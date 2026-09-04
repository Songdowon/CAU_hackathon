"""Fine-tune ViT-B/16 from MAE on the frozen competition split -- M_o and M_r.

    # M_o: all 100 classes
    ENV=/data/hai_ssh/miniconda3/envs/libero-para-openvla-oft/bin
    CUDA_VISIBLE_DEVICES=2,3,4,5 $ENV/torchrun --nproc_per_node=4 train_ft.py \
        --tag M_o_pilot --epochs 50

    # M_r: same recipe, forget classes dropped, head still 100-way
    CUDA_VISIBLE_DEVICES=2,3,4,5 $ENV/torchrun --nproc_per_node=4 train_ft.py \
        --tag M_r_s1 --epochs 50 --forget <forget10.json> --seed 1

Three things this script exists to keep honest:

1. **It trains on `released` only.** The scoring slices live in the same split
   file and are never touched by the optimizer, so M_o cannot leak into the
   leaderboard. The split's sha256 is copied into every checkpoint.

2. **M_r comparability is explicit, not incidental.** Dropping 10 of 100 classes
   removes ~10% of the data, so matching epochs and matching optimizer steps are
   different things and you have to pick:

       --match epoch  (default) M_r does the same number of PASSES over retain
                      data as M_o -> each retain image is seen equally often, and
                      the only difference between the models is that M_o also saw
                      forget images. Fewer total steps.
       --match iter   M_r does the same number of STEPS as M_o -> equal compute
                      and equal samples seen, but each retain image is seen ~11%
                      more often than in M_o.

   Neither is free; `--match epoch` keeps the cleaner definition of M_r and is
   the default, `--match iter` protects Acc_r(M_r) from being depressed by a
   shorter schedule. Decide with the pilot: if Acc_r(M_o) - Acc_r(M_r) > 0.5%p,
   switch.

3. **No checkpoint selection.** Fixed step count, cosine to min_lr, last
   checkpoint wins. Early stopping or best-val picking would make the M_o/M_r
   difference partly a selection artifact rather than the forget data.

Eval is reported per class so any forget/retain split can be scored afterwards
without re-running inference.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from imagenet_vit import (MAE_FT, SCRATCH, ViTWrapper, build_mixup,
                          build_param_groups, build_train_transform, scaled_lr)


class ListDataset(Dataset):
    """(relative path, label) pairs against a fixed root -- the split file is the
    single source of truth for what is trainable and what is held out."""

    def __init__(self, items, root, transform):
        self.items, self.root, self.transform = items, root, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, label = self.items[i]
        with open(os.path.join(self.root, rel), "rb") as f:
            img = Image.open(f).convert("RGB")
        return self.transform(img), label


def eval_transform(cfg):
    """Matches build_train_transform's contract: normalizes internally, so the
    model must be built with in_model_norm=False."""
    import timm

    return timm.data.create_transform(
        input_size=cfg["input_size"][-1], is_training=False,
        crop_pct=cfg.get("crop_pct", 0.9),
        interpolation=cfg.get("interpolation", "bicubic"),
        mean=cfg["mean"], std=cfg["std"])


def lr_mult(it, total, warmup):
    if it < warmup:
        return (it + 1) / max(warmup, 1)
    p = (it - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    correct = torch.zeros(n_classes, dtype=torch.float64, device=device)
    total = torch.zeros(n_classes, dtype=torch.float64, device=device)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x).argmax(1)
        correct.index_add_(0, y, (pred == y).double())
        total.index_add_(0, y, torch.ones_like(y, dtype=torch.float64))
    if dist.is_initialized():
        dist.all_reduce(correct)
        dist.all_reduce(total)
    model.train()
    return correct.cpu().numpy(), total.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="splits/student_split.pt")
    p.add_argument("--tag", required=True, help="checkpoint name, e.g. M_o_pilot")
    p.add_argument("--out", default="/data2/AAAI/hai_ssh/ckpt")
    p.add_argument("--forget", default=None,
                   help="json with a 'wnid' list; those classes are dropped from "
                        "training (M_r). Omit for M_o.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--match", default="epoch", choices=["epoch", "iter"],
                   help="how M_r is made comparable to M_o; see module docstring")
    p.add_argument("--total_iters", type=int, default=None,
                   help="with --match iter: M_o's step count, copied verbatim")
    p.add_argument("--batch_size", type=int, default=128, help="PER GPU")
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=10, help="in epochs")
    p.add_argument("--init", default="mae", choices=["mae", "random"],
                   help="mae: MAE checkpoint + MAE_FT recipe (default). "
                        "random: random init + DeiT-style SCRATCH recipe -- no "
                        "pretraining ever saw the forget classes, at the cost of "
                        "needing far more epochs and landing at lower accuracy.")
    args = p.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    lead = rank == 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = torch.load(args.split, weights_only=False)
    meta, splits = d["meta"], d["splits"]
    dataset_root = os.environ.get("DATASET_ROOT")
    root = (
        os.path.join(dataset_root, "imagenet_released")
        if dataset_root
        else meta["root"]
    )
    wnids = meta["wnids"]
    n_classes = len(wnids)

    forget_labels = set()
    if args.forget:
        fw = json.load(open(args.forget))
        fw = fw["wnid"] if isinstance(fw, dict) else fw
        forget_labels = {wnids.index(w) for w in fw}
        assert len(forget_labels) == len(fw), "forget wnids must be inside the pool"

    train_items = [(p_, l) for p_, l in splits["released"] if l not in forget_labels]

    # --init selects BOTH the weights and the optimisation recipe; they are not
    # interchangeable (layer-wise lr decay protects a pretrained trunk and starves
    # a random one, and the from-scratch lr is half the fine-tuning lr).
    R = MAE_FT if args.init == "mae" else SCRATCH
    model = ViTWrapper(num_classes=n_classes, pretrained=args.init == "mae",
                       drop_path_rate=R["drop_path"], in_model_norm=False)
    cfg = model.data_config
    model = model.to(device)
    net = nn.parallel.DistributedDataParallel(model, device_ids=[rank]) if world > 1 else model

    train_tf, in_model_norm = build_train_transform(cfg, R)
    assert not in_model_norm
    eval_tf = eval_transform(cfg)

    train_ds = ListDataset(train_items, root, train_tf)
    sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed, drop_last=True) \
        if world > 1 else None
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          shuffle=sampler is None, num_workers=args.workers,
                          pin_memory=True, drop_last=True,
                          persistent_workers=args.workers > 0)

    eval_dls = {}
    # 공개 validation은 관찰용일 뿐 optimizer에는 절대 들어가지 않습니다.
    # private test split은 이 manifest 자체에 존재하지 않습니다.
    for name in ("validation",):
        if name not in splits:
            continue
        # exact stride sharding, NOT DistributedSampler: that one pads the last
        # shard by repeating samples, which would double-count them in the
        # accuracy sum whenever len(ds) % world != 0.
        items = splits[name][rank::world] if world > 1 else splits[name]
        eval_dls[name] = DataLoader(ListDataset(items, root, eval_tf),
                                    batch_size=args.batch_size * 2, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)

    total_batch = args.batch_size * world
    iters_per_epoch = len(train_dl)
    if args.match == "iter":
        if args.total_iters is None:
            raise SystemExit("--match iter requires --total_iters (M_o's step count)")
        total_iters = args.total_iters
    else:
        total_iters = iters_per_epoch * args.epochs
    warmup_iters = iters_per_epoch * R["warmup_epochs"]
    lr = scaled_lr(total_batch, R)

    groups = build_param_groups(model, R)
    for g in groups:
        g["base_lr"] = lr * g["lr_scale"]
    opt = torch.optim.AdamW(groups, lr=lr, betas=R["betas"])
    mixup = build_mixup(n_classes, R)
    clip = R.get("clip_grad")
    from timm.loss import SoftTargetCrossEntropy
    crit = SoftTargetCrossEntropy()

    if lead:
        os.makedirs(args.out, exist_ok=True)
        print(f"[cfg] tag={args.tag} classes={n_classes} "
              f"forget_classes={len(forget_labels)}", flush=True)
        print(f"[cfg] train imgs {len(train_items):,} | per-GPU bs {args.batch_size} "
              f"x {world} = {total_batch} | lr {lr:.2e} (blr {R['blr']:.0e}) "
              f"init={args.init}", flush=True)
        print(f"[cfg] match={args.match} iters/epoch {iters_per_epoch} "
              f"total_iters {total_iters} (= {total_iters/iters_per_epoch:.1f} epochs) "
              f"warmup {warmup_iters}", flush=True)
        print(f"[cfg] layer_decay {R['layer_decay']} groups {len(groups)} "
              f"lr_scale {min(g['lr_scale'] for g in groups):.4f}..1.0 | "
              f"split sha256 released={meta['sha256']['released'][:12]}", flush=True)

    it, t0 = 0, time.time()
    epoch = 0
    while it < total_iters:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for x, y in train_dl:
            if it >= total_iters:
                break
            m = lr_mult(it, total_iters, warmup_iters)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * m
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x, yt = mixup(x, y)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(net(x), yt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            it += 1
            if lead and it % 100 == 0:
                el = time.time() - t0
                print(f"[train] it {it}/{total_iters} ep {epoch} loss {loss.item():.3f} "
                      f"lr {opt.param_groups[-1]['lr']:.2e} "
                      f"{it*total_batch/el:.0f} img/s eta {(total_iters-it)*el/it/60:.1f}m",
                      flush=True)
        epoch += 1
        last_eval = {}
        if epoch % args.eval_every == 0 or it >= total_iters:
            for name, dl in eval_dls.items():
                c, t = evaluate(net, dl, device, n_classes)
                last_eval[name] = {"correct": c.tolist(), "total": t.tolist()}
                if lead:
                    acc = c.sum() / max(t.sum(), 1) * 100
                    print(f"[eval] ep {epoch} {name:10s} top-1 {acc:.2f}% "
                          f"({int(t.sum()):,} imgs)", flush=True)

    if lead:
        per_class = last_eval        # the loop always evaluates its final state
        cfg_dump = {
            "tag": args.tag, "seed": args.seed, "epochs_arg": args.epochs,
            "match": args.match, "total_iters": total_iters,
            "iters_per_epoch": iters_per_epoch, "total_batch": total_batch,
            "lr": lr, "n_train": len(train_items),
            "forget_wnids": sorted(wnids[l] for l in forget_labels),
            "init": args.init, "recipe": {k: v for k, v in R.items()},
            "split_sha256": meta["sha256"], "pool_wnids": wnids,
            "arch": "vit_base_patch16_224.mae" if args.init == "mae" else
                    "vit_base_patch16_224 (random init)", "in_model_norm": False,
        }
        torch.save({"model": model.state_dict(), "config": cfg_dump,
                    "per_class_eval": per_class},
                   os.path.join(args.out, f"{args.tag}.pt"))
        with open(os.path.join(args.out, f"{args.tag}_config.json"), "w") as f:
            json.dump({**cfg_dump, "per_class_eval": per_class}, f, indent=2)
        print(f"\n[done] {(time.time()-t0)/60:.1f} min -> {args.out}/{args.tag}.pt",
              flush=True)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
