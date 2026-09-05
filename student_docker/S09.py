# S09 — r019 mixed-loss cosine-tail optimizer experiment
"""Unlearning: retain 표현은 M_o에 고정하고, forget 표현만 retain 분포로 재매핑한다.

(팀원의 S01이 unlearn.py를 쓰고 있어 별도 파일로 둔다. 실행 방식과 저장 포맷은 동일.)

동기 — grading_docker/score_unlearning.py의 채점식에서 직접 유도:

    AUS   = (1 - max(ref_acc_r - Acc_r, 0)/100) / (1 + Acc_f/100)
    RUS_o = harmonic(1 - CKA_f_o, CKA_r_o)      # depth 'pre' = pre-logits CLS 768d
    final = harmonic(AUS, RUS_o)

CKA는 isotropic scaling과 orthogonal 변환에 불변이므로, forget feature를 그냥
작게 줄이거나 회전시키는 방식은 CKA_f_o가 1 근처로 남아 최종 0점이 된다. forget
샘플들 *사이의* 2차 구조 자체를 바꿔야 한다. 그래서 forget 이미지를 매 스텝
무작위로 짝지은 retain 이미지의 teacher feature/label로 끌어당긴다. 구조가 retain
쪽으로 대체되어 원본과 decorrelate되고, feature가 분포 안에 머물러 발산하지
않으며, 예측이 retain 클래스로 가므로 Acc_f도 0으로 간다.

동시에 retain 배치에서는 teacher(M_o)와의 feature 코사인 + logit KD로 표현을
붙잡아 CKA_r_o와 Acc_r을 지킨다.

    python unlearn_remap.py --config configs/remap.yaml
    python validate_submission.py --ckpt models/R01.pt
    python tools/fasteval.py models/R01.pt      # 빠른 로컬 점수
    python score_model.py models/R01.pt         # 실제 grader (제출 전 필수)
"""
import argparse
import copy
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from imagenet_vit import ViTWrapper
from utils.data import get_loaders


def cosine_tail_lr(step, *, total_steps, tail_start, base_lr, final_lr):
    """Return the LR for one zero-based optimizer update.

    The first tail_start + 1 updates use base_lr. Remaining updates follow
    a cosine curve and the final update reaches final_lr exactly.
    """
    if total_steps < 2:
        raise ValueError("total_steps must be at least 2")
    if not 0 <= step < total_steps:
        raise ValueError("step must be within the optimizer-update range")
    if not 0 <= tail_start < total_steps - 1:
        raise ValueError("tail_start must leave at least one decay update")
    if not 0.0 <= final_lr <= base_lr:
        raise ValueError("final_lr must be between zero and base_lr")
    if step <= tail_start:
        return float(base_lr)
    progress = (step - tail_start) / (total_steps - 1 - tail_start)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(final_lr + (base_lr - final_lr) * cosine)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mo(ckpt, num_classes, device):
    """채점 서버가 사용하는 것과 동일한 로더(strict=True)."""
    m = ViTWrapper(num_classes=num_classes, pretrained=False,
                   drop_path_rate=0.0, in_model_norm=False)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(sd.get("model", sd), strict=True)
    return m.to(device)


def set_trainable(model, k, freeze_norm=False):
    """뒤쪽 k개 블록 + final norm + head만 학습 (k < 0이면 전체).

    NegGrad 베이스라인에서도 b4/b8의 CKA는 0.93/0.89로 멀쩡했고 'pre'만 0.21로
    무너졌다. 손상이 후반부에 집중되므로 앞단을 얼리면 retain을 싸게 지킬 수 있다.

    freeze_norm: 채점되는 pre feature는 backbone.norm 직후 값이라, 그 affine
    파라미터가 모든 feature를 차원별로 스케일한다. CKA는 등방 스케일에는
    불변이지만 차원별 스케일에는 불변이 아니므로, 이 층이 전역 왜곡의 직접
    통로가 된다. 얼려서 그 통로를 막는다."""
    if k is not None and k >= 0:
        for p in model.parameters():
            p.requires_grad_(False)
        blocks = model.backbone.blocks
        for blk in blocks[max(len(blocks) - k, 0):]:
            for p in blk.parameters():
                p.requires_grad_(True)
        mods = [model.backbone.head] if freeze_norm else [model.backbone.norm, model.backbone.head]
        for mod in mods:
            for p in mod.parameters():
                p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def batch_linear_cka(x, y):
    """채점식과 동일한 centered linear CKA의 미분 가능한 미니배치 판."""
    x = x.float() - x.float().mean(0, keepdim=True)
    y = y.float() - y.float().mean(0, keepdim=True)
    denom = torch.linalg.norm(x.T @ x) * torch.linalg.norm(y.T @ y)
    return ((x.T @ y) ** 2).sum() / (denom + 1e-12)


def forward(model, x):
    """grader와 동일한 경로: pre-logits feature와 logit을 함께 얻는다."""
    encoded = model.backbone.forward_features(x)
    pre = model.backbone.forward_head(encoded, pre_logits=True)
    return pre, model.head(pre)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/remap.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])
    t = cfg["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_mo(cfg["model"]["mo_ckpt"], cfg["model"]["num_classes"], device)
    teacher = copy.deepcopy(model).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)

    loaders = get_loaders(model, batch_size=cfg["data"]["batch_size"],
                          workers=cfg["data"]["workers"], seed=cfg["seed"],
                          split_pt=cfg["data"]["split"],
                          forget_json=cfg["data"]["forget"])

    # 채점은 증강 없는 eval transform에서 이뤄지는데 get_loaders는 retain에도
    # RandAugment+random erasing을 건다. 앵커를 거는 지점과 측정 지점을 맞추면
    # CKA_r이 더 잘 일반화될 수 있어 선택지로 둔다.
    if t.get("retain_clean", False):
        from torch.utils.data import DataLoader

        from train_ft import ListDataset
        from utils.data import load_split
        root, _, fl, released = load_split(cfg["data"]["split"], cfg["data"]["forget"])
        items = [it for it in released if it[1] not in set(fl)]
        loaders["retain"] = DataLoader(
            ListDataset(items, root, loaders["eval_transform"]),
            batch_size=cfg["data"]["batch_size"], shuffle=True,
            num_workers=cfg["data"]["workers"], pin_memory=True)
        print(f"retain 앵커를 eval transform으로 교체 ({len(items)}장)")

    # retain 보존이 "학습에 쓴 이미지"에만 통하는 암기 과적합이 관측됐다
    # (학습 retain CKA_r 0.983 vs 처음 보는 retain 0.976). 앵커 이미지를 매번 더
    # 크게 변형시켜 특정 이미지 암기를 방해하고, 클래스 표현 자체를 유지하도록 민다.
    if t.get("retain_aug"):
        from torch.utils.data import DataLoader

        from imagenet_vit import MAE_FT, build_train_transform
        from train_ft import ListDataset
        from utils.data import load_split
        recipe = dict(MAE_FT)
        recipe["auto_augment"] = t["retain_aug"]
        recipe["reprob"] = float(t.get("retain_reprob", recipe["reprob"]))
        tf, _ = build_train_transform(model.data_config, recipe)
        root, _, fl, released = load_split(cfg["data"]["split"], cfg["data"]["forget"])
        items = [it for it in released if it[1] not in set(fl)]
        loaders["retain"] = DataLoader(
            ListDataset(items, root, tf), batch_size=cfg["data"]["batch_size"],
            shuffle=True, num_workers=cfg["data"]["workers"], pin_memory=True)
        print(f"retain 증강 강화: {t['retain_aug']}, reprob {recipe['reprob']}")

    # CKA_r은 retain 13,500장에서 측정되는데 앵커는 매 스텝 batch_size장만 본다.
    # forget 압력을 건드리지 않고 retain 표본만 늘려 제약 추정을 개선한다.
    if t.get("batch_retain"):
        from torch.utils.data import DataLoader

        from train_ft import ListDataset
        from utils.data import load_split
        root, _, fl, released = load_split(cfg["data"]["split"], cfg["data"]["forget"])
        items = [it for it in released if it[1] not in set(fl)]
        tf = loaders["eval_transform"] if t.get("retain_clean") else loaders["retain"].dataset.transform
        loaders["retain"] = DataLoader(
            ListDataset(items, root, tf), batch_size=int(t["batch_retain"]),
            shuffle=True, num_workers=cfg["data"]["workers"], pin_memory=True)
        print(f"retain 배치 {t['batch_retain']}로 확대")

    params = set_trainable(model, t.get("trainable_blocks", -1), t.get("freeze_norm", False))
    total = sum(q.numel() for q in model.parameters())
    print(f"학습 파라미터 {sum(q.numel() for q in params)/1e6:.1f}M / 전체 {total/1e6:.1f}M")

    opt = getattr(torch.optim, t.get("optimizer", "AdamW"))(
        params, lr=float(t["lr"]), weight_decay=float(t.get("weight_decay", 0.0)))

    # L2-SP: 가중치를 M_o 쪽으로 당기는 decoupled 정규화.
    # retain 보존이 학습에서 본 이미지에만 맞춰지는 것이 로컬 점수가 private로
    # 이전되지 않는 원인이다(CKA_r 일반화 갭 -0.004~-0.0066). M_o는 모든 이미지에서
    # retain의 정답이므로, 가중치가 M_o 근처에 묶여 있을수록 못 본 이미지에서도
    # retain이 유지된다. 사후 보간과 달리 옵티마이저가 "M_o에 가까우면서 CKA_f도
    # 낮은" 해를 직접 찾는다.
    l2sp = float(t.get("lambda_l2sp", 0) or 0)
    anchor = [q.detach().clone() for q in params] if l2sp else None

    retain_it, forget_it = cycle(loaders["retain"]), cycle(loaders["forget"])
    steps = int(t["steps"])
    lr_schedule = str(t.get("lr_schedule", "constant")).lower()
    if lr_schedule not in {"constant", "cosine_tail"}:
        raise ValueError(f"unsupported lr_schedule: {lr_schedule}")
    tail_start = int(t.get("lr_tail_start", steps - 1))
    final_lr = float(t.get("lr_final", t["lr"]))
    if lr_schedule == "cosine_tail":
        cosine_tail_lr(0, total_steps=steps, tail_start=tail_start,
                       base_lr=float(t["lr"]), final_lr=final_lr)
        print(f"LR schedule cosine_tail: {t['lr']:.3g} through step {tail_start}, "
              f"then decay to {final_lr:.3g} on step {steps - 1}")
    temp = float(t.get("kd_temperature", 2.0))
    w = {k: float(t.get(f"lambda_{k}", d)) for k, d in
         [("feat_r", 1.0), ("kd_r", 1.0), ("ce_r", 0.0),
          ("remap_f", 1.0), ("ce_f", 0.5), ("cka_f", 0.0), ("cka_r", 0.0)]}

    # 학습 궤적 평균. 로컬 점수는 오르는데 private가 안 따라오는 상황이라
    # 마지막 스텝의 우연한 위치보다 평균 지점이 더 잘 일반화될 수 있다.
    decay = float(t.get("ema_decay", 0) or 0)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()} if decay else None

    model.train()
    t0 = time.time()
    for step in range(steps):
        if lr_schedule == "cosine_tail":
            lr_now = cosine_tail_lr(
                step, total_steps=steps, tail_start=tail_start,
                base_lr=float(t["lr"]), final_lr=final_lr)
            for group in opt.param_groups:
                group["lr"] = lr_now
        x_r, y_r = next(retain_it)
        x_f, _ = next(forget_it)
        x_r, y_r = x_r.to(device, non_blocking=True), y_r.to(device)
        x_f = x_f.to(device, non_blocking=True)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            with torch.no_grad():
                z_r_o, logit_r_o = forward(teacher, x_r)
                z_f_o, _ = forward(teacher, x_f)
            z_r_u, logit_r_u = forward(model, x_r)
            z_f_u, logit_f_u = forward(model, x_f)

            # forget 이미지마다 retain 배치에서 무작위 파트너를 뽑아 그 teacher
            # feature/label을 목표로 삼는다 (매 스텝 새로 뽑히므로 구조가 섞인다).
            perm = torch.randint(0, x_r.shape[0], (x_f.shape[0],), device=device)
            tgt_z, tgt_y = z_r_o[perm], y_r[perm]

            # 개별 샘플 코사인은 "이 이미지의 feature를 유지하라"를 가르쳐서 처음 보는
            # retain 이미지에서 무너진다(학습 retain CKA_r 0.983 vs 미본 0.976).
            # 채점 지표 CKA_r 자체가 샘플 간 관계 통계이므로, 배치 CKA를 직접
            # 최대화해 "구조를 유지하라"로 바꾼다. forget 쪽과 대칭이 맞는다.
            l_cka_r = 1 - batch_linear_cka(z_r_u, z_r_o) if w["cka_r"] else z_r_u.sum() * 0

            # retain 가드레일: CKA_r이 이미 충분히 높으면 앵커를 끈다. forget 쪽에
            # 같은 구조(cka_floor)를 걸었을 때 3/3 짝지은 승리에 +0.002가 나왔다.
            # 이미 지켜진 표현을 계속 당기는 힘은 forget 제거를 방해하는 데만 쓰인다.
            guard = float(t.get("retain_guard", 0) or 0)
            if guard:
                with torch.no_grad():
                    keep = (batch_linear_cka(z_r_u, z_r_o) < guard).float()
            l_feat_r = (1 - F.cosine_similarity(z_r_u, z_r_o, dim=1)).mean()
            l_kd_r = F.kl_div(F.log_softmax(logit_r_u / temp, 1),
                              F.softmax(logit_r_o / temp, 1),
                              reduction="batchmean") * temp ** 2
            l_ce_r = F.cross_entropy(logit_r_u, y_r)
            l_remap_f = (1 - F.cosine_similarity(z_f_u, tgt_z, dim=1)).mean()
            l_ce_f = F.cross_entropy(logit_f_u, tgt_y)
            # CKA_f는 0.02 근처에서 포화되는데 손실은 계속 0을 향해 밀어붙인다.
            # 그 잉여 gradient가 공유 백본을 통해 retain을 갉아먹으므로, 충분히
            # 낮아지면(floor 이하) 더 밀지 않는다. 채점상 CKA_f를 0.02에서 0으로
            # 만들어봐야 final +0.006인 반면 CKA_r 개선은 +0.015 가치가 있다.
            if w["cka_f"]:
                l_cka_f = batch_linear_cka(z_f_u, z_f_o)
                floor = float(t.get("cka_floor", 0) or 0)
                if floor:
                    l_cka_f = torch.relu(l_cka_f - floor)
            else:
                l_cka_f = z_f_u.sum() * 0

            loss_r = (w["feat_r"] * l_feat_r + w["kd_r"] * l_kd_r + w["ce_r"] * l_ce_r
                      + w["cka_r"] * l_cka_r)
            if guard:
                loss_r = loss_r * keep
            loss_f = w["remap_f"] * l_remap_f + w["ce_f"] * l_ce_f + w["cka_f"] * l_cka_f
            loss = loss_r + loss_f

        opt.zero_grad(set_to_none=True)
        if t.get("grad_project", False):
            # forget gradient에서 retain gradient와 충돌하는 성분을 제거한다.
            # CKA_f와 CKA_r이 계속 맞바꿔지는 건 forget 업데이트가 retain 표현이
            # 사는 방향으로도 일어나기 때문이다. 그 성분만 빼면 두 지표를 동시에
            # 밀 수 있다.
            loss_r.backward(retain_graph=True)
            g_r = [p.grad.detach().clone() if p.grad is not None else None for p in params]
            opt.zero_grad(set_to_none=True)
            loss_f.backward()
            pairs = [(p, gr) for p, gr in zip(params, g_r) if p.grad is not None and gr is not None]
            dot = sum((p.grad * gr).sum() for p, gr in pairs)
            if dot < 0:
                nrm = sum((gr * gr).sum() for _, gr in pairs) + 1e-12
                for p, gr in pairs:
                    p.grad.sub_(gr, alpha=float(dot / nrm))
            for p, gr in pairs:
                p.grad.add_(gr)
        else:
            loss.backward()
        if t.get("clip_grad"):
            torch.nn.utils.clip_grad_norm_(params, float(t["clip_grad"]))
        if anchor is not None:
            # clip 뒤에 더한다 — 정규화까지 잘려나가면 세기를 조절할 수 없다.
            with torch.no_grad():
                for q, a in zip(params, anchor):
                    if q.grad is not None:
                        q.grad.add_(q.detach() - a, alpha=l2sp)
        opt.step()

        if ema is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema[k].mul_(decay).add_(v.float(), alpha=1 - decay)

        if step % 50 == 0 or step == steps - 1:
            print(f"step {step:4d}/{steps} loss {loss.item():.4f} | feat_r {l_feat_r.item():.4f} "
                  f"kd_r {l_kd_r.item():.4f} remap_f {l_remap_f.item():.4f} "
                  f"ce_f {l_ce_f.item():.4f} cka_f {float(l_cka_f):.4f} "
                  f"cka_r {float(l_cka_r):.4f} lr {opt.param_groups[0]['lr']:.3g} | {time.time()-t0:.0f}s",
                  flush=True)

    out = cfg["output"]["save_path"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if ema is not None:
        sd = model.state_dict()
        model.load_state_dict({k: ema[k].to(sd[k].dtype) for k in sd}, strict=True)
        print(f"EMA 가중치로 저장 (decay {decay})")
    torch.save({"model": model.state_dict()}, out)
    print(f"저장 완료: {out}  ({time.time()-t0:.0f}s)")
    print(f"로컬 점수: python tools/fasteval.py {out}")


if __name__ == "__main__":
    main()
