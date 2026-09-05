"""이미 unlearning된 모델의 classifier head만 retain train split으로 보정한다.

채점 feature는 head 이전('pre')이라 head를 바꿔도 CKA_f/CKA_r은 비트 단위로
그대로다. 따라서 이 단계는 RUS_o를 전혀 건드리지 않고 AUS만 올린다.
relfe_seed0 기준 AUS 0.9954 → 1.0이면 final +0.0023.

STRATEGY.md §2.2의 "head만 조작하면 0점"과 다른 점: 그건 head 조작이 방법
*전체*일 때 얘기다(CKA_f≈1 → final=0). 여기선 표현이 이미 지워진 뒤다.

보정은 반드시 released retain split으로 한다. validation으로 맞추면 로컬만
오르고 private으로 넘어가지 않는다.

    python tools/headfit.py models/relfe_seed0.pt --out models/relfe_seed0_hf.pt
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imagenet_vit import ViTWrapper
from train_ft import ListDataset, eval_transform
from utils.data import load_split

CACHE = Path("cache")


def load_model(ckpt, device):
    m = ViTWrapper(num_classes=100, pretrained=False, drop_path_rate=0.0,
                   in_model_norm=False)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(sd.get("model", sd), strict=True)
    return m.to(device).eval()


@torch.no_grad()
def retain_features(model, ckpt, device, workers=8, batch=256):
    """retain train 이미지의 pre-logits feature를 fp16으로 캐시."""
    path = CACHE / f"feat_{Path(ckpt).stem}.npz"
    if path.exists():
        d = np.load(path)
        return torch.from_numpy(d["f"]), torch.from_numpy(d["y"])
    root, _, forget_labels, released = load_split()
    items = [it for it in released if it[1] not in set(forget_labels)]
    dl = DataLoader(ListDataset(items, root, eval_transform(model.data_config)),
                    batch_size=batch, shuffle=False, num_workers=workers,
                    pin_memory=True)
    feats, ys, t0 = [], [], time.time()
    for x, y in dl:
        with torch.autocast("cuda", torch.float16):
            enc = model.backbone.forward_features(x.to(device, non_blocking=True))
            feats.append(model.backbone.forward_head(enc, pre_logits=True).half().cpu())
        ys.append(y)
    f, y = torch.cat(feats), torch.cat(ys)
    CACHE.mkdir(exist_ok=True)
    np.savez(path, f=f.numpy(), y=y.numpy())
    print(f"feature 캐시 {tuple(f.shape)} {time.time() - t0:.0f}s -> {path}", flush=True)
    return f, y


def fit_head(head, f, y, device, epochs, lr, wd, batch=4096, seed=0):
    """기존 head에서 이어서 학습한다(재초기화 아님). forget 클래스는 타깃에
    한 번도 안 나오므로 그 행의 logit이 자연히 눌린다."""
    torch.manual_seed(seed)
    f, y = f.to(device).float(), y.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    for ep in range(epochs):
        perm = torch.randperm(len(f), device=device)
        tot = 0.0
        for i in range(0, len(f), batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(head(f[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        acc = (head(f).argmax(1) == y).float().mean().item() * 100
        print(f"  ep{ep} loss {tot / len(f):.4f}  train Acc_r {acc:.2f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    device = torch.device("cuda")
    model = load_model(a.ckpt, device)
    f, y = retain_features(model, a.ckpt, device)
    fit_head(model.head, f, y, device, a.epochs, a.lr, a.wd, seed=a.seed)
    torch.save({"model": model.state_dict()}, a.out)
    print(f"저장: {a.out}")


def demo():
    """head만 바뀌고 backbone은 그대로인지 확인하는 최소 체크."""
    import copy
    m = ViTWrapper(num_classes=100, pretrained=False, drop_path_rate=0.0,
                   in_model_norm=False)
    before = copy.deepcopy(m.state_dict())
    f = torch.randn(512, m.head.in_features)
    y = torch.randint(0, 90, (512,))
    fit_head(m.head, f, y, torch.device("cpu"), 1, 1e-3, 0.0, batch=128)
    after = m.state_dict()
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    # ViTWrapper.head는 backbone.head의 별칭이다.
    assert changed == ["backbone.head.weight", "backbone.head.bias"], changed
    print("ok: head만 변경됨")


if __name__ == "__main__":
    if os.environ.get("HEADFIT_DEMO"):
        demo()
    else:
        main()
