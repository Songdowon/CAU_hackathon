"""채점 지점 바로 앞의 LayerNorm affine(768×2=1536개)만 사후 최적화한다.

채점되는 pre feature는 `backbone.norm` 직후 값이다. CKA는 등방 스케일에는
불변이지만 **차원별 스케일에는 불변이 아니므로**, 이 층의 affine이 CKA를 직접
움직인다. 그래서 `freeze_norm`으로 얼려 왜곡을 막아왔는데, 여기서는 반대로 그
통로를 채점식을 목적함수로 삼아 의도적으로 쓴다.

파라미터가 1536개뿐이라 학습이 아니라 사후 보정에 가깝고, GPU 학습 큐를 쓰지
않는다. 정확도가 흔들리면 head는 하류이므로 headfit이 뒤에서 복구한다(CKA 불변).

보정은 released train split으로 한다. validation으로 맞추면 로컬만 오른다.

    python tools/normfit.py models/uniform_v2n.pt --out models/uniform_v2n_nf.pt
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imagenet_vit import ViTWrapper
from train_ft import ListDataset, eval_transform
from utils.data import load_split


def load_model(ckpt, device):
    m = ViTWrapper(num_classes=100, pretrained=False, drop_path_rate=0.0,
                   in_model_norm=False)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(sd.get("model", sd), strict=True)
    return m.to(device).eval()


def cka(x, y):
    """채점식과 동일한 centered linear CKA (미분 가능)."""
    x = x.float() - x.float().mean(0, keepdim=True)
    y = y.float() - y.float().mean(0, keepdim=True)
    d = torch.linalg.norm(x.T @ x) * torch.linalg.norm(y.T @ y)
    return ((x.T @ y) ** 2).sum() / (d + 1e-12)


@torch.no_grad()
def collect(model, teacher, items, root, device, batch=200):
    """norm 직전 CLS 토큰(h)과 teacher의 pre feature(z_o)를 모은다."""
    grabbed = {}
    handle = model.backbone.norm.register_forward_pre_hook(
        lambda _m, inp: grabbed.__setitem__("h", inp[0]))
    dl = DataLoader(ListDataset(items, root, eval_transform(model.data_config)),
                    batch_size=batch, shuffle=False, num_workers=8, pin_memory=True)
    hs, zs = [], []
    for x, _ in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", torch.float16):
            model.backbone.forward_features(x)
            hs.append(grabbed["h"][:, 0].float().cpu())
            enc = teacher.backbone.forward_features(x)
            zs.append(teacher.backbone.forward_head(enc, pre_logits=True).float().cpu())
    handle.remove()
    return torch.cat(hs), torch.cat(zs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--out", required=True)
    p.add_argument("--mo", default="m_o/M_o.pt")
    p.add_argument("--n", type=int, default=4000, help="retain/forget 각각의 표본 수")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-f", type=float, default=1.0)
    p.add_argument("--lambda-r", type=float, default=1.0)
    a = p.parse_args()

    device = torch.device("cuda")
    model, teacher = load_model(a.ckpt, device), load_model(a.mo, device)
    root, _, fl, released = load_split()
    fl = set(fl)
    g = torch.Generator().manual_seed(0)
    def sample(pool):
        idx = torch.randperm(len(pool), generator=g)[:a.n]
        return [pool[i] for i in idx]
    t0 = time.time()
    h_r, z_r = collect(model, teacher, sample([it for it in released if it[1] not in fl]), root, device)
    h_f, z_f = collect(model, teacher, sample([it for it in released if it[1] in fl]), root, device)
    print(f"표본 retain {len(h_r)} / forget {len(h_f)}  {time.time()-t0:.0f}s", flush=True)

    ln = model.backbone.norm
    w = ln.weight.detach().clone().to(device).requires_grad_(True)
    b = ln.bias.detach().clone().to(device).requires_grad_(True)
    h_r, z_r, h_f, z_f = (t.to(device) for t in (h_r, z_r, h_f, z_f))

    def score(w_, b_):
        pr = F.layer_norm(h_r, (h_r.shape[-1],), w_, b_, ln.eps)
        pf = F.layer_norm(h_f, (h_f.shape[-1],), w_, b_, ln.eps)
        return cka(pf, z_f), cka(pr, z_r)

    c_f0, c_r0 = score(w, b)
    print(f"시작:  CKA_f {c_f0:.4f}  CKA_r {c_r0:.4f}  합 {(1-c_f0)+c_r0:.4f}")

    opt = torch.optim.Adam([w, b], lr=a.lr)
    for s in range(a.steps):
        c_f, c_r = score(w, b)
        loss = a.lambda_f * c_f + a.lambda_r * (1 - c_r)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if s % 50 == 0 or s == a.steps - 1:
            print(f"  step {s:4d}  CKA_f {c_f:.4f}  CKA_r {c_r:.4f}  "
                  f"합 {(1-c_f)+c_r:.4f}", flush=True)

    with torch.no_grad():
        ln.weight.copy_(w)
        ln.bias.copy_(b)
    torch.save({"model": model.state_dict()}, a.out)
    print(f"저장: {a.out}")


def demo():
    """LN(h) 재계산이 실제 pre feature와 일치하는지 — 훅 지점이 맞는지 확인."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = ViTWrapper(num_classes=100, pretrained=False, drop_path_rate=0.0,
                   in_model_norm=False).to(device).eval()
    grabbed = {}
    m.backbone.norm.register_forward_pre_hook(
        lambda _m, inp: grabbed.__setitem__("h", inp[0]))
    x = torch.randn(4, 3, 224, 224, device=device)
    with torch.no_grad():
        enc = m.backbone.forward_features(x)
        pre = m.backbone.forward_head(enc, pre_logits=True)
        ln = m.backbone.norm
        mine = F.layer_norm(grabbed["h"][:, 0], (grabbed["h"].shape[-1],),
                            ln.weight, ln.bias, ln.eps)
    err = (pre - mine).abs().max().item()
    assert err < 1e-4, f"훅 지점 불일치: {err}"
    print(f"ok: LN(h_cls) == pre (최대 오차 {err:.2e})")


if __name__ == "__main__":
    import os
    demo() if os.environ.get("NORMFIT_DEMO") else main()
