"""GPM 사전 검증: retain 활성 공간에 여유가 있는가.

선형층 y=Wx에서 가중치 변화가 retain 입력이 span하는 부분공간에 직교하면
retain 출력이 그대로 유지된다. 문제는 retain 90개 클래스가 그 공간을 거의
꽉 채우고 있으면 forget을 바꿀 여지도 함께 사라진다는 것.

그래서 층마다 (a) retain 에너지의 95%를 담는 차원 수와 (b) 그 부분공간 **밖에**
남아 있는 forget 에너지 비율을 잰다. (b)가 충분히 크지 않으면 GPM은 무의미하다.

    python tools/gpm_probe.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imagenet_vit import ViTWrapper
from utils.data import get_loaders

BATCHES = 6
BLOCKS = [6, 8, 11]          # 학습 대상(뒤 6블록) 중 앞/중간/뒤 샘플
NAMES = ["attn.qkv", "mlp.fc1", "mlp.fc2"]

device = torch.device("cuda")
model = ViTWrapper(num_classes=100, pretrained=False, drop_path_rate=0.0, in_model_norm=False)
sd = torch.load("m_o/M_o.pt", map_location="cpu", weights_only=True)
model.load_state_dict(sd.get("model", sd), strict=True)
model = model.eval().to(device)

loaders = get_loaders(model, batch_size=64, workers=8, seed=0)

targets = {}
for b in BLOCKS:
    blk = model.backbone.blocks[b]
    for n in NAMES:
        mod = blk
        for part in n.split("."):
            mod = getattr(mod, part)
        targets[f"b{b}.{n}"] = mod

grams = {k: None for k in targets}
current = {}


def hook(name):
    def fn(_m, inp, _out):
        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
        g = x.T @ x
        current[name] = current.get(name, 0) + g
    return fn


handles = [m.register_forward_hook(hook(k)) for k, m in targets.items()]


@torch.no_grad()
def collect(loader):
    current.clear()
    it = iter(loader)
    for _ in range(BATCHES):
        x, _ = next(it)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x.to(device))
    return {k: v.clone() for k, v in current.items()}


print(f"활성값 수집 중 (배치 {BATCHES}개씩)...", flush=True)
retain_g = collect(loaders["retain"])
forget_g = collect(loaders["forget"])
for h in handles:
    h.remove()

print(f"\n{'층':<16} {'차원':>5} {'95%차원':>8} {'여유차원':>8} {'밖의 forget 에너지':>18}")
for k in targets:
    R, F = retain_g[k].double(), forget_g[k].double()
    evals, evecs = torch.linalg.eigh(R)              # 오름차순
    evals = evals.flip(0); evecs = evecs.flip(1)
    ratio = torch.cumsum(evals, 0) / evals.sum()
    k95 = int((ratio < 0.95).sum().item()) + 1
    d = R.shape[0]
    basis = evecs[:, :k95]                            # retain 주 부분공간
    # forget 에너지 중 그 부분공간 밖에 남은 비율
    inside = torch.einsum("ij,jk,ki->", basis.T, F, basis)
    outside = 1 - (inside / F.diagonal().sum()).item()
    print(f"{k:<16} {d:>5} {k95:>8} {d-k95:>8} {outside*100:>17.2f}%")
