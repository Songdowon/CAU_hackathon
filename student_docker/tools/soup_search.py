"""가중 soup 탐색 — 재료마다 다른 가중치를 주고 최적 조합을 찾는다.

단순 평균은 재료를 동등하게 취급하지만, 성격이 다른 모델을 섞을 때 최적 비율은
균등하지 않을 수 있다. 다만 자유 파라미터가 재료 수만큼 늘어나므로 public
validation 15,000장에 과적합될 위험이 크다(노이즈 대역 ±0.0008).

그래서 클래스별로 절반씩 나눠 **A로 고르고 B로 검증**한다. A에서만 오르고 B에서
안 오르면 과적합이다.

    python tools/soup_search.py 40 models/S03.pt models/r016.pt models/r015.pt models/r012.pt
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fasteval

n_trials = int(sys.argv[1])
paths = sys.argv[2:]
rng = np.random.default_rng(0)

sds = []
for p in paths:
    d = torch.load(p, map_location="cpu", weights_only=True)
    sds.append({k: v.float() for k, v in d.get("model", d).items()})
keys = list(sds[0])

device = torch.device("cuda")
model = fasteval.load_ckpt(paths[0], device)
ia, ib = fasteval.half_index("a"), fasteval.half_index("b")
refs = torch.load(fasteval.DS / "validation_cache/refs.pt", map_location="cpu", weights_only=True)
with np.load(fasteval.DS / "validation_cache/M_o__validation.npz") as z:
    mo_pre = z["f_pre"]


def build(w):
    return {k: sum(float(wi) * sd[k] for wi, sd in zip(w, sds)) for k in keys}


# 균등 가중치를 기준선으로 두고, 그 주변을 Dirichlet으로 탐색한다.
trials = [np.ones(len(paths)) / len(paths)]
trials += [rng.dirichlet(np.ones(len(paths)) * 4) for _ in range(n_trials - 1)]

results = []
for i, w in enumerate(trials):
    model.load_state_dict(build(w), strict=True)
    a = fasteval.score(model, device, refs=refs, mo_pre=mo_pre, index=ia)["final"]
    results.append((a, w))
    tag = "균등" if i == 0 else f"{i}"
    print(f"[{tag:>4}] A {a:.5f}  w={np.round(w, 3)}", flush=True)

results.sort(key=lambda r: -r[0])
print("\nA 상위 5개를 B(홀드아웃)로 검증:")
best = None
for a, w in results[:5]:
    model.load_state_dict(build(w), strict=True)
    b = fasteval.score(model, device, refs=refs, mo_pre=mo_pre, index=ib)["final"]
    print(f"  A {a:.5f} → B {b:.5f}   w={np.round(w, 3)}", flush=True)
    if best is None or b > best[0]:
        best = (b, w)

w = best[1]
model.load_state_dict(build(w), strict=True)
full = fasteval.score(model, device, refs=refs, mo_pre=mo_pre)
print(f"\nB 기준 최적 w={np.round(w, 3)}  전체 validation final={full['final']:.5f}")
out = "models/wsoup_" + "_".join(Path(p).stem for p in paths) + ".pt"
sd0 = torch.load(paths[0], map_location="cpu", weights_only=True)
sd0 = sd0.get("model", sd0)
torch.save({"model": {k: v.to(sd0[k].dtype) for k, v in build(w).items()}}, out)
print(f"저장: {out}")
