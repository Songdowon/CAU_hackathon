"""Greedy soup: 재료를 하나씩 추가해보며 개선될 때만 채택한다.

학습이 필요 없다. 후보 풀 전체를 매 라운드 시험해 가장 크게 개선하는 하나를
넣고, 더 이상 개선이 없으면 멈춘다. 조합당 6초짜리 평가만 쓴다.

    python tools/greedy_soup.py models/s06_seed0.pt models/r019.pt ...
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fasteval

paths = [a for a in sys.argv[1:] if a.endswith(".pt")]
device = torch.device("cuda")
model = fasteval.load_ckpt(paths[0], device)
refs = torch.load(fasteval.DS / "validation_cache/refs.pt", map_location="cpu", weights_only=True)
import numpy as np
with np.load(fasteval.DS / "validation_cache/M_o__validation.npz") as z:
    mo_pre = z["f_pre"]

states = {}
for p in paths:
    d = torch.load(p, map_location="cpu", weights_only=True)
    states[p] = {k: v.float() for k, v in d.get("model", d).items()}
keys = list(next(iter(states.values())))
dtypes = {k: v.dtype for k, v in torch.load(paths[0], map_location="cpu",
                                            weights_only=True).get("model", {}).items()}


# 전체 validation의 final로 고르면 greedy 선택 자체가 그 표본에 과적합한다
# (측정: 재료 단품 |A-B| 0.0009~0.0029 → 소프 0.0040). ROBUST=1이면 절반씩
# 나눈 A/B 중 나쁜 쪽을 목적함수로 써서 표본에 안 휘는 조합을 고른다.
ROBUST = "--robust" in sys.argv
ia, ib = (fasteval.half_index("a"), fasteval.half_index("b")) if ROBUST else (None, None)


def evaluate(members):
    sd = {k: sum(states[p][k] for p in members).div_(len(members)).to(dtypes[k]) for k in keys}
    model.load_state_dict(sd, strict=True)
    if ROBUST:
        a = fasteval.score(model, device, refs=refs, mo_pre=mo_pre, index=ia)["final"]
        b = fasteval.score(model, device, refs=refs, mo_pre=mo_pre, index=ib)["final"]
        return min(a, b), sd
    return fasteval.score(model, device, refs=refs, mo_pre=mo_pre)["final"], sd


print("단일 모델 점수:")
singles = []
for p in paths:
    s, _ = evaluate([p])
    singles.append((s, p))
    print(f"  {Path(p).stem:<26}{s:.5f}")
singles.sort(reverse=True)

chosen = [singles[0][1]]
best = singles[0][0]
print(f"\n시작: {Path(chosen[0]).stem} ({best:.5f})")
while True:
    gains = []
    for p in paths:
        if p in chosen:
            continue
        s, _ = evaluate(chosen + [p])
        gains.append((s, p))
    if not gains:
        break
    gains.sort(reverse=True)
    if gains[0][0] <= best + 1e-5:
        print(f"더 개선 없음 (최선 후보 {Path(gains[0][1]).stem} {gains[0][0]:.5f})")
        break
    best, pick = gains[0]
    chosen.append(pick)
    print(f"  + {Path(pick).stem:<26}→ {best:.5f}  (재료 {len(chosen)}개)")

_, sd = evaluate(chosen)
out = "models/greedy_soup_robust.pt" if ROBUST else "models/greedy_soup.pt"
torch.save({"model": sd}, out)
print(f"\n최종 {best:.5f}: {', '.join(Path(p).stem for p in chosen)}\n저장: {out}")
