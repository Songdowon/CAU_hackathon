"""표본이 바뀌어도 유지되는 모델을 고른다 (robust selection).

지금까지 우리는 validation 전체의 final 최고를 골랐는데, 그건 "그 한 표본에 가장
잘 맞은 모델"을 고르는 것이다. private는 우리가 못 보는 또 하나의 표본이므로,
표본이 바뀌어도 점수가 유지되는 모델이 이전될 가능성이 높다.

validation을 클래스별로 A/B 절반씩 나눠 각각 채점하고, min(A,B)와 |A-B|를 같이
본다. A와 B는 완전히 disjoint한 이미지 집합이다.

    python tools/robust_select.py models/*.pt
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fasteval

paths = [p for p in sys.argv[1:] if Path(p).exists()]
device = torch.device("cuda")
refs = torch.load(fasteval.DS / "validation_cache/refs.pt", map_location="cpu", weights_only=True)
with np.load(fasteval.DS / "validation_cache/M_o__validation.npz") as z:
    mo_pre = z["f_pre"]
ia, ib = fasteval.half_index("a"), fasteval.half_index("b")

rows = []
for p in paths:
    try:
        m = fasteval.load_ckpt(p, device)
    except Exception as e:
        print(f"  건너뜀 {Path(p).stem}: {type(e).__name__}")
        continue
    a = fasteval.score(m, device, refs=refs, mo_pre=mo_pre, index=ia)["final"]
    b = fasteval.score(m, device, refs=refs, mo_pre=mo_pre, index=ib)["final"]
    rows.append((min(a, b), abs(a - b), (a + b) / 2, a, b, Path(p).stem))
    print(f"  {Path(p).stem:<34} A {a:.5f}  B {b:.5f}  min {min(a,b):.5f}  |차| {abs(a-b):.5f}",
          flush=True)

rows.sort(reverse=True)
print(f"\n{'min(A,B) 상위':<36}{'min':>9}{'|A-B|':>9}{'평균':>9}")
for r in rows[:8]:
    print(f"{r[5]:<36}{r[0]:>9.5f}{r[1]:>9.5f}{r[2]:>9.5f}")

print(f"\n{'평균 상위 (기존 방식)':<36}{'min':>9}{'|A-B|':>9}{'평균':>9}")
for r in sorted(rows, key=lambda x: -x[2])[:8]:
    print(f"{r[5]:<36}{r[0]:>9.5f}{r[1]:>9.5f}{r[2]:>9.5f}")
