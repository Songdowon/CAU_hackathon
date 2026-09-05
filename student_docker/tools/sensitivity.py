"""표본 민감도로 private 점수를 예측한다.

제출 3건에서 확인된 관계: private ≈ 로컬 − |A−B| − 0.0005 (오차 ≤2e-4).
|A−B|는 겹치지 않는 검증 절반 두 개의 점수 차이, 즉 "이 모델의 점수가 어느
표본을 보느냐에 얼마나 휘는가"다. private도 우리가 못 보는 또 하나의 표본이므로
그만큼 깎인다.

고정 A/B 분할 하나로 재면 그 분할에 과적합하므로 무작위 층화 분할을 여러 번
쓴다.

    python tools/sensitivity.py models/a.pt models/b.pt          # 단품
    python tools/sensitivity.py --soup models/a.pt models/b.pt   # 균등 소프 1개
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fasteval

N_SPLIT = 5
OFFSET = 0.0005


def rand_halves(seed):
    labels = np.load(fasteval.LB)
    rng = np.random.default_rng(seed)
    a = []
    for c in np.unique(labels):
        pos = rng.permutation(np.flatnonzero(labels == c))
        a.append(pos[: len(pos) // 2])
    a = np.sort(np.concatenate(a))
    return a, np.setdiff1d(np.arange(len(labels)), a)


def measure(model, device, refs, mo_pre):
    full = fasteval.score(model, device, refs=refs, mo_pre=mo_pre)
    gaps = []
    for s in range(N_SPLIT):
        ia, ib = rand_halves(s)
        fa = fasteval.score(model, device, refs=refs, mo_pre=mo_pre, index=ia)["final"]
        fb = fasteval.score(model, device, refs=refs, mo_pre=mo_pre, index=ib)["final"]
        gaps.append(abs(fa - fb))
    g = float(np.mean(gaps))
    return full["final"], g, float(np.std(gaps)), full["final"] - g - OFFSET


def main():
    args = [a for a in sys.argv[1:] if a != "--soup"]
    soup = "--soup" in sys.argv
    device = torch.device("cuda")
    refs = torch.load(fasteval.DS / "validation_cache/refs.pt", map_location="cpu",
                      weights_only=True)
    with np.load(fasteval.DS / "validation_cache/M_o__validation.npz") as z:
        mo_pre = z["f_pre"]
    model = fasteval.load_ckpt(args[0], device)

    print(f"{'모델':<26}{'로컬':>9}{'평균|A-B|':>11}{'(표준편차)':>11}{'예측 private':>13}")
    if soup:
        S = []
        for p in args:
            x = torch.load(p, map_location="cpu", weights_only=True)
            S.append({k: v.float() for k, v in x.get("model", x).items()})
        dt = {k: v.dtype for k, v in
              torch.load(args[0], map_location="cpu", weights_only=True)["model"].items()}
        model.load_state_dict(
            {k: sum(s[k] for s in S).div_(len(S)).to(dt[k]) for k in S[0]}, strict=True)
        f, g, sd, pred = measure(model, device, refs, mo_pre)
        print(f"{'soup(%d개)' % len(args):<26}{f:>9.5f}{g:>11.5f}{sd:>11.5f}{pred:>13.5f}")
        return
    for p in args:
        f, g, sd, pred = measure(fasteval.load_ckpt(p, device), device, refs, mo_pre)
        print(f"{Path(p).stem:<26}{f:>9.5f}{g:>11.5f}{sd:>11.5f}{pred:>13.5f}", flush=True)


if __name__ == "__main__":
    main()
