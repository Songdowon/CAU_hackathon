"""가중치 보간으로 CKA_f / CKA_r trade-off 다이얼 돌리기 (학습 불필요).

    theta(a) = a * theta_unlearn + (1 - a) * theta_Mo

a < 1 이면 M_o 쪽으로 되돌아가 retain(CKA_r)이 오르고 forget(CKA_f)도 같이 오른다.
a > 1 이면 반대. 우리는 CKA_f가 과포화(0.02)이고 CKA_r이 부족하므로 a < 1 구간에
더 좋은 점이 있을 수 있다. 평가가 6초라 전체 곡선을 몇 분이면 그린다.

    python tools/interpolate.py models/r017.pt 0.8 0.9 0.95 1.0 1.05
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fasteval

ckpt, alphas = sys.argv[1], [float(a) for a in sys.argv[2:]]
device = torch.device("cuda")
mo = torch.load("m_o/M_o.pt", map_location="cpu", weights_only=True)["model"]
un = torch.load(ckpt, map_location="cpu", weights_only=True)
un = un.get("model", un)

model = fasteval.load_ckpt(ckpt, device)
best = (None, -1)
for a in alphas:
    sd = {k: (mo[k].float() * (1 - a) + un[k].float() * a).to(mo[k].dtype) for k in mo}
    model.load_state_dict(sd, strict=True)
    r = fasteval.score(model, device)
    print(f"a={a:5.2f}  Acc_f {r['Acc_f']:6.2f}  Acc_r {r['Acc_r']:6.2f}  "
          f"CKA_f {r['CKA_f_o']:.4f}  CKA_r {r['CKA_r_o']:.4f}  final {r['final']:.5f}", flush=True)
    if r["final"] > best[1]:
        best = (a, r["final"], sd)

a, score, sd = best
out = f"models/{Path(ckpt).stem}_a{a}.pt"
torch.save({"model": sd}, out)
print(f"\n최적 a={a} (final {score:.5f}) → {out}")
