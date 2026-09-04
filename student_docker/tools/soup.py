"""여러 체크포인트의 가중치를 평균낸다 (model soup).

모두 같은 M_o에서 출발해 미세조정된 모델이라 같은 손실 분지 안에 있고, 평균이
개별 모델보다 잘 일반화되는 경우가 많다. 로컬 점수는 오르는데 private가 안
따라오는 상황을 겨냥한 시도다. 학습이 필요 없다.

    python tools/soup.py models/r017.pt models/r016.pt
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fasteval

paths = sys.argv[1:]
sds = []
for p in paths:
    d = torch.load(p, map_location="cpu", weights_only=True)
    sds.append(d.get("model", d))

soup = {k: sum(sd[k].float() for sd in sds).div_(len(sds)).to(sds[0][k].dtype) for k in sds[0]}

device = torch.device("cuda")
model = fasteval.load_ckpt(paths[0], device)
model.load_state_dict(soup, strict=True)
r = fasteval.score(model, device)
print(f"soup({', '.join(Path(p).stem for p in paths)})")
print(f"  Acc_f {r['Acc_f']:.2f}  Acc_r {r['Acc_r']:.2f}  CKA_f {r['CKA_f_o']:.4f} "
      f"CKA_r {r['CKA_r_o']:.4f}  final {r['final']:.5f}")

out = "models/soup_" + "_".join(Path(p).stem for p in paths) + ".pt"
torch.save({"model": soup}, out)
print(f"저장: {out}")
