"""Retain recovery: unlearning이 끝난 모델을 retain 데이터로만 짧게 복구한다.

forget 클래스 이미지는 전혀 쓰지 않으므로 삭제는 유지되고, retain 표현만 M_o
쪽으로 되돌아온다는 가설. 손실은 학습 때 쓰던 retain 항(feature 코사인 + logit KD)
그대로이고, forget 항은 없다.

    python tools/recover.py models/soup_s06x3_r019.pt 100 300 600
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fasteval
from unlearn_remap import cycle, forward, load_mo
from utils.data import get_loaders

ckpt = sys.argv[1]
checkpoints = sorted(int(v) for v in sys.argv[2:] if v.isdigit()) or [300]
LR = 1e-5
KD = "--no-kd" not in sys.argv

device = torch.device("cuda")
teacher = load_mo("m_o/M_o.pt", 100, device).eval()
for p in teacher.parameters():
    p.requires_grad_(False)
model = load_mo(ckpt, 100, device)

# 학습 때와 같은 범위만 연다 (뒤 6블록 + head, norm은 동결 상태 유지)
for p in model.parameters():
    p.requires_grad_(False)
for blk in model.backbone.blocks[6:]:
    for p in blk.parameters():
        p.requires_grad_(True)
for p in model.backbone.head.parameters():
    p.requires_grad_(True)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)

loaders = get_loaders(model, batch_size=128, workers=8, seed=0)
retain_it = cycle(loaders["retain"])

base = fasteval.score(fasteval.load_ckpt(ckpt, device), device)
print(f"복구 전  CKA_f {base['CKA_f_o']:.4f}  CKA_r {base['CKA_r_o']:.4f}  final {base['final']:.5f}")

model.train()
done = 0
for target in checkpoints:
    for _ in range(target - done):
        x, y = next(retain_it)
        x, y = x.to(device, non_blocking=True), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                z_o, logit_o = forward(teacher, x)
            z_u, logit_u = forward(model, x)
            loss = 2 * (1 - F.cosine_similarity(z_u, z_o, dim=1)).mean()
            if KD:
                # logit KD는 100차원 분포 전체를 teacher에 맞추므로 retain 이미지만
                # 써도 forget 클래스 logit이 복원된다(600스텝에서 Acc_f 67.8%).
                loss = loss + 2 * F.kl_div(F.log_softmax(logit_u / 2, 1),
                                           F.softmax(logit_o / 2, 1), reduction="batchmean") * 4
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    done = target
    model.eval()
    r = fasteval.score(model, device)
    print(f"{target:5d}스텝  CKA_f {r['CKA_f_o']:.4f}  CKA_r {r['CKA_r_o']:.4f}  final {r['final']:.5f}  "
          f"({r['final']-base['final']:+.5f})", flush=True)
    out = f"{ckpt[:-3]}_rec{'' if KD else 'f'}{target}.pt"
    torch.save({"model": model.state_dict()}, out)
    model.train()
