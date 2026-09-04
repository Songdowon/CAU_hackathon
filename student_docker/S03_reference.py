"""Unlearning: retain 표현은 M_o에 고정하고, forget 표현만 retain 분포로 재매핑한다.

(팀원의 S01이 unlearn.py를 쓰고 있어 별도 파일로 둔다. 실행 방식과 저장 포맷은 동일.)

동기 — grading_docker/score_unlearning.py의 채점식에서 직접 유도:

    AUS   = (1 - max(ref_acc_r - Acc_r, 0)/100) / (1 + Acc_f/100)
    RUS_o = harmonic(1 - CKA_f_o, CKA_r_o)      # depth 'pre' = pre-logits CLS 768d
    final = harmonic(AUS, RUS_o)

CKA는 isotropic scaling과 orthogonal 변환에 불변이므로, forget feature를 그냥
작게 줄이거나 회전시키는 방식은 CKA_f_o가 1 근처로 남아 최종 0점이 된다. forget
샘플들 *사이의* 2차 구조 자체를 바꿔야 한다. 그래서 forget 이미지를 매 스텝
무작위로 짝지은 retain 이미지의 teacher feature/label로 끌어당긴다. 구조가 retain
쪽으로 대체되어 원본과 decorrelate되고, feature가 분포 안에 머물러 발산하지
않으며, 예측이 retain 클래스로 가므로 Acc_f도 0으로 간다.

동시에 retain 배치에서는 teacher(M_o)와의 feature 코사인 + logit KD로 표현을
붙잡아 CKA_r_o와 Acc_r을 지킨다.

    python unlearn_remap.py --config configs/remap.yaml
    python validate_submission.py --ckpt models/R01.pt
    python tools/fasteval.py models/R01.pt      # 빠른 로컬 점수
    python score_model.py models/R01.pt         # 실제 grader (제출 전 필수)
"""
import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from imagenet_vit import ViTWrapper
from utils.data import get_loaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mo(ckpt, num_classes, device):
    """채점 서버가 사용하는 것과 동일한 로더(strict=True)."""
    m = ViTWrapper(num_classes=num_classes, pretrained=False,
                   drop_path_rate=0.0, in_model_norm=False)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(sd.get("model", sd), strict=True)
    return m.to(device)


def set_trainable(model, k):
    """뒤쪽 k개 블록 + final norm + head만 학습 (k < 0이면 전체).

    NegGrad 베이스라인에서도 b4/b8의 CKA는 0.93/0.89로 멀쩡했고 'pre'만 0.21로
    무너졌다. 손상이 후반부에 집중되므로 앞단을 얼리면 retain을 싸게 지킬 수 있다."""
    if k is not None and k >= 0:
        for p in model.parameters():
            p.requires_grad_(False)
        blocks = model.backbone.blocks
        for blk in blocks[max(len(blocks) - k, 0):]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for mod in (model.backbone.norm, model.backbone.head):
            for p in mod.parameters():
                p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def batch_linear_cka(x, y):
    """채점식과 동일한 centered linear CKA의 미분 가능한 미니배치 판."""
    x = x.float() - x.float().mean(0, keepdim=True)
    y = y.float() - y.float().mean(0, keepdim=True)
    denom = torch.linalg.norm(x.T @ x) * torch.linalg.norm(y.T @ y)
    return ((x.T @ y) ** 2).sum() / (denom + 1e-12)


def forward(model, x):
    """grader와 동일한 경로: pre-logits feature와 logit을 함께 얻는다."""
    encoded = model.backbone.forward_features(x)
    pre = model.backbone.forward_head(encoded, pre_logits=True)
    return pre, model.head(pre)


def main(config=None, after_step=None):
    if config is None:
        p = argparse.ArgumentParser()
        p.add_argument("--config", default="configs/remap.yaml")
        args = p.parse_args()
        cfg = yaml.safe_load(open(args.config))
    else:
        cfg = config
    set_seed(cfg["seed"])
    t = cfg["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_mo(cfg["model"]["mo_ckpt"], cfg["model"]["num_classes"], device)
    teacher = copy.deepcopy(model).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)

    loaders = get_loaders(model, batch_size=cfg["data"]["batch_size"],
                          workers=cfg["data"]["workers"], seed=cfg["seed"],
                          split_pt=cfg["data"]["split"],
                          forget_json=cfg["data"]["forget"])

    params = set_trainable(model, t.get("trainable_blocks", -1))
    total = sum(q.numel() for q in model.parameters())
    print(f"학습 파라미터 {sum(q.numel() for q in params)/1e6:.1f}M / 전체 {total/1e6:.1f}M")

    opt = getattr(torch.optim, t.get("optimizer", "AdamW"))(
        params, lr=float(t["lr"]), weight_decay=float(t.get("weight_decay", 0.0)))

    retain_it, forget_it = cycle(loaders["retain"]), cycle(loaders["forget"])
    steps = int(t["steps"])
    temp = float(t.get("kd_temperature", 2.0))
    w = {k: float(t.get(f"lambda_{k}", d)) for k, d in
         [("feat_r", 1.0), ("kd_r", 1.0), ("ce_r", 0.0),
          ("remap_f", 1.0), ("ce_f", 0.5), ("cka_f", 0.0)]}

    model.train()
    t0 = time.time()
    for step in range(steps):
        x_r, y_r = next(retain_it)
        x_f, _ = next(forget_it)
        x_r, y_r = x_r.to(device, non_blocking=True), y_r.to(device)
        x_f = x_f.to(device, non_blocking=True)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            with torch.no_grad():
                z_r_o, logit_r_o = forward(teacher, x_r)
                z_f_o, _ = forward(teacher, x_f)
            z_r_u, logit_r_u = forward(model, x_r)
            z_f_u, logit_f_u = forward(model, x_f)

            # forget 이미지마다 retain 배치에서 무작위 파트너를 뽑아 그 teacher
            # feature/label을 목표로 삼는다 (매 스텝 새로 뽑히므로 구조가 섞인다).
            perm = torch.randint(0, x_r.shape[0], (x_f.shape[0],), device=device)
            tgt_z, tgt_y = z_r_o[perm], y_r[perm]

            l_feat_r = (1 - F.cosine_similarity(z_r_u, z_r_o, dim=1)).mean()
            l_kd_r = F.kl_div(F.log_softmax(logit_r_u / temp, 1),
                              F.softmax(logit_r_o / temp, 1),
                              reduction="batchmean") * temp ** 2
            l_ce_r = F.cross_entropy(logit_r_u, y_r)
            l_remap_f = (1 - F.cosine_similarity(z_f_u, tgt_z, dim=1)).mean()
            l_ce_f = F.cross_entropy(logit_f_u, tgt_y)
            l_cka_f = batch_linear_cka(z_f_u, z_f_o) if w["cka_f"] else z_f_u.sum() * 0

            loss = (w["feat_r"] * l_feat_r + w["kd_r"] * l_kd_r + w["ce_r"] * l_ce_r
                    + w["remap_f"] * l_remap_f + w["ce_f"] * l_ce_f + w["cka_f"] * l_cka_f)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if t.get("clip_grad"):
            torch.nn.utils.clip_grad_norm_(params, float(t["clip_grad"]))
        opt.step()
        if after_step is not None:
            after_step(model, step + 1)

        if step % 50 == 0 or step == steps - 1:
            print(f"step {step:4d}/{steps} loss {loss.item():.4f} | feat_r {l_feat_r.item():.4f} "
                  f"kd_r {l_kd_r.item():.4f} remap_f {l_remap_f.item():.4f} "
                  f"ce_f {l_ce_f.item():.4f} cka_f {float(l_cka_f):.4f} | {time.time()-t0:.0f}s",
                  flush=True)

    if config is not None:
        return model

    out = cfg["output"]["save_path"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict()}, out)
    print(f"저장 완료: {out}  ({time.time()-t0:.0f}s)")
    print(f"로컬 점수: python tools/fasteval.py {out}")


if __name__ == "__main__":
    main()
