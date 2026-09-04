"""Experiment S01: Retain CE + NegGrad.

M_o에서 시작해 retain CE - alpha * forget CE를 최소화합니다.
하이퍼파라미터와 저장 경로는 configs/unlearn.yaml에서 관리합니다.

    python unlearn.py --config configs/unlearn.yaml
    python validate_submission.py --ckpt models/S01_retain_ce_neggrad.pt
    python score_model.py models/S01_retain_ce_neggrad.pt
"""
import argparse
import os
import random

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
    """채점 서버가 사용하는 것과 동일한 로더(strict=True)입니다.
    최종 체크포인트가 이 방식으로 로드되지 않으면 여기가 아니라 제출 시점에
    실패하므로, 반드시 validate_submission.py로 미리 확인하세요."""
    m = ViTWrapper(num_classes=num_classes, pretrained=False,
                   drop_path_rate=0.0, in_model_norm=False)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(sd.get("model", sd), strict=True)
    return m.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/unlearn.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_mo(cfg["model"]["mo_ckpt"], cfg["model"]["num_classes"], device)
    loaders = get_loaders(model, batch_size=cfg["data"]["batch_size"],
                          workers=cfg["data"]["workers"], seed=cfg["seed"],
                          split_pt=cfg["data"]["split"],
                          forget_json=cfg["data"]["forget"])

    # ============================================================
    # Experiment S01
    # Title: Retain CE + NegGrad
    # Goal: Forget 성능을 유지하면서 retain accuracy 붕괴를 완화
    # Change:
    # - forget set: NegGrad (gradient ascent)
    # - retain set: normal CE
    # - forget loss weight alpha = 0.1
    # ============================================================
    train_cfg = cfg["train"]
    if train_cfg["optimizer"] != "AdamW":
        raise ValueError("Experiment S01 requires train.optimizer: AdamW")
    epochs = int(train_cfg["epochs"])
    alpha = float(train_cfg["forget_weight"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]))

    if len(loaders["forget"]) == 0 or len(loaders["retain"]) == 0:
        raise ValueError("S01 requires non-empty forget and retain loaders")

    model.train()
    for epoch in range(epochs):
        retain_iter = iter(loaders["retain"])
        total_sum = retain_sum = forget_sum = 0.0
        batches = 0
        for x_forget, y_forget in loaders["forget"]:
            try:
                x_retain, y_retain = next(retain_iter)
            except StopIteration:
                retain_iter = iter(loaders["retain"])
                x_retain, y_retain = next(retain_iter)

            x_retain, y_retain = x_retain.to(device), y_retain.to(device)
            x_forget, y_forget = x_forget.to(device), y_forget.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                retain_loss = F.cross_entropy(model(x_retain), y_retain)
                forget_loss = F.cross_entropy(model(x_forget), y_forget)
                total_loss = retain_loss - alpha * forget_loss
            total_loss.backward()
            optimizer.step()

            total_sum += total_loss.item()
            retain_sum += retain_loss.item()
            forget_sum += forget_loss.item()
            batches += 1

        print(f"[S01] epoch {epoch + 1}/{epochs} "
              f"total_loss={total_sum / batches:.6f} "
              f"retain_ce={retain_sum / batches:.6f} "
              f"forget_ce={forget_sum / batches:.6f} "
              f"batches={batches}", flush=True)

    os.makedirs(os.path.dirname(cfg["output"]["save_path"]) or ".", exist_ok=True)
    torch.save({"model": model.state_dict()}, cfg["output"]["save_path"])
    print(f"저장 완료: {cfg['output']['save_path']}")
    print(f"구조 검사: python validate_submission.py --ckpt {cfg['output']['save_path']}")
    print(f"로컬 점수: python score_model.py {cfg['output']['save_path']}")


if __name__ == "__main__":
    main()
