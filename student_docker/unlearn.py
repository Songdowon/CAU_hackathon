"""Unlearning 스켈레톤 코드.

알고리즘 본체를 제외한 나머지는 전부 미리 연결해 두었습니다.
(config/seed 설정, M_o 로딩, retain/forget 데이터로더, 채점 서버가 요구하는
저장 포맷) 아래 TODO 블록만 채우면 됩니다.

    python unlearn.py --config configs/unlearn.yaml
    python validate_submission.py --ckpt models/experiment-001.pt
    python score_model.py models/experiment-001.pt
"""
import argparse
import os
import random

import numpy as np
import torch
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

    # ---------------------------------------------------------------- #
    # TODO: 여기에 본인의 unlearning 방법을 구현하세요.
    #
    #   loaders["forget"]         -- 잊어야 할 클래스들의 DataLoader (image, label)
    #   loaders["retain"]         -- 유지해야 할 클래스들의 DataLoader (image, label)
    #   loaders["forget_labels"]  -- 잊어야 할 10개 클래스의 라벨 id 리스트
    #   model                     -- M_o 상태로 시작합니다. 이 모델을 그대로
    #                                학습시키거나, state_dict를 복사해 새 모델을
    #                                만들어 사용해도 됩니다.
    #
    # 점수는 (1) forget 클래스를 얼마나 잘 지웠는지와 (2) retain 클래스 성능을
    # 얼마나 잘 보존했는지를 함께 봅니다. 자세한 지표 정의는 README.md의
    # "평가 지표" 절을 참고하세요.
    #
    # 주의: classifier head만 건드려서 forget 클래스의 logit을 숨기는 방식은
    # 정확도상으로는 완벽해 보여도, 모델 내부 표현(representation)을 실제로
    # 바꾸지 않았기 때문에 표현 유사도 지표(RUS)에서 매우 낮은 점수를 받습니다.
    # 어떤 클래스를 잊어야 하는지, 현재 M_o가 그 클래스들을 어떻게 분류하고
    # 있는지는 walkthrough.ipynb에서 직접 확인할 수 있습니다.
    # ---------------------------------------------------------------- #
    raise NotImplementedError("위 TODO 블록에 unlearning 방법을 구현하세요")

    os.makedirs(os.path.dirname(cfg["output"]["save_path"]) or ".", exist_ok=True)
    torch.save({"model": model.state_dict()}, cfg["output"]["save_path"])
    print(f"저장 완료: {cfg['output']['save_path']}")
    print(f"구조 검사: python validate_submission.py --ckpt {cfg['output']['save_path']}")
    print(f"로컬 점수: python score_model.py {cfg['output']['save_path']}")


if __name__ == "__main__":
    main()
