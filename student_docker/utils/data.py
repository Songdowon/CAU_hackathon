"""`released` 학습 split에 대한 retain/forget 데이터로더.

M_o는 100개 클래스 전부로 학습된 모델이고, `released`는 여러분이 학습에
사용할 수 있는 유일한 split입니다. 이 split을 forget10.json의 클래스
목록에 따라 retain/forget으로 나눠서 제공합니다.

`validation`도 학생 서버에 제공되지만 로컬 점수 확인
전용입니다. 이 모듈은 validation split을 학습 DataLoader에 넣지
않습니다. 최종 순위에 쓰는 private `test` split은 학생 서버에 없습니다.
"""
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from imagenet_vit import build_train_transform
from train_ft import ListDataset, eval_transform


def load_split(split_pt="splits/student_split.pt",
               forget_json="es/es_imagenet_mo/forget10.json"):
    d = torch.load(split_pt, map_location="cpu", weights_only=True)
    meta, sp = d["meta"], d["splits"]
    wnids = meta["wnids"]
    fw = json.load(open(forget_json))
    fw = fw["wnid"] if isinstance(fw, dict) else fw
    forget_labels = sorted(wnids.index(w) for w in fw)
    dataset_root = os.environ.get("DATASET_ROOT")
    image_root = (
        Path(dataset_root) / "imagenet_released"
        if dataset_root
        else Path(meta["root"])
    )
    return str(image_root), wnids, forget_labels, sp["released"]


def get_loaders(
    model,
    batch_size=128,
    workers=8,
    seed=0,
    split_pt="splits/student_split.pt",
    forget_json="es/es_imagenet_mo/forget10.json",
):
    """model은 data_config(입력 크기 / mean / std)를 읽기 위해서만 사용합니다.
    train_ft.py의 transform이 내부에서 정규화를 수행하므로, 모델은 반드시
    in_model_norm=False로 만들어야 합니다. 그렇지 않으면 모든 이미지가 두 번
    정규화됩니다."""
    root, wnids, forget_labels, released = load_split(split_pt, forget_json)
    fl = set(forget_labels)
    forget_items = [it for it in released if it[1] in fl]
    retain_items = [it for it in released if it[1] not in fl]

    cfg = model.data_config
    train_tf, in_model_norm = build_train_transform(cfg)
    assert not in_model_norm, (
        "ViTWrapper(..., in_model_norm=False)로 생성하세요. "
        "build_train_transform이 자체적으로 정규화를 수행하므로 모델이 "
        "다시 정규화하면 안 됩니다")
    eval_tf = eval_transform(cfg)

    g = torch.Generator().manual_seed(seed)
    forget_dl = DataLoader(ListDataset(forget_items, root, train_tf),
                           batch_size=batch_size, shuffle=True,
                           num_workers=workers, pin_memory=True, generator=g)
    retain_dl = DataLoader(ListDataset(retain_items, root, train_tf),
                           batch_size=batch_size, shuffle=True,
                           num_workers=workers, pin_memory=True, generator=g)
    return {"forget": forget_dl, "retain": retain_dl,
           "forget_labels": forget_labels, "root": root, "wnids": wnids,
           "eval_transform": eval_tf}
