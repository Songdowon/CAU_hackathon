"""빠른 로컬 채점기.

score_model.py는 매번 15,000장 JPEG을 디코딩하느라 1.5~4분이 걸린다. 여기서는
grader와 동일한 eval transform으로 한 번만 디코딩해 fp16 캐시로 저장해두고,
이후에는 캐시에서 바로 forward만 돌려 수십 초 안에 같은 지표를 계산한다.

지표 정의는 grading_docker/score_unlearning.py를 그대로 옮긴 것이다.

    python tools/fasteval.py --build-cache            # 최초 1회 (~5분, 4.5GB)
    python tools/fasteval.py models/experiment-001.pt
    python tools/fasteval.py models/ga_example.pt --parity
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imagenet_vit import ViTWrapper
from train_ft import ListDataset, eval_transform

WS = Path(os.environ.get("STUDENT_WORKSPACE_ROOT", "."))
DS = Path(os.environ.get("DATASET_ROOT", WS))
CACHE = WS / "cache"
PX = CACHE / "val_px_fp16.npy"
LB = CACHE / "val_labels.npy"


def _val_items():
    d = torch.load(DS / "splits/student_split.pt", map_location="cpu", weights_only=True)
    return d["splits"]["validation"], str(DS / "imagenet_released")


def build_cache(workers=12, batch=100):
    """validation 15,000장을 grader와 동일한 transform으로 디코딩해 fp16 memmap에 적재."""
    items, root = _val_items()
    m = ViTWrapper(num_classes=100, pretrained=False, drop_path_rate=0.0, in_model_norm=False)
    dl = DataLoader(ListDataset(items, root, eval_transform(m.data_config)),
                    batch_size=batch, shuffle=False, num_workers=workers)
    CACHE.mkdir(exist_ok=True)
    px = np.lib.format.open_memmap(PX, mode="w+", dtype=np.float16,
                                   shape=(len(items), 3, 224, 224))
    labels = np.zeros(len(items), dtype=np.int64)
    i, t0 = 0, time.time()
    for x, y in dl:
        n = x.shape[0]
        px[i:i + n] = x.numpy().astype(np.float16)
        labels[i:i + n] = y.numpy()
        i += n
        if i % 2000 == 0:
            print(f"  {i}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    px.flush()
    np.save(LB, labels)
    print(f"캐시 완료: {PX} ({PX.stat().st_size / 2**30:.2f} GiB), {time.time() - t0:.0f}s")


def linear_cka(left, right):
    """grading_docker/score_unlearning.py:285-297과 동일."""
    left64 = left.astype(np.float64, copy=True)
    right64 = right.astype(np.float64, copy=True)
    left64 -= left64.mean(axis=0, keepdims=True)
    right64 -= right64.mean(axis=0, keepdims=True)
    denominator = np.linalg.norm(left64.T @ left64) * np.linalg.norm(right64.T @ right64)
    if denominator <= 0:
        return 0.0
    value = ((left64.T @ right64) ** 2).sum() / denominator
    return float(min(max(value, 0.0), 1.0))


def harmonic(a, b):
    return 0.0 if a <= 0 or b <= 0 else 2 * a * b / (a + b)


@torch.no_grad()
def extract(model, device, batch=256, index=None):
    """캐시된 픽셀에서 pre-logits feature와 예측을 뽑는다 (grader와 동일한 경로)."""
    px = np.load(PX, mmap_mode="r")
    labels = np.load(LB)
    if index is not None:
        labels = labels[index]
    n = len(index) if index is not None else px.shape[0]
    feats = np.empty((n, 768), dtype=np.float32)
    preds = np.empty(n, dtype=np.int64)
    model.eval()
    for i in range(0, n, batch):
        sl = index[i:i + batch] if index is not None else slice(i, min(i + batch, n))
        x = torch.from_numpy(np.ascontiguousarray(px[sl])).to(device, non_blocking=True).float()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            encoded = model.backbone.forward_features(x)
            pre = model.backbone.forward_head(encoded, pre_logits=True)
            logits = model.head(pre)
        k = x.shape[0]
        feats[i:i + k] = pre.float().cpu().numpy()
        preds[i:i + k] = logits.float().argmax(1).cpu().numpy()
    return feats, preds, labels


def score(model, device, refs=None, mo_pre=None, index=None, batch=256):
    """AUS / RUS_o / final을 계산한다. index로 부분집합(A/B half, quick) 평가 가능."""
    if refs is None:
        refs = torch.load(DS / "validation_cache/refs.pt", map_location="cpu", weights_only=True)
    if mo_pre is None:
        with np.load(DS / "validation_cache/M_o__validation.npz") as z:
            mo_pre = z["f_pre"]
    mo = mo_pre if index is None else mo_pre[index]

    feats, preds, labels = extract(model, device, batch=batch, index=index)
    fmask = np.isin(labels, refs["forget_labels"])
    hit = preds == labels
    acc_f = 100.0 * hit[fmask].sum() / max(fmask.sum(), 1)
    acc_r = 100.0 * hit[~fmask].sum() / max((~fmask).sum(), 1)

    cka_f = linear_cka(feats[fmask], mo[fmask])
    cka_r = linear_cka(feats[~fmask], mo[~fmask])

    ref = refs["reference_accuracy"]
    retain_drop = max(ref["acc_r"] - acc_r, 0.0) / 100
    forget_gap = abs(acc_f - ref["acc_f"]) / 100
    aus = (1 - retain_drop) / (1 + forget_gap)
    rus = harmonic(1 - cka_f, cka_r)
    return {"Acc_f": float(acc_f), "Acc_r": float(acc_r),
            "CKA_f_o": cka_f, "CKA_r_o": cka_r,
            "AUS": float(aus), "RUS_o": float(rus), "final": float(harmonic(aus, rus))}


def load_ckpt(path, device):
    m = ViTWrapper(num_classes=100, pretrained=False, drop_path_rate=0.0, in_model_norm=False)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m.load_state_dict(sd.get("model", sd), strict=True)
    return m.to(device).eval()


def half_index(which):
    """validation을 클래스별로 절반씩 나눈 A/B 인덱스 (튜닝/검증 분리용)."""
    labels = np.load(LB)
    idx = []
    for c in np.unique(labels):
        pos = np.flatnonzero(labels == c)
        idx.append(pos[::2] if which == "a" else pos[1::2])
    return np.sort(np.concatenate(idx))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", nargs="?")
    p.add_argument("--build-cache", action="store_true")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--half", choices=["a", "b"])
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--parity", action="store_true",
                   help="ga_example.pt 기준값과 대조 (AUS .08060857 / RUS_o .21909864 / final .11785655)")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    if a.build_cache:
        build_cache(workers=a.workers)
        if not a.ckpt:
            return
    if not a.ckpt:
        p.error("체크포인트 경로가 필요합니다")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    r = score(load_ckpt(a.ckpt, device), device,
              index=half_index(a.half) if a.half else None, batch=a.batch)
    r["ckpt"] = a.ckpt
    r["half"] = a.half or "full"
    r["seconds"] = round(time.time() - t0, 1)

    if a.json:
        print(json.dumps(r))
    else:
        print(f"{a.ckpt}  [{r['half']}]  {r['seconds']}s")
        print(f"  Acc_f  {r['Acc_f']:8.4f}   Acc_r  {r['Acc_r']:8.4f}")
        print(f"  CKA_f  {r['CKA_f_o']:8.5f}   CKA_r  {r['CKA_r_o']:8.5f}")
        print(f"  AUS    {r['AUS']:8.5f}   RUS_o  {r['RUS_o']:8.5f}   final  {r['final']:8.5f}")

    if a.parity:
        exp = {"AUS": 0.08060857128766592, "RUS_o": 0.219098640712386,
               "final": 0.11785654593384969, "Acc_f": 0.13333333333333333,
               "Acc_r": 3.9703703703703703, "CKA_f_o": 0.20568628148331863,
               "CKA_r_o": 0.12707513452343466}
        # 픽셀 캐시가 fp16이라 경계선 샘플 몇 장의 예측이 흔들린다. 정확도는
        # %p 단위(허용 0.05), 점수/CKA는 [0,1] 단위(허용 1e-3)로 따로 본다.
        print("\nparity vs score_model.py:")
        ok = True
        for k, v in exp.items():
            tol = 0.05 if k.startswith("Acc") else 1e-3
            d = abs(r[k] - v)
            ok &= d < tol
            print(f"  {k:8s} got {r[k]:.6f}  expected {v:.6f}  diff {d:.2e}  (tol {tol})")
        print("  => " + ("일치 (실험 비교용으로 신뢰 가능)" if ok
                         else "불일치 — fasteval을 신뢰하지 말 것"))


if __name__ == "__main__":
    main()
