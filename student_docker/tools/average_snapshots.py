"""한 학습 궤적에서 저장된 스냅샷들의 가중치를 균등 평균한다.

주의: **같은 궤적의 스냅샷끼리만** 평균해야 한다. seed가 다른 궤적을 섞으면
forget 삭제 방향이 서로 달라 상쇄되고 CKA_f가 0.0044에서 0.2205로 붕괴하는 것을
실측했다. 같은 궤적 안에서는 방향이 공유되어 CKA_f와 CKA_r이 함께 개선된다.

    python tools/average_snapshots.py models/ckar8m_s*.pt --out models/mall.pt
"""
import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    sds = []
    for path in a.ckpts:
        d = torch.load(path, map_location="cpu", weights_only=True)
        sds.append(d.get("model", d))
    avg = {k: (sum(sd[k].float() for sd in sds) / len(sds)).to(sds[0][k].dtype)
           for k in sds[0]}
    torch.save({"model": avg}, a.out)
    print(f"{len(sds)}개 균등 평균 -> {a.out}")


if __name__ == "__main__":
    main()
