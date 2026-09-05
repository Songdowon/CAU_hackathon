"""최종 제출 tar.gz를 만든다. 모델을 바꿔도 한 줄로 다시 만들 수 있게 스크립트화.

    python tools/make_final_package.py --model models/mall_hf.pt \
        --config configs/ckar8m.yaml --private 0.99634

`--private`는 리더보드 실측값(모르면 생략). 로컬 점수는 results/의 score_model
결과 json에서 읽고, 없으면 fasteval로 계산한다.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ["unlearn_remap.py", "imagenet_vit.py", "train_ft.py",
        "validate_submission.py", "score_model.py"]
TOOLS = ["average_snapshots.py", "headfit.py", "fasteval.py"]


def scores(model):
    """score_model.py 결과가 있으면 그걸 쓰고, 없으면 fasteval로 잰다."""
    stem = Path(model).stem
    js = sorted(glob.glob(str(ROOT / f"results/{stem}-validation-*.json")))
    if js:
        d = json.load(open(js[-1]))
        acc = d["accuracy_metric"]
        cka = d["cka_per_depth"][d["score_depth"]]
        return dict(Acc_f=acc["Acc_f"], Acc_r=acc["Acc_r"],
                    CKA_f=cka["CKA_f_o"], CKA_r=cka["CKA_r_o"],
                    AUS=d["AUS"], RUS=d["RUS_o"], final=d["final_score"])
    sys.path.insert(0, str(ROOT / "tools"))
    import torch

    import fasteval
    dev = torch.device("cuda")
    r = fasteval.score(fasteval.load_ckpt(model, dev), dev)
    return dict(Acc_f=r["Acc_f"], Acc_r=r["Acc_r"], CKA_f=r["CKA_f_o"],
                CKA_r=r["CKA_r_o"], AUS=r["AUS"], RUS=r["RUS_o"], final=r["final"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--private", default=None, help="리더보드 실측 점수 (선택)")
    p.add_argument("--name", default="Team_노광탈_final")
    p.add_argument("--outdir", default=str(Path.home() / "submissions" / "final"))
    a = p.parse_args()
    os.chdir(ROOT)

    cfg = __import__("yaml").safe_load(open(a.config))
    lam = cfg["train"]["lambda_cka_r"]
    stem = Path(cfg["output"]["save_path"]).stem
    s = scores(a.model)

    block = (f"```\nAcc_f  {s['Acc_f']:.4f}    Acc_r  {s['Acc_r']:.4f}\n"
             f"CKA_f  {s['CKA_f']:.5f}   CKA_r  {s['CKA_r']:.5f}\n"
             f"AUS    {s['AUS']:.5f}   RUS_o  {s['RUS']:.5f}   "
             f"final_score  {s['final']:.5f}\n```")
    if a.private:
        block += f"\n\n리더보드(private) 실측: **{a.private}**"

    readme = Path("FINAL_README.template.md").read_text()
    for k, v in (("{{SCORES}}", block), ("{{LAMBDA}}", str(lam)),
                 ("{{STEM}}", stem), ("{{CFG}}", Path(a.config).name)):
        readme = readme.replace(k, v)

    dest = Path("/tmp") / a.name
    shutil.rmtree(dest, ignore_errors=True)
    for sub in ("configs", "tools", "utils", "splits", "es/es_imagenet_mo"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    (dest / "README.md").write_text(readme)
    shutil.copy(a.model, dest / "model.pt")
    shutil.copy(a.config, dest / "configs/final.yaml")
    for f in CODE:
        shutil.copy(f, dest / f)
    for f in TOOLS:
        shutil.copy(f"tools/{f}", dest / "tools" / f)
    for f in ("utils/__init__.py", "utils/data.py",
              "splits/student_split.pt", "es/es_imagenet_mo/forget10.json"):
        shutil.copy(f, dest / f)

    tar = Path(a.outdir) / f"{a.name}.tar.gz"
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-czf", str(tar), "-C", "/tmp", a.name], check=True)
    n = len(list(dest.rglob("*")))
    print(f"모델 {a.model}  (final {s['final']:.5f}, lambda_cka_r {lam})")
    print(f"-> {tar}  {tar.stat().st_size/2**20:.0f}MB, 파일 {n}개")


if __name__ == "__main__":
    main()
