"""실험 러너: GPU 직렬화 → 학습 → fasteval → EXPERIMENTS.md 기록.

GPU 1장을 둘이 공유하므로 flock으로 한 번에 하나만 돌게 강제한다.
큐에 넣으면 앞 실험이 끝날 때까지 알아서 기다린다.

    python tools/run_exp.py configs/r002.yaml
    python tools/run_exp.py configs/r00*.yaml        # 순차 실행
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

WS = Path(os.environ.get("STUDENT_WORKSPACE_ROOT", "."))
LOCK = Path("/tmp/hackathon_gpu.lock")
LOG = WS / "EXPERIMENTS.md"
HEADER = (
    "| 실험 | 설명 | Acc_f | Acc_r | CKA_f | CKA_r | AUS | RUS_o | **final** | 학습(s) | 시각 |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def headline(cfg_path):
    """config 첫 줄 주석 = 그 실험이 무엇인지."""
    first = Path(cfg_path).read_text().splitlines()[0].strip()
    return first.lstrip("#").strip() if first.startswith("#") else ""


def wait_for_gpu(limit_mib=8000, poll=30):
    """flock 밖에서 도는 남의 작업(팀원의 unlearn.py 등)이 GPU를 비울 때까지 대기.

    ponytail: 전체 메모리 사용량만 본다. PID별로 우리 것/남의 것을 가리려면
    복잡해지는데, 어차피 flock이 우리 큐를 하나로 묶어주므로 그럴 이유가 없다."""
    q = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    while int(subprocess.check_output(q, text=True).split()[0]) > limit_mib:
        print("  GPU 사용 중 — 대기", flush=True)
        time.sleep(poll)


def run(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    ckpt = cfg["output"]["save_path"]
    name = Path(cfg_path).stem
    script = cfg.get("script", "unlearn_remap.py")

    with open(LOCK, "w") as lock:
        print(f"[{name}] GPU 대기 중...", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        wait_for_gpu()
        print(f"[{name}] 학습 시작", flush=True)
        t0 = time.time()
        # 여러 큐가 같은 로그에 append하면 줄이 섞여 결과를 잘못 읽는 사고가 두 번
        # 났다. 실행별 로그를 따로 남겨 원본을 항상 확인할 수 있게 한다.
        train = subprocess.run([sys.executable, script, "--config", str(cfg_path)],
                               capture_output=True, text=True)
        Path("logs").mkdir(exist_ok=True)
        Path(f"logs/{name}.train.log").write_text(train.stdout + train.stderr)
        secs = round(time.time() - t0)
        if train.returncode != 0:
            print(train.stdout[-2000:], train.stderr[-2000:])
            raise SystemExit(f"[{name}] 학습 실패")
        print(train.stdout.strip().splitlines()[-1] if train.stdout.strip() else "", flush=True)

        # 채점은 반드시 별도 프로세스로. in-process로 하면 이 프로세스가 CUDA
        # 컨텍스트(약 2.6GB)를 계속 붙들어 wait_for_gpu가 자기 메모리를 보고
        # 영원히 대기하는 교착이 생긴다.
        out = subprocess.run([sys.executable, "tools/fasteval.py", ckpt, "--json"],
                             capture_output=True, text=True, check=True)
        r = json.loads(out.stdout.strip().splitlines()[-1])

    if not LOG.exists():
        LOG.write_text("# 실험 기록\n\n"
                       "`r###` = 이 라인의 실험, `s###` = 팀원 라인. "
                       "각 config 첫 줄 주석에 그 실험이 무엇인지 적는다.\n\n" + HEADER)
    with LOG.open("a") as f:
        f.write(f"| {name} | {headline(cfg_path)} | {r['Acc_f']:.2f} | {r['Acc_r']:.2f} | "
                f"{r['CKA_f_o']:.4f} | {r['CKA_r_o']:.4f} | {r['AUS']:.4f} | {r['RUS_o']:.4f} | "
                f"**{r['final']:.4f}** | {secs} | {datetime.now():%m-%d %H:%M} |\n")

    print(f"[{name}] Acc_f {r['Acc_f']:.2f}  Acc_r {r['Acc_r']:.2f}  "
          f"CKA_f {r['CKA_f_o']:.4f}  CKA_r {r['CKA_r_o']:.4f}  "
          f"AUS {r['AUS']:.4f}  RUS_o {r['RUS_o']:.4f}  final {r['final']:.4f}", flush=True)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+")
    a = p.parse_args()
    for c in a.configs:
        run(c)


if __name__ == "__main__":
    main()
