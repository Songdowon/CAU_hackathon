"""큐를 일시정지한다. run_exp.py는 락을 잡은 뒤 GPU 메모리가 8000MiB 아래로
내려가야 학습을 시작하므로, 여기서 메모리를 붙들고 있으면 다음 실험이 대기한다.
재개하려면 이 프로세스를 끝내면 된다(파일 tools/.gpu_hold 삭제).
"""
import os, time, torch
from pathlib import Path
flag = Path(__file__).with_name(".gpu_hold"); flag.write_text(str(os.getpid()))
buf = torch.empty(int(9.5e9 // 2), dtype=torch.float16, device="cuda")  # 약 9.5GB
print(f"큐 일시정지 (pid {os.getpid()}) — 해제: rm {flag}", flush=True)
while flag.exists():
    time.sleep(5)
del buf
print("해제됨 — 큐 재개")
