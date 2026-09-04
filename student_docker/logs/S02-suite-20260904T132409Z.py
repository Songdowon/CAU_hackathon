"""One-off S02 queue; reuses tools.run_exp.run and its shared GPU lock."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path('/root/cau-ai-hackathon-26/student_docker')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from tools import run_exp

RUN_ID = Path(__file__).stem
STATE = ROOT / 'logs' / (RUN_ID + '.state.json')
CONFIGS = ['configs/S02-3.yaml', 'configs/S02.yaml', 'configs/S02-1.yaml', 'configs/S02-2.yaml', 'configs/S02-4.yaml', 'configs/S02-5.yaml', 'configs/S02-6.yaml', 'configs/S02-7.yaml']
state = {'run_id': RUN_ID, 'pid': os.getpid(), 'configs': CONFIGS, 'completed': [], 'status': 'queued'}
def update(**values):
    state.update(values, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(STATE)
    print(json.dumps(values, ensure_ascii=False), flush=True)

with open('/tmp/S02-suite.lock', 'a') as suite_lock:
    fcntl.flock(suite_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        update(status='probe_queued_or_running', current='S02-probe')
        subprocess.run([sys.executable, '-u', 'S02.py', '--config', 'configs/S02.yaml', '--probe-only'], check=True)
        for config in CONFIGS:
            name = Path(config).stem
            result_path = ROOT / 'results' / (name + '.validation.json')
            if result_path.exists():
                raise FileExistsError(f'Preserve existing evaluation: {result_path}')
            update(status='training_or_waiting_for_gpu', current=name)
            metrics = run_exp.run(config)
            record = {'experiment': name, 'evaluation': 'local public validation via tools/fasteval.py', 'metrics': metrics,
                      'evaluated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'fasteval_sha256': hashlib.sha256((ROOT/'tools/fasteval.py').read_bytes()).hexdigest()}
            with result_path.open('x') as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False, default=float)
                stream.write('\n')
            state['completed'].append(name)
            update(last_result=str(result_path.relative_to(ROOT)))
        update(status='completed', current=None)
    except BaseException as error:
        update(status='failed', error=type(error).__name__ + ': ' + str(error))
        traceback.print_exc()
        raise
