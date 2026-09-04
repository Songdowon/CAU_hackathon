"""Run one S03 trajectory and evaluate all four variants on local validation."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_exp
import yaml


def write_json(path,data,exclusive=False):
    text=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+chr(10)
    path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with path.open('x') as f: f.write(text)
    else:
        temp=path.with_suffix(path.suffix+'.tmp')
        temp.write_text(text)
        temp.replace(path)


def gpu_command(args):
    with open(run_exp.LOCK,'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run_exp.wait_for_gpu()
        return subprocess.run(args,cwd=ROOT,check=True,capture_output=True,text=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/S03.yaml')
    args=parser.parse_args()
    os.chdir(ROOT)
    cfg=yaml.safe_load(Path(args.config).read_text())
    name=Path(args.config).stem
    paths={'avg':cfg['output']['save_path'],'last':cfg['output']['last_path'],**cfg['output']['ema_paths']}
    state_path=ROOT/'logs'/(name+'.state.json')
    comparison_path=ROOT/'results'/(name+'.comparison.json')
    comparison_md=comparison_path.with_suffix('.md')
    targets=[Path(p) for p in paths.values()]+[Path(cfg['output']['trajectory_report']),Path(cfg['averaging']['snapshot_dir']),Path(cfg['output']['training_log']),state_path,comparison_path,comparison_md]
    if any(p.exists() for p in targets):
        raise FileExistsError('Preserve existing experiment: choose a new run ID')
    state={'experiment':name,'pid':os.getpid(),'status':'queued','config':args.config,'completed_evaluations':[]}
    write_json(state_path,state,exclusive=True)
    comparison={'experiment':name,'evaluation':'local public validation, no private submission',
                'variants':{},'status':'queued','run_exp_sha256':hashlib.sha256(Path(run_exp.__file__).read_bytes()).hexdigest()}
    write_json(comparison_path,comparison,exclusive=True)
    def update(status,**values):
        state.update(status=status,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**values)
        comparison['status']=status
        write_json(state_path,state)
        write_json(comparison_path,comparison)
        print(json.dumps({'status':status,**values},ensure_ascii=False),flush=True)
    try:
        update('training_or_waiting_for_gpu')
        primary=run_exp.run(args.config)
        comparison['variants']['avg']={'checkpoint':paths['avg'],'fast':primary}
        for key,path in paths.items():
            update('fast_evaluation',current=key)
            if key!='avg':
                result=gpu_command([sys.executable,'tools/fasteval.py',path,'--json'])
                metrics=json.loads(result.stdout.strip().splitlines()[-1])
                comparison['variants'][key]={'checkpoint':path,'fast':metrics}
            metrics=comparison['variants'][key]['fast']
            write_json(ROOT/'results'/(name+'-'+key+'.validation.json'),
                       {'experiment':name,'variant':key,'checkpoint':path,'evaluation':'fasteval public validation','metrics':metrics},exclusive=True)
            if key!='avg':
                with run_exp.LOG.open('a') as f:
                    f.write(f"| {name}-{key} | Same S03 trajectory; no additional training | {metrics['Acc_f']:.2f} | {metrics['Acc_r']:.2f} | {metrics['CKA_f_o']:.4f} | {metrics['CKA_r_o']:.4f} | {metrics['AUS']:.4f} | {metrics['RUS_o']:.4f} | **{metrics['final']:.4f}** | 0 | {datetime.datetime.now():%m-%d %H:%M} |"+chr(10))
            update('fast_evaluation',current=key)
        for key,path in paths.items():
            update('score_model_evaluation',current=key)
            scored=gpu_command([sys.executable,'score_model.py',path])
            print(scored.stdout,flush=True)
            lines=[l.split(':',1)[1].strip() for l in scored.stdout.splitlines() if l.startswith('report ') and ':' in l]
            if len(lines)!=1: raise RuntimeError('Cannot locate score_model report')
            report=ROOT/lines[0]
            payload=json.loads(report.read_text())
            if payload['tag']!=Path(path).name or payload['phase']!='validation':
                raise ValueError('Unexpected scorer result identity or split')
            comparison['variants'][key]['score_model']={'report':str(report.relative_to(ROOT)),'metrics':payload}
            state['completed_evaluations'].append(key)
            update('score_model_evaluation',current=key)
        control=comparison['variants']['last']['score_model']['metrics']['final_score']
        for result in comparison['variants'].values():
            result['delta_vs_last']=result['score_model']['metrics']['final_score']-control
        comparison['best_local_variant']=max(comparison['variants'],key=lambda k:comparison['variants'][k]['score_model']['metrics']['final_score'])
        rows=['# '+name+' trajectory comparison','',
              'One r017-matched trajectory; all results below use local score_model.py public validation. Private improvement is unverified.','',
              '| Variant | Acc_f | Acc_r | CKA_f | CKA_r | Final | Delta vs last |',
              '|---|---:|---:|---:|---:|---:|---:|']
        for key,result in comparison['variants'].items():
            p=result['score_model']['metrics']; a=p['accuracy_metric']; r=p['representation_metric']
            rows.append(f"| {key} | {a['Acc_f']:.4f} | {a['Acc_r']:.4f} | {r['CKA_f_o']:.6f} | {r['CKA_r_o']:.6f} | {p['final_score']:.8f} | {result['delta_vs_last']:+.8f} |")
        with comparison_md.open('x') as f: f.write(chr(10).join(rows)+chr(10))
        trajectory=Path(cfg['output']['trajectory_report'])
        record=json.loads(trajectory.read_text())
        record.update(status='local_evaluation_completed',comparison_report=str(comparison_path.relative_to(ROOT)))
        write_json(trajectory,record)
        update('completed',current=None)
        with (ROOT/'S_EXPERIMENTS.md').open('a') as f:
            f.write(chr(10)+'### '+name+' 실행 완료'+chr(10)+chr(10)+'평가 완료: '+str(comparison_md.relative_to(ROOT))+'. 같은 학습의 last 대비 차이를 모든 후보에 기록했다. private 개선 여부는 미검증이다.'+chr(10))
    except BaseException as error:
        update('failed',error=type(error).__name__+': '+str(error))
        traceback.print_exc()
        raise


if __name__=='__main__': main()
