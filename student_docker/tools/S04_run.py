"""S04: locked CUDA smoke, matched control, gradient surgery and local validation."""
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
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+chr(10)
    if exclusive:
        with path.open('x') as stream: stream.write(text)
    else:
        temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(text); temp.replace(path)


def gpu_command(args,on_ready):
    with open(run_exp.LOCK,'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run_exp.wait_for_gpu()
        on_ready()
        result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
        print(result.stdout,flush=True)
        if result.stderr: print(result.stderr,file=sys.stderr,flush=True)
        result.check_returncode()
        return result


def full_metrics(payload):
    a=payload['accuracy_metric']; r=payload['representation_metric']
    return {'Acc_f':a['Acc_f'],'Acc_r':a['Acc_r'],'CKA_f_o':r['CKA_f_o'],'CKA_r_o':r['CKA_r_o'],
            'AUS':payload['AUS'],'RUS_o':payload['RUS_o'],'final':payload['final_score']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    os.chdir(ROOT)
    names=['S04-control','S04']
    configs={name:yaml.safe_load(Path('configs',name+'.yaml').read_text()) for name in names}
    comparable=[]
    for name,cfg in configs.items():
        if cfg['script']!='S04.py': raise ValueError('Unexpected training script')
        comparable.append({k:v for k,v in cfg.items() if k not in ('output','surgery')})
    if comparable[0]!=comparable[1]: raise ValueError('Control and surgery training settings differ')
    if configs['S04-control']['surgery']!={**configs['S04']['surgery'],'enabled':False} or configs['S04']['surgery']['enabled'] is not True:
        raise ValueError('Only projection enablement may differ')
    smoke_paths=['models/S04-smoke.pt','results/S04-smoke.run.json','logs/S04-smoke.training.log','logs/S04-smoke.gradients.csv']
    state_path=ROOT/'logs/S04.state.json'; comparison_path=ROOT/'results/S04.comparison.json'; comparison_md=comparison_path.with_suffix('.md')
    targets=[state_path,comparison_path,comparison_md]+[ROOT/p for p in smoke_paths]
    for name,cfg in configs.items():
        targets.extend(ROOT/p for p in cfg['output'].values())
        targets.append(ROOT/'results'/(name+'.validation.json'))
    if len({p.resolve() for p in targets})!=len(targets) or any(p.exists() for p in targets):
        raise FileExistsError('Existing or overlapping S04 artifacts: preserve and choose a new run ID')
    tracked=['S04.py','S04_reference.py','tools/S04_run.py','tools/run_exp.py','tools/fasteval.py','score_model.py','configs/S04.yaml','configs/S04-control.yaml']
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in tracked}
    state={'experiment':'S04','pid':os.getpid(),'status':'queued','order':names,'completed_evaluations':[]}
    comparison={'experiment':'S04','evaluation':'local public validation; no private submission','status':'queued',
                'method':'retain-prioritized one-way global gradient projection',
                'limitation':'Raw-gradient projection does not guarantee AdamW descent, CKA_r improvement or private transfer.',
                'source_sha256':hashes,'variants':{}}
    write_json(state_path,state,exclusive=True); write_json(comparison_path,comparison,exclusive=True)
    def update(status,**values):
        state.update(status=status,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**values)
        comparison['status']=status
        write_json(state_path,state); write_json(comparison_path,comparison)
        print(json.dumps({'status':status,**values},ensure_ascii=False),flush=True)
    def verify_sources():
        for path,digest in hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Source changed after S04 launch: '+path)
    try:
        update('smoke_or_waiting_for_gpu',current='S04-smoke')
        def smoke_ready():
            verify_sources(); update('gpu_smoke_running',current='S04-smoke')
        gpu_command([sys.executable,'S04.py','--config','configs/S04.yaml','--smoke'],smoke_ready)
        smoke=json.loads(Path('results/S04-smoke.run.json').read_text())
        if smoke['status']!='smoke_completed' or smoke['diagnostics']['steps']!=4 or not smoke.get('cuda_projection_sanity',{}).get('passed'):
            raise RuntimeError('CUDA projection / full-batch smoke did not pass')
        comparison['smoke']={'report':'results/S04-smoke.run.json','steps':4,'batch_size':smoke['config']['data']['batch_size'],
                             'cuda_projection_sanity':smoke['cuda_projection_sanity'],'diagnostics':smoke['diagnostics']}
        for name in names:
            verify_sources(); update('training_or_waiting_for_gpu',current=name)
            fast=run_exp.run('configs/'+name+'.yaml')
            report=json.loads(Path(configs[name]['output']['report']).read_text())
            if report['status']!='training_completed_evaluation_pending' or report['diagnostics']['steps']!=configs[name]['train']['steps']:
                raise RuntimeError('Incomplete training report for '+name)
            comparison['variants'][name]={'checkpoint':configs[name]['output']['save_path'],'fast':fast,
                                          'training_report':configs[name]['output']['report'],'diagnostics':report['diagnostics'],
                                          'elapsed_training_seconds':report['elapsed_seconds']}
            write_json(ROOT/'results'/(name+'.validation.json'),{'experiment':name,'evaluation':'fasteval public validation','metrics':fast},exclusive=True)
            update('fast_evaluation_completed',current=name)
        for name in names:
            verify_sources(); update('score_model_or_waiting_for_gpu',current=name)
            path=configs[name]['output']['save_path']
            def score_ready():
                verify_sources(); update('score_model_running',current=name)
            scored=gpu_command([sys.executable,'score_model.py',path],score_ready)
            lines=[l.split(':',1)[1].strip() for l in scored.stdout.splitlines() if l.startswith('report ') and ':' in l]
            if len(lines)!=1: raise RuntimeError('Cannot locate full validation report')
            report_path=ROOT/lines[0]; payload=json.loads(report_path.read_text())
            if payload['tag']!=Path(path).name or payload['phase']!='validation': raise ValueError('Unexpected checkpoint or evaluation split')
            comparison['variants'][name]['score_model']={'report':str(report_path.relative_to(ROOT)),'metrics':payload}
            state['completed_evaluations'].append(name)
            update('score_model_completed',current=name)
        for kind in ('fast','score_model'):
            control=comparison['variants']['S04-control']; surgery=comparison['variants']['S04']
            a=control['fast'] if kind=='fast' else full_metrics(control[kind]['metrics'])
            b=surgery['fast'] if kind=='fast' else full_metrics(surgery[kind]['metrics'])
            comparison[kind+'_delta_vs_control']={key:b[key]-a[key] for key in ('Acc_f','Acc_r','CKA_f_o','CKA_r_o','AUS','RUS_o','final')}
        rows=['# S04 Gradient Surgery comparison','','Matched r017 settings, seed 0, 4800 steps. Full local public validation below; private improvement remains unverified.','',
              '| Variant | Acc_f | Acc_r | CKA_f | CKA_r | AUS | RUS_o | Final |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name in names:
            result=comparison['variants'][name]; p=full_metrics(result['score_model']['metrics'])
            rows.append(f"| {name} | {p['Acc_f']:.6f} | {p['Acc_r']:.6f} | {p['CKA_f_o']:.6f} | {p['CKA_r_o']:.6f} | {p['AUS']:.6f} | {p['RUS_o']:.6f} | {p['final']:.8f} |")
        delta=comparison['score_model_delta_vs_control']
        rows+=['',f"Surgery minus control: Final {delta['final']:+.8f}, CKA_r {delta['CKA_r_o']:+.8f}, CKA_f {delta['CKA_f_o']:+.8f}.",'',
               'Accuracy entries retain the full scorer JSON units. Gradient diagnostics use weighted feature/KD retain loss, not CKA_r itself.','',
               '| Variant | Conflict steps | Projected steps | Positive retain linear change / sampled AdamW updates | Training seconds |',
               '|---|---:|---:|---:|---:|']
        for name in names:
            v=comparison['variants'][name]; d=v['diagnostics']
            rows.append(f"| {name} | {d['conflicts']}/{d['steps']} | {d['projections']}/{d['steps']} | {d['positive_retain_linear_changes']}/{d['actual_update_samples']} | {v['elapsed_training_seconds']:.1f} |")
        with comparison_md.open('x') as stream: stream.write(chr(10).join(rows)+chr(10))
        for name in names:
            path=Path(configs[name]['output']['report']); record=json.loads(path.read_text())
            record.update(status='local_evaluation_completed',comparison_report=str(comparison_path.relative_to(ROOT)))
            write_json(path,record)
        with (ROOT/'S_EXPERIMENTS.md').open('a') as stream:
            stream.write(chr(10)+'### S04 실행 완료'+chr(10)+chr(10)+'대조군 및 surgery의 로컬 평가 완료: results/S04.comparison.md. Final 차이 '+f"{delta['final']:+.8f}"+'. private 개선 여부는 미검증.'+chr(10))
        update('completed',current=None)
    except BaseException as error:
        update('failed',error=type(error).__name__+': '+str(error))
        traceback.print_exc()
        raise


if __name__=='__main__': main()
