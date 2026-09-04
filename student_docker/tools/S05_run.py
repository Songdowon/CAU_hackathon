"""Bounded S05 weighted-soup search. No training and no CUDA in the parent."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import yaml
from tools import run_exp
from S05 import validate_weights,weight_key,refinement,rank_screen,select_best,validate_metrics,sha256,write_json,verify_manifest


def gpu_command(command,log_path,on_ready):
    with open(run_exp.LOCK,'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run_exp.wait_for_gpu()
        on_ready()
        with Path(log_path).open('x') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            print(Path(log_path).read_text()[-6000:],flush=True)
            result.check_returncode()
        return Path(log_path).read_text()


def full_metrics(payload):
    a=payload['accuracy_metric']; r=payload['representation_metric']
    return validate_metrics({'AUS':payload['AUS'],'RUS_o':payload['RUS_o'],'final':payload['final_score'],
                             'Acc_f':a['Acc_f'],'Acc_r':a['Acc_r'],'CKA_f_o':r['CKA_f_o'],'CKA_r_o':r['CKA_r_o']})


def finalists(rows,threshold,margin,count,equal_id):
    strict=[r for r in rows if r['metrics']['AUS']>=threshold]
    selected=rank_screen(strict,threshold,0)[:count]+rank_screen(rows,threshold,margin)[:count]
    selected+=[max(rows,key=lambda r:(r['metrics']['AUS'],r['metrics']['RUS_o'])),next(r for r in rows if r['id']==equal_id)]
    return list({r['id']:r for r in selected}.values())


def report_markdown(comparison,threshold):
    lines=['# S05 Weighted Soup Search','',
           'Weight order: r016 / r015 / r012. No retraining. Local public validation search; private improvement is unverified.',
           f'Final objective: maximize RUS_o with original scorer AUS >= {threshold}. Search is bounded; no global-optimum claim.','',
           '| Candidate | Weights | Fast AUS | Fast RUS_o | Fast Final | Full AUS | Full RUS_o | Full Final | Feasible |',
           '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in comparison['candidates']:
        f=r['metrics']; full=r.get('full',{}).get('metrics')
        weights='/'.join(f'{v:.6f}' for v in r['weights'])
        end=f"{full['AUS']:.8f} | {full['RUS_o']:.8f} | {full['final']:.8f} | {full['AUS']>=threshold}" if full else '- | - | - | pending/not shortlisted'
        lines.append(f"| {r['id']} | {weights} | {f['AUS']:.8f} | {f['RUS_o']:.8f} | {f['final']:.8f} | {end} |")
    winner=comparison.get('winner')
    if comparison['status'].startswith('completed'):
        lines+=['',('Selected: '+winner['id']+'; checkpoint models/S05.pt.') if winner else 'No confirmed candidate satisfies the AUS constraint; models/S05.pt was not exported.']
    if comparison.get('historical'):
        lines+=['','Historical four-model soup reports (different ingredients):','',
                '| Checkpoint | AUS | RUS_o | Final |','|---|---:|---:|---:|']
        for h in comparison['historical']:
            m=h['metrics']; lines.append(f"| {h['checkpoint']} | {m['AUS']:.8f} | {m['RUS_o']:.8f} | {m['final']:.8f} |")
    return chr(10).join(lines)+chr(10)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/S05.yaml')
    args=parser.parse_args(); os.chdir(ROOT)
    cfg=yaml.safe_load(Path(args.config).read_text())
    if cfg['experiment']!='S05' or [s['name'] for s in cfg['sources']]!=['r016','r015','r012']:
        raise ValueError('S05 source identities/order must match the user design')
    search=cfg['search']; threshold=float(search['min_aus']); margin=float(search['fast_margin'])
    if not 0<=threshold-margin<=threshold<=1 or search['max_candidates']!=27: raise ValueError('Unexpected search constraints')
    initial=[validate_weights(w) for w in search['initial_weights']]
    if len(initial)!=9 or len({weight_key(w) for w in initial})!=9 or weight_key(initial[0])!=weight_key([1/3]*3):
        raise ValueError('Nine unique initial points starting with equal thirds are required')
    output=cfg['output']; model_dir=Path(output['model_dir']); result_dir=Path(output['result_dir']); log_dir=Path(output['log_dir'])
    reserved=[model_dir,result_dir,log_dir,*[Path(output[k]) for k in ('state','manifest','comparison','comparison_md','winner')]]
    if any(p.exists() for p in reserved): raise FileExistsError('Preserve existing S05 outputs; do not overwrite a previous run')
    tracked=[args.config,'S05.py','tools/S05_run.py','tools/run_exp.py','tools/fasteval.py','score_model.py','imagenet_vit.py','train_ft.py',
             'splits/student_split.pt','validation_cache/refs.pt','validation_cache/M_o__validation.npz']+[s['path'] for s in cfg['sources']]
    scorer=Path(os.environ.get('TRUSTED_SCORER_ROOT','/root/cau-ai-hackathon-26/grading_docker'))
    tracked+=[str(scorer/name) for name in ('score_unlearning.py','convert_checkpoint.py','imagenet_vit.py')]
    hashes={p:sha256(p) for p in tracked}
    cache={}
    for path in ('cache/val_px_fp16.npy','cache/val_labels.npy'):
        stat=Path(path).stat(); cache[path]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    if shutil.disk_usage(ROOT).free<12*1024**3: raise RuntimeError('S05 needs at least 12 GiB free for preserved candidates')
    for path in (model_dir,result_dir,log_dir): path.mkdir(parents=True)
    manifest={'experiment':'S05','config':cfg,'source_sha256':hashes,'cache_fingerprints':cache,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(output['manifest'],manifest,exclusive=True)
    state={'experiment':'S05','pid':os.getpid(),'status':'preparing','completed_fast':0,'completed_full':0}
    comparison={'experiment':'S05','status':'preparing','objective':{'constraint':f'AUS >= {threshold}','maximize':'RUS_o'},
                'source_order':[s['name'] for s in cfg['sources']],'candidates':[],'refinements':[],'historical':[],
                'evaluation':'local public validation; no official/private submission','selection_scope':'best feasible candidate among original-scorer-confirmed shortlist','fast_margin_is_heuristic':True,'winner':None,'manifest':output['manifest']}
    write_json(output['state'],state,exclusive=True); write_json(output['comparison'],comparison,exclusive=True)
    with Path(output['comparison_md']).open('x') as stream: stream.write(report_markdown(comparison,threshold))
    def update(status,**values):
        state.update(status=status,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**values)
        comparison['status']=status
        write_json(output['state'],state); write_json(output['comparison'],comparison)
        temp=Path(output['comparison_md']+'.tmp'); temp.write_text(report_markdown(comparison,threshold)); temp.replace(output['comparison_md'])
        print(json.dumps({'status':status,**values},ensure_ascii=False),flush=True)
    try:
        for item in cfg['historical_reports']:
            p=Path(item['report']); payload=json.loads(p.read_text())
            if payload['phase']!='validation' or payload['tag']!=Path(item['checkpoint']).name: raise ValueError('Historical report identity mismatch')
            comparison['historical'].append({**item,'report_sha256':sha256(p),'metrics':full_metrics(payload)})
        def evaluate(weights,stage):
            if len(comparison['candidates'])>=search['max_candidates']: raise RuntimeError('Search budget exceeded')
            ident=f"C{len(comparison['candidates']):03d}"; weights=validate_weights(weights)
            if any(weight_key(r['weights'])==weight_key(weights) for r in comparison['candidates']): raise ValueError('Duplicate candidate')
            update('fast_or_waiting_for_gpu',current=ident,stage=stage,weights=list(weights))
            def ready():
                verify_manifest(manifest); update('fast_running',current=ident,stage=stage,weights=list(weights))
            gpu_command([sys.executable,'S05.py','--config',args.config,'--candidate',ident,'--weights',*[repr(w) for w in weights],'--manifest',output['manifest']],log_dir/(ident+'.fast.log'),ready)
            report=result_dir/('S05-'+ident+'.fast.json'); row=json.loads(report.read_text())
            if row['id']!=ident or weight_key(row['weights'])!=weight_key(weights) or row['source_order']!=comparison['source_order']:
                raise ValueError('Candidate result identity mismatch')
            if sha256(row['checkpoint'])!=row['checkpoint_sha256']: raise ValueError('Candidate checkpoint hash mismatch')
            validate_metrics(row['metrics']); row.update(stage=stage,fast_report=str(report))
            comparison['candidates'].append(row); state['completed_fast']=len(comparison['candidates'])
            m=row['metrics']; label='/'.join(f'{w:.6f}' for w in weights)
            with run_exp.LOG.open('a') as stream:
                stream.write(f"| S05-{ident} | weighted soup r016/r015/r012 = {label}; no training | {m['Acc_f']:.2f} | {m['Acc_r']:.2f} | {m['CKA_f_o']:.4f} | {m['CKA_r_o']:.4f} | {m['AUS']:.4f} | {m['RUS_o']:.4f} | **{m['final']:.4f}** | 0 | {datetime.datetime.now():%m-%d %H:%M} |"+chr(10))
            update('fast_completed',current=ident,AUS=m['AUS'],RUS_o=m['RUS_o'],final=m['final'])
        for weights in initial: evaluate(weights,'initial')
        for spec in search['refinement']:
            centers=rank_screen(comparison['candidates'],threshold,margin)[:spec['centers']]
            points=refinement([r['weights'] for r in centers],spec['step'],seen=[r['weights'] for r in comparison['candidates']])
            comparison['refinements'].append({'step':spec['step'],'centers':[r['id'] for r in centers],'points':[list(p) for p in points]})
            for weights in points: evaluate(weights,'refine-'+str(spec['step']))
        shortlist=finalists(comparison['candidates'],threshold,margin,search['full_top_k'],'C000')
        comparison['shortlist']=[r['id'] for r in shortlist]
        for row in shortlist:
            ident=row['id']; update('full_or_waiting_for_gpu',current=ident)
            def ready():
                verify_manifest(manifest)
                if sha256(row['checkpoint'])!=row['checkpoint_sha256']: raise ValueError('Candidate modified before full evaluation')
                update('full_running',current=ident)
            stdout=gpu_command([sys.executable,'score_model.py',row['checkpoint']],log_dir/(ident+'.full.log'),ready)
            reports=[line.split(':',1)[1].strip() for line in stdout.splitlines() if line.startswith('report ') and ':' in line]
            if len(reports)!=1: raise RuntimeError('Cannot identify full validation report')
            path=Path(reports[0]); payload=json.loads(path.read_text())
            if payload['phase']!='validation' or payload['tag']!=Path(row['checkpoint']).name: raise ValueError('Full report identity mismatch')
            row['full']={'report':str(path),'metrics':full_metrics(payload),'dataset_revision':payload.get('dataset_revision'),'score_version':payload.get('score_version')}
            state['completed_full']+=1; update('full_completed',current=ident)
        confirmed=[{**r,'metrics':r['full']['metrics']} for r in shortlist]
        winner=select_best(confirmed,threshold)
        equal=next(r for r in confirmed if r['id']=='C000')
        if winner:
            if sha256(winner['checkpoint'])!=winner['checkpoint_sha256']: raise ValueError('Winner changed after evaluation')
            with Path(winner['checkpoint']).open('rb') as source,Path(output['winner']).open('xb') as dest: shutil.copyfileobj(source,dest)
            if sha256(output['winner'])!=winner['checkpoint_sha256']: raise ValueError('Winner export differs from evaluated checkpoint')
            comparison['winner']={'id':winner['id'],'weights':winner['weights'],'metrics':winner['metrics'],'checkpoint':output['winner'],
                                  'checkpoint_sha256':winner['checkpoint_sha256'],'evaluated_checkpoint':winner['checkpoint'],'full_report':winner['full']['report'],
                                  'delta_vs_equal':{k:winner['metrics'][k]-equal['metrics'][k] for k in ('AUS','RUS_o','final','CKA_f_o','CKA_r_o','Acc_f','Acc_r')}}
        else: comparison['no_feasible_reason']='No full-confirmed candidate satisfies AUS >= '+str(threshold)
        final_status='completed' if winner else 'completed_no_feasible_candidate'
        with (ROOT/'S_EXPERIMENTS.md').open('a') as stream:
            message='선택 '+winner['id']+' weights='+str(winner['weights'])+' AUS='+str(winner['metrics']['AUS'])+' RUS_o='+str(winner['metrics']['RUS_o']) if winner else 'AUS 제약을 만족한 최종 후보 없음'
            stream.write(chr(10)+'### S05 실행 완료'+chr(10)+chr(10)+message+'. 전체 비교: results/S05.comparison.md. local public validation이며 private 개선은 미검증.'+chr(10))
        update(final_status,current=None)
    except BaseException as error:
        update('failed',error=type(error).__name__+': '+str(error)); traceback.print_exc(); raise


if __name__=='__main__': main()
