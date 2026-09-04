"""S04: retain-prioritized, one-way global gradient projection.
Use tools/S04_run.py for the locked smoke test, matched control and evaluations.
"""
import argparse
import contextlib
import copy
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
import yaml


def sha256(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def write_json(path,data,exclusive=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+chr(10)
    if exclusive:
        with path.open('x') as stream: stream.write(text)
    else:
        temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(text); temp.replace(path)


def save_checkpoint(path,state):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    cpu={k:v.detach().cpu() for k,v in state.items()}
    if any(v.is_floating_point() and not torch.isfinite(v).all().item() for v in cpu.values()):
        raise ValueError('Non-finite checkpoint')
    with path.open('xb') as stream: torch.save({'model':cpu},stream)


class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,text):
        for s in self.streams: s.write(text)
        return len(text)
    def flush(self):
        for s in self.streams: s.flush()


def inner_product(a,b):
    first=next((v for v in [*a,*b] if v is not None),None)
    value=torch.zeros((),dtype=torch.float64,device=first.device if first is not None else 'cpu')
    for x,y in zip(a,b):
        if x is not None and y is not None:
            value+=torch.dot(x.detach().reshape(-1).double(),y.detach().reshape(-1).double())
    return value


@torch.no_grad()
def combine_gradients(retain,forget,*,enabled,min_retain_norm):
    if not retain or len(retain)!=len(forget) or type(enabled) is not bool or not math.isfinite(min_retain_norm) or min_retain_norm<=0:
        raise ValueError('Invalid gradient groups or projection options')
    device=None
    for r,f in zip(retain,forget):
        for g in (r,f):
            if g is not None:
                if not g.is_floating_point() or g.layout!=torch.strided: raise ValueError('Dense floating gradients required')
                if device is not None and g.device!=device: raise ValueError('Gradients must share a device')
                device=g.device
        if r is not None and f is not None and (r.shape!=f.shape or r.dtype!=f.dtype):
            raise ValueError('Gradient shape/dtype mismatch')
    r2=inner_product(retain,retain); f2=inner_product(forget,forget); dot=inner_product(retain,forget)
    if not torch.isfinite(torch.stack([r2,f2,dot])).all().item(): raise ValueError('Non-finite gradient statistics')
    eligible=r2>=min_retain_norm**2
    active=eligible & (dot<0)
    alpha=torch.where(active & enabled,dot/r2.clamp_min(min_retain_norm**2),torch.zeros_like(dot))
    corrected=[]; merged=[]
    for r,f in zip(retain,forget):
        if r is None and f is None:
            corrected.append(None); merged.append(None); continue
        rr=torch.zeros_like(f) if r is None else r.detach()
        ff=torch.zeros_like(r) if f is None else f.detach()
        projected=ff-alpha.to(rr.dtype)*rr
        corrected.append(projected)
        merged.append(rr+projected)
    after=inner_product(retain,corrected); after_f2=inner_product(corrected,corrected)
    merged2=inner_product(merged,merged)
    values=torch.stack([r2,f2,dot,after,after_f2,alpha,merged2]).tolist()
    if not all(math.isfinite(v) for v in values): raise ValueError('Non-finite projected gradient')
    rn,fn=math.sqrt(values[0]),math.sqrt(values[1])
    after_fn=math.sqrt(values[4]); coefficient=values[5]
    projected=enabled and values[2]<0 and rn>=min_retain_norm
    if projected and values[3]<-1e-5*rn*fn-1e-12:
        raise ValueError('Projection has a material negative retain component')
    return merged,{'retain_norm':rn,'forget_norm':fn,'dot_before':values[2],
                   'cosine_before':max(-1.,min(1.,values[2]/(rn*fn))) if rn and fn else None,
                   'dot_after':values[3],
                   'cosine_after':max(-1.,min(1.,values[3]/(rn*after_fn))) if rn and after_fn else None,
                   'forget_norm_after':after_fn,'projection_coefficient':coefficient,
                   'correction_norm':abs(coefficient)*rn,'combined_norm_before_clip':math.sqrt(values[6]),
                   'conflict':values[2]<0,'eligible_retain':rn>=min_retain_norm,'projected':projected}


class SurgeryStep:
    def __init__(self,*,enabled,min_retain_norm,diagnostic_interval):
        if type(diagnostic_interval) is not int or diagnostic_interval<1: raise ValueError('Invalid diagnostic interval')
        self.enabled=enabled; self.min_retain_norm=min_retain_norm; self.interval=diagnostic_interval
        self.step=0; self.pending=None

    def __call__(self,retain_loss,forget_loss,params,step):
        if self.pending is not None or step!=self.step+1: raise ValueError('Gradient/update callbacks out of order')
        if not torch.isfinite(torch.stack([retain_loss.detach(),forget_loss.detach()])).all().item():
            raise ValueError('Non-finite objective')
        # The retained and forgotten examples use independent student forward graphs.
        # Teacher targets are detached, so the first graph can be freed immediately.
        gr=torch.autograd.grad(retain_loss,params,allow_unused=True)
        gf=torch.autograd.grad(forget_loss,params,allow_unused=True)
        combined,stats=combine_gradients(gr,gf,enabled=self.enabled,min_retain_norm=self.min_retain_norm)
        for p,g in zip(params,combined): p.grad=g
        sampled=step==1 or step%self.interval==0
        stats.update(step=step,retain_loss=retain_loss.item(),forget_loss=forget_loss.item())
        self.pending={'stats':stats,'params':list(params),'retain':gr if sampled else None,
                      'before':[p.detach().clone() for p in params] if sampled else None}

    @torch.no_grad()
    def after_update(self,params,step):
        if self.pending is None or self.pending['stats']['step']!=step or len(params)!=len(self.pending['params']) or any(p is not q for p,q in zip(params,self.pending['params'])):
            raise ValueError('Unexpected optimizer update')
        record=self.pending['stats']
        record.update(actual_retain_linear_change=None,actual_update_norm=None,actual_retain_update_cosine=None,combined_norm_after_clip=None)
        if self.pending['before'] is not None:
            delta=[p.detach()-old for p,old in zip(params,self.pending['before'])]
            change=inner_product(self.pending['retain'],delta).item()
            norm=math.sqrt(inner_product(delta,delta).item())
            grads=[p.grad for p in params]
            record.update(actual_retain_linear_change=change,actual_update_norm=norm,
                          actual_retain_update_cosine=change/(record['retain_norm']*norm) if record['retain_norm'] and norm else None,
                          combined_norm_after_clip=math.sqrt(inner_product(grads,grads).item()))
        self.pending=None; self.step=step
        return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/S04.yaml')
    parser.add_argument('--smoke',action='store_true',help='4 steps at configured batch size, released data, distinct smoke outputs')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    experiment=Path(args.config).stem
    if args.smoke:
        experiment+='-smoke'
        cfg=copy.deepcopy(cfg); cfg['train']['steps']=4; cfg['data']['workers']=0
        cfg['output']={'save_path':f'models/{experiment}.pt','report':f'results/{experiment}.run.json',
                       'training_log':f'logs/{experiment}.training.log','gradient_log':f'logs/{experiment}.gradients.csv'}
        cfg['surgery']['diagnostic_interval']=1
    if cfg['model']['arch']!='vit_base_patch16_224.mae' or cfg['model']['num_classes']!=100:
        raise ValueError('Competition architecture must remain unchanged')
    output=cfg['output']; paths=[Path(p) for p in output.values()]
    if len({p.resolve() for p in paths})!=len(paths) or any(p.exists() for p in paths):
        raise FileExistsError('Preserve previous artifacts; use a new run ID')
    import S04_reference as base
    inputs=[Path(__file__),Path(base.__file__),Path(args.config),Path(cfg['model']['mo_ckpt']),Path(cfg['data']['split']),Path(cfg['data']['forget']),Path('utils/data.py'),Path('imagenet_vit.py')]
    metadata={'experiment':experiment,'config':cfg,'smoke_only':args.smoke,'status':'preparing',
              'source_sha256':{str(p):sha256(p) for p in inputs},'software':{'python':sys.version,'torch':str(torch.__version__)},
              'method':'one-way global forget projection against weighted retain feature/KD gradient',
              'limitation':'raw-gradient geometry does not guarantee AdamW descent, CKA retention or private generalization'}
    write_json(output['report'],metadata,exclusive=True)
    try:
        controller=SurgeryStep(**cfg['surgery'])
        if args.smoke:
            if not torch.cuda.is_available(): raise RuntimeError('GPU smoke requires CUDA')
            gr=[torch.tensor([1.,0.],device='cuda')]; gf=[torch.tensor([-2.,3.],device='cuda')]
            merged,stats=combine_gradients(gr,gf,enabled=True,min_retain_norm=1e-8)
            if not stats['projected'] or not torch.equal(merged[0],torch.tensor([1.,3.],device='cuda')):
                raise RuntimeError('CUDA synthetic projection failed')
            metadata['cuda_projection_sanity']={'passed':True,'device':torch.cuda.get_device_name(),'dot_before':stats['dot_before'],'dot_after':stats['dot_after']}
            del gr,gf,merged

        counters={'steps':0,'conflicts':0,'projections':0,'actual_update_samples':0,'positive_retain_linear_changes':0}
        for p in paths: p.parent.mkdir(parents=True,exist_ok=True)
        start=time.monotonic()
        with Path(output['training_log']).open('x') as logfile,Path(output['gradient_log']).open('x',newline='') as csvfile,contextlib.redirect_stdout(Tee(sys.stdout,logfile)),contextlib.redirect_stderr(Tee(sys.stderr,logfile)):
            writer=None
            def after_step(model,step):
                nonlocal writer
                record=controller.after_update([p for p in model.parameters() if p.requires_grad],step)
                record['elapsed_seconds']=time.monotonic()-start
                if writer is None: writer=csv.DictWriter(csvfile,fieldnames=list(record)); writer.writeheader()
                writer.writerow(record)
                counters['steps']+=1; counters['conflicts']+=int(record['conflict']); counters['projections']+=int(record['projected'])
                if record['actual_retain_linear_change'] is not None:
                    counters['actual_update_samples']+=1
                    counters['positive_retain_linear_changes']+=int(record['actual_retain_linear_change']>1e-12)
                if step==1 or step%100==0 or step==cfg['train']['steps']:
                    csvfile.flush()
                    metadata.update(status='training',completed_step=step,diagnostics=dict(counters),updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    write_json(output['report'],metadata)
                    print(f"[{experiment}] step={step} conflict={counters['conflicts']/step:.3f} projected={counters['projections']/step:.3f}",flush=True)
            model=base.main(config=cfg,gradient_step=controller,after_step=after_step)
            if controller.step!=cfg['train']['steps']: raise RuntimeError('Training ended before all gradient callbacks completed')
            save_checkpoint(output['save_path'],model.state_dict())
        metadata.update(status='smoke_completed' if args.smoke else 'training_completed_evaluation_pending',
                        diagnostics=counters,checkpoint_sha256=sha256(output['save_path']),gradient_log_sha256=sha256(output['gradient_log']),elapsed_seconds=time.monotonic()-start)
        write_json(output['report'],metadata)
        print(f"{experiment} saved: {output['save_path']}",flush=True)
    except BaseException as error:
        metadata.update(status='failed',error=type(error).__name__+': '+str(error))
        write_json(output['report'],metadata)
        raise


if __name__=='__main__': main()
