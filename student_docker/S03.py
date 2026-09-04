"""S03: last iterate, tail EMA and checkpoint mean from ONE r017 trajectory.
Run through tools/S03_run.py to share the GPU lock and evaluate every variant.
"""
import argparse
import contextlib
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import yaml


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write_json(path, data, exclusive=False):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+chr(10)
    if exclusive:
        with path.open('x') as stream: stream.write(encoded)
    else:
        temp=path.with_suffix(path.suffix+'.tmp')
        temp.write_text(encoded)
        temp.replace(path)


def save_checkpoint(path,state):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    cpu={k:v.detach().cpu() for k,v in state.items()}
    if any(v.is_floating_point() and not torch.isfinite(v).all().item() for v in cpu.values()):
        raise ValueError('Non-finite checkpoint weights')
    with path.open('xb') as stream:
        torch.save({'model':cpu},stream)


class TrajectoryAverages:
    def __init__(self,*,total_steps,ema_start,decays,checkpoint_steps,snapshot_dir):
        if (type(total_steps) is not int or type(ema_start) is not int
            or not 1<=ema_start<=total_steps or len(checkpoint_steps)<2
            or any(type(s) is not int for s in checkpoint_steps)
            or checkpoint_steps!=sorted(set(checkpoint_steps))
            or not ema_start<=checkpoint_steps[0] or checkpoint_steps[-1]!=total_steps
            or not decays or any(not math.isfinite(b) or not 0<b<1 for b in decays.values())
            or any(k in {'last','avg'} for k in decays)):
            raise ValueError('Invalid trajectory averaging window or decay')
        self.total_steps,self.ema_start=total_steps,ema_start
        self.decays=dict(decays)
        self.checkpoint_steps=list(checkpoint_steps)
        self.snapshot_dir=Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True,exist_ok=False)
        self.step=0
        self.emas={}
        self.mean=None
        self.snapshots=[]

    @torch.no_grad()
    def observe(self,model,step):
        if type(step) is not int or step!=self.step+1 or step>self.total_steps:
            raise ValueError('Expected one observation after each optimizer step')
        if step==1 and any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in model.modules()):
            raise ValueError('S03 supports the LayerNorm ViT; BatchNorm requires explicit statistics recalibration')
        if step>=self.ema_start:
            if not self.emas:
                self.emas={name:AveragedModel(model,multi_avg_fn=get_ema_multi_avg_fn(beta),use_buffers=False)
                           for name,beta in self.decays.items()}
                for average in self.emas.values(): average.requires_grad_(False)
            for average in self.emas.values(): average.update_parameters(model)
        if step in self.checkpoint_steps:
            if self.mean is None:
                self.mean=AveragedModel(model,use_buffers=False)
                self.mean.requires_grad_(False)
            self.mean.update_parameters(model)
            path=self.snapshot_dir/f'step-{step:06d}.pt'
            save_checkpoint(path,model.state_dict())
            self.snapshots.append({'step':step,'path':str(path),'sha256':sha256(path)})
        self.step=step

    def export(self,model,paths):
        if self.step!=self.total_steps or self.mean is None or len(self.snapshots)!=len(self.checkpoint_steps):
            raise ValueError('Cannot export an incomplete trajectory')
        expected=self.total_steps-self.ema_start+1
        if any(int(avg.n_averaged.item())!=expected for avg in self.emas.values()) or int(self.mean.n_averaged.item())!=len(self.checkpoint_steps):
            raise ValueError('Unexpected number of EMA or checkpoint updates')
        if set(paths)!={'last','avg',*self.decays} or len({Path(p).resolve() for p in paths.values()})!=len(paths):
            raise ValueError('Distinct paths required for last, avg and each EMA')
        if any(Path(p).exists() for p in paths.values()):
            raise FileExistsError('Preserve existing S03 output checkpoints')
        states={'last':model.state_dict(),'avg':self.mean.module.state_dict(),
                **{name:average.module.state_dict() for name,average in self.emas.items()}}
        artifacts={}
        for name,path in paths.items():
            save_checkpoint(path,states[name])
            artifacts[name]={'path':str(path),'sha256':sha256(path)}
        saved_last=torch.load(paths['last'],map_location='cpu',weights_only=True)['model']
        if any(not torch.equal(value.detach().cpu(),saved_last[key]) for key,value in model.state_dict().items()):
            raise ValueError('Saved last checkpoint differs from returned training model')
        return {'last_matches_returned_model':True,'steps_observed':self.step,'ema_start':self.ema_start,'ema_decays':self.decays,
                'ema_updates':self.total_steps-self.ema_start+1,
                'checkpoint_steps':self.checkpoint_steps,'checkpoint_count':int(self.mean.n_averaged.item()),
                'snapshots':self.snapshots,'artifacts':artifacts,
                'buffer_policy':'copy from last observation; no BatchNorm in target model'}


class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,text):
        for stream in self.streams: stream.write(text)
        return len(text)
    def flush(self):
        for stream in self.streams: stream.flush()


def output_paths(cfg):
    return {'avg':cfg['output']['save_path'],'last':cfg['output']['last_path'],**cfg['output']['ema_paths']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/S03.yaml')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    if cfg['model']['arch']!='vit_base_patch16_224.mae' or cfg['model']['num_classes']!=100:
        raise ValueError('Competition architecture must remain unchanged')
    paths=output_paths(cfg)
    spec=cfg['averaging']
    if set(cfg['output']['ema_paths'])!=set(spec['ema_decays']):
        raise ValueError('EMA output names must match configured decays')
    report=Path(cfg['output']['trajectory_report'])
    training_log=Path(cfg['output']['training_log'])
    reserved=[*map(Path,paths.values()),Path(spec['snapshot_dir']),report,training_log]
    if len({p.resolve() for p in reserved})!=len(reserved) or any(p.exists() for p in reserved):
        raise FileExistsError('S03 output exists or paths collide; use a new run ID')
    import S03_reference as base
    sources=[Path(__file__),Path(base.__file__),Path(args.config),Path(cfg['model']['mo_ckpt']),
             Path(cfg['data']['split']),Path(cfg['data']['forget']),Path('utils/data.py'),Path('imagenet_vit.py')]
    metadata={'experiment':Path(args.config).stem,'config':cfg,'status':'preparing',
              'source_sha256':{str(p):sha256(p) for p in sources},
              'software':{'python':sys.version,'torch':str(torch.__version__)},
              'hypothesis':'tail trajectory averaging may improve generalization; private improvement is unverified',
              'initialization':'original M_o; averaging begins only after completed optimizer step ema_start',
              'checkpoint_mean':'equal weights at predeclared steps; unchanged r017 LR schedule'}
    write_json(report,metadata,exclusive=True)
    try:
        tracker=TrajectoryAverages(total_steps=cfg['train']['steps'],ema_start=spec['ema_start'],
                                  decays=spec['ema_decays'],checkpoint_steps=spec['checkpoint_steps'],
                                  snapshot_dir=spec['snapshot_dir'])
        def after_step(model,step):
            tracker.observe(model,step)
            if step==1 or step%100==0 or step==cfg['train']['steps']:
                metadata.update(status='training',completed_step=step,
                                updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
                write_json(report,metadata)
        training_log.parent.mkdir(parents=True,exist_ok=True)
        with training_log.open('x') as stream, contextlib.redirect_stdout(Tee(sys.stdout,stream)), contextlib.redirect_stderr(Tee(sys.stderr,stream)):
            model=base.main(config=copy.deepcopy(cfg),after_step=after_step)
            metadata.update(tracker.export(model,paths))
            metadata['status']='training_completed_evaluation_pending'
            write_json(report,metadata)
            print('S03 trajectory and all averaged checkpoints saved',flush=True)
    except BaseException as error:
        metadata.update(status='failed',error=type(error).__name__+': '+str(error))
        write_json(report,metadata)
        raise


if __name__=='__main__': main()
