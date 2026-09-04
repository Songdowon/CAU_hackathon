"""S05 Weighted Soup Search helpers and isolated CUDA candidate evaluation.
Run the suite with python tools/S05_run.py --config configs/S05.yaml.
Importing this module does not import torch or initialize CUDA.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time


def validate_weights(weights):
    if len(weights)!=3 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in weights):
        raise ValueError('Three finite nonnegative coefficients required')
    total=math.fsum(weights)
    if abs(total-1)>1e-9: raise ValueError('Coefficients must sum to one')
    return tuple(float(v)/total for v in weights)


def weight_key(weights): return tuple(round(v,10) for v in validate_weights(weights))


def refinement(centers,step,seen):
    if not math.isfinite(step) or step<=0 or step>1: raise ValueError('Invalid refinement step')
    known={weight_key(w) for w in seen}; points=[]
    for center in centers:
        center=validate_weights(center)
        for donor in range(3):
            for recipient in range(3):
                if donor==recipient or center[donor]<step-1e-12: continue
                w=list(center); w[donor]=max(0.,w[donor]-step); w[recipient]+=step
                w=validate_weights(w); key=weight_key(w)
                if key not in known: points.append(w); known.add(key)
    return points


def validate_metrics(metrics):
    for key in ('AUS','RUS_o','final'):
        value=metrics[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Invalid metric '+key)
    return metrics


def _rank_key(row):
    m=row['metrics']; return (-m['RUS_o'],-m['final'],-m['AUS'],row['id'])


def select_best(rows,min_aus):
    if not math.isfinite(min_aus) or not 0<=min_aus<=1: raise ValueError('Invalid AUS threshold')
    eligible=[r for r in rows if validate_metrics(r['metrics'])['AUS']>=min_aus]
    return min(eligible,key=_rank_key) if eligible else None


def rank_screen(rows,min_aus,margin=0.):
    if not math.isfinite(margin) or margin<0: raise ValueError('Invalid screening margin')
    eligible=[r for r in rows if validate_metrics(r['metrics'])['AUS']>=min_aus-margin]
    if eligible: return sorted(eligible,key=_rank_key)
    return sorted(rows,key=lambda r:(-r['metrics']['AUS'],*_rank_key(r)))


def sha256(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def write_json(path,data,exclusive=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+chr(10)
    if exclusive:
        with path.open('x') as stream: stream.write(text)
    else:
        temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(text); temp.replace(path)


def weighted_state_dict(states,weights):
    import torch
    weights=validate_weights(weights)
    if len(states)!=3 or not all(isinstance(s,dict) and s for s in states): raise ValueError('Three state dictionaries required')
    keys=set(states[0])
    if any(set(s)!=keys for s in states): raise ValueError('Checkpoint keys differ')
    out={}
    for key in states[0]:
        tensors=[s[key] for s in states]
        if any(not isinstance(t,torch.Tensor) or t.device.type!='cpu' or t.layout!=torch.strided or t.is_complex() for t in tensors):
            raise ValueError('Dense real CPU state tensors required: '+key)
        first=tensors[0]
        if any(t.shape!=first.shape or t.dtype!=first.dtype for t in tensors): raise ValueError('Shape/dtype mismatch: '+key)
        if first.is_floating_point() and any(not torch.isfinite(t).all().item() for t in tensors): raise ValueError('Non-finite source: '+key)
        identical=all(torch.equal(t,first) for t in tensors[1:])
        if identical:
            out[key]=first.clone()
        elif not first.is_floating_point():
            raise ValueError('Non-floating buffers must agree: '+key)
        else:
            value=torch.zeros_like(first,dtype=torch.float64)
            for t,w in zip(tensors,weights):
                if w: value.add_(t.to(torch.float64),alpha=w)
            out[key]=value.to(first.dtype)
            if not torch.isfinite(out[key]).all().item(): raise ValueError('Non-finite averaged tensor: '+key)
    return out


def load_states(paths):
    import torch
    states=[]
    for path in paths:
        obj=torch.load(path,map_location='cpu',weights_only=True)
        state=obj.get('model',obj) if isinstance(obj,dict) else obj
        if not isinstance(state,dict): raise ValueError('Unsupported checkpoint format')
        states.append(state)
    return states


def save_checkpoint(path,state):
    import torch
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream: torch.save({'model':state},stream)


def verify_manifest(manifest):
    for path,digest in manifest['source_sha256'].items():
        if sha256(path)!=digest: raise RuntimeError('S05 source changed after launch: '+path)
    for path,stamp in manifest['cache_fingerprints'].items():
        stat=Path(path).stat()
        if {'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}!=stamp: raise RuntimeError('Validation cache changed: '+path)


def candidate_main(args):
    import torch
    import yaml
    from tools import fasteval
    cfg=yaml.safe_load(Path(args.config).read_text())
    manifest=json.loads(Path(args.manifest).read_text())
    if not re.fullmatch(r'C[0-9]{3}',args.candidate): raise ValueError('Invalid candidate ID')
    verify_manifest(manifest)
    weights=validate_weights(args.weights)
    checkpoint=Path(cfg['output']['model_dir'])/('S05-'+args.candidate+'.pt')
    report=Path(cfg['output']['result_dir'])/('S05-'+args.candidate+'.fast.json')
    if checkpoint.exists() or report.exists(): raise FileExistsError('Preserve previous candidate artifacts')
    start=time.monotonic()
    if not torch.cuda.is_available(): raise RuntimeError('S05 evaluation requires the locked CUDA worker')
    state=weighted_state_dict(load_states([s['path'] for s in cfg['sources']]),weights)
    model=fasteval.ViTWrapper(num_classes=100,pretrained=False,drop_path_rate=0.0,in_model_norm=False)
    model.load_state_dict(state,strict=True)
    save_checkpoint(checkpoint,state)
    del state
    metrics=fasteval.score(model.to('cuda').eval(),torch.device('cuda'),batch=cfg['search']['batch_size'])
    validate_metrics(metrics)
    record={'id':args.candidate,'weights':list(weights),'source_order':[s['name'] for s in cfg['sources']],
            'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'metrics':metrics,
            'evaluation':'fasteval full public validation; fp16 cached pixels','seconds':time.monotonic()-start}
    write_json(report,record,exclusive=True)
    print(json.dumps(record,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/S05.yaml')
    p.add_argument('--candidate',required=True)
    p.add_argument('--weights',nargs=3,type=float,required=True)
    p.add_argument('--manifest',required=True)
    candidate_main(p.parse_args())


if __name__=='__main__': main()
