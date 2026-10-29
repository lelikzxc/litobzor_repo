"""Common small CNN fallback; exactly the same weights for every query policy."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BASE=Path(__file__).resolve().parent
torch.set_num_threads(1)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def load_data():
    manifest=json.loads((BASE/'data/MANIFEST.json').read_text())
    path=BASE/'data/benchmark.npz';assert sha(path)==manifest['benchmark_sha256']
    with np.load(path) as z:d={k:z[k] for k in z.files}
    # Pooled target-only templates: one exact division at feature construction.
    d['templates128']=np.stack([np.unpackbits(d[k],axis=1).reshape(-1,128,2,128,2).sum((2,4)) for k in ['a','b','domain','target']],1).astype(np.uint8)
    return d


def features(d,indices,revealed):
    indices=np.asarray(indices);ti=d['template'][indices]
    counts=d['coarse'][indices].astype(np.float32)/64
    coarse=counts.repeat(4,1).repeat(4,2)
    raw=np.unpackbits(d['native'][indices],axis=1).reshape(-1,128,2,128,2).sum((2,4)).astype(np.float32)/4
    known=revealed|(d['coarse'][indices]==0)|(d['coarse'][indices]==64)
    known=known.repeat(4,1).repeat(4,2)
    observation=np.where(known,raw,coarse)
    images=np.concatenate([coarse[:,None],observation[:,None],known[:,None].astype(np.float32),d['templates128'][ti].astype(np.float32)/4],1)
    scalar=np.stack([d['kind'][ti].astype(np.float32),d['ref'][ti]/32],1)
    return np.ascontiguousarray(images),np.ascontiguousarray(scalar)


class FallbackCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(7,8,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(8,16,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(16,24,3,stride=2,padding=1),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(50,64),nn.SiLU(),nn.Linear(64,3))
    def forward(self,images,scalar):
        z=self.encoder(images)
        z=torch.cat([z.mean((2,3)),z.amax((2,3)),scalar],1)
        return self.head(z)


@torch.no_grad()
def predict(model,d,indices,revealed,device):
    pp=[];model.eval()
    for start in range(0,len(indices),64):
        ii=indices[start:start+64];rr=revealed[start:start+64]
        x,s=features(d,ii,rr)
        pp.append(model(torch.as_tensor(x,device=device),torch.as_tensor(s,device=device)).softmax(1).cpu().numpy())
    return np.concatenate(pp)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=BASE/'models');args=parser.parse_args()
    assert torch.backends.mps.is_available() and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
    args.out.mkdir(exist_ok=False);d=load_data();split=json.loads((BASE/'data/FROZEN.json').read_text())
    ii=np.flatnonzero(np.isin(d['geometry'],split['train']));classes=[ii[d['truth'][ii]==k] for k in range(3)]
    assert all(len(v)>0 for v in classes)
    frozen=dict(purpose='COMMON_FALLBACK_ONLY_TRAIN_GEOMETRIES',created_utc=datetime.now(timezone.utc).isoformat(),
        steps=1200,seeds=[17,41],batch=24,optimizer='AdamW',lr=.002,min_lr=.0001,weight_decay=.0001,
        sample='Uniform class then uniform TRAIN example; same fallback for all policies',
        reveal_training='Per-example p from0,.03125,.125,1 with probabilities.3,.3,.3,.1, then Bernoulli tile mask',
        selection='Fixed final step1200; no EVALUATION reads during fitting',device='mps',cpu_fallback=False,
        train=split['train'],evaluation=split['evaluation'],class_counts=[len(v) for v in classes],
        hashes={str(p):sha(p) for p in [Path(__file__),BASE/'PROTOCOL.md',BASE/'data/FROZEN.json',BASE/'data/MANIFEST.json']})
    write(args.out/'FROZEN.json',frozen);device=torch.device('mps');start=time.monotonic();result=[]
    for seed in [17,41]:
        torch.manual_seed(seed);torch.mps.manual_seed(seed);rng=np.random.default_rng(seed)
        model=FallbackCNN().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
        history=[]
        for step in range(1200):
            kk=rng.integers(0,3,24);ids=np.array([rng.choice(classes[k]) for k in kk])
            probability=rng.choice([0.,.03125,.125,1.],size=24,p=[.3,.3,.3,.1])
            reveal=rng.random((24,32,32))<probability[:,None,None]
            xx,ss=features(d,ids,reveal)
            opt.zero_grad(set_to_none=True);out=model(torch.tensor(xx,device=device),torch.tensor(ss,device=device))
            loss=F.cross_entropy(out,torch.tensor(d['truth'][ids],device=device))
            assert torch.isfinite(loss).item();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            lr=.0001+.0019*.5*(1+math.cos(math.pi*step/1199))
            for group in opt.param_groups:group['lr']=lr
            opt.step()
            if (step+1)%100==0:
                row=dict(seed=seed,step=step+1,loss=float(loss.detach().cpu()),elapsed_seconds=time.monotonic()-start)
                history.append(row);write(args.out/'LIVE_PROGRESS.json',dict(status='running',**row));print(json.dumps(row),flush=True)
        state={k:v.detach().cpu() for k,v in model.state_dict().items()};torch.save(state,args.out/f'seed{seed}.pt')
        write(args.out/f'seed{seed}_history.json',history);result.append(dict(seed=seed,steps=1200,weights_sha256=sha(args.out/f'seed{seed}.pt')))
        del model,opt;torch.mps.empty_cache()
    assert all(sha(p)==h for p,h in frozen['hashes'].items())
    write(args.out/'COMPLETE.json',dict(status='COMPLETE',seeds=result,elapsed_seconds=time.monotonic()-start,
        parameters=sum(p.numel() for p in FallbackCNN().parameters()),device='mps',cpu_fallback=False,
        frozen_sha256=sha(args.out/'FROZEN.json')))
    write(args.out/'LIVE_PROGRESS.json',dict(status='complete',elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':main()
