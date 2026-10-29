"""Compare fixed query policies with the same sound decision certificate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from core import ArrayOracle,rollout

BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):
    p=Path(p);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=BASE/'queries');parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();args.out.mkdir(exist_ok=False)
    manifest=json.loads((BASE/'data/MANIFEST.json').read_text())
    assert sha(BASE/'data/benchmark.npz')==manifest['benchmark_sha256']
    with np.load(BASE/'data/benchmark.npz') as z:d={k:z[k] for k in z.files}
    frozen=dict(purpose='DEVELOPMENT_QUERY_COMPARISON',workers=args.workers,policies=['entropy','center','random','witness'],
        budgets=[0,8,32,128],random_order_seed=17,all_cases=len(d['truth']),
        hashes={str(p):sha(p) for p in [Path(__file__),BASE/'core.py',BASE/'PROTOCOL.md',BASE/'data/MANIFEST.json']})
    write(args.out/'FROZEN.json',frozen)
    templates={k:np.unpackbits(d[k],axis=1).reshape(-1,256,256).astype(bool) for k in ['a','b','domain']}
    start=time.monotonic()
    for method in frozen['policies']:
        result={str(b):dict(label=[],queries=[],revealed=[],interval_lower=[],interval_upper=[]) for b in frozen['budgets']}
        def one(i):
            native=np.unpackbits(d['native'][i]).reshape(256,256).astype(bool);ti=d['template'][i]
            return rollout(ArrayOracle(native,d['coarse'][i]),d['coarse'][i],
                templates['a'][ti],templates['b'][ti],templates['domain'][ti],bool(d['kind'][ti]),float(d['ref'][ti]),
                method,seed=17+i,budgets=frozen['budgets'])
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i,r in enumerate(pool.map(one,range(len(d['truth'])))):
                for budget,v in r.items():
                    for key,value in v.items():result[budget][key].append(value)
                    assert v['label']<0 or v['label']==d['truth'][i],(method,i,budget,'UNSOUND CERTIFICATE')
                if (i+1)%112==0:
                    status=dict(status='running',method=method,cases=i+1,total=len(d['truth']),elapsed_seconds=time.monotonic()-start)
                    write(args.out/'LIVE_PROGRESS.json',status);print(json.dumps(status),flush=True)
        for budget,row in result.items():
            np.savez_compressed(args.out/f'{method}_budget{budget}.npz',**{k:np.array(v) for k,v in row.items()})
        write(args.out/f'{method}_COMPLETE.json',dict(status='COMPLETE',cases=len(d['truth']),elapsed_seconds=time.monotonic()-start,
            budgets={b:dict(certified=int(np.sum(np.array(v['label'])>=0)),mean_queries=float(np.mean(v['queries']))) for b,v in result.items()}))
    assert all(sha(p)==h for p,h in frozen['hashes'].items())
    write(args.out/'COMPLETE.json',dict(status='COMPLETE',elapsed_seconds=time.monotonic()-start,frozen_sha256=sha(args.out/'FROZEN.json')))
    write(args.out/'LIVE_PROGRESS.json',dict(status='complete',elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':main()
