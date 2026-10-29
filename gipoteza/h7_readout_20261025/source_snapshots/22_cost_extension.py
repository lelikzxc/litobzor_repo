"""Frozen policy transfer with exact endpoint, no neural fitting or selection."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from core import rollout,ArrayOracle
from learning import write,sha

BASE=Path(__file__).resolve().parent


def main():
    out=BASE/'cost_extension';out.mkdir(exist_ok=False)
    manifest=json.loads((BASE/'extension_data/MANIFEST.json').read_text())
    assert sha(BASE/'extension_data/benchmark.npz')==manifest['benchmark_sha256']
    with np.load(BASE/'extension_data/benchmark.npz') as z:d={k:z[k] for k in z.files}
    original=json.loads((BASE/'queries/FROZEN.json').read_text())
    assert sha(BASE/'core.py')==original['hashes'][str(BASE/'core.py')]
    frozen=dict(purpose='NEW_COST_ENDPOINT_ON_FIXED_TEN_OTHER_GEOMETRIES',policies=['entropy','center','random','witness'],
        main_control='center',budgets=[0,8,32,128,1024],workers=3,bootstrap_seed=71403,bootstrap_draws=10000,
        hashes={str(p):sha(p) for p in [Path(__file__),BASE/'core.py',BASE/'COST_EXTENSION_PROTOCOL.md',BASE/'extension_data/MANIFEST.json']})
    write(out/'FROZEN.json',frozen)
    templates={k:np.unpackbits(d[k],axis=1).reshape(-1,256,256).astype(bool) for k in ['a','b','domain']}
    start=time.monotonic();all_cost={};summaries={}
    for method in frozen['policies']:
        begun=time.monotonic();records={str(b):{k:[] for k in ['label','queries','revealed','interval_lower','interval_upper']} for b in frozen['budgets']}
        def one(i):
            t=d['template'][i];native=np.unpackbits(d['native'][i]).reshape(256,256).astype(bool)
            return rollout(ArrayOracle(native,d['coarse'][i]),d['coarse'][i],templates['a'][t],templates['b'][t],
                templates['domain'][t],bool(d['kind'][t]),float(d['ref'][t]),method,seed=17+i,budgets=frozen['budgets'])
        with ThreadPoolExecutor(max_workers=3) as pool:
            for i,r in enumerate(pool.map(one,range(len(d['truth'])))):
                for budget,v in r.items():
                    for k,value in v.items():records[budget][k].append(value)
                    assert v['label']<0 or v['label']==d['truth'][i]
                assert r['1024']['label']==d['truth'][i]
                if (i+1)%112==0:
                    status=dict(status='running',method=method,cases=i+1,total=len(d['truth']),elapsed_seconds=time.monotonic()-start)
                    write(out/'LIVE_PROGRESS.json',status);print(json.dumps(status),flush=True)
        for budget,r in records.items():np.savez_compressed(out/f'{method}_budget{budget}.npz',**{k:np.array(v) for k,v in r.items()})
        costs=np.array(records['1024']['queries']);all_cost[method]=costs
        summaries[method]=dict(mean_queries=float(costs.mean()),mean_native_area_fraction=float(costs.mean()/1024),
            elapsed_seconds=time.monotonic()-begun,all_native_classes_correct=True,
            by_geometry={g:float(costs[d['geometry']==g].mean()) for g in sorted(set(d['geometry']))})
        write(out/f'{method}_COMPLETE.json',summaries[method])
    geos=sorted(set(d['geometry']));rng=np.random.default_rng(71403);draw=rng.integers(0,len(geos),(10000,len(geos)))
    matrix={m:np.array([v['by_geometry'][g] for g in geos]) for m,v in summaries.items()}
    c=matrix['center'];w=matrix['witness'];den=c[draw].mean(1);diff=(c-w)[draw].mean(1)
    relative=np.divide(diff,den,out=np.zeros_like(diff),where=den>0)
    observed=(c.mean()-w.mean())/c.mean() if c.mean() else 0.
    ci=np.quantile(relative,[.025,.975]);pass_signal=observed>=.1 and ci[0]>0
    positive_states=d['state_truth'][d['state_id'],:2].any(1)
    ambiguous=np.array(np.load(out/'center_budget0.npz')['label'])<0
    strata={}
    for name,mask in [('all',np.ones(len(ambiguous),bool)),('initially_ambiguous',ambiguous),('positive_state_events',positive_states),('positive_local_violations',d['truth']==1)]:
        strata[name]=dict(n=int(mask.sum()),methods={m:float(cost[mask].mean()) if mask.any() else None for m,cost in all_cost.items()})
    assert all(sha(p)==h for p,h in frozen['hashes'].items())
    write(out/'SUMMARY.json',dict(methods=summaries,strata=strata,relative_saving_vs_center=observed,
        descriptive_ci95_relative_saving_vs_center=ci.tolist(),preliminary_cost_transfer_pass=bool(pass_signal),
        all_native_classes_correct=True,global_macro_f1=1.,geometries=len(geos),states=90,local_events=5040,
        elapsed_seconds=time.monotonic()-start,hypothesis_H7_original_confirmed=False,q1_novelty_established=False,
        limitation='Ideal binary readout, ten additional known design clips from same source, endpoint selected after first pilot, no hardware speedup claim.'))
    write(out/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(out/'SUMMARY.json'),elapsed_seconds=time.monotonic()-start))
    write(out/'LIVE_PROGRESS.json',dict(status='complete',elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':main()
