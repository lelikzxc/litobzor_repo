"""Evaluate fixed final shared CNNs with frozen query traces; no fitting."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.metrics import f1_score,confusion_matrix
from learning import FallbackCNN,load_data,predict,sha,write

BASE=Path(__file__).resolve().parent


def global_predictions(labels,d,case_ids,state_ids):
    result=[]
    for state in state_ids:
        selected=d['state_id'][case_ids]==state
        p=labels[selected];kind=d['kind'][d['template'][case_ids[selected]]]
        assert len(p)==56
        result.append([(p[~kind]==1).any(),(p[kind]==1).any(),(p==2).any()])
    return np.array(result,bool)


def score(y,p,geometry):
    w=np.array([1/np.sum(geometry==g) for g in geometry])
    ff=f1_score(y,p,average=None,sample_weight=w,zero_division=0)
    return dict(f1=ff.tolist(),macro_f1=float(ff[:2].mean()),exact_states=int((y==p).all(1).sum()),
        positive_states=y.sum(0).tolist(),states=len(y))


def main():
    assert torch.backends.mps.is_available()
    assert __import__('os').environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
    assert json.loads((BASE/'queries/COMPLETE.json').read_text())['status']=='COMPLETE'
    trained=json.loads((BASE/'models/COMPLETE.json').read_text())
    out=BASE/'evaluation';out.mkdir(exist_ok=False)
    d=load_data();split=json.loads((BASE/'data/FROZEN.json').read_text())
    ids=np.flatnonzero(np.isin(d['geometry'],split['evaluation']))
    states=np.flatnonzero(np.isin(d['state_geometry'],split['evaluation']))
    truth=d['state_truth'][states];geometry=d['state_geometry'][states]
    sources=[Path(__file__),BASE/'learning.py',BASE/'PROTOCOL.md',BASE/'models/COMPLETE.json',BASE/'queries/COMPLETE.json',BASE/'data/MANIFEST.json']
    sources.extend(BASE/'models'/f'seed{seed}.pt' for seed in [17,41])
    sources.extend(BASE/'queries'/f'{m}_budget{b}.npz' for m in ['entropy','center','random','witness'] for b in [0,8,32,128])
    frozen=dict(purpose='DEVELOPMENT_FEASIBILITY_ONLY_NO_NEW_SELECTION',primary_policy='witness',primary_control='entropy',primary_budget=32,
        hashes={str(p):sha(p) for p in sources},evaluation_geometry=split['evaluation'],optimizer_steps=0)
    write(out/'FROZEN.json',frozen);rows=[];preds={};device=torch.device('mps');start=time.monotonic()
    for seed in [17,41]:
        path=BASE/'models'/f'seed{seed}.pt'
        assert sha(path)==next(v for v in trained['seeds'] if v['seed']==seed)['weights_sha256']
        model=FallbackCNN().to(device);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
        zero_prediction=None
        for method in ['entropy','center','random','witness']:
            for budget in [0,8,32,128]:
                with np.load(BASE/'queries'/f'{method}_budget{budget}.npz') as z:query={k:z[k][ids] for k in z.files}
                revealed=np.unpackbits(query['revealed'],axis=1).reshape(-1,32,32).astype(bool)
                cert=query['label'];uncertain=np.flatnonzero(cert<0)
                labels=cert.copy();prob=np.zeros((len(ids),3),np.float32)
                if len(uncertain):
                    pp=predict(model,d,ids[uncertain],revealed[uncertain],device)
                    prob[uncertain]=pp;labels[uncertain]=pp.argmax(1)
                certain=np.flatnonzero(cert>=0);prob[certain,cert[certain]]=1
                assert np.all(labels>=0)
                assert np.all(cert[certain]==d['truth'][ids[certain]])
                if budget==0:
                    if zero_prediction is None:zero_prediction=labels.copy()
                    else:assert np.array_equal(zero_prediction,labels),'Budget0 differs across policies'
                yhat=global_predictions(labels,d,ids,states);sc=score(truth,yhat,geometry)
                w=np.array([1/np.sum(d['geometry'][ids]==g) for g in d['geometry'][ids]])
                local_f1=f1_score(d['truth'][ids],labels,labels=[0,1,2],average=None,sample_weight=w,zero_division=0)
                row=dict(seed=seed,method=method,budget=budget,global_score=sc,local_f1=local_f1.tolist(),
                    local_macro_f1=float(local_f1.mean()),certified_events=int((cert>=0).sum()),events=len(ids),
                    certificate_coverage=float((cert>=0).mean()),mean_queries=float(query['queries'].mean()),
                    local_high_resolution_area_fraction=float(query['queries'].mean()/1024),
                    local_confusion=confusion_matrix(d['truth'][ids],labels,labels=[0,1,2]).tolist())
                rows.append(row);preds[(seed,method,budget)]=yhat
                np.savez_compressed(out/f'seed{seed}_{method}_budget{budget}.npz',probability=prob,local_prediction=labels,
                    local_truth=d['truth'][ids],certificate=cert,case_indices=ids,state_ids=states,
                    prediction=yhat,y=truth,geometry=geometry,queries=query['queries'])
                print(json.dumps(row),flush=True)
        del model;torch.mps.empty_cache()
    geos=sorted(set(geometry));rng=np.random.default_rng(93257);draws=rng.integers(0,len(geos),size=(10000,len(geos)))
    bootstrap={}
    for budget in [0,8,32,128]:
        bootstrap_scores={}
        for seed in [17,41]:
            for method in ['witness','entropy']:
                table=[]
                for g in geos:
                    selected=geometry==g;y=truth[selected,:2];p=preds[(seed,method,budget)][selected,:2]
                    table.append(np.stack([(y&p).sum(0),(~y&p).sum(0),(y&~p).sum(0)],-1))
                # Nine states per geometry; repeated cluster counts preserve
                # the prescribed equal-geometry weighting exactly.
                sampled=np.array(table)[draws].sum(1);tp,fp,fn=np.moveaxis(sampled,-1,0)
                den=2*tp+fp+fn;ff=np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0)
                bootstrap_scores[(seed,method)]=ff.mean(1)
        deltas=np.mean([bootstrap_scores[(s,'witness')]-bootstrap_scores[(s,'entropy')] for s in [17,41]],axis=0)
        seed_delta=[next(r['global_score']['macro_f1'] for r in rows if r['seed']==s and r['method']=='witness' and r['budget']==budget)
            -next(r['global_score']['macro_f1'] for r in rows if r['seed']==s and r['method']=='entropy' and r['budget']==budget) for s in [17,41]]
        bootstrap[str(budget)]=dict(observed_delta_pp=100*float(np.mean(seed_delta)),seed_delta_pp=(100*np.array(seed_delta)).tolist(),
            descriptive_ci95_pp=(100*np.quantile(deltas,[.025,.975])).tolist(),draws=10000,
            limitation='Four development geometry groups; fixed models/traces; no training or dataset-selection uncertainty.')
    support=json.loads((BASE/'data/MANIFEST.json').read_text())['support']['evaluation']
    primary=bootstrap['32'];data_ok=min(support['positive_geometries'][:2])>=2
    signal=data_ok and primary['observed_delta_pp']>=1 and primary['descriptive_ci95_pp'][0]>0 and min(primary['seed_delta_pp'])>0
    assert all(sha(p)==h for p,h in frozen['hashes'].items())
    write(out/'SUMMARY.json',dict(rows=rows,paired_bootstrap=bootstrap,primary=primary,data_support_pass=data_ok,
        preliminary_comparative_signal_pass=bool(signal),hypothesis_confirmed=False,q1_novelty_established=False,
        elapsed_seconds=time.monotonic()-start,frozen_sha256=sha(out/'FROZEN.json')))
    write(out/'COMPLETE.json',dict(status='COMPLETE',summary_sha256=sha(out/'SUMMARY.json'),elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':main()
