"""Independent metrics, input leakage checks and exact-count ambiguity examples."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from core import certificate,initial_bounds,expand,block_sum,native_label
from learning import FallbackCNN,load_data,features,sha,write

BASE=Path(__file__).resolve().parent


def main():
    d=load_data();split=json.loads((BASE/'data/FROZEN.json').read_text())
    summary=json.loads((BASE/'evaluation/SUMMARY.json').read_text())
    metric_checks=0;cert_checks=0;replay=[]
    ids=np.flatnonzero(np.isin(d['geometry'],split['evaluation']))
    for row in summary['rows']:
        path=BASE/'evaluation'/f'seed{row["seed"]}_{row["method"]}_budget{row["budget"]}.npz'
        with np.load(path) as z:a={k:z[k] for k in z.files}
        w=np.array([1/np.sum(a['geometry']==g) for g in a['geometry']])
        f=f1_score(a['y'],a['prediction'],average=None,sample_weight=w,zero_division=0)
        np.testing.assert_allclose(f,row['global_score']['f1'],atol=1e-12);metric_checks+=1
        state_prediction=[]
        for state in a['state_ids']:
            jj=d['state_id'][ids]==state;p=a['local_prediction'][jj];kind=d['kind'][d['template'][ids[jj]]]
            state_prediction.append([(p[~kind]==1).any(),(p[kind]==1).any(),(p==2).any()])
        np.testing.assert_array_equal(state_prediction,a['prediction'])
    for method in ['entropy','center','random','witness']:
        previous=None
        for budget in [0,8,32,128]:
            with np.load(BASE/'queries'/f'{method}_budget{budget}.npz') as z:a={k:z[k] for k in z.files}
            revealed=np.unpackbits(a['revealed'],axis=1).reshape(-1,32,32).astype(bool)
            np.testing.assert_array_equal(revealed.sum((1,2)),a['queries'])
            assert np.all(a['queries']<=budget)
            assert not np.any(revealed&((d['coarse']==0)|(d['coarse']==64)))
            mask=a['label']>=0;np.testing.assert_array_equal(a['label'][mask],d['truth'][mask]);cert_checks+=int(mask.sum())
            if previous is not None:assert np.all(previous<=revealed)
            previous=revealed
            # Replay one predetermined case per geometry using only what its
            # recorded action mask permits the observer to know.
            for g in sorted(set(d['geometry'])):
                i=int(np.flatnonzero(d['geometry']==g)[0]);ti=d['template'][i]
                native=np.unpackbits(d['native'][i]).reshape(256,256).astype(bool)
                lo,hi=initial_bounds(d['coarse'][i]);known=expand(revealed[i])
                lo[known]=native[known];hi[known]=native[known]
                aa,bb,domain=[np.unpackbits(d[k][ti]).reshape(256,256).astype(bool) for k in ['a','b','domain']]
                cert=certificate(lo,hi,aa,bb,domain,bool(d['kind'][ti]),float(d['ref'][ti]))
                assert cert['label']==a['label'][i]
                np.testing.assert_array_equal(cert['lower'],a['interval_lower'][i]);np.testing.assert_array_equal(cert['upper'],a['interval_upper'][i])
    # Changing every unobserved mixed-block pixel arrangement must not change
    # either model input. The coarse sums remain exactly the same.
    sample=ids[:16];reveal=np.zeros((len(sample),32,32),bool)
    original=features(d,sample,reveal);shadow=dict(d);shadow['native']=d['native'].copy()
    for i in sample:
        raw=np.unpackbits(d['native'][i]).reshape(256,256)
        tiles=raw.reshape(32,8,32,8).transpose(0,2,1,3)
        changed=tiles[...,::-1].transpose(0,2,1,3).reshape(256,256)
        np.testing.assert_array_equal(block_sum(raw),block_sum(changed))
        shadow['native'][i]=np.packbits(changed.ravel())
    after=features(shadow,sample,reveal)
    for a,b in zip(original,after):np.testing.assert_array_equal(a,b)
    # Repeat model inference independently on CPU from the saved final weights.
    for seed in [17,41]:
        with np.load(BASE/'evaluation'/f'seed{seed}_entropy_budget0.npz') as z:a={k:z[k] for k in z.files}
        selected=np.flatnonzero(a['certificate']<0)[:32]
        xx,ss=features(d,ids[selected],np.zeros((len(selected),32,32),bool))
        model=FallbackCNN();model.load_state_dict(torch.load(BASE/'models'/f'seed{seed}.pt',map_location='cpu',weights_only=True));model.eval()
        with torch.no_grad():pp=model(torch.tensor(xx),torch.tensor(ss)).softmax(1).numpy()
        err=float(np.max(np.abs(pp-a['probability'][selected])));assert err<1e-4
        replay.append(dict(seed=seed,probability_max_cpu_mps_error=err))
    # Explanatory counterexamples, not synthetic lithography ground truth.
    with np.load(BASE/'queries/entropy_budget0.npz') as z:initial=z['label'].copy()
    candidates=ids[initial[ids]<0];changed_records=[];examples=[];total=0
    for i in candidates:
        raw=np.unpackbits(d['native'][i]).reshape(256,256).astype(bool);ti=d['template'][i]
        aa,bb,domain=[np.unpackbits(d[k][ti]).reshape(256,256).astype(bool) for k in ['a','b','domain']]
        tiles=raw.reshape(32,8,32,8).transpose(0,2,1,3)
        arrangements={'tile_horizontal_flip':tiles[...,::-1], 'tile_vertical_flip':tiles[...,::-1,:],
            'tile_rotation90':np.rot90(tiles,axes=(2,3)),
            'tile_compact':np.sort(tiles.reshape(32,32,64),axis=-1).reshape(32,32,8,8)}
        for name,t in arrangements.items():
            alternative=t.transpose(0,2,1,3).reshape(256,256)
            np.testing.assert_array_equal(block_sum(raw),block_sum(alternative));total+=1
            label=native_label(alternative,aa,bb,domain,bool(d['kind'][ti]),float(d['ref'][ti]))
            if label!=d['truth'][i]:
                changed_records.append(dict(case_index=int(i),geometry=str(d['geometry'][i]),kind='forbidden' if d['kind'][ti] else 'required',
                    original_label=int(d['truth'][i]),alternative_label=int(label),transform=name))
                if len(examples)<8:examples.append((i,raw.copy(),alternative.copy(),aa.copy(),bb.copy()))
    write(BASE/'MECHANISM.json',dict(purpose='INFORMATION_AMBIGUITY_NOT_PHYSICAL_RESIST_SIMULATION',
        initial_uncertain_evaluation_events=len(candidates),fixed_transformations_per_event=4,
        transformations_tested=total,class_changes=len(changed_records),events_with_class_change=len({r['case_index'] for r in changed_records}),
        records=changed_records,limitation='Within-block permutations preserve exact counts but need not be attainable lithographic outcomes. They establish ambiguity only over the declared unconstrained binary-image family.'))
    if examples:
        np.savez_compressed(BASE/'mechanism_examples.npz',case_index=np.array([v[0] for v in examples]),
            original=np.array([v[1] for v in examples]),alternative=np.array([v[2] for v in examples]),
            a=np.array([v[3] for v in examples]),b=np.array([v[4] for v in examples]))
    for folder in ['data','models','queries','evaluation']:
        frozen=json.loads((BASE/folder/'FROZEN.json').read_text())
        for path,h in frozen['hashes'].items():assert sha(path)==h,path
    extension=None
    if (BASE/'cost_extension/COMPLETE.json').exists():
        with np.load(BASE/'extension_data/benchmark.npz') as z:ed={k:z[k] for k in z.files}
        ext=json.loads((BASE/'cost_extension/SUMMARY.json').read_text());methods={}
        for method in ['entropy','center','random','witness']:
            with np.load(BASE/'cost_extension'/f'{method}_budget1024.npz') as z:a={k:z[k] for k in z.files}
            np.testing.assert_array_equal(a['label'],ed['truth'])
            revealed=np.unpackbits(a['revealed'],axis=1).reshape(-1,32,32)
            np.testing.assert_array_equal(revealed.sum((1,2)),a['queries'])
            methods[method]={g:float(a['queries'][ed['geometry']==g].mean()) for g in sorted(set(ed['geometry']))}
            for g,v in methods[method].items():assert v==ext['methods'][method]['by_geometry'][g]
            cert_checks+=len(a['label'])
        c=np.array(list(methods['center'].values()));w=np.array(list(methods['witness'].values()))
        observed=(c.mean()-w.mean())/c.mean()
        assert np.isclose(observed,ext['relative_saving_vs_center'])
        draws=np.random.default_rng(71403).integers(0,len(c),(10000,len(c)))
        ratio=1-w[draws].mean(1)/c[draws].mean(1)
        np.testing.assert_allclose(np.quantile(ratio,[.025,.975]),ext['descriptive_ci95_relative_saving_vs_center'],atol=1e-12)
        extension=dict(exact_labels_checked=4*len(ed['truth']),saving_and_bootstrap_verified=True)
        for folder in ['extension_data','cost_extension']:
            for path,h in json.loads((BASE/folder/'FROZEN.json').read_text())['hashes'].items():assert sha(path)==h,path
    write(BASE/'VERIFICATION.json',dict(status='PASS',sklearn_global_scores_checked=metric_checks,
        correct_certificates_checked=cert_checks,recorded_observation_replay_cases=128,
        hidden_pixel_leakage_test='PASS',cpu_model_replay=replay,
        exact_count_ambiguity_examples_verified=True,extension=extension,
        source_sha256=sha(__file__),evaluation_summary_sha256=sha(BASE/'evaluation/SUMMARY.json')))
    print(json.dumps(dict(status='PASS',metric_checks=metric_checks,certificates=cert_checks,
        ambiguity_cases=len({r['case_index'] for r in changed_records}),extension=extension)),flush=True)


if __name__=='__main__':main()
