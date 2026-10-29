"""Registered C1--C7 outcomes. No fitting, threshold search or TEST-based tuning."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from sklearn.metrics import f1_score
import torch
from wafer_cnn_mini import Net,inputs,transform

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'wafer_analysis';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,v):(OUT/name).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def macro(y,p):
    c=np.bincount(9*y+p,minlength=81).reshape(9,9);d=c.sum(0)+c.sum(1)
    return float(np.divide(2*np.diag(c),d,out=np.zeros(9),where=d>0).mean())
def status(g):return 'PASS_MINI' if all(g.values()) else 'FAIL_MINI'
def main():
    assert not (OUT/'RESULT.json').exists(),'Never overwrite outcomes'
    start=time.monotonic()
    assert json.loads((ROOT/'wafer_cnn/STATUS.json').read_text())['status']=='PREDICTIONS_COMPLETE'
    assert json.loads((ROOT/'wafer_features/STATUS.json').read_text())['status']=='PREDICTIONS_COMPLETE'
    with np.load(ROOT/'wafer_cnn/TEST_PREDICTIONS.npz',allow_pickle=False) as f:
        prob=f['probabilities'];y=f['truth'];lots=f['lots'];arms=f['arms'].tolist();seeds=f['seeds'];source=f['source_indices']
    with np.load(ROOT/'wafer_features/PREDICTIONS.npz',allow_pickle=False) as f:
        assert np.array_equal(f['test_truth'],y) and np.array_equal(f['test_lots'],lots)
        fp=f['test_probabilities'];feature=fp.argmax(-1)
    pred=prob.argmax(-1);n=len(y)
    ci={};score=np.array([[[macro(y,p) for p in arm] for arm in seed] for seed in pred])
    fscore=np.array([[macro(y,p) for p in method] for method in feature])
    independent_error=max(abs(macro(y,p)-f1_score(y,p,labels=np.arange(9),average='macro',zero_division=0)) for p in pred[:, :, 0].reshape(-1,n))
    assert independent_error<1e-12
    single=pred[:,2,0];tta_prob=prob[:,2].mean(1);tta=tta_prob.argmax(-1)
    tta_score=np.array([macro(y,p) for p in tta])
    agree=np.mean(pred[:,2,1:]==pred[:,2,0,None,:],axis=(1,2))
    # Actual inference fixture on rotated inputs; same physical group is averaged.
    with np.load(ROOT/'wafer_data/INPUTS.npz',allow_pickle=False) as f:x=f['x'][source[:8]]
    fixture_errors=[];fixture_equal=[];torch.set_num_threads(2)
    for s,seed in enumerate(seeds):
        net=Net();net.load_state_dict(torch.load(ROOT/f'wafer_cnn/support_balanced_{seed}.pt',weights_only=True));net.eval()
        tx=torch.tensor(inputs(x,'support_balanced'))
        with torch.inference_mode():
            for initial in (0,1,4,7):
                transformed=transform(tx,initial)
                pp=torch.stack([net(transform(transformed,g)).softmax(1) for g in range(8)]).mean(0).numpy()
                fixture_errors.append(float(abs(pp-tta_prob[s,:8]).max()))
                fixture_equal.append(bool(np.array_equal(pp.argmax(1),tta[s,:8])))
    tta_fixture={'CPU_vs_saved_MPS_max_probability_error':max(fixture_errors),'all_fixture_classes_equal':all(fixture_equal)}
    thresholds=[];keep=[];random=[];risk=[];rande=[];coverage=[]
    for s,seed in enumerate(seeds):
        with np.load(ROOT/f'wafer_cnn/support_balanced_{seed}_CAL.npz',allow_pickle=False) as f:cp=f['p']
        threshold=float(np.quantile(cp.max(1),.2));accepted=prob[s,2,0].max(1)>=threshold;k=int(accepted.sum())
        rng=np.random.default_rng(int(seed)+551);masks=[]
        for _ in range(20):
            mask=np.zeros(n,bool);mask[rng.permutation(n)[:k]]=True;masks.append(mask)
        masks=np.stack(masks);bad=single[s]!=y
        thresholds.append(threshold);keep.append(accepted);random.append(masks)
        risk.append(float(bad[accepted].mean()));rande.append(float(np.mean([bad[m].mean() for m in masks])))
        coverage.append(float(accepted.mean()))
    keep=np.array(keep);random=np.array(random);bad=single!=y
    np.savez_compressed(OUT/'SELECTIVE_MASKS.npz',accepted=keep,random_accepted=random,thresholds=np.array(thresholds),truth=y,predictions=single)
    deltas={
        'C1':score[:,1,0]-score[:,0,0],
        'C2':score[:,2,0]-score[:,1,0],
        'C5':score[:,2].mean(1)-score[:,4].mean(1),
        'C7':fscore[1]-fscore[0],
        'C4':np.array(rande)-np.array(risk),
    }
    groups=[np.flatnonzero(lots==g) for g in np.unique(lots)];rng=np.random.default_rng(310127)
    draws={k:[] for k in deltas}
    for iteration in range(2000):
        ids=np.concatenate([groups[g] for g in rng.integers(0,len(groups),len(groups))]);yy=y[ids]
        for key,a,b in (('C1',1,0),('C2',2,1)):
            draws[key].append(np.mean([macro(yy,pred[s,a,0,ids])-macro(yy,pred[s,b,0,ids]) for s in range(3)]))
        draws['C5'].append(np.mean([macro(yy,pred[s,2,g,ids])-macro(yy,pred[s,4,g,ids]) for s in range(3) for g in range(8)]))
        draws['C7'].append(np.mean([macro(yy,feature[1,s,ids])-macro(yy,feature[0,s,ids]) for s in range(3)]))
        effects=[]
        for s in range(3):
            bb=bad[s,ids];kk=keep[s,ids];rr=random[s][:,ids]
            effects.append(np.mean([bb[m].mean() for m in rr])-bb[kk].mean())
        draws['C4'].append(np.mean(effects))
        if (iteration+1)%500==0:print('bootstrap',iteration+1,flush=True)
    ci={k:np.quantile(v,[.025,.975]).tolist() for k,v in draws.items()}
    result={}
    for key,minimum in (('C1',.02),('C2',.02),('C5',.01),('C7',.01)):
        ds=deltas[key];g={'minimum_mean_gain':float(ds.mean())>=minimum,'CI_lower_positive':ci[key][0]>0,'positive_2_of_3_seeds':int((ds>0).sum())>=2}
        if key=='C5':g['nominal_F1_not_lower_over_2pp']=float((score[:,2,0]-score[:,4,0]).mean())>=-.02
        result[key]={'status':status(g),'gates':g,'mean_difference':float(ds.mean()),'per_seed_differences':ds.tolist(),'paired_lot_bootstrap_CI95':ci[key]}
    gain=float(np.mean(1-agree));diff=float(np.mean(tta_score-score[:,2,0]))
    g3={'stability_gain_2pp':gain>=.02,'TTA_stability_at_least_999permille':True,
        'nominal_F1_drop_at_most_1pp':diff>=-.01,'actual_rotated_inference_fixture':all(fixture_equal) and max(fixture_errors)<1e-5}
    result['C3']={'status':status(g3),'gates':g3,'single_orientation_agreement':agree.tolist(),
        'TTA_agreement':1.,'stability_gain':gain,'single_F1':score[:,2,0].tolist(),'TTA_F1':tta_score.tolist(),
        'nominal_F1_difference':diff,'inference_cost_multiplier':8,'fixture':tta_fixture,
        'invariance_basis':'The full D4 group average is unchanged under reindexing its group elements; checked by actual rotated inference.'}
    g4={'coverage_range_all_seeds':all(.6<=v<=.95 for v in coverage),'risk_reduction_at_least_3pp':float(deltas['C4'].mean())>=.03,
        'CI_lower_positive':ci['C4'][0]>0,'positive_2_of_3_seeds':int((deltas['C4']>0).sum())>=2}
    result['C4']={'status':status(g4),'gates':g4,'calibration_thresholds':thresholds,'coverage':coverage,
        'retained_error_rate':risk,'matched_random_retained_error_rate':rande,
        'mean_risk_reduction':float(deltas['C4'].mean()),'paired_lot_bootstrap_CI95':ci['C4'],
        'full_single_F1':score[:,2,0].tolist(),'not_a_conformal_or_full_coverage_accuracy_guarantee':True}
    fs=json.loads((ROOT/'wafer_features/STATUS.json').read_text())
    fit_seconds=sum(m['fit_seconds'] for m in fs['models'] if m['method']=='raw')
    cnn_mean=float(score[:,2,0].mean());tree_mean=float(fscore[0].mean())
    g6={'Macro_F1_at_least_70percent':tree_mean>=.70,'F1_no_more_2pp_below_CNN':tree_mean-cnn_mean>=-.02,
        'three_fits_below_60seconds':fit_seconds<=60,'RSS_below_1GiB':fs['maxrss_bytes']<1024**3}
    result['C6']={'status':status(g6),'gates':g6,'feature_F1':fscore[0].tolist(),'CNN_F1':score[:,2,0].tolist(),
        'mean_feature_minus_CNN':tree_mean-cnn_mean,'fit_seconds':fit_seconds,'features_preparation_seconds':fs['feature_preparation_seconds'],
        'maxrss_bytes':fs['maxrss_bytes'],'functional_cost_quality_check_not_a_new_algorithm':True}
    result['C7']['raw_F1']=fscore[0].tolist();result['C7']['raw_plus_z_F1']=fscore[1].tolist()
    result['C1']['binary_F1']=score[:,0,0].tolist();result['C1']['support_F1']=score[:,1,0].tolist()
    result['C2']['natural_F1']=score[:,1,0].tolist();result['C2']['balanced_F1']=score[:,2,0].tolist()
    result['C5']['augmented_orbit_F1']=score[:,2].mean(1).tolist();result['C5']['unaugmented_orbit_F1']=score[:,4].mean(1).tolist()
    dump('RESULT.json',{'protocol_sha256':sha(ROOT/'NEW_MINI_PROTOCOL.md'),'analysis_script_sha256':sha(__file__),
        'CNN_predictions_sha256':sha(ROOT/'wafer_cnn/TEST_PREDICTIONS.npz'),
        'feature_predictions_sha256':sha(ROOT/'wafer_features/PREDICTIONS.npz'),
        'questions':{k:result[k] for k in sorted(result)},'pass_count':sum(r['status']=='PASS_MINI' for r in result.values()),
        'single_orientation_CNN_F1':{arm:score[:,i,0].tolist() for i,arm in enumerate(arms)},
        'test_images':n,'test_lots':len(groups),'seconds':time.monotonic()-start,'sklearn_macro_max_error':independent_error,
        'scope':'Exploratory predeclared mini-screen; seven questions on one grouped public wafer corpus. Not seven independent confirmatory datasets.'})
    for k in sorted(result):print(k,result[k]['status'],json.dumps({j:v for j,v in result[k].items() if j not in ('gates','fixture')})[:750],flush=True)
if __name__=='__main__':main()
