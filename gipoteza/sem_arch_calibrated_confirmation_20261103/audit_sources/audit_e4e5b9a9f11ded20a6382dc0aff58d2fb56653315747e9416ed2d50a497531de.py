"""Independent prospective calibration/model, native-data and final-result audit.

No downloads, optimizer updates or fresh-model inference. Old-VAL technical
forward evaluation is allowed; saved new predictions are audited numerically.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import re
import shutil
import time
import zipfile
import zlib

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
G, REPO = ROOT.parent, ROOT.parents[1]
DEV = G/'sem_arch_calibration_dev_20261103'
MODEL = G/'sem_arch_groupcnn_20261102'
UNTIED = G/'sem_arch_untied_control_20261103'
PREVIOUS = G/'sem_arch_bce_confirmation_20261103'
GROUP_PREVIOUS = G/'sem_arch_groupcnn_confirmation_20261102'
OLD = G/'sem_arch_orbit_confirmation_20261102'
PILOT, ORIGINAL, SHARED = G/'sem_architecture_pilot_20261102', G/'sem_confirmation_20261028', G/'shared_data/ic_sem_m2'
PRIOR = [ORIGINAL,G/'sem_cldice_confirmation_20261030',G/'sem_cldice_expansion_20261102',
         G/'sem_scarcity_20261102',PILOT,OLD,GROUP_PREVIOUS,PREVIOUS]
ARMS, SEEDS, MODES = ['group6','untied24_single','plain12_C4pooled'],[5701,5721,5757],['final_calibrated','fixed05']
FRAME_PREFIX, EVAL_PREFIX = 'SEM-C4-CALIB-CONFIRM-20261103/','SEM-C4-CALIB-EVAL-20261103/'
CAL_FRAME_PREFIX, CAL_ROW_PREFIX = 'SEM-C4-FINAL-CALIB-FRAME-20261103/','SEM-C4-CALIB-ROW-20261103/'
THRESHOLDS = np.array([round(.1+.01*i,2) for i in range(81)])


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(path): return str(Path(path).resolve().relative_to(REPO))
def resolve(path): return Path(path) if Path(path).is_absolute() else REPO/path
def load(path): return torch.load(path,map_location='cpu',weights_only=True)
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


ref = module('calibrated_confirm_independent_graph_model',MODEL/'audit.py')
native_ref = module('calibrated_confirm_independent_native',GROUP_PREVIOUS/'audit.py')


def error(expected,observed):
    if isinstance(expected,dict):
        if not isinstance(observed,dict) or set(expected)!=set(observed): return float('inf')
        return max([error(v,observed[k]) for k,v in expected.items()]+[0.])
    if isinstance(expected,(list,tuple)):
        if not isinstance(observed,(list,tuple)) or len(expected)!=len(observed): return float('inf')
        return max([error(x,y) for x,y in zip(expected,observed)]+[0.])
    if isinstance(expected,(str,bool)) or expected is None: return 0. if expected==observed else float('inf')
    return abs(float(expected)-float(observed))


def macro(matrix):
    matrix=np.asarray(matrix,dtype=np.float64);den=matrix.sum(-1)+matrix.sum(-2)
    return np.divide(2*np.diagonal(matrix,axis1=-2,axis2=-1),den,out=np.zeros_like(den),where=den>0).mean(-1)


def cm(truth,prediction):
    return np.bincount(9*np.asarray(truth,dtype=np.int64)+np.asarray(prediction,dtype=np.int64),minlength=81).reshape(9,9)


def rank_ap(truth,scores):
    truth=np.asarray(truth,bool).ravel();scores=np.asarray(scores).ravel();positive=int(truth.sum())
    if not positive:return 0.
    order=np.argsort(-scores,kind='stable');ranked=scores[order];labels=truth[order]
    ends=np.flatnonzero(np.r_[ranked[1:]!=ranked[:-1],True]);tp=np.cumsum(labels,dtype=np.int64)[ends]
    return float(np.dot(np.diff(np.r_[0,tp])/positive,tp/(ends+1)))


def quantiles(values):
    ordered=np.sort(np.asarray(values,float));result=[]
    for q in [.0125,.9875]:
        at=(len(ordered)-1)*q;lo=int(np.floor(at));hi=int(np.ceil(at))
        result.append(float(ordered[lo]+(ordered[hi]-ordered[lo])*(at-lo)))
    return result


def choose(scores):
    return max(range(81),key=lambda i:(float(scores[i]),-abs(i-40),-i))


def audited_development():
    result,audit=read(DEV/'RESULT.json'),read(DEV/'AUDIT.json')
    assert result['status']=='COMPLETE_CALIBRATION_DEVELOPMENT_ANALYSIS'
    assert audit['status']=='PASS_CALIBRATION_AUDIT' and all(audit['checks'].values())
    assert audit['checks_passed']==audit['checks_total']==34
    assert all(sha(resolve(p))==s for p,s in audit['input_sha256'].items())
    assert result['next_study_eligibility']['passed'] and all(result['next_study_eligibility']['guards'].values())
    assert result['fit_count']==result['optimizer_updates']==0 and not result['new_confirmation']
    return result,audit


def independent_calibration():
    audited_development()
    with np.load(DEV/'DATA.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(DEV/'CALIBRATION.npz',allow_pickle=False) as z:cache={k:z[k] for k in z.files}
    frames=sorted(np.unique(data['frame']).tolist(),key=lambda f:hashlib.sha256((CAL_FRAME_PREFIX+str(f)).encode()).hexdigest())[:32]
    ids=[]
    for frame in frames:
        candidates=np.flatnonzero(data['frame']==frame).tolist()
        candidates.sort(key=lambda i:hashlib.sha256((CAL_ROW_PREFIX+str(data['raw_D4_sha256'][i])).encode()).hexdigest())
        ids.extend(candidates[:16])
    ids=np.array(sorted(ids),np.int64);pool=cache['calibration_pool_indices']
    positions=np.searchsorted(pool,ids);assert np.array_equal(pool[positions],ids)
    assert len(frames)==32 and len(ids)==512 and len(set(ids.tolist()))==512
    scores=np.empty((3,3,81));selected=np.empty((3,3),np.int64)
    for a in range(3):
        for s in range(3):
            for t in range(81):scores[a,s,t]=macro(cm(data['label'][ids],cache['calibration_pool_classes'][a,s,t,positions]))
            selected[a,s]=choose(scores[a,s])
    return dict(data_indices=ids,frame_ids=np.array(frames),thresholds=THRESHOLDS,scores=scores,
        selected_indices=selected,selected_thresholds=THRESHOLDS[selected],truth=data['label'][ids],source_pool_positions=positions),data


def independent_models(calibration):
    result=read(UNTIED/'RESULT.json');audit=read(UNTIED/'AUDIT.json')
    assert audit['status'].startswith('PASS') and all(audit['checks'].values())
    assert audit['input_sha256'][rel(UNTIED/'RESULT.json')]==sha(UNTIED/'RESULT.json')
    lookup={(r['architecture'],r['seed']):r for r in result['rows']};rows=[];states={}
    for a,arm in enumerate(ARMS):
        for s,seed in enumerate(SEEDS):
            old=lookup[arm,seed];free=resolve(old['model_file']);val=resolve(old['probability_file'])
            pred=free.with_name(free.stem+'_deployed.pt') if arm=='group6' else free
            assert sha(free)==old['model_sha256'] and sha(val)==old['probability_sha256']
            for p in [free,pred,val]:assert audit['input_sha256'][rel(p)]==sha(p)
            state=load(free)['model'];prediction=load(pred)
            count=sum(x.numel() for x in state.values());assert count==[2671,10681,2749][a]
            if arm=='group6':
                expected=ref.reference_deploy_state(state)
                assert prediction['free_model_sha256']==sha(free) and all(torch.equal(v,prediction['model'][k]) for k,v in expected.items())
            states[arm,seed]=prediction['model']
            rows.append(dict(architecture=arm,seed=seed,budget=27,free_model_file=rel(free),free_model_sha256=sha(free),
                prediction_model_file=rel(pred),prediction_model_sha256=sha(pred),prediction_width=12 if a==2 else 24,
                prediction_mode='C4' if a==2 else 'single',free_parameters=count,MACs_per_image=[1527552,1527552,1562112][a],
                final_calibrated_threshold=float(calibration['selected_thresholds'][a,s]),fixed05_threshold=.5,
                source_VAL_file=rel(val),source_VAL_sha256=sha(val)))
    return rows,states


def source_plan_checks(plan=None):
    previous=read(PREVIOUS/'SOURCE_PLAN.json');selection=read(PREVIOUS/'SELECTION.json')
    excluded=set(previous['excluded_previous_frames'])|set(previous['test_frames'])
    assert len(previous['excluded_previous_frames'])==110 and len(previous['test_frames'])==24 and len(excluded)==134
    assert selection['source_plan_sha256']==sha(PREVIOUS/'SOURCE_PLAN.json')
    listing=read(SHARED/'sem_zip_listing.json')
    available=[int(m.group(1)) for r in listing if (m:=re.fullmatch(r'dataset/sems/m2/sem(\d+)\.png',r['name']))]
    chosen=sorted(set(available)-excluded,key=lambda f:hashlib.sha256((FRAME_PREFIX+str(f)).encode()).hexdigest())[:24]
    previous_frames={int(r['frame']) for folder in PRIOR for r in read(folder/'DATA_MANIFEST.json')['records']}
    hashes={str((folder/'DATA_MANIFEST.json').relative_to(G)):sha(folder/'DATA_MANIFEST.json') for folder in PRIOR}
    checks=dict(independent134_frame_exclusion_chain_and_all8_prior_manifest_frames=(previous_frames<=excluded and len(PRIOR)==8),
        independent321_listing_and_fixed24_SHA_frame_rank=(len(available)==len(set(available))==321 and len(chosen)==24 and not set(chosen)&excluded))
    if plan is not None:
        checks['prospective_source_plan24_frames134_excluded8_manifests_exact']=(plan['test_frames']==chosen
            and plan['excluded_previous_frames']==sorted(excluded) and plan['excluded_frame_count']==134
            and plan['prior_manifests_sha256']==hashes and plan['selection_prefix']==FRAME_PREFIX
            and plan['evaluation']['selection_prefix']==EVAL_PREFIX and plan['evaluation']['patches_per_frame']==256
            and plan['evaluation']['target_examples']==6144 and plan['evaluation']['replacement_on_gate_failure'] is False
            and plan['evaluation']['minimum_examples_each_class']==8 and plan['evaluation']['minimum_frames_each_class']==3
            and plan['archive_listing_sha256']==sha(SHARED/'sem_zip_listing.json'))
    return checks,dict(frames=chosen,excluded=134,prior_manifests=8,prior_record_frames=len(previous_frames))


def subject_common():return module('calibrated_confirmation_common_subject',ROOT/'common.py')


def frame_tables(predictions,truth,frames):
    values=np.unique(frames);table=np.zeros((2,3,3,len(values),9,9),np.int64)
    for f,frame in enumerate(values):
        keep=frames==frame
        for m in range(2):
            for a in range(3):
                for s in range(3):table[m,a,s,f]=cm(truth[keep],predictions[m,a,s,keep])
    return table,values


def frame_AP(probability,target,frames):
    values=np.unique(frames);result=np.empty((3,3,len(values)))
    for a in range(3):
        for s in range(3):
            for f,frame in enumerate(values):
                keep=frames==frame;result[a,s,f]=rank_ap(target[keep],probability[a,s,keep])
    return result


def weighted_bootstrap(tables,ap,seed_ids,frame_ids):
    repeats=len(seed_ids);nframes=tables.shape[3]
    flat=tables[0,:2].transpose(2,0,1,3,4).reshape(nframes,-1)
    flat_ap=ap[:2].transpose(2,0,1).reshape(nframes,-1)
    f1=np.empty((2,repeats));average_precision=np.empty_like(f1)
    for start in range(0,repeats,211):
        end=min(repeats,start+211);size=end-start;fw=np.zeros((size,nframes),np.int64);sw=np.zeros((size,3),np.int64)
        for i in range(size):
            fw[i]=np.bincount(frame_ids[start+i],minlength=nframes)
            sw[i]=np.bincount(seed_ids[start+i],minlength=3)
        sampled=(fw@flat).reshape(size,2,3,9,9)
        f1[:,start:end]=((macro(sampled)*sw[:,None,:]).sum(-1)/3).T
        sampled_ap=(fw@flat_ap).reshape(size,2,3)/nframes
        average_precision[:,start:end]=((sampled_ap*sw[:,None,:]).sum(-1)/3).T
    return f1,average_precision,f1[0]-f1[1],average_precision[0]-average_precision[1]


def synthetic_statistics(config):
    subject=module('calibrated_confirm_statistical_subject',ROOT/'stats.py')
    rng=np.random.default_rng(6143);frames=np.repeat(np.arange(24),13);truth=rng.integers(0,9,len(frames))
    target=rng.random((len(frames),12,12))<.45
    pred=np.empty((2,3,3,len(frames)),np.int64);prob=np.empty((3,3,len(frames),12,12))
    for a in range(3):
        for s in range(3):
            for m in range(2):
                p=truth.copy();change=rng.random(len(p))<(.04+.18*s+.05*a+.015*m)
                p[change]=(p[change]+rng.integers(1,9,change.sum()))%9;pred[m,a,s]=p
            prob[a,s]=np.clip(.5+(target-.5)*(.7-.13*s-.045*a)+rng.normal(0,.23, target.shape),0,1)
    iou=rng.uniform(.7,.9,pred.shape)
    analyses,actual=subject.analyze(pred,iou,prob,truth,target,frames,config,True)
    table,values=frame_tables(pred,truth,frames);ap=frame_AP(prob,target,frames)
    draws=np.random.default_rng(config['bootstrap_seed']);si=draws.integers(0,3,(10000,3));fi=draws.integers(0,24,(10000,24))
    f1,aps,df,da=weighted_bootstrap(table,ap,si,fi)
    max_error=max(float(np.abs(table-actual['confusion']).max()),float(np.abs(ap-actual['per_frame_AP']).max()),
        float(np.abs(f1-actual['bootstrap_F1']).max()),float(np.abs(aps-actual['bootstrap_AP']).max()),
        float(np.abs(df-actual['delta_F1']).max()),float(np.abs(da-actual['delta_AP']).max()))
    brute=np.empty((2,7));wrong=np.empty_like(brute)
    for draw in range(7):
        for a in range(2):
            matrices=[sum(table[0,a,s,f] for f in fi[draw]) for s in si[draw]]
            brute[a,draw]=np.mean([macro(matrix) for matrix in matrices]);wrong[a,draw]=macro(sum(matrices))
    max_error=max(max_error,float(np.abs(brute-f1[:,:7]).max()))
    blocked,_=subject.analyze(pred,iou,prob,truth,target,frames,config,False)
    goals=analyses['final_calibrated']['primary_goal']
    ci_error=max(error(quantiles(df),goals['CI97_5_F1']),error(quantiles(da),goals['CI97_5_AP']))
    return (dict(independent_heterogeneous_per_seed_F1_and_frame_AP10000_draws=(max_error<1e-14
        and np.array_equal(si,actual['seed_ids']) and np.array_equal(fi,actual['frame_ids']) and np.array_equal(values,actual['frame_values'])),
        brute_frame_seed_bootstrap_rejects_pooled_seed_estimator=float(np.abs(brute-wrong).max())>1e-3,
        independent_manual97_5_intervals_for_two_outcomes=ci_error<1e-14,
        primary_requires_both_AP_F1_and_eligibility_secondary_no_rescue=(
            not blocked['final_calibrated']['primary_goal']['confirmed'] and not analyses['fixed05']['primary_goal']['confirmed']
            and 'CI97_5_F1' not in analyses['fixed05']['primary_goal'] and 'CI97_5_AP' not in analyses['fixed05']['primary_goal'])),
        dict(maximum_bootstrap_error=max_error,maximum_CI_error=ci_error,wrong_pooled_seed_difference=float(np.abs(brute-wrong).max())))


def old_val_checks(rows,states):
    proof=read(ROOT/'OLD_VAL_PREFLIGHT.json')
    with np.load(ROOT/'OLD_VAL_PREFLIGHT.npz',allow_pickle=False) as z:saved={k:z[k] for k in z.files}
    with np.load(PILOT/'TRAIN_VAL.npz',allow_pickle=False) as z:
        ids=np.flatnonzero(z['role']=='val')[:17];x=5*(z['X'][ids]-.2);truth=z['label'][ids];target=z['Y'][ids,0].astype(bool)
    valid=(proof['status']=='PASS_OLD_VAL_PREFLIGHT' and all(proof['checks'].values())
        and proof['examples']==17 and proof['source_model_count']==9 and proof['fit_count']==proof['optimizer_updates']==0
        and saved['probability'].shape==(3,3,17,12,12) and saved['mask_predictions'].shape==(2,3,3,17)
        and saved['packed_masks'].shape==(2,3,3,17,18) and np.array_equal(saved['truth'],truth)
        and np.array_equal(saved['target'],target))
    probability_error=metric_error=0.;bits=0
    for row in rows:
        arm,seed=row['architecture'],row['seed'];a,s=ARMS.index(arm),SEEDS.index(seed);prob=saved['probability'][a,s]
        with torch.inference_mode():
            parts=[]
            for batch in torch.tensor(x).split(16):
                output=ref.reference_plain(batch,states[arm,seed]).sigmoid() if arm!='plain12_C4pooled' else ref.stable_orbit_logits(batch,states[arm,seed])[1].sigmoid().mean(0)
                parts.append(output[:,0].numpy())
        probability_error=max(probability_error,float(np.abs(np.concatenate(parts)-prob).max()))
        with np.load(resolve(row['source_VAL_file']),allow_pickle=False) as z:archive={k:z[k] for k in z.files}
        probability_error=max(probability_error,float(np.abs(archive['probability'][:17,0]-prob).max()))
        for m,mode in enumerate(MODES):
            threshold=row[mode+'_threshold'];mask=prob>=threshold;prediction=ref.classify(mask)
            packed=np.packbits(mask.reshape(17,144),axis=1);union=(mask|target).sum((1,2));iou=np.divide((mask&target).sum((1,2)),union,out=np.ones(17),where=union>0)
            index=int(np.flatnonzero(archive['thresholds']==threshold)[0]);bits+=int(np.count_nonzero((archive['probability'][:17,0]>=threshold)!=mask))
            valid &= (saved['thresholds'][m,a,s]==threshold and np.array_equal(prediction,saved['mask_predictions'][m,a,s])
                and np.array_equal(packed,saved['packed_masks'][m,a,s]) and np.array_equal(iou,saved['IoU'][m,a,s])
                and np.array_equal(prediction,archive['predictions'][index,:17]) and np.array_equal(packed,archive['packed_masks'][index,:17]))
    return dict(genuine_old_VAL9_states17_examples_both_locked_modes=bool(valid),
        independent_CPU_archived_probability_and_mask_class_parity=probability_error<=1e-5 and bits==0),dict(maximum_probability_error=probability_error,changed_bits=bits)


def audit_pre(lock_required):
    started=time.monotonic();c=subject_common();config=read(ROOT/'CONFIG.json');checks={}
    codes=c.code_hashes();assert all(sha(resolve(p))==h for p,h in codes.items())
    required={rel(ROOT/n) for n in ['common.py','selection.py','evaluate.py','prepare.py','stats.py']}
    checks['all_five_scientific_sources_and_imported_dependencies_hashed']=required<=codes.keys()
    checks['no_fresh_sources_arrays_predictions_before_download']=not any((ROOT/n).exists() for n in ['sources','DOWNLOAD.json','TEST.npz','EVAL.npz','PREDICTIONS.npz','RESULT.json'])
    checks['fixed_24x256_6144_frames10000_paired_draws_Bonferroni2_intervals']=(
        config['architectures']==ARMS and config['seeds']==SEEDS and config['frames']==24 and config['evaluation_per_frame']==256
        and config['evaluation_examples']==6144 and config['bootstrap_repeats']==10000 and config['bootstrap_seed']==202611034
        and config['CI_quantiles']==[.0125,.9875] and config['protocol_sha256']==sha(ROOT/'PROTOCOL.md'))
    calibration,data=independent_calibration()
    with np.load(ROOT/'FINAL_CALIBRATION.npz',allow_pickle=False) as z:saved={k:z[k] for k in z.files}
    cal_error=max(float(np.max(np.abs(v-saved[k]))) for k,v in calibration.items())
    document=read(ROOT/'FINAL_CALIBRATION.json')
    checks['independent512_class_labels32_frame_SHA_selection_and_all81_scores_exact']=(
        cal_error<1e-14 and document['calibration_examples']==512 and document['calibration_frames']==32
        and document['data_indices']==calibration['data_indices'].tolist() and document['frame_ids']==calibration['frame_ids'].tolist()
        and document['class_counts']==np.bincount(calibration['truth'],minlength=9).tolist())
    checks['final_calibration_source_hashes_current_and_no_new_images']=all(sha(resolve(p))==h for p,h in document['source_sha256'].items())
    rows,states=independent_models(calibration);development=c.development_selection()
    rowmap={(r['architecture'],r['seed']):r for r in development['rows']}
    checks['all9_final_states3_exact_group_exports_and_fixed_final_thresholds']=all(
        all(rowmap[row['architecture'],row['seed']][key]==value for key,value in row.items()) for row in rows)
    checks['development_eligibility_and_initial_function_same_compute_proof']=(development['development_eligible']
        and all(read(UNTIED/'PREFLIGHT.json')['checks'].values()) and all(read(UNTIED/'AUDIT.json')['checks'].values())
        and read(UNTIED/'PREFLIGHT.json')['optimizer_updates']==0)
    plan_checks,plan_proof=source_plan_checks(read(ROOT/'SOURCE_PLAN.json') if lock_required else None);checks.update(plan_checks)
    old_checks,old_proof=old_val_checks(rows,states);checks.update(old_checks)
    stat_checks,stat_proof=synthetic_statistics(config);checks.update(stat_checks)
    if lock_required:
        selection,lock=read(ROOT/'SELECTION.json'),read(ROOT/'MODEL_LOCK.json')
        checks['locked_before_fresh_download_models_final_calibration_and_code_exact']=(
            selection['status']=='LOCKED_BEFORE_FRESH_DOWNLOAD' and selection['rows']==lock['rows']==development['rows']
            and selection['model_lock_sha256']==sha(ROOT/'MODEL_LOCK.json') and selection['source_plan_sha256']==lock['source_plan_sha256']==sha(ROOT/'SOURCE_PLAN.json')
            and selection['code_sha256']==codes and selection['fit_count']==selection['optimizer_updates']==0
            and selection['source_model_count']==9 and selection['development_eligible'])
        checks['scientific_source_config_protocol_snapshots_exact']=all((ROOT/'source_snapshot'/n).read_bytes()==(ROOT/n).read_bytes()
            for n in ['common.py','selection.py','evaluate.py','prepare.py','stats.py','CONFIG.json','PROTOCOL.md'])
    inputs={**codes,rel(ROOT/'CONFIG.json'):sha(ROOT/'CONFIG.json'),rel(ROOT/'PROTOCOL.md'):sha(ROOT/'PROTOCOL.md')}
    for row in rows:
        for key in ['free_model_file','prediction_model_file','source_VAL_file']:inputs[row[key]]=sha(resolve(row[key]))
    for n in ['FINAL_CALIBRATION.json','FINAL_CALIBRATION.npz','OLD_VAL_PREFLIGHT.json','OLD_VAL_PREFLIGHT.npz','SELECTION.json','MODEL_LOCK.json','SOURCE_PLAN.json']:
        if (ROOT/n).exists():inputs[rel(ROOT/n)]=sha(ROOT/n)
    return dict(status=('PASS_' if all(checks.values()) else 'FAIL_')+('PRE_DOWNLOAD_AUDIT' if lock_required else 'PRE_LOCK_AUDIT'),
        checks=checks,checks_passed=sum(bool(v) for v in checks.values()),checks_total=len(checks),code_sha256=codes,input_sha256=inputs,
        final_calibration_maximum_error=cal_error,old_VAL_proof=old_proof,independent_source_plan=plan_proof,
        independent_bootstrap_fixture=stat_proof,new_target_access=False,fit_count=0,optimizer_updates=0,seconds=time.monotonic()-started)


def data_inputs(selection):
    c=subject_common();paths=[resolve(p) for p in c.code_hashes()]
    paths += [ROOT/n for n in ['SELECTION.json','MODEL_LOCK.json','SOURCE_PLAN.json','CONFIG.json','PROTOCOL.md',
        'FINAL_CALIBRATION.json','FINAL_CALIBRATION.npz','OLD_VAL_PREFLIGHT.json','OLD_VAL_PREFLIGHT.npz',
        'PRE_LOCK_AUDIT.json','PRE_DOWNLOAD_AUDIT.json','DOWNLOAD.json','DATA_MANIFEST.json','EVAL_INDEX.json',
        'PREPARE_CHECK.json','TEST.npz','EVAL.npz']]
    paths += [Path(__file__),MODEL/'audit.py',GROUP_PREVIOUS/'audit.py',SHARED/'labels.zip',SHARED/'sem_zip_listing.json',
        PREVIOUS/'SOURCE_PLAN.json',PREVIOUS/'SELECTION.json']
    paths += [folder/'DATA_MANIFEST.json' for folder in PRIOR]
    for row in selection['rows']:
        paths.extend(resolve(row[n]) for n in ['free_model_file','prediction_model_file','source_VAL_file'])
    paths += [resolve(p) for p in read(ROOT/'FINAL_CALIBRATION.json')['source_sha256']]
    paths += list((ROOT/'sources').glob('*'))
    return {rel(p):sha(p) for p in sorted(set(paths)) if p.is_file()}


def audit_data():
    started=time.monotonic();c=subject_common();selection=read(ROOT/'SELECTION.json');lock=read(ROOT/'MODEL_LOCK.json')
    plan=read(ROOT/'SOURCE_PLAN.json');manifest=read(ROOT/'DATA_MANIFEST.json');index=read(ROOT/'EVAL_INDEX.json')
    downloaded=read(ROOT/'DOWNLOAD.json');prepared=read(ROOT/'PREPARE_CHECK.json');pre=read(ROOT/'PRE_DOWNLOAD_AUDIT.json')
    inputs=data_inputs(selection);checks={}
    checks['no_prediction_before_independent_native_data_gate']=not any((ROOT/n).exists() for n in ['EVALUATION_FROZEN.json','PREDICTIONS.npz','BOOTSTRAP.npz','RESULT.json'])
    checks['pre_download_audit_and_scientific_calibration_model_locks_current']=(pre['status']=='PASS_PRE_DOWNLOAD_AUDIT'
        and all(pre['checks'].values()) and not pre['new_target_access'] and pre['code_sha256']==selection['code_sha256']==c.code_hashes()
        and selection['rows']==lock['rows'] and len(selection['rows'])==9
        and selection['model_lock_sha256']==sha(ROOT/'MODEL_LOCK.json') and selection['source_plan_sha256']==sha(ROOT/'SOURCE_PLAN.json')
        and all(sha(resolve(p))==h for p,h in pre['input_sha256'].items()))
    plan_checks,plan_proof=source_plan_checks(plan);checks.update(plan_checks)
    checks['download_prepare_all_bound_to_frozen_final_models_and_calibration']=(
        downloaded['status']=='COMPLETE' and len(downloaded['files'])==24 and downloaded['lock_verified_before_new_image_access']
        and manifest['lock_verified_before_new_image_access'] and manifest['no_models_loaded'] and manifest['no_TRAIN_VAL_or_combined_archive']
        and all(doc['selection_sha256']==sha(ROOT/'SELECTION.json') and doc['model_lock_sha256']==sha(ROOT/'MODEL_LOCK.json')
            and doc['source_plan_sha256']==sha(ROOT/'SOURCE_PLAN.json') for doc in [downloaded,manifest,index]))
    expected_files={f'sources/sem{f:04}.png' for f in plan['test_frames']}|{f'sources/label{f:04}.svg' for f in plan['test_frames']}
    source_ok=(set(manifest['sources_sha256'])==expected_files
        and {str(p.relative_to(ROOT)) for p in (ROOT/'sources').iterdir() if p.is_file()}==expected_files
        and all(sha(ROOT/p)==h for p,h in manifest['sources_sha256'].items()))
    listing={r['name']:r for r in read(SHARED/'sem_zip_listing.json')}
    with zipfile.ZipFile(SHARED/'labels.zip') as labels:
        for frame in plan['test_frames']:
            record=next(r for r in downloaded['files'] if r['frame']==frame);png=ROOT/f'sources/sem{frame:04}.png';raw=png.read_bytes()
            svg=(ROOT/f'sources/label{frame:04}.svg').read_bytes();svg_name=f'dataset/labels/m2/label{frame:04}.svg'
            source_ok &= (record['file']==str(png.relative_to(ROOT)) and record['bytes']==len(raw)==listing[f'dataset/sems/m2/sem{frame:04}.png']['size']<=10_000_000
                and sha(png)==record['sha256'] and zlib.crc32(raw)==record['crc32']
                and svg==labels.read(svg_name) and zlib.crc32(svg)==labels.getinfo(svg_name).CRC)
    checks['all24_PNG_CRC_SHA_sizes_and24_SVG_exact_public_archive']=bool(source_ok)
    with np.load(ROOT/'TEST.npz',allow_pickle=False) as z:test={k:z[k] for k in z.files}
    with np.load(ROOT/'EVAL.npz',allow_pickle=False) as z:evaluation={k:z[k] for k in z.files}
    records=manifest['records'];n=len(records)
    checks['TEST_only_new_frames_shapes_and_archive_hashes']=(manifest['status']=='PREPARED_NEW_ONLY_TEST'
        and manifest['TEST_sha256']==index['TEST_sha256']==sha(ROOT/'TEST.npz') and manifest['examples']==len(test['X'])==n
        and test['X'].shape==test['Y'].shape==(n,1,12,12) and np.all(test['role']=='test')
        and test['X'].dtype==np.float32 and test['Y'].dtype==np.uint8 and np.isfinite(test['X']).all()
        and test['X'].min()>=0 and test['X'].max()<=1 and set(np.unique(test['Y']))<={0,1}
        and set(test['frame'])==set(plan['test_frames']) and not set(test['frame'])&set(plan['excluded_previous_frames']))
    prior_mask={r['canonical_D4'] for folder in PRIOR for r in read(folder/'DATA_MANIFEST.json')['records']}
    prior_raw={r['raw_D4_sha256'] for folder in PRIOR for r in read(folder/'DATA_MANIFEST.json')['records']}
    checks['eight_prior_D4_union_and_hashes_exact']=(manifest['prior_manifests_sha256']==plan['prior_manifests_sha256']
        and manifest['prior_union_unique_counts']==dict(mask=len(prior_mask),raw=len(prior_raw)))
    used_mask,used_raw=prior_mask.copy(),prior_raw.copy();cursor=0;native_ok=metadata_ok=True;max_error=0.;reconstructed=[]
    for frame in plan['test_frames']:
        xs,ys,coords,polygons=native_ref.independent_native_patches(ROOT/f'sources/sem{frame:04}.png',ROOT/f'sources/label{frame:04}.svg')
        keys=native_ref.canonical_batch(ys);rawkeys=native_ref.canonical_batch(xs,True);kept=0
        for x,y,xy,key,rawkey in zip(xs,ys,coords,keys,rawkeys):
            if key in used_mask or rawkey in used_raw:continue
            used_mask.add(key);used_raw.add(rawkey);assert cursor<n;row=records[cursor]
            reduced=x.astype(np.float32)/(16*255)
            native_ok &= np.array_equal(reduced,test['X'][cursor,0]) and np.array_equal(y,test['Y'][cursor,0])
            max_error=max(max_error,float(np.abs(reduced-test['X'][cursor,0]).max()))
            metadata_ok &= (row['frame']==frame==int(test['frame'][cursor]) and row['y']==int(xy[0]) and row['x']==int(xy[1])
                and row['canonical_D4']==key and row['raw_D4_sha256']==rawkey and row['role']=='test' and row['label']==int(test['label'][cursor]))
            cursor+=1;kept+=1
        reconstructed.append(dict(frame=frame,SVG_polygons=polygons,candidate_patches=len(xs),excluded_duplicates=len(xs)-kept,kept=kept))
        metadata_ok &= manifest['deduplication'][str(frame)]=={k:reconstructed[-1][k] for k in ['candidate_patches','excluded_duplicates','kept']}
        print(json.dumps(dict(data_audit_frame=frame,completed=len(reconstructed),total=24,independent_retained=cursor)),flush=True)
    checks['all_native4x4_integer_sums_rasterized_SVG_majority_masks_exact']=bool(native_ok) and cursor==n and max_error==0.
    checks['all_stride6_coordinates_D4_exclusion_order_and_metadata_exact']=(bool(metadata_ok) and cursor==n
        and len({r['canonical_D4'] for r in records})==len({r['raw_D4_sha256'] for r in records})==n
        and not {r['canonical_D4'] for r in records}&prior_mask and not {r['raw_D4_sha256'] for r in records}&prior_raw)
    checks['all_TEST_graph_classes_recomputed_independently']=np.array_equal(ref.classify(test['Y'][:,0].astype(bool)),test['label'])
    chosen=[]
    for frame in plan['test_frames']:
        ids=[i for i,row in enumerate(records) if row['frame']==frame]
        ids.sort(key=lambda i:hashlib.sha256((EVAL_PREFIX+records[i]['raw_D4_sha256']).encode()).hexdigest());chosen.extend(ids[:256])
    ids=np.asarray(index['local_TEST_indices'],np.int64)
    checks['EVAL6144_raw_only_SHA_rank_exact_subset_records_and_arrays']=(ids.tolist()==sorted(chosen) and len(set(ids.tolist()))==len(ids)==6144
        and all(np.array_equal(evaluation[k],test[k][ids]) for k in test) and index['records']==[records[i] for i in ids]
        and index['EVAL_sha256']==sha(ROOT/'EVAL.npz') and index['selection_prefix']==EVAL_PREFIX
        and index['mask_not_used_in_ranking'] and index['label_blind_selection'] and index['replacement_on_gate_failure'] is False)
    counts=np.bincount(evaluation['label'],minlength=9);class_frames={str(k):sorted(set(evaluation['frame'][evaluation['label']==k].tolist())) for k in range(9)}
    frame_counts={str(f):int((evaluation['frame']==f).sum()) for f in plan['test_frames']}
    gates=dict(exactly_6144=len(ids)==6144,exactly_256_each_frame=all(v==256 for v in frame_counts.values()),
        min_8_each_class=bool(counts.min()>=8),min_3_frames_each_class=all(len(v)>=3 for v in class_frames.values()))
    checks['predeclared24x256_class8_frame3_coverage_no_replacement_gates']=(all(gates.values()) and index['gates']==gates
        and index['class_counts']==counts.tolist() and index['class_frames']==class_frames and index['frame_counts']==frame_counts)
    checks['all_producer_native_checks_pass_and_exact_crop_reconstruction']=(prepared['status']=='PASS_PREPARE_CHECK'
        and all(prepared['checks'].values()) and prepared['maximum_native_reconstruction_error']==0.)
    checks['all_sources_models_calibration_and_native_data_inputs_unchanged']=all(sha(resolve(p))==h for p,h in inputs.items())
    return dict(status='PASS_DATA_AUDIT' if all(checks.values()) else 'FAIL_DATA_AUDIT',checks=checks,
        checks_passed=sum(bool(v) for v in checks.values()),checks_total=len(checks),input_sha256=inputs,
        source_plan=plan_proof,independent_frames=reconstructed,maximum_native_reconstruction_error=max_error,
        no_model_inference=True,fit_count=0,optimizer_updates=0,seconds=time.monotonic()-started)


def expected_analyses(points,ious,ap,df,da,config,eligible):
    analyses={};ci_f1,ci_ap=quantiles(df),quantiles(da)
    for m,mode in enumerate(MODES):
        cells={}
        for a,arm in enumerate(ARMS):
            cells[arm]=dict(mean=dict(mask_F1=float(points[m,a].mean()),IoU=float(ious[m,a].mean()),mean_frame_AP=float(ap[a].mean())),
                per_seed=[dict(seed=seed,mask_F1=float(points[m,a,s]),IoU=float(ious[m,a,s].mean()),mean_frame_AP=float(ap[a,s].mean())) for s,seed in enumerate(SEEDS)])
        contrasts={}
        for key,reference in [('group_minus_untied',1),('group_minus_narrow',2)]:
            contrasts[key]=dict(mean={metric:cells['group6']['mean'][metric]-cells[ARMS[reference]]['mean'][metric]
                    for metric in ['mask_F1','IoU','mean_frame_AP']},
                per_seed=[dict(seed=seed,mask_F1=float(points[m,0,s]-points[m,reference,s]),
                    IoU=float(ious[m,0,s].mean()-ious[m,reference,s].mean()),mean_frame_AP=float(ap[0,s].mean()-ap[reference,s].mean())) for s,seed in enumerate(SEEDS)])
        effect=contrasts['group_minus_untied'];group,reference=cells['group6']['mean'],cells['untied24_single']['mean']
        goal=dict(group_F1=group['mask_F1'],reference_F1=reference['mask_F1'],group_AP=group['mean_frame_AP'],reference_AP=reference['mean_frame_AP'],
            group_IoU=group['IoU'],reference_IoU=reference['IoU'],F1_delta=effect['mean']['mask_F1'],AP_delta=effect['mean']['mean_frame_AP'],
            IoU_delta=effect['mean']['IoU'],per_seed_F1_deltas=[v['mask_F1'] for v in effect['per_seed']],development_eligible=bool(eligible),
            confirmed=False,descriptive_only=mode!='final_calibrated')
        if m==0:
            limits=config['primary_goal'];guards=dict(mean_F1=goal['F1_delta']>=limits['minimum_mean_F1_delta'],
                F1_CI_lower=ci_f1[0]>limits['CI_lower_strictly_above'],mean_AP=goal['AP_delta']>limits['minimum_mean_AP_delta'],
                AP_CI_lower=ci_ap[0]>limits['CI_lower_strictly_above'],seed_F1=min(goal['per_seed_F1_deltas'])>=limits['minimum_seed_F1_delta'],
                IoU_guard=goal['IoU_delta']>=limits['minimum_mean_IoU_delta'],matched_MACs=abs(1527552/1527552-1)<=limits['maximum_MAC_relative_difference'],
                development_eligible=bool(eligible))
            guards={k:bool(v) for k,v in guards.items()};goal.update(CI97_5_F1=ci_f1,CI97_5_AP=ci_ap,guards=guards,passed=all(guards.values()),confirmed=all(guards.values()))
        analyses[mode]=dict(cells=cells,contrasts=contrasts,primary_goal=goal,other_comparisons_descriptive_only=True,secondary_cannot_replace_primary=mode!='final_calibrated')
    return analyses


def audit_result():
    started=time.monotonic();config=read(ROOT/'CONFIG.json');selection=read(ROOT/'SELECTION.json');lock=read(ROOT/'MODEL_LOCK.json')
    result=read(ROOT/'RESULT.json');data_audit=read(ROOT/'DATA_AUDIT.json');frozen=read(ROOT/'EVALUATION_FROZEN.json')
    inputs={**data_audit['input_sha256'],**{rel(ROOT/n):sha(ROOT/n) for n in ['RESULT.json','DATA_AUDIT.json','EVALUATION_FROZEN.json','PREDICTIONS.npz','BOOTSTRAP.npz']}}
    checks={}
    checks['independent_native_data_audit_bytes_and_all_inputs_unchanged']=(data_audit['status']=='PASS_DATA_AUDIT' and all(data_audit['checks'].values())
        and all(sha(resolve(p))==h for p,h in data_audit['input_sha256'].items())
        and frozen['input_sha256'][rel(ROOT/'DATA_AUDIT.json')]==sha(ROOT/'DATA_AUDIT.json'))
    c=subject_common()
    checks['nine_model_final_calibration_source_and_data_locks_still_current']=(selection['rows']==lock['rows'] and len(selection['rows'])==9
        and selection['code_sha256']==c.code_hashes()==frozen['code_sha256'] and selection['model_lock_sha256']==sha(ROOT/'MODEL_LOCK.json')
        and all(sha(resolve(p))==h for p,h in frozen['input_sha256'].items()) and frozen['fit_count']==frozen['optimizer_updates']==0)
    with np.load(ROOT/'EVAL.npz',allow_pickle=False) as z:evaluation={k:z[k] for k in z.files}
    with np.load(ROOT/'PREDICTIONS.npz',allow_pickle=False) as z:saved={k:z[k] for k in z.files}
    with np.load(ROOT/'BOOTSTRAP.npz',allow_pickle=False) as z:bootstrap={k:z[k] for k in z.files}
    truth,target,frames=evaluation['label'],evaluation['Y'][:,0].astype(bool),evaluation['frame'];n=len(truth)
    checks['fresh6144_examples24_frames_zero_fits_and_array_hashes']=(n==6144 and len(np.unique(frames))==24
        and result['test_examples']==n and result['source_model_count']==result['evaluated_routes']==9
        and result['fit_count']==result['optimizer_updates']==0 and result['predictions_sha256']==sha(ROOT/'PREDICTIONS.npz')
        and result['bootstrap_sha256']==sha(ROOT/'BOOTSTRAP.npz') and result['selection_sha256']==sha(ROOT/'SELECTION.json'))
    valid=(saved['probability'].shape==(3,3,n,12,12) and saved['mask_predictions'].shape==saved['IoU'].shape==(2,3,3,n)
        and saved['packed_masks'].shape==(2,3,3,n,18) and saved['thresholds'].shape==(2,3,3)
        and np.array_equal(saved['truth'],truth) and np.array_equal(saved['target'],target) and np.array_equal(saved['frames'],frames)
        and np.isfinite(saved['probability']).all() and saved['probability'].min()>=0 and saved['probability'].max()<=1)
    for arrays in [saved,bootstrap]:
        valid &= arrays['modes'].tolist()==MODES and arrays['architectures'].tolist()==ARMS and arrays['seeds'].tolist()==SEEDS
    checks['all_prediction_and_bootstrap_axes_targets_probability_range_exact']=bool(valid)
    predictions=np.empty((2,3,3,n),np.int64);ious=np.empty_like(predictions,dtype=float)
    bits=classes=thresholds=True;metric_error=iou_error=0.
    for row in selection['rows']:
        a,s=ARMS.index(row['architecture']),SEEDS.index(row['seed']);p=saved['probability'][a,s]
        record=next(v for v in result['rows'] if v['architecture']==row['architecture'] and v['seed']==row['seed'])
        for m,mode in enumerate(MODES):
            tau=row[mode+'_threshold'];mask=p>=tau;pred=ref.classify(mask);predictions[m,a,s]=pred
            union=(mask|target).sum((1,2));ious[m,a,s]=np.divide((mask&target).sum((1,2)),union,out=np.ones(n),where=union>0)
            bits &= np.array_equal(np.packbits(mask.reshape(n,144),axis=1),saved['packed_masks'][m,a,s])
            classes &= np.array_equal(pred,saved['mask_predictions'][m,a,s]);thresholds &= saved['thresholds'][m,a,s]==tau
            iou_error=max(iou_error,float(np.abs(ious[m,a,s]-saved['IoU'][m,a,s]).max()))
            metric_error=max(metric_error,error(dict(threshold=tau,mask_F1=float(macro(cm(truth,pred))),IoU=float(ious[m,a,s].mean())),record['metrics'][mode]))
        print(json.dumps(dict(final_audit_architecture=row['architecture'],seed=row['seed'],both_modes=True)),flush=True)
    checks['all18_locked_threshold_masks_packed_bits_graph_classes_and_metrics_exact']=(bool(bits and classes and thresholds) and metric_error<1e-14 and iou_error==0.)
    table,values=frame_tables(predictions,truth,frames);ap=frame_AP(saved['probability'],target,frames)
    pooled=np.array([[rank_ap(target,saved['probability'][a,s]) for s in range(3)] for a in range(3)])
    ap_error=max(float(np.abs(ap-bootstrap['per_frame_AP']).max()),float(np.abs(pooled-bootstrap['pooled_pixel_AP']).max()))
    checks['all216_frame_AP_and9_pooled_AP_independent_tie_rank_formula']=ap_error<1e-14
    checks['all_frame_confusions_point_F1_AP_axes_and_prevalence_exact']=(np.array_equal(table,bootstrap['confusion'])
        and np.array_equal(values,bootstrap['frame_values']) and np.max(np.abs(macro(table.sum(3))-bootstrap['point_per_seed']))<1e-14
        and np.array_equal(np.array([target[frames==f].mean() for f in values]),bootstrap['frame_metal_prevalence']))
    draws=np.random.default_rng(config['bootstrap_seed']);si=draws.integers(0,3,(10000,3));fi=draws.integers(0,24,(10000,24))
    bf,ba,df,da=weighted_bootstrap(table,ap,si,fi)
    checks['both_outcomes_share_exact10000_seed_and_frame_indices']=(np.array_equal(si,bootstrap['seed_ids']) and np.array_equal(fi,bootstrap['frame_ids'])
        and config['bootstrap_seed']==result['bootstrap_seed']==202611034 and result['bootstrap_repeats']==10000)
    boot_error=max(float(np.abs(bf-bootstrap['bootstrap_F1']).max()),float(np.abs(ba-bootstrap['bootstrap_AP']).max()),
        float(np.abs(df-bootstrap['delta_F1']).max()),float(np.abs(da-bootstrap['delta_AP']).max()))
    checks['all10000_independent_weighted_crossed_AP_and_per_seed_F1_bootstraps']=boot_error<1e-14
    expected=expected_analyses(macro(table.sum(3)),ious,ap,df,da,config,selection['development_eligible'])
    analysis_error=error(expected,result['analyses'])
    checks['all_cells_seed_contrasts_joint_guards_manual97_5_intervals_exact']=analysis_error<1e-14
    confirmed=expected['final_calibrated']['primary_goal']['confirmed']
    checks['compound_primary_both_AP_F1_and_no_fixed05_narrow_rescue']=(result['confirmed']==confirmed
        and result['status']==('PASS_CONFIRMATION' if confirmed else 'FAIL_CONFIRMATION')
        and result['primary_mode']=='final_calibrated' and result['secondary_mode']=='fixed05'
        and result['CI_quantiles']==config['CI_quantiles']==bootstrap['CI_quantiles'].tolist()==[.0125,.9875]
        and not expected['fixed05']['primary_goal']['confirmed'] and 'CI97_5_F1' not in expected['fixed05']['primary_goal'])
    checks['all_models_calibration_data_code_and_audit_hashes_unchanged']=all(sha(resolve(p))==h for p,h in inputs.items())
    return dict(status='PASS_AUDIT' if all(checks.values()) else 'FAIL_AUDIT',checks=checks,checks_passed=sum(bool(v) for v in checks.values()),
        checks_total=len(checks),input_sha256=inputs,confirmed=bool(confirmed),independent_analyses=expected,
        maximum_errors=dict(metrics=metric_error,IoU=iou_error,AP=ap_error,bootstrap=boot_error,analyses=analysis_error),
        no_model_inference_rerun=True,fit_count=0,optimizer_updates=0,seconds=time.monotonic()-started)


def main():
    parser=argparse.ArgumentParser();actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--pre-lock',action='store_true');actions.add_argument('--pre-download',action='store_true')
    actions.add_argument('--data',action='store_true');actions.add_argument('--result',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        result=audit_result() if args.result else (audit_data() if args.data else audit_pre(args.pre_download))
    name='AUDIT.json' if args.result else ('DATA_AUDIT.json' if args.data else ('PRE_DOWNLOAD_AUDIT.json' if args.pre_download else 'PRE_LOCK_AUDIT.json'))
    snapshot_and_save(name,result)
    print(json.dumps(dict(status=result['status'],checks_passed=result['checks_passed'],checks_total=result['checks_total'],
        failed=[k for k,v in result['checks'].items() if not v],seconds=result['seconds'])),flush=True)
    assert result['status'].startswith('PASS')


def snapshot_and_save(name,result):
    source=sha(__file__);directory=ROOT/'audit_sources';directory.mkdir(exist_ok=True)
    target=directory/f'audit_{source}.py'
    if target.exists():assert target.read_bytes()==Path(__file__).read_bytes()
    else:shutil.copy2(__file__,target)
    result['audit_source_sha256']=source
    def scalar(x):
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    path=ROOT/name;encoded=json.dumps(result,ensure_ascii=False,indent=2,default=scalar)+'\n'
    if path.exists() and path.read_text()!=encoded:shutil.copy2(path,ROOT/f'{path.stem}_PREVIOUS_{sha(path)[:12]}.json')
    path.write_text(encoded)


if __name__=='__main__':main()
