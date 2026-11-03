"""Independent pre-lock, native-data, and saved-prediction confirmation audit.

No downloads, optimizer steps, model fitting, threshold fitting, or fresh-model
inference are performed. Old-VAL technical inference is allowed before lock.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
G, REPO = ROOT.parent, ROOT.parents[1]
DEV = G/'sem_arch_groupcnn_20261102'
FACTORIAL = G/'sem_arch_loss_factorial_20261103'
PREVIOUS = G/'sem_arch_groupcnn_confirmation_20261102'
OLD = G/'sem_arch_orbit_confirmation_20261102'
PILOT = G/'sem_architecture_pilot_20261102'
ORIGINAL = G/'sem_confirmation_20261028'
SHARED = G/'shared_data/ic_sem_m2'
PRIOR = [ORIGINAL,G/'sem_cldice_confirmation_20261030',G/'sem_cldice_expansion_20261102',
         G/'sem_scarcity_20261102',PILOT,OLD,PREVIOUS]
CELLS = ['BCE_group','BCE_plain','aux_group','aux_plain']
BUDGETS, SEEDS, MODES = [18,27], [5701,5721,5757], ['primary_fine','fixed05']
FRAME_PREFIX, EVAL_PREFIX = 'SEM-BCE-CNN-CONFIRM-20261103/','SEM-BCE-CNN-EVAL-20261103/'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(path): return str(Path(path).resolve().relative_to(REPO))
def resolve(path): return Path(path) if Path(path).is_absolute() else REPO/path
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def load(path): return torch.load(path,map_location='cpu',weights_only=True)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


ref = module('bce_independent_model_graph_reference',DEV/'audit.py')
data_ref = module('bce_independent_native_reference',PREVIOUS/'audit.py')


def compare(expected,actual):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(expected)!=set(actual):return float('inf')
        return max([compare(v,actual[k]) for k,v in expected.items()]+[0.])
    if isinstance(expected,(list,tuple)):
        if len(expected)!=len(actual):return float('inf')
        return max([compare(x,y) for x,y in zip(expected,actual)]+[0.])
    if isinstance(expected,(str,bool)) or expected is None:return 0. if expected==actual else float('inf')
    return abs(float(expected)-float(actual))


def macro(confusion):
    confusion=np.asarray(confusion,dtype=np.float64)
    den=confusion.sum(-1)+confusion.sum(-2)
    return np.divide(2*np.diagonal(confusion,axis1=-2,axis2=-1),den,
                     out=np.zeros_like(den),where=den>0).mean(-1)


def linear_quantile(values,quantiles):
    ordered=np.sort(np.asarray(values,dtype=np.float64));result=[]
    for q in quantiles:
        index=(len(ordered)-1)*q;lower=int(np.floor(index));upper=int(np.ceil(index))
        result.append(float(ordered[lower]+(ordered[upper]-ordered[lower])*(index-lower)))
    return result


def independent_confusion(predictions,truth,frames):
    values=np.unique(frames);result=np.zeros((2,4,2,3,len(values),9,9),np.int64)
    for f,frame in enumerate(values):
        subset=frames==frame
        for m in range(2):
            for cell in range(4):
                for b in range(2):
                    for s in range(3):
                        result[m,cell,b,s,f]=np.bincount(9*truth[subset]+predictions[m,cell,b,s,subset],minlength=81).reshape(9,9)
    return result,values


def independent_bootstrap(confusion,seed_ids,frame_ids):
    repeats=len(seed_ids);frames=confusion.shape[4]
    flat=confusion[0,:2].transpose(3,0,1,2,4,5).reshape(frames,-1)
    scores=np.empty((2,2,repeats),np.float64)
    for start in range(0,repeats,211):
        end=min(repeats,start+211);count=end-start
        fw=np.zeros((count,frames),np.int64);sw=np.zeros((count,3),np.int64)
        for i in range(count):
            fw[i]=np.bincount(frame_ids[start+i],minlength=frames)
            sw[i]=np.bincount(seed_ids[start+i],minlength=3)
        sampled=(fw@flat).reshape(count,2,2,3,9,9)
        scores[:,:,start:end]=((macro(sampled)*sw[:,None,None,:]).sum(-1)/3).transpose(1,2,0)
    averaged=scores.mean(1)
    return scores,averaged,averaged[0]-averaged[1]


def expected_analyses(points,ious,config,delta,eligible):
    result={};ci=linear_quantile(delta,[.025,.975])
    for mode_id,mode in enumerate(MODES):
        cells={};f1=points[mode_id];iou=ious[mode_id].mean(-1)
        for r,cell in enumerate(CELLS):
            cells[cell]=dict(mean=dict(mask_F1=float(f1[r].mean()),IoU=float(iou[r].mean())),
                by_budget={str(b):dict(mask_F1=float(f1[r,j].mean()),IoU=float(iou[r,j].mean())) for j,b in enumerate(BUDGETS)},
                by_seed={str(s):dict(mask_F1=float(f1[r,:,j].mean()),IoU=float(iou[r,:,j].mean())) for j,s in enumerate(SEEDS)})
        contrasts={}
        for name,left,right in [('architecture_BCE',0,1),('architecture_auxiliary',2,3),('auxiliary_group',2,0),('auxiliary_plain',3,1)]:
            df,di=f1[left]-f1[right],iou[left]-iou[right]
            contrasts[name]=dict(mean=dict(mask_F1=float(df.mean()),IoU=float(di.mean())),
                by_budget={str(b):dict(mask_F1=float(df[j].mean()),IoU=float(di[j].mean())) for j,b in enumerate(BUDGETS)},
                by_seed={str(s):dict(mask_F1=float(df[:,j].mean()),IoU=float(di[:,j].mean())) for j,s in enumerate(SEEDS)},
                per_budget_seed=[dict(budget=b,seed=s,mask_F1=float(df[j,k]),IoU=float(di[j,k]))
                    for j,b in enumerate(BUDGETS) for k,s in enumerate(SEEDS)])
        contrasts['interaction']=dict(mean={metric:contrasts['architecture_auxiliary']['mean'][metric]-contrasts['architecture_BCE']['mean'][metric]
            for metric in ['mask_F1','IoU']},definition='(Group-Plain)_aux - (Group-Plain)_BCE',descriptive_only=True)
        effect=contrasts['architecture_BCE']
        goal=dict(group_F1=cells['BCE_group']['mean']['mask_F1'],reference_F1=cells['BCE_plain']['mean']['mask_F1'],
            mean_F1_delta=effect['mean']['mask_F1'],delta=effect['mean']['mask_F1'],
            budget_deltas=[effect['by_budget'][str(b)]['mask_F1'] for b in BUDGETS],
            per_seed_deltas=[effect['by_seed'][str(s)]['mask_F1'] for s in SEEDS],IoU_delta=effect['mean']['IoU'],
            group_IoU=cells['BCE_group']['mean']['IoU'],reference_IoU=cells['BCE_plain']['mean']['IoU'],
            development_eligible=bool(eligible),confirmed=False,descriptive_only=mode_id!=0)
        if mode_id==0:
            limits=config['primary_goal'];guards=dict(mean_F1=goal['delta']>=limits['minimum_mean_F1_delta'],
                CI_lower=ci[0]>limits['CI_lower_strictly_above'],budget_guard=min(goal['budget_deltas'])>=limits['minimum_budget_F1_delta'],
                IoU_guard=goal['IoU_delta']>=limits['minimum_mean_IoU_delta'],
                parameter_match=abs(2671/2749-1)<=limits['maximum_budget_ratio_deviation'],
                MAC_match=abs(1527552/1562112-1)<=limits['maximum_budget_ratio_deviation'],development_eligible=bool(eligible))
            guards={k:bool(v) for k,v in guards.items()};goal.update(CI95=ci,guards=guards,passed=all(guards.values()),confirmed=all(guards.values()))
        result[mode]=dict(cells=cells,contrasts=contrasts,primary_goal=goal,secondary_contrasts_descriptive_only=True,
                          secondary_cannot_replace_primary=mode_id!=0)
    return result


def synthetic_statistics(config):
    subject=module('bce_confirmation_statistical_subject',ROOT/'stats.py')
    rng=np.random.default_rng(6037);frames=np.repeat(np.arange(24),31);truth=rng.integers(0,9,len(frames))
    predictions=np.empty((2,4,2,3,len(truth)),np.int64)
    for m in range(2):
        for r in range(4):
            for b in range(2):
                for s in range(3):
                    candidate=truth.copy();change=rng.random(len(truth))<(.025+.21*s+.035*r+.01*b+.015*m)
                    candidate[change]=(candidate[change]+rng.integers(1,9,change.sum()))%9
                    predictions[m,r,b,s]=candidate
    ious=rng.uniform(.6,.9,predictions.shape)
    analysis,actual=subject.analyze(predictions,ious,truth,frames,config,True)
    confusion,values=independent_confusion(predictions,truth,frames)
    draws=np.random.default_rng(config['bootstrap_seed']);seed_ids=draws.integers(0,3,(10000,3));frame_ids=draws.integers(0,24,(10000,24))
    budgets,scores,delta=independent_bootstrap(confusion,seed_ids,frame_ids)
    brute=np.empty((2,2,7));incorrect=np.empty_like(brute)
    for draw in range(7):
        for cell in range(2):
            for b in range(2):
                tables=[sum(confusion[0,cell,b,s,int(f)] for f in frame_ids[draw]) for s in seed_ids[draw]]
                brute[cell,b,draw]=np.mean([macro(cm) for cm in tables]);incorrect[cell,b,draw]=macro(sum(tables))
    points=macro(confusion.sum(4));expected=expected_analyses(points,ious,config,delta,True)
    ineligible,other=subject.analyze(predictions,ious,truth,frames,config,False)
    no_eligibility=expected_analyses(points,ious,config,delta,False)
    error=max(float(np.abs(budgets-actual['primary_budget_scores']).max()),float(np.abs(scores-actual['primary_scores']).max()),
        float(np.abs(delta-actual['delta_F1']).max()),float(np.abs(brute-budgets[:,:,:7]).max()),compare(expected,analysis),compare(no_eligibility,ineligible))
    witness=float(np.abs(brute-incorrect).max())
    checks=dict(independent_heterogeneous_frame_confusions_and_all10000_crossed_draws=(
        np.array_equal(confusion,actual['confusion']) and np.array_equal(values,actual['frame_values'])
        and np.array_equal(seed_ids,actual['seed_ids']) and np.array_equal(frame_ids,actual['frame_ids'])),
        independent_weighted_bootstrap_brute_per_seed_F1_and_manual95CI=error<1e-14,
        wrong_pooled_seed_estimator_sensitive_fixture=witness>1e-3,
        single_primary_eligibility_and_secondary_no_CI_no_rescue=(
            'CI95' not in analysis['fixed05']['primary_goal'] and not analysis['fixed05']['primary_goal']['confirmed']
            and not ineligible['primary_fine']['primary_goal']['confirmed']
            and all('CI95' not in v and 'pvalue' not in v for v in analysis['primary_fine']['contrasts'].values())))
    return checks,dict(maximum_error=error,incorrect_pooled_seed_difference=witness,repeats=10000,
                       brute_draws=7,seed_ids=seed_ids[:7].tolist(),frame_ids=frame_ids[:7].tolist())


def expected_code_hashes():
    paths=[ROOT/name for name in ['common.py','selection.py','evaluate.py','prepare.py','stats.py']]
    paths += [DEV/'model.py',PILOT/'prepare.py',ORIGINAL/'prepare.py',OLD/'prepare.py',
              PREVIOUS/'prepare.py',G/'retained_evidence_20261027/SELECTED_SOURCES.zip',SHARED/'sem_remote_zip.py']
    return {rel(path):sha(path) for path in paths}


def development():
    result=read(FACTORIAL/'RESULT.json');audit=read(FACTORIAL/'AUDIT.json');checks={}
    checks['factorial_audited_12_BCE_12_aux_and_all_audited_inputs_current']=(
        result['status']=='COMPLETE_FACTORIAL_DEVELOPMENT' and result['new_fits']==result['inherited_fits']==12
        and len(result['rows'])==24 and audit['status']=='PASS_FACTORIAL_AUDIT'
        and audit['checks_passed']==audit['checks_total']==31 and all(audit['checks'].values())
        and audit['input_sha256'][rel(FACTORIAL/'RESULT.json')]==sha(FACTORIAL/'RESULT.json')
        and all(sha(resolve(path))==value for path,value in audit['input_sha256'].items()))
    rows=[];states={};seen=set();valid=True
    for row in result['rows']:
        group=row['architecture']=='group6';objective=row['objective']
        assert row['architecture'] in ['group6','plain12_C4pooled'] and objective in ['BCE','BCE_geometry']
        cell=('BCE_' if objective=='BCE' else 'aux_')+('group' if group else 'plain')
        unit=cell,row['budget'],row['seed'];assert unit not in seen;seen.add(unit)
        free=resolve(row['model_file']);prob=resolve(row['probability_file'])
        deployed=free.with_name(free.stem+'_deployed.pt') if group else free
        state=load(free)['model'];prediction=load(deployed)
        valid &= (sha(free)==row['model_sha256'] and sha(prob)==row['probability_sha256']
            and sum(v.numel() for v in state.values())==(2671 if group else 2749)
            and all(torch.isfinite(v).all().item() for v in state.values())
            and row['selected']==max(row['all_thresholds'],key=ref.selection_rank)
            and [v['threshold'] for v in row['all_thresholds']]==[round(.1+.01*i,2) for i in range(81)])
        if group:
            independent=ref.reference_deploy_state(state)
            valid &= (prediction['free_model_sha256']==sha(free) and prediction['free_parameters']==2671
                and prediction['materialized_coefficients']==10681 and set(independent)==set(prediction['model'])
                and all(torch.equal(v,prediction['model'][k]) for k,v in independent.items()))
        rows.append(dict(cell=cell,route=row['route'],architecture=row['architecture'],objective=objective,
            budget=row['budget'],seed=row['seed'],free_model_file=rel(free),free_model_sha256=sha(free),
            prediction_model_file=rel(deployed),prediction_model_sha256=sha(deployed),
            prediction_width=24 if group else 12,prediction_mode='single' if group else 'C4',
            free_parameters=2671 if group else 2749,stored_prediction_coefficients=10681 if group else 2749,
            primary_fine_threshold=row['selected']['threshold'],fixed05_threshold=.5,
            source_VAL_file=rel(prob),source_VAL_sha256=sha(prob)))
        states[unit]=prediction['model']
    rows.sort(key=lambda row:(CELLS.index(row['cell']),BUDGETS.index(row['budget']),SEEDS.index(row['seed'])))
    checks['all24_final_states12_exact_group_exports_and_VAL81_thresholds']=(bool(valid)
        and seen=={(c,b,s) for c in CELLS for b in BUDGETS for s in SEEDS})
    checks['primary_BCE_development_gate_exact_all3_seeds']=(result['primary_development_gate']['passed']
        and all(result['primary_development_gate']['guards'].values()))
    return checks,dict(rows=rows,development_eligible=True,result_file=rel(FACTORIAL/'RESULT.json'),
        result_sha256=sha(FACTORIAL/'RESULT.json'),audit_file=rel(FACTORIAL/'AUDIT.json'),
        audit_sha256=sha(FACTORIAL/'AUDIT.json'),prior_lock_file=rel(FACTORIAL/'LOCK.json'),
        prior_lock_sha256=sha(FACTORIAL/'LOCK.json'),source_model_count=24,deployment_artifact_count=12,
        interpretation='All fixed factorial cells; no selection of best seed or fresh target score'),states


def source_plan_check(plan=None):
    previous=read(PREVIOUS/'SOURCE_PLAN.json');previous_selection=read(PREVIOUS/'SELECTION.json')
    older=read(OLD/'SOURCE_PLAN.json');excluded=set(previous['excluded_previous_frames'])|set(previous['test_frames'])
    hashes={rel(folder/'DATA_MANIFEST.json')[len('gipoteza/'):]:sha(folder/'DATA_MANIFEST.json') for folder in PRIOR}
    prior_frames={int(row['frame']) for folder in PRIOR for row in read(folder/'DATA_MANIFEST.json')['records']}
    available=[int(m.group(1)) for row in read(SHARED/'sem_zip_listing.json')
               if (m:=re.fullmatch(r'dataset/sems/m2/sem(\d+)\.png',row['name']))]
    chosen=sorted(set(available)-excluded,key=lambda n:hashlib.sha256((FRAME_PREFIX+str(n)).encode()).hexdigest())[:24]
    checks=dict(independent110_frame_exclusion_chain_and_all7_manifest_frames=(
        len(previous['excluded_previous_frames'])==94 and len(previous['test_frames'])==16 and len(excluded)==110
        and set(previous['excluded_previous_frames'])==set(older['excluded_previous_frames'])|set(older['test_frames'])
        and len(older['excluded_previous_frames'])==86 and len(older['test_frames'])==8
        and previous_selection['source_plan_sha256']==sha(PREVIOUS/'SOURCE_PLAN.json')
        and previous['exclusion_source_sha256']==sha(OLD/'SOURCE_PLAN.json') and prior_frames<=excluded),
        independent321_frame_listing_fixed24_SHA_rank=(len(available)==len(set(available))==321 and len(chosen)==24
            and not set(chosen)&excluded))
    if plan is not None:
        checks['source_plan_exact24_seven_manifests_raw_only_256_rank_and_coverage']=(
            plan['test_frames']==chosen and plan['excluded_previous_frames']==sorted(excluded)
            and plan['excluded_frame_count']==110 and plan['prior_manifests_sha256']==hashes
            and plan['archive_listing_sha256']==sha(SHARED/'sem_zip_listing.json')
            and plan['selection_prefix']==FRAME_PREFIX and plan['evaluation']['selection_prefix']==EVAL_PREFIX
            and plan['evaluation']['patches_per_frame']==256 and plan['evaluation']['target_examples']==6144
            and plan['evaluation']['minimum_examples_each_class']==8 and plan['evaluation']['minimum_frames_each_class']==3
            and plan['evaluation']['replacement_on_gate_failure'] is False and plan['prepare_script_sha256']==sha(ROOT/'prepare.py'))
    return checks,dict(frames=chosen,excluded_frames=len(excluded),prior_manifest_count=7,
                       prior_record_frame_count=len(prior_frames),prior_manifests_sha256=hashes)


def old_val_proof(expected,states,config):
    proof=read(ROOT/'OLD_VAL_PREFLIGHT.json')
    with np.load(ROOT/'OLD_VAL_PREFLIGHT.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    with np.load(PILOT/'TRAIN_VAL.npz',allow_pickle=False) as z:
        ids=np.flatnonzero(z['role']=='val')[:17];x=5*(z['X'][ids]-.2);truth=z['label'][ids];target=z['Y'][ids,0].astype(bool)
    checks=dict(genuine_old_VAL24_states17_examples_both_modes_nonmultiple_batch=(
        proof['status']=='PASS_OLD_VAL_PREFLIGHT' and all(proof['checks'].values()) and proof['source_model_count']==24
        and proof['examples']==17 and proof['inference_batch_size']==16 and proof['device'] in ['cpu','mps']
        and proof['new_target_access'] is False and proof['fit_count']==proof['optimizer_updates']==0
        and proof['TRAIN_VAL_sha256']==sha(PILOT/'TRAIN_VAL.npz') and proof['code_sha256']==expected_code_hashes()
        and proof['development_selection_digest']==digest(expected) and proof['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
        and proof['config_sha256']==sha(ROOT/'CONFIG.json') and proof['arrays_sha256']==sha(ROOT/'OLD_VAL_PREFLIGHT.npz')))
    valid=(arrays['probability'].shape==(4,2,3,17,12,12) and arrays['mask_predictions'].shape==(2,4,2,3,17)
        and arrays['packed_masks'].shape==(2,4,2,3,17,18) and arrays['IoU'].shape==(2,4,2,3,17)
        and arrays['thresholds'].shape==(2,4,2,3) and len(proof['rows'])==len(proof['agreements'])==24
        and np.array_equal(arrays['truth'],truth) and np.array_equal(arrays['target'],target)
        and np.array_equal(arrays['OLD_VAL_dataset_indices'],ids) and np.array_equal(arrays['normalized_fixture_x'],x)
        and arrays['cells'].tolist()==CELLS and arrays['modes'].tolist()==MODES)
    error=dict(CPU_probability=0.,archived_probability=0.,metrics=0.,source_mask_bit_differences=0)
    for row in expected['rows']:
        unit=row['cell'],row['budget'],row['seed'];r,b,s=CELLS.index(unit[0]),BUDGETS.index(unit[1]),SEEDS.index(unit[2])
        p=arrays['probability'][r,b,s];parts=[]
        with torch.inference_mode():
            for batch in torch.tensor(x).split(16):
                output=ref.reference_plain(batch,states[unit]).sigmoid() if row['prediction_mode']=='single' else ref.stable_orbit_logits(batch,states[unit])[1].sigmoid().mean(0)
                parts.append(output[:,0].numpy())
        error['CPU_probability']=max(error['CPU_probability'],float(np.abs(np.concatenate(parts)-p).max()))
        with np.load(resolve(row['source_VAL_file']),allow_pickle=False) as z:archive={k:z[k] for k in z.files}
        error['archived_probability']=max(error['archived_probability'],float(np.abs(archive['probability'][:17,0]-p).max()))
        metric=next(v for v in proof['rows'] if (v['cell'],v['budget'],v['seed'])==unit)
        for m,mode in enumerate(MODES):
            threshold=row[mode+'_threshold'];mask=p>=threshold;pred=ref.classify(mask)
            packed=np.packbits(mask.reshape(17,144),axis=1);union=(mask|target).sum((1,2))
            iou=np.divide((mask&target).sum((1,2)),union,out=np.ones(17),where=union>0)
            grid_index=archive['thresholds'].tolist().index(threshold)
            valid &= (arrays['thresholds'][m,r,b,s]==threshold and np.array_equal(arrays['mask_predictions'][m,r,b,s],pred)
                and np.array_equal(arrays['packed_masks'][m,r,b,s],packed) and np.array_equal(arrays['IoU'][m,r,b,s],iou)
                and np.array_equal(archive['predictions'][grid_index,:17],pred)
                and np.array_equal(archive['packed_masks'][grid_index,:17],packed))
            point=dict(threshold=threshold,mask_F1=ref.macro_f1(truth,pred),IoU=float(iou.mean()))
            error['metrics']=max(error['metrics'],compare(point,metric['metrics'][mode]))
            error['source_mask_bit_differences']+=int(np.count_nonzero((archive['probability'][:17,0]>=threshold)!=mask))
    checks['all48_old_VAL_masks_classes_IoU_serialized_axes_and_archived_selected_grid_exact']=bool(valid) and error['metrics']<2e-7 and error['source_mask_bit_differences']==0
    checks['all24_old_VAL_independent_CPU_and_archived_probability_parity']=(error['CPU_probability']<=1e-5
        and error['archived_probability']<=config['old_VAL_probability_tolerance'])
    return checks,error


def audit_pre(lock_required):
    started=time.monotonic();config=read(ROOT/'CONFIG.json');checks={}
    codes=expected_code_hashes();protected={**codes,rel(ROOT/'CONFIG.json'):sha(ROOT/'CONFIG.json'),rel(ROOT/'PROTOCOL.md'):sha(ROOT/'PROTOCOL.md')}
    checks['no_fresh_sources_arrays_predictions_or_results_before_lock']=not any((ROOT/name).exists() for name in ['sources','TEST.npz','EVAL.npz','DOWNLOAD.json','PREDICTIONS.npz','RESULT.json'])
    checks['fixed_single_primary_four_cells24_frames6144_examples95CI_and_limits']=(
        config['cells']==CELLS and config['modes']==MODES and config['budgets']==BUDGETS and config['seeds']==SEEDS
        and config['frames']==24 and config['evaluation_per_frame']==256 and config['evaluation_examples']==6144
        and config['frame_prefix']==FRAME_PREFIX and config['evaluation_prefix']==EVAL_PREFIX
        and config['bootstrap_repeats']==10000 and config['bootstrap_seed']==202611033
        and config['CI_quantiles']==[.025,.975] and config['inference_batch_size']==16
        and config['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
        and config['secondary_analyses']=='descriptive_only_no_CI_no_pvalue_no_rescue'
        and config['primary_goal']==dict(minimum_mean_F1_delta=.005,CI_lower_strictly_above=0.,
            minimum_budget_F1_delta=-.0025,minimum_mean_IoU_delta=-.002,maximum_budget_ratio_deviation=.03))
    subject=ast.parse((ROOT/'evaluate.py').read_text())
    calls=[ast.unparse(node.func) for node in ast.walk(subject) if isinstance(node,ast.Call)]
    checks['evaluation_has_no_optimizer_backward_threshold_fit_or_training_calls']=not any(
        any(token in call for token in ['torch.optim.','.backward','.fit(','.step']) for call in calls)
    dev_checks,expected,states=development();checks.update(dev_checks)
    plan_checks,plan_proof=source_plan_check(read(ROOT/'SOURCE_PLAN.json') if lock_required else None);checks.update(plan_checks)
    proof_checks,proof=old_val_proof(expected,states,config);checks.update(proof_checks)
    stat_checks,stat_proof=synthetic_statistics(config);checks.update(stat_checks)
    lock_info=None
    if lock_required:
        selection=read(ROOT/'SELECTION.json');lock=read(ROOT/'MODEL_LOCK.json')
        checks['prospective_selection_and_model_lock_exact_all24_states12_exports_and_oldVAL_proof']=(
            selection['status']=='LOCKED_BEFORE_FRESH_DOWNLOAD' and lock['status']=='LOCKED_FINAL_MODELS_AND_OLD_VAL_THRESHOLDS'
            and selection['rows']==lock['rows']==expected['rows'] and selection['development']==lock['development']==expected
            and selection['model_lock_sha256']==sha(ROOT/'MODEL_LOCK.json') and selection['source_plan_sha256']==lock['source_plan_sha256']==sha(ROOT/'SOURCE_PLAN.json')
            and selection['old_VAL_preflight_sha256']==sha(ROOT/'OLD_VAL_PREFLIGHT.json')
            and selection['old_VAL_preflight_arrays_sha256']==sha(ROOT/'OLD_VAL_PREFLIGHT.npz')
            and selection['code_sha256']==codes and selection['source_model_count']==24 and selection['deployment_artifact_count']==12
            and selection['development_eligible'] and not selection['fresh_images_present']
            and selection['fit_count']==selection['optimizer_updates']==0)
        checks['exact_scientific_source_snapshots_before_download']=all((ROOT/'source_snapshot'/name).read_bytes()==(ROOT/name).read_bytes()
            for name in ['common.py','selection.py','evaluate.py','prepare.py','stats.py','CONFIG.json','PROTOCOL.md'])
        lock_info=dict(selection_sha256=sha(ROOT/'SELECTION.json'),model_lock_sha256=sha(ROOT/'MODEL_LOCK.json'),source_plan_sha256=sha(ROOT/'SOURCE_PLAN.json'))
    checks['scientific_sources_unchanged_by_independent_pre_audit']=all(sha(resolve(name))==value for name,value in protected.items())
    return dict(status=('PASS_' if all(checks.values()) else 'FAIL_')+('PRE_DOWNLOAD_AUDIT' if lock_required else 'PRE_LOCK_AUDIT'),
        checks=checks,checks_passed=sum(bool(v) for v in checks.values()),checks_total=len(checks),
        code_sha256=codes,old_VAL_proof=proof,independent_source_plan=plan_proof,independent_bootstrap_fixture=stat_proof,
        development_digest=digest(expected),lock=lock_info,new_target_access=False,fit_count=0,optimizer_updates=0,
        input_sha256=protected,seconds=time.monotonic()-started)


def all_data_inputs(selection):
    names=['SELECTION.json','MODEL_LOCK.json','SOURCE_PLAN.json','PROTOCOL.md','CONFIG.json','PRE_LOCK_AUDIT.json',
        'PRE_DOWNLOAD_AUDIT.json','OLD_VAL_PREFLIGHT.json','OLD_VAL_PREFLIGHT.npz','DOWNLOAD.json',
        'DATA_MANIFEST.json','EVAL_INDEX.json','PREPARE_CHECK.json','TEST.npz','EVAL.npz']
    paths=[ROOT/name for name in names]+[resolve(name) for name in expected_code_hashes()]
    paths += [folder/'DATA_MANIFEST.json' for folder in PRIOR]+[SHARED/'labels.zip',SHARED/'sem_zip_listing.json',
        PREVIOUS/'SOURCE_PLAN.json',PREVIOUS/'SELECTION.json',OLD/'SOURCE_PLAN.json',
        Path(__file__),DEV/'audit.py',PREVIOUS/'audit.py']
    for row in selection['rows']:paths.extend(resolve(row[name]) for name in ['free_model_file','prediction_model_file','source_VAL_file'])
    paths += [resolve(selection['development'][name]) for name in ['result_file','audit_file','prior_lock_file']]
    paths += list((ROOT/'sources').glob('*'))
    return {rel(path):sha(path) for path in sorted(set(paths)) if path.is_file()}


def audit_data():
    import zipfile,zlib
    started=time.monotonic();selection=read(ROOT/'SELECTION.json');lock=read(ROOT/'MODEL_LOCK.json')
    plan=read(ROOT/'SOURCE_PLAN.json');manifest=read(ROOT/'DATA_MANIFEST.json');index=read(ROOT/'EVAL_INDEX.json')
    download=read(ROOT/'DOWNLOAD.json');prepared=read(ROOT/'PREPARE_CHECK.json');pre=read(ROOT/'PRE_DOWNLOAD_AUDIT.json')
    checks={};inputs=all_data_inputs(selection)
    checks['no_fresh_model_prediction_before_independent_data_gate']=not any((ROOT/name).exists() for name in ['EVALUATION_FROZEN.json','PREDICTIONS.npz','BOOTSTRAP.npz','RESULT.json'])
    checks['pre_download_audit_source_snapshots_and_locked24_models_current']=(pre['status']=='PASS_PRE_DOWNLOAD_AUDIT'
        and all(pre['checks'].values()) and pre['new_target_access'] is False and pre['code_sha256']==expected_code_hashes()==selection['code_sha256']
        and pre['lock']==dict(selection_sha256=sha(ROOT/'SELECTION.json'),model_lock_sha256=sha(ROOT/'MODEL_LOCK.json'),source_plan_sha256=sha(ROOT/'SOURCE_PLAN.json'))
        and selection['rows']==lock['rows'] and len(selection['rows'])==24
        and all(sha(resolve(row[filekey]))==row[hashkey] for row in selection['rows']
                for filekey,hashkey in [('free_model_file','free_model_sha256'),('prediction_model_file','prediction_model_sha256'),('source_VAL_file','source_VAL_sha256')])
        and all(sha(ROOT/'audit_sources'/f"audit_{read(ROOT/name)['audit_source_sha256']}.py")==read(ROOT/name)['audit_source_sha256']
                for name in ['PRE_LOCK_AUDIT.json','PRE_DOWNLOAD_AUDIT.json']))
    plan_checks,plan_proof=source_plan_check(plan);checks.update(plan_checks)
    checks['all_source_access_and_preprocessing_bound_to_same_prospective_lock']=(
        download['status']=='COMPLETE' and len(download['files'])==24 and {r['frame'] for r in download['files']}==set(plan['test_frames'])
        and download['lock_verified_before_new_image_access'] and manifest['lock_verified_before_new_image_access']
        and manifest['no_models_loaded'] and manifest['no_TRAIN_VAL_or_combined_archive']
        and all(doc['selection_sha256']==sha(ROOT/'SELECTION.json') and doc['model_lock_sha256']==sha(ROOT/'MODEL_LOCK.json')
                and doc['source_plan_sha256']==sha(ROOT/'SOURCE_PLAN.json') for doc in [download,manifest,index])
        and all(doc['protocol_sha256']==sha(ROOT/'PROTOCOL.md') and doc['prepare_script_sha256']==sha(ROOT/'prepare.py') for doc in [manifest,index]))
    files={f'sources/sem{f:04}.png' for f in plan['test_frames']}|{f'sources/label{f:04}.svg' for f in plan['test_frames']}
    source_ok=(set(manifest['sources_sha256'])==files and {str(p.relative_to(ROOT)) for p in (ROOT/'sources').iterdir() if p.is_file()}==files
        and all(sha(ROOT/name)==value for name,value in manifest['sources_sha256'].items())
        and sha(SHARED/'labels.zip')==plan['shared_sha256']['labels.zip'])
    listing={row['name']:row for row in read(SHARED/'sem_zip_listing.json')}
    with zipfile.ZipFile(SHARED/'labels.zip') as labels:
        for frame in plan['test_frames']:
            row=next(v for v in download['files'] if v['frame']==frame);png=ROOT/f'sources/sem{frame:04}.png';raw=png.read_bytes()
            item=listing[f'dataset/sems/m2/sem{frame:04}.png'];entry=labels.getinfo(f'dataset/labels/m2/label{frame:04}.svg')
            svg=(ROOT/f'sources/label{frame:04}.svg').read_bytes()
            source_ok &= row['file']==str(png.relative_to(ROOT)) and row['bytes']==len(raw)==item['size']<=10_000_000
            source_ok &= sha(png)==row['sha256'] and zlib.crc32(raw)==row['crc32'] and svg==labels.read(entry) and zlib.crc32(svg)==entry.CRC
    checks['all24_native_PNG_CRC_SHA_sizes_and24_SVG_bytes_exact_archive']=bool(source_ok)
    with np.load(ROOT/'TEST.npz',allow_pickle=False) as z:test={k:z[k] for k in z.files}
    with np.load(ROOT/'EVAL.npz',allow_pickle=False) as z:evaluation={k:z[k] for k in z.files}
    records=manifest['records'];n=len(records)
    checks['TEST_new_only_shapes_dtypes_frames_and_manifest_hash']=(manifest['status']=='PREPARED_NEW_ONLY_TEST'
        and manifest['TEST_sha256']==index['TEST_sha256']==sha(ROOT/'TEST.npz') and manifest['examples']==n==len(test['X'])
        and set(test)=={'X','Y','label','role','frame'} and test['X'].shape==test['Y'].shape==(n,1,12,12)
        and test['X'].dtype==np.float32 and test['Y'].dtype==np.uint8 and set(np.unique(test['Y']))<={0,1}
        and np.all(test['role']=='test') and set(test['frame'])==set(plan['test_frames'])
        and not set(test['frame'])&set(plan['excluded_previous_frames']) and np.isfinite(test['X']).all()
        and test['X'].min()>=0 and test['X'].max()<=1)
    prior_mask={r['canonical_D4'] for folder in PRIOR for r in read(folder/'DATA_MANIFEST.json')['records']}
    prior_raw={r['raw_D4_sha256'] for folder in PRIOR for r in read(folder/'DATA_MANIFEST.json')['records']}
    checks['all7_prior_D4_union_counts_hashes_exact']=(manifest['prior_manifests_sha256']==plan['prior_manifests_sha256']
        and manifest['prior_union_unique_counts']==dict(mask=len(prior_mask),raw=len(prior_raw)))
    used_mask=prior_mask.copy();used_raw=prior_raw.copy();cursor=0;native_ok=metadata_ok=True;max_error=0.;reconstructed=[]
    for frame in plan['test_frames']:
        xs,ys,coords,polygons=data_ref.independent_native_patches(ROOT/f'sources/sem{frame:04}.png',ROOT/f'sources/label{frame:04}.svg')
        keys=data_ref.canonical_batch(ys);raw_keys=data_ref.canonical_batch(xs,True);kept=0
        for raw_x,y,coordinate,key,raw_key in zip(xs,ys,coords,keys,raw_keys):
            if key in used_mask or raw_key in used_raw:continue
            used_mask.add(key);used_raw.add(raw_key);assert cursor<n
            row=records[cursor];normalized=raw_x.astype(np.float32)/(16*255)
            native_ok &= np.array_equal(normalized,test['X'][cursor,0]) and np.array_equal(y,test['Y'][cursor,0])
            max_error=max(max_error,float(np.abs(normalized-test['X'][cursor,0]).max()))
            metadata_ok &= (row['frame']==frame==int(test['frame'][cursor]) and row['y']==int(coordinate[0])
                and row['x']==int(coordinate[1]) and row['canonical_D4']==key and row['raw_D4_sha256']==raw_key
                and row['role']=='test' and row['label']==int(test['label'][cursor]))
            cursor+=1;kept+=1
        summary=dict(candidate_patches=len(xs),excluded_duplicates=len(xs)-kept,kept=kept)
        metadata_ok &= manifest['deduplication'][str(frame)]==summary
        reconstructed.append(dict(frame=frame,SVG_polygons=polygons,**summary))
        print(json.dumps(dict(data_audit_frame=frame,completed=len(reconstructed),total=24,independent_retained=cursor)),flush=True)
    checks['all_native_TEST_4x4_integer_sums_SVG_majority_masks_exact']=bool(native_ok) and cursor==n and max_error==0.
    checks['all_stride6_D4_exclusions_order_coordinates_records_counts_exact']=(bool(metadata_ok) and cursor==n
        and len({r['canonical_D4'] for r in records})==len({r['raw_D4_sha256'] for r in records})==n
        and not {r['canonical_D4'] for r in records}&prior_mask and not {r['raw_D4_sha256'] for r in records}&prior_raw)
    labels=ref.classify(test['Y'][:,0].astype(bool))
    checks['all_TEST_classes_independent_connectivity_and_Cholesky_geometry']=np.array_equal(labels,test['label'])
    selected=[]
    for frame in plan['test_frames']:
        candidates=[i for i,row in enumerate(records) if row['frame']==frame]
        candidates.sort(key=lambda i:hashlib.sha256((EVAL_PREFIX+records[i]['raw_D4_sha256']).encode()).hexdigest())
        selected.extend(candidates[:256])
    ids=np.asarray(index['local_TEST_indices'],np.int64)
    checks['exact6144_EVAL_raw_only_rank_and_TEST_array_record_mapping']=(ids.tolist()==sorted(selected)
        and len(ids)==len(set(ids.tolist()))==6144 and all(np.array_equal(evaluation[k],test[k][ids]) for k in test)
        and index['records']==[records[i] for i in ids] and index['examples']==6144
        and index['selection_prefix']==EVAL_PREFIX and index['label_blind_selection'] and index['mask_not_used_in_ranking']
        and index['replacement_on_gate_failure'] is False and index['EVAL_sha256']==sha(ROOT/'EVAL.npz'))
    counts=np.bincount(evaluation['label'],minlength=9)
    class_frames={str(k):sorted(set(evaluation['frame'][evaluation['label']==k].tolist())) for k in range(9)}
    frame_counts={str(f):int((evaluation['frame']==f).sum()) for f in plan['test_frames']}
    gates=dict(exactly_6144=len(ids)==6144,exactly_256_each_frame=all(v==256 for v in frame_counts.values()),
        min_8_each_class=bool(counts.min()>=8),min_3_frames_each_class=all(len(v)>=3 for v in class_frames.values()))
    checks['all_predeclared24x256_6144_class8_frame3_no_replacement_gates']=(index['status']=='PASS_EVAL_DATA_GATE'
        and all(gates.values()) and index['gates']==gates and index['class_counts']==counts.tolist()
        and index['class_frames']==class_frames and index['frame_counts']==frame_counts)
    checks['producer_native_reconstruction_and_all_self_checks_pass']=(prepared['status']=='PASS_PREPARE_CHECK'
        and all(prepared['checks'].values()) and prepared['maximum_native_reconstruction_error']==0.)
    checks['all_input_source_model_data_hashes_unchanged_during_independent_audit']=all(sha(resolve(path))==value for path,value in inputs.items())
    return dict(status='PASS_DATA_AUDIT' if all(checks.values()) else 'FAIL_DATA_AUDIT',checks=checks,
        checks_passed=sum(bool(v) for v in checks.values()),checks_total=len(checks),input_sha256=inputs,
        test_examples=n,evaluation_examples=len(ids),frames=plan['test_frames'],frame_counts=frame_counts,
        class_counts=counts.tolist(),class_frames=class_frames,independent_native_reconstruction=reconstructed,
        maximum_native_reconstruction_error=max_error,independent_source_plan=plan_proof,
        no_model_predictions=True,fit_count=0,optimizer_updates=0,seconds=time.monotonic()-started)


def audit_result():
    started=time.monotonic();result=read(ROOT/'RESULT.json');frozen=read(ROOT/'EVALUATION_FROZEN.json')
    data_audit=read(ROOT/'DATA_AUDIT.json');selection=read(ROOT/'SELECTION.json');lock=read(ROOT/'MODEL_LOCK.json')
    plan=read(ROOT/'SOURCE_PLAN.json');config=read(ROOT/'CONFIG.json');index=read(ROOT/'EVAL_INDEX.json')
    paths=[ROOT/name for name in ['RESULT.json','EVALUATION_FROZEN.json','DATA_AUDIT.json','PREDICTIONS.npz',
        'BOOTSTRAP.npz','EVAL.npz','EVAL_INDEX.json','DATA_MANIFEST.json','SELECTION.json','MODEL_LOCK.json','SOURCE_PLAN.json','CONFIG.json','PROTOCOL.md']]
    inputs={**data_audit['input_sha256'],**{rel(path):sha(path) for path in paths}}
    checks={}
    checks['data_audit_byte_identical_all_native_data_inputs_still_current']=(
        data_audit['status']=='PASS_DATA_AUDIT' and all(data_audit['checks'].values())
        and all(sha(resolve(path))==value for path,value in data_audit['input_sha256'].items())
        and frozen['input_sha256'][rel(ROOT/'DATA_AUDIT.json')]==sha(ROOT/'DATA_AUDIT.json'))
    checks['scientific_model_threshold_code_and_cohort_locks_current_before_prediction']=(
        frozen['status']=='FROZEN_BEFORE_FIRST_FRESH_PREDICTION' and frozen['fit_count']==frozen['optimizer_updates']==0
        and frozen['code_sha256']==expected_code_hashes()==selection['code_sha256']
        and frozen['selection_sha256']==sha(ROOT/'SELECTION.json')
        and all(sha(resolve(path))==value for path,value in frozen['input_sha256'].items())
        and selection['rows']==lock['rows'] and len(selection['rows'])==24
        and selection['model_lock_sha256']==sha(ROOT/'MODEL_LOCK.json')
        and selection['source_plan_sha256']==lock['source_plan_sha256']==sha(ROOT/'SOURCE_PLAN.json')
        and all(sha(resolve(row[filekey]))==row[hashkey] for row in selection['rows']
                for filekey,hashkey in [('free_model_file','free_model_sha256'),('prediction_model_file','prediction_model_sha256'),('source_VAL_file','source_VAL_sha256')]))
    with np.load(ROOT/'EVAL.npz',allow_pickle=False) as z:evaluation={k:z[k] for k in z.files}
    with np.load(ROOT/'PREDICTIONS.npz',allow_pickle=False) as z:saved={k:z[k] for k in z.files}
    with np.load(ROOT/'BOOTSTRAP.npz',allow_pickle=False) as z:bootstrap={k:z[k] for k in z.files}
    truth=evaluation['label'];target=evaluation['Y'][:,0].astype(bool);frames=evaluation['frame'];n=len(truth)
    checks['final_cohort6144_24x256_and_zero_fits_array_hashes_bound']=(
        result['test_examples']==n==6144 and result['test_frames']==plan['test_frames']
        and len(np.unique(frames))==24 and all((frames==f).sum()==256 for f in plan['test_frames'])
        and result['test_class_counts']==np.bincount(truth,minlength=9).tolist()
        and result['source_model_count']==result['evaluated_routes']==24 and result['fit_count']==result['optimizer_updates']==0
        and result['selection_sha256']==sha(ROOT/'SELECTION.json') and result['evaluation_frozen_sha256']==sha(ROOT/'EVALUATION_FROZEN.json')
        and result['predictions_sha256']==sha(ROOT/'PREDICTIONS.npz') and result['bootstrap_sha256']==sha(ROOT/'BOOTSTRAP.npz'))
    valid=(saved['probability'].shape==(4,2,3,n,12,12) and saved['mask_predictions'].shape==saved['IoU'].shape==(2,4,2,3,n)
        and saved['packed_masks'].shape==(2,4,2,3,n,18) and saved['thresholds'].shape==(2,4,2,3)
        and np.isfinite(saved['probability']).all() and saved['probability'].min()>=0 and saved['probability'].max()<=1
        and np.array_equal(saved['truth'],truth) and np.array_equal(saved['target'],target) and np.array_equal(saved['frames'],frames)
        and saved['source_indices'].tolist()==index['local_TEST_indices']
        and np.array_equal(saved['normalized_fixture_x'],5*(evaluation['X'][:17]-.2)))
    for array in [saved,bootstrap]:
        valid &= array['modes'].tolist()==MODES and array['cells'].tolist()==CELLS and array['budgets'].tolist()==BUDGETS and array['seeds'].tolist()==SEEDS
    checks['all_prediction_bootstrap_axes_targets_source_mapping_and_probability_range']=bool(valid)
    predictions=np.empty((2,4,2,3,n),np.int64);ious=np.empty_like(predictions,dtype=np.float64)
    masks_ok=classes_ok=thresholds_ok=True;metric_error=iou_error=0.
    metadata_ok=(len(result['rows'])==24 and {(r['cell'],r['budget'],r['seed']) for r in result['rows']}==
        {(cell,b,s) for cell in CELLS for b in BUDGETS for s in SEEDS})
    for row in selection['rows']:
        unit=row['cell'],row['budget'],row['seed'];r,b,s=CELLS.index(unit[0]),BUDGETS.index(unit[1]),SEEDS.index(unit[2])
        p=saved['probability'][r,b,s];record=next(v for v in result['rows'] if (v['cell'],v['budget'],v['seed'])==unit)
        metadata_ok &= (record['prediction_model_sha256']==row['prediction_model_sha256'] and record['route']==row['route']
            and record['objective']==row['objective'] and record['CNN_passes_per_image']==(1 if row['architecture']=='group6' else 4))
        for m,mode in enumerate(MODES):
            threshold=row[mode+'_threshold'];mask=p>=threshold;prediction=ref.classify(mask)
            predictions[m,r,b,s]=prediction;packed=np.packbits(mask.reshape(n,144),axis=1)
            masks_ok &= np.array_equal(packed,saved['packed_masks'][m,r,b,s])
            thresholds_ok &= saved['thresholds'][m,r,b,s]==threshold
            classes_ok &= np.array_equal(prediction,saved['mask_predictions'][m,r,b,s])
            union=(mask|target).sum((1,2));intersection=(mask&target).sum((1,2))
            ious[m,r,b,s]=np.divide(intersection,union,out=np.ones(n),where=union>0)
            iou_error=max(iou_error,float(np.abs(ious[m,r,b,s]-saved['IoU'][m,r,b,s]).max()))
            point=dict(threshold=threshold,mask_F1=ref.macro_f1(truth,prediction),IoU=float(ious[m,r,b,s].mean()))
            metric_error=max(metric_error,compare(point,record['metrics'][mode]))
        print(json.dumps(dict(final_audit_cell=row['cell'],budget=row['budget'],seed=row['seed'],both_modes=True)),flush=True)
    checks['all48_locked_threshold_masks_and_packed_bits_exact']=bool(masks_ok and thresholds_ok)
    checks['all48_independent_graph_classes_macroF1_IoU_and24_rows_exact']=(bool(classes_ok and metadata_ok) and metric_error<2e-7 and iou_error==0.)
    confusion,frame_values=independent_confusion(predictions,truth,frames)
    checks['all_independent_frame_confusions_and_frame_axis_exact']=np.array_equal(confusion,bootstrap['confusion']) and np.array_equal(frame_values,bootstrap['frame_values'])
    draws=np.random.default_rng(config['bootstrap_seed'])
    seed_ids=draws.integers(0,3,(10000,3));frame_ids=draws.integers(0,24,(10000,24))
    checks['all10000_shared_paired_seed_then_frame_draws_exact']=(
        config['bootstrap_repeats']==result['bootstrap_repeats']==frozen['bootstrap_repeats']==10000
        and config['bootstrap_seed']==result['bootstrap_seed']==frozen['bootstrap_seed']==202611033
        and np.array_equal(seed_ids,bootstrap['seed_ids']) and np.array_equal(frame_ids,bootstrap['frame_ids']))
    points=macro(confusion.sum(4));budgets,scores,delta=independent_bootstrap(confusion,seed_ids,frame_ids)
    error=max(float(np.abs(points-bootstrap['point_per_seed_budget']).max()),float(np.abs(budgets-bootstrap['primary_budget_scores']).max()),
        float(np.abs(scores-bootstrap['primary_scores']).max()),float(np.abs(delta-bootstrap['delta_F1']).max()))
    checks['all10000_independent_weighted_frame_seed_frequency_bootstrap_scores']=error<1e-14
    analyses=expected_analyses(points,ious,config,delta,selection['development_eligible'])
    analysis_error=compare(analyses,result['analyses'])
    checks['all_four_cells_five_descriptive_contrasts_manual95CI_primary_guards_exact']=(
        analysis_error<1e-14 and config['CI_quantiles']==result['CI_quantiles']==bootstrap['CI_quantiles'].tolist()==[.025,.975])
    confirmed=bool(analyses['primary_fine']['primary_goal']['confirmed'])
    checks['single_primary_confirmation_status_and_fixed05_secondary_no_rescue']=(
        result['confirmed']==confirmed and result['status']==('PASS_CONFIRMATION' if confirmed else 'FAIL_CONFIRMATION')
        and result['primary_mode']=='primary_fine' and result['secondary_mode']=='fixed05'
        and not analyses['fixed05']['primary_goal']['confirmed'] and 'CI95' not in analyses['fixed05']['primary_goal']
        and all('CI95' not in v and 'pvalue' not in v for v in analyses['primary_fine']['contrasts'].values()))
    checks['all_locked_model_code_data_audit_input_hashes_unchanged_after_recomputation']=all(sha(resolve(path))==value for path,value in inputs.items())
    return dict(status='PASS_AUDIT' if all(checks.values()) else 'FAIL_AUDIT',checks=checks,
        checks_passed=sum(bool(v) for v in checks.values()),checks_total=len(checks),input_sha256=inputs,
        result_sha256=sha(ROOT/'RESULT.json'),maximum_errors=dict(all_F1_IoU_metrics=metric_error,
            per_patch_IoU=iou_error,bootstrap_scores=error,full_analyses=analysis_error),
        independent_analyses=analyses,confirmed=confirmed,fit_count=0,optimizer_updates=0,
        no_model_inference_rerun=True,seconds=time.monotonic()-started)


def save_audit(name,result):
    source=sha(__file__);snapshots=ROOT/'audit_sources';snapshots.mkdir(exist_ok=True)
    snapshot=snapshots/f'audit_{source}.py'
    if snapshot.exists():assert snapshot.read_bytes()==Path(__file__).read_bytes()
    else:shutil.copy2(__file__,snapshot)
    result['audit_source_sha256']=source
    target=ROOT/name
    def scalar(value):
        if isinstance(value,np.generic):return value.item()
        raise TypeError(f'Unsupported audit metadata type: {type(value).__name__}')
    encoded=json.dumps(result,ensure_ascii=False,indent=2,default=scalar)+'\n'
    if target.exists() and target.read_text()!=encoded:
        shutil.copy2(target,ROOT/f'{target.stem}_PREVIOUS_{sha(target)[:12]}.json')
    target.write_text(encoded)


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pre-lock',action='store_true');group.add_argument('--pre-download',action='store_true')
    group.add_argument('--data',action='store_true');group.add_argument('--result',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        result=audit_result() if args.result else (audit_data() if args.data else audit_pre(args.pre_download))
    name='AUDIT.json' if args.result else ('DATA_AUDIT.json' if args.data else ('PRE_DOWNLOAD_AUDIT.json' if args.pre_download else 'PRE_LOCK_AUDIT.json'))
    save_audit(name,result)
    print(json.dumps(dict(status=result['status'],checks_passed=result['checks_passed'],checks_total=result['checks_total'],
        failed=[k for k,v in result['checks'].items() if not v],seconds=result['seconds'])),flush=True)
    assert result['status'].startswith('PASS')


if __name__=='__main__':main()
