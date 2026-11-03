"""Independent prospective group-CNN confirmation audit.

Pre-lock and pre-download modes read only metadata, locked development
checkpoints and old VAL. They never download or read prospective pixels.
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
DEV = G / 'sem_arch_groupcnn_20261102'
PILOT = G / 'sem_architecture_pilot_20261102'
OLD = G / 'sem_arch_orbit_confirmation_20261102'
ORIGINAL = G / 'sem_confirmation_20261028'
SHARED = G / 'shared_data/ic_sem_m2'
PRIOR = [ORIGINAL,G/'sem_cldice_confirmation_20261030',G/'sem_cldice_expansion_20261102',G/'sem_scarcity_20261102',PILOT,OLD]
ROUTES, BUDGETS, SEEDS, MODES = ['group6_single','plain12_C4','plain24_C4'],[18,27],[5701,5721,5757],['primary_fine','secondary_seven']
FRAME_PREFIX, EVAL_PREFIX = 'SEM-GROUP-CNN-CONFIRM-20261102/','SEM-GROUP-CNN-EVAL-20261102/'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def resolve(path): return Path(path) if Path(path).is_absolute() else REPO/path


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


independent = module('preserved_stage12_independent_reference',DEV/'audit.py')


def expected_code_hashes():
    paths=[ROOT/name for name in ['confirmation_common.py','confirmation_stats.py','selection.py','prepare.py','evaluate.py']]
    paths += [DEV/'model.py',PILOT/'prepare.py',OLD/'prepare.py',OLD/'evaluate.py',ORIGINAL/'prepare.py',G/'retained_evidence_20261027/SELECTED_SOURCES.zip',SHARED/'sem_remote_zip.py']
    return {str(path.relative_to(REPO)):sha(path) for path in paths}


def development():
    aggregate_files=list((DEV/'trials').glob('*/AGGREGATE_RESULT.json'));assert len(aggregate_files)==1
    aggregate_path=aggregate_files[0];aggregate=read(aggregate_path)
    checks={};eligible={name:bool(aggregate['gates'][name]['passed'] and aggregate['new_seed_gates'][name]['passed']) for name in ['A_matched_quality','B_full_efficiency']}
    checks['same_goal_development_eligibility_all3_and_new2']=eligible['A_matched_quality']==aggregate['robust_primary_pass'] and eligible['B_full_efficiency']==aggregate['robust_secondary_pass'] and any(eligible.values())==aggregate['robust_ANY_GOAL_PASS']
    source_rows={};provenance=[];sources_ok=True
    for name in aggregate['source_trials']:
        folder=DEV/name;result=read(folder/'RESULT.json');audit=read(folder/'AUDIT.json')
        sources_ok &= audit['status']=='PASS_AUDIT' and all(audit['checks'].values()) and audit['result_sha256']==sha(folder/'RESULT.json') and audit['audit_source_sha256']==sha(DEV/'audit.py')
        provenance.append(dict(result_file=str((folder/'RESULT.json').relative_to(REPO)),result_sha256=sha(folder/'RESULT.json'),audit_file=str((folder/'AUDIT.json').relative_to(REPO)),audit_sha256=sha(folder/'AUDIT.json')))
        for row in result['rows']:
            if row['route'] in ROUTES:
                unit=row['route'],row['budget'],row['seed'];assert unit not in source_rows;source_rows[unit]=(row,folder)
    expected={(r,b,s) for r in ROUTES for b in BUDGETS for s in SEEDS}
    checks['both_immutable_development_audits_results_and_unique18_primary_routes']=bool(sources_ok) and set(source_rows)==expected and aggregate['seeds']==SEEDS
    rows=[];states={};model_ok=True
    aggregate_rows={(r['route'],r['budget'],r['seed']):r for r in aggregate['rows'] if r['route'] in ROUTES}
    for route in ROUTES:
        for budget in BUDGETS:
            for seed in SEEDS:
                unit=route,budget,seed;row,folder=source_rows[unit]
                model_ok &= row==aggregate_rows[unit] and row['selected']==row['primary_fine']['selected']==max(row['primary_fine']['all_thresholds'],key=independent.selection_rank)
                model_ok &= row['secondary_seven']['selected']==max(row['secondary_seven']['all_thresholds'],key=independent.selection_rank)
                model_ok &= [r['threshold'] for r in row['primary_fine']['all_thresholds']]==[round(.1+.01*i,2) for i in range(81)]
                free=folder/row['model_file'];prediction=folder/'models'/f'group6_b{budget}_s{seed}_deployed.pt' if route=='group6_single' else free
                payload=torch.load(free,map_location='cpu',weights_only=True);state=payload['model'];deployed=torch.load(prediction,map_location='cpu',weights_only=True)
                count=2671 if route=='group6_single' else (2749 if route=='plain12_C4' else 10681)
                model_ok &= sha(free)==row['model_sha256'] and sum(v.numel() for v in state.values())==count and all(torch.isfinite(v).all().item() for v in state.values())
                if route=='group6_single':
                    expected_deployed=independent.reference_deploy_state(state)
                    model_ok &= deployed['free_model_sha256']==sha(free) and deployed['free_parameters']==2671 and deployed['materialized_coefficients']==10681 and all(torch.equal(v,deployed['model'][key]) for key,v in expected_deployed.items())
                val=folder/'val_probs'/f'{route}_b{budget}_s{seed}.npz';stored=10681 if route!='plain12_C4' else 2749
                descriptor=dict(route=route,architecture=row['architecture'],budget=budget,seed=seed,free_model_file=str(free.relative_to(REPO)),free_model_sha256=sha(free),prediction_model_file=str(prediction.relative_to(REPO)),prediction_model_sha256=sha(prediction),prediction_width=24 if stored==10681 else 12,prediction_mode='single' if route=='group6_single' else 'C4',free_parameters=count,stored_prediction_coefficients=stored,primary_fine_threshold=row['selected']['threshold'],secondary_seven_threshold=row['secondary_seven']['selected']['threshold'],source_VAL_file=str(val.relative_to(REPO)),source_VAL_sha256=sha(val))
                rows.append(descriptor);states[unit]=deployed['model']
    checks['all18_final_free_states_six_exact_deployments_and_frozen81_and7_thresholds']=bool(model_ok)
    result=dict(rows=rows,eligible_by_goal=eligible,aggregate_file=str(aggregate_path.relative_to(REPO)),aggregate_sha256=sha(aggregate_path),source_trials=provenance,source_model_count=18,deployment_artifact_count=6,interpretation='Development eligibility only; not fresh confirmation')
    return checks,result,states


def source_plan_check(plan=None):
    previous=read(OLD/'SOURCE_PLAN.json');pilot=read(PILOT/'SOURCE_PLAN.json');old_selection=read(OLD/'SELECTION.json')
    excluded=set(pilot['excluded_previous_frames'])|set(pilot['test'])|set(previous['test_frames'])
    prior_frames=set();mask=set();raw=set();hashes={}
    for folder in PRIOR:
        manifest=read(folder/'DATA_MANIFEST.json');hashes[str((folder/'DATA_MANIFEST.json').relative_to(G))]=sha(folder/'DATA_MANIFEST.json')
        prior_frames.update(r['frame'] for r in manifest['records']);mask.update(r['canonical_D4'] for r in manifest['records']);raw.update(r['raw_D4_sha256'] for r in manifest['records'])
    listing=read(SHARED/'sem_zip_listing.json');available=[int(m.group(1)) for row in listing if (m:=re.fullmatch(r'dataset/sems/m2/sem(\d+)\.png',row['name']))]
    chosen=sorted(set(available)-excluded,key=lambda n:hashlib.sha256((FRAME_PREFIX+str(n)).encode()).hexdigest())[:16]
    checks=dict(independent94_frame_chain_all_six_manifest_frames_excluded=(len(pilot['excluded_previous_frames'])==78 and len(previous['excluded_previous_frames'])==86 and len(excluded)==94 and set(previous['excluded_previous_frames'])==set(pilot['excluded_previous_frames'])|set(pilot['test']) and previous['exclusion_source_sha256']==sha(PILOT/'SOURCE_PLAN.json') and old_selection['source_plan_sha256']==sha(OLD/'SOURCE_PLAN.json') and prior_frames.issubset(excluded)),
        independent321_listing_and_fixed16_hash_rank=(len(available)==len(set(available))==321 and chosen==[61,87,81,212,100,16,183,252,32,150,157,192,254,200,70,123]))
    if plan is not None:
        checks['locked_source_plan_exact_frames_six_manifests_raw_only_rank_and_coverage']=(plan['test_frames']==chosen and plan['excluded_previous_frames']==sorted(excluded) and plan['excluded_frame_count']==94 and plan['prior_manifests_sha256']==hashes and plan['archive_listing_sha256']==sha(SHARED/'sem_zip_listing.json') and plan['selection_prefix']==FRAME_PREFIX and plan['evaluation']['selection_prefix']==EVAL_PREFIX and plan['evaluation']['patches_per_frame']==256 and plan['evaluation']['target_examples']==4096 and plan['evaluation']['minimum_examples_each_class']==8 and plan['evaluation']['minimum_frames_each_class']==3 and plan['evaluation']['replacement_on_gate_failure'] is False and plan['prepare_script_sha256']==sha(ROOT/'prepare.py'))
    return checks,dict(frames=chosen,excluded_frames=len(excluded),prior_manifest_count=6,prior_record_frame_count=len(prior_frames),prior_unique_D4_mask_count=len(mask),prior_unique_D4_raw_count=len(raw),prior_manifests_sha256=hashes)


def macro(confusion):
    confusion=np.asarray(confusion,dtype=np.float64);den=confusion.sum(-1)+confusion.sum(-2)
    return np.divide(2*np.diagonal(confusion,axis1=-2,axis2=-1),den,out=np.zeros_like(den),where=den>0).mean(-1)


def synthetic_statistics(config):
    sys.path.insert(0,str(ROOT));subject=module('confirmation_statistics_subject',ROOT/'confirmation_stats.py')
    rng=np.random.default_rng(5717);frames=np.repeat(np.arange(16),31);truth=rng.integers(0,9,len(frames));pred=np.empty((2,3,2,3,len(truth)),np.int64)
    for m in range(2):
        for r in range(3):
            for b in range(2):
                for s in range(3):
                    candidate=truth.copy();change=rng.random(len(truth))<(.05+.22*s+.04*r+.01*b+.02*m)
                    candidate[change]=(candidate[change]+rng.integers(1,9,change.sum()))%9;pred[m,r,b,s]=candidate
    iou=rng.uniform(.6,.9,pred.shape);small=dict(config,bootstrap_repeats=7);eligible={name:True for name in config['goals']}
    analyses,actual=subject.analyze(pred,iou,truth,frames,small,eligible)
    expected_conf=np.zeros((2,3,2,3,16,9,9),np.int64)
    for m in range(2):
        for r in range(3):
            for b in range(2):
                for s in range(3):
                    for f in range(16):
                        ids=frames==f;expected_conf[m,r,b,s,f]=np.bincount(9*truth[ids]+pred[m,r,b,s,ids],minlength=81).reshape(9,9)
    draws=np.random.default_rng(config['bootstrap_seed']);seed_ids=draws.integers(0,3,(7,3));frame_ids=draws.integers(0,16,(7,16))
    scores=np.empty((2,3,2,7));wrong=np.empty_like(scores)
    for k in range(7):
        for m in range(2):
            for r in range(3):
                for b in range(2):
                    samples=[sum(expected_conf[m,r,b,s,int(f)] for f in frame_ids[k]) for s in seed_ids[k]]
                    scores[m,r,b,k]=np.mean([macro(cm) for cm in samples]);wrong[m,r,b,k]=macro(sum(samples))
    route=scores.mean(2);deltas=np.stack((route[:,0]-route[:,1],route[:,0]-route[:,2]),1)
    error=max(float(np.max(np.abs(scores-actual['scores']))),float(np.max(np.abs(deltas-actual['deltas']))));witness=float(np.max(np.abs(scores-wrong)))
    point=macro(expected_conf.sum(4));goal_ok=True
    for m,mode in enumerate(MODES):
        for j,name in enumerate(['A_matched_quality','B_full_efficiency']):
            item=analyses[mode]['goals'][name];limits=config['goals'][name];delta=float((point[m,0]-point[m,j+1]).mean());bud=(point[m,0]-point[m,j+1]).mean(1);ci=np.quantile(deltas[m,j],[.0125,.9875])
            tests=[delta>=limits['minimum_mean_F1_delta'],ci[0]>limits['CI_lower_strictly_above'],bud.min()>=limits['minimum_budget_F1_delta'],float((iou[m,0]-iou[m,j+1]).mean())>=limits['minimum_mean_IoU_delta']]
            goal_ok &= item['passed']==all(tests) and item['confirmed']==(mode=='primary_fine' and all(tests)) and np.max(np.abs(ci-item['CI97_5']))<1e-14
    checks=dict(independent_heterogeneous_seed_frame_confusions=np.array_equal(expected_conf,actual['confusion']),shared_seed_first_frame_second_draws=np.array_equal(seed_ids,actual['seed_ids']) and np.array_equal(frame_ids,actual['frame_ids']),independent_brute_crossed_per_seed_macroF1_before_mean=error<1e-14,wrong_pooled_seed_estimator_sensitive_fixture=witness>1e-3,Bonferroni97_5_quantiles_both_goal_predicates_and_secondary_no_rescue=bool(goal_ok))
    return checks,dict(repeats=7,maximum_correct_estimator_error=error,incorrect_pooled_seed_estimator_difference=witness,seed_ids=seed_ids.tolist(),frame_ids=frame_ids.tolist())


def old_val_proof(expected,states,config):
    proof=read(ROOT/'OLD_VAL_PREFLIGHT.json')
    with np.load(ROOT/'OLD_VAL_PREFLIGHT.npz',allow_pickle=False) as z:arrays={key:z[key] for key in z.files}
    with np.load(PILOT/'TRAIN_VAL.npz',allow_pickle=False) as z:
        ids=np.flatnonzero(z['role']=='val')[:17];x=5*(z['X'][ids]-.2);truth=z['label'][ids];target=z['Y'][ids,0].astype(bool)
    checks=dict(real_old_VAL_proof18_routes17_examples_nonmultiple_batch_and_current_code=(proof['status']=='PASS_OLD_VAL_PREFLIGHT' and all(proof['checks'].values()) and proof['source_model_count']==18 and proof['examples']==17 and proof['inference_batch_size']==16 and proof['device'] in ['cpu','mps'] and proof['new_target_access'] is False and proof['fit_count']==0 and proof['TRAIN_VAL_sha256']==sha(PILOT/'TRAIN_VAL.npz') and proof['code_sha256']==expected_code_hashes() and proof['development_selection_digest']==json_digest(expected) and proof['protocol_sha256']==sha(ROOT/'PROTOCOL.md') and proof['config_sha256']==sha(ROOT/'CONFIG.json') and proof['arrays_sha256']==sha(ROOT/'OLD_VAL_PREFLIGHT.npz')))
    valid=(arrays['probability'].shape==(3,2,3,17,12,12) and arrays['mask_predictions'].shape==(2,3,2,3,17) and arrays['packed_masks'].shape==(2,3,2,3,17,18) and arrays['IoU'].shape==(2,3,2,3,17) and arrays['thresholds'].shape==(2,3,2,3) and len(proof['rows'])==len(proof['agreements'])==18 and np.array_equal(arrays['truth'],truth) and np.array_equal(arrays['target'],target) and np.array_equal(arrays['OLD_VAL_dataset_indices'],ids) and np.array_equal(arrays['normalized_fixture_x'],x) and arrays['modes'].tolist()==MODES and arrays['routes'].tolist()==ROUTES and arrays['budgets'].tolist()==BUDGETS and arrays['seeds'].tolist()==SEEDS)
    errors=dict(independent_CPU_probability=0.,source_VAL_probability=0.,metrics=0.,source_mask_bit_differences=0)
    source_parity=True;metrics_ok=True
    for row in expected['rows']:
        unit=row['route'],row['budget'],row['seed'];r,b,s=ROUTES.index(unit[0]),BUDGETS.index(unit[1]),SEEDS.index(unit[2]);p=arrays['probability'][r,b,s]
        state=states[unit];parts=[]
        with torch.inference_mode():
            for images in torch.tensor(x).split(16):
                if row['prediction_mode']=='single':prediction=independent.reference_plain(images,state).sigmoid()
                else:_,views=independent.stable_orbit_logits(images,state);prediction=views.sigmoid().mean(0)
                parts.append(prediction[:,0].numpy())
        errors['independent_CPU_probability']=max(errors['independent_CPU_probability'],float(np.max(np.abs(np.concatenate(parts)-p))))
        assert sha(resolve(row['source_VAL_file']))==row['source_VAL_sha256']
        with np.load(resolve(row['source_VAL_file']),allow_pickle=False) as z:archived={key:z[key] for key in z.files}
        delta=float(np.max(np.abs(archived['probability'][:17,0]-p)));errors['source_VAL_probability']=max(errors['source_VAL_probability'],delta)
        agreement=next(v for v in proof['agreements'] if (v['route'],v['budget'],v['seed'])==unit)
        metric=next(v for v in proof['rows'] if (v['route'],v['budget'],v['seed'])==unit)
        valid &= (agreement['maximum_probability_error']==delta and agreement['source_VAL_file']==row['source_VAL_file'] and agreement['source_VAL_sha256']==row['source_VAL_sha256'] and metric['prediction_model_sha256']==row['prediction_model_sha256'] and metric['CNN_passes_per_image']==(1 if r==0 else 4) and np.array_equal(archived['truth'][:17],truth) and np.array_equal(archived['target'][:17],target))
        for m,mode in enumerate(MODES):
            threshold=row[mode+'_threshold'];mask=p>=threshold;pred=independent.classify(mask);iou=np.divide((mask&target).sum((1,2)),(mask|target).sum((1,2)),out=np.ones(17),where=(mask|target).sum((1,2))>0)
            packed=np.packbits(mask.reshape(17,144),axis=1);valid &= arrays['thresholds'][m,r,b,s]==threshold and np.array_equal(arrays['mask_predictions'][m,r,b,s],pred) and np.array_equal(arrays['packed_masks'][m,r,b,s],packed) and np.array_equal(arrays['IoU'][m,r,b,s],iou)
            scores=dict(threshold=threshold,mask_F1=independent.macro_f1(truth,pred),IoU=float(iou.mean()))
            errors['metrics']=max(errors['metrics'],independent.numeric_compare(scores,metric['metrics'][mode]))
            grid='thresholds' if m==0 else 'secondary_thresholds';predkey='predictions' if m==0 else 'secondary_predictions';i=archived[grid].tolist().index(threshold)
            source_parity &= np.array_equal(archived[predkey][i,:17],pred) and agreement['selected_class_parity'][m]
            source_mask=archived['probability'][:17,0]>=threshold;errors['source_mask_bit_differences']+=int(np.count_nonzero(source_mask!=mask))
    checks['all18_both_threshold_formats_independent_masks_classes_IoU_and_serialized_axes']=bool(valid) and errors['metrics']<2e-7
    checks['all18_locked_state_independent_CPU_inference_and_archived_VAL_class_agreement']=bool(source_parity) and errors['independent_CPU_probability']<1e-4 and errors['source_VAL_probability']<=config['old_VAL_probability_tolerance']
    return checks,dict(device=proof['device'],examples=17,models=18,errors=errors)


def audit_pre(lock_required):
    started=time.monotonic();config=read(ROOT/'CONFIG.json');checks={}
    protected={str(path):sha(path) for path in [DEV/'audit.py',ROOT/'CONFIG.json',ROOT/'PROTOCOL.md']}
    protected.update({str(resolve(name)):value for name,value in expected_code_hashes().items()})
    checks['no_prospective_target_download_pixels_arrays_or_predictions_present']=not any((ROOT/name).exists() for name in ['sources','TEST.npz','EVAL.npz','PREDICTIONS.npz','RESULT.json','DOWNLOAD.json'])
    checks['exact_predeclared_routes_seeds_frame_sample_bootstrap_and_goal_limits']=(config['routes']==ROUTES and config['budgets']==BUDGETS and config['seeds']==SEEDS and config['frames']==16 and config['evaluation_per_frame']==256 and config['evaluation_examples']==4096 and config['minimum_examples_each_class']==8 and config['minimum_frames_each_class']==3 and config['frame_prefix']==FRAME_PREFIX and config['evaluation_prefix']==EVAL_PREFIX and config['bootstrap_repeats']==10000 and config['bootstrap_seed']==202611027 and config['CI_quantiles']==[.0125,.9875] and config['inference_batch_size']==16 and config['protocol_sha256']==sha(ROOT/'PROTOCOL.md') and config['goals']==dict(A_matched_quality=dict(minimum_mean_F1_delta=.005,CI_lower_strictly_above=0.,minimum_budget_F1_delta=-.0025,minimum_mean_IoU_delta=-.002),B_full_efficiency=dict(minimum_mean_F1_delta=-.005,CI_lower_strictly_above=-.005,minimum_budget_F1_delta=-.01,minimum_mean_IoU_delta=-.002)))
    dev_checks,expected,states=development();checks.update(dev_checks)
    plan=read(ROOT/'SOURCE_PLAN.json') if lock_required else None
    plan_checks,plan_proof=source_plan_check(plan);checks.update(plan_checks)
    old_checks,old_proof=old_val_proof(expected,states,config);checks.update(old_checks)
    stats_checks,stats_proof=synthetic_statistics(config);checks.update(stats_checks)
    common_tree=ast.parse((ROOT/'confirmation_common.py').read_text());common_functions={n.name:ast.unparse(n) for n in common_tree.body if isinstance(n,ast.FunctionDef)}
    prepare_tree=ast.parse((ROOT/'prepare.py').read_text());prepare_functions={n.name:ast.unparse(n) for n in prepare_tree.body if isinstance(n,ast.FunctionDef)}
    eval_tree=ast.parse((ROOT/'evaluate.py').read_text());eval_function=next(n for n in eval_tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate');eval_source=ast.unparse(eval_function)
    checks['download_and_prepare_verify_model_code_threshold_lock_before_target_access']=(ast.unparse(next(n for n in prepare_tree.body if isinstance(n,ast.FunctionDef) and n.name=='download').body[0])=='plan = frozen_plan()' and ast.unparse(next(n for n in prepare_tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare').body[0])=='plan = frozen_plan()' and 'verify_lock()' in prepare_functions['frozen_plan'] and 'len(PRIOR) == 6' in prepare_functions['prior_hashes'])
    ranking=prepare_functions['evaluation_subset'];rank_node=next(n for n in ast.walk(next(n for n in prepare_tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluation_subset')) if isinstance(n,ast.Lambda))
    checks['raw_D4_only_256_per_frame_rank_and_no_coverage_resampling']=("records[i]['raw_D4_sha256']" in ast.unparse(rank_node) and 'canonical_D4' not in ast.unparse(rank_node) and 'label' not in ast.unparse(rank_node) and 'candidates[:256]' in ranking and 'replacement_on_gate_failure=False' in ranking and "status='PASS_EVAL_DATA_GATE' if all(gates.values()) else 'INSUFFICIENT_FROZEN_COHORT'" in ranking and 'raise RuntimeError' in ranking)
    checks['all_six_prior_raw_and_mask_D4_motifs_and_new_cohort_duplicates_excluded']=("for folder in PRIOR" in prepare_functions['prepare'] and "prior_mask = {row['canonical_D4'] for row in prior_records}" in prepare_functions['prepare'] and "prior_raw = {row['raw_D4_sha256'] for row in prior_records}" in prepare_functions['prepare'] and 'if canonical in used_mask or raw_sha in used_raw:' in prepare_functions['prepare'] and 'used_mask.add(canonical)' in prepare_functions['prepare'] and 'used_raw.add(raw_sha)' in prepare_functions['prepare'])
    checks['evaluation_requires_passing_current_cohort_hash_bound_independent_data_audit']=("data_audit['status'].startswith('PASS')" in eval_source and "data_audit['input_sha256']" in eval_source and all(name in eval_source for name in ['EVAL.npz','DATA_MANIFEST.json','EVAL_INDEX.json']) and eval_source.index("data_audit['input_sha256']")<eval_source.index('c.predict_models(') and 'c.sha(c.resolve(path)) == expected' in eval_source)
    checks['same_inference_path_real_preflight_and_fresh_evaluation_no_fitting_or_threshold_search']=('def predict_models' in (ROOT/'confirmation_common.py').read_text() and 'c.predict_models(' in (ROOT/'selection.py').read_text() and 'c.predict_models(' in eval_source and 'optimizer' not in eval_source and 'backward' not in eval_source and 'np.quantile' not in common_functions['predict_models'] and "threshold = row[mode + '_threshold']" in common_functions['predict_models'])
    checks['code_hashes_bind_every_executed_workflow_model_and_class_rule_source']=len(expected_code_hashes())==12 and str((PILOT/'prepare.py').relative_to(REPO)) in expected_code_hashes()
    lock_proof={}
    if lock_required:
        selection=read(ROOT/'SELECTION.json');lock=read(ROOT/'MODEL_LOCK.json')
        checks['actual_pre_download_lock_model_states_thresholds_code_and_old_VAL_proof_hashes']=(selection['status']=='LOCKED_BEFORE_FRESH_DOWNLOAD' and lock['status']=='LOCKED_FINAL_MODELS_AND_OLD_VAL_THRESHOLDS' and selection['rows']==lock['rows']==expected['rows'] and selection['development']==lock['development']==expected and selection['eligible_by_goal']==expected['eligible_by_goal'] and selection['code_sha256']==expected_code_hashes() and selection['model_lock_sha256']==sha(ROOT/'MODEL_LOCK.json') and selection['source_plan_sha256']==lock['source_plan_sha256']==sha(ROOT/'SOURCE_PLAN.json') and selection['old_VAL_preflight_sha256']==sha(ROOT/'OLD_VAL_PREFLIGHT.json') and selection['old_VAL_preflight_arrays_sha256']==sha(ROOT/'OLD_VAL_PREFLIGHT.npz') and selection['protocol_sha256']==lock['protocol_sha256']==sha(ROOT/'PROTOCOL.md') and selection['config_sha256']==lock['config_sha256']==sha(ROOT/'CONFIG.json') and selection['fresh_images_present'] is False and selection['fit_count']==selection['optimizer_updates']==0)
        lock_proof=dict(selection_sha256=sha(ROOT/'SELECTION.json'),model_lock_sha256=sha(ROOT/'MODEL_LOCK.json'),source_plan_sha256=sha(ROOT/'SOURCE_PLAN.json'))
    checks['all_protected_development_and_workflow_sources_unchanged_during_audit']=all(sha(path)==value for path,value in protected.items())
    return dict(status=('PASS_PRE_DOWNLOAD_AUDIT' if lock_required else 'PASS_PRE_LOCK_AUDIT') if all(checks.values()) else 'FAIL_PROSPECTIVE_AUDIT',scope='METADATA_LOCK_AND_OLD_VAL_ONLY_NO_FRESH_PIXELS',audit_source_sha256=sha(__file__),checks=checks,checks_passed=sum(bool(v) for v in checks.values()),checks_total=len(checks),development_selection_digest=json_digest(expected),old_VAL_preflight_sha256=sha(ROOT/'OLD_VAL_PREFLIGHT.json'),old_VAL_proof=old_proof,independent_source_plan=plan_proof,independent_bootstrap_fixture=stats_proof,code_sha256=expected_code_hashes(),lock=lock_proof,fit_count=0,optimizer_updates=0,new_target_access=False,seconds=time.monotonic()-started)


def save_audit(name,outcome):
    snapshot=ROOT/'audit_sources'/f"audit_{outcome['audit_source_sha256']}.py";snapshot.parent.mkdir(exist_ok=True)
    if snapshot.exists():assert sha(snapshot)==sha(__file__)
    else:shutil.copy2(__file__,snapshot)
    destination=ROOT/name
    if destination.exists():
        old=read(destination);archive=ROOT/f"{destination.stem}_PREVIOUS_{old['audit_source_sha256'][:12]}.json";assert not archive.exists();destination.rename(archive)
    destination.write_text(json.dumps(outcome,indent=2,ensure_ascii=False,default=lambda value:value.item())+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--pre-lock',action='store_true');g.add_argument('--pre-download',action='store_true');args=parser.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):outcome=audit_pre(args.pre_download)
    save_audit('PRE_DOWNLOAD_AUDIT.json' if args.pre_download else 'PRE_LOCK_AUDIT.json',outcome)
    print(json.dumps({k:outcome[k] for k in ['status','checks_passed','checks_total','old_VAL_proof','independent_source_plan','independent_bootstrap_fixture','seconds']},default=lambda value:value.item()),flush=True)
    assert outcome['status'].startswith('PASS')


if __name__=='__main__':main()
