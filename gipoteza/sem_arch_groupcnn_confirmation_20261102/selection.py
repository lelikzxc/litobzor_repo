"""OLD VAL technical dry run and prospective model/data/code lock. No downloads."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import argparse
import datetime
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import confirmation_common as c
from confirmation_stats import analyze


def synthetic_preflight(config):
    truth = np.tile(np.arange(9),16); frames = np.repeat(np.arange(16),9)
    perfect = np.broadcast_to(truth,(2,3,2,3,len(truth))).copy()
    iou = np.ones_like(perfect,dtype=float)
    eligible = {name:True for name in config['goals']}
    equal, arrays = analyze(perfect,iou,truth,frames,config,eligible)
    assert not equal['primary_fine']['goals']['A_matched_quality']['passed']
    assert equal['primary_fine']['goals']['B_full_efficiency']['passed']
    perfect[:,1] = (perfect[:,1]+1)%9
    separated, second = analyze(perfect,iou,truth,frames,config,eligible)
    assert all(v['passed'] for v in separated['primary_fine']['goals'].values())
    assert np.array_equal(arrays['seed_ids'],second['seed_ids']) and np.array_equal(arrays['frame_ids'],second['frame_ids'])
    return dict(equal_predictions=equal,perfect_group_wrong_narrow=separated,
        paired_seed_draw_digest=c.digest(arrays['seed_ids'].tolist()),paired_frame_draw_digest=c.digest(arrays['frame_ids'].tolist()),
        repeats=config['bootstrap_repeats'],synthetic_only=True), second


def preflight(device):
    config = c.configuration(); development = c.development_selection()
    assert not any((c.ROOT/name).exists() for name in ['sources','TEST.npz','EVAL.npz','SELECTION.json'])
    data_file = c.PILOT/'TRAIN_VAL.npz'
    with np.load(data_file,allow_pickle=False) as data:
        ids = np.flatnonzero(data['role'] == 'val')[:17]
        x = data['X'][ids]; target = data['Y'][ids,0].astype(bool); truth = data['label'][ids]
    assert len(ids) == 17
    arrays, metrics = c.predict_models(development['rows'],x,target,truth,device)
    checks = dict(all18_final_routes=len(metrics)==18,nonmultiple_batch=17%config['inference_batch_size']!=0)
    agreements = []
    for row in development['rows']:
        r,b,s = c.ROUTES.index(row['route']),c.BUDGETS.index(row['budget']),c.SEEDS.index(row['seed'])
        with np.load(c.resolve(row['source_VAL_file']),allow_pickle=False) as source:
            probability = source['probability'][:17,0]
            delta = float(np.max(np.abs(probability-arrays['probability'][r,b,s])))
            assert np.array_equal(truth,source['truth'][:17]) and np.array_equal(target,source['target'][:17])
            class_parity = []
            for m,(grid,predkey) in enumerate([('thresholds','predictions'),('secondary_thresholds','secondary_predictions')]):
                index = int(np.flatnonzero(source[grid] == arrays['thresholds'][m,r,b,s])[0])
                class_parity.append(bool(np.array_equal(source[predkey][index,:17],arrays['mask_predictions'][m,r,b,s])))
        agreements.append(dict(route=row['route'],budget=row['budget'],seed=row['seed'],
            maximum_probability_error=delta,selected_class_parity=class_parity,
            source_VAL_file=row['source_VAL_file'],source_VAL_sha256=row['source_VAL_sha256']))
    checks['source_probability_agreement'] = all(v['maximum_probability_error'] <= config['old_VAL_probability_tolerance'] for v in agreements)
    checks['source_selected_class_agreement'] = all(all(v['selected_class_parity']) for v in agreements)
    synthetic, bootstrap = synthetic_preflight(config)
    arrays.update(OLD_VAL_dataset_indices=ids,normalized_fixture_x=5*(x-.2),
        synthetic_seed_ids=bootstrap['seed_ids'],synthetic_frame_ids=bootstrap['frame_ids'],synthetic_deltas=bootstrap['deltas'])
    path = c.ROOT/'OLD_VAL_PREFLIGHT.npz'; np.savez_compressed(path,**arrays)
    with np.load(path,allow_pickle=False) as saved:
        checks['prediction_serialization_roundtrip'] = all(np.array_equal(saved[name],value) for name,value in arrays.items())
    checks['synthetic_estimator_and_goals'] = True
    result = dict(status='PASS_OLD_VAL_PREFLIGHT' if all(checks.values()) else 'FAIL_OLD_VAL_PREFLIGHT',
        scope='OLD_VAL_PREFLIGHT: technical consistency only; not new confirmation or model selection',device=device,
        checks=checks,rows=metrics,agreements=agreements,synthetic=synthetic,examples=17,source_model_count=18,
        inference_batch_size=config['inference_batch_size'],TRAIN_VAL_sha256=c.sha(data_file),
        development_selection_digest=c.digest(development),code_sha256=c.code_hashes(),
        protocol_sha256=c.sha(c.ROOT/'PROTOCOL.md'),config_sha256=c.sha(c.ROOT/'CONFIG.json'),
        arrays_sha256=c.sha(path),new_target_access=False,fit_count=0)
    c.save(c.ROOT/'OLD_VAL_PREFLIGHT.json',result)
    print(json.dumps(dict(status=result['status'],checks=checks,examples=17,source_model_count=18)),flush=True)
    assert all(checks.values()), 'OLD VAL technical dry run failed before fresh data access'


def lock():
    config = c.configuration(); development = c.development_selection(); preflight = c.read(c.ROOT/'OLD_VAL_PREFLIGHT.json')
    assert not any((c.ROOT/name).exists() for name in ['sources','TEST.npz','EVAL.npz','PREDICTIONS.npz','MODEL_LOCK.json','SELECTION.json','SOURCE_PLAN.json'])
    assert preflight['status'] == 'PASS_OLD_VAL_PREFLIGHT' and all(preflight['checks'].values())
    assert preflight['development_selection_digest'] == c.digest(development)
    assert preflight['code_sha256'] == c.code_hashes() and preflight['arrays_sha256'] == c.sha(c.ROOT/'OLD_VAL_PREFLIGHT.npz')
    plan = c.build_source_plan(); c.save(c.ROOT/'SOURCE_PLAN.json',plan)
    model_lock = dict(status='LOCKED_FINAL_MODELS_AND_OLD_VAL_THRESHOLDS',rows=development['rows'],
        source_model_count=18,deployment_artifact_count=6,source_plan_sha256=c.sha(c.ROOT/'SOURCE_PLAN.json'),
        protocol_sha256=c.sha(c.ROOT/'PROTOCOL.md'),config_sha256=c.sha(c.ROOT/'CONFIG.json'),development=development)
    c.save(c.ROOT/'MODEL_LOCK.json',model_lock)
    selection = dict(status='LOCKED_BEFORE_FRESH_DOWNLOAD',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rows=development['rows'],development=development,eligible_by_goal=development['eligible_by_goal'],
        source_model_count=18,deployment_artifact_count=6,routes=c.ROUTES,budgets=c.BUDGETS,seeds=c.SEEDS,modes=c.MODES,
        source_plan_sha256=c.sha(c.ROOT/'SOURCE_PLAN.json'),model_lock_sha256=c.sha(c.ROOT/'MODEL_LOCK.json'),
        protocol_sha256=c.sha(c.ROOT/'PROTOCOL.md'),config_sha256=c.sha(c.ROOT/'CONFIG.json'),code_sha256=c.code_hashes(),
        old_VAL_preflight_sha256=c.sha(c.ROOT/'OLD_VAL_PREFLIGHT.json'),old_VAL_preflight_arrays_sha256=c.sha(c.ROOT/'OLD_VAL_PREFLIGHT.npz'),
        bootstrap_repeats=10000,bootstrap_seed=config['bootstrap_seed'],CI_quantiles=config['CI_quantiles'],goals=config['goals'],
        fresh_images_present=False,fit_count=0,optimizer_updates=0)
    c.save(c.ROOT/'SELECTION.json',selection); c.verify_lock(require_fresh_data_absent=True)
    print(json.dumps(dict(status=selection['status'],test_frames=plan['test_frames'],models=18,threshold_rows=18,
        selection_sha256=c.sha(c.ROOT/'SELECTION.json'),eligible_by_goal=selection['eligible_by_goal'])),flush=True)


def main():
    parser = argparse.ArgumentParser(); actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--preflight',action='store_true'); actions.add_argument('--lock',action='store_true')
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu'); args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2): preflight(args.device) if args.preflight else lock()


if __name__ == '__main__': main()
