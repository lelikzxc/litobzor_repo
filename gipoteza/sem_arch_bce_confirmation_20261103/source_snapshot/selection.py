"""Genuine OLD VAL dry run followed by a prospective, no-training lock."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import argparse
import datetime
import json

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import common as c


def synthetic_preflight(config):
    from stats import analyze
    truth=np.tile(np.arange(9),24);frames=np.repeat(np.arange(24),9)
    predictions=np.broadcast_to(truth,(2,4,2,3,len(truth))).copy()
    ious=np.ones_like(predictions,dtype=float)
    equal,equal_arrays=analyze(predictions,ious,truth,frames,config,True)
    assert not equal['primary_fine']['primary_goal']['confirmed']
    assert equal['primary_fine']['primary_goal']['CI95']==[0.,0.]
    predictions[:,1]=(predictions[:,1]+1)%9
    separated,arrays=analyze(predictions,ious,truth,frames,config,True)
    assert separated['primary_fine']['primary_goal']['confirmed']
    assert separated['primary_fine']['primary_goal']['CI95']==[1.,1.]
    ineligible,ineligible_arrays=analyze(predictions,ious,truth,frames,config,False)
    assert not ineligible['primary_fine']['primary_goal']['confirmed']
    assert not ineligible['primary_fine']['primary_goal']['guards']['development_eligible']
    for result in [equal,separated,ineligible]:
        assert result['fixed05']['primary_goal']['confirmed'] is False
        assert 'CI95' not in result['fixed05']['primary_goal']
        assert all('CI95' not in row and 'pvalue' not in row for row in result['primary_fine']['contrasts'].values())
    assert np.array_equal(arrays['seed_ids'],equal_arrays['seed_ids'])
    assert np.array_equal(arrays['frame_ids'],equal_arrays['frame_ids'])
    assert np.array_equal(arrays['seed_ids'],ineligible_arrays['seed_ids'])
    return dict(equal_predictions=equal,perfect_group_wrong_plain=separated,ineligible=ineligible,
        synthetic_only=True,optimizer_updates=0,repeats=config['bootstrap_repeats']),arrays


def preflight(device):
    config = c.configuration(); development = c.development_selection()
    assert not any((c.ROOT/name).exists() for name in ['sources','TEST.npz','EVAL.npz','SELECTION.json'])
    data_file = c.PILOT/'TRAIN_VAL.npz'
    with np.load(data_file,allow_pickle=False) as data:
        ids = np.flatnonzero(data['role']=='val')[:17]
        x, target, truth = data['X'][ids],data['Y'][ids,0].astype(bool),data['label'][ids]
    arrays,metrics = c.predict_models(development['rows'],x,target,truth,device)
    checks = dict(all24_final_cells=len(metrics)==24,nonmultiple_batch=len(ids)==17 and 17%config['inference_batch_size']!=0)
    agreements = []
    for row in development['rows']:
        r,b,s = c.CELLS.index(row['cell']),c.BUDGETS.index(row['budget']),c.SEEDS.index(row['seed'])
        with np.load(c.resolve(row['source_VAL_file']),allow_pickle=False) as source:
            probability = source['probability'][:17,0]
            delta = float(np.max(np.abs(probability-arrays['probability'][r,b,s])))
            assert np.array_equal(truth,source['truth'][:17]) and np.array_equal(target,source['target'][:17])
            parity,mask_parity = [],[]
            for m in range(2):
                index = int(np.flatnonzero(source['thresholds']==arrays['thresholds'][m,r,b,s])[0])
                parity.append(bool(np.array_equal(source['predictions'][index,:17],arrays['mask_predictions'][m,r,b,s])))
                mask_parity.append(bool(np.array_equal(source['packed_masks'][index,:17],arrays['packed_masks'][m,r,b,s])))
        agreements.append(dict(cell=row['cell'],budget=row['budget'],seed=row['seed'],
            maximum_probability_error=delta,selected_class_parity=parity,selected_mask_parity=mask_parity,
            source_VAL_file=row['source_VAL_file'],source_VAL_sha256=row['source_VAL_sha256']))
    checks['source_probability_agreement'] = all(v['maximum_probability_error']<=config['old_VAL_probability_tolerance'] for v in agreements)
    checks['source_selected_class_agreement'] = all(all(v['selected_class_parity']) for v in agreements)
    checks['source_selected_mask_agreement'] = all(all(v['selected_mask_parity']) for v in agreements)
    synthetic,bootstrap=synthetic_preflight(config)
    checks['synthetic_primary_goal_eligibility_and_secondary_no_rescue']=True
    arrays.update(OLD_VAL_dataset_indices=ids,normalized_fixture_x=5*(x-.2),
        synthetic_seed_ids=bootstrap['seed_ids'],synthetic_frame_ids=bootstrap['frame_ids'],
        synthetic_delta_F1=bootstrap['delta_F1'])
    path = c.ROOT/'OLD_VAL_PREFLIGHT.npz'; np.savez_compressed(path,**arrays)
    with np.load(path,allow_pickle=False) as stored:
        checks['prediction_serialization_roundtrip'] = all(np.array_equal(stored[name],value) for name,value in arrays.items())
    result = dict(status='PASS_OLD_VAL_PREFLIGHT' if all(checks.values()) else 'FAIL_OLD_VAL_PREFLIGHT',
        scope='Technical OLD VAL parity only; no target-image access or model selection',device=device,
        checks=checks,rows=metrics,agreements=agreements,synthetic=synthetic,examples=17,source_model_count=24,
        inference_batch_size=config['inference_batch_size'],TRAIN_VAL_sha256=c.sha(data_file),
        development_selection_digest=c.digest(development),code_sha256=c.code_hashes(),
        protocol_sha256=c.sha(c.ROOT/'PROTOCOL.md'),config_sha256=c.sha(c.ROOT/'CONFIG.json'),
        arrays_sha256=c.sha(path),new_target_access=False,fit_count=0,optimizer_updates=0)
    c.save(c.ROOT/'OLD_VAL_PREFLIGHT.json',result)
    print(json.dumps(dict(status=result['status'],checks=checks,examples=17,source_model_count=24)),flush=True)
    assert all(checks.values()),'Technical OLD VAL parity failed before new image access'


def lock():
    config = c.configuration(); development = c.development_selection(); pre = c.read(c.ROOT/'OLD_VAL_PREFLIGHT.json')
    assert not any((c.ROOT/name).exists() for name in ['sources','TEST.npz','EVAL.npz','PREDICTIONS.npz','MODEL_LOCK.json','SELECTION.json','SOURCE_PLAN.json'])
    assert pre['status']=='PASS_OLD_VAL_PREFLIGHT' and all(pre['checks'].values())
    assert pre['development_selection_digest']==c.digest(development)
    assert pre['code_sha256']==c.code_hashes() and pre['arrays_sha256']==c.sha(c.ROOT/'OLD_VAL_PREFLIGHT.npz')
    plan = c.build_source_plan();c.save(c.ROOT/'SOURCE_PLAN.json',plan)
    model_lock = dict(status='LOCKED_FINAL_MODELS_AND_OLD_VAL_THRESHOLDS',rows=development['rows'],
        source_model_count=24,deployment_artifact_count=12,source_plan_sha256=c.sha(c.ROOT/'SOURCE_PLAN.json'),
        protocol_sha256=c.sha(c.ROOT/'PROTOCOL.md'),config_sha256=c.sha(c.ROOT/'CONFIG.json'),development=development)
    c.save(c.ROOT/'MODEL_LOCK.json',model_lock)
    selection = dict(status='LOCKED_BEFORE_FRESH_DOWNLOAD',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rows=development['rows'],development=development,development_eligible=True,
        source_model_count=24,deployment_artifact_count=12,cells=c.CELLS,budgets=c.BUDGETS,seeds=c.SEEDS,modes=c.MODES,
        source_plan_sha256=c.sha(c.ROOT/'SOURCE_PLAN.json'),model_lock_sha256=c.sha(c.ROOT/'MODEL_LOCK.json'),
        protocol_sha256=c.sha(c.ROOT/'PROTOCOL.md'),config_sha256=c.sha(c.ROOT/'CONFIG.json'),code_sha256=c.code_hashes(),
        old_VAL_preflight_sha256=c.sha(c.ROOT/'OLD_VAL_PREFLIGHT.json'),old_VAL_preflight_arrays_sha256=c.sha(c.ROOT/'OLD_VAL_PREFLIGHT.npz'),
        bootstrap_repeats=config['bootstrap_repeats'],bootstrap_seed=config['bootstrap_seed'],CI_quantiles=config['CI_quantiles'],
        fresh_images_present=False,fit_count=0,optimizer_updates=0)
    c.save(c.ROOT/'SELECTION.json',selection);c.verify_lock(require_fresh_data_absent=True)
    snapshot = c.ROOT/'source_snapshot';snapshot.mkdir(exist_ok=False)
    for name in ['common.py','selection.py','evaluate.py','prepare.py','stats.py','CONFIG.json','PROTOCOL.md']:
        (snapshot/name).write_bytes((c.ROOT/name).read_bytes())
    print(json.dumps(dict(status=selection['status'],test_frames=plan['test_frames'],models=24,
        selection_sha256=c.sha(c.ROOT/'SELECTION.json'))),flush=True)


def main():
    parser=argparse.ArgumentParser();actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--preflight',action='store_true');actions.add_argument('--lock',action='store_true')
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu');args=parser.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):preflight(args.device) if args.preflight else lock()


if __name__=='__main__':main()
