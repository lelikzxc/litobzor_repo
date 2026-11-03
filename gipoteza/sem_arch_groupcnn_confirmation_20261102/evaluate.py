"""One locked fresh-frame evaluation. No training or threshold search."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import argparse
import json
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import confirmation_common as c
from confirmation_stats import analyze


def evaluate(device):
    started = time.monotonic(); config = c.configuration(); plan,selection = c.verify_lock()
    assert not any((c.ROOT/name).exists() for name in ['EVALUATION_FROZEN.json','PREDICTIONS.npz','RESULT.json'])
    data_audit = c.read(c.ROOT/'DATA_AUDIT.json')
    assert data_audit['status'].startswith('PASS'), 'Independent data audit required before prediction'
    if 'checks' in data_audit: assert all(data_audit['checks'].values())
    audited_inputs = data_audit['input_sha256']
    required_audit_inputs = {str((c.ROOT/name).relative_to(c.REPO))
        for name in ['EVAL.npz','DATA_MANIFEST.json','EVAL_INDEX.json']}
    assert required_audit_inputs <= audited_inputs.keys(), 'Audit must bind the evaluated cohort and its manifest/index'
    assert all(c.sha(c.resolve(path)) == expected for path,expected in audited_inputs.items()), 'Passing data audit inputs changed'
    inputs = [c.ROOT/name for name in ['SELECTION.json','MODEL_LOCK.json','PROTOCOL.md','CONFIG.json','SOURCE_PLAN.json',
        'DATA_AUDIT.json','DATA_MANIFEST.json','EVAL_INDEX.json','EVAL.npz']]
    hashes = {str(path.relative_to(c.REPO)):c.sha(path) for path in inputs}
    c.save(c.ROOT/'EVALUATION_FROZEN.json',dict(status='FROZEN_BEFORE_FIRST_FRESH_PREDICTION',input_sha256=hashes,
        code_sha256=c.code_hashes(),bootstrap_repeats=10000,bootstrap_seed=config['bootstrap_seed'],fit_count=0,
        selection_sha256=c.sha(c.ROOT/'SELECTION.json'),device=device))
    index = c.read(c.ROOT/'EVAL_INDEX.json')
    assert all(index['gates'].values()) and index['EVAL_sha256'] == c.sha(c.ROOT/'EVAL.npz')
    with np.load(c.ROOT/'EVAL.npz',allow_pickle=False) as archive:
        x=archive['X'].copy(); target=archive['Y'][:,0].astype(bool)
        truth=archive['label'].copy(); frames=archive['frame'].copy(); assert np.all(archive['role']=='test')
    assert x.shape == (4096,1,12,12) and set(np.unique(frames)) == set(plan['test_frames'])
    assert all((frames==f).sum()==256 for f in plan['test_frames'])
    assert np.bincount(truth,minlength=9).min()>=8 and min(len(np.unique(frames[truth==k])) for k in range(9))>=3
    def progress(rows):
        c.save(c.ROOT/'PARTIAL_RESULTS.json',rows)
        c.save(c.ROOT/'PROGRESS.json',dict(stage='LOCKED_FRESH_PREDICTION',completed=len(rows),total=18,seconds=time.monotonic()-started))
        print(json.dumps(dict(stage='LOCKED_FRESH_PREDICTION',completed=len(rows),total=18,route=rows[-1]['route'],
            budget=rows[-1]['budget'],seed=rows[-1]['seed'])),flush=True)
    arrays,rows = c.predict_models(selection['rows'],x,target,truth,device,progress)
    arrays.update(frames=frames,source_indices=np.asarray(index['local_TEST_indices']),normalized_fixture_x=5*(x[:17]-.2))
    np.savez_compressed(c.ROOT/'PREDICTIONS.npz',**arrays)
    analyses,bootstrap = analyze(arrays['mask_predictions'],arrays['IoU'],truth,frames,config,selection['eligible_by_goal'])
    np.savez_compressed(c.ROOT/'BOOTSTRAP.npz',**bootstrap,modes=np.array(c.MODES),routes=np.array(c.ROUTES),budgets=np.array(c.BUDGETS),seeds=np.array(c.SEEDS))
    assert all(c.sha(c.resolve(path))==expected for path,expected in hashes.items()); c.verify_lock()
    confirmed = {name:goal['confirmed'] for name,goal in analyses['primary_fine']['goals'].items()}
    result = dict(status='PASS_CONFIRMATION' if any(confirmed.values()) else 'FAIL_CONFIRMATION',
        primary_mode='primary_fine',secondary_mode='secondary_seven',analyses=analyses,confirmed_by_goal=confirmed,
        rows=rows,test_examples=4096,test_frames=plan['test_frames'],test_class_counts=np.bincount(truth,minlength=9).tolist(),
        source_model_count=18,evaluated_routes=18,fit_count=0,optimizer_updates=0,bootstrap_repeats=10000,
        bootstrap_seed=config['bootstrap_seed'],CI_quantiles=config['CI_quantiles'],
        estimator='MacroF1 per sampled seed/budget on sampled frames, then mean seeds and budgets; paired crossed indices',
        axes=dict(predictions='mode,route,budget,seed,example',confusion='mode,route,budget,seed,frame,truth,prediction',
            bootstrap_scores='mode,route,budget,draw',bootstrap_deltas='mode,comparison,draw'),
        selection_sha256=c.sha(c.ROOT/'SELECTION.json'),evaluation_frozen_sha256=c.sha(c.ROOT/'EVALUATION_FROZEN.json'),
        predictions_sha256=c.sha(c.ROOT/'PREDICTIONS.npz'),bootstrap_sha256=c.sha(c.ROOT/'BOOTSTRAP.npz'),
        seconds=time.monotonic()-started,scope='Fixed new fields of the same IC; known architecture; no external-chip or novelty claim')
    c.save(c.ROOT/'RESULT.json',result)
    c.save(c.ROOT/'PROGRESS.json',dict(status=result['status'],completed=18,total=18,seconds=result['seconds']))
    print(json.dumps(dict(status=result['status'],confirmed_by_goal=confirmed,analyses=analyses)),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',action='store_true',required=True)
    parser.add_argument('--device',choices=['cpu','mps'],default='mps'); args=parser.parse_args(); torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        try: evaluate(args.device)
        except BaseException as error:
            c.save(c.ROOT/'EVALUATION_FAILURE.json',dict(status='FAILED_EVALUATION',error_type=type(error).__name__,
                error=str(error),traceback=traceback.format_exc()))
            raise


if __name__ == '__main__': main()
