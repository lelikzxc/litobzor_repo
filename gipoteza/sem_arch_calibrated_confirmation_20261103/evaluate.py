"""One prospective evaluation of fixed CNNs and fixed final scalar thresholds."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import argparse
import json
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import common as c
from stats import analyze


def evaluate(device):
    started = time.monotonic(); settings = c.configuration(); plan, selection = c.verify_lock()
    assert not any((c.ROOT / name).exists() for name in ['EVALUATION_FROZEN.json', 'PREDICTIONS.npz', 'RESULT.json'])
    if device == 'mps': assert torch.backends.mps.is_available() and os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] == '0'
    audit = c.read(c.ROOT / 'DATA_AUDIT.json')
    assert audit['status'].startswith('PASS') and all(audit['checks'].values()), 'Independent native-data audit required'
    required = {c.relative(c.ROOT / name) for name in ['EVAL.npz', 'DATA_MANIFEST.json', 'EVAL_INDEX.json']}
    assert required <= audit['input_sha256'].keys()
    assert all(c.sha(c.resolve(name)) == expected for name, expected in audit['input_sha256'].items()), 'Audited data inputs changed'
    inputs = [c.ROOT / name for name in ['SELECTION.json', 'MODEL_LOCK.json', 'PROTOCOL.md', 'CONFIG.json',
        'SOURCE_PLAN.json', 'FINAL_CALIBRATION.json', 'FINAL_CALIBRATION.npz',
        'DATA_AUDIT.json', 'DATA_MANIFEST.json', 'EVAL_INDEX.json', 'EVAL.npz']]
    hashes = {c.relative(path): c.sha(path) for path in inputs}
    c.save(c.ROOT / 'EVALUATION_FROZEN.json', dict(status='FROZEN_BEFORE_FIRST_FRESH_PREDICTION',
        input_sha256=hashes, code_sha256=c.code_hashes(), fit_count=0, optimizer_updates=0,
        bootstrap_repeats=settings['bootstrap_repeats'], bootstrap_seed=settings['bootstrap_seed'], device=device))
    index = c.read(c.ROOT / 'EVAL_INDEX.json')
    assert all(index['gates'].values()) and index['EVAL_sha256'] == c.sha(c.ROOT / 'EVAL.npz')
    with np.load(c.ROOT / 'EVAL.npz', allow_pickle=False) as z:
        x, target, truth, frames = z['X'].copy(), z['Y'][:, 0].astype(bool), z['label'].copy(), z['frame'].copy()
        assert np.all(z['role'] == 'test')
    assert x.shape == (6144, 1, 12, 12) and set(np.unique(frames)) == set(plan['test_frames'])
    assert all((frames == frame).sum() == 256 for frame in plan['test_frames'])
    assert np.bincount(truth, minlength=9).min() >= 8
    assert min(len(np.unique(frames[truth == value])) for value in range(9)) >= 3

    def progress(rows):
        c.save(c.ROOT / 'PARTIAL_RESULTS.json', rows)
        value = dict(stage='LOCKED_FRESH_PREDICTION', completed=len(rows), total=9,
            architecture=rows[-1]['architecture'], seed=rows[-1]['seed'], seconds=time.monotonic() - started)
        c.save(c.ROOT / 'PROGRESS.json', value); print(json.dumps(value), flush=True)

    arrays, rows = c.predict_models(selection['rows'], x, target, truth, device, progress)
    arrays.update(frames=frames, source_indices=np.asarray(index['local_TEST_indices']), normalized_fixture_x=5 * (x[:17] - .2))
    np.savez_compressed(c.ROOT / 'PREDICTIONS.npz', **arrays)
    analyses, bootstrap = analyze(arrays['mask_predictions'], arrays['IoU'], arrays['probability'],
        truth, target, frames, settings, selection['development_eligible'])
    np.savez_compressed(c.ROOT / 'BOOTSTRAP.npz', **bootstrap,
        modes=np.array(c.MODES), architectures=np.array(c.ARCHITECTURES), seeds=np.array(c.SEEDS))
    assert all(c.sha(c.resolve(name)) == expected for name, expected in hashes.items()); c.verify_lock()
    confirmed = bool(analyses['final_calibrated']['primary_goal']['confirmed'])
    result = dict(status='PASS_CONFIRMATION' if confirmed else 'FAIL_CONFIRMATION', confirmed=confirmed,
        primary_mode='final_calibrated', secondary_mode='fixed05', analyses=analyses, rows=rows,
        test_examples=6144, test_frames=plan['test_frames'], test_class_counts=np.bincount(truth, minlength=9).tolist(),
        source_model_count=9, evaluated_routes=9, fit_count=0, optimizer_updates=0, calibration_examples=512,
        bootstrap_repeats=settings['bootstrap_repeats'], bootstrap_seed=settings['bootstrap_seed'],
        CI_quantiles=settings['CI_quantiles'],
        selection_sha256=c.sha(c.ROOT / 'SELECTION.json'),
        evaluation_frozen_sha256=c.sha(c.ROOT / 'EVALUATION_FROZEN.json'),
        final_calibration_sha256=c.sha(c.ROOT / 'FINAL_CALIBRATION.json'),
        predictions_sha256=c.sha(c.ROOT / 'PREDICTIONS.npz'), bootstrap_sha256=c.sha(c.ROOT / 'BOOTSTRAP.npz'),
        axes=dict(predictions='mode,architecture,seed,example', confusion='mode,architecture,seed,frame,truth,prediction',
            per_frame_AP='architecture,seed,frame', bootstrap_F1='primary_architecture,draw',
            bootstrap_AP='primary_architecture,draw', delta_F1='draw', delta_AP='draw'),
        estimator='Paired crossed seed/frame bootstrap; per-seed frame-aggregated MacroF1, frame AP means; then seed mean',
        seconds=time.monotonic() - started,
        scope='New frames of one IC; fixed final calibration; both AP/F1 primary; narrow/fixed0.5 descriptive only')
    c.save(c.ROOT / 'RESULT.json', result)
    c.save(c.ROOT / 'PROGRESS.json', dict(status=result['status'], completed=9, total=9, seconds=result['seconds']))
    print(json.dumps(dict(status=result['status'], confirmed=confirmed, analyses=analyses)), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', action='store_true', required=True)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps'); args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        try: evaluate(args.device)
        except BaseException as error:
            c.save(c.ROOT / 'EVALUATION_FAILURE.json', dict(status='TECHNICAL_EVALUATION_FAILURE',
                error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
            raise


if __name__ == '__main__': main()
