"""Final weak-label calibration, OLD VAL technical proof, then prospective lock."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import argparse
import datetime
import json
import shutil

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import common as c


def save_calibration():
    metadata, arrays = c.final_calibration()
    json_file, array_file = c.ROOT / 'FINAL_CALIBRATION.json', c.ROOT / 'FINAL_CALIBRATION.npz'
    if json_file.exists() or array_file.exists():
        assert json_file.exists() and array_file.exists() and c.read(json_file) == metadata
        with np.load(array_file, allow_pickle=False) as z:
            assert set(z.files) == set(arrays) and all(np.array_equal(z[name], value) for name, value in arrays.items())
    else:
        c.save(json_file, metadata); np.savez_compressed(array_file, **arrays)
    return metadata, arrays


def synthetic_preflight(config):
    from stats import analyze
    truth = np.tile(np.arange(9), 24); frames = np.repeat(np.arange(24), 9)
    checker = (np.indices((12, 12)).sum(0) % 2).astype(bool)
    target = np.broadcast_to(checker, (len(truth), 12, 12)).copy()
    probability = np.broadcast_to(target, (3, 3, *target.shape)).astype(np.float32).copy()
    predictions = np.broadcast_to(truth, (2, 3, 3, len(truth))).copy()
    ious = np.ones_like(predictions, dtype=np.float64)
    equal, equal_arrays = analyze(predictions, ious, probability, truth, target, frames, config, True)
    assert not equal['final_calibrated']['primary_goal']['confirmed']
    assert equal['final_calibrated']['primary_goal']['CI97_5_F1'] == [0., 0.]
    assert equal['final_calibrated']['primary_goal']['CI97_5_AP'] == [0., 0.]
    predictions[:, 1] = (predictions[:, 1] + 1) % 9
    f1_only, ignored = analyze(predictions, ious, probability, truth, target, frames, config, True)
    assert not f1_only['final_calibrated']['primary_goal']['confirmed']
    probability[1] = 1 - probability[1]
    ap_only, ignored = analyze(np.broadcast_to(truth, predictions.shape).copy(), ious,
        probability, truth, target, frames, config, True)
    assert not ap_only['final_calibrated']['primary_goal']['confirmed']
    separated, arrays = analyze(predictions, ious, probability, truth, target, frames, config, True)
    assert separated['final_calibrated']['primary_goal']['confirmed']
    assert separated['final_calibrated']['primary_goal']['CI97_5_F1'] == [1., 1.]
    assert separated['final_calibrated']['primary_goal']['CI97_5_AP'][0] > 0
    ineligible, ignored = analyze(predictions, ious, probability, truth, target, frames, config, False)
    assert not ineligible['final_calibrated']['primary_goal']['confirmed']
    for result in [equal, f1_only, ap_only, separated, ineligible]:
        secondary = result['fixed05']['primary_goal']
        assert secondary['confirmed'] is False
        assert not any(key.startswith('CI') for key in secondary)
        assert not any(any(key.startswith('CI') for key in row) for row in result['final_calibrated']['contrasts'].values())
    assert np.array_equal(arrays['seed_ids'], equal_arrays['seed_ids'])
    assert np.array_equal(arrays['frame_ids'], equal_arrays['frame_ids'])
    return dict(equal=equal, f1_only=f1_only, AP_only=ap_only, separated=separated, ineligible=ineligible, synthetic_only=True,
                optimizer_updates=0), arrays


def preflight(device):
    config = c.configuration()
    assert not any((c.ROOT / name).exists() for name in ['sources', 'TEST.npz', 'EVAL.npz', 'SELECTION.json'])
    calibration, calibration_arrays = save_calibration()
    development = c.development_selection()
    with np.load(c.PILOT / 'TRAIN_VAL.npz', allow_pickle=False) as data:
        ids = np.flatnonzero(data['role'] == 'val')[:17]
        x, target, truth = data['X'][ids], data['Y'][ids, 0].astype(bool), data['label'][ids]
    arrays, metrics = c.predict_models(development['rows'], x, target, truth, device)
    checks = dict(all_nine_final_models=len(metrics) == 9,
                  nonmultiple_batch=len(ids) == 17 and 17 % config['inference_batch_size'] != 0)
    agreements = []
    for row in development['rows']:
        a, s = c.ARCHITECTURES.index(row['architecture']), c.SEEDS.index(row['seed'])
        with np.load(c.resolve(row['source_VAL_file']), allow_pickle=False) as source:
            original = source['probability'][:17, 0]
            difference = float(np.max(np.abs(original - arrays['probability'][a, s])))
            assert np.array_equal(source['truth'][:17], truth) and np.array_equal(source['target'][:17], target)
            classes, masks = [], []
            for m in range(2):
                index = int(np.flatnonzero(source['thresholds'] == arrays['thresholds'][m, a, s])[0])
                classes.append(bool(np.array_equal(source['predictions'][index, :17], arrays['mask_predictions'][m, a, s])))
                masks.append(bool(np.array_equal(source['packed_masks'][index, :17], arrays['packed_masks'][m, a, s])))
        agreements.append(dict(architecture=row['architecture'], seed=row['seed'],
            maximum_probability_error=difference, class_parity=classes, mask_parity=masks,
            source_VAL_file=row['source_VAL_file'], source_VAL_sha256=row['source_VAL_sha256']))
    checks['archived_probabilities_match'] = all(row['maximum_probability_error'] <= config['old_VAL_probability_tolerance'] for row in agreements)
    checks['both_thresholds_class_and_mask_parity'] = all(all(row['class_parity']) and all(row['mask_parity']) for row in agreements)
    checks['final_calibration_512_shared_class_labels'] = len(calibration_arrays['data_indices']) == 512 and calibration['labels_shared_by_all_models']
    synthetic, bootstrap = synthetic_preflight(config)
    checks['synthetic_two_outcome_confirmation_and_secondary_no_rescue'] = True
    arrays.update(OLD_VAL_dataset_indices=ids, normalized_fixture_x=5 * (x - .2),
        synthetic_seed_ids=bootstrap['seed_ids'], synthetic_frame_ids=bootstrap['frame_ids'],
        synthetic_delta_F1=bootstrap['delta_F1'], synthetic_delta_AP=bootstrap['delta_AP'])
    path = c.ROOT / 'OLD_VAL_PREFLIGHT.npz'; np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        checks['serialization_roundtrip'] = all(np.array_equal(saved[name], value) for name, value in arrays.items())
    result = dict(status='PASS_OLD_VAL_PREFLIGHT' if all(checks.values()) else 'FAIL_OLD_VAL_PREFLIGHT',
        checks=checks, rows=metrics, agreements=agreements, synthetic=synthetic, examples=17,
        source_model_count=9, device=device, fit_count=0, optimizer_updates=0,
        final_calibration_sha256=c.sha(c.ROOT / 'FINAL_CALIBRATION.json'),
        final_calibration_arrays_sha256=c.sha(c.ROOT / 'FINAL_CALIBRATION.npz'),
        development_selection_digest=c.digest(development), code_sha256=c.code_hashes(),
        protocol_sha256=c.sha(c.ROOT / 'PROTOCOL.md'), config_sha256=c.sha(c.ROOT / 'CONFIG.json'),
        arrays_sha256=c.sha(path), TRAIN_VAL_sha256=c.sha(c.PILOT / 'TRAIN_VAL.npz'),
        new_target_access=False, scope='OLD_VAL_PARITY_AND_OLD_DEV_FINAL_CALIBRATION_ONLY')
    c.save(c.ROOT / 'OLD_VAL_PREFLIGHT.json', result)
    print(json.dumps(dict(status=result['status'], checks=checks, source_model_count=9, examples=17)), flush=True)
    assert all(checks.values()), 'Final calibration technical preflight failed'


def lock():
    config = c.configuration(); development = c.development_selection()
    pre = c.read(c.ROOT / 'OLD_VAL_PREFLIGHT.json')
    assert not any((c.ROOT / name).exists() for name in ['sources', 'TEST.npz', 'EVAL.npz', 'PREDICTIONS.npz',
        'MODEL_LOCK.json', 'SELECTION.json', 'SOURCE_PLAN.json'])
    assert pre['status'] == 'PASS_OLD_VAL_PREFLIGHT' and all(pre['checks'].values())
    assert pre['development_selection_digest'] == c.digest(development)
    assert pre['code_sha256'] == c.code_hashes() and pre['arrays_sha256'] == c.sha(c.ROOT / 'OLD_VAL_PREFLIGHT.npz')
    calibration, arrays = c.final_calibration()
    assert c.read(c.ROOT / 'FINAL_CALIBRATION.json') == calibration
    plan = c.build_source_plan(); c.save(c.ROOT / 'SOURCE_PLAN.json', plan)
    model_lock = dict(status='LOCKED_FINAL_MODELS_AND_FINAL_CALIBRATION', rows=development['rows'],
        source_model_count=9, deployment_artifact_count=3, source_plan_sha256=c.sha(c.ROOT / 'SOURCE_PLAN.json'),
        development=development, final_calibration_sha256=c.sha(c.ROOT / 'FINAL_CALIBRATION.json'),
        final_calibration_arrays_sha256=c.sha(c.ROOT / 'FINAL_CALIBRATION.npz'),
        protocol_sha256=c.sha(c.ROOT / 'PROTOCOL.md'), config_sha256=c.sha(c.ROOT / 'CONFIG.json'))
    c.save(c.ROOT / 'MODEL_LOCK.json', model_lock)
    selection = dict(status='LOCKED_BEFORE_FRESH_DOWNLOAD', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rows=development['rows'], development=development, development_eligible=True,
        source_model_count=9, deployment_artifact_count=3, architectures=c.ARCHITECTURES, seeds=c.SEEDS,
        modes=c.MODES, budget=27, calibration_examples=512,
        source_plan_sha256=c.sha(c.ROOT / 'SOURCE_PLAN.json'), model_lock_sha256=c.sha(c.ROOT / 'MODEL_LOCK.json'),
        final_calibration_sha256=c.sha(c.ROOT / 'FINAL_CALIBRATION.json'),
        final_calibration_arrays_sha256=c.sha(c.ROOT / 'FINAL_CALIBRATION.npz'),
        protocol_sha256=c.sha(c.ROOT / 'PROTOCOL.md'), config_sha256=c.sha(c.ROOT / 'CONFIG.json'),
        code_sha256=c.code_hashes(), old_VAL_preflight_sha256=c.sha(c.ROOT / 'OLD_VAL_PREFLIGHT.json'),
        old_VAL_preflight_arrays_sha256=c.sha(c.ROOT / 'OLD_VAL_PREFLIGHT.npz'),
        bootstrap_repeats=config['bootstrap_repeats'], bootstrap_seed=config['bootstrap_seed'],
        CI_quantiles=config['CI_quantiles'], fresh_images_present=False, fit_count=0, optimizer_updates=0)
    c.save(c.ROOT / 'SELECTION.json', selection); c.verify_lock(require_fresh_data_absent=True)
    snapshot = c.ROOT / 'source_snapshot'; snapshot.mkdir(exist_ok=False)
    for name in ['common.py', 'selection.py', 'evaluate.py', 'prepare.py', 'stats.py', 'CONFIG.json', 'PROTOCOL.md']:
        shutil.copy2(c.ROOT / name, snapshot / name)
    print(json.dumps(dict(status=selection['status'], source_model_count=9,
        selection_sha256=c.sha(c.ROOT / 'SELECTION.json'), test_frames=plan['test_frames'],
        final_thresholds=calibration['selected_thresholds'])), flush=True)


def main():
    parser = argparse.ArgumentParser(); action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preflight', action='store_true'); action.add_argument('--lock', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    args = parser.parse_args(); torch.set_num_threads(2)
    with threadpool_limits(limits=2): preflight(args.device) if args.preflight else lock()


if __name__ == '__main__': main()
