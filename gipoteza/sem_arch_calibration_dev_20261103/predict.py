"""Freeze and infer nine unchanged models on previously evaluated DEV cohorts."""
import argparse
import datetime
import json
import os
import shutil
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import common as c


def preflight(device):
    config = c.config(); rows = c.model_descriptors()
    assert not any((c.ROOT / name).exists() for name in ['LOCK.json', 'PREDICTIONS.npz', 'INFERENCE.json'])
    with np.load(c.PILOT / 'TRAIN_VAL.npz', allow_pickle=False) as z:
        ids = np.flatnonzero(z['role'] == 'val')[:17]
        x, target, truth = z['X'][ids], z['Y'][ids, 0].astype(bool), z['label'][ids]
    probability = c.predict(rows, x, device)
    masks = probability >= .5
    classes = np.empty((3, 3, 17), np.int64)
    agreements = []
    for row in rows:
        a, s = c.ARCHITECTURES.index(row['architecture']), c.SEEDS.index(row['seed'])
        classes[a, s] = c.hard_classes(masks[a, s])
        with np.load(c.resolve(row['source_VAL_file']), allow_pickle=False) as z:
            original = z['probability'][:17, 0]
            at = int(np.flatnonzero(z['thresholds'] == .5)[0])
            difference = float(np.max(np.abs(original - probability[a, s])))
            mask_parity = bool(np.array_equal(np.packbits(masks[a, s].reshape(17, 144), axis=1), z['packed_masks'][at, :17]))
            class_parity = bool(np.array_equal(classes[a, s], z['predictions'][at, :17]))
            assert np.array_equal(truth, z['truth'][:17]) and np.array_equal(target, z['target'][:17])
        agreements.append(dict(architecture=row['architecture'], seed=row['seed'],
            maximum_probability_error=difference, mask_parity=mask_parity, class_parity=class_parity))
    arrays = dict(probability=probability, masks=masks, classes=classes, target=target, truth=truth,
                  dataset_indices=ids, normalized_x=5 * (x - .2))
    path = c.ROOT / 'PREFLIGHT_ARRAYS.npz'; np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as z:
        roundtrip = all(np.array_equal(value, z[key]) for key, value in arrays.items())
    data, metadata = c.build_data()
    checks = dict(all_nine_models=len(rows) == 9,
        nonmultiple_batch=len(ids) == 17 and len(ids) % config['inference_batch_size'] != 0,
        probability_parity=all(v['maximum_probability_error'] <= config.get('probability_tolerance', 1e-5) for v in agreements),
        mask_and_class_parity=all(v['mask_parity'] and v['class_parity'] for v in agreements),
        serialization_roundtrip=roundtrip,
        reused_40_frame_cohort=len(metadata['frames']) == 40 and len(data['X']) == 10240,
        raw_and_mask_D4_disjoint=metadata['exact_raw_and_mask_D4_disjoint'])
    protected = c.protected_inputs()
    result = dict(status='PASS_PREDICTION_PREFLIGHT' if all(checks.values()) else 'FAIL_PREDICTION_PREFLIGHT',
        checks=checks, checks_passed=sum(checks.values()), checks_total=len(checks), agreements=agreements,
        input_digest=c.digest(protected), arrays_sha256=c.sha(path), device=device,
        optimizer_updates=0, fit_count=0, development_pool_inference_performed=False,
        scope='OLD_VAL_17_TECHNICAL_PARITY_AND_ALREADY_SEEN_COHORT_METADATA')
    c.save(c.ROOT / 'PREFLIGHT.json', result)
    print(json.dumps({key: result[key] for key in ['status', 'checks_passed', 'checks_total', 'optimizer_updates']}), flush=True)
    assert all(checks.values()), 'Pre-inference technical parity failed'


def freeze():
    c.config(); assert not (c.ROOT / 'LOCK.json').exists(), 'Refuse to rewrite a frozen workflow'
    pre = c.read(c.ROOT / 'PREFLIGHT.json')
    assert pre['status'] == 'PASS_PREDICTION_PREFLIGHT'
    protected = c.protected_inputs()
    assert pre['input_digest'] == c.digest(protected)
    assert pre['arrays_sha256'] == c.sha(c.ROOT / 'PREFLIGHT_ARRAYS.npz')
    assert not any((c.ROOT / name).exists() for name in ['DATA.npz', 'SOURCES.json', 'PREDICTIONS.npz'])
    data, metadata = c.build_data()
    np.savez_compressed(c.ROOT / 'DATA.npz', **data)
    metadata.update(models=c.model_descriptors(), input_sha256={key: value['sha256'] for key, value in protected.items()})
    c.save(c.ROOT / 'SOURCES.json', metadata)
    snapshot = c.ROOT / 'source_snapshot'; snapshot.mkdir(exist_ok=False)
    for name in ['common.py', 'predict.py', 'analysis.py', 'CONFIG.json', 'PROTOCOL.md']:
        shutil.copy2(c.ROOT / name, snapshot / name)
    c.save(c.ROOT / 'LOCK.json', dict(status='FROZEN_ALREADY_SEEN_DEVELOPMENT_ANALYSIS',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), inputs=protected, models=c.model_descriptors(),
        DATA_sha256=c.sha(c.ROOT / 'DATA.npz'), SOURCES_sha256=c.sha(c.ROOT / 'SOURCES.json'),
        PREFLIGHT_sha256=c.sha(c.ROOT / 'PREFLIGHT.json'), PREFLIGHT_ARRAYS_sha256=c.sha(c.ROOT / 'PREFLIGHT_ARRAYS.npz'),
        examples=10240, frames=40, fit_count=0, optimizer_updates=0, independent_confirmation=False))
    c.verify_lock()
    print(json.dumps(dict(status='FROZEN', locked_inputs=len(protected), models=9, examples=10240,
                         LOCK_sha256=c.sha(c.ROOT / 'LOCK.json'))), flush=True)


def run(device):
    started = time.monotonic(); lock = c.verify_lock()
    assert not any((c.ROOT / name).exists() for name in ['PREDICTIONS.npz', 'INFERENCE.json', 'INFERENCE_FROZEN.json'])
    if device == 'mps': assert torch.backends.mps.is_available() and os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] == '0'
    audit = c.read(c.ROOT / 'PREFLIGHT_AUDIT.json')
    assert audit['status'].startswith('PASS') and all(audit['checks'].values())
    required = {c.relative(c.ROOT / name) for name in ['LOCK.json', 'DATA.npz', 'SOURCES.json']}
    assert required <= audit['input_sha256'].keys()
    assert all(c.sha(c.resolve(name)) == expected for name, expected in audit['input_sha256'].items())
    input_hashes = {c.relative(c.ROOT / name): c.sha(c.ROOT / name)
                    for name in ['LOCK.json', 'DATA.npz', 'SOURCES.json', 'PREFLIGHT_AUDIT.json']}
    c.save(c.ROOT / 'INFERENCE_FROZEN.json', dict(status='FROZEN_BEFORE_DEVELOPMENT_POOL_INFERENCE',
        input_sha256=input_hashes, device=device, fit_count=0, optimizer_updates=0))
    with np.load(c.ROOT / 'DATA.npz', allow_pickle=False) as z:
        data = {key: z[key].copy() for key in z.files}

    def progress(completed, row):
        value = dict(stage='FIXED_DEVELOPMENT_INFERENCE', completed=completed, total=9,
            architecture=row['architecture'], seed=row['seed'], seconds=time.monotonic() - started)
        c.save(c.ROOT / 'PROGRESS.json', value); print(json.dumps(value), flush=True)

    probability = c.predict(lock['models'], data['X'], device, progress)
    # Six historical Group/Plain routes already have predictions on both cohorts.
    # This is parity/provenance checking, not use of their historical test scores.
    archival = []
    offset = 0
    for cohort, folder in enumerate(c.COHORTS):
        with np.load(folder / 'PREDICTIONS.npz', allow_pickle=False) as z:
            count = len(z['truth'])
            for architecture in ['group6', 'plain12_C4pooled']:
                route = (0 if architecture == 'group6' else 1) if cohort == 0 else (2 if architecture == 'group6' else 3)
                expected = z['probability'][route, 1]
                actual = probability[c.ARCHITECTURES.index(architecture), :, offset:offset + count]
                difference = float(np.abs(expected - actual).max())
                assert difference <= c.config().get('probability_tolerance', 1e-5)
                archival.append(dict(cohort=folder.name, architecture=architecture,
                                     maximum_probability_error=difference))
        offset += count
    assert offset == 10240
    np.savez_compressed(c.ROOT / 'PREDICTIONS.npz', probability=probability, truth=data['label'],
        target=data['Y'], frames=data['frame'], raw_key=data['raw_key'], raw_D4_sha256=data['raw_key'],
        architectures=np.array(c.ARCHITECTURES), seeds=np.array(c.SEEDS), source_cohort=data['source_cohort'])
    assert all(c.sha(c.resolve(name)) == expected for name, expected in input_hashes.items())
    c.verify_lock()
    result = dict(status='COMPLETE_FROZEN_DEVELOPMENT_INFERENCE', PREDICTIONS_sha256=c.sha(c.ROOT / 'PREDICTIONS.npz'),
        DATA_sha256=c.sha(c.ROOT / 'DATA.npz'), LOCK_sha256=c.sha(c.ROOT / 'LOCK.json'),
        input_sha256=input_hashes, archival_probability_agreements=archival,
        models=9, examples=10240, frames=40, fit_count=0, optimizer_updates=0, device=device,
        seconds=time.monotonic() - started, independent_confirmation=False)
    c.save(c.ROOT / 'INFERENCE.json', result)
    c.save(c.ROOT / 'PROGRESS.json', dict(status=result['status'], completed=9, total=9, seconds=result['seconds']))
    print(json.dumps(dict(status=result['status'], models=9, examples=10240, seconds=result['seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser(); actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--preflight', action='store_true'); actions.add_argument('--freeze', action='store_true')
    actions.add_argument('--run', action='store_true'); parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    args = parser.parse_args(); torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.preflight: preflight(args.device)
        elif args.freeze: freeze()
        else: run(args.device)


if __name__ == '__main__': main()
