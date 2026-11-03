"""Frozen-model AP and frame-held-out calibration on explicitly reused DEV data."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
import common as c


def confusion_matrix(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    assert truth.shape == prediction.shape
    return np.bincount(9 * truth.astype(np.int64) + prediction.astype(np.int64), minlength=81).reshape(9, 9)


def macro_from_confusion(confusion):
    denominator = confusion.sum(-1) + confusion.sum(-2)
    numerator = 2 * np.diagonal(confusion, axis1=-2, axis2=-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0).mean(-1)


def macro_f1(truth, prediction):
    return float(macro_from_confusion(confusion_matrix(truth, prediction)))


def select_threshold(scores, thresholds):
    """Class labels only: no pixel masks or IoU participate in this ranking."""
    assert len(scores) == len(thresholds)
    hundredths = [int(round(100 * float(threshold))) for threshold in thresholds]
    return max(range(len(thresholds)), key=lambda j: (float(scores[j]), -abs(hundredths[j] - 50), -hundredths[j]))


def frame_plan(frames, raw_keys, settings):
    frames, raw_keys = np.asarray(frames), np.asarray(raw_keys)
    values = np.unique(frames)
    assert len(values) == settings['frames'] == 40 and len(frames) == settings['examples'] == 10240
    assert len(set(raw_keys.tolist())) == len(raw_keys)
    ranked = sorted(values.tolist(), key=lambda f: hashlib.sha256((settings['fold_prefix'] + str(f)).encode()).hexdigest())
    assignments = {int(frame): position % settings['folds'] for position, frame in enumerate(ranked)}
    frame_fold = np.array([assignments[int(frame)] for frame in values], np.int8)
    example_fold = np.array([assignments[int(frame)] for frame in frames], np.int8)
    row_indices = []
    for frame in values:
        candidates = np.flatnonzero(frames == frame).tolist()
        assert len(candidates) == settings['examples_per_frame'] == 256
        candidates.sort(key=lambda i: hashlib.sha256((settings['row_prefix'] + str(raw_keys[i])).encode()).hexdigest())
        row_indices.append(candidates[:settings['calibration_per_frame']])
    row_indices = np.array(row_indices, np.int64)
    pool = np.sort(row_indices.reshape(-1))
    calibrations, holdouts = [], []
    for fold in range(settings['folds']):
        calibration = np.sort(row_indices[frame_fold != fold].reshape(-1))
        holdout = np.flatnonzero(example_fold == fold)
        assert len(calibration) == settings['calibration_examples'] == 512
        assert len(holdout) == 2048 and not set(frames[calibration]) & set(frames[holdout])
        calibrations.append(calibration); holdouts.append(holdout)
    assert np.array_equal(np.sort(np.concatenate(holdouts)), np.arange(len(frames)))
    assert len(pool) == 640 and len(set(pool.tolist())) == len(pool)
    return dict(frame_values=values, frame_fold=frame_fold, example_fold=example_fold,
        ranked_frames=np.array(ranked), row_candidate_indices=row_indices,
        calibration_pool_indices=pool, calibration_indices=np.array(calibrations),
        heldout_indices=np.array(holdouts))


def pixel_ap(probability, target, frames, callback=None):
    """Non-interpolated average precision; the primary average weights frames equally."""
    probability, target, frames = np.asarray(probability), np.asarray(target, bool), np.asarray(frames)
    assert probability.shape[:2] == (3, 3) and probability.shape[2:] == target.shape
    values = np.unique(frames)
    per_frame = np.empty((3, 3, len(values)), np.float64)
    pooled = np.empty((3, 3), np.float64)
    prevalence = np.array([target[frames == frame].mean() for frame in values])
    for architecture in range(3):
        for seed in range(3):
            scores = probability[architecture, seed]
            for j, frame in enumerate(values):
                ids = frames == frame
                per_frame[architecture, seed, j] = average_precision_score(target[ids].reshape(-1), scores[ids].reshape(-1))
            pooled[architecture, seed] = average_precision_score(target.reshape(-1), scores.reshape(-1))
            if callback: callback(architecture, seed)
    return dict(frame_values=values, per_frame_ap=per_frame, mean_frame_ap=per_frame.mean(2),
                pooled_pixel_ap=pooled, frame_metal_prevalence=prevalence,
                pooled_metal_prevalence=np.array(float(target.mean())))


def mask_iou(mask, target):
    union, intersection = (mask | target).sum((1, 2)), (mask & target).sum((1, 2))
    return np.divide(intersection, union, out=np.ones(len(mask)), where=union > 0)


def progress(stage, **values):
    row = dict(stage=stage, **values)
    c.save(c.ROOT / 'PROGRESS.json', row)
    with (c.ROOT / 'analysis_progress.jsonl').open('a') as handle:
        handle.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def score_masks(probability, target, truth, thresholds, callback=None):
    n = len(truth)
    prediction = np.empty((3, 3, n), np.int64)
    packed = np.empty((3, 3, n, 18), np.uint8)
    ious = np.empty((3, 3, n), np.float64)
    confusion = np.empty((3, 3, 9, 9), np.int64)
    for a in range(3):
        for s in range(3):
            mask = probability[a, s] >= float(thresholds[a, s])
            prediction[a, s] = c.hard_classes(mask)
            packed[a, s] = np.packbits(mask.reshape(n, 144), axis=1)
            ious[a, s] = mask_iou(mask, target)
            confusion[a, s] = confusion_matrix(truth, prediction[a, s])
            if callback: callback(a, s)
    return dict(prediction=prediction, packed_masks=packed, IoU=ious, confusion=confusion,
                F1=macro_from_confusion(confusion), mean_IoU=ious.mean(2))


def calibrate(probability, target, truth, plan, settings):
    n = len(truth); thresholds = np.asarray(settings['thresholds'])
    pool = plan['calibration_pool_indices']; size = len(pool)
    pool_position = np.full(n, -1, np.int64); pool_position[pool] = np.arange(size)
    pool_classes = np.empty((3, 3, len(thresholds), size), np.int64)
    pool_packed = np.empty((3, 3, len(thresholds), size, 18), np.uint8)
    scores = np.empty((3, 3, 5, len(thresholds)), np.float64)
    selected_index = np.empty((3, 3, 5), np.int64)
    selected_thresholds = np.empty((3, 3, 5), np.float64)
    oof_prediction = np.full((3, 3, n), -1, np.int64)
    oof_packed = np.empty((3, 3, n, 18), np.uint8)
    oof_iou = np.empty((3, 3, n), np.float64)
    fold_confusion = np.empty((3, 3, 5, 9, 9), np.int64)
    fold_f1, fold_iou = np.empty((3, 3, 5), np.float64), np.empty((3, 3, 5), np.float64)
    calibration_counts = np.array([np.bincount(truth[ids], minlength=9) for ids in plan['calibration_indices']])
    for a in range(3):
        for s in range(3):
            for j, threshold in enumerate(thresholds):
                mask = probability[a, s, pool] >= threshold
                pool_classes[a, s, j] = c.hard_classes(mask)
                pool_packed[a, s, j] = np.packbits(mask.reshape(size, 144), axis=1)
                if (j + 1) % 10 == 0 or j + 1 == len(thresholds):
                    progress('CALIBRATION_POOL_CLASSES', architecture=c.ARCHITECTURES[a], seed=c.SEEDS[s],
                             completed_thresholds=j + 1, target_thresholds=len(thresholds))
            for fold in range(5):
                calibration = plan['calibration_indices'][fold]
                holdout = plan['heldout_indices'][fold]
                positions = pool_position[calibration]; assert (positions >= 0).all()
                scores[a, s, fold] = [macro_f1(truth[calibration], row[positions]) for row in pool_classes[a, s]]
                chosen = select_threshold(scores[a, s, fold], thresholds)
                threshold = float(thresholds[chosen])
                selected_index[a, s, fold], selected_thresholds[a, s, fold] = chosen, threshold
                mask = probability[a, s, holdout] >= threshold
                prediction = c.hard_classes(mask)
                iou = mask_iou(mask, target[holdout])
                oof_prediction[a, s, holdout] = prediction
                oof_packed[a, s, holdout] = np.packbits(mask.reshape(len(holdout), 144), axis=1)
                oof_iou[a, s, holdout] = iou
                fold_confusion[a, s, fold] = confusion_matrix(truth[holdout], prediction)
                fold_f1[a, s, fold] = macro_from_confusion(fold_confusion[a, s, fold])
                fold_iou[a, s, fold] = iou.mean()
                progress('FRAME_HELD_OUT_CALIBRATION', architecture=c.ARCHITECTURES[a], seed=c.SEEDS[s],
                         fold=fold, completed_units=(3 * a + s) * 5 + fold + 1, target_units=45)
            progress('FRAME_HELD_OUT_CALIBRATION', architecture=c.ARCHITECTURES[a], seed=c.SEEDS[s],
                     completed_models=3 * a + s + 1, target_models=9)
    assert (oof_prediction >= 0).all()
    oof_confusion = fold_confusion.sum(2)
    return dict(**plan, thresholds=thresholds, calibration_class_counts=calibration_counts,
        calibration_pool_classes=pool_classes, calibration_pool_packed_masks=pool_packed,
        calibration_scores=scores, selected_threshold_indices=selected_index, selected_thresholds=selected_thresholds,
        oof_prediction=oof_prediction, oof_packed_masks=oof_packed, oof_IoU=oof_iou,
        oof_confusion=oof_confusion, oof_F1=macro_from_confusion(oof_confusion), oof_mean_IoU=oof_iou.mean(2),
        fold_confusion=fold_confusion, fold_F1=fold_f1, fold_IoU=fold_iou)


def summarize(ap, calibration, fixed, old):
    cells = {}
    for a, architecture in enumerate(c.ARCHITECTURES):
        cells[architecture] = dict(
            mean_frame_AP=float(ap['mean_frame_ap'][a].mean()), pooled_pixel_AP=float(ap['pooled_pixel_ap'][a].mean()),
            OOF_F1=float(calibration['oof_F1'][a].mean()), OOF_IoU=float(calibration['oof_mean_IoU'][a].mean()),
            fixed05_F1=float(fixed['F1'][a].mean()), fixed05_IoU=float(fixed['mean_IoU'][a].mean()),
            old_VAL_threshold_F1=float(old['F1'][a].mean()), old_VAL_threshold_IoU=float(old['mean_IoU'][a].mean()),
            per_seed=[dict(seed=seed, mean_frame_AP=float(ap['mean_frame_ap'][a, s]),
                pooled_pixel_AP=float(ap['pooled_pixel_ap'][a, s]), OOF_F1=float(calibration['oof_F1'][a, s]),
                OOF_IoU=float(calibration['oof_mean_IoU'][a, s]), fixed05_F1=float(fixed['F1'][a, s]),
                old_VAL_threshold_F1=float(old['F1'][a, s]), thresholds=calibration['selected_thresholds'][a, s].tolist())
                for s, seed in enumerate(c.SEEDS)])

    def contrast(reference):
        metrics = ['mean_frame_AP', 'pooled_pixel_AP', 'OOF_F1', 'OOF_IoU', 'fixed05_F1', 'fixed05_IoU',
                   'old_VAL_threshold_F1', 'old_VAL_threshold_IoU']
        return dict(mean={metric: cells[c.ARCHITECTURES[0]][metric] - cells[c.ARCHITECTURES[reference]][metric]
                          for metric in metrics},
            per_seed=[dict(seed=seed, mean_frame_AP=float(ap['mean_frame_ap'][0, s] - ap['mean_frame_ap'][reference, s]),
                OOF_F1=float(calibration['oof_F1'][0, s] - calibration['oof_F1'][reference, s]),
                OOF_IoU=float(calibration['oof_mean_IoU'][0, s] - calibration['oof_mean_IoU'][reference, s]))
                for s, seed in enumerate(c.SEEDS)],
            per_fold=[dict(fold=fold, F1=float((calibration['fold_F1'][0, :, fold] - calibration['fold_F1'][reference, :, fold]).mean()),
                IoU=float((calibration['fold_IoU'][0, :, fold] - calibration['fold_IoU'][reference, :, fold]).mean()))
                for fold in range(5)])
    return dict(cells=cells, contrasts=dict(group_minus_untied=contrast(1), group_minus_narrow=contrast(2)))


def eligibility(summary, settings):
    effect = summary['contrasts']['group_minus_untied']; limits = settings['eligibility']
    guards = dict(mean_frame_AP=effect['mean']['mean_frame_AP'] > limits['minimum_AP_delta'],
        OOF_F1=effect['mean']['OOF_F1'] >= limits['minimum_OOF_F1_delta'],
        OOF_IoU=effect['mean']['OOF_IoU'] >= limits['minimum_OOF_IoU_delta'],
        each_fold_F1=min(row['F1'] for row in effect['per_fold']) >= limits['minimum_fold_F1_delta'])
    guards = {key: bool(value) for key, value in guards.items()}
    return dict(passed=all(guards.values()), guards=guards, failed=[key for key, value in guards.items() if not value],
                scope='descriptive_development_eligibility_only_not_independent_confirmation')


def preflight():
    settings = c.config(); checks = {}
    scores = np.zeros(81); thresholds = settings['thresholds']
    checks['threshold_tie_uses_nearest_half'] = select_threshold(scores, thresholds) == 40
    scores[30] = scores[50] = 1.
    checks['symmetric_threshold_tie_uses_lower'] = thresholds[select_threshold(scores, thresholds)] == .4
    scores[:] = 0.; scores[20] = scores[60] = 1.
    checks['symmetric_0_3_0_7_tie_uses_lower_despite_float_roundoff'] = thresholds[select_threshold(scores, thresholds)] == .3
    checks['macro_F1_fixed_nine_classes'] = macro_f1(np.arange(9), np.arange(9)) == 1. and macro_f1(np.arange(9), (np.arange(9) + 1) % 9) == 0.
    truth = np.array([0, 1, 0, 1])
    checks['AP_perfect_and_constant_ties'] = average_precision_score(truth, truth) == 1. and average_precision_score(truth, np.ones(4)) == .5
    ranking_truth = np.array([0, 1, 0, 1, 0, 1, 1])
    ranking_scores = np.array([-9., -2., -2., 1., 3., 3., 12.])
    checks['AP_invariant_to_strictly_increasing_transform_with_gaps_and_ties'] = (
        average_precision_score(ranking_truth, ranking_scores) ==
        average_precision_score(ranking_truth, np.exp(ranking_scores)))
    frames = np.repeat(np.arange(40), 256)
    raw = np.array([hashlib.sha256(str(i).encode()).hexdigest() for i in range(10240)])
    plan = frame_plan(frames, raw, settings)
    checks['five_frame_disjoint_folds_and_exact_512_shared_labels'] = all(
        len(calibration) == 512 and not set(frames[calibration]) & set(frames[heldout])
        for calibration, heldout in zip(plan['calibration_indices'], plan['heldout_indices']))
    checks['all_examples_held_out_exactly_once'] = np.array_equal(np.sort(plan['heldout_indices'].reshape(-1)), np.arange(10240))
    checks['calibration_candidate_rows_do_not_accept_labels_or_masks'] = frame_plan.__code__.co_varnames[:3] == ('frames', 'raw_keys', 'settings')
    result = dict(status='PASS_ANALYSIS_PREFLIGHT' if all(checks.values()) else 'FAIL_ANALYSIS_PREFLIGHT', checks=checks,
        checks_passed=sum(checks.values()), checks_total=len(checks), optimizer_updates=0, model_inference=False,
        analysis_source_sha256=c.sha(Path(__file__)), config_sha256=c.sha(c.ROOT / 'CONFIG.json'))
    c.save(c.ROOT / 'ANALYSIS_PREFLIGHT.json', result)
    print(json.dumps(result), flush=True)
    assert all(checks.values())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight', action='store_true'); args = parser.parse_args()
    if args.preflight:
        preflight(); return
    started = time.monotonic(); settings = c.config(); c.verify_lock()
    assert not any((c.ROOT / name).exists() for name in ['RESULT.json', 'PRAUC.npz', 'CALIBRATION.npz']), 'Immutable analysis outputs already exist'
    inference = c.read(c.ROOT / 'INFERENCE.json')
    assert inference['status'] == 'COMPLETE_FROZEN_DEVELOPMENT_INFERENCE' and inference['fit_count'] == 0
    assert inference['PREDICTIONS_sha256'] == c.sha(c.ROOT / 'PREDICTIONS.npz')
    assert inference['DATA_sha256'] == c.sha(c.ROOT / 'DATA.npz')
    assert inference['LOCK_sha256'] == c.sha(c.ROOT / 'LOCK.json')
    for path, expected in inference['input_sha256'].items(): assert c.sha(c.resolve(path)) == expected
    inputs = {c.relative(c.ROOT / name): c.sha(c.ROOT / name) for name in
              ['DATA.npz', 'PREDICTIONS.npz', 'INFERENCE.json', 'LOCK.json', 'CONFIG.json', 'PROTOCOL.md', 'analysis.py']}
    c.save(c.ROOT / 'ANALYSIS_FROZEN.json', dict(input_sha256=inputs, optimizer_updates=0,
        scope='PREVIOUSLY_EVALUATED_FRAMES_REUSED_FOR_DEVELOPMENT', new_confirmation=False))
    with np.load(c.ROOT / 'DATA.npz', allow_pickle=False) as z:
        frames, truth, target, raw = z['frame'].copy(), z['label'].copy(), z['Y'].copy().astype(bool), z['raw_D4_sha256'].copy()
    with np.load(c.ROOT / 'PREDICTIONS.npz', allow_pickle=False) as z:
        probability = z['probability'].copy()
        assert np.array_equal(frames, z['frames']) and np.array_equal(truth, z['truth']) and np.array_equal(target, z['target'])
        assert np.array_equal(z['architectures'], np.array(c.ARCHITECTURES)) and np.array_equal(z['seeds'], np.array(c.SEEDS))
    assert probability.shape == (3, 3, 10240, 12, 12)
    assert np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
    plan = frame_plan(frames, raw, settings)
    ap = pixel_ap(probability, target, frames,
        lambda a, s: progress('AP_DEVELOPMENT', architecture=c.ARCHITECTURES[a], seed=c.SEEDS[s],
                              completed_models=3 * a + s + 1, target_models=9, seconds=time.monotonic() - started))
    progress('AP_DEVELOPMENT_COMPLETE', models=9, frames=40, seconds=time.monotonic() - started)
    descriptors = c.model_descriptors(); old_thresholds = np.empty((3, 3), np.float64)
    old_probability = np.empty((3, 3, 695, 12, 12), np.float32)
    with np.load(c.PILOT / 'TRAIN_VAL.npz', allow_pickle=False) as z:
        ids = np.flatnonzero(z['role'] == 'val'); old_frames = z['frame'][ids]; old_truth = z['label'][ids]; old_target = z['Y'][ids, 0].astype(bool)
    for row in descriptors:
        a, s = c.ARCHITECTURES.index(row['architecture']), c.SEEDS.index(row['seed'])
        with np.load(c.resolve(row['source_VAL_file']), allow_pickle=False) as z:
            assert np.array_equal(z['truth'], old_truth) and np.array_equal(z['target'], old_target)
            old_probability[a, s] = z['probability'][:, 0]
        old_thresholds[a, s] = row['primary_fine_threshold']
    old_ap = pixel_ap(old_probability, old_target, old_frames,
        lambda a, s: progress('AP_OLD_VAL', architecture=c.ARCHITECTURES[a], seed=c.SEEDS[s],
                              completed_models=3 * a + s + 1, target_models=9, seconds=time.monotonic() - started))
    ap.update(old_VAL_frame_values=old_ap['frame_values'], old_VAL_per_frame_ap=old_ap['per_frame_ap'],
              old_VAL_mean_frame_ap=old_ap['mean_frame_ap'], old_VAL_pooled_pixel_ap=old_ap['pooled_pixel_ap'],
              old_VAL_frame_metal_prevalence=old_ap['frame_metal_prevalence'])
    np.savez_compressed(c.ROOT / 'PRAUC.npz', **ap, architectures=np.array(c.ARCHITECTURES), seeds=np.array(c.SEEDS))
    calibration = calibrate(probability, target, truth, plan, settings)
    fixed = score_masks(probability, target, truth, np.full((3, 3), .5),
        lambda a, s: progress('FIXED05_DIAGNOSTIC', architecture=c.ARCHITECTURES[a], seed=c.SEEDS[s],
                              completed_models=3 * a + s + 1, target_models=9, seconds=time.monotonic() - started))
    old = score_masks(probability, target, truth, old_thresholds,
        lambda a, s: progress('OLD_VAL_THRESHOLD_DIAGNOSTIC', architecture=c.ARCHITECTURES[a], seed=c.SEEDS[s],
                              completed_models=3 * a + s + 1, target_models=9, seconds=time.monotonic() - started))
    arrays = dict(calibration, truth=truth, target=target, frames=frames, old_VAL_thresholds=old_thresholds)
    arrays.update({f'fixed05_{key}': value for key, value in fixed.items()})
    arrays.update({f'old_VAL_threshold_{key}': value for key, value in old.items()})
    np.savez_compressed(c.ROOT / 'CALIBRATION.npz', **arrays, architectures=np.array(c.ARCHITECTURES), seeds=np.array(c.SEEDS))
    summary = summarize(ap, calibration, fixed, old)
    result = dict(status='COMPLETE_CALIBRATION_DEVELOPMENT_ANALYSIS', **summary,
        next_study_eligibility=eligibility(summary, settings), new_confirmation=False, optimizer_updates=0, fit_count=0,
        examples=10240, frames=40, folds=5, calibration_labels_per_fold=512,
        unique_calibration_examples=640, calibration_indices_shared_all_models=True,
        calibration_class_counts=calibration['calibration_class_counts'].tolist(),
        missing_calibration_classes=[np.flatnonzero(row == 0).tolist() for row in calibration['calibration_class_counts']],
        frame_folds={str(int(frame)): int(fold) for frame, fold in zip(plan['frame_values'], plan['frame_fold'])},
        AP_definition='noninterpolated_average_precision_macro_frame_then_seed',
        calibration_tie_rule='MacroF1_then_nearest_0.5_then_lower; no IoU or pixel mask labels',
        PRAUC_sha256=c.sha(c.ROOT / 'PRAUC.npz'), CALIBRATION_sha256=c.sha(c.ROOT / 'CALIBRATION.npz'),
        PREDICTIONS_sha256=c.sha(c.ROOT / 'PREDICTIONS.npz'), LOCK_sha256=c.sha(c.ROOT / 'LOCK.json'),
        ANALYSIS_FROZEN_sha256=c.sha(c.ROOT / 'ANALYSIS_FROZEN.json'), seconds=time.monotonic() - started,
        caveats=['previously_evaluated_frames_now_development_only', 'no_new_independent_confirmation',
                 'class_labels_used_for_calibration_512_per_fold_shared_all_models', 'same_IC_only',
                 'overlapping_calibration_training_folds_no_CI_or_pvalues', 'AP_is_segmentation_not_classification_F1'])
    c.verify_lock()
    assert all(c.sha(c.resolve(path)) == expected for path, expected in inputs.items())
    c.save(c.ROOT / 'RESULT.json', result)
    progress(result['status'], next_study_eligibility=result['next_study_eligibility'], seconds=result['seconds'])


if __name__ == '__main__':
    with threadpool_limits(limits=2): main()
