"""Independent CPU audit of AP and frame-held-out threshold calibration.

The previously inspected forty frames are development data in this analysis.
No fitting of networks, optimizer updates, downloads or unseen-target access.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
G = ROOT.parent
STAGE = G / 'sem_arch_groupcnn_20261102'
UNTIED = G / 'sem_arch_untied_control_20261103'
COHORTS = [G / 'sem_arch_groupcnn_confirmation_20261102',
           G / 'sem_arch_bce_confirmation_20261103']
ARMS = ['group6', 'untied24_single', 'plain12_C4pooled']
SEEDS = [5701, 5721, 5757]
THRESHOLDS = np.array([round(.1 + .01 * i, 2) for i in range(81)])
FOLD_PREFIX = 'SEM-C4-CALIB-FOLD-20261103/'
ROW_PREFIX = 'SEM-C4-CALIB-ROW-20261103/'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


ref = module('calibration_independent_model_graph_reference', STAGE / 'audit.py')


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(path): return str(Path(path).resolve().relative_to(REPO))
def resolve(path): return Path(path) if Path(path).is_absolute() else REPO / path
def load(path): return torch.load(path, map_location='cpu', weights_only=True)


def scalar_json(value):
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=scalar_json) + '\n')


def max_error(expected, observed):
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(expected) != set(observed): return float('inf')
        return max([max_error(value, observed[key]) for key, value in expected.items()] + [0.])
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(observed): return float('inf')
        return max([max_error(a, b) for a, b in zip(expected, observed)] + [0.])
    if isinstance(expected, (str, bool)) or expected is None:
        return 0. if expected == observed else float('inf')
    return abs(float(expected) - float(observed))


def confusion(truth, prediction):
    return np.bincount(9 * np.asarray(truth) + np.asarray(prediction), minlength=81).reshape(9, 9)


def macro_from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    denominator = matrix.sum(-1) + matrix.sum(-2)
    return np.divide(2 * np.diagonal(matrix, axis1=-2, axis2=-1), denominator,
                     out=np.zeros_like(denominator), where=denominator > 0).mean(-1)


def macro(truth, prediction): return float(macro_from_confusion(confusion(truth, prediction)))


def rank_ap(truth, scores):
    """Noninterpolated AP from complete equal-score blocks, independently."""
    truth = np.asarray(truth, dtype=bool).ravel()
    scores = np.asarray(scores).ravel()
    assert truth.shape == scores.shape and np.isfinite(scores).all()
    positives = int(truth.sum())
    if positives == 0: return 0.
    order = np.argsort(-scores, kind='stable')
    sorted_scores, target = scores[order], truth[order]
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    true_positive = np.cumsum(target, dtype=np.int64)[ends]
    increments = np.diff(np.r_[0, true_positive])
    precision = true_positive / (ends + 1)
    return float(np.dot(increments / positives, precision))


def plan(frames, rawkeys):
    frame_values = np.array(sorted(np.unique(frames).tolist()))
    ranked = sorted(frame_values.tolist(), key=lambda f: hashlib.sha256((FOLD_PREFIX + str(int(f))).encode()).hexdigest())
    assignment = {frame: position % 5 for position, frame in enumerate(ranked)}
    frame_fold = np.array([assignment[int(frame)] for frame in frame_values], np.int64)
    example_fold = np.array([assignment[int(frame)] for frame in frames], np.int64)
    selected_by_frame = {}
    for frame in frame_values:
        ids = np.flatnonzero(frames == frame).tolist()
        ids.sort(key=lambda i: hashlib.sha256((ROW_PREFIX + str(rawkeys[i])).encode()).hexdigest())
        selected_by_frame[int(frame)] = ids[:16]
    calibration, heldout = [], []
    for fold in range(5):
        calibration.append(np.array(sorted(i for f in frame_values if assignment[int(f)] != fold
                                           for i in selected_by_frame[int(f)]), np.int64))
        heldout.append(np.flatnonzero(example_fold == fold))
    pool = np.array(sorted(i for ids in selected_by_frame.values() for i in ids), np.int64)
    return dict(frame_values=frame_values, frame_fold=frame_fold, example_fold=example_fold,
                ranked_frames=np.array(ranked),
                row_candidate_indices=np.array([selected_by_frame[int(f)] for f in frame_values]),
                calibration_indices=np.array(calibration), heldout_indices=np.array(heldout),
                calibration_pool_indices=pool)


def selection_index(scores):
    # The fixed grid is .10+.01*i. Integer distance avoids asymmetric binary
    # float errors such as |.3-.5| versus |.7-.5| in exact mathematical ties.
    return max(range(len(THRESHOLDS)), key=lambda i: (float(scores[i]), -abs(i - 40), -i))


def expected_models():
    source = read(UNTIED / 'RESULT.json')
    audit = read(UNTIED / 'AUDIT.json')
    assert audit['status'].startswith('PASS') and all(audit['checks'].values())
    assert audit['input_sha256'][rel(UNTIED / 'RESULT.json')] == sha(UNTIED / 'RESULT.json')
    by_key = {(row['architecture'], row['seed']): row for row in source['rows']}
    assert len(by_key) == 9
    result = []
    for architecture in ARMS:
        for seed in SEEDS:
            row = by_key[architecture, seed]
            free = resolve(row['model_file'])
            prediction = free.with_name(free.stem + '_deployed.pt') if architecture == 'group6' else free
            assert sha(free) == row['model_sha256'] and sha(resolve(row['probability_file'])) == row['probability_sha256']
            if architecture == 'group6':
                exported = load(prediction)
                assert exported['free_model_sha256'] == sha(free)
                expected = ref.reference_deploy_state(load(free)['model'])
                assert set(expected) == set(exported['model'])
                assert all(torch.equal(value, exported['model'][name]) for name, value in expected.items())
            result.append(dict(architecture=architecture, seed=seed, budget=27,
                free_model_file=rel(free), free_model_sha256=sha(free),
                prediction_model_file=rel(prediction), prediction_model_sha256=sha(prediction),
                primary_fine_threshold=row['selected']['threshold'],
                source_VAL_file=row['probability_file'], source_VAL_sha256=row['probability_sha256'],
                prediction_width=12 if architecture == 'plain12_C4pooled' else 24,
                prediction_mode='C4' if architecture == 'plain12_C4pooled' else 'single'))
    return result


def expected_data():
    parts = []; raws = []; masks = []
    for cohort in COHORTS:
        index, manifest = read(cohort / 'EVAL_INDEX.json'), read(cohort / 'DATA_MANIFEST.json')
        assert index['EVAL_sha256'] == sha(cohort / 'EVAL.npz')
        assert index['TEST_sha256'] == manifest['TEST_sha256']
        with np.load(cohort / 'EVAL.npz', allow_pickle=False) as z:
            parts.append({key: z[key].copy() for key in z.files})
        records = index['records']
        assert len(records) == len(parts[-1]['X'])
        for i, row in enumerate(records):
            assert row['frame'] == int(parts[-1]['frame'][i])
            assert row['label'] == int(parts[-1]['label'][i])
        raws.extend(row['raw_D4_sha256'] for row in records)
        masks.extend(row['canonical_D4'] for row in records)
    result = dict(X=np.concatenate([p['X'] for p in parts]), Y=np.concatenate([p['Y'][:, 0].astype(bool) for p in parts]),
        label=np.concatenate([p['label'] for p in parts]), frame=np.concatenate([p['frame'] for p in parts]),
        source_cohort=np.r_[np.zeros(len(parts[0]['X']), np.int64), np.ones(len(parts[1]['X']), np.int64)],
        rawkeys=np.array(raws))
    assert len(result['X']) == 10240 and len(np.unique(result['frame'])) == 40
    assert all(int((result['frame'] == frame).sum()) == 256 for frame in np.unique(result['frame']))
    assert len(set(raws)) == len(set(masks)) == 10240
    return result


def source_checks(checks, details):
    sys.path.insert(0, str(ROOT))
    c = module('calibration_audit_common_subject', ROOT / 'common.py')
    settings = read(ROOT / 'CONFIG.json')
    expected = expected_models(); actual = c.model_descriptors()
    assert len(actual) == 9
    for left, right in zip(expected, actual):
        assert all(right[key] == value for key, value in left.items()), (left, right)
    checks['all_nine_fixed_budget27_auxiliary_model_descriptors_and_exports_match'] = True
    data = expected_data()
    checks['forty_previously_viewed_frames_exact10240_and_disjoint_D4_cohorts'] = True
    checks['configuration_folds_budget_AP_and_future_eligibility_fixed'] = (
        settings['architectures'] == ARMS and settings['seeds'] == SEEDS and settings['budget'] == 27 and
        settings['frames'] == 40 and settings['examples'] == 10240 and settings['examples_per_frame'] == 256 and
        settings['folds'] == 5 and settings['calibration_per_frame'] == 16 and settings['calibration_examples'] == 512 and
        settings['unique_calibration_examples'] == 640 and
        settings['thresholds'] == THRESHOLDS.tolist() and settings['fold_prefix'] == FOLD_PREFIX and
        settings['row_prefix'] == ROW_PREFIX and settings['probability_tolerance'] == 1e-5 and
        settings['AP_definition'] == 'non_interpolated_average_precision_macro_frame_then_seed' and
        settings['threshold_selection'] == 'class_macro_F1_then_nearest_0.5_then_lower_no_IoU' and
        settings['eligibility'] == dict(minimum_AP_delta=0., minimum_OOF_F1_delta=.005,
            minimum_OOF_IoU_delta=-.002, minimum_fold_F1_delta=-.01) and
        settings['scope'] == 'FORMERLY_EVALUATED_FRAMES_NOW_DEVELOPMENT_NO_NEW_CONFIRMATION' and
        settings['protocol_sha256'] == sha(ROOT / 'PROTOCOL.md'))
    lock = c.verify_lock()
    checks['immutable_inputs_sources_and_model_data_lock_match'] = all(
        sha(resolve(name)) == row['sha256'] for name, row in lock['inputs'].items())
    details['locked_input_count'] = len(lock['inputs'])
    details['model_count'] = len(actual)
    details['frames'] = sorted(np.unique(data['frame']).tolist())
    with np.load(ROOT / 'DATA.npz', allow_pickle=False) as z:
        checks['all_saved_DEVELOPMENT_data_and_rawkeys_exactly_match_two_audited_sources'] = (
            all(np.array_equal(z[name], data[name]) for name in ['X', 'Y', 'label', 'frame', 'source_cohort']) and
            np.array_equal(z['frames'], data['frame']) and np.array_equal(z['raw_key'], data['rawkeys']) and
            np.array_equal(z['raw_D4_sha256'], data['rawkeys']))
    return c, settings, expected, data


def synthetic_checks(checks, details, settings):
    analysis = module('calibration_audited_analysis_subject', ROOT / 'analysis.py')
    from sklearn.metrics import average_precision_score
    examples = [(np.array([1, 0, 1, 0, 1, 0]), np.array([.9, .8, .8, .3, .3, .1])),
                (np.array([1, 1, 0, 0]), np.array([.5, .5, .5, .5])),
                (np.array([0, 0, 0]), np.array([.8, .4, .1]))]
    error = 0.
    for truth, values in examples:
        with np.errstate(all='ignore'):
            expected = rank_ap(truth, values)
            observed = average_precision_score(truth, values)
        error = max(error, abs(expected - observed), abs(rank_ap(truth, 2 * values + 3) - expected))
    checks['independent_AP_tie_blocks_and_monotone_score_invariance'] = error < 1e-14
    checks['threshold_ties_are_closest_half_then_lower_without_IoU'] = (
        selection_index(np.ones(81)) == 40 and
        selection_index(np.array([1. if i in [39, 41] else 0. for i in range(81)])) == 39 and
        selection_index(np.array([1. if i in [20, 60] else 0. for i in range(81)])) == 20)
    checks['producer_threshold_tie_rule_matches_independent_reference'] = all(
        analysis.select_threshold(scores, THRESHOLDS) == selection_index(scores)
        for scores in [np.zeros(81), np.array([1. if i in [39, 41] else 0. for i in range(81)]),
                       np.array([1. if i in [20, 60] else 0. for i in range(81)]),
                       np.array([.8 if i == 2 else .5 for i in range(81)])])
    # Exercise the producer's per-frame AP operator without reading target data.
    target = np.tile(np.array([0, 1, 0, 1], bool).reshape(1, 2, 2), (4, 1, 1))
    frames = np.repeat([10, 20], 2)
    values = np.array([.9, .8, .8, .1, .2, .7, .7, .3, .1, .6, .4, .9, .8, .8, .2, .2]).reshape(4, 2, 2)
    p = np.broadcast_to(values, (3, 3, 4, 2, 2)).copy()
    observed = analysis.pixel_ap(p, target, frames)
    expected = reference_ap(p, target, frames)
    transformed = analysis.pixel_ap(p * 2 + 3, target, frames)
    producer_error = max(float(np.abs(observed[key] - expected[key]).max()) for key in expected)
    producer_error = max(producer_error, max(float(np.abs(observed[key] - transformed[key]).max())
        for key in ['per_frame_ap', 'mean_frame_ap', 'pooled_pixel_ap']))
    checks['producer_AP_frame_averaging_ties_and_monotone_ranking_match_reference'] = producer_error < 1e-14
    details['synthetic_AP_maximum_error'] = error
    details['synthetic_producer_AP_maximum_error'] = producer_error


def design_checks(checks, details, settings, data):
    layout = plan(data['frame'], data['rawkeys'])
    calibration, heldout = layout['calibration_indices'], layout['heldout_indices']
    checks['five_exact8_frame_holdouts_32_calibration_frames_512_shared_labels'] = (
        calibration.shape == (5, 512) and heldout.shape == (5, 2048) and
        len(layout['calibration_pool_indices']) == 640 and
        all(len(np.unique(data['frame'][calibration[f]])) == 32 and
            len(np.unique(data['frame'][heldout[f]])) == 8 and
            not set(data['frame'][calibration[f]]) & set(data['frame'][heldout[f]])
            for f in range(5)))
    checks['each_example_exactly_once_OOF_and_no_frame_or_example_calibration_leak'] = (
        np.array_equal(np.bincount(heldout.ravel(), minlength=10240), np.ones(10240)) and
        all(len(set(calibration[f])) == 512 and not set(calibration[f]) & set(heldout[f]) for f in range(5)))
    analysis = module('calibration_plan_audit_subject', ROOT / 'analysis.py')
    actual = analysis.frame_plan(data['frame'], data['rawkeys'], settings)
    checks['producer_plan_reproduces_independent_class_blind_SHA_frame_and_row_rank'] = all(
        np.array_equal(actual[name], layout[name]) for name in layout)
    source = (ROOT / 'common.py').read_text() + (ROOT / 'predict.py').read_text() + (ROOT / 'analysis.py').read_text()
    tree = ast.parse(source)
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    checks['active_workflow_has_no_optimizer_updates_model_fits_or_download_calls'] = not any(
        name.endswith('.step') or name.endswith('.fit') or name.startswith('torch.optim.') or
        name in ['requests.get', 'urllib.request.urlopen', 'urlopen'] for name in calls)
    details['calibration_class_counts'] = [np.bincount(data['label'][ids], minlength=9).tolist() for ids in calibration]
    details['unique_calibration_label_union'] = 640


def preflight_checks(checks, details, models):
    pre = read(ROOT / 'PREFLIGHT.json')
    checks['producer_nine_model_oldVAL_preflight_passes_zero_updates'] = (
        pre['status'] == 'PASS_PREDICTION_PREFLIGHT' and all(pre['checks'].values()) and
        pre['fit_count'] == pre['optimizer_updates'] == 0 and not pre['development_pool_inference_performed'] and
        pre['arrays_sha256'] == sha(ROOT / 'PREFLIGHT_ARRAYS.npz'))
    with np.load(ROOT / 'PREFLIGHT_ARRAYS.npz', allow_pickle=False) as z:
        saved = {key: z[key].copy() for key in z.files}
    with np.load(G / 'sem_architecture_pilot_20261102' / 'TRAIN_VAL.npz', allow_pickle=False) as z:
        ids = np.flatnonzero(z['role'] == 'val')[:17]
        x = torch.tensor(5 * (z['X'][ids] - .2))
        truth, target = z['label'][ids], z['Y'][ids, 0].astype(bool)
    checks['oldVAL17_nonmultiple_batch_fixture_source_examples_are_exact'] = (
        np.array_equal(saved['dataset_indices'], ids) and np.array_equal(saved['normalized_x'], x.numpy()) and
        np.array_equal(saved['truth'], truth) and np.array_equal(saved['target'], target))
    mask_ok = class_ok = archive_ok = True; forward_error = 0.; archive_error = 0.
    for a, architecture in enumerate(ARMS):
        for s, seed in enumerate(SEEDS):
            row = next(row for row in models if row['architecture'] == architecture and row['seed'] == seed)
            state = load(resolve(row['prediction_model_file']))['model']
            with torch.no_grad():
                logits = ref.stable_orbit_logits(x, state)[0] if architecture == 'plain12_C4pooled' else ref.reference_plain(x, state)
                probability = logits.sigmoid().numpy()[:, 0]
            forward_error = max(forward_error, float(np.abs(probability - saved['probability'][a, s]).max()))
            masks = saved['probability'][a, s] >= .5
            prediction = ref.classify(masks)
            mask_ok &= np.array_equal(masks, saved['masks'][a, s])
            class_ok &= np.array_equal(prediction, saved['classes'][a, s])
            with np.load(resolve(row['source_VAL_file']), allow_pickle=False) as z:
                t = int(np.flatnonzero(z['thresholds'] == .5)[0])
                archive_error = max(archive_error, float(np.abs(z['probability'][:17, 0] - saved['probability'][a, s]).max()))
                archive_ok &= (np.array_equal(z['truth'][:17], truth) and np.array_equal(z['target'][:17], target) and
                    np.array_equal(z['predictions'][t, :17], prediction) and
                    np.array_equal(z['packed_masks'][t, :17], np.packbits(masks.reshape(17, 144), axis=1)))
    checks['independent_CPU_forward_all_nine_archived_VAL_probabilities_match'] = max(forward_error, archive_error) <= 1e-5
    checks['all_nine_oldVAL_half_threshold_masks_and_graph_classes_match'] = bool(mask_ok and class_ok and archive_ok)
    details['preflight_max_CPU_probability_error'] = forward_error
    details['preflight_max_archive_probability_error'] = archive_error


def reference_ap(probability, target, frames):
    values = np.unique(frames)
    per_frame = np.empty((3, 3, len(values)), np.float64)
    pooled = np.empty((3, 3), np.float64)
    for a in range(3):
        for s in range(3):
            for f, frame in enumerate(values):
                keep = frames == frame
                per_frame[a, s, f] = rank_ap(target[keep], probability[a, s, keep])
            pooled[a, s] = rank_ap(target, probability[a, s])
    prevalence = np.array([target[frames == frame].mean() for frame in values])
    return dict(frame_values=values, per_frame_ap=per_frame, mean_frame_ap=per_frame.mean(-1),
                pooled_pixel_ap=pooled, frame_metal_prevalence=prevalence,
                pooled_metal_prevalence=np.array(float(target.mean())))


def mask_metrics(probability, target, truth, thresholds):
    """Threshold may vary per model and example; never fitted here."""
    output = dict(prediction=np.empty((3, 3, len(truth)), np.int64),
        packed=np.empty((3, 3, len(truth), 18), np.uint8),
        iou=np.empty((3, 3, len(truth)), np.float64),
        F1=np.empty((3, 3), np.float64), confusion=np.empty((3, 3, 9, 9), np.int64))
    thresholds = np.broadcast_to(thresholds, (3, 3, len(truth)))
    for a in range(3):
        for s in range(3):
            mask = probability[a, s] >= thresholds[a, s, :, None, None]
            prediction = ref.classify(mask)
            union = (mask | target).sum((-2, -1))
            intersection = (mask & target).sum((-2, -1))
            iou = np.divide(intersection, union, out=np.ones(len(union)), where=union > 0)
            matrix = confusion(truth, prediction)
            output['prediction'][a, s] = prediction
            output['packed'][a, s] = np.packbits(mask.reshape(len(truth), 144), axis=1)
            output['iou'][a, s] = iou
            output['confusion'][a, s] = matrix
            output['F1'][a, s] = macro_from_confusion(matrix)
    return output


def reference_calibration(probability, target, truth, frames, rawkeys):
    design = plan(frames, rawkeys)
    pool = design['calibration_pool_indices']
    classes = np.empty((3, 3, 81, 640), np.int64)
    packed_pool = np.empty((3, 3, 81, 640, 18), np.uint8)
    scores = np.empty((3, 3, 5, 81), np.float64)
    selected = np.empty((3, 3, 5), np.int64)
    threshold_per_example = np.empty((3, 3, len(truth)), np.float64)
    for a in range(3):
        for s in range(3):
            probabilities = probability[a, s, pool]
            for t, threshold in enumerate(THRESHOLDS):
                masks = probabilities >= threshold
                classes[a, s, t] = ref.classify(masks)
                packed_pool[a, s, t] = np.packbits(masks.reshape(640, 144), axis=1)
            for f in range(5):
                calibration = design['calibration_indices'][f]
                indices = np.searchsorted(pool, calibration)
                assert np.array_equal(pool[indices], calibration)
                for t in range(81):
                    scores[a, s, f, t] = macro(truth[calibration], classes[a, s, t, indices])
                selected[a, s, f] = selection_index(scores[a, s, f])
                threshold_per_example[a, s, design['heldout_indices'][f]] = THRESHOLDS[selected[a, s, f]]
    held = mask_metrics(probability, target, truth, threshold_per_example)
    fold_f1, fold_iou = np.empty((3, 3, 5)), np.empty((3, 3, 5))
    fold_confusion = np.empty((3, 3, 5, 9, 9), np.int64)
    for a in range(3):
        for s in range(3):
            for f in range(5):
                ids = design['heldout_indices'][f]
                fold_f1[a, s, f] = macro(truth[ids], held['prediction'][a, s, ids])
                fold_iou[a, s, f] = held['iou'][a, s, ids].mean()
                fold_confusion[a, s, f] = confusion(truth[ids], held['prediction'][a, s, ids])
    return dict(**design, thresholds=THRESHOLDS, calibration_pool_classes=classes, calibration_pool_packed_masks=packed_pool,
        calibration_scores=scores, selected_threshold_indices=selected,
        selected_thresholds=THRESHOLDS[selected], oof_prediction=held['prediction'],
        oof_packed_masks=held['packed'], oof_IoU=held['iou'], oof_confusion=held['confusion'],
        oof_F1=held['F1'], oof_mean_IoU=held['iou'].mean(-1),
        fold_confusion=fold_confusion, fold_F1=fold_f1, fold_IoU=fold_iou,
        calibration_class_counts=np.array([np.bincount(truth[ids], minlength=9)
                                          for ids in design['calibration_indices']]))


def expected_summary(ap, calibration, fixed, old):
    cells = {}
    for a, architecture in enumerate(ARMS):
        cells[architecture] = dict(
            mean_frame_AP=float(ap['mean_frame_ap'][a].mean()),
            pooled_pixel_AP=float(ap['pooled_pixel_ap'][a].mean()),
            OOF_F1=float(calibration['oof_F1'][a].mean()),
            OOF_IoU=float(calibration['oof_mean_IoU'][a].mean()),
            fixed05_F1=float(fixed['F1'][a].mean()), fixed05_IoU=float(fixed['iou'][a].mean()),
            old_VAL_threshold_F1=float(old['F1'][a].mean()), old_VAL_threshold_IoU=float(old['iou'][a].mean()),
            per_seed=[dict(seed=seed, mean_frame_AP=float(ap['mean_frame_ap'][a, s]),
                pooled_pixel_AP=float(ap['pooled_pixel_ap'][a, s]), OOF_F1=float(calibration['oof_F1'][a, s]),
                OOF_IoU=float(calibration['oof_mean_IoU'][a, s]), fixed05_F1=float(fixed['F1'][a, s]),
                old_VAL_threshold_F1=float(old['F1'][a, s]), thresholds=calibration['selected_thresholds'][a, s].tolist())
                for s, seed in enumerate(SEEDS)])
    contrasts = {}
    metrics = ['mean_frame_AP', 'pooled_pixel_AP', 'OOF_F1', 'OOF_IoU', 'fixed05_F1', 'fixed05_IoU',
               'old_VAL_threshold_F1', 'old_VAL_threshold_IoU']
    for name, reference in [('group_minus_untied', 1), ('group_minus_narrow', 2)]:
        contrasts[name] = dict(
            mean={m: cells[ARMS[0]][m] - cells[ARMS[reference]][m] for m in metrics},
            per_seed=[dict(seed=seed, mean_frame_AP=float(ap['mean_frame_ap'][0, s] - ap['mean_frame_ap'][reference, s]),
                OOF_F1=float(calibration['oof_F1'][0, s] - calibration['oof_F1'][reference, s]),
                OOF_IoU=float(calibration['oof_mean_IoU'][0, s] - calibration['oof_mean_IoU'][reference, s]))
                for s, seed in enumerate(SEEDS)],
            per_fold=[dict(fold=f, F1=float((calibration['fold_F1'][0, :, f] - calibration['fold_F1'][reference, :, f]).mean()),
                IoU=float((calibration['fold_IoU'][0, :, f] - calibration['fold_IoU'][reference, :, f]).mean()))
                for f in range(5)])
    effect = contrasts['group_minus_untied']
    guards = dict(mean_frame_AP=effect['mean']['mean_frame_AP'] > 0.,
                  OOF_F1=effect['mean']['OOF_F1'] >= .005,
                  OOF_IoU=effect['mean']['OOF_IoU'] >= -.002,
                  each_fold_F1=min(row['F1'] for row in effect['per_fold']) >= -.01)
    eligibility = dict(passed=all(guards.values()), guards=guards,
        failed=[key for key, value in guards.items() if not value],
        scope='descriptive_development_eligibility_only_not_independent_confirmation')
    return dict(cells=cells, contrasts=contrasts), eligibility


def array_errors(expected, observed, prefix=''):
    assert set(expected) <= set(observed), (prefix, set(expected) - set(observed))
    errors = {}
    for name, value in expected.items():
        value, actual = np.asarray(value), np.asarray(observed[name])
        assert value.shape == actual.shape, (prefix + name, value.shape, actual.shape)
        if value.dtype.kind in 'biuUS':
            errors[prefix + name] = 0. if np.array_equal(value, actual) else float('inf')
        else:
            errors[prefix + name] = float(np.abs(value - actual).max()) if value.size else 0.
    return errors


def final_checks(checks, details, settings, models, data):
    inference = read(ROOT / 'INFERENCE.json'); result = read(ROOT / 'RESULT.json')
    checks['nine_fixed_models_and_no_new_training_or_confirmation'] = (
        inference['status'] == 'COMPLETE_FROZEN_DEVELOPMENT_INFERENCE' and inference['models'] == 9 and
        inference['frames'] == 40 and inference['examples'] == 10240 and
        inference['fit_count'] == inference['optimizer_updates'] == 0 and not inference['independent_confirmation'] and
        result['status'] == 'COMPLETE_CALIBRATION_DEVELOPMENT_ANALYSIS' and
        result['fit_count'] == result['optimizer_updates'] == 0 and not result['new_confirmation'] and
        result['examples'] == 10240 and result['frames'] == 40 and result['folds'] == 5 and
        result['calibration_labels_per_fold'] == 512)
    checks['inference_analysis_and_all_output_hashes_bound_to_current_inputs'] = (
        inference['PREDICTIONS_sha256'] == sha(ROOT / 'PREDICTIONS.npz') and
        inference['DATA_sha256'] == sha(ROOT / 'DATA.npz') and inference['LOCK_sha256'] == sha(ROOT / 'LOCK.json') and
        result['PRAUC_sha256'] == sha(ROOT / 'PRAUC.npz') and
        result['CALIBRATION_sha256'] == sha(ROOT / 'CALIBRATION.npz') and
        result['PREDICTIONS_sha256'] == sha(ROOT / 'PREDICTIONS.npz') and
        result['LOCK_sha256'] == sha(ROOT / 'LOCK.json') and
        result['ANALYSIS_FROZEN_sha256'] == sha(ROOT / 'ANALYSIS_FROZEN.json') and
        all(sha(resolve(name)) == expected for name, expected in inference['input_sha256'].items()) and
        all(sha(resolve(name)) == expected for name, expected in read(ROOT / 'ANALYSIS_FROZEN.json')['input_sha256'].items()))
    with np.load(ROOT / 'PREDICTIONS.npz', allow_pickle=False) as z:
        probability = z['probability'].copy()
        correct = (np.array_equal(z['truth'], data['label']) and np.array_equal(z['target'], data['Y']) and
            np.array_equal(z['frames'], data['frame']) and np.array_equal(z['raw_key'], data['rawkeys']) and
            np.array_equal(z['raw_D4_sha256'], data['rawkeys']) and np.array_equal(z['architectures'], ARMS) and
            np.array_equal(z['seeds'], SEEDS) and np.array_equal(z['source_cohort'], data['source_cohort']))
    checks['all_saved_probabilities_axes_targets_and_rawkeys_match_fixed_cohort'] = (
        correct and probability.shape == (3, 3, 10240, 12, 12) and np.isfinite(probability).all() and
        probability.min() >= 0 and probability.max() <= 1)
    # The six Group/Plain routes have independent historical whole-cohort
    # probability archives. The three new Untied routes are recomputed on CPU.
    archive_error, cpu_error = 0., 0.
    offset = 0
    for cohort, folder in enumerate(COHORTS):
        with np.load(folder / 'PREDICTIONS.npz', allow_pickle=False) as z:
            count = len(z['truth'])
            for a, architecture in [(0, 'group6'), (2, 'plain12_C4pooled')]:
                route = (0 if a == 0 else 1) if cohort == 0 else (2 if a == 0 else 3)
                archive_error = max(archive_error, float(np.abs(
                    probability[a, :, offset:offset + count] - z['probability'][route, 1]).max()))
        offset += count
    x = torch.tensor(5 * (data['X'] - .2))
    for s, seed in enumerate(SEEDS):
        row = next(row for row in models if row['architecture'] == 'untied24_single' and row['seed'] == seed)
        state = load(resolve(row['prediction_model_file']))['model']
        with torch.no_grad():
            parts = [ref.reference_plain(part, state).sigmoid().numpy()[:, 0] for part in x.split(64)]
        cpu_error = max(cpu_error, float(np.abs(np.concatenate(parts) - probability[1, s]).max()))
    checks['all_historical_Group_Plain_and_new_Untied_CPU_probabilities_agree'] = max(archive_error, cpu_error) <= 1e-5
    details['maximum_historical_probability_error'] = archive_error
    details['maximum_Untied_CPU_vs_MPS_probability_error'] = cpu_error

    ap = reference_ap(probability, data['Y'], data['frame'])
    with np.load(G / 'sem_architecture_pilot_20261102' / 'TRAIN_VAL.npz', allow_pickle=False) as z:
        ids = np.flatnonzero(z['role'] == 'val')
        old_frames, old_target = z['frame'][ids], z['Y'][ids, 0].astype(bool)
    old_probability = np.empty((3, 3, 695, 12, 12), np.float32)
    old_thresholds = np.empty((3, 3))
    for a, architecture in enumerate(ARMS):
        for s, seed in enumerate(SEEDS):
            row = next(row for row in models if row['architecture'] == architecture and row['seed'] == seed)
            with np.load(resolve(row['source_VAL_file']), allow_pickle=False) as z:
                old_probability[a, s] = z['probability'][:, 0]
            old_thresholds[a, s] = row['primary_fine_threshold']
    old_ap = reference_ap(old_probability, old_target, old_frames)
    ap.update(old_VAL_frame_values=old_ap['frame_values'], old_VAL_per_frame_ap=old_ap['per_frame_ap'],
        old_VAL_mean_frame_ap=old_ap['mean_frame_ap'], old_VAL_pooled_pixel_ap=old_ap['pooled_pixel_ap'],
        old_VAL_frame_metal_prevalence=old_ap['frame_metal_prevalence'])
    with np.load(ROOT / 'PRAUC.npz', allow_pickle=False) as z:
        saved_ap = {key: z[key].copy() for key in z.files}
    checks['AP_artifact_architecture_and_seed_axes_match'] = (
        np.array_equal(saved_ap['architectures'], ARMS) and np.array_equal(saved_ap['seeds'], SEEDS))
    ap_errors = array_errors(ap, saved_ap)
    checks['all_360_frame_AP_values_pooled_AP_and_prevalence_match_rank_tie_reference'] = max(ap_errors.values()) < 1e-12

    calibration = reference_calibration(probability, data['Y'], data['label'], data['frame'], data['rawkeys'])
    fixed = mask_metrics(probability, data['Y'], data['label'], np.array(.5))
    old = mask_metrics(probability, data['Y'], data['label'], old_thresholds[:, :, None])
    expected = dict(calibration, truth=data['label'], target=data['Y'], frames=data['frame'], old_VAL_thresholds=old_thresholds)
    for prefix, value in [('fixed05_', fixed), ('old_VAL_threshold_', old)]:
        expected.update({prefix + 'prediction': value['prediction'], prefix + 'packed_masks': value['packed'],
            prefix + 'IoU': value['iou'], prefix + 'confusion': value['confusion'],
            prefix + 'F1': value['F1'], prefix + 'mean_IoU': value['iou'].mean(-1)})
    with np.load(ROOT / 'CALIBRATION.npz', allow_pickle=False) as z:
        saved = {key: z[key].copy() for key in z.files}
    checks['calibration_artifact_architecture_and_seed_axes_match'] = (
        np.array_equal(saved['architectures'], ARMS) and np.array_equal(saved['seeds'], SEEDS))
    errors = array_errors(expected, saved)
    groups = {
        'all_frame_folds_shared_512_labels_and_640_candidate_indices_exact': list(plan(data['frame'], data['rawkeys'])),
        'all81_calibration_pool_graph_classes_and_packed_masks_exact': ['calibration_pool_classes', 'calibration_pool_packed_masks'],
        'all405_model_fold_calibration_scores_and_selected_class_only_thresholds_exact':
            ['calibration_scores', 'selected_threshold_indices', 'selected_thresholds', 'calibration_class_counts'],
        'all_OOF_graph_classes_masks_IoU_confusions_and_seed_F1_exact':
            ['oof_prediction', 'oof_packed_masks', 'oof_IoU', 'oof_confusion', 'oof_F1', 'oof_mean_IoU',
             'fold_confusion', 'fold_F1', 'fold_IoU'],
        'fixed_half_and_archived_VAL_threshold_baselines_match':
            [key for key in expected if key.startswith('fixed05_') or key.startswith('old_VAL_threshold_')],
    }
    for name, keys in groups.items(): checks[name] = max(errors[key] for key in keys) < 1e-12
    summary, eligibility = expected_summary(ap, calibration, fixed, old)
    summary_error = max_error(summary, {key: result[key] for key in ['cells', 'contrasts']})
    eligibility_error = max_error(eligibility, result['next_study_eligibility'])
    checks['all_model_seed_fold_contrasts_use_per_seed_F1_before_averaging'] = summary_error < 1e-12
    checks['future_study_eligibility_requires_all_four_predeclared_guards'] = eligibility_error < 1e-12
    checks['missing_calibration_classes_recorded_without_resampling'] = (
        result['calibration_class_counts'] == calibration['calibration_class_counts'].tolist() and
        result['missing_calibration_classes'] == [np.flatnonzero(row == 0).tolist()
                                                 for row in calibration['calibration_class_counts']])
    checks['explicit_seen_frame_development_scope_AP_definition_and_no_new_confirmation'] = (
        result['AP_definition'] == 'noninterpolated_average_precision_macro_frame_then_seed' and
        result['calibration_tie_rule'] == 'MacroF1_then_nearest_0.5_then_lower; no IoU or pixel mask labels' and
        result['frame_folds'] == {str(int(f)): int(k) for f, k in zip(calibration['frame_values'], calibration['frame_fold'])} and
        'previously_evaluated_frames_now_development_only' in result['caveats'] and
        'no_new_independent_confirmation' in result['caveats'] and
        'AP_is_segmentation_not_classification_F1' in result['caveats'])
    details.update(maximum_AP_error=max(ap_errors.values()), maximum_calibration_array_error=max(errors.values()),
        maximum_summary_error=summary_error, maximum_eligibility_error=eligibility_error,
        group_minus_untied=summary['contrasts']['group_minus_untied'], next_study_eligibility=eligibility)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(2); started = time.monotonic()
    checks, details = {}, {}
    c, settings, models, data = source_checks(checks, details)
    synthetic_checks(checks, details, settings)
    design_checks(checks, details, settings, data)
    preflight_checks(checks, details, models)
    if not args.preflight:
        final_checks(checks, details, settings, models, data)
    inputs = {name: row['sha256'] for name, row in c.protected_inputs().items()}
    for name in ['LOCK.json', 'PREFLIGHT.json', 'PREFLIGHT_ARRAYS.npz', 'DATA.npz', 'SOURCES.json']:
        path = ROOT / name
        if path.exists(): inputs[rel(path)] = sha(path)
    inputs[rel(Path(__file__))] = sha(Path(__file__))
    inputs[rel(STAGE / 'audit.py')] = sha(STAGE / 'audit.py')
    if not args.preflight:
        for name in ['PREDICTIONS.npz', 'INFERENCE.json', 'INFERENCE_FROZEN.json', 'ANALYSIS_FROZEN.json',
                     'ANALYSIS_PREFLIGHT.json', 'RESULT.json', 'PRAUC.npz', 'CALIBRATION.npz', 'PREFLIGHT_AUDIT.json']:
            path = ROOT / name
            if path.exists(): inputs[rel(path)] = sha(path)
    checks['all_audited_input_hashes_current_and_immutable'] = all(sha(resolve(name)) == value for name, value in inputs.items())
    status = 'PASS' if all(checks.values()) else 'FAIL'
    result = dict(status=status + ('_CALIBRATION_PREFLIGHT_AUDIT' if args.preflight else '_CALIBRATION_AUDIT'), checks=checks,
        checks_passed=sum(bool(value) for value in checks.values()), checks_total=len(checks),
        details=details, input_sha256=inputs, optimizer_updates=0, model_fitting_performed=False,
        unseen_target_access=False, scientific_scope='DESCRIPTIVE_SEEN_FRAME_DEVELOPMENT_ONLY',
        seconds=time.monotonic() - started)
    folder = ROOT / 'audit_sources'; folder.mkdir(exist_ok=True)
    (folder / f'audit_{sha(Path(__file__))}.py').write_bytes(Path(__file__).read_bytes())
    save(ROOT / ('PREFLIGHT_AUDIT.json' if args.preflight else 'AUDIT.json'), result)
    print(json.dumps(dict(status=result['status'], checks_passed=result['checks_passed'],
                         checks_total=result['checks_total'], seconds=result['seconds'])))
    assert status == 'PASS'


if __name__ == '__main__':
    with threadpool_limits(limits=2): main()
