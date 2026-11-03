"""Locked final calibration and new-cohort utilities; no fitting or import-time data reads."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import re
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
G, REPO = ROOT.parent, ROOT.parents[1]
DEV = G / 'sem_arch_calibration_dev_20261103'
STAGE = G / 'sem_arch_groupcnn_20261102'
UNTIED = G / 'sem_arch_untied_control_20261103'
PILOT = G / 'sem_architecture_pilot_20261102'
SHARED = G / 'shared_data/ic_sem_m2'
ORIGINAL = G / 'sem_confirmation_20261028'
OLD_CONFIRM = G / 'sem_arch_orbit_confirmation_20261102'
PREVIOUS_GROUP = G / 'sem_arch_groupcnn_confirmation_20261102'
PREVIOUS_BCE = G / 'sem_arch_bce_confirmation_20261103'
PRIOR = [ORIGINAL, G / 'sem_cldice_confirmation_20261030',
         G / 'sem_cldice_expansion_20261102', G / 'sem_scarcity_20261102',
         PILOT, OLD_CONFIRM, PREVIOUS_GROUP, PREVIOUS_BCE]
ARCHITECTURES = ['group6', 'untied24_single', 'plain12_C4pooled']
SEEDS, MODES = [5701, 5721, 5757], ['final_calibrated', 'fixed05']
FRAME_PREFIX = 'SEM-C4-CALIB-CONFIRM-20261103/'
EVAL_PREFIX = 'SEM-C4-CALIB-EVAL-20261103/'
FINAL_CALIBRATION_FRAME_PREFIX = 'SEM-C4-FINAL-CALIB-FRAME-20261103/'
CALIBRATION_ROW_PREFIX = 'SEM-C4-CALIB-ROW-20261103/'
FREE_PARAMETERS = {'group6': 2671, 'untied24_single': 10681, 'plain12_C4pooled': 2749}
MACS = {'group6': 1527552, 'untied24_single': 1527552, 'plain12_C4pooled': 1562112}
CLASS_CACHE = {}


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
def relative(path): return str(Path(path).resolve().relative_to(REPO))
def resolve(path): return Path(path) if Path(path).is_absolute() else REPO / path


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def configuration():
    value = read(ROOT / 'CONFIG.json')
    assert value['architectures'] == ARCHITECTURES and value['seeds'] == SEEDS and value['budget'] == 27
    assert value['frames'] == 24 and value['evaluation_per_frame'] == 256 and value['evaluation_examples'] == 6144
    assert value['frame_prefix'] == FRAME_PREFIX and value['evaluation_prefix'] == EVAL_PREFIX
    assert value['final_calibration_frame_prefix'] == FINAL_CALIBRATION_FRAME_PREFIX
    assert value['calibration_row_prefix'] == CALIBRATION_ROW_PREFIX
    assert value['calibration_frames'] == 32 and value['calibration_per_frame'] == 16 and value['calibration_examples'] == 512
    assert value['thresholds'] == [round(.1 + .01 * i, 2) for i in range(81)]
    assert value['inference_batch_size'] == 16 and value['old_VAL_probability_tolerance'] == 1e-5
    assert value['bootstrap_repeats'] == 10000 and value['CI_quantiles'] == [.0125, .9875]
    assert value['protocol_sha256'] == sha(ROOT / 'PROTOCOL.md')
    return value


def code_hashes():
    paths = [ROOT / name for name in ['common.py', 'selection.py', 'evaluate.py', 'prepare.py', 'stats.py']]
    paths += [DEV / 'common.py', STAGE / 'model.py', PILOT / 'prepare.py', ORIGINAL / 'prepare.py',
              PREVIOUS_BCE / 'prepare.py', G / 'retained_evidence_20261027/SELECTED_SOURCES.zip',
              SHARED / 'sem_remote_zip.py']
    return {relative(path): sha(path) for path in paths}


def source_evidence():
    """Read prior DEV evidence and model descriptors, never a new image."""
    settings = configuration()
    result, audit = read(DEV / 'RESULT.json'), read(DEV / 'AUDIT.json')
    assert result['status'] == 'COMPLETE_CALIBRATION_DEVELOPMENT_ANALYSIS'
    assert result['next_study_eligibility']['passed'] and not result['new_confirmation']
    assert audit['status'] == 'PASS_CALIBRATION_AUDIT' and all(audit['checks'].values())
    names = ['RESULT.json', 'AUDIT.json', 'DATA.npz', 'CALIBRATION.npz', 'PREDICTIONS.npz',
             'LOCK.json', 'SOURCES.json', 'CONFIG.json', 'PROTOCOL.md', 'common.py']
    paths = [DEV / name for name in names]
    for name in ['RESULT.json', 'DATA.npz', 'CALIBRATION.npz', 'PREDICTIONS.npz', 'LOCK.json']:
        assert audit['input_sha256'][relative(DEV / name)] == sha(DEV / name)
    base = module('final_calibration_prior_development_common', DEV / 'common.py')
    rows = base.model_descriptors()
    assert len(rows) == 9
    for row in rows:
        for file_key, hash_key in [('free_model_file', 'free_model_sha256'),
                                   ('prediction_model_file', 'prediction_model_sha256'),
                                   ('source_VAL_file', 'source_VAL_sha256')]:
            path = resolve(row[file_key]); assert sha(path) == row[hash_key]
            assert audit['input_sha256'][relative(path)] == sha(path)
            paths.append(path)
    mechanism, mechanism_audit = read(UNTIED / 'RESULT.json'), read(UNTIED / 'AUDIT.json')
    assert mechanism_audit['status'].startswith('PASS') and all(mechanism_audit['checks'].values())
    mechanism_preflight = read(UNTIED / 'PREFLIGHT.json')
    assert mechanism_preflight['status'] == 'PASS_UNTIED_CPU_PREFLIGHT' and all(mechanism_preflight['checks'].values())
    assert read(UNTIED / 'LOCK.json')['preflight_sha256'] == sha(UNTIED / 'PREFLIGHT.json')
    assert mechanism_audit['input_sha256'][relative(UNTIED / 'LOCK.json')] == sha(UNTIED / 'LOCK.json')
    paths += [UNTIED / 'RESULT.json', UNTIED / 'AUDIT.json', UNTIED / 'PREFLIGHT.json', UNTIED / 'LOCK.json']
    initialization = []
    for fit in mechanism['fits']:
        assert fit['architecture'] == 'untied24_single' and fit['budget'] == 27
        item = dict(seed=fit['seed'])
        for file_key, hash_key in [('initial_file', 'initial_sha256'), ('group_initial_file', 'group_initial_sha256')]:
            path = resolve(fit[file_key]); assert sha(path) == fit[hash_key]
            assert mechanism_audit['input_sha256'][relative(path)] == sha(path)
            item[file_key], item[hash_key] = relative(path), sha(path); paths.append(path)
        initialization.append(item)
    assert sorted(row['seed'] for row in initialization) == SEEDS
    sources = {relative(path): sha(path) for path in sorted(set(paths))}
    return dict(rows=rows, source_sha256=sources, initialization_proof=initialization,
                development_eligible=True, source_model_count=9, deployment_artifact_count=3,
                source_development_result_sha256=sha(DEV / 'RESULT.json'),
                source_development_audit_sha256=sha(DEV / 'AUDIT.json'))


def macro_from_confusion(confusion):
    denominator = confusion.sum(-1) + confusion.sum(-2)
    numerator = 2 * np.diagonal(confusion, axis1=-2, axis2=-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0).mean(-1)


def final_calibration():
    """Fit nine scalar thresholds on exactly the same 512 old DEV class labels."""
    settings = configuration(); evidence = source_evidence()
    with np.load(DEV / 'DATA.npz', allow_pickle=False) as z:
        frames, truth, raw = z['frame'].copy(), z['label'].copy(), z['raw_D4_sha256'].copy()
    values = np.unique(frames); assert len(values) == 40 and len(frames) == 10240
    chosen_frames = sorted(values.tolist(), key=lambda f: hashlib.sha256(
        (FINAL_CALIBRATION_FRAME_PREFIX + str(f)).encode()).hexdigest())[:32]
    ids = []
    for frame in chosen_frames:
        candidates = np.flatnonzero(frames == frame).tolist(); assert len(candidates) == 256
        candidates.sort(key=lambda i: hashlib.sha256((CALIBRATION_ROW_PREFIX + str(raw[i])).encode()).hexdigest())
        ids.extend(candidates[:16])
    ids = np.array(sorted(ids), np.int64); assert len(ids) == len(set(ids.tolist())) == 512
    thresholds = np.asarray(settings['thresholds'])
    with np.load(DEV / 'CALIBRATION.npz', allow_pickle=False) as z:
        pool = z['calibration_pool_indices'].copy()
        cached_classes = z['calibration_pool_classes'].copy()
        assert np.array_equal(z['thresholds'], thresholds)
        assert np.array_equal(z['truth'], truth) and np.array_equal(z['frames'], frames)
    assert cached_classes.shape == (3, 3, 81, 640) and len(set(pool.tolist())) == 640
    lookup = {int(index): position for position, index in enumerate(pool)}
    positions = np.array([lookup[int(i)] for i in ids], np.int64)
    scores = np.empty((3, 3, 81), np.float64)
    selected_indices = np.empty((3, 3), np.int64)
    selected_thresholds = np.empty((3, 3), np.float64)
    selected_rows = []
    source_rows = {(row['architecture'], row['seed']): row for row in evidence['rows']}
    for a, architecture in enumerate(ARCHITECTURES):
        for s, seed in enumerate(SEEDS):
            for t in range(81):
                confusion = np.bincount(9 * truth[ids] + cached_classes[a, s, t, positions], minlength=81).reshape(9, 9)
                scores[a, s, t] = macro_from_confusion(confusion)
            chosen = max(range(81), key=lambda t: (float(scores[a, s, t]), -abs(t - 40), -t))
            selected_indices[a, s], selected_thresholds[a, s] = chosen, thresholds[chosen]
            selected_rows.append(dict(source_rows[architecture, seed],
                free_parameters=FREE_PARAMETERS[architecture], MACs_per_image=MACS[architecture],
                stored_prediction_coefficients=10681 if architecture != 'plain12_C4pooled' else 2749,
                final_calibrated_threshold=float(thresholds[chosen]), fixed05_threshold=.5,
                final_calibration_F1=float(scores[a, s, chosen])))
    metadata = dict(status='FIXED_FINAL_CALIBRATION_FROM_OLD_DEVELOPMENT',
        calibration_examples=512, calibration_frames=32, frame_ids=chosen_frames,
        frame_selection_prefix=FINAL_CALIBRATION_FRAME_PREFIX, row_selection_prefix=CALIBRATION_ROW_PREFIX,
        data_indices=ids.tolist(), class_counts=np.bincount(truth[ids], minlength=9).tolist(),
        labels_shared_by_all_models=True, masks_or_IoU_used_for_selection=False,
        thresholds=thresholds.tolist(), selected_thresholds=selected_thresholds.tolist(),
        selected_indices=selected_indices.tolist(), rows=selected_rows,
        source_sha256=evidence['source_sha256'], development_eligible=True,
        initialization_proof=evidence['initialization_proof'],
        rule='MacroF1 on 512 class labels, then integer-hundredths distance to0.5, then lower threshold',
        fit_count=0, optimizer_updates=0, new_target_images_accessed=False)
    arrays = dict(data_indices=ids, frame_ids=np.array(chosen_frames), thresholds=thresholds,
        scores=scores, selected_indices=selected_indices, selected_thresholds=selected_thresholds,
        truth=truth[ids], source_pool_positions=positions)
    return metadata, arrays


def development_selection():
    metadata, arrays = final_calibration()
    return dict(rows=metadata['rows'], source_sha256=metadata['source_sha256'],
        final_calibration_digest=digest(metadata), development_eligible=True,
        initialization_proof=metadata['initialization_proof'], source_model_count=9,
        deployment_artifact_count=3, calibration_examples=512,
        interpretation='Frozen previously fitted CNNs and identical final class-label calibration budget')


def build_source_plan():
    configuration(); previous_file = PREVIOUS_BCE / 'SOURCE_PLAN.json'
    previous, previous_selection = read(previous_file), read(PREVIOUS_BCE / 'SELECTION.json')
    assert sha(previous_file) == previous_selection['source_plan_sha256']
    excluded = set(previous['excluded_previous_frames']) | set(previous['test_frames'])
    assert len(previous['excluded_previous_frames']) == 110 and len(previous['test_frames']) == 24 and len(excluded) == 134
    listing = SHARED / 'sem_zip_listing.json'
    available = [int(match.group(1)) for row in read(listing)
                 if (match := re.fullmatch(r'dataset/sems/m2/sem(\d+)\.png', row['name']))]
    assert len(available) == len(set(available)) == 321
    old_records = [row for folder in PRIOR for row in read(folder / 'DATA_MANIFEST.json')['records']]
    assert {int(row['frame']) for row in old_records} <= excluded
    chosen = sorted([frame for frame in available if frame not in excluded],
        key=lambda f: hashlib.sha256((FRAME_PREFIX + str(f)).encode()).hexdigest())[:24]
    return dict(status='FROZEN_FRESH_DATA_PLAN', test_frames=chosen, selection_prefix=FRAME_PREFIX,
        excluded_previous_frames=sorted(excluded), excluded_frame_count=134, available_M2_frames=321,
        exclusion_source_file=str(previous_file.relative_to(G)), exclusion_source_sha256=sha(previous_file),
        previous_selection_sha256=sha(PREVIOUS_BCE / 'SELECTION.json'),
        archive_listing_file=str(listing.relative_to(G)), archive_listing_sha256=sha(listing),
        prior_manifests_sha256={str((folder / 'DATA_MANIFEST.json').relative_to(G)):sha(folder / 'DATA_MANIFEST.json') for folder in PRIOR},
        helper_prepare_sha256=sha(ORIGINAL / 'prepare.py'),
        shared_sha256={name:sha(SHARED / name) for name in ['labels.zip', 'sem_remote_zip.py']},
        evaluation=dict(selection_prefix=EVAL_PREFIX, patches_per_frame=256, target_examples=6144,
            ranking='SHA256(prefix+raw_D4_sha256),first256/frame; no mask/class/model in ranking',
            minimum_examples_each_class=8, minimum_frames_each_class=3, replacement_on_gate_failure=False),
        protocol_sha256=sha(ROOT / 'PROTOCOL.md'), config_sha256=sha(ROOT / 'CONFIG.json'),
        prepare_script_sha256=sha(ROOT / 'prepare.py'), reused_prepare_source_sha256=sha(PREVIOUS_BCE / 'prepare.py'),
        maximum_png_bytes_each=10_000_000, remote_zip_url='https://edmond.mpg.de/api/access/datafile/223856',
        scope='Eight prior manifests only for exact exclusion; final calibration uses old DEV only')


def verify_lock(require_fresh_data_absent=False):
    configuration()
    selection, lock, plan = [read(ROOT / name) for name in ['SELECTION.json', 'MODEL_LOCK.json', 'SOURCE_PLAN.json']]
    assert selection['status'] == 'LOCKED_BEFORE_FRESH_DOWNLOAD'
    assert lock['status'] == 'LOCKED_FINAL_MODELS_AND_FINAL_CALIBRATION'
    assert selection['model_lock_sha256'] == sha(ROOT / 'MODEL_LOCK.json')
    assert selection['source_plan_sha256'] == sha(ROOT / 'SOURCE_PLAN.json') == lock['source_plan_sha256']
    assert selection['protocol_sha256'] == sha(ROOT / 'PROTOCOL.md') and selection['config_sha256'] == sha(ROOT / 'CONFIG.json')
    assert selection['code_sha256'] == code_hashes()
    assert selection['old_VAL_preflight_sha256'] == sha(ROOT / 'OLD_VAL_PREFLIGHT.json')
    assert selection['old_VAL_preflight_arrays_sha256'] == sha(ROOT / 'OLD_VAL_PREFLIGHT.npz')
    assert selection['final_calibration_sha256'] == sha(ROOT / 'FINAL_CALIBRATION.json')
    assert selection['final_calibration_arrays_sha256'] == sha(ROOT / 'FINAL_CALIBRATION.npz')
    development = development_selection()
    assert selection['development'] == lock['development'] == development
    assert selection['rows'] == lock['rows'] == development['rows']
    metadata, arrays = final_calibration()
    assert read(ROOT / 'FINAL_CALIBRATION.json') == metadata
    with np.load(ROOT / 'FINAL_CALIBRATION.npz', allow_pickle=False) as z:
        assert set(z.files) == set(arrays) and all(np.array_equal(z[name], value) for name, value in arrays.items())
    assert selection['development_eligible'] is True
    for row in selection['rows']:
        for file_key, hash_key in [('free_model_file','free_model_sha256'), ('prediction_model_file','prediction_model_sha256'), ('source_VAL_file','source_VAL_sha256')]:
            assert sha(resolve(row[file_key])) == row[hash_key]
    assert plan == build_source_plan()
    if require_fresh_data_absent:
        assert not any((ROOT / name).exists() for name in ['sources', 'TEST.npz', 'EVAL.npz', 'PREDICTIONS.npz'])
    return plan, selection


def load_network(row, device):
    factory = module('final_calibration_locked_factory', STAGE / 'model.py')
    payload = torch.load(resolve(row['prediction_model_file']), map_location='cpu', weights_only=True)
    network = factory.make_plain(row['prediction_width']); network.load_state_dict(payload['model'], strict=True)
    assert factory.parameters(network) == row['stored_prediction_coefficients']
    if row['architecture'] == 'group6': assert payload['free_model_sha256'] == row['free_model_sha256']
    return network.requires_grad_(False).to(device).eval(), factory


def hard_classes(masks):
    masks = np.asarray(masks, bool).reshape(-1, 12, 12)
    keys = [row.tobytes() for row in np.packbits(masks.reshape(len(masks), 144), axis=1)]
    missing = sorted(set(keys) - CLASS_CACHE.keys())
    if missing:
        helper = module('final_calibration_unchanged_class_rule', PILOT / 'prepare.py')
        values = np.array([np.unpackbits(np.frombuffer(key, np.uint8)).reshape(12, 12) for key in missing], bool)
        CLASS_CACHE.update(zip(missing, helper.class_labels(values).tolist()))
    return np.array([CLASS_CACHE[key] for key in keys], np.int64)


@torch.no_grad()
def predict_models(rows, x, target, truth, device, progress=None):
    settings = configuration(); n = len(x)
    assert x.shape == (n, 1, 12, 12) and target.shape == (n, 12, 12)
    probability = np.empty((3, 3, n, 12, 12), np.float32)
    predictions = np.empty((2, 3, 3, n), np.int64)
    packed = np.empty((2, 3, 3, n, 18), np.uint8)
    ious = np.empty((2, 3, 3, n), np.float64)
    thresholds = np.empty((2, 3, 3), np.float64)
    lookup = {(row['architecture'], row['seed']):row for row in rows}; metrics = []
    tensor = torch.tensor(5 * (x - .2), device=device)
    for a, architecture in enumerate(ARCHITECTURES):
        for s, seed in enumerate(SEEDS):
            row = lookup[architecture, seed]; network, factory = load_network(row, device); started = time.monotonic()
            prob = np.concatenate([factory.predict_probability(network, part, row['prediction_mode']).cpu().numpy()
                                   for part in tensor.split(settings['inference_batch_size'])])[:, 0]
            assert np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1
            probability[a, s] = prob
            result = dict(architecture=architecture, seed=seed, budget=27,
                prediction_model_sha256=row['prediction_model_sha256'], metrics={},
                CNN_passes_per_image=4 if architecture == 'plain12_C4pooled' else 1,
                inference_and_transfer_seconds=time.monotonic() - started)
            for m, mode in enumerate(MODES):
                threshold = row[mode + '_threshold']; thresholds[m, a, s] = threshold
                mask = prob >= threshold; prediction = hard_classes(mask); predictions[m, a, s] = prediction
                packed[m, a, s] = np.packbits(mask.reshape(n, 144), axis=1)
                union, intersection = (mask | target).sum((1, 2)), (mask & target).sum((1, 2))
                iou = np.divide(intersection, union, out=np.ones(n), where=union > 0); ious[m, a, s] = iou
                confusion = np.bincount(9 * truth + prediction, minlength=81).reshape(9, 9)
                result['metrics'][mode] = dict(threshold=threshold, mask_F1=float(macro_from_confusion(confusion)), IoU=float(iou.mean()))
            metrics.append(result)
            if progress: progress(metrics)
            del network
    return dict(probability=probability, mask_predictions=predictions, packed_masks=packed, IoU=ious,
        thresholds=thresholds, truth=truth, target=target, modes=np.array(MODES),
        architectures=np.array(ARCHITECTURES), seeds=np.array(SEEDS)), metrics
