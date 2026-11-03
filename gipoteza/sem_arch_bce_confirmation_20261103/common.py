"""Prospectively locked 2x2 confirmation utilities; no data access at import."""
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
DEV = G / 'sem_arch_groupcnn_20261102'
FACTORIAL = G / 'sem_arch_loss_factorial_20261103'
OLD_CONFIRM = G / 'sem_arch_orbit_confirmation_20261102'
PREVIOUS_GROUP = G / 'sem_arch_groupcnn_confirmation_20261102'
PILOT = G / 'sem_architecture_pilot_20261102'
SHARED = G / 'shared_data/ic_sem_m2'
ORIGINAL = G / 'sem_confirmation_20261028'
PRIOR = [ORIGINAL, G / 'sem_cldice_confirmation_20261030',
         G / 'sem_cldice_expansion_20261102', G / 'sem_scarcity_20261102',
         PILOT, OLD_CONFIRM, PREVIOUS_GROUP]
CELLS = ['BCE_group', 'BCE_plain', 'aux_group', 'aux_plain']
ROUTES = CELLS
BUDGETS, SEEDS, MODES = [18, 27], [5701, 5721, 5757], ['primary_fine', 'fixed05']
FRAME_PREFIX = 'SEM-BCE-CNN-CONFIRM-20261103/'
EVAL_PREFIX = 'SEM-BCE-CNN-EVAL-20261103/'
CLASS_CACHE = {}


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
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
    assert value['cells'] == CELLS and value['budgets'] == BUDGETS and value['seeds'] == SEEDS
    assert value['frames'] == 24 and value['evaluation_per_frame'] == 256 and value['evaluation_examples'] == 6144
    assert value['frame_prefix'] == FRAME_PREFIX and value['evaluation_prefix'] == EVAL_PREFIX
    assert value['bootstrap_repeats'] == 10000 and value['CI_quantiles'] == [.025, .975]
    assert value['inference_batch_size'] == 16 and value['old_VAL_probability_tolerance'] == 1e-5
    assert value['protocol_sha256'] == sha(ROOT / 'PROTOCOL.md')
    return value


def code_hashes():
    paths = [ROOT / name for name in ['common.py', 'selection.py', 'evaluate.py', 'prepare.py', 'stats.py']]
    paths += [DEV / 'model.py', PILOT / 'prepare.py', ORIGINAL / 'prepare.py',
              OLD_CONFIRM / 'prepare.py', PREVIOUS_GROUP / 'prepare.py',
              G / 'retained_evidence_20261027/SELECTED_SOURCES.zip', SHARED / 'sem_remote_zip.py']
    return {str(path.relative_to(REPO)): sha(path) for path in paths}


def key(row): return row['cell'], row['budget'], row['seed']


def development_selection():
    """Bind all 24 old-VAL fits and their existing threshold choices; no TEST reads."""
    configuration()
    result_file, audit_file = FACTORIAL / 'RESULT.json', FACTORIAL / 'AUDIT.json'
    result, audit = read(result_file), read(audit_file)
    assert result['status'] == 'COMPLETE_FACTORIAL_DEVELOPMENT'
    assert audit['status'] == 'PASS_FACTORIAL_AUDIT' and all(audit['checks'].values())
    assert result['new_fits'] == result['inherited_fits'] == 12 and len(result['rows']) == 24
    assert result['primary_development_gate']['passed']
    source_rows = {}
    for row in result['rows']:
        objective, architecture = row['objective'], row['architecture']
        assert objective in ['BCE', 'BCE_geometry']
        assert architecture in ['group6', 'plain12_C4pooled']
        cell = ('BCE_' if objective == 'BCE' else 'aux_') + ('group' if architecture == 'group6' else 'plain')
        identifier = (cell, row['budget'], row['seed'])
        assert identifier not in source_rows
        source_rows[identifier] = row
    expected = {(cell, budget, seed) for cell in CELLS for budget in BUDGETS for seed in SEEDS}
    assert set(source_rows) == expected
    selected = []
    for cell in CELLS:
        for budget in BUDGETS:
            for seed in SEEDS:
                row = source_rows[cell, budget, seed]
                assert [v['threshold'] for v in row['all_thresholds']] == [round(.1 + .01*i, 2) for i in range(81)]
                selected_score = max(row['all_thresholds'], key=lambda r:(r['mask_F1'],r['IoU'],-abs(r['threshold']-.5),-r['threshold']))
                assert selected_score == row['selected']
                free_file, val_file = resolve(row['model_file']), resolve(row['probability_file'])
                assert sha(free_file) == row['model_sha256'] and sha(val_file) == row['probability_sha256']
                group = row['architecture'] == 'group6'
                prediction_file = free_file.with_name(free_file.stem+'_deployed.pt') if group else free_file
                selected.append(dict(cell=cell, route=row['route'], architecture=row['architecture'],
                    objective=row['objective'], budget=budget, seed=seed,
                    free_model_file=str(free_file.relative_to(REPO)), free_model_sha256=sha(free_file),
                    prediction_model_file=str(prediction_file.relative_to(REPO)), prediction_model_sha256=sha(prediction_file),
                    prediction_width=24 if group else 12, prediction_mode='single' if group else 'C4',
                    free_parameters=2671 if group else 2749, stored_prediction_coefficients=10681 if group else 2749,
                    primary_fine_threshold=row['selected']['threshold'], fixed05_threshold=.5,
                    source_VAL_file=str(val_file.relative_to(REPO)), source_VAL_sha256=sha(val_file)))
    audit_inputs = audit['input_sha256']
    for path in [result_file] + [resolve(row[name]) for row in selected for name in ['free_model_file','source_VAL_file']]:
        assert audit_inputs[str(path.relative_to(REPO))] == sha(path)
    return dict(rows=selected, development_eligible=True,
        result_file=str(result_file.relative_to(REPO)), result_sha256=sha(result_file),
        audit_file=str(audit_file.relative_to(REPO)), audit_sha256=sha(audit_file),
        prior_lock_file=str((FACTORIAL/'LOCK.json').relative_to(REPO)), prior_lock_sha256=sha(FACTORIAL/'LOCK.json'),
        source_model_count=24, deployment_artifact_count=12,
        interpretation='All fixed factorial cells; no selection of best seed or fresh target score')


def build_source_plan():
    config = configuration()
    previous_file = PREVIOUS_GROUP / 'SOURCE_PLAN.json'
    previous, previous_selection = read(previous_file), read(PREVIOUS_GROUP / 'SELECTION.json')
    assert sha(previous_file) == previous_selection['source_plan_sha256']
    excluded = set(previous['excluded_previous_frames']) | set(previous['test_frames'])
    assert len(previous['excluded_previous_frames']) == 94 and len(previous['test_frames']) == 16 and len(excluded) == 110
    listing_path = SHARED / 'sem_zip_listing.json'
    available = [int(match.group(1)) for row in read(listing_path)
                 if (match := re.fullmatch(r'dataset/sems/m2/sem(\d+)\.png', row['name']))]
    assert len(available) == len(set(available)) == 321
    previous_records = [row for folder in PRIOR for row in read(folder/'DATA_MANIFEST.json')['records']]
    assert {int(row['frame']) for row in previous_records} <= excluded
    chosen = sorted([frame for frame in available if frame not in excluded],
        key=lambda n: hashlib.sha256((FRAME_PREFIX + str(n)).encode()).hexdigest())[:24]
    return dict(status='FROZEN_FRESH_DATA_PLAN', test_frames=chosen, selection_prefix=FRAME_PREFIX,
        excluded_previous_frames=sorted(excluded), excluded_frame_count=110, available_M2_frames=321,
        exclusion_source_file=str(previous_file.relative_to(G)), exclusion_source_sha256=sha(previous_file),
        previous_selection_sha256=sha(PREVIOUS_GROUP/'SELECTION.json'),
        archive_listing_file=str(listing_path.relative_to(G)), archive_listing_sha256=sha(listing_path),
        prior_manifests_sha256={str((folder/'DATA_MANIFEST.json').relative_to(G)):sha(folder/'DATA_MANIFEST.json') for folder in PRIOR},
        helper_prepare_sha256=sha(ORIGINAL/'prepare.py'),
        shared_sha256={name:sha(SHARED/name) for name in ['labels.zip','sem_remote_zip.py']},
        evaluation=dict(selection_prefix=EVAL_PREFIX, patches_per_frame=256, target_examples=6144,
            ranking='SHA256(prefix+raw_D4_sha256),first256/frame; no mask/class in ranking',
            minimum_examples_each_class=8, minimum_frames_each_class=3,replacement_on_gate_failure=False),
        protocol_sha256=sha(ROOT/'PROTOCOL.md'), config_sha256=sha(ROOT/'CONFIG.json'),
        prepare_script_sha256=sha(ROOT/'prepare.py'), reused_prepare_source_sha256=sha(PREVIOUS_GROUP/'prepare.py'),
        maximum_png_bytes_each=10_000_000, remote_zip_url='https://edmond.mpg.de/api/access/datafile/223856',
        scope='Seven prior manifests for exclusion only; no old TEST predictions/metrics inspected')


def verify_lock(require_fresh_data_absent=False):
    configuration()
    selection, lock, plan = [read(ROOT/name) for name in ['SELECTION.json','MODEL_LOCK.json','SOURCE_PLAN.json']]
    assert selection['status'] == 'LOCKED_BEFORE_FRESH_DOWNLOAD'
    assert lock['status'] == 'LOCKED_FINAL_MODELS_AND_OLD_VAL_THRESHOLDS'
    assert selection['model_lock_sha256'] == sha(ROOT/'MODEL_LOCK.json')
    assert selection['source_plan_sha256'] == sha(ROOT/'SOURCE_PLAN.json') == lock['source_plan_sha256']
    assert selection['protocol_sha256'] == sha(ROOT/'PROTOCOL.md') and selection['config_sha256'] == sha(ROOT/'CONFIG.json')
    assert selection['code_sha256'] == code_hashes()
    assert selection['old_VAL_preflight_sha256'] == sha(ROOT/'OLD_VAL_PREFLIGHT.json')
    assert selection['old_VAL_preflight_arrays_sha256'] == sha(ROOT/'OLD_VAL_PREFLIGHT.npz')
    development = development_selection()
    assert selection['development'] == lock['development'] == development
    assert selection['rows'] == lock['rows'] == development['rows']
    assert selection['development_eligible'] is True
    for row in selection['rows']:
        for file_key, hash_key in [('free_model_file','free_model_sha256'),('prediction_model_file','prediction_model_sha256'),('source_VAL_file','source_VAL_sha256')]:
            assert sha(resolve(row[file_key])) == row[hash_key]
    assert plan == build_source_plan()
    if require_fresh_data_absent:
        assert not any((ROOT/name).exists() for name in ['sources','TEST.npz','EVAL.npz','PREDICTIONS.npz'])
    return plan, selection


def load_network(row, device):
    factory = module('bce_confirmation_locked_factory', DEV/'model.py')
    payload = torch.load(resolve(row['prediction_model_file']),map_location='cpu',weights_only=True)
    network = factory.make_plain(row['prediction_width'])
    network.load_state_dict(payload['model'],strict=True)
    assert factory.parameters(network) == row['stored_prediction_coefficients']
    if row['architecture'] == 'group6':
        assert payload['free_model_sha256'] == row['free_model_sha256']
    return network.requires_grad_(False).to(device).eval(), factory


def hard_classes(masks):
    masks = np.asarray(masks,bool).reshape(-1,12,12)
    keys = [row.tobytes() for row in np.packbits(masks.reshape(len(masks),144),axis=1)]
    missing = sorted(set(keys)-CLASS_CACHE.keys())
    if missing:
        helper = module('bce_confirmation_unchanged_class_rule',PILOT/'prepare.py')
        values = np.array([np.unpackbits(np.frombuffer(k,np.uint8)).reshape(12,12) for k in missing],bool)
        CLASS_CACHE.update(zip(missing,helper.class_labels(values).tolist()))
    return np.array([CLASS_CACHE[key] for key in keys],np.int64)


@torch.no_grad()
def predict_models(rows, x, target, truth, device, progress=None):
    config = configuration(); n = len(x)
    assert x.shape == (n,1,12,12) and target.shape == (n,12,12)
    probability = np.empty((4,2,3,n,12,12),np.float32)
    predictions = np.empty((2,4,2,3,n),np.int64)
    packed = np.empty((2,4,2,3,n,18),np.uint8)
    ious = np.empty((2,4,2,3,n),np.float64)
    thresholds = np.empty((2,4,2,3),np.float64)
    by_key = {key(row):row for row in rows}; metrics = []
    tensor = torch.tensor(5*(x-.2),device=device)
    from stats import macro_from_confusion
    for r,cell in enumerate(CELLS):
        for b,budget in enumerate(BUDGETS):
            for s,seed in enumerate(SEEDS):
                row = by_key[cell,budget,seed]
                net, factory = load_network(row,device); started = time.monotonic()
                parts = [factory.predict_probability(net,part,row['prediction_mode']).cpu().numpy()
                         for part in tensor.split(config['inference_batch_size'])]
                prob = np.concatenate(parts)[:,0]; probability[r,b,s] = prob
                assert np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1
                result = dict(cell=cell,route=row['route'],objective=row['objective'],budget=budget,seed=seed,
                    prediction_model_sha256=row['prediction_model_sha256'],metrics={},
                    CNN_passes_per_image=1 if row['architecture']=='group6' else 4,
                    inference_and_transfer_seconds=time.monotonic()-started)
                for m,mode in enumerate(MODES):
                    threshold = row[mode+'_threshold']; thresholds[m,r,b,s] = threshold
                    mask = prob >= threshold; prediction = hard_classes(mask); predictions[m,r,b,s] = prediction
                    packed[m,r,b,s] = np.packbits(mask.reshape(n,144),axis=1)
                    union, intersection = (mask|target).sum((1,2)), (mask&target).sum((1,2))
                    iou = np.divide(intersection,union,out=np.ones(n),where=union>0); ious[m,r,b,s] = iou
                    confusion = np.zeros((9,9),np.int64); np.add.at(confusion,(truth,prediction),1)
                    result['metrics'][mode] = dict(threshold=threshold,mask_F1=float(macro_from_confusion(confusion)),IoU=float(iou.mean()))
                metrics.append(result)
                if progress: progress(metrics)
                del net
    return dict(probability=probability,mask_predictions=predictions,packed_masks=packed,IoU=ious,
        thresholds=thresholds,truth=truth,target=target,modes=np.array(MODES),cells=np.array(CELLS),
        budgets=np.array(BUDGETS),seeds=np.array(SEEDS)),metrics
