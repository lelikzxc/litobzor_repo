"""Shared prospective-confirmation utilities; importing this file reads no data."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import re
import os
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
G, REPO = ROOT.parent, ROOT.parents[1]
DEV = G / 'sem_arch_groupcnn_20261102'
OLD_CONFIRM = G / 'sem_arch_orbit_confirmation_20261102'
PILOT = G / 'sem_architecture_pilot_20261102'
SHARED = G / 'shared_data/ic_sem_m2'
ORIGINAL = G / 'sem_confirmation_20261028'
PRIOR = [ORIGINAL, G / 'sem_cldice_confirmation_20261030', G / 'sem_cldice_expansion_20261102',
         G / 'sem_scarcity_20261102', PILOT, OLD_CONFIRM]
ROUTES = ['group6_single', 'plain12_C4', 'plain24_C4']
BUDGETS, SEEDS, MODES = [18, 27], [5701, 5721, 5757], ['primary_fine', 'secondary_seven']
FRAME_PREFIX = 'SEM-GROUP-CNN-CONFIRM-20261102/'
EVAL_PREFIX = 'SEM-GROUP-CNN-EVAL-20261102/'
CLASS_CACHE = {}


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
def resolve(path): return Path(path) if Path(path).is_absolute() else REPO / path


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temp.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def configuration():
    c = read(ROOT / 'CONFIG.json')
    assert c['routes'] == ROUTES and c['budgets'] == BUDGETS and c['seeds'] == SEEDS
    assert c['frames'] == 16 and c['evaluation_per_frame'] == 256 and c['evaluation_examples'] == 4096
    assert c['bootstrap_repeats'] == 10000 and c['CI_quantiles'] == [.0125, .9875]
    assert c['frame_prefix'] == FRAME_PREFIX and c['evaluation_prefix'] == EVAL_PREFIX
    assert c['protocol_sha256'] == sha(ROOT / 'PROTOCOL.md')
    return c


def code_hashes():
    names = ['confirmation_common.py', 'confirmation_stats.py', 'selection.py', 'prepare.py', 'evaluate.py']
    paths = [ROOT / name for name in names] + [DEV / 'model.py', PILOT / 'prepare.py', OLD_CONFIRM / 'prepare.py', OLD_CONFIRM / 'evaluate.py',
        ORIGINAL / 'prepare.py', G / 'retained_evidence_20261027/SELECTED_SOURCES.zip', SHARED / 'sem_remote_zip.py']
    return {str(path.relative_to(REPO)): sha(path) for path in paths}


def key(row): return row['route'], row['budget'], row['seed']


def development_selection(require_audit=True):
    """Only old-VAL development artifacts are inspected; no prior TEST results."""
    configuration()
    paths = list((DEV / 'trials').glob('*/AGGREGATE_RESULT.json'))
    assert len(paths) == 1, 'Require the unique preregistered replication aggregate'
    aggregate_path = paths[0]; aggregate = read(aggregate_path)
    assert aggregate['seeds'] == SEEDS and len(aggregate['rows']) == 30
    eligible = {name: bool(aggregate['gates'][name]['passed'] and aggregate['new_seed_gates'][name]['passed'])
                for name in ['A_matched_quality', 'B_full_efficiency']}
    assert eligible['A_matched_quality'] == aggregate['robust_primary_pass']
    assert eligible['B_full_efficiency'] == aggregate['robust_secondary_pass']
    assert any(eligible.values()) == aggregate['robust_ANY_GOAL_PASS']
    source_rows, provenance = {}, []
    for trial_name in aggregate['source_trials']:
        trial = DEV / trial_name; result_path = trial / 'RESULT.json'; audit_path = trial / 'AUDIT.json'
        result = read(result_path)
        assert result['status'] == 'COMPLETE_GROUPCNN_DEVELOPMENT'
        if require_audit:
            audit = read(audit_path); assert audit['status'] == 'PASS_AUDIT' and all(audit['checks'].values())
        provenance.append(dict(result_file=str(result_path.relative_to(REPO)), result_sha256=sha(result_path),
            audit_file=str(audit_path.relative_to(REPO)), audit_sha256=sha(audit_path) if audit_path.exists() else None))
        for row in result['rows']:
            if row['route'] not in ROUTES: continue
            assert key(row) not in source_rows
            source_rows[key(row)] = (row, trial)
    expected = {(route, b, s) for route in ROUTES for b in BUDGETS for s in SEEDS}
    assert set(source_rows) == expected
    aggregate_rows = {key(row): row for row in aggregate['rows'] if row['route'] in ROUTES}
    selected = []
    for route, budget, seed in sorted(expected, key=lambda t: (ROUTES.index(t[0]), BUDGETS.index(t[1]), SEEDS.index(t[2]))):
        row, trial = source_rows[route, budget, seed]
        assert row == aggregate_rows[route, budget, seed]
        assert row['selected'] == row['primary_fine']['selected']
        assert [v['threshold'] for v in row['primary_fine']['all_thresholds']] == [round(.1 + .01*i, 2) for i in range(81)]
        free_file = trial / row['model_file']; assert sha(free_file) == row['model_sha256']
        group = route == 'group6_single'
        prediction_file = trial / 'models' / f'group6_b{budget}_s{seed}_deployed.pt' if group else free_file
        prediction_count = 10681 if group or route == 'plain24_C4' else 2749
        prediction_mode = 'single' if group else 'C4'
        val_file = trial / 'val_probs' / f'{route}_b{budget}_s{seed}.npz'
        selected.append(dict(route=route, architecture=row['architecture'], budget=budget, seed=seed,
            free_model_file=str(free_file.relative_to(REPO)), free_model_sha256=sha(free_file),
            prediction_model_file=str(prediction_file.relative_to(REPO)), prediction_model_sha256=sha(prediction_file),
            prediction_width=24 if prediction_count == 10681 else 12, prediction_mode=prediction_mode,
            free_parameters=row['free_parameters'], stored_prediction_coefficients=prediction_count,
            primary_fine_threshold=row['selected']['threshold'], secondary_seven_threshold=row['secondary_seven']['selected']['threshold'],
            source_VAL_file=str(val_file.relative_to(REPO)), source_VAL_sha256=sha(val_file)))
    if require_audit: assert any(eligible.values()), 'No audited coherent same-goal replication eligibility'
    return dict(rows=selected, eligible_by_goal=eligible, aggregate_file=str(aggregate_path.relative_to(REPO)),
        aggregate_sha256=sha(aggregate_path), source_trials=provenance, source_model_count=18,
        deployment_artifact_count=6, interpretation='Development eligibility only; not fresh confirmation')


def build_source_plan():
    c = configuration(); previous = read(OLD_CONFIRM / 'SOURCE_PLAN.json'); previous_selection = read(OLD_CONFIRM / 'SELECTION.json')
    assert sha(OLD_CONFIRM / 'SOURCE_PLAN.json') == previous_selection['source_plan_sha256']
    excluded = set(previous['excluded_previous_frames']) | set(previous['test_frames'])
    assert len(previous['excluded_previous_frames']) == 86 and len(previous['test_frames']) == 8 and len(excluded) == 94
    listing_path = SHARED / 'sem_zip_listing.json'
    available = [int(match.group(1)) for row in read(listing_path)
                 if (match := re.fullmatch(r'dataset/sems/m2/sem(\d+)\.png', row['name']))]
    assert len(available) == len(set(available)) == 321
    chosen = sorted([n for n in available if n not in excluded],
        key=lambda n: hashlib.sha256((FRAME_PREFIX + str(n)).encode()).hexdigest())[:16]
    return dict(status='FROZEN_FRESH_DATA_PLAN', test_frames=chosen, selection_prefix=FRAME_PREFIX,
        excluded_previous_frames=sorted(excluded), excluded_frame_count=94, available_M2_frames=321,
        exclusion_source_file=str((OLD_CONFIRM / 'SOURCE_PLAN.json').relative_to(G)), exclusion_source_sha256=sha(OLD_CONFIRM / 'SOURCE_PLAN.json'),
        previous_selection_sha256=sha(OLD_CONFIRM / 'SELECTION.json'), archive_listing_file=str(listing_path.relative_to(G)),
        archive_listing_sha256=sha(listing_path), prior_manifests_sha256={str((folder/'DATA_MANIFEST.json').relative_to(G)):
            sha(folder/'DATA_MANIFEST.json') for folder in PRIOR},
        helper_prepare_sha256=sha(ORIGINAL/'prepare.py'),
        shared_sha256={name: sha(SHARED/name) for name in ['labels.zip','sem_remote_zip.py']},
        evaluation=dict(selection_prefix=EVAL_PREFIX, patches_per_frame=256, target_examples=4096,
            ranking='SHA256(prefix+raw_D4_sha256),first256/frame; no mask/class in ranking',
            minimum_examples_each_class=8,minimum_frames_each_class=3,replacement_on_gate_failure=False),
        protocol_sha256=sha(ROOT/'PROTOCOL.md'),config_sha256=sha(ROOT/'CONFIG.json'),prepare_script_sha256=sha(ROOT/'prepare.py'),
        reused_prepare_source_sha256=sha(OLD_CONFIRM/'prepare.py'),maximum_png_bytes_each=10_000_000,
        remote_zip_url='https://edmond.mpg.de/api/access/datafile/223856',
        scope='Six prior manifests used only for exact D4 exclusion; no old TEST prediction metrics inspected')


def verify_lock(require_fresh_data_absent=False):
    configuration(); selection = read(ROOT/'SELECTION.json'); lock = read(ROOT/'MODEL_LOCK.json'); plan = read(ROOT/'SOURCE_PLAN.json')
    assert selection['status'] == 'LOCKED_BEFORE_FRESH_DOWNLOAD' and lock['status'] == 'LOCKED_FINAL_MODELS_AND_OLD_VAL_THRESHOLDS'
    assert selection['model_lock_sha256'] == sha(ROOT/'MODEL_LOCK.json') and selection['rows'] == lock['rows']
    assert selection['source_plan_sha256'] == sha(ROOT/'SOURCE_PLAN.json') and selection['source_plan_sha256'] == lock['source_plan_sha256']
    assert selection['protocol_sha256'] == sha(ROOT/'PROTOCOL.md') and selection['config_sha256'] == sha(ROOT/'CONFIG.json')
    assert selection['code_sha256'] == code_hashes()
    assert selection['old_VAL_preflight_sha256'] == sha(ROOT/'OLD_VAL_PREFLIGHT.json')
    assert selection['old_VAL_preflight_arrays_sha256'] == sha(ROOT/'OLD_VAL_PREFLIGHT.npz')
    assert selection['development'] == development_selection()
    assert selection['rows'] == selection['development']['rows']
    assert selection['eligible_by_goal'] == selection['development']['eligible_by_goal']
    assert any(selection['eligible_by_goal'].values())
    for row in selection['rows']:
        for file_key, hash_key in [('free_model_file','free_model_sha256'),('prediction_model_file','prediction_model_sha256')]:
            assert sha(resolve(row[file_key])) == row[hash_key]
    for source in selection['development']['source_trials']:
        assert sha(resolve(source['result_file'])) == source['result_sha256']
        assert sha(resolve(source['audit_file'])) == source['audit_sha256']
    assert sha(resolve(selection['development']['aggregate_file'])) == selection['development']['aggregate_sha256']
    assert plan == build_source_plan()
    if require_fresh_data_absent:
        assert not any((ROOT/name).exists() for name in ['sources','TEST.npz','EVAL.npz','PREDICTIONS.npz'])
    return plan, selection


def load_network(row, device):
    model = module('group_confirmation_locked_factory', DEV/'model.py')
    payload = torch.load(resolve(row['prediction_model_file']),map_location='cpu',weights_only=True)
    network = model.make_plain(row['prediction_width']); network.load_state_dict(payload['model'],strict=True)
    assert model.parameters(network) == row['stored_prediction_coefficients']
    if row['route'] == 'group6_single': assert payload['free_model_sha256'] == row['free_model_sha256']
    return network.requires_grad_(False).to(device).eval(), model


def hard_classes(masks):
    masks=np.asarray(masks,bool).reshape(-1,12,12); keys=[r.tobytes() for r in np.packbits(masks.reshape(len(masks),144),axis=1)]
    missing=sorted(set(keys)-CLASS_CACHE.keys())
    if missing:
        helper=module('group_confirmation_unchanged_class_rule',PILOT/'prepare.py')
        values=np.array([np.unpackbits(np.frombuffer(k,np.uint8)).reshape(12,12) for k in missing],bool)
        CLASS_CACHE.update(zip(missing,helper.class_labels(values).tolist()))
    return np.array([CLASS_CACHE[k] for k in keys],np.int64)


@torch.no_grad()
def predict_models(rows, x, target, truth, device, progress=None):
    """One inference path shared by genuine OLD VAL preflight and locked evaluation."""
    config = configuration(); n = len(x)
    assert x.shape == (n,1,12,12) and target.shape == (n,12,12)
    probability = np.empty((3,2,3,n,12,12),np.float32)
    predictions = np.empty((2,3,2,3,n),np.int64)
    packed = np.empty((2,3,2,3,n,18),np.uint8)
    ious = np.empty((2,3,2,3,n),np.float64)
    thresholds = np.empty((2,3,2,3),np.float64)
    by_key = {key(row):row for row in rows}; metrics = []
    tensor = torch.tensor(5*(x-.2),device=device)
    for r,route in enumerate(ROUTES):
        for b,budget in enumerate(BUDGETS):
            for s,seed in enumerate(SEEDS):
                row = by_key[route,budget,seed]; net, model = load_network(row,device)
                started = time.monotonic()
                parts = [model.predict_probability(net,part,row['prediction_mode']).cpu().numpy()
                    for part in tensor.split(config['inference_batch_size'])]
                prob = np.concatenate(parts)[:,0]; probability[r,b,s] = prob
                assert np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1
                result = dict(route=route,budget=budget,seed=seed,
                    prediction_model_sha256=row['prediction_model_sha256'],metrics={},
                    CNN_passes_per_image=1 if r == 0 else 4,inference_and_transfer_seconds=time.monotonic()-started)
                for m,mode in enumerate(MODES):
                    threshold = row[mode+'_threshold']; thresholds[m,r,b,s] = threshold
                    mask = prob >= threshold; pred = hard_classes(mask); predictions[m,r,b,s] = pred
                    packed[m,r,b,s] = np.packbits(mask.reshape(n,144),axis=1)
                    union = (mask|target).sum((1,2)); intersection = (mask&target).sum((1,2))
                    iou = np.divide(intersection,union,out=np.ones(n),where=union>0); ious[m,r,b,s] = iou
                    confusion = np.zeros((9,9),np.int64); np.add.at(confusion,(truth,pred),1)
                    from confirmation_stats import macro_from_confusion
                    result['metrics'][mode] = dict(threshold=threshold,mask_F1=float(macro_from_confusion(confusion)),IoU=float(iou.mean()))
                metrics.append(result)
                if progress: progress(metrics)
                del net
    return dict(probability=probability,mask_predictions=predictions,packed_masks=packed,IoU=ious,
        thresholds=thresholds,truth=truth,target=target,modes=np.array(MODES),routes=np.array(ROUTES),
        budgets=np.array(BUDGETS),seeds=np.array(SEEDS)), metrics
