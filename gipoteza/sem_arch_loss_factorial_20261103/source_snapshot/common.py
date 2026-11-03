"""Frozen stage12 imports and old-TRAIN_VAL-only provenance for the ablation."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
STAGE = ROOT.parent / 'sem_arch_groupcnn_20261102'
PILOT = ROOT.parent / 'sem_architecture_pilot_20261102'
ARMS = ['group6', 'plain12_C4pooled']
ROUTES = {'group6': ('group6_single', 'single'), 'plain12_C4pooled': ('plain12_C4', 'C4')}
AGG = STAGE / 'trials/groupcnn_replicate_s5721-5757_metadata_recovery_27b9055b42/AGGREGATE_RESULT.json'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


model = module('factorial_frozen_groupcnn_model', STAGE / 'model.py')
old_model = sys.modules.get('model')
sys.modules['model'] = model
try:
    historical = module('factorial_frozen_groupcnn_runner', STAGE / 'runner.py')
finally:
    if old_model is None: del sys.modules['model']
    else: sys.modules['model'] = old_model


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def relative(path): return str(Path(path).resolve().relative_to(ROOT.parent.parent))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temp.replace(path)


def config():
    c = read(ROOT / 'CONFIG.json')
    assert c['architectures'] == ARMS and c['budgets'] == [18, 27] and c['seeds'] == [5701, 5721, 5757]
    assert c['steps'] == 2000 and c['warmup_steps'] == 400 and c['batch_size'] == 8
    assert c['learning_rate'] == .0005 and c['weight_decay'] == .0001 and c['batch_seed_offset'] == 1000
    assert c['thresholds'] == [round(.1 + .01*i, 2) for i in range(81)] and c['diagnostic_threshold'] == .5
    assert c['development_gate'] == dict(minimum_mean_F1_gain=.005, minimum_budget_F1_delta=-.0025,
        minimum_mean_IoU_delta=-.002, maximum_budget_ratio_deviation=.03)
    assert c['protocol_sha256'] == sha(ROOT / 'PROTOCOL.md')
    return c


def old_rows():
    aggregate = read(AGG); result = []
    assert aggregate['seeds'] == [5701, 5721, 5757]
    for trial in aggregate['source_trials']:
        folder = STAGE / trial; old = read(folder / 'RESULT.json'); audit = read(folder / 'AUDIT.json')
        assert audit['checks_passed'] == audit['checks_total']
        for fit in old['fits']:
            arch = fit['architecture']
            if arch not in ARMS: continue
            route, _ = ROUTES[arch]
            rows = [r for r in old['rows'] if (r['architecture'], r['budget'], r['seed'], r['route']) ==
                    (arch, fit['budget'], fit['seed'], route)]
            assert len(rows) == 1
            row = rows[0]
            prefix = f"{route}_b{fit['budget']}_s{fit['seed']}"
            final = folder / fit['model_file']; initial = folder / fit['initial_model_file']
            assert sha(final) == fit['model_sha256'] and sha(initial) == fit['initial_model_sha256']
            assert row['model_sha256'] == fit['model_sha256']
            result.append(dict(architecture=arch, route=route, budget=fit['budget'], seed=fit['seed'],
                row=row, fit=fit, folder=folder, model=final, initial=initial,
                probability=folder / 'val_probs' / f'{prefix}.npz'))
    assert len(result) == 12
    assert len({(v['architecture'], v['budget'], v['seed']) for v in result}) == 12
    return result


def protected_inputs():
    paths = [ROOT/'common.py', ROOT/'runner.py', ROOT/'CONFIG.json', ROOT/'PROTOCOL.md',
             STAGE/'model.py', STAGE/'runner.py', STAGE/'recovery.py', STAGE/'CONFIG.json', STAGE/'PROTOCOL.md',
             PILOT/'TRAIN_VAL.npz', PILOT/'SUBSETS.json', AGG]
    for name in historical.source_hashes(): paths.append(STAGE/name)
    for item in old_rows():
        folder, fit = item['folder'], item['fit']
        paths.extend([item['model'], item['initial'], item['probability'], folder/'RESULT.json', folder/'AUDIT.json',
                      folder/'FROZEN.json', folder/'CONFIG_SNAPSHOT.json', folder/'PROTOCOL.md'])
        paths.extend(folder/p['file'] for p in fit['gradient_probes'])
        if item['architecture'] == 'group6':
            paths.append(item['model'].with_name(item['model'].stem+'_deployed.pt'))
    unique = sorted(set(path.resolve() for path in paths))
    return {relative(path): dict(sha256=sha(path), bytes=path.stat().st_size) for path in unique}


def verify_lock():
    lock = read(ROOT/'LOCK.json'); current = protected_inputs()
    assert lock['inputs'] == current, 'Immutable source/model/data hash mismatch'
    assert lock['preflight_sha256'] == sha(ROOT/'PREFLIGHT.json')
    assert read(ROOT/'PREFLIGHT.json')['status'] == 'PASS_FACTORIAL_CPU_PREFLIGHT'
    return lock


def data(device):
    return historical.inputs(read(STAGE/'CONFIG.json'), device)


def matching_old(architecture, budget, seed):
    matches = [v for v in old_rows() if (v['architecture'], v['budget'], v['seed']) == (architecture, budget, seed)]
    assert len(matches) == 1
    return matches[0]
