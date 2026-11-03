"""Unchanged training/data operators for paired seed replication; no import-time test reads."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import numpy as np
import torch

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
ROOT = Path(__file__).resolve().parent
G, REPO = ROOT.parent, ROOT.parents[1]
OLD = G / 'sem_arch_calibrated_confirmation_20261103'
DEV = G / 'sem_arch_calibration_dev_20261103'
STAGE = G / 'sem_arch_groupcnn_20261102'
PILOT = G / 'sem_architecture_pilot_20261102'
UNTIED = G / 'sem_arch_untied_control_20261103'
ARCHITECTURES = ['group6', 'untied24_single']
NEW_SEEDS, OLD_SEEDS = [5801, 5821], [5701, 5721, 5757]
SEEDS, MODES = OLD_SEEDS + NEW_SEEDS, ['final_calibrated', 'fixed05']

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result

legacy = module('seed_replication_historical_untied_common', UNTIED / 'common.py')
model, historical = legacy.model, legacy.historical
calibrated = module('seed_replication_original_calibrated_common', OLD / 'common.py')

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def relative(path): return str(Path(path).resolve().relative_to(REPO))
def resolve(path): return REPO / path
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temporary.replace(path)

def configuration():
    value = read(ROOT / 'CONFIG.json')
    assert value['new_seeds'] == NEW_SEEDS and value['old_seeds'] == OLD_SEEDS
    assert value['architectures'] == ARCHITECTURES and value['budget'] == 27
    assert value['steps'] == 2000 and value['warmup_steps'] == 400 and value['batch_size'] == 8
    assert value['learning_rate'] == .0005 and value['weight_decay'] == .0001
    assert value['batch_seed_offset'] == 1000 and value['physics_weight'] == 1.
    assert value['thresholds'] == [round(.1 + .01*i, 2) for i in range(81)]
    assert value['bootstrap_repeats'] == 10000 and value['bootstrap_seed'] == 202611035
    assert value['CI_quantiles'] == [.0125, .9875]
    assert value['protocol_sha256'] == sha(ROOT / 'PROTOCOL.md')
    return value

def protected_inputs():
    paths = [ROOT / name for name in ['common.py','runner.py','evaluate.py','report.py','CONFIG.json','PROTOCOL.md']]
    paths += [UNTIED / 'common.py', STAGE / 'model.py', STAGE / 'runner.py', STAGE / 'CONFIG.json',
              PILOT / 'TRAIN_VAL.npz', PILOT / 'SUBSETS.json', PILOT / 'physics_adapter.py',
              PILOT / 'prepare.py', G / 'sem_physics_materials/physics_loss_example.py',
              G / 'sem_confirmation_20261028/prepare.py', G / 'retained_evidence_20261027/SELECTED_SOURCES.zip',
              OLD / 'common.py', OLD / 'stats.py', OLD / 'CONFIG.json', OLD / 'RESULT.json',
              OLD / 'AUDIT.json', OLD / 'EVAL.npz', OLD / 'PREDICTIONS.npz', OLD / 'BOOTSTRAP.npz',
              OLD / 'FINAL_CALIBRATION.json', OLD / 'FINAL_CALIBRATION.npz', DEV / 'DATA.npz']
    return {relative(path): sha(path) for path in sorted(set(paths))}

def verify_lock():
    value = read(ROOT / 'LOCK.json')
    assert value['input_sha256'] == protected_inputs()
    assert value['preflight_sha256'] == sha(ROOT / 'PREFLIGHT.json')
    assert read(ROOT / 'PREFLIGHT.json')['status'] == 'PASS_SEED_REPLICATION_PREFLIGHT'
    return value

def pair(seed):
    assert seed in NEW_SEEDS
    group = model.make_model('group6', seed)
    untied = group.deploy().requires_grad_(True)
    return group, untied

def data(device):
    return legacy.data(device)

def objective(network, dx, wx, target, labels, joint):
    from torch.nn import functional as F
    logits = network(torch.cat((dx, wx)) if joint else dx)
    dense, weak = logits[:len(dx)], logits[len(dx):] if joint else None
    bce = F.binary_cross_entropy_with_logits(dense, target)
    geometry = None if weak is None else historical.physics.geometry_loss(weak, torch.stack((labels//3, labels%3), 1))
    return bce if geometry is None else bce + geometry, dense, weak, bce, geometry

def hard_classes(masks): return calibrated.hard_classes(masks)
def macro(confusion): return calibrated.macro_from_confusion(confusion)
