"""Fixed models and previously inspected frames, explicitly reused as development."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO, G = ROOT.parents[1], ROOT.parent
STAGE = G / 'sem_arch_groupcnn_20261102'
PILOT = G / 'sem_architecture_pilot_20261102'
UNTIED = G / 'sem_arch_untied_control_20261103'
COHORTS = [G / 'sem_arch_groupcnn_confirmation_20261102', G / 'sem_arch_bce_confirmation_20261103']
ARCHITECTURES = ['group6', 'untied24_single', 'plain12_C4pooled']
SEEDS = [5701, 5721, 5757]
CLASS_CACHE = {}


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def relative(path): return str(Path(path).resolve().relative_to(REPO))
def resolve(path): return Path(path) if Path(path).is_absolute() else REPO / path
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def config():
    value = read(ROOT / 'CONFIG.json')
    assert value['architectures'] == ARCHITECTURES and value['seeds'] == SEEDS
    assert value.get('budget', 27) == 27 and value['frames'] == 40
    assert value['inference_batch_size'] == 16
    assert value['protocol_sha256'] == sha(ROOT / 'PROTOCOL.md')
    return value


def model_descriptors():
    result, audit = read(UNTIED / 'RESULT.json'), read(UNTIED / 'AUDIT.json')
    assert result['status'] == 'COMPLETE_UNTIED_CONTROL_DEVELOPMENT'
    assert audit['status'].startswith('PASS') and all(audit['checks'].values())
    assert audit['input_sha256'][relative(UNTIED / 'RESULT.json')] == sha(UNTIED / 'RESULT.json')
    lookup = {(row['architecture'], row['seed']): row for row in result['rows']}
    assert set(lookup) == {(a, s) for a in ARCHITECTURES for s in SEEDS}
    rows = []
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            source = lookup[architecture, seed]
            assert source['budget'] == 27
            free, val = resolve(source['model_file']), resolve(source['probability_file'])
            assert sha(free) == source['model_sha256'] and sha(val) == source['probability_sha256']
            assert audit['input_sha256'][relative(free)] == sha(free)
            assert audit['input_sha256'][relative(val)] == sha(val)
            prediction = free.with_name(free.stem + '_deployed.pt') if architecture == 'group6' else free
            assert audit['input_sha256'][relative(prediction)] == sha(prediction)
            rows.append(dict(architecture=architecture, seed=seed, budget=27,
                free_model_file=relative(free), free_model_sha256=sha(free),
                prediction_model_file=relative(prediction), prediction_model_sha256=sha(prediction),
                prediction_width=12 if architecture == 'plain12_C4pooled' else 24,
                prediction_mode='C4' if architecture == 'plain12_C4pooled' else 'single',
                source_VAL_file=relative(val), source_VAL_sha256=sha(val),
                primary_fine_threshold=source['selected']['threshold'],
                source_VAL_tau=source['selected']['threshold']))
    return rows


def selection():
    return dict(rows=model_descriptors(), source_model_count=9, budget=27,
                independent_confirmation=False)


def build_data():
    parts, metadata = [], []
    mask_keys, raw_keys, seen_frames = set(), set(), set()
    for cohort, folder in enumerate(COHORTS):
        index = read(folder / 'EVAL_INDEX.json')
        data_audit, audit = read(folder / 'DATA_AUDIT.json'), read(folder / 'AUDIT.json')
        assert data_audit['status'].startswith('PASS') and all(data_audit['checks'].values())
        assert audit['status'].startswith('PASS') and all(audit['checks'].values())
        assert index['EVAL_sha256'] == sha(folder / 'EVAL.npz') and all(index['gates'].values())
        assert data_audit['input_sha256'][relative(folder / 'EVAL.npz')] == sha(folder / 'EVAL.npz')
        with np.load(folder / 'EVAL.npz', allow_pickle=False) as z:
            item = {key: z[key].copy() for key in ['X', 'Y', 'label', 'frame']}
        item['Y'] = item['Y'][:, 0].astype(bool)
        records = index['records']; n = len(records)
        assert len(item['X']) == n == [4096, 6144][cohort]
        frames = set(item['frame'].tolist()); assert not frames & seen_frames
        seen_frames.update(frames)
        for local, row in enumerate(records):
            assert row['frame'] == int(item['frame'][local]) and row['label'] == int(item['label'][local])
            raw, mask = row['raw_D4_sha256'], row['canonical_D4']
            assert raw not in raw_keys and mask not in mask_keys
            raw_keys.add(raw); mask_keys.add(mask)
            metadata.append(dict(cohort=cohort, source_cohort=folder.name, local_index=local,
                original_TEST_index=index['local_TEST_indices'][local], **row))
        item.update(source_cohort=np.full(n, cohort, np.int8),
                    raw_key=np.array([row['raw_D4_sha256'] for row in records]))
        parts.append(item)
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    arrays['frames'] = arrays['frame'].copy()
    arrays['raw_D4_sha256'] = arrays['raw_key'].copy()
    assert arrays['X'].shape == (10240, 1, 12, 12) and len(seen_frames) == 40
    assert all(int((arrays['frame'] == frame).sum()) == 256 for frame in seen_frames)
    return arrays, dict(scope='PREVIOUSLY_EVALUATED_COHORTS_NOW_DEVELOPMENT_ONLY',
        independent_confirmation=False, examples=10240, frames=sorted(seen_frames),
        cohort_order=[folder.name for folder in COHORTS], records=metadata,
        exact_raw_and_mask_D4_disjoint=True, source_model_count=9)


def protected_inputs():
    paths = [ROOT / name for name in ['common.py', 'predict.py', 'analysis.py', 'CONFIG.json', 'PROTOCOL.md']]
    paths += [UNTIED / name for name in ['RESULT.json', 'AUDIT.json', 'LOCK.json', 'CONFIG.json', 'PROTOCOL.md']]
    paths += [STAGE / 'model.py', PILOT / 'TRAIN_VAL.npz', PILOT / 'prepare.py',
              G / 'sem_confirmation_20261028/prepare.py', G / 'retained_evidence_20261027/SELECTED_SOURCES.zip']
    for folder in COHORTS:
        paths += [folder / name for name in ['EVAL.npz', 'EVAL_INDEX.json', 'DATA_MANIFEST.json',
                  'PREDICTIONS.npz', 'RESULT.json', 'AUDIT.json', 'DATA_AUDIT.json', 'SELECTION.json', 'MODEL_LOCK.json']]
    for row in model_descriptors():
        paths += [resolve(row[key]) for key in ['free_model_file', 'prediction_model_file', 'source_VAL_file']]
    return {relative(path): dict(sha256=sha(path), bytes=path.stat().st_size)
            for path in sorted(set(path.resolve() for path in paths))}


def verify_lock():
    config(); lock = read(ROOT / 'LOCK.json')
    assert lock['inputs'] == protected_inputs(), 'Frozen source/model/cohort inputs changed'
    for name, key in [('DATA.npz', 'DATA_sha256'), ('SOURCES.json', 'SOURCES_sha256'),
                      ('PREFLIGHT.json', 'PREFLIGHT_sha256'), ('PREFLIGHT_ARRAYS.npz', 'PREFLIGHT_ARRAYS_sha256')]:
        assert sha(ROOT / name) == lock[key]
    assert lock['models'] == model_descriptors()
    return lock


def load_network(row, device):
    factory = module('calibration_development_fixed_factory', STAGE / 'model.py')
    payload = torch.load(resolve(row['prediction_model_file']), map_location='cpu', weights_only=True)
    network = factory.make_plain(row['prediction_width']); network.load_state_dict(payload['model'], strict=True)
    if row['architecture'] == 'group6': assert payload['free_model_sha256'] == row['free_model_sha256']
    return network.requires_grad_(False).to(device).eval(), factory


def hard_classes(masks):
    masks = np.asarray(masks, bool).reshape(-1, 12, 12)
    keys = [row.tobytes() for row in np.packbits(masks.reshape(len(masks), 144), axis=1)]
    missing = sorted(set(keys) - CLASS_CACHE.keys())
    if missing:
        helper = module('calibration_development_fixed_class_rule', PILOT / 'prepare.py')
        values = np.array([np.unpackbits(np.frombuffer(key, np.uint8)).reshape(12, 12) for key in missing], bool)
        CLASS_CACHE.update(zip(missing, helper.class_labels(values).tolist()))
    return np.array([CLASS_CACHE[key] for key in keys], np.int64)


@torch.no_grad()
def predict(rows, x, device, callback=None):
    result = np.empty((3, 3, len(x), 12, 12), np.float32)
    tensor = torch.tensor(5 * (x - .2), device=device)
    for i, row in enumerate(rows):
        net, factory = load_network(row, device)
        probability = np.concatenate([factory.predict_probability(net, part, row['prediction_mode']).cpu().numpy()
            for part in tensor.split(config()['inference_batch_size'])])[:, 0]
        assert np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
        result[ARCHITECTURES.index(row['architecture']), SEEDS.index(row['seed'])] = probability
        if callback: callback(i + 1, row)
        del net
    return result
