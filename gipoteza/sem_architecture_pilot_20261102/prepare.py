"""Freeze fresh same-chip TEST for the architecture pilot.

TRAIN/VAL is an exact byte copy of the completed label-scarcity study.
Dense18/27 subsets are selected later from its original54 masks; weak1152 stays fixed.
No training or model inference is performed by this module.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import re
import shutil
import struct
import time
import zipfile
import zlib

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
G = ROOT.parent
ORIGINAL = G / 'sem_confirmation_20261028'
SECOND = G / 'sem_cldice_confirmation_20261030'
THIRD = G / 'sem_cldice_expansion_20261102'
OLD = G / 'sem_scarcity_20261102'
SHARED = G / 'shared_data/ic_sem_m2'
PRIOR = [ORIGINAL, SECOND, THIRD, OLD]
TEST = [217, 245, 120, 92, 41, 287, 47, 263]
PREFIX = 'SEM-ARCHITECTURE-PILOT-20261102/'
EVAL_PREFIX = 'SEM-ARCHITECTURE-EVAL-20261102/'
EVAL_PER_FRAME = 256
URL = 'https://edmond.mpg.de/api/access/datafile/223856'
EXPECTED_TRAIN_VAL_SHA256 = '119cf2f5286067ad06c7e83034516abb0bd2a946383fe4c0987ba7a16fa15ade'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def old_helper():
    spec = importlib.util.spec_from_file_location('sem_architecture_original_prepare_20261028', ORIGINAL / 'prepare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT  # In-memory path override; original file stays unchanged.
    return module


def class_labels(masks):
    return old_helper().class_labels(masks)


def remote_module():
    spec = importlib.util.spec_from_file_location('sem_architecture_public_remote', SHARED / 'sem_remote_zip.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prior_paths(name):
    return {f'../{folder.name}/{name}': sha(folder / name) for folder in PRIOR}


def source_plan():
    assert (ROOT / 'PROTOCOL.md').exists(), 'Protocol must be frozen before new data access'
    source = json.loads((SHARED / 'SOURCE_MEMBERS.json').read_text())
    historical = {int(re.search(r'sem(\d+)\.png', row['path']).group(1))
                  for row in source['png_original_sources']}
    used = historical.copy()
    for folder in PRIOR:
        used.update(json.loads((folder / 'RESULT.json').read_text())['test_frames'])
    assert len(historical) == 46 and len(used) == 78
    listing = json.loads((SHARED / 'sem_zip_listing.json').read_text())
    candidates = [int(m.group(1)) for row in listing
        if (m := re.fullmatch(r'dataset/sems/m2/sem(\d+)\.png', row['name']))
        and int(m.group(1)) not in used]
    candidates.sort(key=lambda frame: hashlib.sha256((PREFIX + str(frame)).encode()).hexdigest())
    assert candidates[:8] == TEST
    config = json.loads((ROOT / 'CONFIG.json').read_text())
    assert config['test_frames'] == TEST
    assert sha(OLD / 'TRAIN_VAL.npz') == EXPECTED_TRAIN_VAL_SHA256
    plan = {
        'test': TEST, 'train': list(range(6)), 'val': [6, 7],
        'selection_prefix': PREFIX, 'excluded_previous_frames': sorted(used),
        'protocol_sha256': sha(ROOT / 'PROTOCOL.md'),
        'prepare_script_sha256': sha(Path(__file__)),
        'inherited_TRAIN_VAL_sha256': sha(OLD / 'TRAIN_VAL.npz'),
        'inherited_data_sha256': sha(OLD / 'DATA.npz'),
        'inherited_manifest_sha256': sha(OLD / 'DATA_MANIFEST.json'),
        'prior_manifests_sha256': prior_paths('DATA_MANIFEST.json'),
        'prior_data_sha256': prior_paths('DATA.npz'),
        'prior_results_sha256': prior_paths('RESULT.json'),
        'original_prepare_sha256': sha(ORIGINAL / 'prepare.py'),
        'shared_sha256': {name: sha(SHARED / name) for name in
            ['SOURCE_MEMBERS.json', 'labels.zip', 'sem_zip_listing.json', 'sem_remote_zip.py']},
        'remote_zip_url': URL, 'files': 8, 'maximum_png_bytes_each': 10_000_000,
        'evaluation_subset': {'selection_prefix': EVAL_PREFIX, 'maximum_patches_each_frame': EVAL_PER_FRAME,
            'rule': 'First256 per frame by SHA256(prefix+canonical_D4+slash+raw_D4_sha256), no label in key; then original DATA order',
            'minimum_examples_each_class': 8, 'minimum_frames_each_class': 3, 'replace_on_gate_failure': False},
        'scope': 'Fixed new frames from the same physical IC; exact inherited TRAIN/VAL; exclude all four-prior-pilot motifs.'}
    path = ROOT / 'SOURCE_PLAN.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan, 'Frozen source selection/inputs changed'
    else:
        save(path, plan)
    return plan


def download():
    assert not (ROOT / 'DATA.npz').exists(), 'Refuse to overwrite prepared data'
    plan = source_plan()
    (ROOT / 'sources').mkdir(exist_ok=True)
    started = time.monotonic()
    remote = remote_module()
    attempts = []
    infos = None
    for attempt in range(1, 4):
        try:
            with zipfile.ZipFile(remote.RemoteZip()) as archive:
                infos = {frame: archive.getinfo(f'dataset/sems/m2/sem{frame:04}.png') for frame in TEST}
            attempts.append({'stage': 'CENTRAL_DIRECTORY', 'attempt': attempt, 'status': 'COMPLETE'})
            break
        except Exception as error:
            attempts.append({'stage': 'CENTRAL_DIRECTORY', 'attempt': attempt,
                             'status': 'FAILED', 'error': str(error), 'error_type': type(error).__name__})
            save(ROOT / 'DOWNLOAD_ATTEMPTS.json', attempts)
            print(json.dumps(attempts[-1]), flush=True)
    assert infos is not None, 'Unable to read unchanged remote ZIP directory after three attempts'

    def fetch(frame):
        path = ROOT / 'sources' / f'sem{frame:04}.png'
        info = infos[frame]
        assert info.file_size <= 10_000_000 and info.compress_type == 0
        if path.exists():
            raw = path.read_bytes()
        else:
            last_error = None
            for attempt in range(1, 4):
                try:
                    stream = remote.RemoteZip()
                    stream.seek(info.header_offset)
                    length = min(30 + len(info.filename.encode()) + 2048 + info.file_size,
                                 remote.SIZE - info.header_offset)
                    block = stream.read(length)
                    header = struct.unpack('<4s5H3L2H', block[:30])
                    assert header[0] == b'PK\x03\x04'
                    name_size, extra_size = header[-2:]
                    assert block[30:30 + name_size].decode() == info.filename
                    offset = 30 + name_size + extra_size
                    raw = block[offset:offset + info.file_size]
                    assert len(raw) == info.file_size and zlib.crc32(raw) == info.CRC
                    attempts.append({'stage': 'PNG', 'frame': frame, 'attempt': attempt, 'status': 'COMPLETE'})
                    break
                except Exception as error:
                    last_error = error
                    row = {'stage': 'PNG', 'frame': frame, 'attempt': attempt,
                           'status': 'FAILED', 'error': str(error), 'error_type': type(error).__name__}
                    attempts.append(row)
                    print(json.dumps(row), flush=True)
            else:
                raise RuntimeError(f'Unable to fetch unchanged frame {frame}') from last_error
            temporary = path.with_suffix('.png.partial')
            temporary.write_bytes(raw); temporary.replace(path)
        assert len(raw) == info.file_size and zlib.crc32(raw) == info.CRC
        return {'frame': frame, 'file': str(path.relative_to(ROOT)), 'bytes': len(raw),
                'sha256': hashlib.sha256(raw).hexdigest(), 'crc32': info.CRC}

    files = []
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fetch, frame): frame for frame in TEST}
            for future in as_completed(futures):
                row = future.result(); files.append(row)
                save(ROOT / 'DOWNLOAD_PROGRESS.json', {
                    'stage': 'DOWNLOAD', 'completed': len(files), 'total': 8,
                    'seconds': time.monotonic() - started,
                    'files': sorted(files, key=lambda r: r['frame'])})
                save(ROOT / 'DOWNLOAD_ATTEMPTS.json', attempts)
                print(json.dumps({'stage': 'DOWNLOAD', 'completed': len(files), 'total': 8, **row}), flush=True)
    finally:
        save(ROOT / 'DOWNLOAD_ATTEMPTS.json', attempts)
    with zipfile.ZipFile(SHARED / 'labels.zip') as archive:
        for frame in TEST:
            raw = archive.read(f'dataset/labels/m2/label{frame:04}.svg')
            path = ROOT / 'sources' / f'label{frame:04}.svg'
            if path.exists():
                assert path.read_bytes() == raw
            else:
                path.write_bytes(raw)
    assert sha(ROOT / 'PROTOCOL.md') == plan['protocol_sha256']
    save(ROOT / 'DOWNLOAD.json', {'status': 'COMPLETE', 'files': sorted(files, key=lambda r: r['frame']),
        'seconds': time.monotonic() - started, 'plan_sha256': sha(ROOT / 'SOURCE_PLAN.json')})



def evaluation_subset(test, test_records, manifest):
    """Freeze a bounded, label-blind subset; gate labels only AFTER selection.

    The subset and unselected patches share the same eight physical frames;
    the remainder must not later be described as an independent TEST cohort.
    """
    selected = []
    for frame in TEST:
        candidates = [i for i, row in enumerate(test_records) if row['frame'] == frame]
        # Read frame and motif keys only here. Labels do not enter the ordering.
        candidates.sort(key=lambda i: hashlib.sha256(
            (EVAL_PREFIX + test_records[i]['canonical_D4'] + '/' +
             test_records[i]['raw_D4_sha256']).encode()).hexdigest())
        selected.extend(candidates[:EVAL_PER_FRAME])
    ids = np.array(sorted(selected), dtype=np.int64)
    assert len(ids) == len(set(ids.tolist()))
    assert all(test_records[int(i)]['role'] == 'test' for i in ids)
    selected_data = {key: value[ids] for key, value in test.items()}
    counts = np.bincount(selected_data['label'], minlength=9)
    class_frames = {str(c): sorted(set(selected_data['frame'][selected_data['label'] == c].tolist()))
                    for c in range(9)}
    gates = {'min_8_each_class': bool(counts.min() >= 8),
             'min_3_frames_each_class': all(len(frames) >= 3 for frames in class_frames.values())}
    np.savez_compressed(ROOT / 'EVAL.npz', **selected_data)
    index = {
        'status': 'PASS_EVAL_DATA_GATE' if all(gates.values()) else 'FAIL_EVAL_DATA_GATE',
        'selection_prefix': EVAL_PREFIX, 'maximum_patches_each_frame': EVAL_PER_FRAME,
        'selection_rule': 'SHA256(prefix+canonical_D4+slash+raw_D4_sha256), first256 per frame; original DATA order',
        'label_blind_selection': True, 'replacement_on_gate_failure': False,
        'local_TEST_indices': ids.tolist(), 'global_DATA_indices': (1901 + ids).tolist(),
        'records': [test_records[int(i)].copy() for i in ids],
        'examples': len(ids), 'class_counts': counts.tolist(), 'class_frames': class_frames,
        'frame_counts': {str(frame): int((selected_data['frame'] == frame).sum()) for frame in TEST},
        'gates': gates, 'EVAL_sha256': sha(ROOT / 'EVAL.npz'),
        'TEST_sha256': manifest['TEST_sha256'], 'DATA_sha256': manifest['DATA_sha256'],
        'TRAIN_VAL_sha256': manifest['TRAIN_VAL_sha256'],
        'protocol_sha256': manifest['protocol_sha256'],
        'prepare_script_sha256': sha(Path(__file__)), 'source_plan_sha256': sha(ROOT / 'SOURCE_PLAN.json'),
        'scope': 'Bounded label-blind subset of the frozen eight TEST frames. Remaining patches are not an independent cohort.'}
    save(ROOT / 'EVAL_INDEX.json', index)
    print(json.dumps({k: v for k, v in index.items() if k not in ['records', 'local_TEST_indices', 'global_DATA_indices']}, ensure_ascii=False), flush=True)
    if not all(gates.values()):
        save(ROOT / 'FAIL_EVAL_DATA_GATE.json', {k: v for k, v in index.items() if k != 'records'})
        raise RuntimeError('Frozen label-blind EVAL subset failed its gate; no resampling or training permitted')
    return index


def prepare():
    assert not (ROOT / 'DATA.npz').exists(), 'Refuse to overwrite completed data'
    assert json.loads((ROOT / 'DOWNLOAD.json').read_text())['status'] == 'COMPLETE'
    plan = source_plan()
    started = time.monotonic()
    old_manifest = json.loads((OLD / 'DATA_MANIFEST.json').read_text())
    prior_manifests = [json.loads((folder / 'DATA_MANIFEST.json').read_text()) for folder in PRIOR]
    prior_records = [row for manifest in prior_manifests for row in manifest['records']]
    with np.load(OLD / 'DATA.npz', allow_pickle=False) as data:
        inherited_ids = np.flatnonzero(data['role'] != 'test')
    assert inherited_ids.tolist() == list(range(1901))
    inherited_records = [old_manifest['records'][int(i)].copy() for i in inherited_ids]
    inherited_path = ROOT / 'TRAIN_VAL.npz'
    if inherited_path.exists():
        assert sha(inherited_path) == plan['inherited_TRAIN_VAL_sha256']
    else:
        shutil.copy2(OLD / 'TRAIN_VAL.npz', inherited_path)
    assert sha(inherited_path) == plan['inherited_TRAIN_VAL_sha256']
    with np.load(inherited_path, allow_pickle=False) as archive:
        trainval = {key: archive[key].copy() for key in archive.files}
    assert set(trainval) == {'X', 'Y', 'label', 'role', 'frame'}
    assert {role: int((trainval['role'] == role).sum()) for role in ['dense', 'weak', 'val']} == {
        'dense': 54, 'weak': 1152, 'val': 695}
    used_mask = {row['canonical_D4'] for row in prior_records}
    used_raw = {row['raw_D4_sha256'] for row in prior_records}
    prior_unique_counts = {'mask': len(used_mask), 'raw': len(used_raw), 'records_with_repeated_trainval': len(prior_records)}
    helper = old_helper()
    rows = []; xs = []; ys = []; labels_all = []
    counters = {}; dedup = {}
    for frame in TEST:
        x, y, coordinates = helper.patches(frame)
        keys = helper.canonical_keys(y)
        raw_keys = helper.canonical_keys(x, raw=True)
        kept = []; records = []; duplicates = 0
        for i, (key, rawkey) in enumerate(zip(keys, raw_keys)):
            canonical = key.hex(); rawhash = hashlib.sha256(rawkey).hexdigest()
            if canonical in used_mask or rawhash in used_raw:
                duplicates += 1
                continue
            used_mask.add(canonical); used_raw.add(rawhash); kept.append(i)
            records.append({'frame': frame, 'y': int(4*coordinates[i, 0]+512),
                'x': int(4*coordinates[i, 1]+512), 'canonical_D4': canonical,
                'raw_D4_sha256': rawhash, 'role': 'test'})
        ids = np.array(kept, dtype=np.int64)
        classes = helper.class_labels(y[ids])
        for row, label in zip(records, classes): row['label'] = int(label)
        rows.extend(records)
        xs.extend(x[ids].astype(np.float32)/(16*255)); ys.extend(y[ids]); labels_all.extend(classes)
        counters[str(frame)] = np.bincount(classes, minlength=9).tolist()
        dedup[str(frame)] = {'candidate_patches': len(x), 'excluded_duplicates': duplicates, 'kept': len(ids)}
        save(ROOT / 'PREPARE_PROGRESS.json', {'stage': 'PREPARE_TEST', 'frame': frame,
            'completed_frames': len(counters), 'total_frames': 8, 'unique_test_patches': len(rows),
            'seconds': time.monotonic()-started})
        print(json.dumps({'stage': 'PREPARE_TEST', 'frame': frame, 'unique_test_patches': len(rows),
                          'class_counts': counters[str(frame)], 'seconds': time.monotonic()-started}), flush=True)
    counts = np.bincount(labels_all, minlength=9)
    class_frames = {str(c): sorted({row['frame'] for row in rows if row['label'] == c}) for c in range(9)}
    gates = {'min_16_each_class': bool(counts.min() >= 16),
             'min_4_frames_each_class': all(len(frames) >= 4 for frames in class_frames.values())}
    if not all(gates.values()):
        failed = {'status': 'FAIL_DATA_GATE', 'test_class_counts': counts.tolist(),
                  'test_class_frames': class_frames, 'gates': gates, 'frames': TEST,
                  'scope': 'Fixed TEST gate failed; no replacement frames or training permitted.'}
        save(ROOT / 'FAIL_DATA_GATE.json', failed)
        raise RuntimeError(json.dumps(failed))
    test = {'X': np.array(xs, dtype=np.float32)[:, None],
            'Y': np.array(ys, dtype=np.uint8)[:, None],
            'label': np.array(labels_all, dtype=np.int64),
            'role': np.array(['test']*len(rows)),
            'frame': np.array([row['frame'] for row in rows], dtype=trainval['frame'].dtype)}
    np.savez_compressed(ROOT / 'TEST.npz', **test)
    combined = {key: np.concatenate((trainval[key], test[key])) for key in trainval}
    np.savez_compressed(ROOT / 'DATA.npz', **combined)
    records = inherited_records + rows
    assert len(records) == len(combined['X'])
    assert len({row['canonical_D4'] for row in records}) == len(records)
    assert len({row['raw_D4_sha256'] for row in records}) == len(records)
    assert not set(trainval['frame']) & set(test['frame'])
    assert not {row['canonical_D4'] for row in rows} & {row['canonical_D4'] for row in prior_records}
    assert not {row['raw_D4_sha256'] for row in rows} & {row['raw_D4_sha256'] for row in prior_records}
    assert sha(ROOT / 'PROTOCOL.md') == plan['protocol_sha256']
    assert plan['prior_manifests_sha256'] == prior_paths('DATA_MANIFEST.json')
    assert plan['prior_data_sha256'] == prior_paths('DATA.npz')
    original_manifest = prior_manifests[0]
    inherited_sources = {name: h for name, h in original_manifest['sources_sha256'].items()
        if int(re.search(r'(?:sem|label)(\d+)\.', Path(name).name).group(1)) < 8}
    manifest = {
        'status': 'PASS_DATA_GATE',
        'split_counts': {role: int((combined['role'] == role).sum()) for role in ['dense', 'weak', 'val', 'test']},
        'records': records,
        'sources_sha256': {str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT / 'sources').iterdir()) if path.is_file()},
        'inherited_source_files_sha256': inherited_sources, 'inherited_source_root': str(ORIGINAL),
        'inherited_data_root': str(OLD),
        'inherited_data_sha256': plan['inherited_data_sha256'],
        'inherited_manifest_sha256': plan['inherited_manifest_sha256'],
        'inherited_record_indices': inherited_ids.tolist(),
        'prior_manifests_sha256': plan['prior_manifests_sha256'], 'prior_data_sha256': plan['prior_data_sha256'],
        'prior_results_sha256': plan['prior_results_sha256'],
        'prior_union_unique_counts': prior_unique_counts,
        'TRAIN_VAL_sha256': sha(inherited_path), 'TEST_sha256': sha(ROOT / 'TEST.npz'),
        'DATA_sha256': sha(ROOT / 'DATA.npz'), 'protocol_sha256': sha(ROOT / 'PROTOCOL.md'),
        'prepare_script_sha256': sha(Path(__file__)), 'source_plan_sha256': sha(ROOT / 'SOURCE_PLAN.json'),
        'test_class_counts': counts.tolist(), 'test_class_frames': class_frames,
        'test_frame_class_counts': counters, 'deduplication': dedup, 'gates': gates,
        'exact_old_train_val_reused': True, 'prior_TEST_motifs_excluded': True,
        'seconds': time.monotonic()-started,
        'scope': 'New fixed held-out frames of one IC; exact inherited TRAIN/VAL; all motifs from all four prior pilots excluded.'}
    save(ROOT / 'DATA_MANIFEST.json', manifest)
    evaluation_subset(test, rows, manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in
        ['records', 'sources_sha256', 'inherited_source_files_sha256', 'inherited_record_indices']}, ensure_ascii=False), flush=True)


def check():
    manifest = json.loads((ROOT / 'DATA_MANIFEST.json').read_text())
    plan = json.loads((ROOT / 'SOURCE_PLAN.json').read_text())
    checks = {}
    checks['manifest_protocol_hash'] = manifest['protocol_sha256'] == sha(ROOT / 'PROTOCOL.md')
    checks['script_and_plan_hashes'] = (manifest['prepare_script_sha256'] == sha(Path(__file__))
        and manifest['source_plan_sha256'] == sha(ROOT / 'SOURCE_PLAN.json'))
    checks['archive_hashes'] = all(manifest[name+'_sha256'] == sha(ROOT / (name+'.npz'))
                                  for name in ['TRAIN_VAL', 'TEST', 'DATA'])
    checks['TRAIN_VAL_byte_exact'] = sha(ROOT / 'TRAIN_VAL.npz') == sha(OLD / 'TRAIN_VAL.npz') == plan['inherited_TRAIN_VAL_sha256']
    with np.load(ROOT / 'TRAIN_VAL.npz', allow_pickle=False) as z: tv = {k: z[k] for k in z.files}
    with np.load(ROOT / 'TEST.npz', allow_pickle=False) as z: test = {k: z[k] for k in z.files}
    with np.load(ROOT / 'DATA.npz', allow_pickle=False) as z: data = {k: z[k] for k in z.files}
    with np.load(OLD / 'TRAIN_VAL.npz', allow_pickle=False) as z: old_tv = {k: z[k] for k in z.files}
    checks['TRAIN_VAL_arrays_exact'] = all(tv[k].dtype == old_tv[k].dtype and np.array_equal(tv[k], old_tv[k]) for k in tv)
    checks['TRAIN_VAL_contains_no_TEST'] = len(tv['role']) == 1901 and set(tv['role']) == {'dense', 'weak', 'val'}
    checks['inherited_roles_frames_and_counts'] = (
        {role: int((tv['role'] == role).sum()) for role in ['dense', 'weak', 'val']}
        == {'dense': 54, 'weak': 1152, 'val': 695}
        and set(tv['frame'][tv['role'] == 'val']) == {6, 7}
        and set(tv['frame'][np.isin(tv['role'], ['dense', 'weak'])]) == set(range(6)))
    checks['combined_archives_exact'] = all(np.array_equal(data[k][:1901], tv[k]) and np.array_equal(data[k][1901:], test[k]) for k in tv)
    records = manifest['records']
    checks['record_metadata_alignment'] = (len(records) == len(data['role']) and
        all([r[k] for r in records] == data[k].tolist() for k in ['role', 'frame', 'label']))
    prior_records = [r for folder in PRIOR for r in json.loads((folder / 'DATA_MANIFEST.json').read_text())['records']]
    checks['mask_and_raw_D4_unique'] = all(len({r[k] for r in records}) == len(records) for k in ['canonical_D4', 'raw_D4_sha256'])
    checks['new_TEST_excludes_all_four_prior_motifs'] = all(not {r[k] for r in records[1901:]} & {r[k] for r in prior_records}
        for k in ['canonical_D4', 'raw_D4_sha256'])
    checks['all_source_hashes'] = (all(sha(ROOT / name) == h for name, h in manifest['sources_sha256'].items())
        and all(sha(ORIGINAL / name) == h for name, h in manifest['inherited_source_files_sha256'].items()))
    checks['prior_inputs_unchanged'] = (manifest['prior_manifests_sha256'] == prior_paths('DATA_MANIFEST.json')
        and manifest['prior_data_sha256'] == prior_paths('DATA.npz')
        and manifest['prior_results_sha256'] == prior_paths('RESULT.json'))
    checks['frozen_source_selection'] = set(test['frame']) == set(TEST) and not set(TEST) & set(plan['excluded_previous_frames'])
    checks['data_gates'] = all(manifest['gates'].values())
    # One exact native crop per test frame: no display normalization or scaling.
    errors = []
    for frame in TEST:
        i = next(i for i, r in enumerate(records) if r['role'] == 'test' and r['frame'] == frame)
        r = records[i]
        image = np.asarray(Image.open(ROOT / 'sources' / f'sem{frame:04}.png'))
        crop = image[r['y']:r['y']+48, r['x']:r['x']+48]
        reconstruction = crop.reshape(12, 4, 12, 4).sum((1, 3)).astype(np.float32)/(16*255)
        errors.append(float(np.max(np.abs(reconstruction-data['X'][i, 0]))))
    checks['native_input_reconstruction_exact'] = max(errors) == 0
    index = json.loads((ROOT / 'EVAL_INDEX.json').read_text())
    with np.load(ROOT / 'EVAL.npz', allow_pickle=False) as z: evaluation = {k: z[k] for k in z.files}
    ids = np.array(index['local_TEST_indices'], dtype=np.int64)
    global_ids = np.array(index['global_DATA_indices'], dtype=np.int64)
    checks['EVAL_index_provenance'] = (index['status'] == 'PASS_EVAL_DATA_GATE'
        and index['EVAL_sha256'] == sha(ROOT / 'EVAL.npz')
        and index['TEST_sha256'] == manifest['TEST_sha256']
        and index['DATA_sha256'] == manifest['DATA_sha256']
        and index['TRAIN_VAL_sha256'] == manifest['TRAIN_VAL_sha256']
        and index['protocol_sha256'] == manifest['protocol_sha256']
        and index['prepare_script_sha256'] == sha(Path(__file__))
        and index['source_plan_sha256'] == sha(ROOT / 'SOURCE_PLAN.json'))
    checks['EVAL_original_DATA_order'] = bool((np.diff(ids) > 0).all() and np.array_equal(global_ids, 1901+ids))
    checks['EVAL_arrays_match_TEST'] = all(evaluation[k].dtype == test[k].dtype and
        np.array_equal(evaluation[k], test[k][ids]) for k in test)
    checks['EVAL_records_match_DATA'] = index['records'] == [records[int(i)] for i in global_ids]
    # Independent reconstruction of the ordering from full-manifest motif keys.
    expected = []
    for frame in TEST:
        ranked = sorted((hashlib.sha256((EVAL_PREFIX+r['canonical_D4']+'/'+r['raw_D4_sha256']).encode()).hexdigest(), i)
                        for i, r in enumerate(records[1901:]) if r['frame'] == frame)
        expected.extend(i for _, i in ranked[:EVAL_PER_FRAME])
    checks['EVAL_label_blind_SHA_selection'] = ids.tolist() == sorted(expected)
    checks['EVAL_all_eight_frames_and_counts'] = (set(evaluation['frame']) == set(TEST)
        and all(int((evaluation['frame'] == frame).sum()) == min(EVAL_PER_FRAME, int((test['frame'] == frame).sum())) for frame in TEST))
    eval_counts = np.bincount(evaluation['label'], minlength=9)
    eval_class_frames = {str(c): sorted(set(evaluation['frame'][evaluation['label'] == c].tolist())) for c in range(9)}
    checks['EVAL_gates_recomputed'] = (bool(eval_counts.min() >= 8)
        and all(len(frames) >= 3 for frames in eval_class_frames.values())
        and index['class_counts'] == eval_counts.tolist() and index['class_frames'] == eval_class_frames
        and all(index['gates'].values()))
    result = {'status': 'PASS_PREPARE_CHECK' if all(checks.values()) else 'FAIL_PREPARE_CHECK',
              'checks': checks, 'new_TEST_patches': len(test['role']), 'records': len(records),
              'native_input_max_error': max(errors), 'EVAL_patches': len(ids), 'EVAL_sha256': sha(ROOT / 'EVAL.npz'),
              'script_sha256': sha(Path(__file__)),
              'scope': 'Data/provenance checks only; no model, training or test quality evaluation.'}
    save(ROOT / 'PREPARE_CHECK.json', result)
    assert all(checks.values()), checks
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['plan', 'download', 'prepare', 'check'])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.stage == 'plan': print(json.dumps(source_plan(), ensure_ascii=False), flush=True)
        elif args.stage == 'download': download()
        elif args.stage == 'prepare': prepare()
        else: check()
