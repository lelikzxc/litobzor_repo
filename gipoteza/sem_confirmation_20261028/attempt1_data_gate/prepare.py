"""Restore a bounded public SEM subset and freeze a new same-chip pilot."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import ast
import hashlib
import importlib.util
import json
import re
import struct
import time
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import zipfile
import zlib

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
G = ROOT.parent
SHARED = G / 'shared_data/ic_sem_m2'
TEST = [179, 77, 88, 199, 37, 206, 84, 171]
URL = 'https://edmond.mpg.de/api/access/datafile/223856'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(p)


def remote_module():
    spec = importlib.util.spec_from_file_location('sem_public_remote', SHARED / 'sem_remote_zip.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def download():
    plan_path = ROOT / 'SOURCE_PLAN.json'
    assert not (ROOT / 'DATA.npz').exists(), 'Data already prepared'
    source = json.loads((SHARED / 'SOURCE_MEMBERS.json').read_text())
    used = {int(re.search(r'sem(\d+)\.png', r['path']).group(1)) for r in source['png_original_sources']}
    listing = json.loads((SHARED / 'sem_zip_listing.json').read_text())
    available = [int(m.group(1)) for r in listing
                 if (m := re.fullmatch(r'dataset/sems/m2/sem(\d+)\.png', r['name'])) and int(m.group(1)) not in used]
    available.sort(key=lambda i: hashlib.sha256(('SEM-FAIR-20261028/' + str(i)).encode()).hexdigest())
    assert available[:8] == TEST
    plan = {'protocol_sha256': sha(ROOT / 'PROTOCOL.md'), 'prepare_script_sha256': sha(Path(__file__)),
            'shared_sha256': {n: sha(SHARED / n) for n in ['SOURCE_MEMBERS.json', 'labels.zip', 'sem_zip_listing.json', 'sem_remote_zip.py']},
            'train': list(range(6)), 'val': [6, 7], 'test': TEST, 'excluded_previous_frames': sorted(used),
            'remote_zip_url': URL, 'scope': 'Fresh frames of the same historical IC, not independent chips.'}
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan
    else:
        save(plan_path, plan)
    expected = {Path(r['path']).name: r['sha256'] for r in source['png_original_sources']}
    remote = remote_module()
    with zipfile.ZipFile(remote.RemoteZip()) as z:
        infos = {i: z.getinfo(f'dataset/sems/m2/sem{i:04}.png') for i in TEST}
    files = []

    def fetch(frame):
        path = ROOT / f'sources/sem{frame:04}.png'
        if path.exists():
            raw = path.read_bytes()
        elif frame < 8:
            address = f'https://raw.githubusercontent.com/emsec/unsupervised-ic-sem-segmentation/main/dataset/sems/sem{frame:04}.png'
            with urlopen(address, timeout=60) as response:
                raw = response.read(15_000_001)
            assert len(raw) < 15_000_000
        else:
            info = infos[frame]
            assert info.compress_type == 0
            r = remote.RemoteZip(); r.seek(info.header_offset)
            block = r.read(min(30 + len(info.filename.encode()) + 2048 + info.file_size, remote.SIZE - info.header_offset))
            header = struct.unpack('<4s5H3L2H', block[:30])
            assert header[0] == b'PK\x03\x04'
            size_name, size_extra = header[-2:]
            assert block[30:30 + size_name].decode() == info.filename
            offset = 30 + size_name + size_extra
            raw = block[offset:offset + info.file_size]
            assert len(raw) == info.file_size and zlib.crc32(raw) == info.CRC
        h = hashlib.sha256(raw).hexdigest()
        if frame < 8:
            assert h == expected[path.name], ('Original source changed', frame)
        else:
            assert len(raw) == infos[frame].file_size and zlib.crc32(raw) == infos[frame].CRC
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            temporary = path.with_suffix('.partial'); temporary.write_bytes(raw); temporary.replace(path)
        return {'frame': frame, 'file': str(path.relative_to(ROOT)), 'bytes': len(raw), 'sha256': h}

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(fetch, i): i for i in list(range(8)) + TEST}
        for future in as_completed(pending):
            row = future.result(); files.append(row)
            save(ROOT / 'DOWNLOAD_PROGRESS.json', {'completed': len(files), 'total': 16,
                 'seconds': time.monotonic() - start, 'files': sorted(files, key=lambda r: r['frame'])})
            print(json.dumps({'stage': 'download', 'completed': len(files), 'total': 16, **row}), flush=True)
    with zipfile.ZipFile(SHARED / 'labels.zip') as z:
        for frame in list(range(8)) + TEST:
            raw = z.read(f'dataset/labels/m2/label{frame:04}.svg')
            path = ROOT / f'sources/label{frame:04}.svg'
            if path.exists():
                assert path.read_bytes() == raw
            else:
                path.write_bytes(raw)
    save(ROOT / 'DOWNLOAD.json', {'status': 'COMPLETE', 'files': sorted(files, key=lambda r: r['frame']),
                                 'seconds': time.monotonic() - start, 'plan_sha256': sha(plan_path)})


def binary_functions():
    """Use the exact historical hard-mask solver with explicit immutable globals."""
    with zipfile.ZipFile(G / 'retained_evidence_20261027/SELECTED_SOURCES.zip') as z:
        text = z.read('gipoteza/search_20261026/structured12_learning.py').decode()
    functions = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in ('binary_response', 'label')]
    from types import SimpleNamespace
    edges = [(y * 12 + x, yy * 12 + xx) for y in range(12) for x in range(12)
             for yy, xx in [(y + 1, x), (y, x + 1)] if yy < 12 and xx < 12]
    src = np.array([a for a, _ in edges]); dst = np.array([b for _, b in edges])
    incidence = np.zeros((len(edges), 144)); incidence[np.arange(len(edges)), src] = 1; incidence[np.arange(len(edges)), dst] = 1
    physics = SimpleNamespace(SRC=src, DST=dst, ABS=incidence,
        FREE=np.array([y * 12 + x for y in range(12) for x in range(1, 11)]),
        RIGHT=np.array([y * 12 + 11 for y in range(12)]), FULL=12 / 11)
    scope = dict(np=np, ndi=ndi, physics=physics, B=12, P=144)
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<archived binary functions>', 'exec'), scope)
    return scope['binary_response'], scope['label']


def class_labels(masks):
    response, label = binary_functions()
    return 3 * label(response(masks)) + label(response(np.asarray(masks).transpose(0, 2, 1)))


def canonical_keys(masks, raw=False):
    variants = [np.rot90(masks, k, axes=(1, 2)) for k in range(4)]
    variants += [v[:, :, ::-1] for v in variants]
    if raw:
        return [min(v[i].astype('<u2').tobytes() for v in variants) for i in range(len(masks))]
    packed = [np.packbits(v.reshape(len(v), -1), axis=1) for v in variants]
    return [min(v[i].tobytes() for v in packed) for i in range(len(masks))]


def patches(frame):
    p = ROOT / f'sources/sem{frame:04}.png'
    image = np.array(Image.open(p))
    assert image.shape == (3536, 4096) and image.ndim == 2 and image.dtype == np.uint8
    tree = ET.parse(ROOT / f'sources/label{frame:04}.svg')
    canvas = Image.new('L', (image.shape[1], image.shape[0])); draw = ImageDraw.Draw(canvas)
    for element in tree.iter():
        assert 'transform' not in element.attrib
        if element.tag.endswith('}path'):
            data = element.attrib['d']
            assert set(re.findall('[A-Za-z]', data)) <= set('MLZz')
            points = [(int(x), int(y)) for x, y in re.findall(r'[ML](\d+),(\d+) ', data)]
            assert len(points) >= 3; draw.polygon(points, fill=1)
    image = image[512:-512, 512:-512]; mask = np.array(canvas)[512:-512, 512:-512]
    h, w = image.shape; h -= h % 4; w -= w % 4
    sums = image[:h, :w].reshape(h // 4, 4, w // 4, 4).sum((1, 3)).astype(np.uint16)
    binary = mask[:h, :w].reshape(h // 4, 4, w // 4, 4).mean((1, 3)) >= .5
    from numpy.lib.stride_tricks import sliding_window_view
    y = sliding_window_view(binary, (12, 12))[::6, ::6].reshape(-1, 12, 12)
    x = sliding_window_view(sums, (12, 12))[::6, ::6].reshape(-1, 12, 12)
    rows, cols = np.meshgrid(np.arange(0, binary.shape[0] - 11, 6), np.arange(0, binary.shape[1] - 11, 6), indexing='ij')
    keep = (y.sum((1, 2)) > 0) & (y.sum((1, 2)) < 144)
    return x[keep], y[keep], np.stack((rows.ravel(), cols.ravel()), 1)[keep]


def prepare():
    assert not (ROOT / 'DATA.npz').exists()
    assert json.loads((ROOT / 'DOWNLOAD.json').read_text())['status'] == 'COMPLETE'
    used_mask = set(); used_raw = set(); all_rows = []; all_x = []; all_y = []; all_labels = []; counters = {}
    start = time.monotonic()
    for split, frames in [('train', list(range(6))), ('val', [6, 7]), ('test', TEST)]:
        pool = []
        for frame in frames:
            x, y, coordinates = patches(frame); keys = canonical_keys(y); rawkeys = canonical_keys(x, raw=True)
            selected = []
            for i, (key, rawkey) in enumerate(zip(keys, rawkeys)):
                if key in used_mask or rawkey in used_raw:
                    continue
                used_mask.add(key); used_raw.add(rawkey); selected.append(i)
            ids = np.array(selected, dtype=int)
            lab = class_labels(y[ids])
            for i, c in zip(ids, lab):
                row = dict(frame=frame, y=int(4 * coordinates[i, 0] + 512), x=int(4 * coordinates[i, 1] + 512),
                           canonical_D4=keys[i].hex(), raw_D4_sha256=hashlib.sha256(rawkeys[i]).hexdigest(), label=int(c))
                pool.append((row, x[i].astype(np.float32) / (16 * 255), y[i].copy()))
            print(json.dumps({'stage': 'prepare', 'split': split, 'frame': frame, 'unique': len(pool),
                              'seconds': time.monotonic() - start}), flush=True)
        counts = np.bincount([p[0]['label'] for p in pool], minlength=9); counters[split] = counts.tolist()
        groups = {c: sorted([v for v in pool if v[0]['label'] == c],
                           key=lambda p: hashlib.sha256(('SEM-FAIR-SELECT/' + p[0]['canonical_D4']).encode()).hexdigest()) for c in range(9)}
        if split == 'train':
            assert min(counts) >= 134, ('TRAIN gate failed', counts.tolist())
            chosen = [(v, 'dense') for c in range(9) for v in groups[c][:6]]
            chosen += [(v, 'weak') for c in range(9) for v in groups[c][6:134]]
        elif split == 'val':
            assert min(counts) >= 32, ('VAL gate failed', counts.tolist())
            chosen = [(v, 'val') for c in range(9) for v in groups[c][:96]]
        else:
            assert min(counts) >= 16, ('TEST count gate failed', counts.tolist())
            assert all(len({v[0]['frame'] for v in groups[c]}) >= 4 for c in range(9)), 'TEST group gate failed'
            chosen = [(v, 'test') for v in pool]
        for (row, x, y), role in chosen:
            all_rows.append(dict(row, role=role)); all_x.append(x); all_y.append(y); all_labels.append(row['label'])
    role = np.array([r['role'] for r in all_rows]); frame = np.array([r['frame'] for r in all_rows])
    assert not set(frame[np.isin(role, ['dense', 'weak'])]) & set(frame[role == 'val'])
    assert not set(frame[np.isin(role, ['dense', 'weak', 'val'])]) & set(frame[role == 'test'])
    assert len({r['canonical_D4'] for r in all_rows}) == len(all_rows)
    assert len({r['raw_D4_sha256'] for r in all_rows}) == len(all_rows)
    data = ROOT / 'DATA.npz'
    np.savez_compressed(data, X=np.array(all_x, dtype=np.float32)[:, None], Y=np.array(all_y, dtype=np.uint8)[:, None],
                        label=np.array(all_labels, dtype=np.int64), role=role, frame=frame)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'sources').iterdir())}
    manifest = {'status': 'PASS_DATA_GATE', 'split_counts': {r: int((role == r).sum()) for r in np.unique(role)},
                'candidate_class_counts': counters, 'records': all_rows, 'sources_sha256': sources,
                'DATA_sha256': sha(data), 'protocol_sha256': sha(ROOT / 'PROTOCOL.md'),
                'prepare_script_sha256': sha(Path(__file__)), 'seconds': time.monotonic() - start,
                'scope': 'D4-exact motif and raw-patch disjoint; 512px border exclusion; one physical IC, no claim of independent manufacturing sources.'}
    save(ROOT / 'DATA_MANIFEST.json', manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['records', 'sources_sha256']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['download', 'prepare'])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        download() if args.stage == 'download' else prepare()
