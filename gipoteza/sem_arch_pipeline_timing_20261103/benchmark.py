"""Paired CPU→MPS→CPU→geometric-class timings, without training or tuning."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import time

import numpy as np
from PIL import Image
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCE = ROOT.parent / 'sem_arch_calibrated_confirmation_20261103'
ARCH = ['group6', 'untied24_single']
SEEDS = [5701, 5721, 5757]


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def relative(p): return str(Path(p).resolve().relative_to(REPO))
def save(p, v):
    p = Path(p); temporary = p.with_suffix(p.suffix + '.partial')
    temporary.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(p)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def selection():
    index = read(SOURCE / 'EVAL_INDEX.json'); records = index['records']
    ids = []
    for frame in sorted({r['frame'] for r in records}):
        candidates = [i for i, r in enumerate(records) if r['frame'] == frame]
        candidates.sort(key=lambda i: hashlib.sha256(('SEM-C4-TIMING-20261103/' + records[i]['raw_D4_sha256']).encode()).hexdigest())
        ids.extend(candidates[:4])
    ids = np.array(sorted(ids), np.int64)
    assert len(ids) == 96
    rows = [r for r in read(SOURCE / 'MODEL_LOCK.json')['rows'] if r['architecture'] in ARCH]
    assert len(rows) == 6 and all(r['prediction_mode'] == 'single' for r in rows)
    return ids, records, rows


def sources():
    ids, records, rows = selection()
    paths = [ROOT / 'benchmark.py', ROOT / 'PROTOCOL.md']
    paths += [SOURCE / n for n in ['common.py', 'MODEL_LOCK.json', 'SELECTION.json', 'AUDIT.json',
                                  'DATA_AUDIT.json', 'EVAL.npz', 'EVAL_INDEX.json', 'PREDICTIONS.npz']]
    paths += [ROOT.parent / 'sem_arch_groupcnn_20261102/model.py',
              ROOT.parent / 'sem_confirmation_20261028/prepare.py',
              ROOT.parent / 'retained_evidence_20261027/SELECTED_SOURCES.zip']
    for row in rows:
        paths += [REPO / row['free_model_file'], REPO / row['prediction_model_file']]
    paths += [SOURCE / f'sources/sem{frame:04}.png' for frame in sorted({records[i]['frame'] for i in ids})]
    return {relative(p): sha(p) for p in sorted(set(paths))}


def raw_inputs(ids, records):
    images = {}; values = []
    for i in ids:
        r = records[i]; frame = r['frame']
        if frame not in images:
            images[frame] = np.asarray(Image.open(SOURCE / f'sources/sem{frame:04}.png'))
        crop = images[frame][r['y']:r['y'] + 48, r['x']:r['x'] + 48]
        assert crop.shape == (48, 48) and crop.dtype == np.uint8
        values.append(crop.copy())
    return np.stack(values)


def normalize(raw):
    n = len(raw)
    sums = raw.reshape(n, 12, 4, 12, 4).sum((2, 4)).astype(np.uint16)
    x = sums.astype(np.float32) / np.float32(16 * 255)
    return np.ascontiguousarray(5 * (x[:, None] - np.float32(.2)))


class Classifier:
    def __init__(self):
        helper = module('pipeline_timing_original_solver', ROOT.parent / 'sem_confirmation_20261028/prepare.py')
        self.response, self.label = helper.binary_functions()

    def __call__(self, mask, cache):
        keys = [v.tobytes() for v in np.packbits(mask.reshape(len(mask), 144), axis=1)]
        missing = sorted(set(keys) - cache.keys())
        if missing:
            values = np.array([np.unpackbits(np.frombuffer(k, np.uint8)).reshape(12, 12) for k in missing], bool)
            classes = 3 * self.label(self.response(values)) + self.label(self.response(values.transpose(0, 2, 1)))
            cache.update(zip(missing, classes.tolist()))
        return np.array([cache[k] for k in keys], np.int64)


def preflight():
    ids, records, rows = selection(); raw = raw_inputs(ids, records)
    with np.load(SOURCE / 'EVAL.npz') as z:
        x = z['X'][ids]; assert np.array_equal(normalize(raw), 5 * (x - .2))
    classify = Classifier()
    with np.load(SOURCE / 'PREDICTIONS.npz') as z:
        # Slice N after materialization to avoid mixed advanced-index axis movement.
        packed = z['packed_masks'][0, :2][:, :, ids]
        prediction = z['mask_predictions'][0, :2][:, :, ids]
    for a in range(2):
        for s in range(3):
            masks = np.unpackbits(packed[a, s], axis=-1).reshape(96, 12, 12).astype(bool)
            cache = {}; assert np.array_equal(classify(masks, cache), prediction[a, s])
            assert np.array_equal(classify(masks, cache), prediction[a, s])
            assert all(np.array_equal(classify(masks[i:i+16], {}), prediction[a, s, i:i+16]) for i in range(0, 96, 16))
    result = dict(status='PASS_CPU_PREFLIGHT', checks=dict(native_preprocessing_exact=True,
        all_six_classifier_cold_warm_exact=True, batch16_exact=True), input_sha256=sources(), indices=ids.tolist())
    save(ROOT / 'PREFLIGHT.json', result); print(json.dumps(dict(status=result['status'], examples=96)), flush=True)


@torch.inference_mode()
def run():
    assert os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] == '0' and torch.backends.mps.is_available()
    assert not (ROOT / 'TIMING.json').exists(), 'Do not overwrite completed measurement'
    lock = read(ROOT / 'LOCK.json'); assert lock['input_sha256'] == sources()
    torch.set_num_threads(2); started = time.perf_counter()
    ids, records, rows = selection(); raw = raw_inputs(ids, records)
    c = module('pipeline_timing_confirmation', SOURCE / 'common.py')
    classifier = Classifier(); nets = {}; expected = {}; warm_caches = {}
    with np.load(SOURCE / 'PREDICTIONS.npz') as z:
        probs = z['probability'][:2][:, :, ids]
        classes = z['mask_predictions'][0, :2][:, :, ids]
        packs = z['packed_masks'][0, :2][:, :, ids]
    for row in rows:
        key = (row['architecture'], row['seed']); a, s = ARCH.index(key[0]), SEEDS.index(key[1])
        net, factory = c.load_network(row, 'mps'); nets[key] = net
        expected[key] = dict(prob=probs[a, s], classes=classes[a, s], packed=packs[a, s], tau=row['final_calibrated_threshold'])
        cache = {}; classifier(np.unpackbits(packs[a, s], axis=-1).reshape(96, 12, 12).astype(bool), cache)
        warm_caches[key] = cache
        fixture = torch.tensor(normalize(raw[:16]), device='mps')
        for _ in range(20): net(fixture).sigmoid().cpu()
    torch.mps.synchronize(); setup_seconds = time.perf_counter() - started

    def pipeline(key, part, cache, profile=False):
        torch.mps.synchronize(); t0 = time.perf_counter_ns()
        x = normalize(raw[part]); t1 = time.perf_counter_ns()
        tensor = torch.from_numpy(x).to('mps')
        if profile: torch.mps.synchronize()
        t2 = time.perf_counter_ns()
        p = nets[key](tensor).sigmoid()
        if profile: torch.mps.synchronize()
        t3 = time.perf_counter_ns()
        p = p.cpu().numpy()[:, 0]; t4 = time.perf_counter_ns()
        mask = p >= expected[key]['tau']; t5 = time.perf_counter_ns()
        labels = classifier(mask, cache); torch.mps.synchronize(); t6 = time.perf_counter_ns()
        return p, mask, labels, t6 - t0, np.diff([t0, t1, t2, t3, t4, t5, t6])

    rng = np.random.default_rng(202611035); timings = []; profiles = []
    for batch in [1, 16]:
        for repeat in range(3):
            for start in range(0, 96, batch):
                part = np.arange(start, start + batch)
                for seed in SEEDS:
                    jobs = [(arch, cachemode) for arch in ARCH for cachemode in ['cold', 'warm']]
                    for j in rng.permutation(len(jobs)):
                        arch, cachemode = jobs[j]; key = (arch, seed)
                        cache = {} if cachemode == 'cold' else warm_caches[key]
                        p, mask, labels, elapsed, stages = pipeline(key, part, cache)
                        err = float(np.max(np.abs(p - expected[key]['prob'][part])))
                        assert err <= 1e-5
                        assert np.array_equal(np.packbits(mask.reshape(batch, 144), axis=1), expected[key]['packed'][part])
                        assert np.array_equal(labels, expected[key]['classes'][part])
                        timings.append(dict(architecture=arch, seed=seed, cache=cachemode, batch=batch,
                            repeat=repeat, start=start, nanoseconds=int(elapsed), probability_error=err))
            save(ROOT / 'PROGRESS.json', dict(stage='TIMING', batch=batch, repeat=repeat + 1,
                 requests=len(timings), seconds=time.perf_counter() - started))
            print(json.dumps(read(ROOT / 'PROGRESS.json')), flush=True)
    for repeat in range(3):
        for start in range(0, 96, 16):
            for seed in SEEDS:
                for arch in np.array(ARCH)[rng.permutation(2)]:
                    _, _, _, elapsed, stages = pipeline((str(arch), seed), np.arange(start, start + 16), {}, profile=True)
                    profiles.append(dict(architecture=str(arch), seed=seed, repeat=repeat, start=start,
                                         nanoseconds=int(elapsed), stages_ns=stages.tolist()))
    summary = []
    for batch in [1, 16]:
        for cachemode in ['cold', 'warm']:
            for arch in ARCH:
                values = np.array([r['nanoseconds'] / 1e6 for r in timings
                                   if (r['batch'], r['cache'], r['architecture']) == (batch, cachemode, arch)])
                summary.append(dict(architecture=arch, batch=batch, cache=cachemode, requests=len(values),
                    median_ms=float(np.median(values)), p95_ms=float(np.quantile(values, .95)),
                    median_ms_per_patch=float(np.median(values) / batch),
                    p95_request_ms_per_patch=float(np.quantile(values, .95) / batch),
                    mean_ms=float(np.mean(values))))
    profile_summary = []
    for arch in ARCH:
        stages = np.array([r['stages_ns'] for r in profiles if r['architecture'] == arch]) / 1e6
        profile_summary.append(dict(architecture=arch, batch=16, cache='cold',
            stages=['CPU_preprocessing', 'host_to_MPS', 'CNN_sigmoid', 'MPS_to_host', 'threshold', 'CPU_geometry_and_final_sync'],
            mean_ms=stages.mean(0).tolist(), median_ms=np.median(stages, axis=0).tolist(),
            separate_barriers_not_primary_total=True))
    assert sources() == lock['input_sha256']
    result = dict(status='COMPLETE_PIPELINE_TIMING', device='mps',
        hardware=dict(machine=platform.machine(), platform=platform.platform(), torch=torch.__version__,
                      CPU_threads=torch.get_num_threads(), mps_fallback=False),
        lock_sha256=sha(ROOT / 'LOCK.json'), setup_seconds=setup_seconds, total_seconds=time.perf_counter() - started,
        summary=summary, profile_summary=profile_summary, measurements=timings, profiles=profiles,
        correctness_checks_all_passed=True, new_fits=0, selected_indices=ids.tolist(), input_sha256=sources(),
        scope='96 selected resident-RAM native48 patches; three historical seeds; no full-frame/disk timing; descriptive')
    save(ROOT / 'TIMING.json', result)
    print(json.dumps(dict(status=result['status'], seconds=result['total_seconds'], summary=summary)), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['preflight', 'lock', 'run']); args = parser.parse_args()
    torch.set_num_threads(2)
    if args.action == 'preflight': preflight()
    elif args.action == 'lock':
        pre = read(ROOT / 'PREFLIGHT.json'); assert pre['input_sha256'] == sources() and all(pre['checks'].values())
        assert not (ROOT / 'LOCK.json').exists()
        save(ROOT / 'LOCK.json', dict(status='LOCKED_BEFORE_TIMING', input_sha256=sources(),
             preflight_sha256=sha(ROOT / 'PREFLIGHT.json'), batch_sizes=[1, 16], repeats=3, random_seed=202611035,
             UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        print('LOCKED', flush=True)
    else: run()


if __name__ == '__main__':
    with threadpool_limits(limits=2): main()
