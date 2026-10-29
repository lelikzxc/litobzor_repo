"""Same-information SEM controls; selection on VAL only, no TEST inference."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
G = ROOT.parent
sys.path.insert(0, str(G / 'sem_physics_materials'))
from physics_loss_example import make_segmenter, geometry_loss, smoke_check
from prepare import class_labels, sha, save

DEV_SEEDS = [1701, 1741]
SEEDS = [2801, 2833, 2887]
WEIGHTS = [.1, .4, 1.]
START = time.monotonic()
TOTAL = 44000
COMPLETED = 0
CLASS_CACHE = {}


def progress(**details):
    record = dict(seconds=time.monotonic() - START, completed_backbone_steps=COMPLETED,
                  total_backbone_steps=TOTAL, **details)
    save(ROOT / 'PROGRESS.json', record)
    with (ROOT / 'progress.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


def hard_classes(masks):
    masks = np.asarray(masks, bool).reshape(-1, 12, 12)
    keys = [r.tobytes() for r in np.packbits(masks.reshape(len(masks), -1), axis=1)]
    missing = sorted(set(keys) - CLASS_CACHE.keys())
    if missing:
        values = np.array([np.unpackbits(np.frombuffer(k, dtype=np.uint8)).reshape(12, 12) for k in missing], dtype=bool)
        CLASS_CACHE.update(zip(missing, class_labels(values).tolist()))
    return np.array([CLASS_CACHE[k] for k in keys], dtype=np.int64)


def make_head(family):
    return nn.Sequential(nn.Linear(3456, 64), nn.GELU(), nn.Linear(64, 9 if family == 'head9' else 4))


def head_classes(logits, family):
    if family == 'head9':
        return logits.argmax(1)
    z = logits.reshape(-1, 2, 2)
    scores = torch.stack((F.logsigmoid(-z[:, :, 0]) + F.logsigmoid(-z[:, :, 1]),
                          F.logsigmoid(z[:, :, 0]) + F.logsigmoid(-z[:, :, 1]),
                          F.logsigmoid(z[:, :, 0]) + F.logsigmoid(z[:, :, 1])), -1)
    c = scores.argmax(-1)
    return 3 * c[:, 0] + c[:, 1]


def head_loss(logits, targets, family):
    if family == 'head9':
        return F.cross_entropy(logits, targets)
    directional = torch.stack((targets // 3, targets % 3), 1)
    ordinal = (directional[:, :, None] > torch.arange(2, device=targets.device)).float()
    return F.binary_cross_entropy_with_logits(logits.reshape(-1, 2, 2), ordinal)


def load_training():
    with np.load(ROOT / 'DATA.npz', allow_pickle=False) as z:
        role = z['role']; data = {}
        for r in ['dense', 'weak', 'val']:
            ids = np.flatnonzero(role == r)
            data[r] = {'x': torch.tensor(5 * (z['X'][ids] - .2), device='mps'),
                       'label': torch.tensor(z['label'][ids], device='mps')}
            if r != 'weak':
                data[r]['mask'] = torch.tensor(z['Y'][ids].astype(np.float32), device='mps')
    # Weak segmentation masks and TEST tensors are not exposed to training.
    assert set(data['weak']) == {'x', 'label'}
    return data


def evaluate(net, head, data, family):
    net.eval()
    if head is not None:
        head.eval()
    masks = []; direct = []
    with torch.inference_mode():
        for x in data['x'].split(256):
            if head is None:
                logits = net(x)
            else:
                features = net[:-1](x); logits = net[-1](features)
                direct.append(head_classes(head(features.flatten(1)), family).cpu().numpy())
            masks.append((logits[:, 0] >= 0).cpu().numpy())
    masks = np.concatenate(masks); truth = data['label'].cpu().numpy()
    target = data['mask'][:, 0].cpu().numpy().astype(bool)
    prediction = hard_classes(masks); union = (masks | target).sum((1, 2))
    iou = np.divide((masks & target).sum((1, 2)), union, out=np.ones(len(union), dtype=float), where=union > 0)
    result = dict(mask_F1=float(f1_score(truth, prediction, labels=range(9), average='macro', zero_division=0)),
                  IoU=float(iou.mean()), direct_head_F1=None if head is None else float(f1_score(
                      truth, np.concatenate(direct), labels=range(9), average='macro', zero_division=0)))
    net.train()
    if head is not None:
        head.train()
    return result


def base_model(seed, data):
    global COMPLETED
    torch.manual_seed(seed)
    net = make_segmenter().to('mps')
    opt = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=.0001)
    rng = np.random.default_rng(seed)
    for step in range(1, 401):
        ids = torch.tensor(rng.integers(0, len(data['x']), 16), device='mps')
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(net(data['x'][ids]), data['mask'][ids])
        assert torch.isfinite(loss)
        loss.backward(); opt.step(); COMPLETED += 1
        if step % 100 == 0:
            progress(stage='base', seed=seed, step=step, target_steps=400)
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def train_method(seed, family, weight, steps, initial, data):
    global COMPLETED
    net = make_segmenter().to('mps'); net.load_state_dict(initial)
    torch.manual_seed(seed + 3)
    head = None if family in ['soft', 'replay'] else make_head(family).to('mps')
    weak, dense = data['weak'], data['dense']
    if head is not None:
        with torch.inference_mode():
            frozen_features = torch.cat([net[:-1](x).flatten(1) for x in weak['x'].split(128)]).clone()
        # clone outside inference mode gives ordinary tensors for head gradients.
        frozen_features = frozen_features.clone()
        opt_head = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.0001)
        rng = np.random.default_rng(seed + 700)
        for _ in range(200):
            wi = torch.tensor(rng.integers(0, len(weak['x']), 16), device='mps')
            opt_head.zero_grad(set_to_none=True)
            loss = head_loss(head(frozen_features[wi]), weak['label'][wi], family)
            loss.backward(); opt_head.step()
        del frozen_features, opt_head
    parameters = list(net.parameters()) + ([] if head is None else list(head.parameters()))
    opt = torch.optim.AdamW(parameters, lr=.001, weight_decay=.0001)
    rng = np.random.default_rng(seed + 10); started = time.monotonic()
    for step in range(1, steps + 1):
        ai = torch.tensor(rng.integers(0, len(dense['x']), 16), device='mps')
        wi = torch.tensor(rng.integers(0, len(weak['x']), 16), device='mps')
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(net(dense['x'][ai]), dense['mask'][ai])
        if family == 'soft':
            labels = weak['label'][wi]
            directional = torch.stack((labels // 3, labels % 3), 1)
            loss = loss + weight * geometry_loss(net(weak['x'][wi]), directional)
        elif head is not None:
            features = net[:-1](weak['x'][wi])
            loss = loss + weight * head_loss(head(features.flatten(1)), weak['label'][wi], family)
        assert torch.isfinite(loss), (seed, family, weight, step)
        loss.backward(); opt.step(); COMPLETED += 1
        if step % 100 == 0:
            progress(stage='train', seed=seed, family=family, weight=weight, step=step,
                     target_steps=steps, method_seconds=time.monotonic() - started,
                     method_steps_per_second=step / (time.monotonic() - started))
    return net, head, time.monotonic() - started


def checkpoint(path, net, head, **settings):
    payload = {'segmenter': {k:v.detach().cpu() for k,v in net.state_dict().items()},
               'head': None if head is None else {k:v.detach().cpu() for k,v in head.state_dict().items()},
               'settings': settings}
    torch.save(payload, path)
    return sha(path)


def main():
    assert torch.backends.mps.is_available(), 'MPS required; no silent CPU fallback'
    assert not (ROOT / 'TRAINING_COMPLETE.json').exists()
    assert not (ROOT / 'FROZEN.json').exists(), 'Refuse to overwrite frozen run; document an interrupted attempt first'
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        checks = smoke_check()
        manifest = json.loads((ROOT / 'DATA_MANIFEST.json').read_text())
        assert manifest['status'] == 'PASS_DATA_GATE'
        assert manifest['protocol_sha256'] == sha(ROOT / 'PROTOCOL.md')
        assert manifest['DATA_sha256'] == sha(ROOT / 'DATA.npz')
        frozen = {'protocol_sha256': sha(ROOT / 'PROTOCOL.md'), 'data_sha256': sha(ROOT / 'DATA.npz'),
                  'manifest_sha256': sha(ROOT / 'DATA_MANIFEST.json'),
                  'script_sha256': sha(Path(__file__)), 'prepare_sha256': sha(ROOT / 'prepare.py'),
                  'physics_example_sha256': sha(G / 'sem_physics_materials/physics_loss_example.py'),
                  'torch': torch.__version__, 'MPS_available': True, 'DEV_SEEDS': DEV_SEEDS,
                  'FINAL_SEEDS': SEEDS, 'weights': WEIGHTS, 'physics_checks': checks,
                  'training_has_no_TEST_tensors': True}
        save(ROOT / 'FROZEN.json', frozen)
        (ROOT / 'models').mkdir(exist_ok=True)
        data = load_training(); rows = []; hashes = {}
        for seed in DEV_SEEDS:
            initial = base_model(seed, data['dense'])
            for family in ['soft', 'head9', 'head4']:
                for weight in WEIGHTS:
                    net, head, seconds = train_method(seed, family, weight, 1000, initial, data)
                    row = dict(seed=seed, family=family, weight=weight, seconds=seconds,
                               **evaluate(net, head, data['val'], family))
                    rows.append(row)
                    name = f'dev_{family}_{weight}_{seed}.pt'
                    hashes[name] = checkpoint(ROOT / 'models' / name, net, head, **row)
                    save(ROOT / 'DEV_RESULTS.json', rows)
                    print(json.dumps({'stage': 'VAL', **row}), flush=True)
                    del net, head
        configurations = []
        for family in ['soft', 'head9', 'head4']:
            for weight in WEIGHTS:
                selected = [r for r in rows if r['family'] == family and r['weight'] == weight]
                configurations.append(dict(family=family, weight=weight,
                    mask_F1=float(np.mean([r['mask_F1'] for r in selected])),
                    IoU=float(np.mean([r['IoU'] for r in selected]))))
        def rank(r):
            return (r['mask_F1'], r['IoU'], -r['weight'], r['family'])
        chosen = {family: max([r for r in configurations if r['family'] == family], key=rank)
                  for family in ['soft', 'head9', 'head4']}
        best_head = max([chosen['head9'], chosen['head4']], key=rank)['family']
        assert all(r['mask_F1'] >= .45 for r in chosen.values()), 'VAL learning gate failed; do not open TEST'
        selection = {'status': 'FROZEN_BEFORE_TEST', 'chosen': chosen, 'primary_head': best_head,
                     'all_VAL_configurations': configurations, 'rule': 'highest VAL mask-F1, then IoU, then lower weight'}
        save(ROOT / 'SELECTION.json', selection)
        print(json.dumps({'stage': 'SELECTION', **selection}), flush=True)
        final_rows = []
        for seed in SEEDS:
            initial = base_model(seed, data['dense'])
            base = make_segmenter(); base.load_state_dict(initial)
            name = f'final_base_{seed}.pt'; hashes[name] = checkpoint(ROOT / 'models' / name, base, None, seed=seed, family='base')
            for family in ['replay', 'soft', 'head9', 'head4']:
                weight = 0. if family == 'replay' else chosen[family]['weight']
                net, head, seconds = train_method(seed, family, weight, 2000, initial, data)
                row = dict(seed=seed, family=family, weight=weight, seconds=seconds,
                           **evaluate(net, head, data['val'], family))
                final_rows.append(row)
                name = f'final_{family}_{seed}.pt'
                hashes[name] = checkpoint(ROOT / 'models' / name, net, head, **row)
                save(ROOT / 'FINAL_VAL_RESULTS.json', final_rows)
                save(ROOT / 'CHECKPOINT_HASHES.json', hashes)
                print(json.dumps({'stage': 'FINAL_VAL', **row}), flush=True)
                del net, head
        assert COMPLETED == TOTAL, (COMPLETED, TOTAL)
        assert sha(ROOT / 'DATA.npz') == frozen['data_sha256']
        assert sha(ROOT / 'PROTOCOL.md') == frozen['protocol_sha256']
        complete = dict(status='COMPLETE_TRAINING_NO_TEST_OPENED', seconds=time.monotonic() - START,
                        backbone_steps=COMPLETED, total_weights=len(hashes), checkpoint_hashes=hashes,
                        selection_sha256=sha(ROOT / 'SELECTION.json'), frozen_sha256=sha(ROOT / 'FROZEN.json'))
        save(ROOT / 'TRAINING_COMPLETE.json', complete)
        progress(stage='COMPLETE_TRAINING_NO_TEST_OPENED')


if __name__ == '__main__':
    main()
