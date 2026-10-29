"""One evaluation of frozen models on the fixed new SEM frames."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import torch
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from prepare import sha, save
from train import make_segmenter, make_head, head_classes, hard_classes, SEEDS

ARMS = ['base', 'replay', 'soft', 'head9', 'head4']


def macro_from_confusion(c):
    denominator = c.sum(-1) + c.sum(-2)
    numerator = 2 * np.diagonal(c, axis1=-2, axis2=-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0).mean(-1)


def main():
    start = time.monotonic()
    assert torch.backends.mps.is_available(), 'Explicit MPS inference; no silent fallback'
    assert not (ROOT / 'RESULT.json').exists() and not (ROOT / 'PREDICTIONS.npz').exists()
    complete = json.loads((ROOT / 'TRAINING_COMPLETE.json').read_text())
    assert complete['status'] == 'COMPLETE_TRAINING_NO_TEST_OPENED'
    selected = json.loads((ROOT / 'SELECTION.json').read_text())
    assert sha(ROOT / 'SELECTION.json') == complete['selection_sha256']
    frozen = json.loads((ROOT / 'FROZEN.json').read_text())
    assert sha(ROOT / 'train.py') == frozen['script_sha256']
    assert sha(ROOT / 'DATA.npz') == frozen['data_sha256']
    assert sha(ROOT / 'PROTOCOL.md') == frozen['protocol_sha256']
    for name, h in complete['checkpoint_hashes'].items():
        assert sha(ROOT / 'models' / name) == h
    save(ROOT / 'EVALUATION_FROZEN.json', {'script_sha256': sha(Path(__file__)),
        'selection_sha256': sha(ROOT / 'SELECTION.json'), 'training_complete_sha256': sha(ROOT / 'TRAINING_COMPLETE.json'),
        'data_sha256': sha(ROOT / 'DATA.npz'), 'bootstrap_repetitions': 4000, 'bootstrap_seed': 281028})
    torch.set_num_threads(2)
    with threadpool_limits(limits=2), np.load(ROOT / 'DATA.npz', allow_pickle=False) as z:
        ids = np.flatnonzero(z['role'] == 'test')
        x = z['X'][ids]; target = z['Y'][ids, 0].astype(bool)
        truth = z['label'][ids]; frames = z['frame'][ids]
    predictions = []; packed = []; ious = []; fixtures = []; direct = {'head9': [], 'head4': []}; logits_all = {'head9': [], 'head4': []}
    rows = []
    for seed in SEEDS:
        sp = []; sm = []; si = []; sf = []
        for family in ARMS:
            payload = torch.load(ROOT / 'models' / f'final_{family}_{seed}.pt', map_location='cpu', weights_only=True)
            net = make_segmenter().to('mps'); net.load_state_dict(payload['segmenter']); net.eval()
            head = None if payload['head'] is None else make_head(family).to('mps')
            if head is not None:
                head.load_state_dict(payload['head']); head.eval()
            masks = []; probabilities = []; hp = []; hl = []
            with torch.inference_mode():
                for offset in range(0, len(x), 256):
                    xx = torch.tensor(5 * (x[offset:offset + 256] - .2), device='mps')
                    if head is None:
                        zz = net(xx)
                    else:
                        f = net[:-1](xx); zz = net[-1](f); h = head(f.flatten(1))
                        hp.append(head_classes(h, family).cpu().numpy()); hl.append(h.cpu().numpy())
                    masks.append((zz[:, 0] >= 0).cpu().numpy())
                    if offset == 0:
                        probabilities = zz[:24].sigmoid().cpu().numpy()
            mask = np.concatenate(masks); pred = hard_classes(mask)
            union = (mask | target).sum((1, 2)); intersection = (mask & target).sum((1, 2))
            iou = np.divide(intersection, union, out=np.ones(len(union), dtype=float), where=union > 0)
            sp.append(pred); sm.append(np.packbits(mask.reshape(len(mask), -1), axis=1)); si.append(iou); sf.append(probabilities)
            row = {'seed': seed, 'family': family,
                   'mask_F1': float(f1_score(truth, pred, labels=range(9), average='macro', zero_division=0)), 'IoU': float(iou.mean())}
            if head is not None:
                direct[family].append(np.concatenate(hp)); logits_all[family].append(np.concatenate(hl))
                row['direct_head_F1'] = float(f1_score(truth, np.concatenate(hp), labels=range(9), average='macro', zero_division=0))
            rows.append(row); save(ROOT / 'EVALUATION_PROGRESS.json', {'rows': rows, 'completed': len(rows), 'total': 15})
            print(json.dumps({'stage': 'TEST_EVALUATION', **row}), flush=True)
            del net, head
        predictions.append(sp); packed.append(sm); ious.append(si); fixtures.append(sf)
    predictions = np.array(predictions); ious = np.array(ious)
    np.savez_compressed(ROOT / 'PREDICTIONS.npz', mask_predictions=predictions,
        packed_masks=np.array(packed), IoU=ious, fixture_probabilities=np.array(fixtures), source_indices=ids,
        truth=truth, frames=frames, seeds=np.array(SEEDS), arms=np.array(ARMS),
        direct_head9=np.array(direct['head9']), direct_head4=np.array(direct['head4']),
        logits_head9=np.array(logits_all['head9']), logits_head4=np.array(logits_all['head4']))
    unique, gid = np.unique(frames, return_inverse=True)
    confusion = np.zeros((3, 5, len(unique), 9, 9), dtype=np.int32)
    for s in range(3):
        for a in range(5):
            np.add.at(confusion[s, a], (gid, truth, predictions[s, a]), 1)
    points = macro_from_confusion(confusion.sum(2))
    rng = np.random.default_rng(281028)
    seed_ids = rng.integers(0, 3, (4000, 3)); frame_ids = rng.integers(0, len(unique), (4000, len(unique)))
    draws = []
    for a in range(5):
        sampled = confusion[seed_ids[:, :, None], a, frame_ids[:, None, :]].sum(2)
        draws.append(macro_from_confusion(sampled).mean(1))
    draws = np.array(draws)
    metrics = {family: {'mask_F1': float(points[:, a].mean()), 'per_seed_mask_F1': points[:, a].tolist(),
                         'IoU': float(ious[:, a].mean())} for a, family in enumerate(ARMS)}
    for family in ['head9', 'head4']:
        fs = [float(f1_score(truth, pred, labels=range(9), average='macro', zero_division=0)) for pred in direct[family]]
        metrics[family].update(direct_head_F1=float(np.mean(fs)), per_seed_direct_head_F1=fs)
    comparisons = {}
    for family in ['base', 'replay', 'head9', 'head4']:
        a = ARMS.index(family); delta = draws[2] - draws[a]
        point = metrics['soft']['mask_F1'] - metrics[family]['mask_F1']
        ci = np.quantile(delta, [.025, .975]); iou_delta = metrics['soft']['IoU'] - metrics[family]['IoU']
        comparisons[family] = {'delta': point, 'CI95': ci.tolist(), 'IoU_delta': iou_delta,
                              'gates': {'gain_at_least_one_pp': point >= .01, 'positive_CI_lower': bool(ci[0] > 0),
                                        'IoU_guard': iou_delta >= -.02},
                              'per_seed_deltas': (points[:, 2] - points[:, a]).tolist()}
    primary = ['replay', selected['primary_head']]
    passed = all(all(comparisons[k]['gates'].values()) for k in primary)
    np.savez_compressed(ROOT / 'BOOTSTRAP.npz', scores=draws, seed_ids=seed_ids, frame_ids=frame_ids, confusion=confusion)
    result = {'status': 'PASS_PILOT' if passed else 'FAIL_PILOT', 'primary_controls': primary,
              'test_examples': len(truth), 'test_frames': unique.tolist(), 'test_class_counts': np.bincount(truth, minlength=9).tolist(),
              'seeds': SEEDS, 'metrics': metrics, 'soft_minus_control': comparisons,
              'selection': selected, 'seconds': time.monotonic() - start,
              'predictions_sha256': sha(ROOT / 'PREDICTIONS.npz'), 'analysis_sha256': sha(Path(__file__)),
              'scope': 'New same-chip pilot with matched weak images/class labels. Primary output is class from binary segmentation. Direct head metrics secondary. No external chip replication or publication novelty claim.'}
    save(ROOT / 'RESULT.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
