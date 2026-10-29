"""Independent arithmetic, source, classifier and CPU replay audit."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train import make_segmenter, make_head
from prepare import sha, save


def independent_direction(mask):
    """Sparse Kirchhoff solution on components touching both electrodes."""
    a = np.asarray(mask, bool); size = 12
    edges = [(y * size + x, yy * size + xx) for y in range(size) for x in range(size)
             for yy, xx in [(y + 1, x), (y, x + 1)]
             if yy < size and xx < size and a[y, x] and a[yy, xx]]
    if not edges:
        return 0
    u = np.array([r[0] for r in edges]); v = np.array([r[1] for r in edges])
    graph = sparse.csr_matrix((np.ones(2 * len(u)), (np.r_[u, v], np.r_[v, u])), shape=(144, 144))
    _, labels = connected_components(graph, directed=False)
    left = np.arange(0, 144, 12); right = np.arange(11, 144, 12)
    common = np.intersect1d(labels[left[a[:, 0]]], labels[right[a[:, -1]]])
    if not len(common):
        return 0
    active = np.isin(labels, common)
    graph = graph.multiply(active[:, None] * active[None, :]).tocsr()
    lap = sparse.diags(np.asarray(graph.sum(1)).ravel()) - graph
    free = np.array([y * 12 + x for y in range(12) for x in range(1, 11) if active[y * 12 + x]])
    matrix = lap[free][:, free]
    rhs = -np.asarray(lap[free][:, right].sum(1)).ravel()
    potential = spsolve(matrix.tocsc(), rhs)
    g = float(lap[right][:, right].sum() - rhs @ potential) / (12 / 11)
    return 0 if g < 1e-8 else 1 if g < .5 - 1e-8 else 2


def main():
    torch.set_num_threads(2)
    result = json.loads((ROOT / 'RESULT.json').read_text())
    complete = json.loads((ROOT / 'TRAINING_COMPLETE.json').read_text())
    frozen = json.loads((ROOT / 'FROZEN.json').read_text())
    selection = json.loads((ROOT / 'SELECTION.json').read_text())
    manifest = json.loads((ROOT / 'DATA_MANIFEST.json').read_text())
    source_hashes = all(sha(ROOT / p) == h for p, h in manifest['sources_sha256'].items())
    checkpoint_hashes = all(sha(ROOT / 'models' / p) == h for p, h in complete['checkpoint_hashes'].items())
    inputs_ok = (sha(ROOT / 'DATA.npz') == frozen['data_sha256']
                 and sha(ROOT / 'PROTOCOL.md') == frozen['protocol_sha256']
                 and sha(ROOT / 'train.py') == frozen['script_sha256']
                 and sha(ROOT / 'prepare.py') == frozen['prepare_sha256']
                 and sha(ROOT / 'physics_adapter.py') == frozen['physics_adapter_sha256']
                 and sha(ROOT.parent / 'sem_physics_materials/physics_loss_example.py') == frozen['physics_example_sha256'])
    predictions_ok = sha(ROOT / 'PREDICTIONS.npz') == result['predictions_sha256']
    with np.load(ROOT / 'DATA.npz', allow_pickle=False) as z:
        roles = z['role']; all_frames = z['frame']; ids = np.flatnonzero(roles == 'test')
        x = z['X'][ids[:24]]; ground_masks = z['Y'][ids, 0].astype(bool); labels = z['label'][ids]
    with np.load(ROOT / 'PREDICTIONS.npz', allow_pickle=False) as z:
        pred = z['mask_predictions']; packed = z['packed_masks']; ious = z['IoU']; prob = z['fixture_probabilities']
        truth = z['truth']; frame = z['frames']; seeds = z['seeds'].tolist(); arms = z['arms'].tolist()
        direct9 = z['direct_head9']; direct4 = z['direct_head4']; logits4 = z['logits_head4']
    assert np.array_equal(labels, truth)
    split_disjoint = not set(all_frames[np.isin(roles, ['dense', 'weak'])]) & set(all_frames[roles == 'val'])
    split_disjoint &= not set(all_frames[roles != 'test']) & set(all_frames[roles == 'test'])
    records = manifest['records']
    unique = len({r['canonical_D4'] for r in records}) == len(records) and len({r['raw_D4_sha256'] for r in records}) == len(records)
    f1_error = 0.; iou_error = 0.; cpu_probability_error = 0.
    for s, seed in enumerate(seeds):
        for a, family in enumerate(arms):
            score = f1_score(truth, pred[s, a], labels=range(9), average='macro', zero_division=0)
            f1_error = max(f1_error, abs(score - result['metrics'][family]['per_seed_mask_F1'][s]))
            mask = np.unpackbits(packed[s, a], axis=1).reshape(-1, 12, 12).astype(bool)
            union = (mask | ground_masks).sum((1, 2)); intersection = (mask & ground_masks).sum((1, 2))
            actual_iou = np.divide(intersection, union, out=np.ones(len(union), dtype=float), where=union > 0)
            iou_error = max(iou_error, float(abs(actual_iou - ious[s, a]).max()))
            ck = torch.load(ROOT / 'models' / f'final_{family}_{seed}.pt', map_location='cpu', weights_only=True)
            net = make_segmenter(); net.load_state_dict(ck['segmenter']); net.eval()
            with torch.inference_mode():
                out = net(torch.tensor(5 * (x - .2))).sigmoid().numpy()
            cpu_probability_error = max(cpu_probability_error, float(abs(out - prob[s, a]).max()))
    for family, direct in [('head9', direct9), ('head4', direct4)]:
        for s in range(3):
            value = f1_score(truth, direct[s], labels=range(9), average='macro', zero_division=0)
            f1_error = max(f1_error, abs(value - result['metrics'][family]['per_seed_direct_head_F1'][s]))
    # Enumerate all nine joint ordinal target patterns, independently of the decoder.
    patterns = np.array([[[a > k for k in range(2)], [b > k for k in range(2)]]
                         for a in range(3) for b in range(3)], dtype=float).reshape(9, 4)
    logit = logits4.reshape(3, -1, 4)
    joint_scores = np.einsum('snk,ck->snc', logit, patterns) - np.logaddexp(0, logit).sum(-1)[..., None]
    head4_decode = np.array_equal(joint_scores.argmax(-1), direct4)
    with np.load(ROOT / 'BOOTSTRAP.npz', allow_pickle=False) as z:
        stored = z['scores']; si = z['seed_ids']; fi = z['frame_ids']
    groups, gid = np.unique(frame, return_inverse=True)
    matrices = np.zeros((3, 5, len(groups), 9, 9), int)
    for s in range(3):
        for a in range(5):
            np.add.at(matrices[s, a], (gid, truth, pred[s, a]), 1)
    weights = np.array([np.bincount(row, minlength=len(groups)) for row in fi])
    reweighted = np.einsum('bg,smgij->bsmij', weights.astype(float), matrices.astype(float))
    diagonal = np.diagonal(reweighted, axis1=-2, axis2=-1)
    denominator = reweighted.sum(-1) + reweighted.sum(-2)
    f = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal), where=denominator > 0).mean(-1)
    independent = f[np.arange(len(si))[:, None], si].mean(1).T
    bootstrap_error = float(abs(independent - stored).max())
    gates = {}
    for family in result['primary_controls']:
        a = arms.index(family); ci = np.quantile(independent[2] - independent[a], [.025, .975])
        delta = result['metrics']['soft']['mask_F1'] - result['metrics'][family]['mask_F1']
        gates[family] = bool(delta >= .01 and ci[0] > 0 and result['metrics']['soft']['IoU'] - result['metrics'][family]['IoU'] >= -.02)
    scientific_gate_ok = (result['status'] == 'PASS_PILOT') == all(gates.values())
    # A different sparse solver checks a fixed, balanced truth subset and predicted masks.
    selected_truth = np.concatenate([np.flatnonzero(truth == c)[:16] for c in range(9)])
    checks = [(ground_masks[i], truth[i]) for i in selected_truth]
    rng = np.random.default_rng(812028)
    for s in range(3):
        for a in range(5):
            sample = rng.choice(len(truth), 16, replace=False)
            decoded = np.unpackbits(packed[s, a, sample], axis=1).reshape(-1, 12, 12).astype(bool)
            checks.extend(zip(decoded, pred[s, a, sample]))
    with threadpool_limits(limits=2):
        sparse_classes_ok = all(3 * independent_direction(m) + independent_direction(m.T) == int(y) for m, y in checks)
    protected = json.loads((ROOT / 'PROTECTED_BEFORE.json').read_text())
    unchanged = all(sha(ROOT.parent / p) == h for p, h in protected.items())
    development = json.loads((ROOT / 'DEV_RESULTS.json').read_text())
    configurations = []
    for family in ['soft', 'head9', 'head4']:
        for weight in [.1, .4, 1.]:
            rows = [v for v in development if v['family'] == family and v['weight'] == weight]
            assert len(rows) == 2
            configurations.append({'family': family, 'weight': weight,
                                   'mask_F1': float(np.mean([v['mask_F1'] for v in rows])),
                                   'IoU': float(np.mean([v['IoU'] for v in rows]))})
    rank = lambda v: (v['mask_F1'], v['IoU'], -v['weight'], v['family'])
    chosen = {family: max([v for v in configurations if v['family'] == family], key=rank)
              for family in ['soft', 'head9', 'head4']}
    independent_head = max([chosen['head9'], chosen['head4']], key=rank)['family']
    chosen_ok = (chosen == selection['chosen'] and independent_head == selection['primary_head']
                 and result['primary_controls'] == ['replay', independent_head])
    audit = {'status': 'PASS_AUDIT' if source_hashes and checkpoint_hashes and inputs_ok and predictions_ok and split_disjoint
             and unique and head4_decode and f1_error < 1e-12 and iou_error < 1e-12 and cpu_probability_error < 1e-4
             and bootstrap_error < 1e-12 and scientific_gate_ok and sparse_classes_ok and unchanged and chosen_ok else 'FAIL_AUDIT',
             'source_hashes': source_hashes, 'checkpoint_hashes': checkpoint_hashes, 'frozen_inputs': inputs_ok,
             'prediction_hash': predictions_ok, 'frame_split_disjoint': bool(split_disjoint), 'exact_D4_unique': unique,
             'independent_sklearn_F1_max_error': f1_error, 'independent_IoU_max_error': iou_error,
             'CPU_MPS_max_probability_error': cpu_probability_error, 'independent_head4_decoder': head4_decode,
             'independent_bootstrap_max_error': bootstrap_error, 'scientific_gate_consistent': scientific_gate_ok,
             'independent_sparse_class_cases': len(checks), 'independent_sparse_classes_match': bool(sparse_classes_ok),
             'existing_hypothesis_document_unchanged': unchanged, 'VAL_selected_control_respected': chosen_ok,
             'scope': 'Technical provenance and arithmetic audit. PASS_AUDIT does not mean the scientific hypothesis passed.'}
    save(ROOT / 'AUDIT.json', audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    assert audit['status'] == 'PASS_AUDIT'


if __name__ == '__main__':
    main()
