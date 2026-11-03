"""Joint AP/F1 confirmation with paired seed/frame resampling, fixed in advance."""
import numpy as np
from sklearn.metrics import average_precision_score

ARCHITECTURES = ['group6', 'untied24_single', 'plain12_C4pooled']
MODES = ['final_calibrated', 'fixed05']
SEEDS = [5701, 5721, 5757]


def macro_from_confusion(confusion):
    denominator = confusion.sum(-1) + confusion.sum(-2)
    numerator = 2 * np.diagonal(confusion, axis1=-2, axis2=-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                     where=denominator > 0).mean(-1)


def describe(points, ious, per_frame_ap):
    cells = {}
    for a, name in enumerate(ARCHITECTURES):
        cells[name] = dict(mean=dict(mask_F1=float(points[a].mean()), IoU=float(ious[a].mean()),
                                    mean_frame_AP=float(per_frame_ap[a].mean())),
            per_seed=[dict(seed=seed, mask_F1=float(points[a, s]), IoU=float(ious[a, s].mean()),
                           mean_frame_AP=float(per_frame_ap[a, s].mean()))
                      for s, seed in enumerate(SEEDS)])

    def difference(reference):
        return dict(mean={metric: cells['group6']['mean'][metric] - cells[reference]['mean'][metric]
                          for metric in ['mask_F1', 'IoU', 'mean_frame_AP']},
            per_seed=[dict(seed=seed, mask_F1=float(points[0, s] - points[a, s]),
                           IoU=float(ious[0, s].mean() - ious[a, s].mean()),
                           mean_frame_AP=float(per_frame_ap[0, s].mean() - per_frame_ap[a, s].mean()))
                      for s, seed in enumerate(SEEDS) for a in [ARCHITECTURES.index(reference)]])

    return dict(cells=cells, contrasts=dict(group_minus_untied=difference('untied24_single'),
                                           group_minus_narrow=difference('plain12_C4pooled')))


def analyze(predictions, ious, probability, truth, target, frames, config, eligible=True):
    predictions, ious, probability = np.asarray(predictions), np.asarray(ious), np.asarray(probability)
    truth, target, frames = np.asarray(truth), np.asarray(target, bool), np.asarray(frames)
    n = len(truth)
    assert predictions.shape == ious.shape == (2, 3, 3, n)
    assert probability.shape == (3, 3, n, 12, 12) and target.shape == (n, 12, 12)
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    assert np.isfinite(ious).all() and ((ious >= 0) & (ious <= 1)).all()
    assert ((truth >= 0) & (truth < 9)).all() and ((predictions >= 0) & (predictions < 9)).all()
    values, groups = np.unique(frames, return_inverse=True)
    assert len(values) == config['frames'] == 24
    assert config['bootstrap_repeats'] == 10000 and config['CI_quantiles'] == [.0125, .9875]
    confusion = np.zeros((2, 3, 3, 24, 9, 9), np.int32)
    for m in range(2):
        for a in range(3):
            for s in range(3):
                np.add.at(confusion[m, a, s], (groups, truth, predictions[m, a, s]), 1)
    points = macro_from_confusion(confusion.sum(3))
    per_frame_ap = np.empty((3, 3, 24), np.float64)
    pooled_ap = np.empty((3, 3), np.float64)
    for a in range(3):
        for s in range(3):
            for f, value in enumerate(values):
                ids = frames == value
                per_frame_ap[a, s, f] = average_precision_score(target[ids].reshape(-1),
                                                               probability[a, s, ids].reshape(-1))
            pooled_ap[a, s] = average_precision_score(target.reshape(-1), probability[a, s].reshape(-1))
    rng = np.random.default_rng(config['bootstrap_seed'])
    repeats = config['bootstrap_repeats']
    seed_ids = rng.integers(0, 3, (repeats, 3))
    frame_ids = rng.integers(0, 24, (repeats, 24))
    bootstrap_f1, bootstrap_ap = np.empty((2, repeats)), np.empty((2, repeats))
    for a in range(2):
        table = confusion[0, a]
        for start in range(0, repeats, 256):
            end = min(start + 256, repeats)
            sampled = table[seed_ids[start:end, :, None], frame_ids[start:end, None, :]].sum(2)
            bootstrap_f1[a, start:end] = macro_from_confusion(sampled).mean(1)
        bootstrap_ap[a] = per_frame_ap[a][seed_ids[:, :, None], frame_ids[:, None, :]].mean((1, 2))
    delta_f1, delta_ap = bootstrap_f1[0] - bootstrap_f1[1], bootstrap_ap[0] - bootstrap_ap[1]
    ci_f1 = np.quantile(delta_f1, config['CI_quantiles']).tolist()
    ci_ap = np.quantile(delta_ap, config['CI_quantiles']).tolist()
    analyses = {}
    for m, mode in enumerate(MODES):
        summary = describe(points[m], ious[m], per_frame_ap)
        effect = summary['contrasts']['group_minus_untied']
        group, reference = summary['cells']['group6']['mean'], summary['cells']['untied24_single']['mean']
        goal = dict(group_F1=group['mask_F1'], reference_F1=reference['mask_F1'],
                    group_AP=group['mean_frame_AP'], reference_AP=reference['mean_frame_AP'],
                    group_IoU=group['IoU'], reference_IoU=reference['IoU'],
                    F1_delta=effect['mean']['mask_F1'], AP_delta=effect['mean']['mean_frame_AP'],
                    IoU_delta=effect['mean']['IoU'],
                    per_seed_F1_deltas=[row['mask_F1'] for row in effect['per_seed']],
                    development_eligible=bool(eligible), confirmed=False,
                    descriptive_only=mode != 'final_calibrated')
        if mode == 'final_calibrated':
            limits = config['primary_goal']
            guards = dict(mean_F1=goal['F1_delta'] >= limits['minimum_mean_F1_delta'],
                F1_CI_lower=ci_f1[0] > limits['CI_lower_strictly_above'],
                mean_AP=goal['AP_delta'] > limits['minimum_mean_AP_delta'],
                AP_CI_lower=ci_ap[0] > limits['CI_lower_strictly_above'],
                seed_F1=min(goal['per_seed_F1_deltas']) >= limits['minimum_seed_F1_delta'],
                IoU_guard=goal['IoU_delta'] >= limits['minimum_mean_IoU_delta'],
                matched_MACs=abs(1527552 / 1527552 - 1) <= limits['maximum_MAC_relative_difference'],
                development_eligible=bool(eligible))
            guards = {name: bool(value) for name, value in guards.items()}
            goal.update(CI97_5_F1=ci_f1, CI97_5_AP=ci_ap, guards=guards,
                        passed=all(guards.values()), confirmed=all(guards.values()))
        summary.update(primary_goal=goal, other_comparisons_descriptive_only=True,
                       secondary_cannot_replace_primary=mode != 'final_calibrated')
        analyses[mode] = summary
    arrays = dict(confusion=confusion, point_per_seed=points, per_frame_AP=per_frame_ap,
        pooled_pixel_AP=pooled_ap, frame_values=values, seed_ids=seed_ids, frame_ids=frame_ids,
        bootstrap_F1=bootstrap_f1, bootstrap_AP=bootstrap_ap,
        delta_F1=delta_f1, delta_AP=delta_ap, CI_quantiles=np.array(config['CI_quantiles']),
        frame_metal_prevalence=np.array([target[frames == value].mean() for value in values]))
    return analyses, arrays
