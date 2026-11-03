"""One confirmatory crossed-bootstrap contrast and descriptive factorial means."""
import numpy as np

CELLS = ['BCE_group', 'BCE_plain', 'aux_group', 'aux_plain']
MODES = ['primary_fine', 'fixed05']
BUDGETS, SEEDS = [18, 27], [5701, 5721, 5757]


def macro_from_confusion(confusion):
    denominator = confusion.sum(-1) + confusion.sum(-2)
    numerator = 2 * np.diagonal(confusion, axis1=-2, axis2=-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                     where=denominator > 0).mean(-1)


def describe(points, ious):
    cells = {}
    for index, cell in enumerate(CELLS):
        cells[cell] = dict(mean=dict(mask_F1=float(points[index].mean()), IoU=float(ious[index].mean())),
            by_budget={str(b): dict(mask_F1=float(points[index, j].mean()),
                IoU=float(ious[index, j].mean())) for j, b in enumerate(BUDGETS)},
            by_seed={str(s): dict(mask_F1=float(points[index, :, j].mean()),
                IoU=float(ious[index, :, j].mean())) for j, s in enumerate(SEEDS)})

    def difference(left, right):
        f1 = points[left] - points[right]
        iou = ious[left].mean(-1) - ious[right].mean(-1)
        return dict(mean=dict(mask_F1=float(f1.mean()), IoU=float(iou.mean())),
            by_budget={str(b): dict(mask_F1=float(f1[j].mean()), IoU=float(iou[j].mean()))
                       for j, b in enumerate(BUDGETS)},
            by_seed={str(s): dict(mask_F1=float(f1[:, j].mean()), IoU=float(iou[:, j].mean()))
                     for j, s in enumerate(SEEDS)},
            per_budget_seed=[dict(budget=b, seed=s, mask_F1=float(f1[j, k]), IoU=float(iou[j, k]))
                             for j, b in enumerate(BUDGETS) for k, s in enumerate(SEEDS)])

    contrasts = dict(architecture_BCE=difference(0, 1), architecture_auxiliary=difference(2, 3),
                     auxiliary_group=difference(2, 0), auxiliary_plain=difference(3, 1))
    architecture, auxiliary = contrasts['architecture_BCE'], contrasts['architecture_auxiliary']
    contrasts['interaction'] = dict(
        mean={m: auxiliary['mean'][m] - architecture['mean'][m] for m in ['mask_F1', 'IoU']},
        definition='(Group-Plain)_aux - (Group-Plain)_BCE', descriptive_only=True)
    return dict(cells=cells, contrasts=contrasts)


def analyze(predictions, ious, truth, frames, config, eligible=True):
    predictions, ious = np.asarray(predictions), np.asarray(ious)
    truth, frames = np.asarray(truth), np.asarray(frames)
    n = len(truth)
    assert predictions.shape == ious.shape == (2, 4, 2, 3, n)
    assert np.isfinite(ious).all() and ((ious >= 0) & (ious <= 1)).all()
    assert ((predictions >= 0) & (predictions < 9)).all() and ((truth >= 0) & (truth < 9)).all()
    values, groups = np.unique(frames, return_inverse=True)
    assert len(values) == config['frames'] == 24
    assert config['bootstrap_repeats'] == 10000 and config['CI_quantiles'] == [.025, .975]
    confusion = np.zeros((2, 4, 2, 3, 24, 9, 9), np.int32)
    for m in range(2):
        for r in range(4):
            for b in range(2):
                for s in range(3):
                    np.add.at(confusion[m, r, b, s], (groups, truth, predictions[m, r, b, s]), 1)
    points = macro_from_confusion(confusion.sum(4))
    rng = np.random.default_rng(config['bootstrap_seed'])
    repeats = config['bootstrap_repeats']
    seed_ids = rng.integers(0, 3, (repeats, 3))
    frame_ids = rng.integers(0, 24, (repeats, 24))
    budget_scores = np.empty((2, 2, repeats), np.float64)
    # Chunking changes memory use only. F1 is computed for each sampled seed
    # after frame aggregation and before averaging seed/budget values.
    for cell in range(2):
        for budget in range(2):
            table = confusion[0, cell, budget]
            for start in range(0, repeats, 256):
                end = min(start + 256, repeats)
                sampled = table[seed_ids[start:end, :, None], frame_ids[start:end, None, :]].sum(2)
                budget_scores[cell, budget, start:end] = macro_from_confusion(sampled).mean(1)
    primary_scores = budget_scores.mean(1)
    delta = primary_scores[0] - primary_scores[1]
    interval = np.quantile(delta, config['CI_quantiles']).tolist()
    analyses = {}
    for m, mode in enumerate(MODES):
        summary = describe(points[m], ious[m])
        effect = summary['contrasts']['architecture_BCE']
        goal = dict(group_F1=summary['cells']['BCE_group']['mean']['mask_F1'],
            reference_F1=summary['cells']['BCE_plain']['mean']['mask_F1'],
            mean_F1_delta=effect['mean']['mask_F1'], delta=effect['mean']['mask_F1'],
            budget_deltas=[effect['by_budget'][str(b)]['mask_F1'] for b in BUDGETS],
            per_seed_deltas=[effect['by_seed'][str(s)]['mask_F1'] for s in SEEDS],
            IoU_delta=effect['mean']['IoU'], group_IoU=summary['cells']['BCE_group']['mean']['IoU'],
            reference_IoU=summary['cells']['BCE_plain']['mean']['IoU'],
            development_eligible=bool(eligible), confirmed=False,
            descriptive_only=mode != 'primary_fine')
        if mode == 'primary_fine':
            limits = config['primary_goal']
            guards = dict(mean_F1=goal['mean_F1_delta'] >= limits['minimum_mean_F1_delta'],
                CI_lower=interval[0] > limits['CI_lower_strictly_above'],
                budget_guard=min(goal['budget_deltas']) >= limits['minimum_budget_F1_delta'],
                IoU_guard=goal['IoU_delta'] >= limits['minimum_mean_IoU_delta'],
                parameter_match=abs(2671 / 2749 - 1) <= limits['maximum_budget_ratio_deviation'],
                MAC_match=abs(1527552 / 1562112 - 1) <= limits['maximum_budget_ratio_deviation'],
                development_eligible=bool(eligible))
            guards = {key: bool(value) for key, value in guards.items()}
            goal.update(CI95=interval, guards=guards, passed=all(guards.values()),
                        confirmed=all(guards.values()))
        summary.update(primary_goal=goal, secondary_contrasts_descriptive_only=True,
                       secondary_cannot_replace_primary=mode != 'primary_fine')
        analyses[mode] = summary
    arrays = dict(confusion=confusion, point_per_seed_budget=points,
                  seed_ids=seed_ids, frame_ids=frame_ids, frame_values=values,
                  primary_budget_scores=budget_scores, primary_scores=primary_scores,
                  delta_F1=delta, CI_quantiles=np.array(config['CI_quantiles']))
    return analyses, arrays
