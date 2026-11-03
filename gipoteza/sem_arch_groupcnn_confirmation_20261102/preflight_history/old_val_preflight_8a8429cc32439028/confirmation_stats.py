"""Paired seed/frame estimators; no file or target access at import."""
import ast
import numpy as np
import confirmation_common as c


def macro_from_confusion(confusion):
    # Exact primitive from the frozen prior confirmation, without importing its runner.
    text = (c.OLD_CONFIRM / 'evaluate.py').read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'macro_from_confusion')
    scope = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(c.OLD_CONFIRM/'evaluate.py'), 'exec'), scope)
    return scope['macro_from_confusion'](confusion)


def analyze(predictions, ious, truth, frames, config, eligible_by_goal):
    """Axes: predictions/IoU [mode,route,budget,seed,example]."""
    predictions, ious = np.asarray(predictions), np.asarray(ious)
    truth, frames = np.asarray(truth), np.asarray(frames)
    assert predictions.shape == ious.shape == (2,3,2,3,len(truth))
    values, gid = np.unique(frames, return_inverse=True)
    assert len(values) == 16
    confusion = np.zeros((2,3,2,3,16,9,9),np.int32)
    for m in range(2):
        for r in range(3):
            for b in range(2):
                for s in range(3): np.add.at(confusion[m,r,b,s],(gid,truth,predictions[m,r,b,s]),1)
    points = macro_from_confusion(confusion.sum(4))  # [mode,route,budget,seed]
    rng = np.random.default_rng(config['bootstrap_seed'])
    repeats = config['bootstrap_repeats']
    seed_ids = rng.integers(0,3,(repeats,3)); frame_ids = rng.integers(0,16,(repeats,16))
    scores = np.empty((2,3,2,repeats),np.float64)
    for m in range(2):
        for r in range(3):
            for b in range(2):
                table = confusion[m,r,b]
                sampled = table[seed_ids[:,:,None],frame_ids[:,None,:]].sum(2)
                scores[m,r,b] = macro_from_confusion(sampled).mean(1)
    route_scores = scores.mean(2)
    deltas = np.stack([route_scores[:,0]-route_scores[:,1],route_scores[:,0]-route_scores[:,2]],axis=1)
    analyses = {}
    names = ['A_matched_quality','B_full_efficiency']
    for m, mode in enumerate(c.MODES):
        goals = {}
        for j,name in enumerate(names):
            ref = j+1; limits = config['goals'][name]
            delta = float((points[m,0]-points[m,ref]).mean())
            budget_deltas = (points[m,0]-points[m,ref]).mean(1).tolist()
            iou_delta = float((ious[m,0]-ious[m,ref]).mean())
            ci = np.quantile(deltas[m,j],config['CI_quantiles']).tolist()
            checks = dict(mean_F1=delta >= limits['minimum_mean_F1_delta'],
                CI_lower=ci[0] > limits['CI_lower_strictly_above'],
                budget_guard=min(budget_deltas) >= limits['minimum_budget_F1_delta'],
                IoU_guard=iou_delta >= limits['minimum_mean_IoU_delta'])
            if j == 0:
                checks['parameter_match'] = abs(2671/2749-1) <= .03
                checks['MAC_match'] = abs(1527552/1562112-1) <= .03
            else: checks['MAC_ratio'] = 1527552/6110208 <= .30
            passed = bool(all(checks.values()))
            goals[name] = dict(group_F1=float(points[m,0].mean()), reference_F1=float(points[m,ref].mean()),
                reference_route=c.ROUTES[ref], delta=delta, CI97_5=ci, budget_deltas=budget_deltas,
                per_seed_deltas=(points[m,0]-points[m,ref]).mean(0).tolist(), IoU_delta=iou_delta,
                group_IoU=float(ious[m,0].mean()), reference_IoU=float(ious[m,ref].mean()),
                checks={k:bool(v) for k,v in checks.items()}, passed=passed,
                development_eligible=bool(eligible_by_goal[name]),
                confirmed=bool(mode == 'primary_fine' and passed and eligible_by_goal[name]))
        analyses[mode] = dict(goals=goals, any_confirmed=any(g['confirmed'] for g in goals.values()),
            secondary_cannot_replace_primary=mode == 'secondary_seven')
    arrays = dict(confusion=confusion,point_per_seed_budget=points,scores=scores,route_scores=route_scores,
        deltas=deltas,seed_ids=seed_ids,frame_ids=frame_ids,frame_values=values,CI_quantiles=np.array(config['CI_quantiles']))
    return analyses, arrays
