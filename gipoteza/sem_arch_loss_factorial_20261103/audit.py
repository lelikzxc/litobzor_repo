"""Independent CPU audit of the old-VAL 2x2 architecture/supervision ablation.

No optimizer steps, new fitting, or closed-confirmation data access occur here.
The already independent stage12 auditor supplies explicit reference kernels and
an independent binary-graph classifier; new factorial statistics are rebuilt.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
STAGE = ROOT.parent / 'sem_arch_groupcnn_20261102'
PILOT = ROOT.parent / 'sem_architecture_pilot_20261102'
sys.path.insert(0, str(ROOT))
import common as c
import runner


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


ref = module('factorial_independent_reference', STAGE / 'audit.py')
ARMS = ['group6', 'plain12_C4pooled']
SEEDS = [5701, 5721, 5757]
BUDGETS = [18, 27]
THRESHOLDS = [round(.1 + .01*i, 2) for i in range(81)]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def relative(path): return str(Path(path).resolve().relative_to(REPO))
def load(path): return torch.load(path, map_location='cpu', weights_only=True)
def digest(value): return ref.digest(value)


def errors(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b), (set(a), set(b))
        return max([errors(v, b[k]) for k, v in a.items()] + [0.])
    if isinstance(a, list):
        assert len(a) == len(b)
        return max([errors(x, y) for x, y in zip(a, b)] + [0.])
    if isinstance(a, (str, bool)) or a is None:
        assert a == b, (a, b); return 0.
    return abs(float(a) - float(b))


def data():
    with np.load(PILOT/'TRAIN_VAL.npz', allow_pickle=False) as z:
        # Weak masks are not used for training or any weak-target audit.
        roles = z['role'].copy(); vi = np.flatnonzero(roles == 'val')
        values = dict(X=z['X'].copy(), label=z['label'].copy(), role=roles,
                      frame=z['frame'].copy(), val_masks=z['Y'][vi, 0].astype(bool))
        subsets = read(PILOT/'SUBSETS.json')
        values['dense_masks'] = {b: z['Y'][np.array(subsets['dense_indices'][str(b)])].copy() for b in BUDGETS}
    return values, subsets, vi, np.flatnonzero(roles == 'weak')


def source_and_design_checks(checks, details):
    settings = read(ROOT/'CONFIG.json'); text = (ROOT/'PROTOCOL.md').read_text()
    checks['fixed_2x2_three_seeds_two_budgets_2000_steps_and_81_thresholds'] = (
        settings['architectures'] == ARMS and settings['seeds'] == SEEDS and settings['budgets'] == BUDGETS and
        settings['steps'] == 2000 and settings['warmup_steps'] == 400 and settings['batch_size'] == 8 and
        settings['learning_rate'] == .0005 and settings['weight_decay'] == .0001 and settings['batch_seed_offset'] == 1000 and
        settings['thresholds'] == THRESHOLDS and settings['diagnostic_threshold'] == .5 and
        settings['new_fits'] == settings['inherited_fits'] == 12 and
        settings['protocol_sha256'] == sha(ROOT/'PROTOCOL.md') and
        settings['development_gate'] == dict(minimum_mean_F1_gain=.005, minimum_budget_F1_delta=-.0025,
            minimum_mean_IoU_delta=-.002, maximum_budget_ratio_deviation=.03))
    source = ast.parse((ROOT/'runner.py').read_text())
    functions = {n.name: n for n in source.body if isinstance(n, ast.FunctionDef)}
    objective = ast.unparse(functions['objective']); fit = ast.unparse(functions['fit'])
    objective_calls = [ast.unparse(n.func) for n in ast.walk(functions['objective']) if isinstance(n, ast.Call)]
    checks['source_dense_only_BCE_shared_warmup_sampler_and_no_new_auxiliary_fits'] = (
        'F.binary_cross_entropy_with_logits' in objective_calls and not any('geometry' in n for n in objective_calls) and
        "loss = F.binary_cross_entropy_with_logits(logits[:len(dx)], dense_target)" in objective and
        "torch.cat((dx, wx)) if joint else dx" in objective and
        "joint = step > settings['warmup_steps']" in fit and
        "seed + settings['batch_seed_offset']" in fit and
        fit.index("ai = rng.integers") < fit.index("wi = rng.integers") and
        'optimizer.step()' in fit and "torch.count_nonzero(lg[settings['batch_size']:])" in fit)
    forbidden = ['EVAL.npz', 'TEST.npz', 'sem_arch_groupcnn_confirmation_20261102',
                 'sem_arch_orbit_confirmation_20261102', 'PREDICTIONS.npz']
    active_sources = (ROOT/'common.py').read_text() + (ROOT/'runner.py').read_text()
    checks['new_workflow_has_no_closed_confirmation_data_source_or_call'] = not any(v in active_sources for v in forbidden)
    calls = [n for n in ast.walk(functions['run']) if isinstance(n, ast.Call)]
    checks['all12_runs_unconditional_no_seed_replacement_or_checkpoint_choice'] = (
        sum(ast.unparse(n.func) == 'fit' for n in calls) == 1 and
        len([n for n in ast.walk(functions['run']) if isinstance(n, (ast.Break, ast.Continue))]) == 0 and
        "historical_rows(settings)" in ast.unparse(functions['run']) and
        "assert len(fits) == 12 and len(rows) == 24" in ast.unparse(functions['run']))
    checks['explicit_reused_VAL_and_weak_label_information_caveats'] = (
        'не новая закрытая проверка' in text and 'информацию weak-меток' in text and
        'изолированная оценка двух лоссов' in text and 'Доверительные интервалы' in text and
        'не объявляется новым оператором' in text)
    old = c.old_rows()
    checks['12_reused_auxiliary_rows_distinct_and_previously_independently_audited'] = (
        len(old) == 12 and {(v['architecture'],v['budget'],v['seed']) for v in old} ==
        {(a,b,s) for a in ARMS for b in BUDGETS for s in SEEDS} and
        all(read(v['folder']/'AUDIT.json')['checks_passed'] == read(v['folder']/'AUDIT.json')['checks_total'] for v in old))
    initial_ok = True
    for item in old:
        model = c.model.make_model(item['architecture'], item['seed'])
        expected = load(item['initial'])['model']
        initial_ok &= digest(model.state_dict()) == digest(expected) == item['fit']['initial_state_sha256']
        initial_ok &= sha(item['model']) == item['fit']['model_sha256']
    checks['all12_fresh_initial_states_match_historical_auxiliary_initializations'] = bool(initial_ok)
    values, subsets, vi, wi = data()
    checks['unchanged_data_roles_native_frames_dense_balance_and_weak_masks_excluded'] = (
        sha(PILOT/'TRAIN_VAL.npz') == subsets['train_val_sha256'] == '119cf2f5286067ad06c7e83034516abb0bd2a946383fe4c0987ba7a16fa15ade' and
        len(vi) == 695 and len(wi) == 1152 and set(values['frame'][vi]) == {6,7} and
        set(values['frame'][values['role'] != 'val']) == set(range(6)) and
        all(np.array_equal(np.bincount(values['label'][subsets['dense_indices'][str(b)]], minlength=9),
                           np.full(9,b//9)) for b in BUDGETS))
    if (ROOT/'PREFLIGHT.json').exists():
        pre = read(ROOT/'PREFLIGHT.json')
        checks['saved_CPU_preflight_has_zero_updates_and_current_protected_inputs'] = (
            pre['status'] == 'PASS_FACTORIAL_CPU_PREFLIGHT' and pre['optimizer_updates'] == 0 and
            pre['checks_passed'] == pre['checks_total'] == 8 and not pre['fresh_test_access'] and
            pre['inputs_sha256'] == c.historical.digest(c.protected_inputs()))
    if (ROOT/'LOCK.json').exists():
        lock = c.verify_lock()
        checks['immutable_lock_matches_all_sources_models_old_probabilities_data_and_preflight'] = (
            not lock['fresh_test_access'] and lock['optimizer_updates_planned'] == 24000 and
            lock['new_fits'] == lock['inherited_fits'] == 12 and
            all(sha(ROOT/'source_snapshot'/name) == sha(ROOT/name) for name in ['common.py','runner.py','CONFIG.json','PROTOCOL.md']))
        details['lock_sha256'] = sha(ROOT/'LOCK.json')
    details['protected_input_count'] = len(c.protected_inputs())
    return settings, old, values, subsets, vi, wi


def cpu_objective_check(checks, details):
    torch.manual_seed(6031); inputs = torch.randn(4,1,12,12,dtype=torch.float64)
    target = torch.randint(0,2,(2,1,12,12)).double(); diagnostics = []
    for arm in ARMS:
        for joint in [False, True]:
            net = c.model.make_model(arm,5701).double()
            state = {k:v.detach().clone().requires_grad_() for k,v in net.state_dict().items()}
            loss,actual = runner.objective(net,arm,inputs[:2],inputs[2:],target,joint)
            x = inputs if joint else inputs[:2]
            expected = ref.reference_group(x,state) if arm == 'group6' else ref.stable_orbit_logits(x,state)[0]
            reference_loss = F.binary_cross_entropy_with_logits(expected[:2],target)
            grad = torch.autograd.grad(loss,list(net.parameters()),retain_graph=True)
            independent = torch.autograd.grad(reference_loss,list(state.values()),retain_graph=True)
            upstream = torch.autograd.grad(loss,actual['logits'])[0]
            errors = dict(logits=float((actual['logits']-expected).detach().abs().max()),
                loss=abs(float(loss.detach())-float(reference_loss.detach())),
                parameter_gradient=max(float((a-b).abs().max()) for a,b in zip(grad,independent)),
                weak_gradient=0. if not joint else float(upstream[2:].abs().max()))
            diagnostics.append(dict(architecture=arm,joint=joint,errors=errors))
    checks['independent_CPU64_logits_BCE_parameter_VJP_and_exact_zero_weak_gradient'] = all(
        max(v['errors'].values()) < 1e-10 for v in diagnostics)
    # Dense predictions/gradients cannot depend on the additional images.
    invariant = True
    for arm in ARMS:
        net = c.model.make_model(arm,5701).double()
        one,_ = runner.objective(net,arm,inputs[:2],inputs[2:],target,False)
        grad_one = torch.autograd.grad(one,list(net.parameters()))
        both,_ = runner.objective(net,arm,inputs[:2],inputs[2:],target,True)
        grad_both = torch.autograd.grad(both,list(net.parameters()))
        invariant &= max(float((a-b).abs().max()) for a,b in zip(grad_one,grad_both)) < 1e-10
        invariant &= not any(isinstance(n,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout)) for n in net.modules())
    checks['no_batch_statistics_or_dropout_and_weak_forward_cannot_influence_dense_gradient'] = bool(invariant)
    details['independent_CPU64_objective_probes'] = diagnostics


def factorial(rows, mode):
    table = {(v['objective'],v['architecture'],v['budget'],v['seed']):v[mode] for v in rows}
    assert len(table) == 24
    def average(points): return {m:float(np.mean([p[m] for p in points])) for m in ['mask_F1','IoU']}
    cells = {obj:{arm:dict(mean=average([table[obj,arm,b,s] for b in BUDGETS for s in SEEDS]),
        by_budget={str(b):average([table[obj,arm,b,s] for s in SEEDS]) for b in BUDGETS}) for arm in ARMS}
        for obj in ['BCE','BCE_geometry']}
    def contrast(a,b):
        units = [dict(budget=budget,seed=seed,**{m:table[a[0],a[1],budget,seed][m]-table[b[0],b[1],budget,seed][m]
            for m in ['mask_F1','IoU']}) for budget in BUDGETS for seed in SEEDS]
        return dict(mean=average(units),by_budget={str(v):average([u for u in units if u['budget']==v]) for v in BUDGETS},
                    per_budget_seed=units)
    contrasts = dict(architecture_BCE=contrast(('BCE','group6'),('BCE','plain12_C4pooled')),
        architecture_auxiliary=contrast(('BCE_geometry','group6'),('BCE_geometry','plain12_C4pooled')),
        auxiliary_group=contrast(('BCE_geometry','group6'),('BCE','group6')),
        auxiliary_plain=contrast(('BCE_geometry','plain12_C4pooled'),('BCE','plain12_C4pooled')))
    contrasts['interaction'] = dict(mean={m:contrasts['architecture_auxiliary']['mean'][m]-contrasts['architecture_BCE']['mean'][m]
        for m in ['mask_F1','IoU']},definition='(Group-Plain)_aux - (Group-Plain)_BCE')
    return dict(cells=cells,contrasts=contrasts)


def expected_gate(summary):
    effect = summary['contrasts']['architecture_BCE']
    p = abs(2671/2749-1); mac = abs(1527552/1562112-1)
    guards = dict(mean_F1=effect['mean']['mask_F1']>=.005,
        budget_F1=all(v['mask_F1']>=-.0025 for v in effect['by_budget'].values()),
        mean_IoU=effect['mean']['IoU']>=-.002,matched_parameters=p<=.03,matched_MACs=mac<=.03)
    return dict(passed=all(guards.values()),guards=guards,parameter_relative_difference=p,MAC_relative_difference=mac,
        failed=[k for k,v in guards.items() if not v],interpretation='descriptive_reused_VAL_gate_not_independent_confirmation')


def audit_probe(path, fit, old, training):
    probe = load(path); step = probe['step']; expected_hash, indices = ref.expected_training_batches(fit['budget'],fit['seed'])
    di,wi = indices[step]['dense'],indices[step]['weak']; joint=step==401
    oldpath = REPO/probe['historical_probe_file']; oldprobe=load(oldpath)
    state=probe['state_before']; actual_max=max(float((state[k]-v).abs().max()) for k,v in oldprobe['state_before'].items())
    matches=all(torch.allclose(state[k],v,atol=1e-6,rtol=1e-5) for k,v in oldprobe['state_before'].items())
    metadata = (probe['architecture']==fit['architecture'] and probe['budget']==fit['budget'] and probe['seed']==fit['seed'] and
        probe['objective']=='BCE' and probe['joint_forward']==joint and not probe['weak_loss_used'] and
        probe['optimizer_step_not_yet_applied'] and sha(oldpath)==probe['historical_probe_sha256'] and
        np.array_equal(probe['dense_indices'].numpy(),di) and np.array_equal(probe['weak_indices'].numpy(),wi) and
        torch.equal(probe['dense_x'],training['dense_x'][di]) and torch.equal(probe['weak_x'],training['weak_x'][wi]) and
        torch.equal(probe['dense_mask'],training['dense_mask'][di]) and matches and
        actual_max==probe['paired_warmup_state_max_absolute_error'] and probe['paired_warmup_state_matches'] and
        probe['paired_warmup_state_tolerance']==dict(absolute=1e-6,relative=1e-5) and
        oldprobe['step']==step and oldprobe['architecture']==fit['architecture'] and
        probe['total_loss']==probe['BCE'])
    independent_state={k:v.clone().requires_grad_() for k,v in state.items()}
    x=torch.cat((probe['dense_x'],probe['weak_x'])) if joint else probe['dense_x']
    reference=(ref.reference_group(x,independent_state) if fit['architecture']=='group6' else
               ref.stable_orbit_logits(x,independent_state)[0])
    z=probe['all_logits'].clone().requires_grad_()
    bce=F.binary_cross_entropy_with_logits(z[:8],probe['dense_mask']); upstream=torch.autograd.grad(bce,z)[0]
    dense_error=float((upstream-probe['logits_gradient']).abs().max())
    vjp=torch.autograd.grad(reference,list(independent_state.values()),grad_outputs=probe['logits_gradient'])
    scaled=relative_l2=absolute=0.; finite=True
    for (key,value),gradient in zip(independent_state.items(),vjp):
        recorded=probe['gradients'][key];delta=gradient-recorded
        scaled=max(scaled,float(delta.abs().max())/max(1.,float(recorded.abs().max())))
        relative_l2=max(relative_l2,float(delta.norm())/max(1e-6,float(recorded.norm())))
        absolute=max(absolute,float(delta.abs().max()))
        finite &= bool(torch.isfinite(recorded).all() and torch.isfinite(gradient).all())
    forward=float((reference-probe['all_logits']).detach().abs().max())
    loss_error=abs(float(bce.detach())-probe['BCE'])
    weak_zero=not joint or int(torch.count_nonzero(probe['logits_gradient'][8:]))==0
    return dict(metadata=bool(metadata),independent_CPU_forward=forward<1e-3,
        independent_BCE_and_full_logit_gradient=loss_error<2e-5 and dense_error<2e-6 and weak_zero,
        independent_CPU_parameter_VJP=bool(finite) and scaled<2e-4 and relative_l2<2e-3),dict(
        architecture=fit['architecture'],budget=fit['budget'],seed=fit['seed'],step=step,
        forward=forward,BCE=loss_error,upstream=dense_error,gradient_scaled=scaled,
        gradient_relative_L2=relative_l2,gradient_absolute=absolute,warmup_state_max_absolute_error=actual_max)


def final_checks(checks,details,settings,old,values,subsets,vi,wi):
    result=read(ROOT/'RESULT.json'); trial=ROOT/'trial'; frozen=read(trial/'FROZEN.json')
    checks['complete_12_new_12_reused_fits_24000_updates_and_immutable_result'] = (
        result['status']=='COMPLETE_FACTORIAL_DEVELOPMENT' and len(result['rows'])==24 and len(result['fits'])==12 and
        result['new_fits']==result['inherited_fits']==12 and result['new_optimizer_updates']==24000 and
        sha(ROOT/'RESULT.json')==sha(trial/'RESULT.json') and result['source_and_old_inputs_unchanged'] and
        not result['fresh_test_access'] and not frozen['fresh_test_access'] and
        result['lock_sha256']==frozen['lock_sha256']==sha(ROOT/'LOCK.json') and
        frozen['config_sha256']==sha(ROOT/'CONFIG.json') and frozen['new_optimizer_updates']==24000)
    oldmap={(v['architecture'],v['budget'],v['seed']):v for v in old}
    truth=values['label'][vi]; target=values['val_masks'];x=torch.tensor(5*(values['X'][vi]-.2))
    fits=result['fits']; fitmap={(v['architecture'],v['budget'],v['seed']):v for v in fits}
    checks['all_factorial_cells_and_fit_keys_complete_without_replacements'] = (
        set(fitmap)==set(oldmap) and len(fits)==12 and
        {(v['objective'],v['architecture'],v['budget'],v['seed']) for v in result['rows']}==
        {(o,a,b,s) for o in ['BCE','BCE_geometry'] for a in ARMS for b in BUDGETS for s in SEEDS})
    sampler_ok=weight_ok=training_ok=initial_ok=deploy_ok=provenance_ok=True
    probe_checks={}; diagnostics=[]; max_probability=0.
    for key,fit in fitmap.items():
        arm,budget,seed=key;prior=oldmap[key]
        initial=load(REPO/fit['initial_file'])['model'];payload=load(REPO/fit['model_file']);final=payload['model']
        expected_hash,_=ref.expected_training_batches(budget,seed)
        sampler_ok &= fit['batch_sequence_sha256']==expected_hash==prior['fit']['batch_sequence_sha256']
        initial_ok &= digest(initial)==fit['initial_state_sha256']==prior['fit']['initial_state_sha256']==digest(load(prior['initial'])['model'])
        weight_ok &= digest(final)==fit['final_state_sha256'] and digest(final)!=digest(initial)
        weight_ok &= sha(REPO/fit['model_file'])==fit['model_sha256'] and sha(REPO/fit['initial_file'])==fit['initial_sha256']
        training_ok &= (fit['optimizer_updates']==fit['steps']==fit['dense_BCE_steps']==2000 and
            fit['warmup_shape_steps']==400 and fit['joint_shape_steps']==1600 and fit['weak_loss_steps']==0 and
            not fit['weak_labels_used'] and not fit['weak_masks_loaded'] and fit['fresh_initialization'] and
            fit['paired_auxiliary_initialization_match'] and fit['paired_auxiliary_batch_sequence_match'] and
            len(fit['gradient_probes'])==2 and {p['step'] for p in fit['gradient_probes']}=={1,401})
        training_ok &= all(payload['settings'][k]==v for k,v in fit.items() if k in payload['settings'])
        di=np.array(subsets['dense_indices'][str(budget)])
        training=dict(dense_x=torch.tensor(5*(values['X'][di]-.2)),weak_x=torch.tensor(5*(values['X'][wi]-.2)),
                      dense_mask=torch.tensor(values['dense_masks'][budget].astype(np.float32)))
        for p in fit['gradient_probes']:
            path=trial/p['file'];training_ok &= sha(path)==p['sha256']
            pc,pd=audit_probe(path,fit,prior,training);diagnostics.append(pd)
            for name,value in pc.items():probe_checks[name]=probe_checks.get(name,True) and value
        if arm=='group6':
            deployed=load((REPO/fit['model_file']).with_name(Path(fit['model_file']).stem+'_deployed.pt'))
            expected=ref.reference_deploy_state(final)
            deploy_ok &= (set(expected)==set(deployed['model']) and all(torch.equal(v,deployed['model'][k]) for k,v in expected.items()) and
                deployed['free_model_sha256']==fit['model_sha256'] and deployed['free_parameters']==2671 and deployed['materialized_coefficients']==10681)
    checks['all12_exact_sampler_digests_match_archived_auxiliary_runs']=bool(sampler_ok)
    checks['all12_initial_states_and_final_checkpoint_digests_match']=bool(initial_ok and weight_ok)
    checks['exact2000_BCE_updates_400warmup_and_zero_weak_target_supervision']=bool(training_ok)
    checks['all6_group_exported_weights_exact_reference_expansion']=bool(deploy_ok)
    for name,value in probe_checks.items():checks['all24_gradient_probes_'+name]=bool(value)
    details['gradient_probes']=diagnostics
    maximum_metric_error=0.; masks_ok=predictions_ok=targets_ok=historical_ok=True
    for row in result['rows']:
        key=(row['architecture'],row['budget'],row['seed']);prior=oldmap[key]
        path=REPO/row['probability_file'];provenance_ok &= sha(path)==row['probability_sha256']
        provenance_ok &= sha(REPO/row['model_file'])==row['model_sha256']
        if row['objective']=='BCE_geometry':
            historical_ok &= (path==prior['probability'] and REPO/row['model_file']==prior['model'] and
                REPO/row['historical_trial']==prior['folder'] and row['origin']=='immutable_historical_fit_no_new_optimizer_steps' and
                row['selected']==prior['row']['selected'] and row['primary_fine']==prior['row']['primary_fine'])
        else:
            provenance_ok &= row['model_sha256']==fitmap[key]['model_sha256'] and row['origin']=='new_fresh_fit'
        with np.load(path,allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
        probability=arrays['probability']; targets_ok &= (np.array_equal(arrays['truth'],truth) and np.array_equal(arrays['target'],target) and
            np.array_equal(arrays['thresholds'],THRESHOLDS) and probability.shape==(695,1,12,12) and np.isfinite(probability).all() and
            probability.min()>=0 and probability.max()<=1)
        metric_rows,prediction=ref.score_probability(probability,truth,target,THRESHOLDS)
        maximum_metric_error=max(maximum_metric_error,errors(metric_rows,row['all_thresholds']),
            errors(max(metric_rows,key=ref.selection_rank),row['selected']),
            errors(metric_rows[40],row['fixed05']),errors(row['primary_fine'],dict(selected=row['selected'],all_thresholds=metric_rows)))
        predictions_ok &= np.array_equal(prediction,arrays['predictions'])
        expected_packed=np.array([np.packbits((probability[:,0]>=t).reshape(695,144),axis=1) for t in THRESHOLDS])
        masks_ok &= np.array_equal(expected_packed,arrays['packed_masks'])
        state=load(REPO/row['model_file'])['model']
        with torch.no_grad():
            parts=[]
            for batch in x.split(16):
                logits=ref.reference_group(batch,state) if row['architecture']=='group6' else ref.stable_orbit_logits(batch,state)[0]
                parts.append(logits.sigmoid().numpy())
            maximum_error=float(np.max(np.abs(np.concatenate(parts)-probability)))
        max_probability=max(max_probability,maximum_error)
    checks['all24_model_probability_paths_and_sha_provenance']=bool(provenance_ok and historical_ok)
    checks['all24_VAL_truth_masks_and_threshold_arrays_match_old_archive']=bool(targets_ok)
    checks['all24x81_hard_graph_classes_and_packed_masks_reconstructed']=bool(predictions_ok and masks_ok)
    checks['all24x81_macro_F1_IoU_selection_rank_fixed05_and_aliases_exact']=maximum_metric_error<1e-12
    checks['all24_CPU_reference_final_forward_matches_recorded_MPS_probabilities']=max_probability<=1e-5
    primary=factorial(result['rows'],'selected');fixed=factorial(result['rows'],'fixed05')
    factorial_error=max(errors(primary,result['primary']),errors(fixed,result['fixed05_diagnostic']))
    gate=expected_gate(primary);gate_error=errors(gate,result['primary_development_gate'])
    checks['all_four_cells_five_contrasts_seed_then_budget_averages_and_interaction']=factorial_error<1e-12
    checks['unchanged_descriptive_BCE_architecture_gate_all_guards']=gate_error<1e-12
    checks['required_limits_no_new_confirmation_no_pvalues_and_weak_label_confound']=result['caveats']==[
        'reused_adaptive_VAL_no_new_statistical_confirmation','auxiliary_effect_changes_weak_label_supervision',
        'threshold_selection_and_scoring_on_same_VAL','known_group_convolution_no_operator_novelty_claim']
    details.update(maximum_metric_error=maximum_metric_error,maximum_CPU_vs_MPS_probability_error=max_probability,
                   maximum_factorial_error=factorial_error,maximum_gate_error=gate_error,
                   primary_gate=gate,primary_architecture_BCE=primary['contrasts']['architecture_BCE'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    started=time.monotonic();torch.set_num_threads(2);checks={};details={}
    before=c.protected_inputs();audit_source=sha(Path(__file__));reference_sha=sha(STAGE/'audit.py')
    settings,old,values,subsets,vi,wi=source_and_design_checks(checks,details)
    cpu_objective_check(checks,details)
    if not args.preflight: final_checks(checks,details,settings,old,values,subsets,vi,wi)
    checks['protected_sources_and_historical_artifacts_unchanged_by_audit']=before==c.protected_inputs()
    inputs={**{k:v['sha256'] for k,v in before.items()},relative(Path(__file__)):audit_source,
            relative(STAGE/'audit.py'):reference_sha}
    if (ROOT/'LOCK.json').exists():inputs[relative(ROOT/'LOCK.json')]=sha(ROOT/'LOCK.json')
    if not args.preflight:
        for path in sorted((ROOT/'trial').rglob('*')):
            if path.is_file():inputs[relative(path)]=sha(path)
        inputs[relative(ROOT/'RESULT.json')]=sha(ROOT/'RESULT.json')
    passed=all(checks.values());result=dict(status=('PASS_' if passed else 'FAIL_')+
        ('FACTORIAL_PREFLIGHT_AUDIT' if args.preflight else 'FACTORIAL_AUDIT'),checks=checks,
        checks_passed=sum(bool(v) for v in checks.values()),checks_total=len(checks),details=details,
        input_sha256=inputs,optimizer_updates=0,model_fitting_performed=False,fresh_confirmation_data_read=False,
        scientific_scope='independent_CPU_audit_of_reused_VAL_factorial_development',seconds=time.monotonic()-started)
    output=ROOT/('PREFLIGHT_AUDIT.json' if args.preflight else 'AUDIT.json');c.save(output,result)
    print(json.dumps(dict(status=result['status'],checks_passed=result['checks_passed'],checks_total=result['checks_total'],
        failed=[k for k,v in checks.items() if not v],seconds=result['seconds'],output=str(output))),flush=True)
    assert passed,'Independent factorial audit failed'


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
