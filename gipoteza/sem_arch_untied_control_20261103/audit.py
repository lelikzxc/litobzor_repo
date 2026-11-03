"""Independent CPU audit of a same-function-initialized, untied CNN control.

Only the historical TRAIN_VAL and development artifacts are read. No optimizer
steps, model fitting, or closed-test arrays/metrics are accessed by this tool.
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
REPO = ROOT.parents[1]
STAGE = ROOT.parent / 'sem_arch_groupcnn_20261102'
PILOT = ROOT.parent / 'sem_architecture_pilot_20261102'
sys.path.insert(0, str(ROOT))
import common as c


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


ref = module('untied_independent_reference', STAGE / 'audit.py')
physics = module('untied_independent_legacy_physics', ROOT.parent / 'sem_physics_materials/physics_loss_example.py')
SEEDS = [5701, 5721, 5757]
ARMS = ['group6', 'untied24_single', 'plain12_C4pooled']
THRESHOLDS = [round(.1 + .01*i, 2) for i in range(81)]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def relative(path): return str(Path(path).resolve().relative_to(REPO))
def load(path): return torch.load(path, map_location='cpu', weights_only=True)
def scalar_json(value):
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def difference(actual, expected):
    if isinstance(actual, dict):
        assert set(actual) == set(expected), (set(actual), set(expected))
        return max([difference(value, expected[key]) for key, value in actual.items()] + [0.])
    if isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        return max([difference(a, b) for a, b in zip(actual, expected)] + [0.])
    if isinstance(actual, (str, bool)) or actual is None:
        assert actual == expected, (actual, expected)
        return 0.
    return abs(float(actual) - float(expected))


def constraint_error(state):
    errors = []
    w = state['0.weight'].reshape(6, 4, 1, 3, 3)
    for orientation in range(4):
        errors.append(float((w[:, orientation] - torch.rot90(w[:, 0], orientation, (-2, -1))).abs().max()))
    for layer in [0, 2, 4]:
        b = state[f'{layer}.bias'].reshape(6, 4)
        errors.append(float((b - b[:, 0, None]).abs().max()))
    for layer in [2, 4]:
        w = state[f'{layer}.weight'].reshape(6, 4, 6, 4, 3, 3)
        for output in range(4):
            for input in range(4):
                expected = torch.rot90(w[:, 0, :, (input-output) % 4], output, (-2, -1))
                errors.append(float((w[:, output, :, input] - expected).abs().max()))
    h = state['6.weight'].reshape(1, 6, 4, 1, 1)
    errors.append(float((h - h[:, :, :1]).abs().max()))
    return max(errors)


def dataset():
    subsets = read(PILOT / 'SUBSETS.json')
    with np.load(PILOT / 'TRAIN_VAL.npz', allow_pickle=False) as z:
        role = z['role'].copy(); val = np.flatnonzero(role == 'val')
        dense = np.array(subsets['dense_indices']['27'])
        weak = np.flatnonzero(role == 'weak')
        x, label, frame = z['X'].copy(), z['label'].copy(), z['frame'].copy()
        dense_mask = z['Y'][dense].copy().astype(np.float32)
        target = z['Y'][val, 0].copy().astype(bool)
    values = dict(dense_x=torch.tensor(5*(x[dense]-.2)), dense_mask=torch.tensor(dense_mask),
                  weak_x=torch.tensor(5*(x[weak]-.2)), weak_label=torch.tensor(label[weak]),
                  val_x=torch.tensor(5*(x[val]-.2)), truth=label[val], target=target)
    valid = (len(dense) == 27 and len(weak) == 1152 and len(val) == 695 and
             np.array_equal(np.bincount(label[dense], minlength=9), np.full(9, 3)) and
             set(frame[val]) == {6, 7} and set(frame[role != 'val']) == set(range(6)) and
             sha(PILOT / 'TRAIN_VAL.npz') == subsets['train_val_sha256'] ==
             '119cf2f5286067ad06c7e83034516abb0bd2a946383fe4c0987ba7a16fa15ade')
    return values, valid


def design(checks, details):
    settings = read(ROOT / 'CONFIG.json')
    checks['fixed_three_seeds_budget27_common_2000_step_auxiliary_regime'] = (
        settings['budget'] == 27 and settings['seeds'] == SEEDS and settings['steps'] == 2000 and
        settings['warmup_steps'] == 400 and settings['batch_size'] == 8 and settings['physics_weight'] == 1. and
        settings['learning_rate'] == .0005 and settings['weight_decay'] == .0001 and
        settings['batch_seed_offset'] == 1000 and settings['new_fits'] == 3 and settings['inherited_fits'] == 6 and
        settings['thresholds'] == THRESHOLDS and settings['diagnostic_threshold'] == .5 and
        settings['protocol_sha256'] == sha(ROOT / 'PROTOCOL.md') and
        settings['development_gate'] == dict(minimum_mean_F1_gain=.005, minimum_seed_F1_delta=-.01,
             minimum_mean_IoU_delta=-.002, maximum_MAC_relative_difference=0.) and
        settings['confirmation_admission'] == dict(minimum_fixed05_F1_delta=0., minimum_fixed05_IoU_delta=-.002))
    source = (ROOT / 'common.py').read_text() + (ROOT / 'runner.py').read_text()
    forbidden = ['EVAL.npz', 'TEST.npz', 'PREDICTIONS.npz', 'sem_arch_bce_confirmation_20261103',
                 'sem_arch_groupcnn_confirmation_20261102', 'sem_arch_orbit_confirmation_20261102']
    checks['active_workflow_has_no_closed_test_input'] = not any(name in source for name in forbidden)
    tree = ast.parse((ROOT / 'runner.py').read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    objective_text, fit_text = ast.unparse(functions['objective']), ast.unparse(functions['fit'])
    checks['source_single_pass_auxiliary_loss_400warmup_and_same_sampler'] = (
        'net(torch.cat((dx, wx)) if joint else dx)' in objective_text and
        'F.binary_cross_entropy_with_logits' in objective_text and 'geometry_loss' in objective_text and
        'training_details' not in objective_text and 'aligned_view_logits' not in objective_text and
        "joint = step > settings['warmup_steps']" in fit_text and
        fit_text.index('ai = rng.integers') < fit_text.index('wi = rng.integers') and
        'optimizer.step()' in fit_text and
        len([node for node in ast.walk(functions['run']) if isinstance(node, (ast.Break, ast.Continue))]) == 0)
    old = c.old_rows()
    checks['six_original_audited_auxiliary_fits_no_best_seed_selection'] = (
        {(v['architecture'], v['budget'], v['seed']) for v in old} ==
        {(a, 27, s) for a in ['group6', 'plain12_C4pooled'] for s in SEEDS} and len(old) == 6 and
        all(read(v['folder'] / 'AUDIT.json')['checks_passed'] == read(v['folder'] / 'AUDIT.json')['checks_total'] for v in old))
    values, valid = dataset()
    checks['unchanged_dense27_weak1152_VAL695_no_weak_masks_used'] = bool(valid)
    if (ROOT / 'PREFLIGHT.json').exists():
        pre = read(ROOT / 'PREFLIGHT.json')
        checks['producer_preflight_passes_without_optimizer_updates'] = (
            pre['status'] == 'PASS_UNTIED_CPU_PREFLIGHT' and pre['optimizer_updates'] == 0)
    if (ROOT / 'LOCK.json').exists():
        lock = c.verify_lock()
        checks['immutable_source_data_model_and_preflight_lock_matches'] = (
            lock['inputs'] == c.protected_inputs() and
            all(sha(ROOT / 'source_snapshot' / name) == sha(ROOT / name)
                for name in ['common.py', 'runner.py', 'CONFIG.json', 'PROTOCOL.md']))
        details['lock_sha256'] = sha(ROOT / 'LOCK.json')
    details['protected_inputs'] = len(c.protected_inputs())
    return settings, old, values


def initial_mathematics(checks, details):
    all_equal = all_free = all_forward = all_gradient = gradients_release = True
    diagnostic = []
    torch.manual_seed(6011)
    x = torch.randn(3, 1, 12, 12, dtype=torch.float64)
    target = torch.randint(0, 2, x.shape, dtype=torch.int64).double()
    for seed in SEEDS:
        net, group, old = c.initial_control(seed)
        original = load(old['initial'])['model']
        expected = ref.reference_deploy_state(original)
        actual = net.state_dict()
        all_equal &= set(actual) == set(expected) and all(torch.equal(actual[k], value) for k, value in expected.items())
        all_equal &= ref.digest(group.state_dict()) == old['fit']['initial_state_sha256'] == ref.digest(original)
        all_free &= sum(p.numel() for p in net.parameters()) == 10681 and len(list(net.parameters())) == 8 and all(p.requires_grad for p in net.parameters())
        mac = sum(value.numel() for name, value in actual.items() if name.endswith('.weight')) * 144
        all_free &= mac == 1527552 and constraint_error(actual) == 0.
        net = net.double(); group = group.double()
        plain = net(x); free = group(x)
        state = {name: value.detach().clone().requires_grad_() for name, value in net.state_dict().items()}
        reference = ref.reference_plain(x, state)
        loss = F.binary_cross_entropy_with_logits(plain, target)
        comparison = F.binary_cross_entropy_with_logits(reference, target)
        gradients = torch.autograd.grad(loss, list(net.parameters()))
        independent = torch.autograd.grad(comparison, list(state.values()))
        forward_error = max(float((plain-free).detach().abs().max()), float((plain-reference).detach().abs().max()))
        gradient_error = max(float((a-b).abs().max()) for a, b in zip(gradients, independent))
        grad_state = {name: gradient for (name, _), gradient in zip(net.named_parameters(), gradients)}
        released = constraint_error(grad_state)
        all_forward &= forward_error < 1e-10
        all_gradient &= gradient_error < 1e-10 and all(torch.isfinite(g).all() and bool(g.abs().sum() > 0) for g in gradients)
        gradients_release &= released > 1e-7
        diagnostic.append(dict(seed=seed, forward_max_error=forward_error,
            gradient_max_error=gradient_error, off_constraint_gradient_max=released))
    checks['all_three_initial_states_exact_independent_C4_kernel_and_head_expansion'] = bool(all_equal)
    checks['all_10681_coefficients_free_same_24_channels_single_pass_and_1527552_MAC'] = bool(all_free)
    checks['CPU64_initial_function_and_ordinary_operator_exact_agreement'] = bool(all_forward)
    checks['independent_CPU64_all_parameter_gradients_finite_nonzero_and_correct'] = bool(all_gradient)
    checks['unconstrained_gradient_can_leave_C4_and_shared_head_parameter_subspace'] = bool(gradients_release)
    details['initial_reference_checks'] = diagnostic
    # Exercise the actual objective while computing the comparison from an
    # independent ordinary-convolution operator and the original CPU loss.
    run = module('untied_audited_runner', ROOT / 'runner.py')
    objective_diagnostics = []
    for joint in [False, True]:
        net, _, _ = c.initial_control(5701); net = net.double()
        state = {name: value.detach().clone().requires_grad_() for name, value in net.state_dict().items()}
        labels = torch.tensor([5], dtype=torch.long)
        loss, dense, weak, bce, geometry = run.objective(net, x[:2], x[2:], target[:2], labels, joint)
        output = ref.reference_plain(x if joint else x[:2], state)
        expected_loss = F.binary_cross_entropy_with_logits(output[:2], target[:2])
        if joint:
            expected_loss = expected_loss + physics.geometry_loss(output[2:], torch.stack((labels//3, labels%3), 1))
        actual_gradient = torch.autograd.grad(loss, list(net.parameters()))
        expected_gradient = torch.autograd.grad(expected_loss, list(state.values()))
        maximum = max(float((a-b).abs().max()) for a, b in zip(actual_gradient, expected_gradient))
        objective_diagnostics.append(dict(joint=joint, loss_error=abs(float(loss.detach())-float(expected_loss.detach())),
            gradient_error=maximum))
    checks['actual_objective_CPU64_single_pass_BCE_and_original_geometry_VJP_match'] = all(
        max(row['loss_error'], row['gradient_error']) < 1e-10 for row in objective_diagnostics)
    details['objective_reference_checks'] = objective_diagnostics


def statistics(rows, mode):
    table = {(r['architecture'], r['seed']): r[mode] for r in rows}
    assert len(table) == 9
    means = {arm: {metric: float(np.mean([table[arm, s][metric] for s in SEEDS]))
                   for metric in ['mask_F1', 'IoU']} for arm in ARMS}
    def compare(reference):
        units = [dict(seed=s, **{m: table['group6', s][m]-table[reference, s][m]
                               for m in ['mask_F1', 'IoU']}) for s in SEEDS]
        return dict(mean={m: float(np.mean([row[m] for row in units])) for m in ['mask_F1', 'IoU']}, per_seed=units)
    return dict(means=means, contrasts=dict(group_minus_untied=compare('untied24_single'),
                                           group_minus_narrow=compare('plain12_C4pooled')))


def probe(path, fit, values, initial):
    p = load(path); step = p['step']; warmup = step == 1
    _, indices = ref.expected_training_batches(27, fit['seed'])
    dense, weak = indices[step]['dense'], indices[step]['weak']
    meta = (p['architecture'] == 'untied24_single' and p['budget'] == 27 and p['seed'] == fit['seed'] and
        step in [1, 401] and p['optimizer_step_not_yet_applied'] and p['weak_loss_used'] == (not warmup) and
        np.array_equal(p['dense_indices'].numpy(), dense) and np.array_equal(p['weak_indices'].numpy(), weak) and
        torch.equal(p['dense_x'], values['dense_x'][dense]) and torch.equal(p['weak_x'], values['weak_x'][weak]) and
        torch.equal(p['dense_mask'], values['dense_mask'][dense]) and torch.equal(p['weak_label'], values['weak_label'][weak]))
    if warmup: meta &= ref.digest(p['state_before']) == ref.digest(initial)
    state = {name: tensor.clone().requires_grad_() for name, tensor in p['state_before'].items()}
    x = p['dense_x'] if warmup else torch.cat((p['dense_x'], p['weak_x']))
    reference = ref.reference_plain(x, state)
    recorded = p['dense_logits'] if warmup else torch.cat((p['dense_logits'], p['weak_logits']))
    upstream = p['dense_logits_gradient'] if warmup else torch.cat((p['dense_logits_gradient'], p['weak_logits_gradient']))
    forward_error = float((reference-recorded).detach().abs().max())
    gradients = torch.autograd.grad(reference, list(state.values()), grad_outputs=upstream)
    scaled = relative_l2 = absolute = 0.; finite = True
    for (name, tensor), gradient in zip(state.items(), gradients):
        actual = p['gradients'][name]; delta = gradient-actual
        scaled = max(scaled, float(delta.abs().max())/max(1., float(actual.abs().max())))
        relative_l2 = max(relative_l2, float(delta.norm())/max(1e-6, float(actual.norm())))
        absolute = max(absolute, float(delta.abs().max()))
        finite &= bool(torch.isfinite(actual).all() and torch.isfinite(gradient).all())
    dense_logits = p['dense_logits'].clone().requires_grad_()
    bce = F.binary_cross_entropy_with_logits(dense_logits, p['dense_mask'])
    dense_grad = torch.autograd.grad(bce, dense_logits)[0]
    bce_error = abs(float(bce.detach())-p['BCE'])
    dense_error = float((dense_grad-p['dense_logits_gradient']).abs().max())
    geometry_error = geometry_upstream = 0.
    if warmup:
        meta &= p['weak_logits'] is None and p['geometry_loss'] is None and p['weak_logits_gradient'] is None and p['total_loss'] == p['BCE']
    else:
        weak_logits = p['weak_logits'].clone().requires_grad_()
        label = p['weak_label']; directions = torch.stack((label//3, label%3), 1)
        geometry = physics.geometry_loss(weak_logits, directions)
        weak_grad = torch.autograd.grad(geometry, weak_logits)[0]
        geometry_error = abs(float(geometry.detach())-p['geometry_loss'])/max(1., abs(p['geometry_loss']))
        geometry_upstream = float((weak_grad-p['weak_logits_gradient']).abs().max())
        meta &= abs(p['total_loss']-p['BCE']-p['geometry_loss']) < 2e-5 and bool(p['weak_logits_gradient'].abs().sum() > 0)
    return dict(metadata=bool(meta), CPU_single_pass_forward=forward_error < 1e-3,
        CPU_VJP_given_actual_MPS_upstream=finite and scaled < 2e-4 and relative_l2 < 2e-3,
        legacy_BCE_geometry_and_dense_upstream=bce_error < 2e-5 and dense_error < 2e-6 and geometry_error < 1e-4), dict(
        seed=fit['seed'], step=step, forward_error=forward_error, gradient_max_absolute=absolute,
        gradient_max_scaled=scaled, gradient_relative_L2=relative_l2, BCE_error=bce_error,
        dense_upstream_error=dense_error, geometry_relative_error=geometry_error,
        geometry_upstream_diagnostic=geometry_upstream)


def final(checks, details, settings, old, values):
    result = read(ROOT / 'RESULT.json'); trial = ROOT / 'trial'
    checks['complete_three_new_six_reused_rows_no_replacements'] = (
        result['status'] == 'COMPLETE_UNTIED_CONTROL_DEVELOPMENT' and
        result['new_fits'] == 3 and result['reused_fits'] == 6 and result['new_optimizer_updates'] == 6000 and
        result['lock_sha256'] == sha(ROOT / 'LOCK.json') and not result['fresh_test_access'] and
        len(result['fits']) == 3 and len(result['rows']) == 9 and
        {f['seed'] for f in result['fits']} == set(SEEDS) and
        {(r['architecture'], r['seed']) for r in result['rows']} == {(a, s) for a in ARMS for s in SEEDS} and
        sha(ROOT / 'RESULT.json') == sha(trial / 'RESULT.json'))
    oldmap = {(r['architecture'], r['seed']): r for r in old}
    initial_ok = sampler_ok = final_ok = training_ok = released = True
    probes = {}; diagnostics = []; tie_errors = []
    for fit in result['fits']:
        seed = fit['seed']; initial = load(REPO / fit['initial_file'])['model']
        payload = load(REPO / fit['model_file']); state = payload['model']
        group_initial = load(oldmap['group6', seed]['initial'])['model']
        expected = ref.reference_deploy_state(group_initial)
        initial_ok &= set(initial) == set(expected) and all(torch.equal(initial[k], v) for k, v in expected.items())
        initial_ok &= ref.digest(initial) == fit['initial_state_sha256']
        final_ok &= ref.digest(state) == fit['final_state_sha256'] and ref.digest(state) != ref.digest(initial)
        final_ok &= sha(REPO / fit['model_file']) == fit['model_sha256'] and sha(REPO / fit['initial_file']) == fit['initial_sha256']
        batch_hash, _ = ref.expected_training_batches(27, seed)
        sampler_ok &= fit['batch_sequence_sha256'] == batch_hash == oldmap['group6', seed]['fit']['batch_sequence_sha256']
        training_ok &= fit['steps'] == fit['optimizer_updates'] == 2000 and fit['warmup_steps'] == 400 and fit['joint_steps'] == 1600
        training_ok &= {v['step'] for v in fit['gradient_probes']} == {1, 401} and len(fit['gradient_probes']) == 2
        error = constraint_error(state); tie_errors.append(dict(seed=seed, error=error)); released &= error > 1e-7
        training_ok &= abs(error-fit['final_tie_residual']) < 1e-12 and fit['free_parameters'] == 10681 and fit['MACs_per_image'] == 1527552
        training_ok &= fit['training_route'] == 'single_untied24' and fit['fresh_initialization'] and not fit['pretrained_weights_loaded'] and not fit['weak_masks_loaded']
        for item in fit['gradient_probes']:
            path = trial / item['file']; training_ok &= sha(path) == item['sha256']
            checks_probe, diagnostic = probe(path, fit, values, initial); diagnostics.append(diagnostic)
            for name, value in checks_probe.items(): probes[name] = probes.get(name, True) and value
    checks['all_three_initial_checkpoints_exact_original_group_initial_expansion'] = bool(initial_ok)
    checks['all_three_final_digests_fresh_training_and_artifact_hashes_match'] = bool(final_ok)
    checks['same_2000_dense_and_weak_batch_digests_as_paired_GroupCNN'] = bool(sampler_ok)
    checks['three_unconditional_2000_update_fits_and_six_gradient_probes'] = bool(training_ok)
    checks['trained_control_weights_have_genuinely_left_shared_C4_constraints'] = bool(released)
    for name, value in probes.items(): checks['all_six_probes_'+name] = bool(value)
    details.update(gradient_probes=diagnostics, final_constraint_deviations=tie_errors)
    provenance = truth_ok = masks_ok = classes_ok = True; metric_error = 0.; forward_error = 0.
    for row in result['rows']:
        path = REPO / row['probability_file']
        provenance &= sha(path) == row['probability_sha256'] and sha(REPO / row['model_file']) == row['model_sha256']
        if row['architecture'] != 'untied24_single':
            prior = oldmap[row['architecture'], row['seed']]
            provenance &= path == prior['probability'] and REPO / row['model_file'] == prior['model'] and row['selected'] == prior['row']['selected']
            provenance &= row['origin'] == 'immutable_historical_fit'
        else:
            provenance &= row['origin'] == 'new_untied_fit'
        with np.load(path, allow_pickle=False) as z: arrays = {k: z[k] for k in z.files}
        probability = arrays['probability']
        truth_ok &= np.array_equal(arrays['truth'], values['truth']) and np.array_equal(arrays['target'], values['target']) and np.array_equal(arrays['thresholds'], THRESHOLDS)
        scores, classes = ref.score_probability(probability, values['truth'], values['target'], THRESHOLDS)
        selected = max(scores, key=ref.selection_rank)
        metric_error = max(metric_error, difference(row['selected'], selected), difference(row['all_thresholds'], scores), difference(row['fixed05'], scores[40]))
        metric_error = max(metric_error, difference(row['primary_fine'], dict(selected=selected, all_thresholds=scores)))
        classes_ok &= np.array_equal(classes, arrays['predictions'])
        packed = np.array([np.packbits((probability[:, 0] >= t).reshape(695, 144), axis=1) for t in THRESHOLDS])
        masks_ok &= np.array_equal(packed, arrays['packed_masks'])
        state = load(REPO / row['model_file'])['model']; pieces = []
        with torch.no_grad():
            for x in values['val_x'].split(16):
                logits = (ref.reference_group(x, state) if row['architecture'] == 'group6' else
                          ref.stable_orbit_logits(x, state)[0] if row['architecture'] == 'plain12_C4pooled' else ref.reference_plain(x, state))
                pieces.append(logits.sigmoid().numpy())
        forward_error = max(forward_error, float(np.max(np.abs(np.concatenate(pieces)-probability))))
    checks['all_nine_rows_models_and_saved_probability_provenance_exact'] = bool(provenance)
    checks['all_nine_truth_masks_81_thresholds_match_old_VAL'] = bool(truth_ok)
    checks['all_nine_times81_independent_graph_classes_and_mask_bits_exact'] = bool(classes_ok and masks_ok)
    checks['all_threshold_F1_IoU_fixed05_and_selection_rank_recomputed_exactly'] = metric_error < 1e-12
    checks['all_nine_independent_CPU_forwards_match_saved_MPS_probabilities'] = forward_error <= 1e-5
    primary, fixed = statistics(result['rows'], 'selected'), statistics(result['rows'], 'fixed05')
    statistic_error = max(difference(primary, result['primary']), difference(fixed, result['fixed05_diagnostic']))
    checks['all_three_means_and_paired_seed_architecture_contrasts_correct'] = statistic_error < 1e-12
    effect = primary['contrasts']['group_minus_untied']
    guard = dict(mean_F1=effect['mean']['mask_F1'] >= .005,
                 seed_F1=all(row['mask_F1'] >= -.01 for row in effect['per_seed']),
                 mean_IoU=effect['mean']['IoU'] >= -.002, matched_MACs=True, initial_function_match=True)
    details.update(maximum_metric_error=metric_error, maximum_final_probability_error=forward_error,
                   maximum_statistic_error=statistic_error, independently_expected_primary_guards=guard)
    expected_pass = all(guard.values())
    expected_gate = dict(passed=expected_pass, guards=guard, failed=[name for name, value in guard.items() if not value],
        interpretation='descriptive_reused_VAL_mechanism_gate_not_independent_confirmation')
    checks['primary_same_compute_gate_all_guards_match_frozen_limits'] = difference(result['primary_development_gate'], expected_gate) < 1e-12
    fixed_effect = fixed['contrasts']['group_minus_untied']['mean']
    expected_admission = expected_pass and fixed_effect['mask_F1'] >= 0. and fixed_effect['IoU'] >= -.002
    admission_guards = dict(primary_gate=expected_pass, fixed05_F1=fixed_effect['mask_F1'] >= 0., fixed05_IoU=fixed_effect['IoU'] >= -.002)
    admission = dict(passed=expected_admission, guards=admission_guards,
        failed=[name for name, value in admission_guards.items() if not value],
        interpretation='Primary gate AND both fixed0.5 safeguards; no secondary rescue')
    checks['confirmation_admission_requires_primary_and_nonnegative_fixed05_F1'] = difference(result['confirmation_admission'], admission) < 1e-12


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight', action='store_true'); args = parser.parse_args()
    started = time.monotonic(); torch.set_num_threads(2); checks = {}; details = {}
    protected = c.protected_inputs()
    settings, old, values = design(checks, details)
    initial_mathematics(checks, details)
    if not args.preflight: final(checks, details, settings, old, values)
    checks['protected_scientific_sources_and_historical_artifacts_unchanged'] = protected == c.protected_inputs()
    inputs = {name: row['sha256'] for name, row in protected.items()}
    for path in [Path(__file__), STAGE / 'audit.py', ROOT.parent / 'sem_physics_materials/physics_loss_example.py']:
        inputs[relative(path)] = sha(path)
    if (ROOT / 'LOCK.json').exists(): inputs[relative(ROOT / 'LOCK.json')] = sha(ROOT / 'LOCK.json')
    if not args.preflight:
        for path in sorted((ROOT / 'trial').rglob('*')):
            if path.is_file(): inputs[relative(path)] = sha(path)
        inputs[relative(ROOT / 'RESULT.json')] = sha(ROOT / 'RESULT.json')
    passed = all(checks.values())
    result = dict(status=('PASS_' if passed else 'FAIL_') + ('UNTIED_PREFLIGHT_AUDIT' if args.preflight else 'UNTIED_AUDIT'),
        checks=checks, checks_passed=sum(bool(v) for v in checks.values()), checks_total=len(checks), details=details,
        input_sha256=inputs, optimizer_updates=0, model_fitting_performed=False, fresh_test_access=False,
        scope='Independent audit on reused historical TRAIN_VAL; not fresh statistical confirmation', seconds=time.monotonic()-started)
    snapshots = ROOT / 'audit_sources'; snapshots.mkdir(exist_ok=True)
    snapshot = snapshots / f'audit_{sha(Path(__file__))}.py'
    if not snapshot.exists(): snapshot.write_bytes(Path(__file__).read_bytes())
    path = ROOT / ('PREFLIGHT_AUDIT.json' if args.preflight else 'AUDIT.json')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=scalar_json)+'\n')
    print(json.dumps(dict(status=result['status'], checks_passed=result['checks_passed'], checks_total=result['checks_total'],
        failed=[name for name, value in checks.items() if not value], seconds=result['seconds'])))
    assert passed, 'Independent untied-control audit failed'


if __name__ == '__main__':
    with threadpool_limits(limits=2): main()
