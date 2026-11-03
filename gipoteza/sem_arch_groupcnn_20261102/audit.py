"""Independent CPU audit of fresh C4 group-CNN development on old TRAIN/VAL.

No fitting or fresh TEST access. Group indexing, deployed kernels, training
operators, gradients, provenance, thresholds and gates are reconstructed here.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import time

import numpy as np
from scipy import linalg, ndimage
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
G = ROOT.parent
REPO = G.parent
PILOT = G / 'sem_architecture_pilot_20261102'
_DIRECTION_CACHE = {}
_CLASS_CACHE = {}
ARMS = ['group6', 'plain12_C4pooled', 'plain24_C4pooled']
CLASS_SWAP = [0, 3, 6, 1, 4, 7, 2, 5, 8]
G0 = .5 / (143 * (12 / 11))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(state):
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()

def macro_f1(truth, prediction):
    confusion = np.bincount(9 * truth + prediction, minlength=81).reshape(9, 9)
    denominator = confusion.sum(0) + confusion.sum(1)
    return float(np.divide(2 * np.diag(confusion), denominator,
                           out=np.zeros(9, float), where=denominator > 0).mean())

def mean_iou(masks, target):
    union = (masks | target).sum((-2, -1))
    intersection = (masks & target).sum((-2, -1))
    return float(np.divide(intersection, union, out=np.ones(len(union)),
                           where=union > 0).mean())

def direction_class(mask):
    """Independent active-node Cholesky solve of the binary unit-edge graph."""
    mask = np.ascontiguousarray(mask, dtype=bool)
    key = np.packbits(mask).tobytes()
    if key in _DIRECTION_CACHE:
        return _DIRECTION_CACHE[key]
    components, _ = ndimage.label(mask, ndimage.generate_binary_structure(2, 1))
    spanning = np.intersect1d(components[:, 0], components[:, -1])
    spanning = spanning[spanning > 0]
    if not len(spanning):
        answer = 0
    else:
        active = np.isin(components, spanning)
        vertices = np.flatnonzero(active.ravel())
        lookup = np.full(144, -1, dtype=int)
        lookup[vertices] = np.arange(len(vertices))
        laplacian = np.zeros((len(vertices), len(vertices)), dtype=np.float64)
        for vertex in vertices:
            y, x = divmod(int(vertex), 12)
            for yy, xx in [(y + 1, x), (y, x + 1)]:
                if yy < 12 and xx < 12 and active[yy, xx]:
                    a, b = lookup[vertex], lookup[yy * 12 + xx]
                    laplacian[a, a] += 1; laplacian[b, b] += 1
                    laplacian[a, b] -= 1; laplacian[b, a] -= 1
        left = np.flatnonzero(vertices % 12 == 0)
        right = np.flatnonzero(vertices % 12 == 11)
        free = np.setdiff1d(np.arange(len(vertices)), np.r_[left, right])
        potential = np.zeros(len(vertices)); potential[right] = 1
        if len(free):
            potential[free] = linalg.solve(
                laplacian[np.ix_(free, free)],
                -laplacian[np.ix_(free, right)].sum(1), assume_a='pos',
                check_finite=False)
        conductance = float((laplacian[right] @ potential).sum()) / (12 / 11)
        answer = 0 if conductance < 1e-8 else (1 if conductance < .5 - 1e-8 else 2)
    _DIRECTION_CACHE[key] = answer
    return answer

def classify(masks):
    result = []
    for mask in np.asarray(masks, dtype=bool).reshape(-1, 12, 12):
        key = np.packbits(mask).tobytes()
        if key not in _CLASS_CACHE:
            _CLASS_CACHE[key] = 3 * direction_class(mask) + direction_class(mask.T)
        result.append(_CLASS_CACHE[key])
    return np.asarray(result, dtype=np.int64)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def reference_plain(x, state):
    for index, dilation in [(0, 1), (2, 2), (4, 4)]:
        x = F.gelu(F.conv2d(x, state[str(index) + '.weight'], state[str(index) + '.bias'], padding=dilation, dilation=dilation))
    return F.conv2d(x, state['6.weight'], state['6.bias'])


def expanded_group_kernel(kernel):
    # Explicit [output-base, output-orientation, input-base,
    # input-orientation] indexing, independent of source stack/reshape order.
    output = kernel.new_empty(kernel.shape[0], 4, kernel.shape[1], 4, 3, 3)
    for input_orientation in range(4):
        for output_orientation in range(4):
            relative = (input_orientation - output_orientation) % 4
            output[:, output_orientation, :, input_orientation] = torch.rot90(kernel[:, :, relative], output_orientation, (-2, -1))
    return output.flatten(0, 1).flatten(1, 2)


def reference_group(x, state, return_features=False):
    lift = torch.stack([torch.rot90(state['lift.weight'], k, (-2, -1)) for k in range(4)], 1).flatten(0, 1)
    z = F.gelu(F.conv2d(x, lift, state['lift.bias'].repeat_interleave(4), padding=1))
    for name, dilation in [('group2', 2), ('group4', 4)]:
        z = F.gelu(F.conv2d(z, expanded_group_kernel(state[name + '.weight']), state[name + '.bias'].repeat_interleave(4), padding=dilation, dilation=dilation))
    pooled = z.reshape(len(z), 6, 4, *z.shape[-2:]).mean(2)
    logits = F.conv2d(pooled, state['head.weight'], state['head.bias'])
    return (logits, z) if return_features else logits


def slow_group_layer(x, kernel, bias, dilation):
    # Independent direct sum of relative-orientation convolutions, with no
    # materialized block kernel; used on CPU64 for gradient/indexing proofs.
    x = x.reshape(len(x), 6, 4, *x.shape[-2:]); outputs = []
    for orientation in range(4):
        value = sum(F.conv2d(x[:, :, (orientation + relative) % 4],
            torch.rot90(kernel[:, :, relative], orientation, (-2, -1)), padding=dilation, dilation=dilation)
            for relative in range(4))
        outputs.append(value + bias[None, :, None, None])
    return torch.stack(outputs, 2).flatten(1, 2)


def stable_orbit_logits(x, state, operator=reference_plain):
    aligned = torch.stack([torch.rot90(operator(torch.rot90(x, k, (-2, -1)).contiguous(), state), -k, (-2, -1)) for k in range(4)])
    # odds of the mean Bernoulli probability, with no clipping or epsilon.
    logits = torch.logsumexp(F.logsigmoid(aligned), 0) - torch.logsumexp(F.logsigmoid(-aligned), 0)
    return logits, aligned


def reference_deploy_state(state):
    deployed = {}
    deployed['0.weight'] = torch.stack([torch.rot90(state['lift.weight'], k, (-2, -1)) for k in range(4)], 1).flatten(0, 1)
    deployed['0.bias'] = state['lift.bias'].repeat_interleave(4)
    for index, name in [(2, 'group2'), (4, 'group4')]:
        deployed[str(index) + '.weight'] = expanded_group_kernel(state[name + '.weight'])
        deployed[str(index) + '.bias'] = state[name + '.bias'].repeat_interleave(4)
    deployed['6.weight'] = state['head.weight'].repeat_interleave(4, 1) / 4
    deployed['6.bias'] = state['head.bias']
    return deployed


def expected_training_batches(budget, seed, steps=2000, batch_size=8):
    rng = np.random.default_rng(seed + 1000); result = hashlib.sha256(); first = {}
    for step in range(1, steps + 1):
        dense = rng.integers(0, budget, batch_size).astype(np.int64)
        weak = rng.integers(0, 1152, batch_size).astype(np.int64)
        result.update(dense.tobytes()); result.update(weak.tobytes())
        if step in [1, 401]: first[step] = dict(dense=dense, weak=weak)
    return result.hexdigest(), first


def selection_rank(row):
    return row['mask_F1'], row['IoU'], -abs(row['threshold'] - .5), -row['threshold']


def score_probability(probability, truth, target, thresholds):
    rows = []; predictions = []
    for threshold in thresholds:
        mask = probability[:, 0] >= threshold; prediction = classify(mask)
        rows.append(dict(threshold=float(threshold), mask_F1=macro_f1(truth, prediction), IoU=mean_iou(mask, target)))
        predictions.append(prediction)
    return rows, np.stack(predictions)


def audit_mathematics(model_module):
    torch.manual_seed(9503)
    net = model_module.GroupCNN().double()
    rng = np.random.default_rng(9503)
    images = torch.tensor(rng.normal(size=(3, 1, 12, 12)), dtype=torch.float64, requires_grad=True)
    state = {key: value.detach().clone().requires_grad_() for key, value in net.state_dict().items()}
    logits, features = reference_group(images, state, return_features=True)
    actual = net(images); source_features = net.forward_features(images)
    errors = dict(functional_logit=float((actual - logits).detach().abs().max()),
        functional_features=float((source_features - features).detach().abs().max()),
        direct_group_layer=0., group_layer_gradient=0., feature_covariance=0., output_covariance=0., deploy=0., full_parameter_gradient=0.)
    group_input = torch.tensor(rng.normal(size=(2, 24, 12, 12)), dtype=torch.float64, requires_grad=True)
    for name, dilation in [('group2', 2), ('group4', 4)]:
        kernel = state[name + '.weight']; bias = state[name + '.bias']
        direct = slow_group_layer(group_input, kernel, bias, dilation)
        block = F.conv2d(group_input, expanded_group_kernel(kernel), bias.repeat_interleave(4), padding=dilation, dilation=dilation)
        errors['direct_group_layer'] = max(errors['direct_group_layer'], float((direct - block).detach().abs().max()))
        probe = torch.tensor(rng.normal(size=direct.shape), dtype=torch.float64)
        direct_gradient = torch.autograd.grad((direct * probe).sum(), [group_input, kernel, bias], retain_graph=True)
        block_gradient = torch.autograd.grad((block * probe).sum(), [group_input, kernel, bias], retain_graph=True)
        errors['group_layer_gradient'] = max(errors['group_layer_gradient'], *(float((a - b).detach().abs().max()) for a, b in zip(direct_gradient, block_gradient)))
    reference_probe = torch.tensor(rng.normal(size=logits.shape), dtype=torch.float64)
    source_gradients = torch.autograd.grad((actual * reference_probe).sum(), list(net.parameters()))
    reference_gradients = torch.autograd.grad((logits * reference_probe).sum(), list(state.values()))
    errors['full_parameter_gradient'] = max(float((a - b).detach().abs().max()) for a, b in zip(source_gradients, reference_gradients))
    with torch.inference_mode():
        for k in range(4):
            rotated = torch.rot90(images.detach(), k, (-2, -1)).contiguous()
            expected_feature = torch.rot90(features.detach().reshape(3, 6, 4, 12, 12).roll(k, 2), k, (-2, -1)).flatten(1, 2)
            errors['feature_covariance'] = max(errors['feature_covariance'], float((net.forward_features(rotated) - expected_feature).abs().max()))
            errors['output_covariance'] = max(errors['output_covariance'], float((net(rotated) - torch.rot90(logits.detach(), k, (-2, -1))).abs().max()))
        deployed = net.deploy(); expected_deployed = reference_deploy_state(net.state_dict())
        deploy_exact = all(torch.equal(value, deployed.state_dict()[key]) for key, value in expected_deployed.items())
        errors['deploy'] = float((deployed(images.detach()) - logits.detach()).abs().max())
    counts = dict(group=sum(value.numel() for value in net.parameters()), deployed=sum(value.numel() for value in deployed.parameters()),
        narrow=sum(value.numel() for value in model_module.make_plain(12).parameters()), full=sum(value.numel() for value in model_module.make_plain(24).parameters()))
    mac_one = lambda width: 144 * (9 * width + 2 * 9 * width * width + width)
    operations = dict(group_deployed=mac_one(24), narrow_C4=4 * mac_one(12), full_C4=4 * mac_one(24))
    checks = dict(independent_C4_relative_orientation_indexing_and_channel_layout=max(errors['functional_logit'], errors['functional_features'], errors['direct_group_layer']) < 1e-11,
        direct_relative_orientation_sum_and_full_network_VJP=max(errors['group_layer_gradient'], errors['full_parameter_gradient']) < 1e-10,
        group_feature_action_roll_and_output_spatial_covariance=max(errors['feature_covariance'], errors['output_covariance']) < 1e-11,
        exact_materialized_kernels_biases_and_orientation_mean_head=deploy_exact and errors['deploy'] < 1e-11,
        effective2671_materialized10681_near_parameter2749_full10681=counts == dict(group=2671, deployed=10681, narrow=2749, full=10681),
        exact_operation_accounting=operations == dict(group_deployed=1527552, narrow_C4=1562112, full_C4=6110208))
    return checks, dict(errors=errors, parameters=counts, multiply_accumulate_operations=operations)


def numeric_compare(actual, observed):
    if isinstance(actual, dict):
        assert set(actual) == set(observed), (set(actual), set(observed))
        return max([numeric_compare(value, observed[key]) for key, value in actual.items()] + [0.])
    if isinstance(actual, list):
        assert len(actual) == len(observed)
        return max([numeric_compare(a, b) for a, b in zip(actual, observed)] + [0.])
    if isinstance(actual, bool):
        assert actual == observed
        return 0.
    return abs(float(actual) - float(observed))


def fresh_fanin_bounds(state, arm):
    checks = []
    if arm == 'group6':
        for prefix, fanin in [('lift', 9), ('group2', 216), ('group4', 216), ('head', 6)]:
            bound = 1 / np.sqrt(fanin)
            checks += [bool(value.abs().max() <= bound + 1e-7) and bool(torch.count_nonzero(value) > 0)
                for key, value in state.items() if key.startswith(prefix + '.')]
    else:
        width = 12 if arm == 'plain12_C4pooled' else 24
        for prefix, fanin in [('0', 9), ('2', 9 * width), ('4', 9 * width), ('6', width)]:
            bound = 1 / np.sqrt(fanin)
            checks += [bool(value.abs().max() <= bound + 1e-7) and bool(torch.count_nonzero(value) > 0)
                for key, value in state.items() if key.startswith(prefix + '.')]
    return len(checks) == 8 and all(checks)


def audit_training_operators(model_module):
    physics = load_module('group_audit_legacy_physics', PILOT / 'physics_adapter.py')
    historical = load_module('group_audit_original_loss', G / 'sem_physics_materials/physics_loss_example.py')
    rng = np.random.default_rng(5703)
    images = torch.tensor(rng.normal(size=(3, 1, 12, 12)), dtype=torch.float64)
    targets = torch.tensor(rng.integers(0, 2, size=(3, 1, 12, 12)), dtype=torch.float64)
    labels = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long)
    errors = dict(logits=0., probability=0., total_loss=0., common_loss_gradient=0., adapter_original_loss=0., adapter_original_gradient=0.)
    gradient_live = True; bounds_ok = True; fresh_counts = {}
    for arm in ARMS:
        net = model_module.make_model(arm, 5703).double()
        weights = {key: value.detach().clone().requires_grad_() for key, value in net.state_dict().items()}
        fresh_counts[arm] = sum(value.numel() for value in weights.values())
        bounds_ok &= fresh_fanin_bounds(weights, arm)
        result = model_module.training_details(net, arm, images)
        if arm == 'group6':
            reference = reference_group(images, weights)
            expected_probability = reference.sigmoid()
        else:
            reference, views = stable_orbit_logits(images, weights)
            expected_probability = views.sigmoid().mean(0)
        errors['logits'] = max(errors['logits'], float((result['logits'] - reference).detach().abs().max()))
        errors['probability'] = max(errors['probability'], float((result['probability'] - expected_probability).detach().abs().max()),
            float((result['logits'].sigmoid() - expected_probability).detach().abs().max()))
        # Same untouched historical loss on the canonical output for all arms.
        dense = F.binary_cross_entropy_with_logits(result['logits'], targets)
        weak = physics.geometry_loss(result['logits'], labels)
        reference_loss = F.binary_cross_entropy_with_logits(reference, targets) + historical.geometry_loss(reference, labels)
        total = dense + weak
        errors['total_loss'] = max(errors['total_loss'], float((total - reference_loss).detach().abs()))
        gradients = torch.autograd.grad(total, list(net.parameters()), retain_graph=True)
        independent = torch.autograd.grad(reference_loss, list(weights.values()), retain_graph=True)
        errors['common_loss_gradient'] = max(errors['common_loss_gradient'], *(float((a - b).detach().abs().max()) for a, b in zip(gradients, independent)))
        gradient_live &= all(torch.isfinite(grad).all().item() and grad.abs().sum().item() > 0 for grad in gradients)
        old = historical.geometry_loss(result['logits'], labels)
        errors['adapter_original_loss'] = max(errors['adapter_original_loss'], float((weak - old).detach().abs()))
        adapted_gradient = torch.autograd.grad(weak, result['logits'], retain_graph=True)[0]
        original_gradient = torch.autograd.grad(old, result['logits'])[0]
        errors['adapter_original_gradient'] = max(errors['adapter_original_gradient'], float((adapted_gradient - original_gradient).detach().abs().max()))
    checks = dict(stable_logit_of_mean_four_view_probability_training_before_legacy_loss=max(errors['logits'], errors['probability'], errors['total_loss']) < 1e-11,
        independent_common_BCE_geometry_loss_parameter_gradients=errors['common_loss_gradient'] < 1e-9 and gradient_live,
        legacy_transfer_adapter_and_original_loss_gradient_equal=errors['adapter_original_loss'] == errors['adapter_original_gradient'] == 0,
        fresh_uniform_fanin_bounds_all_layers_and_parameter_counts=bool(bounds_ok) and fresh_counts == dict(group6=2671, plain12_C4pooled=2749, plain24_C4pooled=10681))
    return checks, dict(errors=errors, fresh_parameter_counts=fresh_counts)


ROUTES = {'group6': [('group6_single', 'single')],
    'plain12_C4pooled': [('plain12_identity', 'identity'), ('plain12_C4', 'C4')],
    'plain24_C4pooled': [('plain24_identity', 'identity'), ('plain24_C4', 'C4')]}
ROUTE_NAMES = [name for arm in ARMS for name, _ in ROUTES[arm]]
COUNTS = dict(group6=2671, plain12_C4pooled=2749, plain24_C4pooled=10681)
MACS = dict(group6_single=1527552, plain12_identity=390528, plain12_C4=1562112,
    plain24_identity=1527552, plain24_C4=6110208)


def independent_gates(rows, c):
    means = {str(b): {route: {metric: float(np.mean([row['selected'][metric] for row in rows if row['budget'] == b and row['route'] == route]))
        for metric in ['mask_F1', 'IoU']} for route in ROUTE_NAMES} for b in [18, 27]}
    def compare(route, limits):
        delta = [means[str(b)]['group6_single']['mask_F1'] - means[str(b)][route]['mask_F1'] for b in [18, 27]]
        iou = float(np.mean([means[str(b)]['group6_single']['IoU'] - means[str(b)][route]['IoU'] for b in [18, 27]]))
        checks = dict(mean_F1_gain=float(np.mean(delta)) >= limits['minimum_mean_F1_gain'],
            budget_F1_guard=min(delta) >= limits['minimum_budget_F1_delta'], mean_IoU_guard=iou >= limits['minimum_mean_IoU_delta'])
        return dict(delta=float(np.mean(delta)), budget_deltas=list(map(float, delta)), IoU_delta=iou, checks=checks)
    a = compare('plain12_C4', c['primary_gates']); p = COUNTS['group6'] / COUNTS['plain12_C4pooled']; m = MACS['group6_single'] / MACS['plain12_C4']
    a['checks'].update(parameter_budget_match=abs(1 - p) <= c['primary_gates']['maximum_budget_ratio_deviation'],
        MAC_budget_match=abs(1 - m) <= c['primary_gates']['maximum_budget_ratio_deviation'])
    a.update(free_parameter_ratio=p, MAC_ratio=m, passed=all(a['checks'].values()))
    b = compare('plain24_C4', c['secondary_gates']); m = MACS['group6_single'] / MACS['plain24_C4']
    b['checks']['MAC_efficiency_guard'] = m <= c['secondary_gates']['maximum_MAC_ratio']
    b.update(MAC_ratio=m, passed=all(b['checks'].values()))
    return dict(A_matched_quality=a, B_full_efficiency=b, ANY_GOAL_PASS=bool(a['passed'] or b['passed'])), means


def audit_gradient_probe(probe, arm, budget, seed, expected_indices, initial, dense_data, weak_data, physics):
    checks = {}; step = probe['step']; warm = step == 1; expected = expected_indices[step]
    metadata = (probe['architecture'] == arm and probe['budget'] == budget and probe['seed'] == seed and step in [1, 401] and
        probe['phase'] == ('BCE_WARMUP' if warm else 'BCE_PLUS_OLD_GEOMETRY') and probe['weak_loss_used'] == (not warm) and
        probe['optimizer_step_not_yet_applied'] and np.array_equal(probe['dense_indices'].numpy(), expected['dense']) and
        np.array_equal(probe['weak_indices'].numpy(), expected['weak']) and
        torch.equal(probe['dense_x'], dense_data['x'][expected['dense']]) and torch.equal(probe['weak_x'], weak_data['x'][expected['weak']]) and
        torch.equal(probe['dense_mask'], dense_data['mask'][expected['dense']]) and torch.equal(probe['weak_label'], weak_data['label'][expected['weak']]))
    if warm: metadata &= digest(probe['state_before']) == digest(initial)
    state = {key: value.detach().clone().requires_grad_() for key, value in probe['state_before'].items()}
    images = probe['dense_x'] if warm else torch.cat((probe['dense_x'], probe['weak_x']))
    if arm == 'group6': z = reference_group(images, state); views = None
    else: z, views = stable_orbit_logits(images, state)
    dense, weak = z[:8], None if warm else z[8:]
    dense_error = float((dense - probe['dense_logits']).detach().abs().max())
    weak_error = 0. if warm else float((weak - probe['weak_logits']).detach().abs().max())
    views_error = 0. if views is None else float((views - probe['aligned_logits']).detach().abs().max())
    actual_upstream = probe['dense_logits_gradient'] if warm else torch.cat((probe['dense_logits_gradient'], probe['weak_logits_gradient']))
    gradients = torch.autograd.grad(z, list(state.values()), grad_outputs=actual_upstream)
    max_abs = max_scaled = max_l2 = 0.; finite = True
    for (key, value), gradient in zip(state.items(), gradients):
        recorded = probe['gradients'][key]; difference = gradient - recorded
        max_abs = max(max_abs, float(difference.abs().max()))
        max_scaled = max(max_scaled, float(difference.abs().max()) / max(1., float(recorded.abs().max())))
        max_l2 = max(max_l2, float(difference.norm()) / max(1e-6, float(recorded.norm())))
        finite &= bool(torch.isfinite(recorded).all()) and bool(torch.isfinite(gradient).all())
    operating_dense = probe['dense_logits'].detach().clone().requires_grad_()
    bce = F.binary_cross_entropy_with_logits(operating_dense, probe['dense_mask'])
    dense_upstream = torch.autograd.grad(bce, operating_dense)[0]
    bce_error = abs(float(bce.detach()) - probe['BCE'])
    upstream_error = float((dense_upstream - probe['dense_logits_gradient']).abs().max())
    geometry_error = geometry_gradient_diagnostic = 0.
    if warm:
        metadata &= probe['weak_logits'] is None and probe['weak_logits_gradient'] is None and probe['geometry_loss'] is None and probe['total_loss'] == probe['BCE']
    else:
        operating_weak = probe['weak_logits'].detach().clone().requires_grad_()
        labels = probe['weak_label']; directions = torch.stack((labels // 3, labels % 3), 1)
        geometry = physics.geometry_loss(operating_weak, directions)
        geometry_error = abs(float(geometry.detach()) - probe['geometry_loss']) / max(1., abs(probe['geometry_loss']))
        weak_upstream = torch.autograd.grad(geometry, operating_weak)[0]
        geometry_gradient_diagnostic = float((weak_upstream - probe['weak_logits_gradient']).abs().max())
        metadata &= abs(probe['total_loss'] - probe['BCE'] - probe['geometry_loss']) < 2e-5 and bool(torch.isfinite(probe['weak_logits_gradient']).all())
    metadata &= set(state) == set(probe['gradients']) == set(initial) and all(torch.isfinite(v).all().item() for v in state.values())
    checks['metadata'] = bool(metadata)
    checks['operator_forward'] = max(dense_error, weak_error, views_error) < 1e-3
    checks['actual_MPS_upstream_independent_CPU_parameter_VJP'] = bool(finite) and max_scaled < 2e-4 and max_l2 < 2e-3
    checks['legacy_components_and_dense_BCE_upstream'] = bce_error < 2e-5 and upstream_error < 2e-6 and geometry_error < 1e-4
    return checks, dict(architecture=arm, budget=budget, seed=seed, step=step, dense_logit=dense_error, weak_logit=weak_error,
        aligned_logits=views_error, parameter_gradient_absolute=max_abs, parameter_gradient_scaled=max_scaled,
        parameter_gradient_relative_L2=max_l2, BCE=bce_error, BCE_upstream=upstream_error, geometry_relative=geometry_error,
        geometry_upstream_CPU_MPS_diagnostic=geometry_gradient_diagnostic)


def audit_benchmark(folder, bench, images, c):
    with np.load(folder / bench['raw_file'], allow_pickle=False) as z: values = {key: z[key] for key in z.files}
    orders, times = values['randomized_route_order'], values['latency_ns']; rows = {}
    rng = np.random.default_rng(c['benchmark_seed'] + bench['budget'] + bench['seed'])
    expected = np.array([rng.permutation(5) for _ in range(50)])
    valid = (sha(folder / bench['raw_file']) == bench['raw_sha256'] and values['route_names'].tolist() == ROUTE_NAMES and
        np.array_equal(orders, expected) and orders.shape == times.shape == (50, 5) and np.all(times > 0) and
        np.array_equal(values['x'], images[:32].numpy()) and bench['batch_size'] == 32 and
        bench['warmup_batches_per_route'] == 10 and bench['warmup_images_per_route'] == 320 and bench['repeats'] == 50 and
        bench['random_seed'] == c['benchmark_seed'] + bench['budget'] + bench['seed'] and bench['materialization_setup_seconds'] >= 0 and
        bench['MPS_sync_before_and_after_every_measurement'] and bench['materialization_excluded'] and bench['CPU_geometry_readout_excluded'] and
        bench['data_and_host_transfer_excluded'] and not bench['end_to_end_measured'] and bench['CNN_batch_calls_per_route'] == 1 and
        bench['plain_C4_internal_batch_multiplier'] == 4)
    for i, route in enumerate(ROUTE_NAMES):
        rows[route] = dict(median_seconds=float(np.median(times[:, i])) / 1e9, p10_seconds=float(np.quantile(times[:, i], .1)) / 1e9,
            p90_seconds=float(np.quantile(times[:, i], .9)) / 1e9, images_per_batch=32, MACs_per_image=MACS[route])
    valid &= numeric_compare(rows, bench['routes']) < 1e-12
    valid &= bench['paired_group_over_narrow_C4_median'] == float(np.median(times[:, 0] / times[:, 2]))
    valid &= bench['paired_group_over_full_C4_median'] == float(np.median(times[:, 0] / times[:, 4]))
    return bool(valid)


def audit_trial(folder):
    started = time.monotonic(); read = lambda name: json.loads((folder / name).read_text())
    result, frozen, c = (read(name) for name in ['RESULT.json', 'FROZEN.json', 'CONFIG_SNAPSHOT.json'])
    before = {str(path): sha(path) for path in [folder / 'RESULT.json', folder / 'FROZEN.json', folder / 'CONFIG_SNAPSHOT.json',
        folder / 'PROTOCOL.md', PILOT / 'TRAIN_VAL.npz', PILOT / 'SUBSETS.json']}
    before.update({str(ROOT / name): sha(ROOT / name) for name in frozen['source_sha256']})
    checks = {}
    checks['frozen_sources_config_protocol_and_old_TRAIN_VAL_only_scope'] = (
        result['status'] == 'COMPLETE_GROUPCNN_DEVELOPMENT' and frozen['fresh_test_access'] is False and
        sha(folder / 'FROZEN.json') == result['frozen_sha256'] and sha(folder / 'CONFIG_SNAPSHOT.json') == frozen['config_sha256'] and
        sha(folder / 'PROTOCOL.md') == frozen['protocol_sha256'] == c['protocol_sha256'] and
        all(sha(ROOT / name) == value for name, value in frozen['source_sha256'].items()) and
        all(sha(folder / 'source' / name) == frozen['source_sha256'][name] for name in ['model.py', 'runner.py']))
    checks['fixed_fresh_2000step_400warmup_legacy_loss_design_and_both_separate_goals'] = (
        c['candidate'] == 'known_C4_group_CNN' and c['training_arms'] == ARMS and c['budgets'] == [18, 27] and
        c['screen_seeds'] == [5701] and c['replication_seeds'] == [5701, 5721, 5757] and c['subset_seed'] == 5303 and
        c['steps'] == 2000 and c['warmup_steps'] == 400 and c['batch_size'] == 8 and c['learning_rate'] == .0005 and
        c['weight_decay'] == .0001 and c['physics_weight'] == 1. and c['batch_seed_offset'] == 1000 and
        c['thresholds'] == [round(.1 + .01 * i, 2) for i in range(81)] and c['secondary_thresholds'] == [.3, .4, .45, .5, .55, .6, .7] and
        c['primary_gates'] == dict(minimum_mean_F1_gain=.005, minimum_budget_F1_delta=-.0025, minimum_mean_IoU_delta=-.002, maximum_budget_ratio_deviation=.03) and
        c['secondary_gates'] == dict(minimum_mean_F1_gain=-.005, minimum_budget_F1_delta=-.01, minimum_mean_IoU_delta=-.002, maximum_MAC_ratio=.30) and
        c['replication_success_rule'] == '(A_all3 AND A_new2) OR (B_all3 AND B_new2); no cross-goal mixing' and
        c['benchmark_batch_size'] == 32 and c['benchmark_warmup'] == 10 and c['benchmark_repeats'] == 50)
    tree = ast.parse((folder / 'source' / 'runner.py').read_text()); functions = {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    fit_source, input_source = functions['fit'], functions['inputs']
    checks['source_training_schedule_all_weights_canonical_targets_and_weak_mask_exclusion'] = (
        "joint = step > c['warmup_steps']" in fit_source and "loss = bce if geometry is None else bce + c['physics_weight'] * geometry" in fit_source and
        'params = list(net.parameters())' in fit_source and 'optimizer.step()' in fit_source and
        "training_details(net, architecture, torch.cat((dx, wx)) if joint else dx)" in fit_source and
        "physics.geometry_loss(weak, torch.stack((labels // 3, labels % 3), dim=1))" in fit_source and
        "assert 'mask' not in data[budget]['weak']" in input_source and 'torch.load' not in input_source and
        'weak_masks_loaded=False' in input_source and 'pretrained_weights_read=False' in input_source)
    model_tree = ast.parse((folder / 'source' / 'model.py').read_text()); model_functions = {n.name: ast.unparse(n) for n in model_tree.body if isinstance(n, ast.FunctionDef)}
    checks['source_fresh_seed_and_batched_four_view_shared_CNN_without_augmentation'] = (
        'torch.manual_seed(int(seed))' in model_functions['make_model'] and 'load_state_dict' not in model_functions['make_model'] and
        'torch.stack' in model_functions['aligned_view_logits'] and model_functions['aligned_view_logits'].count('network(') == 1 and
        'torch.logsumexp(F.logsigmoid(views), 0) - torch.logsumexp(F.logsigmoid(-views), 0)' in model_functions['training_details'])
    checks['source_benchmark_synchronizes_raw_forward_timing_and_excludes_setup_readout'] = (
        functions['benchmark'].count('torch.mps.synchronize()') >= 3 and 'time.perf_counter_ns()' in functions['benchmark'] and
        'predict_probability(net, x, mode)' in functions['benchmark'] and 'deploy(' not in functions['benchmark'] and
        'optimizer' not in functions['benchmark'] and 'hard_classes' not in functions['benchmark'])
    preflight = json.loads((ROOT / 'PREFLIGHT.json').read_text())
    checks['saved_CPU_preflight_current_sources_no_updates'] = preflight['status'] == 'PASS_PREFLIGHT' and preflight['optimizer_updates'] == 0 and all(sha(ROOT / name) == value for name, value in preflight['source_sha256'].items())
    with np.load(PILOT / 'TRAIN_VAL.npz', allow_pickle=False) as z:
        data = {name: z[name] for name in ['X', 'Y', 'label', 'role', 'frame']}
    subsets = json.loads((PILOT / 'SUBSETS.json').read_text()); vi = np.flatnonzero(data['role'] == 'val'); wi = np.flatnonzero(data['role'] == 'weak')
    images = torch.tensor(5 * (data['X'][vi] - .2)); truth = data['label'][vi]; target = data['Y'][vi, 0].astype(bool)
    lineage = frozen['lineage']; expected_lineage = {}
    training = {}
    for budget in [18, 27]:
        di = np.array(subsets['dense_indices'][str(budget)], dtype=np.int64)
        expected_lineage[str(budget)] = dict(dense=di.tolist(), weak=wi.tolist(), val=vi.tolist())
        training[budget] = dict(dense=dict(x=torch.tensor(5 * (data['X'][di] - .2)), mask=torch.tensor(data['Y'][di].astype(np.float32))),
            weak=dict(x=torch.tensor(5 * (data['X'][wi] - .2)), label=torch.tensor(data['label'][wi])))
    checks['exact_original_archive_subsets_role_frame_exclusion_and_balanced_dense_training'] = (
        sha(PILOT / 'TRAIN_VAL.npz') == lineage['TRAIN_VAL_sha256'] == subsets['train_val_sha256'] == '119cf2f5286067ad06c7e83034516abb0bd2a946383fe4c0987ba7a16fa15ade' and
        sha(PILOT / 'SUBSETS.json') == lineage['SUBSETS_sha256'] and lineage['indices'] == expected_lineage and
        not lineage['pretrained_weights_read'] and not lineage['weak_masks_loaded'] and len(wi) == 1152 and len(vi) == 695 and
        set(data['frame'][vi]) == {6, 7} and set(data['frame'][np.flatnonzero(data['role'] != 'val')]) == set(range(6)) and
        set(subsets['dense_indices']['18']).issubset(subsets['dense_indices']['27']) and
        all(np.array_equal(np.bincount(data['label'][subsets['dense_indices'][str(b)]], minlength=9), np.full(9, b // 9)) for b in [18, 27]))
    checks['independent_binary_graph_truth_and_odd_rotation_class_swap'] = np.array_equal(classify(target), truth) and all(np.array_equal(classify(np.rot90(target, k, (-2, -1))), np.array(CLASS_SWAP)[truth] if k % 2 else truth) for k in range(4))
    model_module = load_module('group_audit_frozen_model', ROOT / 'model.py'); physics = load_module('group_audit_fixed_loss', PILOT / 'physics_adapter.py')
    math_checks, math_proof = audit_mathematics(model_module); checks.update(math_checks)
    operator_checks, operator_proof = audit_training_operators(model_module); checks.update(operator_checks)
    seeds = [5701] if result['phase'] == 'screen' else [5721, 5757]
    units = {(b, seed) for b in [18, 27] for seed in seeds}; fits, rows = result['fits'], result['rows']
    checks['exact_six_fits_ten_routes_twelve_thousand_updates_per_seed'] = (
        result['seeds'] == frozen['seeds'] == seeds and result['fitted_models'] == frozen['fitted_models'] == len(fits) == 6 * len(seeds) and
        result['updates'] == frozen['total_updates'] == 12000 * len(seeds) and len(rows) == frozen['evaluation_routes'] == 10 * len(seeds) and
        {(r['architecture'], r['budget'], r['seed']) for r in fits} == {(a, b, s) for a in ARMS for b, s in units} and
        {(r['route'], r['budget'], r['seed']) for r in rows} == {(route, b, s) for route in ROUTE_NAMES for b, s in units} and
        read('FIT_RESULTS.json') == fits and read('CHECKPOINT_HASHES.json') == result['checkpoint_hashes'])
    checks['all_initial_final_and_materialized_checkpoint_file_hashes'] = len(result['checkpoint_hashes']) == 14 * len(seeds) and all(sha(folder / 'models' / name) == value for name, value in result['checkpoint_hashes'].items())
    fit_ok = initial_ok = probe_ok = deploy_ok = True; states = {}; fresh_sha = {}; gradient_proofs = []
    for fit in fits:
        arm, budget, seed = fit['architecture'], fit['budget'], fit['seed']; unit = arm, budget, seed
        initial_payload = torch.load(folder / fit['initial_model_file'], map_location='cpu', weights_only=True)
        final_payload = torch.load(folder / fit['model_file'], map_location='cpu', weights_only=True)
        initial, state = initial_payload['model'], final_payload['model']; states[unit] = state
        fresh = model_module.make_model(arm, seed).state_dict(); fresh_sha.setdefault((arm, seed), set()).add(digest(initial))
        initial_ok &= (sha(folder / fit['initial_model_file']) == fit['initial_model_sha256'] and initial_payload['seed'] == seed and initial_payload['architecture'] == arm and
            digest(initial) == digest(fresh) == fit['initial_state_sha256'] != fit['final_state_sha256'] and fresh_fanin_bounds(initial, arm))
        batch_sha, indices = expected_training_batches(budget, seed)
        fitting = {k: v for k, v in fit.items() if k not in ['architecture', 'budget', 'seed', 'model_file', 'model_sha256', 'initial_model_file', 'initial_model_sha256']}
        fit_ok &= (final_payload['settings'] == fitting and final_payload['architecture'] == arm and final_payload['budget'] == budget and final_payload['seed'] == seed and
            sha(folder / fit['model_file']) == fit['model_sha256'] and digest(state) == fit['final_state_sha256'] and sum(v.numel() for v in state.values()) == fit['free_parameters'] == COUNTS[arm] and
            all(torch.isfinite(v).all().item() for v in state.values()) and fit['steps'] == fit['optimizer_updates'] == 2000 and fit['warmup_steps'] == 400 and fit['joint_steps'] == 1600 and
            fit['fit_count'] == 1 and fit['fresh_initialization'] and fit['batch_sequence_sha256'] == batch_sha and
            fit['training_route'] == ('single_group_equivariant' if arm == 'group6' else 'four_view_probability_pool'))
        probe_ok &= len(fit['gradient_probes']) == 2 and {p['step'] for p in fit['gradient_probes']} == {1, 401}
        for ref in fit['gradient_probes']:
            probe_ok &= sha(folder / ref['file']) == ref['sha256']
            probe = torch.load(folder / ref['file'], map_location='cpu', weights_only=True)
            pc, pe = audit_gradient_probe(probe, arm, budget, seed, indices, initial, training[budget]['dense'], training[budget]['weak'], physics)
            probe_ok &= all(pc.values()); gradient_proofs.append(dict(checks=pc, **pe))
        if arm == 'group6':
            path = folder / 'models' / f'group6_b{budget}_s{seed}_deployed.pt'; payload = torch.load(path, map_location='cpu', weights_only=True)
            deployed = payload['model']; expected_deployed = reference_deploy_state(state)
            deploy_ok &= (payload['free_model_sha256'] == fit['model_sha256'] and payload['free_parameters'] == 2671 and payload['materialized_coefficients'] == 10681 and
                all(torch.equal(value, deployed[key]) for key, value in expected_deployed.items()))
    initial_ok &= all(len(value) == 1 for value in fresh_sha.values())
    checks['all_fresh_seed_initial_states_fanin_bounds_and_same_seed_across_budgets'] = bool(initial_ok)
    checks['all_2000steps400plus1600_metadata_final_states_and_paired_dense_weak_batches'] = bool(fit_ok) and all(len({fit['batch_sequence_sha256'] for fit in fits if (fit['budget'], fit['seed']) == unit}) == 1 for unit in units)
    checks['all_step1_and401_actual_MPS_metadata_legacy_loss_and_independent_parameter_VJPs'] = bool(probe_ok)
    checks['all_trained_deployed_kernels_biases_and_repeated_head_coefficients_exact'] = bool(deploy_ok)
    errors = dict(full_probability=0., metrics=0., canonical_pool_probability=0., fixture_probability=0., group_features_relative=0., covariance=0., deployment=0.)
    scores_ok = technical_ok = row_ok = True; recomputed = []
    for row in rows:
        arm, route, budget, seed = row['architecture'], row['route'], row['budget'], row['seed']; state = states[arm, budget, seed]
        mode = dict(ROUTES[arm])[route]; stem = f'{route}_b{budget}_s{seed}'
        fit = next(f for f in fits if (f['architecture'], f['budget'], f['seed']) == (arm, budget, seed))
        row_ok &= (row['model_file'] == fit['model_file'] and row['model_sha256'] == fit['model_sha256'] and row['free_parameters'] == COUNTS[arm] and
            row['materialized_coefficients'] == (10681 if arm == 'group6' else COUNTS[arm]) and row['MACs_per_image'] == MACS[route] and
            row['CNN_passes_per_image'] == row['internal_batch_multiplier'] == (4 if mode == 'C4' else 1) and row['CNN_batch_calls'] == 1)
        with np.load(folder / 'val_probs' / f'{stem}.npz', allow_pickle=False) as z: saved = {key: z[key] for key in z.files}
        scores_ok &= (saved['probability'].shape == (695, 1, 12, 12) and np.isfinite(saved['probability']).all() and saved['probability'].min() >= 0 and saved['probability'].max() <= 1 and
            np.array_equal(saved['truth'], truth) and np.array_equal(saved['target'], target) and saved['thresholds'].tolist() == c['thresholds'] and saved['secondary_thresholds'].tolist() == c['secondary_thresholds'])
        with torch.inference_mode():
            pieces = []
            for x in images.split(16):
                if arm == 'group6': p = reference_group(x, state).sigmoid()
                elif mode == 'identity': p = reference_plain(x, state).sigmoid()
                else: _, z = stable_orbit_logits(x, state); p = z.sigmoid().mean(0)
                pieces.append(p.numpy())
        errors['full_probability'] = max(errors['full_probability'], float(np.max(np.abs(np.concatenate(pieces) - saved['probability']))))
        all_scores, predictions = score_probability(saved['probability'], truth, target, c['thresholds'])
        scores_ok &= np.array_equal(predictions, saved['predictions'])
        packed = np.stack([np.packbits((saved['probability'][:, 0] >= threshold).reshape(695, 144), axis=1) for threshold in c['thresholds']])
        scores_ok &= np.array_equal(packed, saved['packed_masks'])
        secondary_indices = [c['thresholds'].index(value) for value in c['secondary_thresholds']]
        secondary = [all_scores[i] for i in secondary_indices]
        selected, selected_secondary = max(all_scores, key=selection_rank), max(secondary, key=selection_rank)
        errors['metrics'] = max(errors['metrics'], numeric_compare(dict(selected=selected, all_thresholds=all_scores), row['primary_fine']),
            numeric_compare(dict(selected=selected_secondary, all_thresholds=secondary), row['secondary_seven']), numeric_compare(selected, row['selected']))
        scores_ok &= np.array_equal(saved['secondary_predictions'], predictions[secondary_indices]) and np.array_equal(saved['secondary_packed_masks'], packed[secondary_indices])
        with np.load(folder / 'fixtures' / f'{stem}.npz', allow_pickle=False) as z: fixture = {key: z[key] for key in z.files}
        error = float(np.max(np.abs(fixture['restored_probability'] - fixture['probability'][None])))
        technical_ok &= (np.array_equal(fixture['x'], images[:16].numpy()) and np.array_equal(fixture['probability'], saved['probability'][:16]) and
            row['C4_equivariance_required'] == (mode != 'identity') and abs(error - row['C4_probability_max_error']) < 1e-12)
        if mode != 'identity':
            errors['covariance'] = max(errors['covariance'], error); technical_ok &= error <= c['trained_probability_tolerance']
        if mode == 'C4':
            restored_probability = torch.tensor(fixture['aligned_logits']).sigmoid().mean(0).numpy()
            errors['canonical_pool_probability'] = max(errors['canonical_pool_probability'], float(np.max(np.abs(restored_probability - fixture['probability']))))
        if arm == 'group6':
            free_error = float(np.max(np.abs(fixture['free_group_probability'] - fixture['probability'])))
            errors['deployment'] = max(errors['deployment'], free_error)
            technical_ok &= abs(free_error - row['deployment_probability_max_error']) < 1e-12 and free_error <= c['trained_probability_tolerance'] and row['feature_action_relative_max_error'] <= c['trained_probability_tolerance']
            with torch.inference_mode(): _, features = reference_group(images[:16], state, True)
            errors['group_features_relative'] = max(errors['group_features_relative'], float(np.max(np.abs(features.numpy().reshape(16, 6, 4, 12, 12) - fixture['group_features']))) / max(1., float(np.max(np.abs(fixture['group_features'])))))
        recomputed.append(dict(route=route, budget=budget, seed=seed, selected=selected))
        print(json.dumps(dict(audit_progress=stem, independent_thresholds=81, maximum_probability_error=errors['full_probability'])), flush=True)
    checks['all_route_parameter_materialized_storage_operation_and_pass_accounting'] = bool(row_ok)
    checks['independent_all_81_hard_graph_classes_packed_masks_primary_and_seven_grid_metrics_rank'] = bool(scores_ok) and errors['metrics'] < 2e-7
    checks['independent_full_VAL_group_and_plain_single_four_view_probability_forward'] = errors['full_probability'] < 1e-4
    checks['trained_operating_MPS_required_C4_and_deployment_probability_tolerances'] = bool(technical_ok) and max(errors['covariance'], errors['deployment']) <= c['trained_probability_tolerance'] and errors['canonical_pool_probability'] < 8 * np.finfo(np.float32).eps and errors['group_features_relative'] < 1e-4
    benchmark_ok = len(result['benchmarks']) == len(units) and {(b['budget'], b['seed']) for b in result['benchmarks']} == units and read('BENCHMARK_RESULTS.json') == result['benchmarks']
    for bench in result['benchmarks']: benchmark_ok &= audit_benchmark(folder, bench, images, c)
    checks['all_actual_latency_samples_randomized_orders_warmups_summaries_and_CNN_only_scope'] = bool(benchmark_ok)
    gates, means = independent_gates(recomputed, c)
    checks['both_goals_primary81_grid_gains_budget_IoU_and_compute_gates_recomputed'] = numeric_compare(gates, {k: v for k, v in result['gates'].items() if k != 'interpretation'}) < 2e-7 and numeric_compare(means, result['means']) < 2e-7
    replication = {}
    if result['phase'] == 'replicate':
        aggregate = read('AGGREGATE_RESULT.json'); old_folder = ROOT / aggregate['source_trials'][0]
        old = json.loads((old_folder / 'RESULT.json').read_text()); old_audit = json.loads((old_folder / 'AUDIT.json').read_text())
        valid = old['seeds'] == [5701] and old['gates']['ANY_GOAL_PASS'] and old_audit['status'] == 'PASS_AUDIT' and old_audit['result_sha256'] == sha(old_folder / 'RESULT.json')
        valid &= aggregate['rows'] == old['rows'] + rows and aggregate['fits'] == old['fits'] + fits and aggregate['benchmarks'] == old['benchmarks'] + result['benchmarks']
        all_gates, all_means = independent_gates(old['rows'] + recomputed, c)
        robust_a = all_gates['A_matched_quality']['passed'] and gates['A_matched_quality']['passed']; robust_b = all_gates['B_full_efficiency']['passed'] and gates['B_full_efficiency']['passed']
        valid &= (numeric_compare(all_gates, {k: v for k, v in aggregate['gates'].items() if k != 'interpretation'}) < 2e-7 and
            numeric_compare(gates, {k: v for k, v in aggregate['new_seed_gates'].items() if k != 'interpretation'}) < 2e-7 and numeric_compare(all_means, aggregate['means']) < 2e-7 and
            aggregate['robust_primary_pass'] == robust_a and aggregate['robust_secondary_pass'] == robust_b and aggregate['robust_ANY_GOAL_PASS'] == (robust_a or robust_b))
        checks['valid_screen_all3_new2_exact_aggregation_and_same_goal_robust_conjunction'] = bool(valid)
        replication = dict(all_three=all_gates, new_two=gates)
    recovery = {}
    if 'recovery' in frozen:
        recovery_checks, recovery = audit_recovery(folder, result, frozen)
        checks.update(recovery_checks)
    checks['all_sources_protocol_config_models_and_TRAIN_VAL_unchanged_during_CPU_audit'] = all(sha(path) == value for path, value in before.items())
    return dict(status='PASS_AUDIT' if all(checks.values()) else 'FAIL_AUDIT', trial=str(folder.relative_to(ROOT)),
        scope='INDEPENDENT_CPU_FRESH_GROUPCNN_OLD_TRAIN_VAL_ONLY', audit_source_sha256=sha(__file__), result_sha256=sha(folder / 'RESULT.json'),
        checks=checks, checks_passed=sum(bool(value) for value in checks.values()), checks_total=len(checks), maximum_errors=errors,
        mathematical_proof=math_proof, training_operator_proof=operator_proof, actual_MPS_gradient_proofs=gradient_proofs,
        independent_gates=gates, replication=replication, recovery=recovery, seconds=time.monotonic() - started,
        numerical_tolerances=dict(CPU_MPS_probability=1e-4, operator_logits=1e-3, saved_MPS_trained_probability=c['trained_probability_tolerance'], parameter_VJP_scaled=2e-4, parameter_VJP_relative_L2=2e-3),
        gradient_note='Independent CPU operator VJPs use the recorded actual MPS loss upstreams. Legacy CPU/MPS geometry upstream difference is separately disclosed; no altered loss or solver is used.',
        caveat='Known C4 weight sharing; reused VAL development only, no independent TEST, no originality claim. Materialized group storage is10681 repeated coefficients despite2671 free training parameters. Latency is descriptive CNN-only, not end-to-end.')


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def audit_recovery(folder, result, frozen):
    recovery = frozen['recovery']; proof = json.loads((folder / 'RECOVERY_PROOF.json').read_text())
    failed = ROOT / recovery['original_failed_trial']; old_frozen = json.loads((failed / 'FROZEN.json').read_text()); failure = json.loads((failed / 'FAILURE.json').read_text())
    files = {str(path.relative_to(ROOT)): dict(sha256=sha(path), bytes=path.stat().st_size) for path in sorted(failed.rglob('*')) if path.is_file()}
    protected = recovery['preserved_source_config_protocol_sha256']; c = json.loads((folder / 'CONFIG_SNAPSHOT.json').read_text())
    signature = json_digest(dict(configuration=c, source_sha256=old_frozen['source_sha256'], TRAIN_VAL_sha256=sha(PILOT / 'TRAIN_VAL.npz')))
    checks = {}
    checks['failed_metadata_trial_entire_inventory_and_original_scientific_sources_preserved'] = (
        failure['status'] == 'FAILED' and failure['error_type'] == 'TypeError' and "multiple values for keyword argument 'free_parameters'" in failure['error'] and
        not (failed / 'RESULT.json').exists() and files == recovery['failed_inventory'] and json_digest(files) == recovery['failed_inventory_sha256'] and
        recovery == result['recovery'] == json.loads((folder / 'RECOVERY_MANIFEST.json').read_text()) and
        all(sha(ROOT / name) == value for name, value in old_frozen['source_sha256'].items()) and all(sha(ROOT / name) == value for name, value in protected.items()) and
        (failed / 'CONFIG_SNAPSHOT.json').read_bytes() == (folder / 'CONFIG_SNAPSHOT.json').read_bytes() and (failed / 'PROTOCOL.md').read_bytes() == (folder / 'PROTOCOL.md').read_bytes() and
        signature == recovery['scientific_signature_unchanged'] == frozen['signature'] == result['signature'])
    preflight = json.loads((ROOT / 'RECOVERY_PREFLIGHT.json').read_text())
    source = (folder / 'source' / 'recovery.py').read_text(); tree = ast.parse(source); unparsed = ast.unparse(tree)
    functions = {node.name: ast.unparse(node) for node in tree.body if isinstance(node, ast.FunctionDef)}
    source_ok = (recovery['recovery_source_sha256'] == sha(ROOT / 'recovery.py') == sha(folder / 'source' / 'recovery.py') == preflight['executed_recovery_sha256'] and
        preflight['status'] == 'PASS_METADATA_RECOVERY_PREFLIGHT' and preflight['conflicting_overlap_rejected'] and preflight['no_optimizer_steps'] and
        preflight['same_entrypoint_supports_replication'] and len(preflight['row_construction_variants']) == 5 and
        {row['route'] for row in preflight['row_construction_variants']} == set(ROUTE_NAMES) and
        all(row['row_constructs'] and row['JSON_serializes'] for row in preflight['row_construction_variants']) and
        "assert result[key] == value" in functions['merge_checked'] and
        all('r.' + name + '(' in unparsed for name in ['fit', 'evaluate', 'route_fixture', 'benchmark', 'gates', 'inputs']) and
        not any(isinstance(node, ast.Attribute) and node.attr in ['step', 'backward', 'AdamW'] for node in ast.walk(tree)))
    expected_fields = {'group6_single': ['free_parameters', 'materialized_coefficients'], 'plain12_identity': [], 'plain12_C4': [], 'plain24_identity': [], 'plain24_C4': []}
    source_ok &= all(row['overlapping_fields_checked'] == expected_fields[row['route']] for row in preflight['row_construction_variants'])
    source_ok &= recovery['no_model_data_loss_gate_changes'] and proof['scientific_functions_reused_unchanged'] and proof['metadata_merge_equal_overlap_asserted']
    checks['metadata_only_equal_overlap_merge_all_five_variants_and_unchanged_scientific_functions'] = bool(source_ok)
    inherited_count = 1 if result['phase'] == 'screen' else 0; total = len(result['fits']); new = total - inherited_count
    count_ok = (recovery['inherited_fits'] == result['inherited_fits'] == proof['inherited_fits'] == inherited_count and
        recovery['inherited_updates'] == result['inherited_updates'] == proof['inherited_updates'] == 2000 * inherited_count and
        recovery['actual_new_fits'] == result['actual_new_fits'] == proof['actual_new_fits'] == new and
        recovery['actual_new_updates'] == result['actual_new_updates'] == proof['actual_new_updates'] == 2000 * new and
        result['total_distinct_fits'] == proof['total_distinct_fits'] == total and proof['total_unique_updates'] == result['updates'] == 2000 * total and
        proof['status'] == 'PASS_METADATA_ONLY_RECOVERY_PROOF' and sha(folder / 'RECOVERY_PROOF.json') == result['recovery_proof_sha256'] and
        proof['failed_inventory_unchanged'] and proof['original_model_runner_config_protocol_unchanged'] and proof['no_inherited_fit_retrained'] and
        proof['fit_provenance'] == result['fit_provenance'] == json.loads((folder / 'FIT_PROVENANCE.json').read_text()))
    key = lambda row: (row['architecture'], row['budget'], row['seed'])
    count_ok &= len(result['fit_provenance']) == total and {key(row) for row in result['fit_provenance']} == {key(row) for row in result['fits']}
    for origin in result['fit_provenance']:
        inherited = inherited_count and key(origin) == ('group6', 18, 5701)
        count_ok &= origin['origin'] == ('inherited_completed_fit' if inherited else 'new_fit') and origin['source_trial'] == (recovery['original_failed_trial'] if inherited else str(folder.relative_to(ROOT)))
    checks['exact_distinct_fit_origins_one_plus_five_screen_or_twelve_new_replication'] = bool(count_ok)
    inherited_ok = True
    if inherited_count:
        inherited = recovery['inherited_fit']; inherited_ok &= (inherited['architecture'], inherited['budget'], inherited['seed']) == ('group6', 18, 5701)
        row = next(row for row in result['fits'] if key(row) == ('group6', 18, 5701))
        payload = torch.load(failed / 'models/group6_b18_s5701.pt', map_location='cpu', weights_only=True)
        inherited_ok &= inherited['fitting_metadata'] == payload['settings'] and inherited['fitting_metadata_sha256'] == json_digest(payload['settings'])
        inherited_ok &= all(row[name] == value for name, value in payload['settings'].items()) and len(inherited['artifacts']) == 5
        for artifact in inherited['artifacts']:
            old = ROOT / artifact['file']; subdir = 'gradient_probes' if 'gradient_probes' in old.parts else 'models'; new_path = folder / subdir / old.name
            inherited_ok &= old.is_relative_to(failed) and sha(old) == artifact['sha256'] == sha(new_path)
        origin = next(row for row in result['fit_provenance'] if key(row) == ('group6', 18, 5701))
        inherited_ok &= origin['fitting_metadata_sha256'] == inherited['fitting_metadata_sha256']
    else: inherited_ok &= recovery['inherited_fit'] is None
    checks['inherited_free_initial_final_materialized_and_both_gradient_probe_artifacts_exact'] = bool(inherited_ok)
    return checks, dict(original_failed_trial=recovery['original_failed_trial'], failed_inventory_sha256=json_digest(files),
        inherited_fits=inherited_count, new_fits=new, total_distinct_fits=total, inherited_updates=2000 * inherited_count, new_updates=2000 * new,
        correction='Equal-overlap metadata dictionary merge only; original fit/evaluation/threshold/benchmark/gate functions unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--trial', type=Path); args = parser.parse_args()
    folders = [args.trial if args.trial.is_absolute() else ROOT / args.trial] if args.trial else sorted(path.parent for path in (ROOT / 'trials').glob('*/RESULT.json'))
    assert folders, 'No completed trial'; torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        for folder in folders:
            destination = folder / 'AUDIT.json'
            if destination.exists():
                previous = json.loads(destination.read_text()); assert previous['result_sha256'] == sha(folder / 'RESULT.json')
                if previous['audit_source_sha256'] == sha(__file__): outcome = previous
                else:
                    archive = folder / f"AUDIT_PREVIOUS_{previous['audit_source_sha256'][:12]}.json"; assert not archive.exists(); destination.rename(archive); outcome = audit_trial(folder)
            else: outcome = audit_trial(folder)
            if not destination.exists(): destination.write_text(json.dumps(outcome, ensure_ascii=False, indent=2, default=lambda value: value.item()) + '\n')
            print(json.dumps({key: outcome[key] for key in ['status', 'trial', 'checks_passed', 'checks_total', 'maximum_errors', 'independent_gates', 'replication', 'recovery', 'seconds']}, default=lambda value: value.item()), flush=True)
            assert outcome['status'] == 'PASS_AUDIT'


if __name__ == '__main__': main()
