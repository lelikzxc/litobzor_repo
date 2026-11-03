"""Fresh group/plain-C4 segmentation fits under the unchanged historical loss.

Only TRAIN_VAL and SUBSETS are loaded; no historical weights or fresh tests.
Both plain CNNs train THROUGH their four-view probability pool. Benchmarks
measure CNN routes, not the common CPU geometry classification pipeline.
"""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import time
import traceback

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from model import (FREE_PARAMETERS, MACS, ROUTES, TRAINING_ARMS, GroupCNN,
    aligned_view_logits, architecture_spec, make_model, parameters,
    predict_probability, state_digest, training_details)

ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent / 'sem_architecture_pilot_20261102'
ROUTE_NAMES = [route for arch in TRAINING_ARMS for route, mode in ROUTES[arch]]
CLASS_CACHE = {}
START = None


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


physics = module('groupcnn_unchanged_physics', PILOT / 'physics_adapter.py')
preparation = module('groupcnn_unchanged_class_rule', PILOT / 'prepare.py')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temp.replace(path)


def configuration(path):
    c = json.loads(path.read_text())
    assert c['candidate'] == 'known_C4_group_CNN' and c['training_arms'] == TRAINING_ARMS
    assert c['budgets'] == [18, 27] and c['screen_seeds'] == [5701] and c['replication_seeds'] == [5701, 5721, 5757]
    assert c['subset_seed'] == 5303 and c['steps'] == 2000 and c['warmup_steps'] == 400 and c['batch_size'] == 8
    assert c['learning_rate'] == .0005 and c['weight_decay'] == .0001 and c['physics_weight'] == 1.
    assert c['batch_seed_offset'] == 1000 and c['thresholds'] == [round(.1 + .01 * i, 2) for i in range(81)]
    assert c['secondary_thresholds'] == [.3, .4, .45, .5, .55, .6, .7]
    assert c['benchmark_batch_size'] == 32 and c['benchmark_warmup'] == 10 and c['benchmark_repeats'] == 50
    assert c['primary_gates'] == dict(minimum_mean_F1_gain=.005, minimum_budget_F1_delta=-.0025,
        minimum_mean_IoU_delta=-.002, maximum_budget_ratio_deviation=.03)
    assert c['secondary_gates'] == dict(minimum_mean_F1_gain=-.005, minimum_budget_F1_delta=-.01,
        minimum_mean_IoU_delta=-.002, maximum_MAC_ratio=.30)
    assert c['protocol_sha256'] == sha(ROOT / 'PROTOCOL.md')
    return c


def source_hashes():
    names = ['model.py', 'runner.py', '../sem_architecture_pilot_20261102/physics_adapter.py',
        '../sem_physics_materials/physics_loss_example.py', '../sem_architecture_pilot_20261102/prepare.py',
        '../sem_confirmation_20261028/prepare.py', '../retained_evidence_20261027/SELECTED_SOURCES.zip']
    return {name: sha(ROOT / name) for name in names}


def append_ledger(event):
    with (ROOT / '.ledger.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); path = ROOT / 'LEDGER.json'
        ledger = json.loads(path.read_text()) if path.exists() else dict(version=1, scope='FRESH_INIT_OLD_TRAIN_VAL_GROUPCNN', events=[])
        ledger['events'].append(dict(event, index=len(ledger['events']), UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        save(path, ledger); fcntl.flock(lock, fcntl.LOCK_UN)


def inputs(c, device):
    subsets = json.loads((PILOT / 'SUBSETS.json').read_text())
    assert sha(PILOT / 'TRAIN_VAL.npz') == subsets['train_val_sha256']
    with np.load(PILOT / 'TRAIN_VAL.npz', allow_pickle=False) as z:
        x, labels, roles = [z[key].copy() for key in ['X', 'label', 'role']]
        vi = np.flatnonzero(roles == 'val'); val_masks = z['Y'][vi].astype(np.float32)
        dense_masks = {b: z['Y'][np.array(subsets['dense_indices'][str(b)])].astype(np.float32) for b in c['budgets']}
    assert set(roles) == {'dense', 'weak', 'val'}
    data = {}
    for budget in c['budgets']:
        dense = np.array(subsets['dense_indices'][str(budget)], dtype=np.int64)
        weak = np.flatnonzero(roles == 'weak')
        assert len(dense) == budget and len(weak) == 1152 and len(vi) == 695
        assert np.array_equal(np.bincount(labels[dense], minlength=9), np.full(9, budget // 9))
        make = lambda ids: dict(x=torch.tensor(5 * (x[ids] - .2), device=device),
            label=torch.tensor(labels[ids], device=device), indices=ids.tolist())
        data[budget] = dict(dense=make(dense), weak=make(weak), val=make(vi))
        data[budget]['dense']['mask'] = torch.tensor(dense_masks[budget], device=device)
        data[budget]['val']['mask'] = torch.tensor(val_masks, device=device)
        assert 'mask' not in data[budget]['weak']
    return data, dict(TRAIN_VAL_sha256=sha(PILOT / 'TRAIN_VAL.npz'), SUBSETS_sha256=sha(PILOT / 'SUBSETS.json'),
        indices={str(b): {role: v['indices'] for role, v in values.items()} for b, values in data.items()},
        pretrained_weights_read=False, weak_masks_loaded=False)


def hard_classes(masks):
    masks = np.asarray(masks, bool).reshape(-1, 12, 12)
    keys = [row.tobytes() for row in np.packbits(masks.reshape(len(masks), 144), axis=1)]
    missing = sorted(set(keys) - CLASS_CACHE.keys())
    if missing:
        values = np.array([np.unpackbits(np.frombuffer(key, np.uint8)).reshape(12, 12) for key in missing], bool)
        CLASS_CACHE.update(zip(missing, preparation.class_labels(values).tolist()))
    return np.array([CLASS_CACHE[key] for key in keys], np.int64)


def rank(row): return row['mask_F1'], row['IoU'], -abs(row['threshold'] - .5), -row['threshold']


def score(probability, values, thresholds):
    truth = values['label'].cpu().numpy(); target = values['mask'][:, 0].cpu().numpy().astype(bool)
    rows, predictions, packed = [], [], []
    for threshold in thresholds:
        mask = probability[:, 0] >= threshold; prediction = hard_classes(mask); union = (mask | target).sum((1, 2))
        iou = np.divide((mask & target).sum((1, 2)), union, out=np.ones(len(mask)), where=union > 0)
        rows.append(dict(threshold=threshold, mask_F1=float(f1_score(truth, prediction, labels=range(9), average='macro', zero_division=0)), IoU=float(iou.mean())))
        predictions.append(prediction); packed.append(np.packbits(mask.reshape(len(mask), 144), axis=1))
    return dict(selected=max(rows, key=rank), all_thresholds=rows), dict(predictions=np.array(predictions), packed_masks=np.array(packed))


def fit(c, net, architecture, values, budget, seed, trial, counter):
    net.train(); initial = state_digest(net.state_dict()); started = time.monotonic()
    params = list(net.parameters()); optimizer = torch.optim.AdamW(params, lr=c['learning_rate'], weight_decay=c['weight_decay'])
    rng = np.random.default_rng(seed + c['batch_seed_offset']); batches = hashlib.sha256(); probes = []
    for step in range(1, c['steps'] + 1):
        ai = rng.integers(0, len(values['dense']['x']), c['batch_size']); wi = rng.integers(0, len(values['weak']['x']), c['batch_size'])
        batches.update(ai.astype(np.int64).tobytes()); batches.update(wi.astype(np.int64).tobytes())
        ai, wi = torch.tensor(ai, device='mps'), torch.tensor(wi, device='mps')
        dx, wx = values['dense']['x'][ai], values['weak']['x'][wi]
        optimizer.zero_grad(set_to_none=True); joint = step > c['warmup_steps']
        details = training_details(net, architecture, torch.cat((dx, wx)) if joint else dx)
        dense = details['logits'][:c['batch_size']]
        weak = details['logits'][c['batch_size']:] if joint else None
        if step in [1, c['warmup_steps'] + 1]:
            dense.retain_grad()
            if weak is not None: weak.retain_grad()
        bce = F.binary_cross_entropy_with_logits(dense, values['dense']['mask'][ai]); geometry = None
        if joint:
            labels = values['weak']['label'][wi]
            geometry = physics.geometry_loss(weak, torch.stack((labels // 3, labels % 3), dim=1))
        loss = bce if geometry is None else bce + c['physics_weight'] * geometry
        assert torch.isfinite(loss); loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert any(float(p.grad.abs().sum()) > 0 for p in params)
        if step in [1, c['warmup_steps'] + 1]:
            probe = dict(architecture=architecture, budget=budget, seed=seed, step=step,
                phase='BCE_WARMUP' if not joint else 'BCE_PLUS_OLD_GEOMETRY',
                state_before={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                gradients={k: p.grad.detach().cpu().clone() for k, p in net.named_parameters()},
                dense_indices=ai.cpu(), weak_indices=wi.cpu(), dense_x=dx.cpu(), weak_x=wx.cpu(),
                dense_mask=values['dense']['mask'][ai].cpu(), weak_label=values['weak']['label'][wi].cpu(),
                dense_logits=dense.detach().cpu(), weak_logits=None if weak is None else weak.detach().cpu(),
                dense_logits_gradient=dense.grad.detach().cpu().clone(),
                weak_logits_gradient=None if weak is None else weak.grad.detach().cpu().clone(),
                aligned_logits=None if details['aligned_logits'] is None else details['aligned_logits'].detach().cpu(),
                BCE=float(bce.detach().cpu()), geometry_loss=None if geometry is None else float(geometry.detach().cpu()),
                total_loss=float(loss.detach().cpu()), weak_loss_used=joint, optimizer_step_not_yet_applied=True)
            path = trial / 'gradient_probes' / f'{architecture}_b{budget}_s{seed}_step{step}.pt'; torch.save(probe, path)
            probes.append(dict(file=str(path.relative_to(trial)), sha256=sha(path), step=step))
        optimizer.step(); counter[0] += 1
        if step % 200 == 0:
            progress = dict(stage='FRESH_GROUP_CNN_TRAIN_VAL', trial=trial.name, architecture=architecture, budget=budget, seed=seed,
                step=step, target_steps=2000, completed_updates=counter[0], phase='BCE_WARMUP' if not joint else 'BCE_PLUS_OLD_GEOMETRY',
                BCE=float(bce.detach().cpu()), geometry_loss=None if geometry is None else float(geometry.detach().cpu()),
                total_loss=float(loss.detach().cpu()), seconds=time.monotonic() - START)
            save(trial / 'PROGRESS.json', progress); save(ROOT / 'PROGRESS.json', progress)
            with (trial / 'progress.jsonl').open('a') as h: h.write(json.dumps(progress) + '\n')
            print(json.dumps(progress), flush=True)
    torch.mps.synchronize(); final = state_digest(net.state_dict()); assert final != initial
    return dict(steps=2000, warmup_steps=400, joint_steps=1600, optimizer_updates=2000, fit_count=1,
        fresh_initialization=True, initial_state_sha256=initial, final_state_sha256=final, free_parameters=parameters(net),
        training_route='single_group_equivariant' if architecture == 'group6' else 'four_view_probability_pool',
        batch_sequence_sha256=batches.hexdigest(), gradient_probes=probes, training_seconds=time.monotonic() - started)


@torch.no_grad()
def evaluate(c, network, values, mode):
    network.eval(); parts = [predict_probability(network, x, mode).cpu().numpy() for x in values['x'].split(16)]
    probability = np.concatenate(parts)
    primary, pa = score(probability, values, c['thresholds']); secondary, sa = score(probability, values, c['secondary_thresholds'])
    return dict(selected=primary['selected'], primary_fine=primary, secondary_seven=secondary), dict(probability=probability,
        truth=values['label'].cpu().numpy(), target=values['mask'][:, 0].cpu().numpy().astype(bool),
        thresholds=np.array(c['thresholds']), secondary_thresholds=np.array(c['secondary_thresholds']),
        predictions=pa['predictions'], packed_masks=pa['packed_masks'], secondary_predictions=sa['predictions'], secondary_packed_masks=sa['packed_masks'])


@torch.no_grad()
def route_fixture(c, net, x, mode, free_group=None):
    probability = predict_probability(net, x, mode)
    restored = torch.stack([torch.rot90(predict_probability(net, torch.rot90(x, k, (-2, -1)).contiguous(), mode), -k, (-2, -1)) for k in range(4)])
    error = float((restored - probability[None]).abs().max().cpu()); required = mode != 'identity'
    if required: assert error <= c['trained_probability_tolerance'], error
    fixture = dict(x=x.cpu().numpy(), probability=probability.cpu().numpy(), restored_probability=restored.cpu().numpy())
    technical = dict(C4_probability_max_error=error, C4_equivariance_required=required)
    if mode == 'C4': fixture['aligned_logits'] = aligned_view_logits(net, x).cpu().numpy()
    if free_group is not None:
        original = free_group(x).sigmoid(); parity = float((original - probability).abs().max().cpu())
        assert parity <= c['trained_probability_tolerance']
        features = free_group.forward_features(x).reshape(len(x), 6, 4, 12, 12)
        relative = 0.
        for k in range(4):
            actual = free_group.forward_features(torch.rot90(x, k, (-2, -1)).contiguous()).reshape_as(features)
            expected = torch.rot90(torch.roll(features, k, 2), k, (-2, -1))
            relative = max(relative, float((actual - expected).abs().max().cpu()) / max(1., float(features.abs().max().cpu())))
        assert relative <= c['trained_probability_tolerance']
        fixture.update(free_group_probability=original.cpu().numpy(), group_features=features.cpu().numpy())
        technical.update(deployment_probability_max_error=parity, feature_action_relative_max_error=relative,
                         free_parameters=2671, materialized_coefficients=parameters(net))
    return fixture, technical


@torch.no_grad()
def benchmark(c, networks, x, budget, seed, trial, setup_seconds):
    routes = [(name, networks[arch], mode) for arch in TRAINING_ARMS for name, mode in ROUTES[arch]]
    assert len(x) == c['benchmark_batch_size'] == 32
    for name, net, mode in routes:
        net.eval()
        for _ in range(c['benchmark_warmup']): predict_probability(net, x, mode)
    torch.mps.synchronize(); rng = np.random.default_rng(c['benchmark_seed'] + budget + seed)
    orders = np.array([rng.permutation(len(routes)) for _ in range(c['benchmark_repeats'])]); ns = np.zeros_like(orders, dtype=np.int64)
    for repeat, order in enumerate(orders):
        for index in order:
            _, net, mode = routes[int(index)]; torch.mps.synchronize(); start = time.perf_counter_ns()
            predict_probability(net, x, mode); torch.mps.synchronize(); ns[repeat, index] = time.perf_counter_ns() - start
    path = trial / 'benchmarks' / f'b{budget}_s{seed}.npz'
    np.savez_compressed(path, route_names=np.array(ROUTE_NAMES), randomized_route_order=orders, latency_ns=ns, x=x.cpu().numpy())
    rows = {name: dict(median_seconds=float(np.median(ns[:, i])) / 1e9, p10_seconds=float(np.quantile(ns[:, i], .1)) / 1e9,
        p90_seconds=float(np.quantile(ns[:, i], .9)) / 1e9, images_per_batch=32, MACs_per_image=MACS[name]) for i, name in enumerate(ROUTE_NAMES)}
    return dict(budget=budget, seed=seed, routes=rows, batch_size=32, warmup_batches_per_route=c['benchmark_warmup'],
        warmup_images_per_route=32 * c['benchmark_warmup'], repeats=c['benchmark_repeats'], random_seed=c['benchmark_seed'] + budget + seed,
        raw_file=str(path.relative_to(trial)), raw_sha256=sha(path), materialization_setup_seconds=setup_seconds,
        MPS_sync_before_and_after_every_measurement=True, materialization_excluded=True,
        CNN_batch_calls_per_route=1, plain_C4_internal_batch_multiplier=4,
        CPU_geometry_readout_excluded=True, data_and_host_transfer_excluded=True, end_to_end_measured=False,
        paired_group_over_narrow_C4_median=float(np.median(ns[:, 0] / ns[:, 2])),
        paired_group_over_full_C4_median=float(np.median(ns[:, 0] / ns[:, 4])))


def gates(rows, c):
    means = {b: {route: {metric: float(np.mean([r['selected'][metric] for r in rows if r['budget'] == b and r['route'] == route]))
        for metric in ['mask_F1', 'IoU']} for route in ROUTE_NAMES} for b in c['budgets']}
    def compare(reference, limits):
        f1 = [means[b]['group6_single']['mask_F1'] - means[b][reference]['mask_F1'] for b in c['budgets']]
        iou = float(np.mean([means[b]['group6_single']['IoU'] - means[b][reference]['IoU'] for b in c['budgets']]))
        mean = float(np.mean(f1)); checks = dict(mean_F1_gain=mean >= limits['minimum_mean_F1_gain'],
            budget_F1_guard=min(f1) >= limits['minimum_budget_F1_delta'], mean_IoU_guard=iou >= limits['minimum_mean_IoU_delta'])
        return dict(delta=mean, budget_deltas=f1, IoU_delta=iou, checks=checks)
    primary = compare('plain12_C4', c['primary_gates'])
    parameter_ratio = 2671 / 2749; MAC_ratio = MACS['group6_single'] / MACS['plain12_C4']
    primary['checks'].update(parameter_budget_match=abs(parameter_ratio - 1) <= c['primary_gates']['maximum_budget_ratio_deviation'],
        MAC_budget_match=abs(MAC_ratio - 1) <= c['primary_gates']['maximum_budget_ratio_deviation'])
    primary.update(free_parameter_ratio=parameter_ratio, MAC_ratio=MAC_ratio, passed=all(primary['checks'].values()))
    secondary = compare('plain24_C4', c['secondary_gates']); full_ratio = MACS['group6_single'] / MACS['plain24_C4']
    secondary['checks']['MAC_efficiency_guard'] = full_ratio <= c['secondary_gates']['maximum_MAC_ratio']
    secondary.update(MAC_ratio=full_ratio, passed=all(secondary['checks'].values()))
    return dict(A_matched_quality=primary, B_full_efficiency=secondary,
        ANY_GOAL_PASS=bool(primary['passed'] or secondary['passed']),
        interpretation='Separate reused-VAL development goals; secondary does not replace primary; latency is descriptive'), means


def signature(c): return digest(dict(configuration=c, source_sha256=source_hashes(), TRAIN_VAL_sha256=sha(PILOT / 'TRAIN_VAL.npz')))


def previous_screen(expected):
    found = []
    for path in (ROOT / 'trials').glob('*/RESULT.json'):
        result = json.loads(path.read_text())
        if result['phase'] == 'screen' and result['signature'] == expected and result['status'] == 'COMPLETE_GROUPCNN_DEVELOPMENT': found.append((path, result))
    assert len(found) == 1 and found[0][1]['gates']['ANY_GOAL_PASS']
    audit = json.loads((found[0][0].parent / 'AUDIT.json').read_text()); assert audit['status'] == 'PASS_AUDIT' and all(audit['checks'].values())
    return found[0]


def preflight(c):
    data, lineage = inputs(c, 'cpu'); torch.manual_seed(c['technical_seed']); x = torch.randn(3, 1, 12, 12)
    records = []; group = make_model('group6', 5701); deployed = group.deploy()
    features = group.forward_features(x).reshape(3, 6, 4, 12, 12); output = group(x)
    feature_error = output_error = 0.
    with torch.no_grad():
        for k in range(4):
            rotated = torch.rot90(x, k, (-2, -1)).contiguous()
            actual = group.forward_features(rotated).reshape_as(features)
            expected = torch.rot90(torch.roll(features, k, 2), k, (-2, -1))
            feature_error = max(feature_error, float((actual - expected).abs().max()))
            output_error = max(output_error, float((group(rotated) - torch.rot90(output, k, (-2, -1))).abs().max()))
        deploy_error = float((deployed(x) - output).abs().max())
    assert max(feature_error, output_error, deploy_error) <= c['preflight_tolerance']
    for arch in TRAINING_ARMS:
        net = make_model(arch, 5701); before = state_digest(net.state_dict())
        values = data[18]; dx, wx = values['dense']['x'][:2], values['weak']['x'][:2]
        details = training_details(net, arch, torch.cat((dx, wx))); z = details['logits']
        bce = F.binary_cross_entropy_with_logits(z[:2], values['dense']['mask'][:2]); y = values['weak']['label'][:2]
        geometry = physics.geometry_loss(z[2:], torch.stack((y // 3, y % 3), 1)); loss = bce + geometry; loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        assert all(float(p.grad.abs().sum()) > 0 for p in net.parameters())
        if arch != 'group6': assert details['aligned_logits'].shape[0] == 4
        assert float((z.sigmoid() - details['probability']).abs().max().detach()) < 3e-7
        assert state_digest(net.state_dict()) == before
        records.append(dict(architecture=arch, parameters=parameters(net), fresh_initial_state_sha256=before,
            all_named_gradients_nonzero=True, old_BCE_geometry_finite=True, training_through_C4_pool=arch != 'group6'))
    result = dict(status='PASS_PREFLIGHT', rows=records, feature_group_action_max_error=feature_error,
        scalar_C4_logit_max_error=output_error, deployment_logit_max_error=deploy_error,
        free_group_parameters=2671, materialized_coefficients=parameters(deployed), no_pretrained_weights=True,
        data_lineage=lineage, architecture=architecture_spec(), source_sha256=source_hashes(), optimizer_updates=0)
    save(ROOT / 'PREFLIGHT.json', result)
    print(json.dumps({key: value for key, value in result.items() if key not in ['data_lineage', 'architecture']}), flush=True)


def run(c, phase, config_path):
    global START
    START = time.monotonic(); assert torch.backends.mps.is_available() and os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] == '0'
    seeds = c['screen_seeds'] if phase == 'screen' else [s for s in c['replication_seeds'] if s not in c['screen_seeds']]
    expected = signature(c); inherited = previous_screen(expected) if phase == 'replicate' else None
    data, lineage = inputs(c, 'mps'); trial_id = f'groupcnn_{phase}_s{"-".join(map(str,seeds))}_{digest(source_hashes())[:10]}'
    trial = ROOT / 'trials' / trial_id; assert not trial.exists(), 'Immutable trial exists; no overwrite or implicit resume'
    for folder in [trial, trial / 'models', trial / 'val_probs', trial / 'fixtures', trial / 'source', trial / 'gradient_probes', trial / 'benchmarks']:
        folder.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes(); save(trial / 'CONFIG_SNAPSHOT.json', c); shutil.copy2(ROOT / 'PROTOCOL.md', trial / 'PROTOCOL.md')
    for name in ['model.py', 'runner.py']: shutil.copy2(ROOT / name, trial / 'source' / name)
    frozen = dict(candidate=c['candidate'], phase=phase, seeds=seeds, signature=expected, source_sha256=hashes, lineage=lineage,
        architecture=architecture_spec(), protocol_sha256=sha(ROOT / 'PROTOCOL.md'), config_sha256=sha(config_path),
        fitted_models=len(seeds) * 6, total_updates=len(seeds) * 12000, evaluation_routes=len(seeds) * 10,
        initialization='fresh torchseed=seed; CPUparameterinitialization; no historicalcheckpoint read',
        objective='400denseBCE+1600denseBCEplusoldgeometry1; allplainstrainthroughC4pool', fresh_test_access=False)
    save(trial / 'FROZEN.json', frozen); append_ledger(dict(status='RUNNING', trial=trial_id, phase=phase, signature=expected))
    rows, fits, benches, checkpoints, counter = [], [], [], {}, [0]
    try:
        for budget in c['budgets']:
            for seed in seeds:
                networks = {}; setup_seconds = 0.; fitted = []
                for arch in TRAINING_ARMS:
                    net = make_model(arch, seed).to('mps'); assert parameters(net) == FREE_PARAMETERS[arch]
                    initial_path = trial / 'models' / f'{arch}_b{budget}_s{seed}_initial.pt'
                    torch.save(dict(model={k: v.detach().cpu() for k, v in net.state_dict().items()}, seed=seed, architecture=arch), initial_path)
                    checkpoints[initial_path.name] = sha(initial_path)
                    fitting = fit(c, net, arch, data[budget], budget, seed, trial, counter)
                    final_path = trial / 'models' / f'{arch}_b{budget}_s{seed}.pt'
                    torch.save(dict(model={k: v.detach().cpu() for k, v in net.state_dict().items()}, settings=fitting,
                        architecture=arch, budget=budget, seed=seed), final_path); checkpoints[final_path.name] = sha(final_path)
                    fit_row = dict(architecture=arch, budget=budget, seed=seed, **fitting, model_file=str(final_path.relative_to(trial)),
                        model_sha256=sha(final_path), initial_model_file=str(initial_path.relative_to(trial)), initial_model_sha256=sha(initial_path))
                    fitted.append(fit_row); fits.append(fit_row); net.eval()
                    if arch == 'group6':
                        torch.mps.synchronize(); setup = time.monotonic(); deployment = net.deploy(); torch.mps.synchronize()
                        setup_seconds = time.monotonic() - setup; networks[arch] = deployment
                        dp = trial / 'models' / f'group6_b{budget}_s{seed}_deployed.pt'
                        torch.save(dict(model={k: v.detach().cpu() for k, v in deployment.state_dict().items()},
                            free_model_sha256=sha(final_path), free_parameters=2671, materialized_coefficients=parameters(deployment)), dp)
                        checkpoints[dp.name] = sha(dp)
                    else: networks[arch] = net
                    for route, mode in ROUTES[arch]:
                        network = networks[arch]
                        evaluation, arrays = evaluate(c, network, data[budget]['val'], mode)
                        fixture, technical = route_fixture(c, network, data[budget]['val']['x'][:16], mode, net if arch == 'group6' else None)
                        row = dict(architecture=arch, route=route, budget=budget, seed=seed, **evaluation, **technical,
                            free_parameters=FREE_PARAMETERS[arch], materialized_coefficients=parameters(network), MACs_per_image=MACS[route],
                            CNN_passes_per_image=4 if mode == 'C4' else 1, CNN_batch_calls=1,
                            internal_batch_multiplier=4 if mode == 'C4' else 1,
                            model_file=fit_row['model_file'], model_sha256=fit_row['model_sha256'])
                        name = f'{route}_b{budget}_s{seed}'; np.savez_compressed(trial / 'val_probs' / f'{name}.npz', **arrays)
                        np.savez_compressed(trial / 'fixtures' / f'{name}.npz', **fixture); rows.append(row)
                        save(trial / 'PARTIAL_RESULTS.json', rows)
                    save(trial / 'FIT_RESULTS.json', fits); save(trial / 'CHECKPOINT_HASHES.json', checkpoints)
                assert len({row['batch_sequence_sha256'] for row in fitted}) == 1
                bench = benchmark(c, networks, data[budget]['val']['x'][:32], budget, seed, trial, setup_seconds)
                benches.append(bench); save(trial / 'BENCHMARK_RESULTS.json', benches); del networks
        assert counter[0] == len(seeds) * 12000 and len(fits) == len(seeds) * 6 and len(rows) == len(seeds) * 10
        assert all(sha(ROOT / name) == value for name, value in hashes.items())
        assert sha(config_path) == frozen['config_sha256'] and sha(ROOT / 'PROTOCOL.md') == frozen['protocol_sha256']
        outcome, means = gates(rows, c)
        result = dict(status='COMPLETE_GROUPCNN_DEVELOPMENT', candidate=c['candidate'], phase=phase, seeds=seeds, signature=expected,
            rows=rows, fits=fits, benchmarks=benches, gates=outcome, means=means, fitted_models=len(fits), updates=counter[0],
            checkpoint_hashes=checkpoints, frozen_sha256=sha(trial / 'FROZEN.json'), seconds=time.monotonic() - START,
            scope='FreshknownC4backbone comparison, separatequality/efficiencygoals onreusedVAL; nonewlossorTEST')
        save(trial / 'RESULT.json', result); aggregate = None
        if inherited is not None:
            old_path, old = inherited; all_rows = old['rows'] + rows; ag, am = gates(all_rows, c)
            robust_primary = bool(ag['A_matched_quality']['passed'] and outcome['A_matched_quality']['passed'])
            robust_secondary = bool(ag['B_full_efficiency']['passed'] and outcome['B_full_efficiency']['passed'])
            aggregate = dict(status='COMPLETE_REUSED_VAL_GROUPCNN_REPLICATION', seeds=c['replication_seeds'], rows=all_rows,
                fits=old['fits'] + fits, benchmarks=old['benchmarks'] + benches, gates=ag, means=am, new_seed_gates=outcome,
                robust_primary_pass=robust_primary, robust_secondary_pass=robust_secondary,
                robust_ANY_GOAL_PASS=bool(robust_primary or robust_secondary),
                source_trials=[str(old_path.parent.relative_to(ROOT)), str(trial.relative_to(ROOT))])
            save(trial / 'AGGREGATE_RESULT.json', aggregate)
        summary = dict(status=result['status'], trial=str(trial.relative_to(ROOT)), phase=phase, result_sha256=sha(trial / 'RESULT.json'),
            gates=aggregate['gates'] if aggregate else outcome, means=aggregate['means'] if aggregate else means)
        append_ledger(summary); save(ROOT / 'LATEST_RESULT.json', summary); print(json.dumps(summary), flush=True)
    except BaseException as error:
        failure = dict(status='FAILED', trial=trial_id, phase=phase, error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
        save(trial / 'FAILURE.json', failure); append_ledger(failure); raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['screen', 'replicate'], default='screen')
    parser.add_argument('--config', type=Path, default=ROOT / 'CONFIG.json'); parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(); c = configuration(args.config); torch.set_num_threads(2)
    if args.preflight: preflight(c)
    else: run(c, args.phase, args.config)


if __name__ == '__main__':
    with threadpool_limits(limits=2): main()
