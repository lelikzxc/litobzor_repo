"""Metadata-only, append-only stage12 entrypoint for screen and replication.

The first screen preserves and reuses the completed group6/b18 fit. Scientific
model, sampler, optimizer, losses, calibration, benchmark and gates are called
unchanged from the original frozen runner. Row metadata is merged explicitly
with equality checks, avoiding duplicate-key keyword expansion.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
FAILED = ROOT / 'trials/groupcnn_screen_s5701_a571c57fcf'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


previous_model = sys.modules.get('model')
sys.modules['model'] = module('group_recovery_frozen_model', ROOT / 'model.py')
try: r = module('group_recovery_frozen_runner', ROOT / 'runner.py')
finally:
    if previous_model is None: del sys.modules['model']
    else: sys.modules['model'] = previous_model


def read(path): return json.loads(Path(path).read_text())


def inventory(folder):
    return {str(path.relative_to(ROOT)): dict(sha256=r.sha(path), bytes=path.stat().st_size)
            for path in sorted(folder.rglob('*')) if path.is_file()}


def merge_checked(*parts):
    result = {}
    for part in parts:
        for key, value in part.items():
            if key in result: assert result[key] == value, f'Conflicting metadata field: {key}'
            result[key] = value
    return result


def route_row(arch, route, mode, budget, seed, evaluation, technical, network, fitted):
    return merge_checked(dict(architecture=arch, route=route, budget=budget, seed=seed), evaluation, technical,
        dict(free_parameters=r.FREE_PARAMETERS[arch], materialized_coefficients=r.parameters(network),
            MACs_per_image=r.MACS[route], CNN_passes_per_image=4 if mode == 'C4' else 1,
            CNN_batch_calls=1, internal_batch_multiplier=4 if mode == 'C4' else 1,
            model_file=fitted['model_file'], model_sha256=fitted['model_sha256']))


def validate_failed(c):
    frozen, failure = read(FAILED / 'FROZEN.json'), read(FAILED / 'FAILURE.json')
    assert failure['status'] == 'FAILED' and failure['error_type'] == 'TypeError'
    assert "multiple values for keyword argument 'free_parameters'" in failure['error']
    assert frozen['signature'] == r.signature(c) and frozen['source_sha256'] == r.source_hashes()
    assert not (FAILED / 'RESULT.json').exists() and read(FAILED / 'CONFIG_SNAPSHOT.json') == c
    assert r.sha(FAILED / 'PROTOCOL.md') == r.sha(ROOT / 'PROTOCOL.md')
    for name, expected in frozen['source_sha256'].items(): assert r.sha(ROOT / name) == expected
    paths = [FAILED / 'models' / f'group6_b18_s5701{ending}.pt' for ending in ['', '_initial', '_deployed']]
    assert {path.name for path in (FAILED / 'models').glob('*.pt')} == {path.name for path in paths}
    payload = torch.load(paths[0], map_location='cpu', weights_only=True); fitting = payload['settings']
    initial = torch.load(paths[1], map_location='cpu', weights_only=True)
    deployed = torch.load(paths[2], map_location='cpu', weights_only=True)
    assert fitting['steps'] == fitting['optimizer_updates'] == 2000 and fitting['fit_count'] == 1
    assert fitting['warmup_steps'] == 400 and fitting['joint_steps'] == 1600
    assert fitting['initial_state_sha256'] == r.state_digest(initial['model'])
    assert fitting['final_state_sha256'] == r.state_digest(payload['model'])
    assert deployed['free_model_sha256'] == r.sha(paths[0]) and deployed['free_parameters'] == 2671 and deployed['materialized_coefficients'] == 10681
    for probe in fitting['gradient_probes']:
        path = FAILED / probe['file']; assert r.sha(path) == probe['sha256']; paths.append(path)
    inherited = dict(architecture='group6', budget=18, seed=5701, fitting_metadata=fitting,
        fitting_metadata_sha256=r.digest(fitting),
        artifacts=[dict(file=str(path.relative_to(ROOT)), sha256=r.sha(path)) for path in paths])
    return frozen, inherited, inventory(FAILED), payload, initial, deployed


def preflight(c):
    frozen, inherited, files, payload, initial, deployed = validate_failed(c)
    torch.manual_seed(c['technical_seed']); x = torch.randn(3, 1, 12, 12)
    group = r.make_model('group6', 5701); group.load_state_dict(payload['model']); group.eval()
    materialized = group.deploy(); assert r.state_digest(materialized.state_dict()) == r.state_digest(deployed['model'])
    networks = {'group6': materialized,
        'plain12_C4pooled': r.make_model('plain12_C4pooled', 5701).eval(),
        'plain24_C4pooled': r.make_model('plain24_C4pooled', 5701).eval()}
    evidence = []
    for arch in r.TRAINING_ARMS:
        for route, mode in r.ROUTES[arch]:
            _, technical = r.route_fixture(c, networks[arch], x, mode, group if arch == 'group6' else None)
            dummy_evaluation = dict(selected=dict(threshold=.5, mask_F1=0., IoU=0.),
                primary_fine=dict(selected=dict(threshold=.5, mask_F1=0., IoU=0.), all_thresholds=[]),
                secondary_seven=dict(selected=dict(threshold=.5, mask_F1=0., IoU=0.), all_thresholds=[]))
            row = route_row(arch, route, mode, 18, 5701, dummy_evaluation, technical, networks[arch],
                dict(model_file='models/probe.pt', model_sha256='technical_placeholder'))
            encoded = json.dumps(row); assert json.loads(encoded) == row
            evidence.append(dict(route=route, row_constructs=True, JSON_serializes=True,
                overlapping_fields_checked=[key for key in ['free_parameters', 'materialized_coefficients'] if key in technical]))
    rejected = False
    try: merge_checked(dict(free_parameters=2671), dict(free_parameters=1))
    except AssertionError: rejected = True
    assert rejected and len(evidence) == 5
    result = dict(status='PASS_METADATA_RECOVERY_PREFLIGHT', row_construction_variants=evidence,
        conflicting_overlap_rejected=True, inherited_fit_valid=True, inherited_fit_count=1, inherited_updates=2000,
        actual_new_screen_fits_planned=5, actual_new_screen_updates_planned=10000,
        failed_inventory_sha256=r.digest(files), original_source_sha256=frozen['source_sha256'],
        executed_recovery_sha256=r.sha(Path(__file__)), no_optimizer_steps=True,
        same_entrypoint_supports_replication=True, inherited=inherited)
    r.save(ROOT / 'RECOVERY_PREFLIGHT.json', result); print(json.dumps(result), flush=True)


def run(c, phase):
    assert torch.backends.mps.is_available() and r.os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] == '0'
    started = time.monotonic(); expected = r.signature(c)
    previous = r.previous_screen(expected) if phase == 'replicate' else None
    seeds = c['screen_seeds'] if phase == 'screen' else [s for s in c['replication_seeds'] if s not in c['screen_seeds']]
    old_frozen, inherited, failed_files, old_payload, old_initial, old_deploy = validate_failed(c)
    inherited_count = 1 if phase == 'screen' else 0
    data, lineage = r.inputs(c, 'mps')
    trial_id = f'groupcnn_{phase}_s{"-".join(map(str,seeds))}_metadata_recovery_{r.sha(Path(__file__))[:10]}'
    trial = ROOT / 'trials' / trial_id; assert not trial.exists(), 'Immutable recovery trial; no overwrite or implicit resume'
    for folder in [trial, trial / 'models', trial / 'val_probs', trial / 'fixtures', trial / 'source', trial / 'gradient_probes', trial / 'benchmarks']:
        folder.mkdir(parents=True, exist_ok=False)
    for name in ['model.py', 'runner.py', 'recovery.py']: shutil.copy2(ROOT / name, trial / 'source' / name)
    shutil.copy2(ROOT / 'PROTOCOL.md', trial / 'PROTOCOL.md'); r.save(trial / 'CONFIG_SNAPSHOT.json', c)
    hashes = dict(r.source_hashes(), **{'recovery.py': r.sha(Path(__file__))})
    protected = {name: r.sha(ROOT / name) for name in ['model.py', 'runner.py', 'CONFIG.json', 'PROTOCOL.md']}
    recovery = dict(change='Metadata row merge only; equal overlapping values asserted',
        original_failed_trial=str(FAILED.relative_to(ROOT)), failed_inventory=failed_files, failed_inventory_sha256=r.digest(failed_files),
        inherited_fit=inherited if inherited_count else None, inherited_fits=inherited_count, inherited_updates=2000 * inherited_count,
        actual_new_fits=6 * len(seeds) - inherited_count, actual_new_updates=12000 * len(seeds) - 2000 * inherited_count,
        recovery_source_sha256=r.sha(Path(__file__)), preserved_source_config_protocol_sha256=protected,
        scientific_signature_unchanged=expected, no_model_data_loss_gate_changes=True)
    frozen = dict(candidate=c['candidate'], phase=phase, seeds=seeds, signature=expected, source_sha256=hashes, lineage=lineage,
        architecture=r.architecture_spec(), protocol_sha256=r.sha(ROOT / 'PROTOCOL.md'), config_sha256=r.sha(ROOT / 'CONFIG.json'),
        fitted_models=6 * len(seeds), total_updates=12000 * len(seeds), evaluation_routes=10 * len(seeds),
        initialization=old_frozen['initialization'], objective=old_frozen['objective'], fresh_test_access=False, recovery=recovery)
    if previous is not None:
        frozen['audited_screen_file'] = str(previous[0].relative_to(ROOT)); frozen['audited_screen_sha256'] = r.sha(previous[0])
    r.save(trial / 'FROZEN.json', frozen); r.save(trial / 'RECOVERY_MANIFEST.json', recovery)
    r.append_ledger(dict(status='RUNNING_METADATA_RECOVERY', phase=phase, trial=trial_id, signature=expected,
        inherited_fits=inherited_count, actual_new_fits_planned=recovery['actual_new_fits']))
    rows, fits, benches, checkpoints, origins = [], [], [], {}, []
    counter = [2000 * inherited_count]; r.START = started
    try:
        for budget in c['budgets']:
            for seed in seeds:
                networks = {}; setup_seconds = 0.; fitted = []
                for arch in r.TRAINING_ARMS:
                    inherit_this = phase == 'screen' and (arch, budget, seed) == ('group6', 18, 5701)
                    net = r.make_model(arch, seed).to('mps')
                    initial_path = trial / 'models' / f'{arch}_b{budget}_s{seed}_initial.pt'
                    final_path = trial / 'models' / f'{arch}_b{budget}_s{seed}.pt'
                    if inherit_this:
                        net.load_state_dict(old_payload['model']); fitting = dict(inherited['fitting_metadata'])
                        for artifact in inherited['artifacts']:
                            source = ROOT / artifact['file']; subdir = 'gradient_probes' if 'gradient_probes' in source.parts else 'models'
                            target = trial / subdir / source.name; shutil.copy2(source, target); assert r.sha(target) == artifact['sha256']
                        origin = dict(architecture=arch, budget=budget, seed=seed, origin='inherited_completed_fit',
                            source_trial=str(FAILED.relative_to(ROOT)), fitting_metadata_sha256=inherited['fitting_metadata_sha256'])
                    else:
                        torch.save(dict(model={k: v.detach().cpu() for k, v in net.state_dict().items()}, seed=seed, architecture=arch), initial_path)
                        fitting = r.fit(c, net, arch, data[budget], budget, seed, trial, counter)
                        torch.save(dict(model={k: v.detach().cpu() for k, v in net.state_dict().items()}, settings=fitting,
                            architecture=arch, budget=budget, seed=seed), final_path)
                        origin = dict(architecture=arch, budget=budget, seed=seed, origin='new_fit', source_trial=str(trial.relative_to(ROOT)))
                    checkpoints[initial_path.name] = r.sha(initial_path); checkpoints[final_path.name] = r.sha(final_path)
                    fit_row = merge_checked(dict(architecture=arch, budget=budget, seed=seed), fitting,
                        dict(model_file=str(final_path.relative_to(trial)), model_sha256=r.sha(final_path),
                            initial_model_file=str(initial_path.relative_to(trial)), initial_model_sha256=r.sha(initial_path)))
                    fitted.append(fit_row); fits.append(fit_row); origins.append(origin); net.eval()
                    if arch == 'group6':
                        torch.mps.synchronize(); start_setup = time.monotonic(); deployment = net.deploy(); torch.mps.synchronize()
                        setup_seconds = time.monotonic() - start_setup; networks[arch] = deployment
                        dp = trial / 'models' / f'group6_b{budget}_s{seed}_deployed.pt'
                        if inherit_this:
                            assert r.state_digest(deployment.state_dict()) == r.state_digest(old_deploy['model'])
                            assert r.sha(dp) == next(v['sha256'] for v in inherited['artifacts'] if v['file'].endswith('_deployed.pt'))
                        else:
                            torch.save(dict(model={k: v.detach().cpu() for k, v in deployment.state_dict().items()},
                                free_model_sha256=r.sha(final_path), free_parameters=2671, materialized_coefficients=r.parameters(deployment)), dp)
                        checkpoints[dp.name] = r.sha(dp)
                    else: networks[arch] = net
                    for route, mode in r.ROUTES[arch]:
                        network = networks[arch]; evaluation, arrays = r.evaluate(c, network, data[budget]['val'], mode)
                        fixture, technical = r.route_fixture(c, network, data[budget]['val']['x'][:16], mode, net if arch == 'group6' else None)
                        row = route_row(arch, route, mode, budget, seed, evaluation, technical, network, fit_row)
                        name = f'{route}_b{budget}_s{seed}'; np.savez_compressed(trial / 'val_probs' / f'{name}.npz', **arrays)
                        np.savez_compressed(trial / 'fixtures' / f'{name}.npz', **fixture); rows.append(row)
                        r.save(trial / 'PARTIAL_RESULTS.json', rows)
                    r.save(trial / 'FIT_RESULTS.json', fits); r.save(trial / 'CHECKPOINT_HASHES.json', checkpoints)
                    r.save(trial / 'FIT_PROVENANCE.json', origins)
                assert len({row['batch_sequence_sha256'] for row in fitted}) == 1
                benches.append(r.benchmark(c, networks, data[budget]['val']['x'][:32], budget, seed, trial, setup_seconds))
                r.save(trial / 'BENCHMARK_RESULTS.json', benches); del networks
        assert counter[0] == len(seeds) * 12000 and len(fits) == len(seeds) * 6 and len(rows) == len(seeds) * 10
        assert inventory(FAILED) == failed_files
        assert all(r.sha(ROOT / name) == value for name, value in protected.items())
        assert all(r.sha(ROOT / name) == value for name, value in hashes.items())
        if inherited_count:
            for artifact in inherited['artifacts']:
                source = ROOT / artifact['file']; subdir = 'gradient_probes' if 'gradient_probes' in source.parts else 'models'
                assert r.sha(trial / subdir / source.name) == artifact['sha256']
        outcome, means = r.gates(rows, c)
        proof = dict(status='PASS_METADATA_ONLY_RECOVERY_PROOF', failed_inventory_unchanged=True,
            original_model_runner_config_protocol_unchanged=True, fit_provenance=origins,
            inherited_fits=inherited_count, actual_new_fits=len(fits) - inherited_count,
            inherited_updates=2000 * inherited_count, actual_new_updates=counter[0] - 2000 * inherited_count,
            total_distinct_fits=len(fits), total_unique_updates=counter[0], scientific_functions_reused_unchanged=True,
            metadata_merge_equal_overlap_asserted=True, no_inherited_fit_retrained=True)
        r.save(trial / 'RECOVERY_PROOF.json', proof)
        result = dict(status='COMPLETE_GROUPCNN_DEVELOPMENT', candidate=c['candidate'], phase=phase, seeds=seeds, signature=expected,
            rows=rows, fits=fits, benchmarks=benches, gates=outcome, means=means, fitted_models=len(fits), updates=counter[0],
            actual_new_fits=len(fits) - inherited_count, actual_new_updates=counter[0] - 2000 * inherited_count,
            inherited_fits=inherited_count, inherited_updates=2000 * inherited_count, total_distinct_fits=len(fits), fit_provenance=origins,
            checkpoint_hashes=checkpoints, frozen_sha256=r.sha(trial / 'FROZEN.json'), seconds=time.monotonic() - started,
            recovery=recovery, recovery_proof_sha256=r.sha(trial / 'RECOVERY_PROOF.json'),
            scope='Metadata-only recovery; original scientific model/data/loss/gates unchanged; no fresh TEST')
        r.save(trial / 'RESULT.json', result); aggregate = None
        if previous is not None:
            old_path, old = previous; all_rows = old['rows'] + rows; ag, am = r.gates(all_rows, c)
            primary = bool(ag['A_matched_quality']['passed'] and outcome['A_matched_quality']['passed'])
            secondary = bool(ag['B_full_efficiency']['passed'] and outcome['B_full_efficiency']['passed'])
            aggregate = dict(status='COMPLETE_REUSED_VAL_GROUPCNN_REPLICATION', seeds=c['replication_seeds'], rows=all_rows,
                fits=old['fits'] + fits, benchmarks=old['benchmarks'] + benches, gates=ag, means=am, new_seed_gates=outcome,
                robust_primary_pass=primary, robust_secondary_pass=secondary, robust_ANY_GOAL_PASS=bool(primary or secondary),
                source_trials=[str(old_path.parent.relative_to(ROOT)), str(trial.relative_to(ROOT))])
            r.save(trial / 'AGGREGATE_RESULT.json', aggregate)
        summary = dict(status=result['status'], trial=str(trial.relative_to(ROOT)), phase=phase, result_sha256=r.sha(trial / 'RESULT.json'),
            gates=aggregate['gates'] if aggregate else outcome, means=aggregate['means'] if aggregate else means,
            actual_new_fits=result['actual_new_fits'], inherited_fits=inherited_count, total_distinct_fits=len(fits))
        r.append_ledger(summary); r.save(ROOT / 'LATEST_RESULT.json', summary); print(json.dumps(summary), flush=True)
    except BaseException as error:
        failure = dict(status='FAILED_METADATA_RECOVERY', trial=trial_id, phase=phase, error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
        r.save(trial / 'FAILURE.json', failure); r.append_ledger(failure); raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['screen', 'replicate'], default='screen')
    parser.add_argument('--preflight', action='store_true'); args = parser.parse_args()
    c = r.configuration(ROOT / 'CONFIG.json'); torch.set_num_threads(2)
    if args.preflight: preflight(c)
    else: run(c, args.phase)


if __name__ == '__main__':
    with threadpool_limits(limits=2): main()
