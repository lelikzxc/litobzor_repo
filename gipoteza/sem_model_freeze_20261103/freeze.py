"""Freeze and verify the accepted model family without loading data or torch.

The manifest is deterministic. Existing experiment files are only read as bytes;
the only writes are FROZEN_MODEL.json and VERIFY.json in this new directory.
"""
from pathlib import Path
import argparse
import hashlib
import json

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEVELOPMENT = REPO / 'gipoteza/sem_arch_groupcnn_20261102'
CONFIRMATION = REPO / 'gipoteza/sem_arch_groupcnn_confirmation_20261102'
SCREEN = DEVELOPMENT / 'trials/groupcnn_screen_s5701_metadata_recovery_27b9055b42'
REPLICATION = DEVELOPMENT / 'trials/groupcnn_replicate_s5721-5757_metadata_recovery_27b9055b42'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(REPO))


def build():
    files = {}

    def include(path, purpose, expected=None):
        path = Path(path).resolve()
        assert path.is_file(), path
        key = relative(path)
        digest = sha(path)
        assert expected is None or digest == expected, (key, digest, expected)
        row = files.setdefault(key, dict(sha256=digest, bytes=path.stat().st_size, purposes=[]))
        assert row['sha256'] == digest
        if purpose not in row['purposes']:
            row['purposes'].append(purpose)
        return key

    # Preserve the complete scientific input set underlying both audited reports.
    # Local stage-relative names have precedence over repository-relative names.
    for stage in [DEVELOPMENT, CONFIRMATION]:
        manifest = stage / 'REPORT_MANIFEST.json'
        data = read(manifest)
        include(manifest, 'historical_report_manifest')
        for name, expected in data['scientific_inputs_sha256'].items():
            options = [stage / name, REPO / name]
            candidates = [path for path in options if path.is_file() and sha(path) == expected]
            assert candidates, (name, expected)
            include(candidates[0], 'audited_scientific_input', expected)
        for name, expected in data['outputs_sha256'].items():
            options = [stage / name, REPO / name]
            candidates = [path for path in options if path.is_file() and sha(path) == expected]
            assert candidates, (name, expected)
            include(candidates[0], 'historical_report_snapshot', expected)

    for name in ['model.py', 'runner.py', 'recovery.py', 'CONFIG.json', 'PROTOCOL.md',
                 'HYPOTHESIS.md', 'PREFLIGHT.json', 'RECOVERY_PREFLIGHT.json', 'audit.py']:
        include(DEVELOPMENT / name, 'development_source_or_specification')
    for name in ['confirmation_common.py', 'confirmation_stats.py', 'selection.py', 'prepare.py',
                 'evaluate.py', 'audit.py', 'CONFIG.json', 'PROTOCOL.md', 'SELECTION.json',
                 'MODEL_LOCK.json', 'SOURCE_PLAN.json', 'EVALUATION_FROZEN.json', 'RESULT.json',
                 'AUDIT.json', 'DATA_AUDIT.json', 'DATA_MANIFEST.json', 'EVAL_INDEX.json',
                 'EVAL.npz', 'TEST.npz', 'PREDICTIONS.npz', 'BOOTSTRAP.npz']:
        include(CONFIRMATION / name, 'locked_confirmation_source_data_or_result')
    for name in ['TRAIN_VAL.npz', 'SUBSETS.json']:
        include(DEVELOPMENT.parent / 'sem_architecture_pilot_20261102' / name, 'development_training_data')

    aggregate = read(REPLICATION / 'AGGREGATE_RESULT.json')
    include(REPLICATION / 'AGGREGATE_RESULT.json', 'development_aggregate_result')
    assert len(aggregate['fits']) == 18
    model_family = []
    final_count = initial_count = deployment_count = 0
    for fit in aggregate['fits']:
        trial = SCREEN if fit['seed'] == 5701 else REPLICATION
        assert fit['seed'] in [5701, 5721, 5757]
        assert fit['budget'] in [18, 27]
        assert fit['architecture'] in ['group6', 'plain12_C4pooled', 'plain24_C4pooled']
        assert fit['fit_count'] == 1 and fit['optimizer_updates'] == 2000
        final = trial / fit['model_file']
        initial = trial / fit['initial_model_file']
        include(final, 'final_learned_checkpoint', fit['model_sha256'])
        include(initial, 'initial_checkpoint', fit['initial_model_sha256'])
        final_count += 1
        initial_count += 1
        if fit['architecture'] == 'group6':
            deployed = final.with_name(final.stem + '_deployed.pt')
            include(deployed, 'materialized_group_checkpoint')
            deployment_count += 1
            model_family.append(dict(architecture='group6', budget=fit['budget'], seed=fit['seed'],
                final_checkpoint=relative(final), final_checkpoint_sha256=sha(final),
                deployed_checkpoint=relative(deployed), deployed_checkpoint_sha256=sha(deployed),
                initial_checkpoint=relative(initial),
                independent_learned_parameters=2671, materialized_coefficients=10681))
    assert (final_count, initial_count, deployment_count) == (18, 18, 6)
    assert len(model_family) == 6
    lock = read(CONFIRMATION / 'MODEL_LOCK.json')
    for row in lock['rows']:
        include(REPO / row['free_model_file'], 'confirmation_locked_checkpoint', row['free_model_sha256'])
        include(REPO / row['prediction_model_file'], 'confirmation_locked_prediction_checkpoint', row['prediction_model_sha256'])
        include(REPO / row['source_VAL_file'], 'confirmation_locked_threshold_source', row['source_VAL_sha256'])
    assert lock['source_model_count'] == 18 and lock['deployment_artifact_count'] == 6
    result = read(CONFIRMATION / 'RESULT.json')
    assert result['confirmed_by_goal'] == {'A_matched_quality': True, 'B_full_efficiency': False}
    goals = result['analyses']['primary_fine']['goals']
    include(Path(__file__), 'freeze_tool_source')
    example = next(row for row in model_family if row['budget'] == 27 and row['seed'] == 5701)
    return dict(version=1, status='FROZEN_ACCEPTED_MODEL_FAMILY', freeze_date='2026-11-03',
        date_timezone='Europe/Moscow', model_family='known_C4_GroupCNN_6_base_features',
        selection_policy='Freeze all six existing budget/seed group models; no best-seed selection; '
                        'seed5701/budget27 is an illustrative checkpoint only.',
        input_transform='float32 5*(X12-0.2), shape [N,1,12,12]',
        target='M2 metal mask followed by nine SVG-mask-derived geometric states of one IC',
        task_scope='SEM metallization segmentation plus fixed mask-geometric classification; '
                   'not expert production-defect labels or measured electrical response',
        training=dict(budgets=[18, 27], seeds=[5701, 5721, 5757], weak_class_examples=1152,
            updates_per_fit=2000, warmup_dense_BCE_updates=400,
            joint_dense_BCE_and_legacy_geometry_updates=1600,
            geometry_weight=1.0, optimizer='AdamW', learning_rate=0.0005,
            weight_decay=0.0001, dense_batch=8, weak_batch=8),
        architecture=dict(base_features=6, orientations=4, independent_learned_parameters=2671,
            materialized_coefficients=10681, MACs_per_image=1527552,
            layers=['lifting 3x3, GELU', 'C4 group 3x3 dilation2, GELU',
                    'C4 group 3x3 dilation4, GELU', 'orientation mean', '1x1 head']),
        frozen_model_family=sorted(model_family, key=lambda row: (row['budget'], row['seed'])),
        illustrative_checkpoint=example, learned_final_checkpoints=final_count,
        initial_checkpoints=initial_count, materialized_checkpoints=deployment_count,
        confirmation=dict(examples=result['test_examples'], frames=result['test_frames'],
            new_training_fits=result['fit_count'], confirmed_by_goal=result['confirmed_by_goal'],
            primary_A=goals['A_matched_quality'], secondary_B=goals['B_full_efficiency'],
            scope='New frames of the same IC; original locked weights and VAL thresholds'),
        limitations=['The known C4 operator itself is not claimed as a new invention.',
            'Positive same-IC confirmation does not establish between-IC transfer.',
            'The observed >=0.5pp gate is not a population minimum-effect proof.',
            'Noninferiority versus full Plain24 C4 is not confirmed.',
            'Further ablations use old TRAIN/VAL; the locked confirmation must not become tuning data.'],
        files={key: dict(row, purposes=sorted(row['purposes'])) for key, row in sorted(files.items())})


def verify(manifest):
    failures = []
    for name, row in manifest['files'].items():
        path = REPO / name
        if not path.is_file() or sha(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            failures.append(name)
    return dict(status='PASS_FROZEN_MODEL_HASH_VERIFICATION' if not failures else 'FAIL_FROZEN_MODEL_HASH_VERIFICATION',
        files_checked=len(manifest['files']), learned_final_checkpoints=manifest['learned_final_checkpoints'],
        initial_checkpoints=manifest['initial_checkpoints'],
        materialized_checkpoints=manifest['materialized_checkpoints'],
        frozen_manifest_sha256=sha(HERE / 'FROZEN_MODEL.json'), mismatches=failures,
        historical_files_modified=False, model_inference_or_training_performed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    target = HERE / 'FROZEN_MODEL.json'
    if not args.verify:
        manifest = build()
        encoded = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
        if target.exists():
            assert target.read_text() == encoded, 'Refuse to rewrite an existing model freeze with a different inventory'
        else:
            target.write_text(encoded)
    report = verify(read(target))
    (HERE / 'VERIFY.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report))
    assert not report['mismatches']


if __name__ == '__main__':
    main()
