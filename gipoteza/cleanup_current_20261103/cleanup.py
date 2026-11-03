"""Prune abandoned experiments while preserving current model and hash bindings.

Plan is explicit and immutable. Archive compact evidence before removing any
unused binary artifacts. Never traverse symlink targets or touch the user project
outside gipoteza. No git reset/clean or rewriting scientific records.
"""
from pathlib import Path
import argparse
import collections
import datetime
import hashlib
import json
import re
import zipfile

HERE = Path(__file__).resolve().parent
G, REPO = HERE.parent, HERE.parents[1]
ACTIVE = [
    'sem_arch_groupcnn_20261102', 'sem_arch_groupcnn_confirmation_20261102',
    'sem_arch_calibrated_confirmation_20261103', 'sem_arch_calibration_dev_20261103',
    'sem_arch_seed_extension_20261103', 'sem_arch_untied_control_20261103',
    'sem_arch_bce_confirmation_20261103', 'sem_arch_loss_factorial_20261103',
    'sem_model_freeze_20261103', 'sem_arch_error_analysis_20261103',
    'sem_arch_final_validation_20261103', 'sem_arch_pipeline_timing_20261103',
    'eda_ic_sem_m2_20261102', 'sem_physics_materials', 'slides_revision_20261029',
]
CACHE_NAMES = {'__pycache__', 'matplotlib_cache', '.pytest_cache', '.mypy_cache', '.ruff_cache'}
TEXT_SUFFIXES = {'.md', '.py', '.json', '.jsonl', '.txt', '.yaml', '.yml', '.toml', '.csv', '.tsv', '.sh'}


def read(p): return json.loads(Path(p).read_text())
def relative(p): return str(Path(p).absolute().relative_to(REPO))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def save(p, value): Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
def cache(p): return any(part in CACHE_NAMES for part in p.relative_to(G).parts) or p.name in {'.DS_Store', 'Thumbs.db'} or p.suffix == '.pyc'


def inventory():
    # Path.rglob does not descend directory symlinks. Keep node_modules external.
    return sorted(p for p in G.rglob('*') if HERE not in p.parents and (p.is_symlink() or p.is_file()))


def protected_files():
    protected = {}; registries = {}

    def add(p, expected=None, reason='active_material'):
        p = Path(p).absolute()
        if not p.is_file() or p.is_symlink(): return False
        try: p.relative_to(G)
        except ValueError: return False
        digest = sha(p)
        assert expected is None or digest == expected, (str(p), expected, digest)
        item = protected.setdefault(relative(p), dict(sha256=digest, bytes=p.stat().st_size, reasons=[]))
        if reason not in item['reasons']: item['reasons'].append(reason)
        return True

    def resolve(name, parent, expected):
        if not isinstance(name, str) or len(name) > 1500 or '\n' in name: return
        path = Path(name)
        options = [path] if path.is_absolute() else [REPO / path, parent / path, G / path]
        for option in options:
            if option.is_file() and not option.is_symlink() and sha(option) == expected:
                add(option, expected, 'current_hash_binding'); return

    def visit(value, parent):
        if isinstance(value, list):
            for row in value: visit(row, parent)
        elif isinstance(value, dict):
            for name, row in value.items():
                if isinstance(row, str) and re.fullmatch('[0-9a-f]{64}', row): resolve(name, parent, row)
                if isinstance(row, dict) and isinstance(row.get('sha256'), str): resolve(name, parent, row['sha256'])
                visit(row, parent)
            for key, name in value.items():
                if not isinstance(name, str): continue
                if key in ['file', 'path']:
                    expected = value.get('sha256')
                elif key.endswith('_file'):
                    expected = value.get(key[:-5] + '_sha256')
                else: continue
                if isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected): resolve(name, parent, expected)

    for name in ACTIVE:
        for p in (G / name).rglob('*'):
            if p.is_file() and not p.is_symlink() and not cache(p): add(p)
            if p.is_file() and p.suffix == '.json' and not cache(p):
                visit(read(p), p.parent)
                registries[relative(p)] = sha(p)
    frozen = G / 'sem_model_freeze_20261103/FROZEN_MODEL.json'
    for name, row in read(frozen)['files'].items(): add(REPO / name, row['sha256'], 'frozen_275_family')

    # EDA dataset and runtime inputs are not all explicitly listed in audits.
    pilot = G / 'sem_architecture_pilot_20261102'
    for name in ['DATA.npz', 'DATA_MANIFEST.json', 'TRAIN_VAL.npz', 'SUBSETS.json', 'EVAL.npz', 'EVAL_INDEX.json',
                 'prepare.py', 'physics_adapter.py']:
        add(pilot / name, reason='EDA_or_training_dependency')
    for p in (pilot / 'sources').glob('*'): add(p, reason='EDA_native_sources')
    for frame in range(8):
        for stem, suffix in [('sem', '.png'), ('label', '.svg')]:
            add(G / 'sem_confirmation_20261028/sources' / f'{stem}{frame:04}{suffix}', reason='EDA_native_sources')
    for p in (G / 'shared_data/ic_sem_m2').rglob('*'):
        if p.is_file() and not cache(p): add(p, reason='SEM_source_dataset')
    for name in ['EDA_IC_SEM_M2.ipynb', 'literatura_20261019.md', 'gipoteza.md']:
        add(G / name, reason='thesis_document')
    return protected, registries


def make_plan():
    assert not (HERE / 'PLAN.json').exists(), 'Do not overwrite a cleanup plan'
    protected, registries = protected_files(); files = inventory(); remove = []; summaries = collections.defaultdict(lambda: [0, 0])
    for p in files:
        name = relative(p)
        if name in protected or p == G / 'README.md': continue
        row = dict(path=name, bytes=0 if p.is_symlink() else p.stat().st_size,
                   symlink=p.is_symlink(), target=str(p.readlink()) if p.is_symlink() else None,
                   sha256=None if p.is_symlink() else sha(p),
                   reason='regenerable_cache' if cache(p) else 'outside_current_model_dependency_set')
        row['archive'] = not row['symlink'] and not cache(p) and (p.suffix in TEXT_SUFFIXES or
            (p.suffix.lower() in {'.png', '.svg', '.pdf'} and any(x in {'figures', 'plots'} for x in p.parts)))
        remove.append(row)
        top = p.relative_to(G).parts[0]; summaries[top][0] += 1; summaries[top][1] += row['bytes']
    value = dict(status='PLANNED_BEFORE_DELETION', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cleanup_source_sha256=sha(Path(__file__)), active_directories=ACTIVE, protected_files=protected,
        active_json_registry_sha256=registries, remove=remove, files_to_remove=len(remove),
        bytes_to_remove=sum(r['bytes'] for r in remove), compact_archive_files=sum(r['archive'] for r in remove),
        per_directory={k:dict(files=v[0],bytes=v[1]) for k,v in sorted(summaries.items())},
        outside_gipoteza_untouched=True, user_original_notebooks_data_bibliography_environment_preserved=True)
    save(HERE / 'PLAN.json', value)
    print(json.dumps({k:value[k] for k in ['status','files_to_remove','bytes_to_remove','compact_archive_files','per_directory']}), flush=True)


def apply():
    assert not (HERE / 'APPLIED.json').exists(), 'Already applied'
    plan = read(HERE / 'PLAN.json'); assert plan['cleanup_source_sha256'] == sha(Path(__file__))
    for name, row in plan['protected_files'].items(): assert sha(REPO / name) == row['sha256'], name
    for row in plan['remove']:
        p = REPO / row['path']; assert G in p.parents and HERE not in p.parents
        if row['symlink']: assert p.is_symlink() and str(p.readlink()) == row['target']
        else: assert p.is_file() and not p.is_symlink() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
    archive = HERE / 'ARCHIVED_EVIDENCE.zip'; assert not archive.exists()
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for row in plan['remove']:
            if row['archive']: z.write(REPO / row['path'], row['path'])
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == plan['compact_archive_files']
        for row in plan['remove']:
            if row['archive']: assert hashlib.sha256(z.read(row['path'])).hexdigest() == row['sha256']
    for row in plan['remove']: (REPO / row['path']).unlink()
    empty = 0
    for p in sorted([q for q in G.rglob('*') if q.is_dir() and not q.is_symlink() and HERE not in q.parents], key=lambda q:len(q.parts), reverse=True):
        try: p.rmdir(); empty += 1
        except OSError: pass
    result = dict(status='CLEANUP_APPLIED', removed_files=plan['files_to_remove'],
        removed_bytes=plan['bytes_to_remove'], archive_bytes=archive.stat().st_size,
        net_reclaimed_bytes=plan['bytes_to_remove']-archive.stat().st_size,
        archived_evidence_files=plan['compact_archive_files'], empty_directories_removed=empty,
        archive_sha256=sha(archive), plan_sha256=sha(HERE / 'PLAN.json'),
        protected_files=len(plan['protected_files']), destructive_scope='Only explicitly inventoried unused files under gipoteza',
        note='Old unused checkpoints/data/predictions are removed; compact archive retains code/reports/results, not full rerun data for discarded branches.')
    save(HERE / 'APPLIED.json', result); print(json.dumps(result), flush=True)


def verify():
    plan, result = read(HERE / 'PLAN.json'), read(HERE / 'APPLIED.json')
    assert result['plan_sha256'] == sha(HERE / 'PLAN.json')
    assert result['archive_sha256'] == sha(HERE / 'ARCHIVED_EVIDENCE.zip')
    for name, row in plan['protected_files'].items(): assert sha(REPO / name) == row['sha256'], name
    for row in plan['remove']: assert not (REPO / row['path']).exists() and not (REPO / row['path']).is_symlink(), row['path']
    value = dict(status='PASS_CLEANUP_HASH_VERIFICATION', preserved_files=len(plan['protected_files']),
                 removed_files_absent=len(plan['remove']), archive_verified=True, scientific_file_changes=0)
    save(HERE / 'VERIFY.json', value); print(json.dumps(value), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['plan','apply','verify']); args=p.parse_args()
    {'plan':make_plan,'apply':apply,'verify':verify}[args.action]()
