"""Plan and execute the user-authorized cleanup for the retained eleven options.

Use --plan first; --execute accepts only that saved plan. Never follows symlinks.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil

PROJECT = Path(__file__).resolve().parents[1]
G = PROJECT / 'gipoteza'
OUT = G / 'cleanup_11_20261027'
PLAN = OUT / 'PLAN.json'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def files(p):
    if p.is_symlink() or p.is_file():
        yield p
    elif p.is_dir():
        for directory, dirs, names in os.walk(p, followlinks=False):
            for d in list(dirs):
                candidate = Path(directory) / d
                if candidate.is_symlink():
                    yield candidate
                    dirs.remove(d)
            for n in names:
                yield Path(directory) / n


def safe(rel):
    p = PROJECT / rel
    assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
    assert Path(rel).parts[0] not in ('.git', '.venv', '.agents', '.codex', '.aws')
    assert p.parent.resolve().is_relative_to(PROJECT)
    return p


def make_plan():
    assert not PLAN.exists(), 'Do not replace an existing deletion plan'
    OUT.mkdir(exist_ok=True)
    explicit = [
        'gipoteza/search_20261027', 'gipoteza/search_20261027_round2',
        'gipoteza/shared_models', 'gipoteza/shared_data/carinthia_s',
        'gipoteza/shared_data/mixed_wm', 'gipoteza/shared_data/dspcbsd_plus',
        'gipoteza/cleanup_history', 'gipoteza/PREVIOUS_RESULTS.md',
        'gipoteza/10_gipotez_dlya_vkr.md',
        'gipoteza/diploma_selection_20261027/CANDIDATES_10_DRAFT.md',
        'papers/ctm_yolov10', 'papers/semiwafernet', 'papers/vmamba',
        'papers/vit_tiny', 'papers/transformer_segmentation',
        'papers/wafer_defect_classifier', 'papers/reproduction.py',
        'datasets/ctm_yolov10', 'datasets/semiwafernet',
        'datasets/transformer_segmentation', 'datasets/vmamba_author',
        'third_party', 'outputs', 'predict.py',
        '_semiwafernet_extract.txt', '_vmamba_extract.txt',
        'semiwafernet.pdf', 'transformer_segm.pdf', 'vit.pdf', 'vmamba.pdf', 'yolov10n.pt',
        'scripts/create_wm38k_dataset.py', 'scripts/create_wm811k_segmentation_dataset.py',
        'scripts/image_size_distribution.py',
        'tests/test_vmamba_windows_data.py', 'tests/test_semiwafernet_seed.py',
        'tests/test_semiwafernet_ssl_collation.py', 'tests/test_reproduction_protocol.py',
        'checkpoints/radon_cnn_mps_full_20261008/superseded_before_loss_fix',
        '.pytest_cache', '.ruff_cache', '.DS_Store', 'datasets/.DS_Store',
    ]
    kept_checkpoints = {'polar_cnn_accepted_mps_20261011', 'radon_cnn_mps_full_20261008'}
    explicit += [str(p.relative_to(PROJECT)) for p in (PROJECT / 'checkpoints').iterdir()
                 if p.name not in kept_checkpoints]
    # Caches are unrelated derived files, except inside the immutable H7 package.
    for base in ('common', 'papers', 'tests', 'checkpoints'):
        for directory, dirs, names in os.walk(PROJECT / base, followlinks=False):
            for d in list(dirs):
                if d in ('__pycache__', '.pytest_cache', '.ruff_cache'):
                    explicit.append(str((Path(directory) / d).relative_to(PROJECT)))
                    dirs.remove(d)
            for n in names:
                if n == '.DS_Store':
                    explicit.append(str((Path(directory) / n).relative_to(PROJECT)))
    roots = []
    for rel in sorted(set(explicit), key=lambda s: (len(Path(s).parts), s)):
        p = safe(rel)
        if not p.exists() and not p.is_symlink():
            continue
        if not any(p.is_relative_to(PROJECT / v) for v in roots):
            roots.append(rel)
    protected = json.loads((G / 'retained_evidence_20261027/H7_PROTECTED.json').read_text())
    preserve = {**protected['h7_sha256'], **protected['outside_sha256']}
    retain_roots = [
        'gipoteza/diploma_selection_20261027', 'gipoteza/retained_evidence_20261027',
        'gipoteza/shared_data/ic_sem_m2', 'gipoteza/shared_data/refics',
        'papers/radon_cnn', 'papers/polar_cnn', 'common',
        'checkpoints/radon_cnn_mps_full_20261008',
        'checkpoints/polar_cnn_accepted_mps_20261011',
    ]

    def deleting(p):
        return any(p.is_relative_to(PROJECT / rel) for rel in roots)

    for rel in retain_roots:
        for p in files(PROJECT / rel):
            if not p.is_symlink() and not deleting(p):
                preserve[str(p.relative_to(PROJECT))] = sha(p)
    for rel in ('litobzor_corrected.xlsx', 'datasets/wm811k/labels.csv',
                'gipoteza/11_gipotez_dlya_vkr.md', 'gipoteza/H7_1.md', 'gipoteza/literatura_20261019.md'):
        preserve[rel] = sha(PROJECT / rel)
    # Protect exact selected native source PNGs; leave the whole source corpus in place.
    data = json.loads((G / 'diploma_selection_20261027/wafer_data/MANIFEST.json').read_text())
    for r in data['records']:
        preserve['datasets/wm811k/images/' + r['filename']] = r['source_file_sha256']
    for rel, expected in preserve.items():
        p = PROJECT / rel
        assert not deleting(p), 'Protected file in deletion scope: ' + rel
        assert sha(p) == expected, 'Retained bytes changed: ' + rel
    for rel, target in protected['symlinks'].items():
        p = PROJECT / rel
        assert p.is_symlink() and str(p.readlink()) == target and p.resolve().exists()
        assert not deleting(p)
    rows = []
    for rel in roots:
        for p in files(PROJECT / rel):
            rows.append({'path': str(p.relative_to(PROJECT)), 'bytes': p.lstat().st_size,
                         'symlink': p.is_symlink(),
                         'sha256': None if p.is_symlink() else sha(p)})
    plan = {
        'status': 'PREPARED_NOT_EXECUTED', 'date': '2026-10-27',
        'authorization': 'User explicitly requested deletion of everything unrelated to the retained eleven hypotheses across the repository.',
        'selected_total': 11, 'delete_roots': roots, 'delete_files': rows,
        'delete_file_count': len(rows), 'delete_logical_bytes': sum(r['bytes'] for r in rows),
        'retained_sha256': preserve, 'retained_symlinks': protected['symlinks'],
        'source_data_policy': 'Keep the whole native WM811K corpus and original LSWMD.pkl; protect every selected PNG. Keep IC SEM M2 and REFICS provenance.',
        'baseline_policy': 'Keep accepted RadonCNN and PolarCNN code/checkpoints as external comparisons for the five wafer options, and their common runtime. No prior metric is comparable without matching splits/classes.',
        'old_evidence_policy': 'Selected SEM/EP2 records and dependency code were extracted with exact hashes before deletion of full mixed archives. Negative controls are retained.',
    }
    save(PLAN, plan)
    (OUT / 'README.md').write_text('''# Очистка под 11 оставшихся гипотез

Авторизована пользователем 27 октября 2026: очистить репозиторий от материалов остальных веток. [Точный план и SHA](PLAN.json), [результат выполнения](COMPLETE.json), [проверка сохранности и расчётов](VERIFICATION.json).

Сохранены:

- Все 292 файла H7.1 и три внутренние ссылки на её зависимости.
- Новый wafer-стенд: протокол, код, данные, 15 CNN и 6 ExtraTrees, прогнозы и отрицательные контроли C1/C7.
- Исходный WM-811K целиком, LSWMD.pkl и метаданные; они относятся к оставшимся wafer-гипотезам и нужны для увеличения эксперимента.
- Исходники IC SEM M2 и REFICS; отчёты/протоколы и доступный исторический код SEM/EP2, в том числе отрицательные расширения, ограничивающие выводы.
- RadonCNN и PolarCNN с принятыми результатами и весами как готовые внешние сравнения для wafer-блока; общая библиотека и её проверки. Это два базовых метода, не ещё две выбранные гипотезы. Их исторические цифры нельзя прямо сравнивать с новым lot-split и девятью классами.
- Исследовательский контекст, список литературы, исходная литературная таблица и существующая таблица воспроизведения. Контекст и таблица результатов сохранены побайтно.

Удалены другие поисковые ветки, их отдельные датасеты/модели, старые смешанные архивы после извлечения нужного, неиспользуемые реализации ViT/VMamba/SemiWaferNet/YOLO/Transformer Segmentation и их диагностические обучения. Git-история и рабочее окружение не очищались.

Старые отчёты и frozen manifests сохраняют исходные даты и ссылки на прежнее расположение: это историческое свидетельство. Актуальные входные точки — [список 11](../11_gipotez_dlya_vkr.md), [H7.1](../h7_readout_20261025/README.md), [выбранные исторические материалы](../retained_evidence_20261027/README.md). Полные ранние данные SEM/EP2 были удалены до этой очистки и ею не восстанавливались.

Скрипт `curate_11_20261027.py` — одноразовая подготовка из старых архивов; после их удаления повторять его нельзя. `cleanup_11_20261027.py` исполняет только сохранённый точный план и отказывается повторно выполнять завершённую очистку.
''')
    print(json.dumps({'status': plan['status'], 'delete_roots': len(roots),
                      'files': len(rows), 'logical_GB': plan['delete_logical_bytes'] / 1e9,
                      'protected_files': len(preserve)}, ensure_ascii=False))


def execute():
    assert not (OUT / 'COMPLETE.json').exists(), 'Cleanup already completed'
    plan = json.loads(PLAN.read_text())
    assert plan['status'] == 'PREPARED_NOT_EXECUTED' and plan['selected_total'] == 11
    for rel, expected in plan['retained_sha256'].items():
        assert sha(PROJECT / rel) == expected, 'Changed retained file: ' + rel
    for row in plan['delete_files']:
        p = safe(row['path'])
        assert p.lstat().st_size == row['bytes']
        if not row['symlink']:
            assert sha(p) == row['sha256'], 'Changed deletion candidate: ' + row['path']
    # Reject any newly appeared file: only the reviewed inventory can be removed.
    expected_paths = {r['path'] for r in plan['delete_files']}
    actual_paths = {str(p.relative_to(PROJECT)) for rel in plan['delete_roots']
                    for p in files(safe(rel))}
    assert actual_paths == expected_paths, 'Deletion inventory changed'
    for rel in plan['delete_roots']:
        p = safe(rel)
        assert not any((PROJECT / keep).is_relative_to(p) for keep in plan['retained_sha256'])
        if p.is_symlink() or p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p)
    for rel, expected in plan['retained_sha256'].items():
        assert sha(PROJECT / rel) == expected, 'Changed retained file after cleanup: ' + rel
    for rel, target in plan['retained_symlinks'].items():
        p = PROJECT / rel
        assert p.is_symlink() and str(p.readlink()) == target and p.resolve().exists()
    assert all(not (PROJECT / p).exists() for p in plan['delete_roots'])
    complete = {'status': 'COMPLETE', 'selected_total': 11,
                'plan_sha256': sha(PLAN), 'deleted_files': plan['delete_file_count'],
                'deleted_logical_bytes': plan['delete_logical_bytes'],
                'retained_hashes_verified': len(plan['retained_sha256']),
                'H7_original_files_unchanged': 292, 'internal_symlinks_verified': 3,
                'units': 'Logical file bytes, not measured APFS free-space change; git history retained.'}
    save(OUT / 'COMPLETE.json', complete)
    print(json.dumps(complete, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--plan', action='store_true')
    group.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    make_plan() if args.plan else execute()
