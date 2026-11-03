"""Make a thesis-ready overview from completed, audited experiments only."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/sem_final_validation_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
G = ROOT.parent
REPLICATION = G / 'sem_arch_seed_extension_20261103'
ERRORS = G / 'sem_arch_error_analysis_20261103'
TIMING = G / 'sem_arch_pipeline_timing_20261103'


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    d, audit = read(REPLICATION / 'RESULT.json'), read(REPLICATION / 'AUDIT.json')
    error_checks, timing_checks = read(ERRORS / 'CHECKS.json'), read(TIMING / 'CHECKS.json')
    assert d['status'] == 'PASS_SEED_REPLICATION' and all(audit['checks'].values())
    assert all(error_checks['checks'].values()) and all(timing_checks['checks'].values())
    source = d['analyses']['final_calibrated']; all5 = source['all5']; new2 = source['new2']
    cells = all5['cells']; delta = all5['group_minus_untied']['mean']
    rows = all5['group_minus_untied']['per_seed']
    figure_dir = ROOT / 'figures'; figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), constrained_layout=True)
    for ax, metric, ci_key, title in zip(axes, ['mask_F1', 'mean_frame_AP'],
            ['CI97_5_F1', 'CI97_5_AP'], ['Classification Macro-F1', 'Segmentation mean-frame AP']):
        for i, row in enumerate(rows):
            value = 100 * row[metric]
            ax.barh(i, value, height=.6, color='#2c759e' if i < 3 else '#d18032')
            ax.text(value + .025 if metric == 'mean_frame_AP' else value + .08, i,
                    f'{value:+.3f}', va='center', fontsize=10)
        lo, hi = np.array(all5[ci_key]) * 100; point = 100 * delta[metric]
        ax.errorbar(point, 6, xerr=np.array([[point - lo], [hi - point]]),
                    fmt='o', color='#151515', capsize=5, lw=2)
        ax.axhline(5, color='.6', lw=.8)
        ax.axvline(0, color='.3', lw=.8)
        ax.set(yticks=[0, 1, 2, 3, 4, 6],
            yticklabels=[str(r['seed']) + (' (new)' if i >= 3 else '') for i, r in enumerate(rows)] + ['All 5 + CI'],
            xlabel='Group minus Untied, percentage points', title=title)
        ax.invert_yaxis(); ax.set_xlim(0, max(hi, max(100 * r[metric] for r in rows)) * 1.22)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Seed replication: fixed calibrated pipeline, same 6,144 patches / 24 frames\n'
                 'Aggregate intervals: descriptive central 97.5%, paired seeds and frames')
    fig.savefig(figure_dir / 'five_seed_effects.png', dpi=180); plt.close(fig)
    lines = ['# Завершающие проверки архитектурной гипотезы', '',
        '3 ноября 2026. Выполнены четыре дополнительных обучения, разбор ошибок и замер '
        'всего конвейера обработки. Основной вариант модели, обучение с общим вспомогательным '
        'геометрическим лоссом, данные и процедура калибровки сохранены.', '',
        '## Устойчивость к инициализации', '',
        'Для seed 5801 и 5821 обучены обе архитектуры. У каждой пары одинаковая начальная '
        'функция, фактическая ширина 24 канала и 1527552 MAC на фрагмент. Group имеет '
        '2671 свободный параметр, Untied24 — 10681. Гиперпараметры, 27 полных масок, '
        '1152 слабые метки, 2000 шагов и бюджет окончательной калибровки 512 классовых '
        'меток общие. Все новые веса и четыре порога сохранены. Лучшие seed не отбирались.', '',
        '| Показатель, среднее пяти запусков | GroupCNN | Untied24 | Разность, п.п. |',
        '|---|---:|---:|---:|']
    for metric, name in [('mask_F1', 'Macro-F1 классификации, %'),
                         ('mean_frame_AP', 'Покадровый AP сегментации, %'), ('IoU', 'IoU сегментации, %')]:
        lines.append(f"| {name} | {100 * cells['group6']['mean'][metric]:.4f} | "
                     f"{100 * cells['untied24_single']['mean'][metric]:.4f} | {100 * delta[metric]:+.4f} |")
    new_delta = new2['group_minus_untied']['mean']
    lines += ['', f"Два новых seed отдельно: ΔMacro-F1 {100 * new_delta['mask_F1']:+.4f} п.п., "
        f"ΔAP {100 * new_delta['mean_frame_AP']:+.4f} п.п., ΔIoU {100 * new_delta['IoU']:+.4f} п.п. "
        'Оба дали положительный прирост по всем трём показателям. По пяти seed прирост F1 '
        'положителен в каждой паре.', '',
        f"Приближённые описательные 97,5%-интервалы пяти seed: ΔF1 "
        f"[{100 * all5['CI97_5_F1'][0]:+.4f}; {100 * all5['CI97_5_F1'][1]:+.4f}] п.п.; "
        f"ΔAP [{100 * all5['CI97_5_AP'][0]:+.4f}; {100 * all5['CI97_5_AP'][1]:+.4f}] п.п. "
        'Пересэмплируются парные инициализации и кадры, 10000 повторов, поправка на два исхода. '
        'Среднее — по отдельным запускам, не ансамбль.', '',
        '![Разности по пяти инициализациям](figures/five_seed_effects.png)', '',
        'Репликация проходит заранее заданные описательные критерии как для новых двух '
        'пар отдельно, так и для всех пяти. Тестовые изображения — те же 6144 фрагмента '
        '24 кадров: это устойчивость к инициализации, не новая независимая когорта. '
        'Первое закрытое подтверждение трёх seed остаётся отдельным неизменённым опытом.', '',
        'Диагностика при общем пороге 0,5 также даёт положительное среднее: '
        f"ΔF1 пяти seed {100 * d['analyses']['fixed05']['all5']['group_minus_untied']['mean']['mask_F1']:+.4f} п.п. "
        'Но один прежний seed при 0,5 имеет отрицательную разность, поэтому универсальное '
        'преимущество при любом пороге не заявляется.', '',
        f"[Независимый аудит репликации](../{REPLICATION.name}/AUDIT.json): "
        f"{audit.get('checks_passed', len(audit['checks']))}/{audit.get('checks_total', len(audit['checks']))}. "
        f"[Полный отчёт, код и веса](../{REPLICATION.name}/REPORT.md).", '',
        '## Разбор ошибок', '',
        'На трёх прежних seed средний F1 Group выше для всех девяти классов. '
        'Исправлено 726 классификаций, ухудшено 434; это суммы по 18432 парам '
        '«фрагмент × seed» с повторным использованием 6144 изображений. По направлениям '
        'наблюдаются компромиссы: уменьшаются пропуски связей LR и ложные связи TB, '
        'но немного растут ложные связи LR и пропуски TB. Причинный механизм этими '
        'наблюдениями не доказывается.', '',
        f"[Таблицы, матрицы и восемь примеров, включая ухудшения](../{ERRORS.name}/REPORT.md). "
        'Примеры выбраны воспроизводимым SHA-ранжированием внутри фиксированных категорий; '
        'метрики и нативные входы проверены. Этот разбор относится к прежним трём seed.', '',
        '## Время полного конвейера', '',
        'На MPS и CPU медиана одиночного запроса с пустым кэшем классов — 0,9341 мс '
        'у Group и 0,9344 мс у Untied. Для пакета 16 — 2,5776 и 2,5830 мс соответственно '
        '(около 0,161 мс на фрагмент в пакете). По этому замеру скорости практически равны.', '',
        'Учтены подготовка нативного фрагмента 48×48 до 12×12, переносы CPU↔MPS, CNN, '
        'бинаризация и геометрический классификатор. Исходный фрагмент уже в оперативной '
        'памяти; диск, загрузка весов и обработка целого кадра сюда не входят. '
        '3672 запроса, три прежние инициализации, размеры пакета 1 и 16; warm/cold '
        'кэш разделены. Во всех запросах маски и классы совпали с архивом.', '',
        f"[Таблица задержек, p95, профиль этапов и код](../{TIMING.name}/REPORT.md). "
        'Проверки артефактов и расчётов 13/13; отдельная проверка кода блокирующих '
        'замечаний не выявила. Это не независимый повтор аппаратного замера.', '',
        '## Что теперь обосновано', '',
        'В данной постановке архитектура с C4-связями весов воспроизводимо улучшает '
        'классификацию геометрии металлизации и качество её маски относительно свободной '
        'CNN24 при одинаковых ширине, MAC, обучающем надзоре и калибровке. '
        'Свободных параметров меньше примерно в четыре раза, время обработки сопоставимо. '
        'Результата достаточно для описания и обсуждения выполненного сравнительного '
        'эксперимента в ВКР; следующий практический шаг — оформить метод и экспериментальную '
        'главу, сохраняя неудачные абляции и ограничения.', '',
        'Данные — один слой M2 одной ИС. Классы вычислены из SVG-масок и описывают '
        'геометрию, а не экспертные категории производственных дефектов или измеренные '
        'электрические обрывы/замыкания. Фильтрация кандидатов использует истинные маски. '
        'Перенос на другую ИС, промышленная эффективность и новизна самого известного '
        'C4-оператора остаются за пределами этих проверок.', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    inputs = [REPLICATION / n for n in ['RESULT.json', 'AUDIT.json', 'LOCK.json', 'REPORT.md']]
    inputs += [ERRORS / n for n in ['RESULT.json', 'CHECKS.json', 'REPORT.md']]
    inputs += [TIMING / n for n in ['TIMING.json', 'CHECKS.json', 'REPORT.md']]
    inputs += [Path(__file__)]
    outputs = [ROOT / 'REPORT.md', figure_dir / 'five_seed_effects.png']
    manifest = dict(input_sha256={str(p.relative_to(G)): sha(p) for p in inputs},
                    output_sha256={str(p.relative_to(ROOT)): sha(p) for p in outputs})
    (ROOT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__': main()
