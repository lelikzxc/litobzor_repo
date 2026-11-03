"""Render saved timing measurements; no model inference or fitting."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/sem_pipeline_timing_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    d = json.loads((ROOT / 'TIMING.json').read_text())
    assert d['status'] == 'COMPLETE_PIPELINE_TIMING' and d['correctness_checks_all_passed']
    lookup = {(r['batch'], r['cache'], r['architecture']): r for r in d['summary']}
    (ROOT / 'figures').mkdir(exist_ok=True)
    colors = {'group6': '#206f9d', 'untied24_single': '#d17b33'}
    names = {'group6': 'GroupCNN', 'untied24_single': 'Untied24'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, batch in zip(axes, [1, 16]):
        for a, arch in enumerate(colors):
            for m, mode in enumerate(['cold', 'warm']):
                row = lookup[batch, mode, arch]; x = m + (a - .5) * .2
                median, p95 = row['median_ms_per_patch'], row['p95_request_ms_per_patch']
                ax.plot([x, x], [median, p95], color=colors[arch], lw=2)
                ax.scatter(x, median, s=65, color=colors[arch], label=names[arch] if m == 0 else None)
                ax.plot(x, p95, marker='_', ms=10, color=colors[arch])
        ax.set(xticks=[0, 1], xticklabels=['Cold cache', 'Warm cache'],
               ylabel='Milliseconds per patch', title=f'Batch {batch}: median and request p95 / batch')
        ax.set_ylim(bottom=0); ax.grid(axis='y', alpha=.2); ax.legend(frameon=False)
    fig.suptitle('Complete resident-input pipeline on MPS + CPU, 3 initializations')
    path = ROOT / 'figures/pipeline_latency.png'; fig.savefig(path, dpi=180); plt.close(fig)
    lines = [
        '# Время полного конвейера на MPS и CPU', '',
        '3 ноября 2026. Замер шести неизменённых моделей на 96 фрагментах из всех 24 кадров. '
        'Для GroupCNN и Untied24 использованы одинаковые фрагменты, три прежние инициализации '
        'и ранее зафиксированные окончательные пороги. Новых обучений в этом замере нет.', '',
        'Время включает подготовку нативного входа 48×48, усреднение до 12×12 и нормировку, '
        'перенос на MPS, CNN, sigmoid, перенос вероятностей на CPU, бинаризацию и вычисление '
        'геометрического класса на CPU. Исходный фрагмент уже находится в оперативной памяти. '
        'Чтение PNG, загрузка весов, материализация групповых ядер, склейка кадров и запуск '
        'интерпретатора вынесены за пределы запроса. Это не задержка обработки целой пластины.', '',
        '| Пакет | Кэш классов | Модель | Медиана запроса, мс | p95 запроса, мс | Медиана / фрагмент, мс |',
        '|---:|---|---|---:|---:|---:|'
    ]
    for batch in [1, 16]:
        for mode in ['cold', 'warm']:
            for arch in names:
                r = lookup[batch, mode, arch]
                lines.append(f"| {batch} | {mode} | {names[arch]} | {r['median_ms']:.4f} | {r['p95_ms']:.4f} | {r['median_ms_per_patch']:.4f} |")
    lines += ['', 'Cold: кэш ранее полученных классов пуст в начале каждого запроса; '
              'повторяющиеся маски внутри пакета разрешено объединять. Warm: все маски '
              'выбранных фрагментов заранее присутствуют в кэше. Warm относится к повторным '
              'маскам и не характеризует обычный поток новых изображений.', '',
              'Нормировка времени пакета на число фрагментов показывает пропускную способность '
              'последовательной обработки пакетов. Она не равна задержке отдельного запроса '
              'размера 1. p95 характеризует запрос, а не независимые длительности его элементов.', '',
              'Три повтора для каждого размера пакета, парный случайный порядок моделей и режимов. '
              'Медианы и p95 объединяют запросы трёх инициализаций; измерения не являются '
              'независимыми образцами производительности разных машин. Статистическое превосходство '
              'по скорости не проверяется.', '',
              '![Медиана и p95 полного конвейера](figures/pipeline_latency.png)', '',
              '## Отношение медиан Group / Untied', '']
    for batch in [1, 16]:
        for mode in ['cold', 'warm']:
            ratio = lookup[batch, mode, 'group6']['median_ms'] / lookup[batch, mode, 'untied24_single']['median_ms']
            lines.append(f'- Пакет {batch}, {mode}: {ratio:.3f}.')
    lines += ['', 'Отношение меньше 1 означает меньшую измеренную медиану Group. '
              'Архитектуры используют одинаковые 1527552 MAC и одинаковую развёрнутую '
              '24-канальную форму свёрток, поэтому эти замеры не обосновывают универсальное '
              'ускорение GroupCNN.', '', '## Вспомогательный профиль этапов', '',
              'Среднее время в мс для пакета 16/cold. В этом отдельном профиле добавлены '
              'синхронизации между этапами; суммы не заменяют основной замер полного запроса.', '',
              '| Этап | GroupCNN | Untied24 |', '|---|---:|---:|']
    p = {r['architecture']: r for r in d['profile_summary']}
    stage_labels = ['Подготовка на CPU', 'Перенос CPU → MPS', 'CNN и sigmoid',
                    'Перенос MPS → CPU', 'Бинаризация', 'Геометрический класс и финальная синхронизация']
    for i, name in enumerate(stage_labels):
        lines.append(f"| {name} | {p['group6']['mean_ms'][i]:.4f} | {p['untied24_single']['mean_ms'][i]:.4f} |")
    lines += ['', f"Подготовка измерительного окружения: {d['setup_seconds']:.2f} с. "
              'Включает чтение исходных PNG, загрузку моделей, подготовку решателя и кэша и прогрев; '
              'это не отдельно измеренное время развёртывания одной модели.', '',
              f"Среда: `{d['hardware']['platform']}`, PyTorch `{d['hardware']['torch']}`, "
              f"MPS, {d['hardware']['CPU_threads']} потока CPU. Автоматический MPS fallback отключён.", '',
              'Все выходные вероятности проверены с допуском 1e-5 относительно архива; '
              'маски и классы совпали точно во всех основных измерениях. Исходные файлы и веса '
              'проверены по SHA-256 до и после замера. Длительности каждого запроса и '
              'синхронизированного профиля сохранены в [TIMING.json](TIMING.json).', '',
              'Воспроизведение: [benchmark.py](benchmark.py), [протокол](PROTOCOL.md), '
              '[фиксация входов](LOCK.json). Скрипт отказывается перезаписывать завершённый замер.', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    (ROOT / 'README.md').write_text('# Время полного конвейера\n\n[Отчёт](REPORT.md), [протокол](PROTOCOL.md), '
                                  '[код](benchmark.py), [измерения](TIMING.json).\n')
    outputs = ['REPORT.md', 'README.md', 'figures/pipeline_latency.png']
    manifest = dict(input_sha256={n: sha(ROOT / n) for n in ['TIMING.json', 'LOCK.json', 'report.py']},
                    output_sha256={n: sha(ROOT / n) for n in outputs})
    (ROOT / 'REPORT_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__': main()
