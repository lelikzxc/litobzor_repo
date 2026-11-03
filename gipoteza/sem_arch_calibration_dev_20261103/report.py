"""Render the audited reused-DEV AP/calibration study; no inference or fitting."""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/sem_factorial_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ARCHS = ['group6', 'untied24_single', 'plain12_C4pooled']
NAMES = {'group6': 'GroupCNN', 'untied24_single': 'Untied24', 'plain12_C4pooled': 'Plain12 C4'}
COLORS = ['#197d86', '#b66530', '#6a7a89']


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def percent(value, decimals=3): return f'{100 * value:.{decimals}f}'
def pp(value, decimals=3): return f'{100 * value:+.{decimals}f}'


def validate():
    result, audit = read(ROOT / 'RESULT.json'), read(ROOT / 'AUDIT.json')
    assert result['status'] == 'COMPLETE_CALIBRATION_DEVELOPMENT_ANALYSIS'
    assert result['fit_count'] == result['optimizer_updates'] == 0 and not result['new_confirmation']
    assert audit['status'].startswith('PASS') and all(audit['checks'].values())
    assert audit['checks_passed'] == audit['checks_total']
    for name, expected in audit['input_sha256'].items():
        path = Path(name) if Path(name).is_absolute() else REPO / name
        assert sha(path) == expected, f'Audit input changed: {name}'
    for name in ['RESULT.json', 'PREDICTIONS.npz', 'DATA.npz', 'CALIBRATION.npz', 'PRAUC.npz', 'LOCK.json']:
        assert str((ROOT / name).relative_to(REPO)) in audit['input_sha256']
    for name, field in [('PREDICTIONS.npz', 'PREDICTIONS_sha256'), ('CALIBRATION.npz', 'CALIBRATION_sha256'),
                        ('PRAUC.npz', 'PRAUC_sha256'), ('LOCK.json', 'LOCK_sha256')]:
        assert sha(ROOT / name) == result[field]
    with np.load(ROOT / 'PRAUC.npz', allow_pickle=False) as z:
        ap = {key: z[key].copy() for key in z.files}
    assert ap['per_frame_ap'].shape == (3, 3, 40)
    assert ap['old_VAL_mean_frame_ap'].shape == (3, 3)
    return result, audit, ap


def figures(result, ap):
    folder = ROOT / 'figures'; folder.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    means = []
    for i, arch in enumerate(ARCHS):
        values = 100 * ap['mean_frame_ap'][i]
        means.append(float(values.mean()))
        ax.hlines(values.mean(), i - .18, i + .18, color=COLORS[i], linewidth=3.5, zorder=2)
        ax.scatter(i + np.array([-.09, 0., .09]), values, color=COLORS[i], s=40,
                   edgecolor='white', linewidth=.8, zorder=3)
        ax.annotate(f'{values.mean():.4f}', (i, values.mean()), xytext=(0, 8),
                    textcoords='offset points', ha='center', fontsize=10)
    ax.set_xticks(range(3), [NAMES[arch] for arch in ARCHS])
    low = min(100 * ap['mean_frame_ap'].min() - .15, min(means) - .2)
    ax.set_ylim(max(0., low), min(100.05, 100 * ap['mean_frame_ap'].max() + .35))
    ax.set_ylabel('Average precision сегментации, %')
    ax.set_title('Ранее просмотренные 40 кадров: AP без выбора порога\n'
                 'Черта — среднее; точки — три инициализации, без интервалов')
    ax.grid(axis='y', alpha=.2)
    fig.savefig(folder / 'calibration_dev_average_precision.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    mode_fields = [('OOF_F1', 'OOF_IoU'), ('fixed05_F1', 'fixed05_IoU'),
                   ('old_VAL_threshold_F1', 'old_VAL_threshold_IoU')]
    for panel, (ax, title) in enumerate(zip(axes, ['Macro-F1 классификации, %', 'IoU сегментации, %'])):
        for j, arch in enumerate(ARCHS):
            x = np.arange(3) + (j - 1) * .25
            fields = [pair[panel] for pair in mode_fields]
            values = [100 * result['cells'][arch][field] for field in fields]
            ax.bar(x, values, width=.23, color=COLORS[j], alpha=.85, label=NAMES[arch])
            for k, value in enumerate(values):
                ax.annotate(f'{value:.2f}', (x[k], value), xytext=(0, 5),
                            textcoords='offset points', ha='center', fontsize=8)
            if panel == 0:
                for k, field in enumerate(fields):
                    points = [100 * row[field] for row in result['cells'][arch]['per_seed']]
                    ax.scatter(x[k] + np.array([-.045, 0., .045]), points, color=COLORS[j],
                               edgecolor='white', linewidth=.5, s=17, zorder=3)
        ax.set_xticks(range(3), ['Калибровка по\nдругим кадрам', 'Общий порог\n0,5', 'Прежние пороги\nиз двух VAL-кадров'])
        ax.set_title(title); ax.set_ylim(0, 105); ax.grid(axis='y', alpha=.2)
    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle('Фиксированные модели: три способа бинаризации на данных разработки\n'
                 'Новых обучений нет; результаты не являются новым закрытым подтверждением', fontsize=11)
    fig.savefig(folder / 'calibration_dev_quality.png', dpi=180)
    plt.close(fig)


def main():
    result, audit, ap = validate(); figures(result, ap)
    eligibility = result['next_study_eligibility']
    effect = result['contrasts']['group_minus_untied']['mean']
    admission = 'пройден' if eligibility['passed'] else 'не пройден'
    lines = ['# AP сегментации и перенос калибровки между кадрами', '',
        '3 ноября 2026 года. Выполнены две проверки фиксированных CNN: '
        'порогонезависимое ранжирование пикселей и покадровая проверка калибровки. '
        'Использованы 10 240 фрагментов 40 ранее оценённых кадров одной ИС. '
        'В этом новом опыте они явно являются данными разработки. '
        'Это не новая закрытая выборка и не независимое подтверждение.', '',
        f"Group−Untied24: ΔAP {pp(effect['mean_frame_AP'], 6)} п.п., "
        f"ΔMacro-F1 после покадровой калибровки {pp(effect['OOF_F1'])} п.п., "
        f"ΔIoU {pp(effect['OOF_IoU'])} п.п. Заранее установленный описательный "
        f'допуск к следующему исследованию {admission}.', '',
        'Новых обучений и обновлений оптимизатора — 0. Сравниваются девять '
        'сохранённых состояний: GroupCNN, свободная Untied24 и Plain12 C4 × '
        'три seed 5701/5721/5757. Везде бюджет обучения 27 масок и 1152 слабых '
        'классовых метки; дополнительный компонент обучения одинаков.', '',
        '## 1. AP сегментации без выбора порога', '',
        'Используется non-interpolated average precision: ступенчато взвешенная '
        'точность по приращениям полноты, а не площадь трапециями под PR-кривой. '
        'Метрика сначала вычисляется по всем пикселям выбранных фрагментов '
        'каждого кадра, затем усредняется по 40 кадрам и трём seed. '
        'Она оценивает разделение металла и фона, а не девять классов геометрии. '
        'Выбор порога 0,5 или VAL-порога на неё не влияет.', '',
        '| Модель | Среднее покадровое AP, % | Общее пиксельное AP, % |',
        '|---|---:|---:|']
    for arch in ARCHS:
        cell = result['cells'][arch]
        lines.append(f"| {NAMES[arch]} | {percent(cell['mean_frame_AP'], 4)} | {percent(cell['pooled_pixel_AP'], 4)} |")
    lines += ['', 'По покадровому AP GroupCNN превосходит Untied24, но уступает '
        'узкому контролю Plain12 C4. Поэтому превосходство над всеми контролями '
        'по качеству ранжирования пикселей не заявляется.', '',
        'Общее пиксельное AP приведено только как дополнительная диагностика. '
        'Оно может иначе ранжировать сети из-за объединения кадров с разными '
        'распределениями яркости и долями металла. Основное AP даёт кадрам равные веса.', '',
        '![Среднее покадровое AP](figures/calibration_dev_average_precision.png)', '',
        'Для сопоставления отдельно сохранено AP прежней VAL (695 фрагментов двух '
        'других кадров). Абсолютные оценки разных наборов нельзя трактовать как '
        'парный прирост: распределение и доля металла различаются.', '',
        '| Модель | AP прежней VAL, среднее кадров и seed, % |', '|---|---:|']
    for i, arch in enumerate(ARCHS):
        lines.append(f"| {NAMES[arch]} | {percent(float(ap['old_VAL_mean_frame_ap'][i].mean()), 4)} |")
    lines += ['', '## 2. Калибровка с разделением по кадрам', '',
        '40 кадров разделены хеш-ранжированием на пять групп по восемь. '
        'Для каждого цикла порог выбирается на 32 остальных кадрах, затем '
        'используется на восьми отложенных. Каждый фрагмент получает отложенное '
        'предсказание ровно один раз. Кадры калибровки и оценки в одном цикле '
        'не пересекаются; веса моделей остаются неизменными.', '',
        'На каждом кадре заранее выбираются 16 калибровочных фрагментов по хешу '
        'интенсивностей. В одном цикле используются 512 классовых меток. '
        'Всего во всех пяти циклах задействованы 640 уникальных меток, общих '
        'для всех девяти состояний моделей. Эти два числа нельзя смешивать: '
        '512 — бюджет одного цикла, 640 — размер объединения.', '',
        'Порог каждого состояния выбирается на фиксированной сетке '
        '0,10…0,90 с шагом 0,01: максимум Macro-F1 девяти классов, затем '
        'близость к 0,5, затем меньший порог. Пиксельная маска и IoU '
        'в выборе порога не участвуют. Если какой-то класс отсутствует '
        'в фиксированных 512 метках, набор не заменяется; покрытие записано ниже.', '',
        'OOF Macro-F1 считается отдельно для каждого seed по всем его отложенным '
        'предсказаниям, затем усредняется по seed. Confusion разных seed '
        'не объединяются. IoU отдельно измеряет качество маски металла.', '',
        '| Модель | OOF Macro-F1, % | OOF IoU, % | F1 при 0,5, % | IoU при 0,5, % | F1 старых VAL-порогов, % | IoU старых VAL-порогов, % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arch in ARCHS:
        cell = result['cells'][arch]
        values = [cell[key] for key in ['OOF_F1', 'OOF_IoU', 'fixed05_F1', 'fixed05_IoU',
                                        'old_VAL_threshold_F1', 'old_VAL_threshold_IoU']]
        lines.append('| ' + NAMES[arch] + ' | ' + ' | '.join(percent(value) for value in values) + ' |')
    lines += ['', '![Сравнение калибровок](figures/calibration_dev_quality.png)', '',
        '| Разность Group−контроль | ΔAP, п.п. | ΔOOF F1, п.п. | ΔOOF IoU, п.п. |',
        '|---|---:|---:|---:|']
    for key, name in [('group_minus_untied', 'Untied24'), ('group_minus_narrow', 'Plain12 C4')]:
        row = result['contrasts'][key]['mean']
        lines.append(f"| {name} | {pp(row['mean_frame_AP'], 6)} | {pp(row['OOF_F1'])} | {pp(row['OOF_IoU'])} |")
    lines += ['', '## Все девять состояний', '',
        '| Модель | Seed | Покадровое AP, % | OOF F1, % | OOF IoU, % | F1 при 0,5, % | Пороги пяти циклов |',
        '|---|---:|---:|---:|---:|---:|---|']
    for arch in ARCHS:
        for row in result['cells'][arch]['per_seed']:
            thresholds = ', '.join(f'{value:.2f}' for value in row['thresholds'])
            lines.append(f"| {NAMES[arch]} | {row['seed']} | {percent(row['mean_frame_AP'], 4)} | "
                f"{percent(row['OOF_F1'])} | {percent(row['OOF_IoU'])} | {percent(row['fixed05_F1'])} | {thresholds} |")
    lines += ['', '## Покадровые циклы и покрытие классов', '',
        '| Цикл | Отложенные кадры | ΔF1 Group−Untied, п.п. | ΔIoU Group−Untied, п.п. | Классы в 512 метках: количества 0…8 |',
        '|---|---|---:|---:|---|']
    fold_effects = result['contrasts']['group_minus_untied']['per_fold']
    for fold, fold_effect in enumerate(fold_effects):
        heldout = sorted(int(frame) for frame, value in result['frame_folds'].items() if value == fold)
        counts = ', '.join(str(value) for value in result['calibration_class_counts'][fold])
        lines.append(f"| {fold + 1} | {', '.join(map(str, heldout))} | {pp(fold_effect['F1'])} | {pp(fold_effect['IoU'])} | {counts} |")
    lines += ['', '## Заранее установленный допуск', '',
        'Этот допуск определяет, есть ли основание планировать отдельную '
        'проверку на ещё не использованных кадрах. Он не является статистическим '
        'подтверждением в этом опыте.', '',
        '| Условие Group−Untied | Исход |', '|---|---|']
    guard_names = {'mean_frame_AP': 'Среднее покадровое AP строго выше',
        'OOF_F1': 'Средний OOF ΔF1 ≥ +0,5 п.п.',
        'OOF_IoU': 'Средний OOF ΔIoU ≥ −0,2 п.п.',
        'each_fold_F1': 'ΔF1 каждого цикла ≥ −1 п.п.'}
    for key, value in eligibility['guards'].items():
        lines.append(f"| {guard_names[key]} | {'PASS' if value else 'FAIL'} |")
    lines += ['', f'Итоговый описательный допуск: {admission}. Все ограничения '
        'применяются одновременно; метрики не выбираются после получения исхода.', '',
        '## Проверка и границы', '',
        f"Независимый аудит — {audit['checks_passed']}/{audit['checks_total']}. "
        'Перед построением отчёта проверены хеши аудированных результатов, '
        'данных, вероятностей, калибровочных таблиц и фиксации источников.', '',
        'Пять калибровочных обучающих частей пересекаются. По ним не строятся '
        'независимые доверительные интервалы и p-value; пиксели и фрагменты '
        'не считаются независимыми повторениями эксперимента. AP относится '
        'к сегментации, а не к Macro-F1 классификации.', '',
        'Данные состоят из двух прежде оценённых наборов: 4096 фрагментов '
        '16 кадров и 6144 фрагментов 24 кадров. Их повторное использование '
        'явно объявлено разработкой новой калибровки. Прежние результаты '
        'сохраняют исторический статус своих протоколов; этот опыт не является '
        'новым подтверждением. Для него потребуется другой, ещё не использованный '
        'набор с заранее зафиксированными моделями и правилом калибровки.', '',
        'Все кадры относятся к одной ИС и металлизации M2. Девять классов '
        'вычислены из SVG-масок геометрическим правилом; это не экспертные '
        'типы производственных дефектов или электрические измерения. '
        'У Group и Untied одинаковые ширина и MAC, но разное число свободных '
        'параметров. Известная C4-конструкция не объявляется новым оператором.', '',
        '[Протокол](PROTOCOL.md) · [Конфигурация](CONFIG.json) · '
        '[Фиксация](LOCK.json) · [Источники](SOURCES.json) · '
        '[Предсказания](predict.py) · [Анализ](analysis.py) · '
        '[Результат](RESULT.json) · [Аудит](AUDIT.json) · '
        '[Предыдущая архитектурная проверка](../sem_arch_untied_control_20261103/REPORT.md).', '',
        'Перестроение отчёта без новых предсказаний:', '',
        '```bash', '.venv/bin/python -B gipoteza/sem_arch_calibration_dev_20261103/report.py', '```', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    (ROOT / 'README.md').write_text('# AP и покадровая калибровка фиксированных CNN\n\n'
        '40 ранее оценённых кадров явно используются как данные разработки. '
        'Новых обучений нет; нового закрытого подтверждения нет.\n\n'
        f"Group−Untied: ΔAP {pp(effect['mean_frame_AP'], 6)} п.п.; "
        f"ΔOOF F1 {pp(effect['OOF_F1'])} п.п.; ΔOOF IoU {pp(effect['OOF_IoU'])} п.п. "
        f'Описательный допуск к следующему исследованию {admission}.\n\n'
        'Калибровка: 512 классовых меток на цикл, 640 уникальных меток '
        'в объединении пяти циклов, одинаковых для всех девяти моделей.\n\n'
        '[Отчёт и графики](REPORT.md) · [Результаты](RESULT.json) · '
        '[Аудит](AUDIT.json) · [Анализ](analysis.py) · [Протокол](PROTOCOL.md).\n')
    inputs = ['RESULT.json', 'AUDIT.json', 'PRAUC.npz', 'CALIBRATION.npz', 'PREDICTIONS.npz',
        'DATA.npz', 'SOURCES.json', 'LOCK.json', 'INFERENCE.json', 'ANALYSIS_FROZEN.json',
        'CONFIG.json', 'PROTOCOL.md', 'common.py', 'predict.py', 'analysis.py', 'audit.py', 'report.py']
    outputs = ['REPORT.md', 'README.md', 'figures/calibration_dev_average_precision.png',
               'figures/calibration_dev_quality.png']
    manifest = dict(scientific_inputs_sha256={name: sha(ROOT / name) for name in inputs},
        outputs_sha256={name: sha(ROOT / name) for name in outputs},
        scope='REUSED_DEVELOPMENT_AP_CALIBRATION_REPORT_NO_NEW_CONFIRMATION_OR_FIT')
    (ROOT / 'REPORT_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(report=str(ROOT / 'REPORT.md'), next_study_eligibility=eligibility['passed'],
                         audit_checks=audit['checks_total'], new_confirmation=False)), flush=True)


if __name__ == '__main__': main()
