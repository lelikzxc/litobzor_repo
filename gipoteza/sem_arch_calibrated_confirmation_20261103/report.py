"""Render the audited final-calibration confirmation without inference or fitting."""
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
    assert result['status'] in ['PASS_CONFIRMATION', 'FAIL_CONFIRMATION']
    assert result['fit_count'] == result['optimizer_updates'] == 0
    assert result['source_model_count'] == result['evaluated_routes'] == len(result['rows']) == 9
    assert result['test_examples'] == 6144 and len(result['test_frames']) == 24
    assert audit['status'].startswith('PASS') and all(audit['checks'].values())
    assert audit['checks_passed'] == audit['checks_total']
    for name, expected in audit['input_sha256'].items():
        path = Path(name) if Path(name).is_absolute() else REPO / name
        assert sha(path) == expected, f'Changed audited input: {name}'
    required = ['RESULT.json', 'SELECTION.json', 'MODEL_LOCK.json', 'FINAL_CALIBRATION.json',
                'FINAL_CALIBRATION.npz', 'EVAL.npz', 'PREDICTIONS.npz', 'BOOTSTRAP.npz']
    assert {str((ROOT / name).relative_to(REPO)) for name in required} <= audit['input_sha256'].keys()
    for name, field in [('SELECTION.json', 'selection_sha256'),
                        ('EVALUATION_FROZEN.json', 'evaluation_frozen_sha256'),
                        ('FINAL_CALIBRATION.json', 'final_calibration_sha256'),
                        ('PREDICTIONS.npz', 'predictions_sha256'), ('BOOTSTRAP.npz', 'bootstrap_sha256')]:
        assert sha(ROOT / name) == result[field]
    primary = result['analyses']['final_calibrated']
    assert result['confirmed'] == primary['primary_goal']['confirmed']
    assert result['analyses']['fixed05']['primary_goal']['confirmed'] is False
    assert not any(key.startswith('CI') for key in result['analyses']['fixed05']['primary_goal'])
    selection, calibration, plan, index = [read(ROOT / name) for name in
        ['SELECTION.json', 'FINAL_CALIBRATION.json', 'SOURCE_PLAN.json', 'EVAL_INDEX.json']]
    assert len(selection['rows']) == 9 and len(calibration['data_indices']) == calibration['calibration_examples'] == 512
    assert len(calibration['frame_ids']) == calibration['calibration_frames'] == 32
    assert not calibration['masks_or_IoU_used_for_selection']
    assert plan['excluded_frame_count'] == 134 and len(plan['prior_manifests_sha256']) == 8
    assert plan['test_frames'] == result['test_frames'] and index['examples'] == 6144 and all(index['gates'].values())
    assert index['EVAL_sha256'] == sha(ROOT / 'EVAL.npz')
    return result, audit, selection, calibration, plan, index


def figures(result):
    output = ROOT / 'figures'; output.mkdir(exist_ok=True)
    primary = result['analyses']['final_calibrated']; goal = primary['primary_goal']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    color = '#197d86' if goal['confirmed'] else '#b66530'
    for ax, metric, interval, title in zip(axes, ['F1_delta', 'AP_delta'],
            ['CI97_5_F1', 'CI97_5_AP'], ['Macro-F1 классификации', 'AP сегментации']):
        value = 100 * goal[metric]; low, high = np.array(goal[interval]) * 100
        ax.hlines(0, low, high, color=color, linewidth=2.2)
        ax.vlines([low, high], -.055, .055, color=color, linewidth=2.2)
        ax.scatter([value], [0], s=60, color=color, zorder=3)
        ax.axvline(0, color='#45515a', linewidth=1.1, label='Нижняя граница должна быть > 0')
        bounds = [low, high, value, 0.]
        if metric == 'F1_delta':
            ax.axvline(.5, color='#b66530', linestyle='--', linewidth=1.1,
                       label='Порог наблюдаемого прироста +0,5 п.п.')
            bounds.append(.5)
        ax.annotate(f'{value:+.3f}; 97,5% [{low:+.3f}; {high:+.3f}]',
                    (value, 0), xytext=(0, 20), textcoords='offset points', ha='center', fontsize=9)
        pad = max(.06, .18 * (max(bounds) - min(bounds)))
        ax.set_xlim(min(bounds) - pad, max(bounds) + pad)
        ax.set_ylim(-.52, .5); ax.set_yticks([0], ['Group − Untied'])
        ax.set_title(title); ax.set_xlabel('Разность качества, процентные пункты')
        ax.grid(axis='x', alpha=.2); ax.legend(loc='lower left', fontsize=7)
    fig.suptitle('Две обязательные части основной гипотезы: 24 новых кадра одной ИС\n'
                 'Парные 97,5%-интервалы; поправка на два показателя', fontsize=11)
    fig.savefig(output / 'final_calibration_intervals.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout='constrained')
    for ax, metric, title in zip(axes, ['mean_frame_AP', 'mask_F1', 'IoU'],
            ['AP сегментации, %', 'Macro-F1 классификации, %', 'IoU сегментации, %']):
        all_values = []
        for i, architecture in enumerate(ARCHS):
            cell = primary['cells'][architecture]
            mean = 100 * cell['mean'][metric]
            values = np.array([100 * row[metric] for row in cell['per_seed']])
            all_values.extend(values.tolist())
            if metric == 'mean_frame_AP':
                ax.hlines(mean, i - .2, i + .2, color=COLORS[i], linewidth=3)
            else:
                ax.bar(i, mean, width=.62, color=COLORS[i], alpha=.8)
            ax.scatter(i + np.array([-.08, 0., .08]), values, color=COLORS[i],
                       edgecolor='white', linewidth=.7, s=25, zorder=3)
            ax.annotate(f'{mean:.4f}' if metric == 'mean_frame_AP' else f'{mean:.3f}',
                        (i, mean), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=8)
        ax.set_xticks(range(3), [NAMES[name] for name in ARCHS])
        if metric == 'mean_frame_AP':
            ax.set_ylim(max(0, min(all_values) - .2), min(100.02, max(all_values) + .3))
        else:
            ax.set_ylim(0, 105)
        ax.set_title(title); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Фиксированные модели и финальная калибровка: 6 144 новых фрагмента\n'
                 'Средние индивидуальных метрик; точки — три инициализации', fontsize=11)
    fig.savefig(output / 'final_calibration_quality.png', dpi=180)
    plt.close(fig)


def main():
    result, audit, selection, calibration, plan, index = validate()
    figures(result)
    primary = result['analyses']['final_calibrated']; goal = primary['primary_goal']
    decision = 'подтверждена' if result['confirmed'] else 'не подтверждена'
    data_audit = read(ROOT / 'DATA_AUDIT.json')
    f1_ci, ap_ci = goal['CI97_5_F1'], goal['CI97_5_AP']
    lines = ['# Закрытая проверка CNN после фиксированной классовой калибровки', '',
        f'3 ноября 2026 года. Основная составная гипотеза {decision}. '
        'Проверены 6 144 фрагмента из 24 новых кадров металлизации M2 одной ИС. '
        'GroupCNN сравнивается со свободной Untied24 при одинаковой ширине и MAC. '
        'Обязательны одновременно положительные AP и Macro-F1 с установленными ограничениями.', '',
        f"Group−Untied: ΔMacro-F1 {pp(goal['F1_delta'])} п.п., 97,5% интервал "
        f"[{pp(f1_ci[0])}; {pp(f1_ci[1])}] п.п.; ΔAP {pp(goal['AP_delta'], 6)} п.п., "
        f"97,5% интервал [{pp(ap_ci[0], 6)}; {pp(ap_ci[1], 6)}] п.п.; "
        f"ΔIoU {pp(goal['IoU_delta'])} п.п.", '',
        'Новых обучений CNN и обновлений оптимизатора — 0. Девять конечных весов '
        'и девять скалярных порогов закреплены до загрузки новых кадров. '
        'Пороги определены только по одним и тем же 512 прежним классовым меткам. '
        'Новые маски, классы и предсказания не участвовали в их подборе.', '',
        '## Основной исход и критерии', '',
        'Два первичных показателя — Macro-F1 девяти классов после фиксированной '
        'бинаризации маски и AP сегментации до выбора порога. '
        'Оба 97,5%-интервала должны иметь нижнюю границу строго выше нуля. '
        'Использована поправка Бонферрони на два показателя, с номинальным '
        'семейным уровнем 95%. Остальные условия применяются одновременно.', '',
        '| Условие | Исход |', '|---|---|']
    guard_names = dict(mean_F1='Наблюдаемый средний ΔF1 ≥ +0,5 п.п.',
        F1_CI_lower='Нижняя граница 97,5%-интервала ΔF1 строго >0',
        mean_AP='Наблюдаемый средний ΔAP строго >0',
        AP_CI_lower='Нижняя граница 97,5%-интервала ΔAP строго >0',
        seed_F1='ΔF1 каждой инициализации ≥−1 п.п.',
        IoU_guard='Средний ΔIoU ≥−0,2 п.п.', matched_MACs='MAC совпадает',
        development_eligible='Предшествующая разработка и её независимый аудит пройдены')
    for name, passed in goal['guards'].items():
        lines.append(f"| {guard_names[name]} | {'PASS' if passed else 'FAIL'} |")
    lines += ['', 'Порог +0,5 п.п. относится к наблюдаемой точечной разности F1. '
        'Нижняя граница >0 поддерживает положительный эффект в заданной схеме '
        'пересэмплирования, но не доказывает минимальный популяционный выигрыш '
        'не менее +0,5 п.п.', '',
        '![Два первичных интервала](figures/final_calibration_intervals.png)', '',
        '## Качество трёх фиксированных моделей', '',
        'AP — non-interpolated average precision, вычисленная по пикселям каждого '
        'кадра, затем усреднённая по кадрам и seed. Macro-F1 вычислена отдельно '
        'для каждой инициализации, затем усреднена. IoU — среднее по фрагментам. '
        'Это средние метрик отдельных сетей, а не результат ансамбля.', '',
        '| Модель | Покадровое AP, % | Macro-F1, % | IoU, % | Свободных параметров | MAC / фрагмент |',
        '|---|---:|---:|---:|---:|---:|']
    parameters = [2671, 10681, 2749]; macs = [1527552, 1527552, 1562112]
    for a, architecture in enumerate(ARCHS):
        cell = primary['cells'][architecture]['mean']
        lines.append(f"| {NAMES[architecture]} | {percent(cell['mean_frame_AP'], 4)} | "
            f"{percent(cell['mask_F1'])} | {percent(cell['IoU'])} | {parameters[a]} | {macs[a]} |")
    lines += ['', 'Узкая Plain12 C4 имеет более высокое AP, чем GroupCNN, '
        'хотя её Macro-F1 и IoU ниже. Основной подтверждённый исход относится '
        'к сравнению Group−Untied; превосходство над всеми контролями по каждой '
        'метрике не заявляется.', '',
        '![Качество трёх сетей](figures/final_calibration_quality.png)', '',
        'У Group и Untied совпадают фактическая ширина 24, одиночный маршрут '
        'исполнения, свёртки и дилатации. Начальная функция Untied получена '
        'развёртыванием архивных начальных весов Group, до обучения. '
        'Untied затем обучалась с независимыми коэффициентами, включая выходной слой. '
        'Число независимых параметров и геометрия оптимизации различаются; '
        'этот опыт не отделяет их от эффекта C4-связывания. Вспомогательное '
        'обучение одинаково у всех сетей: 27 полных масок и 1152 классовые метки.', '',
        '## Описательные сравнения', '',
        'Plain12 C4 — дополнительный контроль. Его оценки и общий порог 0,5 '
        'не имеют новых подтверждающих интервалов и не могут заменить основной '
        'исход Group−Untied. Общий порог применяется к тем же вероятностям без '
        'нового обучения.', '',
        '| Сравнение | ΔF1, п.п. | ΔAP, п.п. | ΔIoU, п.п. |', '|---|---:|---:|---:|']
    for comparison, title in [('group_minus_untied', 'Group−Untied'), ('group_minus_narrow', 'Group−Plain12 C4')]:
        contrast = primary['contrasts'][comparison]['mean']
        lines.append(f"| {title}, финальная калибровка | {pp(contrast['mask_F1'])} | "
                     f"{pp(contrast['mean_frame_AP'], 6)} | {pp(contrast['IoU'])} |")
    for comparison, title in [('group_minus_untied', 'Group−Untied'), ('group_minus_narrow', 'Group−Plain12 C4')]:
        contrast = result['analyses']['fixed05']['contrasts'][comparison]['mean']
        lines.append(f"| {title}, общий порог 0,5 | {pp(contrast['mask_F1'])} | "
                     f"{pp(contrast['mean_frame_AP'], 6)} | {pp(contrast['IoU'])} |")
    lines += ['', '## Все девять состояний', '',
        '| Модель | Seed | Фиксированный порог | AP, % | F1, % | IoU, % | F1 при 0,5, % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    locked = {(row['architecture'], row['seed']):row for row in selection['rows']}
    for architecture in ARCHS:
        per_seed = {row['seed']:row for row in primary['cells'][architecture]['per_seed']}
        for row in result['rows']:
            if row['architecture'] != architecture: continue
            seed = row['seed']; score = row['metrics']['final_calibrated']; state = per_seed[seed]
            assert score['threshold'] == locked[architecture, seed]['final_calibrated_threshold']
            lines.append(f"| {NAMES[architecture]} | {seed} | {score['threshold']:.2f} | "
                f"{percent(state['mean_frame_AP'], 4)} | {percent(score['mask_F1'])} | {percent(score['IoU'])} | "
                f"{percent(row['metrics']['fixed05']['mask_F1'])} |")
    lines += ['', '## Финальная калибровка до новых кадров', '',
        'Из 40 ранее просмотренных DEV-кадров выбраны 32 по заранее заданному '
        'SHA-ранжированию. В каждом взяты 16 фрагментов по неизменённому хеш-правилу '
        'исходных интенсивностей. Все девять состояний используют одинаковые '
        '512 классовых меток. Отбор не использует класс, маску или качество сети.', '',
        'Для каждой сети выбирается максимум Macro-F1 на общей сетке '
        '0,10…0,90 с шагом 0,01. При равенстве выбирается ближайший к 0,5 '
        'порог, затем меньший; сравнение расстояний выполняется в целых сотых. '
        'IoU и полные пиксельные маски не входят в выбор порога. '
        'Сохранены все 81 калибровочных оценки и выбранные пороги.', '',
        'Кадры калибровки: `'+str(calibration['frame_ids'])+'`.', '',
        'Количество классов 0…8 среди 512 меток: `'+str(calibration['class_counts'])+'`.', '',
        '## Новая выборка и происхождение', '',
        'Из 321 доступного кадра M2 исключены 134 кадра из прежних планов и данных. '
        'Оставшиеся ранжированы SHA-256 до загрузки; выбраны первые 24 без замены. '
        'По восьми прежним манифестам и внутри нового набора исключены точные '
        'D4-повторы исходных интенсивностей и бинарных масок.', '',
        'Фиксированные новые кадры: `'+str(plan['test_frames'])+'`.', '',
        'Подготовка прежняя: отступ 512 нативных пикселей, уменьшение 4×4, '
        'маска по доле металла ≥0,5, фрагменты 12×12 с шагом 6. '
        'Кандидаты используют маски для удаления однородных фрагментов и точных '
        'повторов. Последующее ранжирование 256 фрагментов на кадр использует '
        'только raw-D4 хеш интенсивностей, без маски, класса и предсказания. '
        'Требования покрытия проверены без замены кадров или пересэмплирования.', '',
        '| Геометрический класс | Фрагментов | Различных кадров |', '|---|---:|---:|']
    for cls, count in enumerate(index['class_counts']):
        lines.append(f"| {cls} | {count} | {len(index['class_frames'][str(cls)])} |")
    lines += ['', 'Остальные фрагменты того же TEST разделяют эти кадры и не являются '
        'ещё одним независимым подтверждением. Геометрическая регистрация полей '
        'в физических координатах микросхемы отдельно не проводилась.', '',
        '## Статистика и аудит', '',
        'Выполнены 10 000 парных crossed-bootstrap повторов с общими индексами '
        'трёх seed и 24 кадров для обеих первичных метрик и сетей. Для Macro-F1 '
        'confusion сначала суммируется по выбранным кадрам отдельно каждого '
        'выбранного seed, затем вычисляется F1 и усредняется по seed. '
        'AP сначала считается по каждому кадру/seed, затем усредняется по '
        'выбранным кадрам и seed. Confusion разных seed до F1 не объединяются.', '',
        'Центральные квантили 0,0125 и 0,9875 дают два 97,5%-интервала с '
        'поправкой на два первичных показателя. Это приблизительные интервалы '
        'в выбранной схеме пересэмплирования. Три инициализации ограничивают '
        'оценку изменчивости обучения; зависимые пиксели и фрагменты не считаются '
        'независимыми экспериментами.', '',
        f"Независимый аудит данных — {data_audit['checks_passed']}/{data_audit['checks_total']}; "
        f"итоговый аудит — {audit['checks_passed']}/{audit['checks_total']}. "
        'Перед построением отчёта проверены хеши текущих результатов, весов, '
        'калибровки, данных, вероятностей и bootstrap. '
        f"Оценка и статистика заняли {result['seconds']:.2f} с; "
        'это длительность прогона, а не измерение полной задержки приложения.', '',
        '## Границы результата', '',
        'Это новые поля одной физической ИС и одного слоя металлизации M2. '
        'Девять классов вычислены из SVG-масок геометрическим правилом. '
        'Они не являются экспертными типами производственных дефектов, '
        'электрическими измерениями или измеренным выходом годных. '
        'Перенос между изделиями не проверялся. Известная C4-конструкция '
        'не заявляется новым оператором; результат не устанавливает публикационную новизну.', '',
        'Прежние [проверка BCE без дополнительного надзора]'
        '(../sem_arch_bce_confirmation_20261103/REPORT.md), '
        '[архитектурный пилот с общим порогом 0,5]'
        '(../sem_arch_untied_control_20261103/REPORT.md) и '
        '[разработка покадровой калибровки]'
        '(../sem_arch_calibration_dev_20261103/REPORT.md) сохраняют свои исходы. '
        'Новая постановка с финальной калибровкой не переписывает их критерии '
        'и не превращает прежние кадры в новый тест.', '',
        '[Протокол](PROTOCOL.md) · [Конфигурация](CONFIG.json) · '
        '[Выбор и фиксация](selection.py) · [Финальная калибровка](FINAL_CALIBRATION.json) · '
        '[Веса и пороги](MODEL_LOCK.json) · [План данных](SOURCE_PLAN.json) · '
        '[Оценка](evaluate.py) · [Статистика](stats.py) · [Результат](RESULT.json) · '
        '[Аудит данных](DATA_AUDIT.json) · [Итоговый аудит](AUDIT.json).', '',
        'Повторное построение отчёта без обучения и предсказаний:', '',
        '```bash', '.venv/bin/python -B gipoteza/sem_arch_calibrated_confirmation_20261103/report.py', '```', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    (ROOT / 'README.md').write_text('# Закрытая проверка CNN с финальной классовой калибровкой\n\n'
        f'Составная гипотеза Group−Untied по AP и Macro-F1 {decision}. '
        '24 новых кадра, 6144 фрагмента, девять фиксированных состояний, '
        '512 прежних классовых меток для одинаковой финальной калибровки; '
        'новых обучений CNN нет.\n\n'
        f"ΔF1 {pp(goal['F1_delta'])} п.п., 97,5% [{pp(f1_ci[0])}; {pp(f1_ci[1])}] п.п.; "
        f"ΔAP {pp(goal['AP_delta'], 6)} п.п., 97,5% [{pp(ap_ci[0], 6)}; {pp(ap_ci[1], 6)}] п.п.\n\n"
        '[Отчёт и графики](REPORT.md) · [Результат](RESULT.json) · [Аудит](AUDIT.json) · '
        '[Протокол](PROTOCOL.md).\n\n'
        'Plain12 C4 и общий порог 0,5 — описательные контроли. '
        'Результат относится к геометрическим классам одной ИС.\n')
    inputs = ['RESULT.json', 'AUDIT.json', 'SELECTION.json', 'MODEL_LOCK.json', 'SOURCE_PLAN.json',
        'FINAL_CALIBRATION.json', 'FINAL_CALIBRATION.npz', 'CONFIG.json', 'PROTOCOL.md',
        'DATA_AUDIT.json', 'DATA_MANIFEST.json', 'EVAL_INDEX.json', 'EVAL.npz',
        'PREDICTIONS.npz', 'BOOTSTRAP.npz', 'EVALUATION_FROZEN.json',
        'OLD_VAL_PREFLIGHT.json', 'OLD_VAL_PREFLIGHT.npz', 'common.py', 'selection.py',
        'evaluate.py', 'prepare.py', 'stats.py', 'audit.py', 'report.py']
    inputs += [name for name in ['PRE_LOCK_AUDIT.json', 'PRE_DOWNLOAD_AUDIT.json'] if (ROOT / name).exists()]
    outputs = ['REPORT.md', 'README.md', 'figures/final_calibration_intervals.png', 'figures/final_calibration_quality.png']
    manifest = dict(scientific_inputs_sha256={name:sha(ROOT / name) for name in inputs},
        outputs_sha256={name:sha(ROOT / name) for name in outputs},
        scope='LOCKED_FINAL_CALIBRATION_NEW_FRAME_CONFIRMATION_NO_FIT')
    (ROOT / 'REPORT_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(report=str(ROOT / 'REPORT.md'), scientific_status=result['status'],
        confirmed=result['confirmed'], audit_checks=audit['checks_total'], inputs=len(inputs), outputs=len(outputs))), flush=True)


if __name__ == '__main__': main()
