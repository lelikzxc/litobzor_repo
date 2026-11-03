"""Describe a frozen weight-sharing ablation; no fits or target-data reads."""
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
ARMS = ['group6', 'untied24_single', 'plain12_C4pooled']
SEEDS = [5701, 5721, 5757]
NAMES = {'group6': 'GroupCNN: связанные веса',
         'untied24_single': 'CNN24: независимые веса',
         'plain12_C4pooled': 'CNN12: четыре вида C4'}
SHORT = {'group6': 'GroupCNN', 'untied24_single': 'Untied24',
         'plain12_C4pooled': 'Plain12 C4'}
PARAMETERS = {'group6': 2671, 'untied24_single': 10681, 'plain12_C4pooled': 2749}
MACS = {'group6': 1527552, 'untied24_single': 1527552, 'plain12_C4pooled': 1562112}
COLORS = {'group6': '#197d86', 'untied24_single': '#ad6735', 'plain12_C4pooled': '#718291'}


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def percent(value): return f'{100 * value:.3f}'
def pp(value): return f'{100 * value:+.3f}'


def table_rows(row):
    return row['all_thresholds'] if 'all_thresholds' in row else row['primary_fine']['all_thresholds']


def validate():
    result, audit = read(ROOT/'RESULT.json'), read(ROOT/'AUDIT.json')
    assert result['status'].startswith('COMPLETE')
    assert len(result['rows']) == 9 and len(result['fits']) == 3
    assert audit['status'].startswith('PASS') and all(audit['checks'].values())
    assert audit['checks_passed'] == audit['checks_total']
    for name, expected in audit['input_sha256'].items():
        path = Path(name) if Path(name).is_absolute() else REPO/name
        assert sha(path) == expected, f'Stale independent audit input: {name}'
    for name in ['RESULT.json', 'LOCK.json']:
        assert str((ROOT/name).relative_to(REPO)) in audit['input_sha256']
    expected = {(arch, seed) for arch in ARMS for seed in SEEDS}
    actual = {(row['architecture'], row['seed']) for row in result['rows']}
    assert actual == expected and all(row['budget'] == 27 for row in result['rows'])
    thresholds = [round(.1+.01*i, 2) for i in range(81)]
    for row in result['rows']:
        curve = table_rows(row)
        assert [score['threshold'] for score in curve] == thresholds
        chosen = max(curve, key=lambda v: (v['mask_F1'], v['IoU'],
                                          -abs(v['threshold']-.5), -v['threshold']))
        assert row['selected'] == chosen
        assert row['fixed05'] == curve[40]
        for file_key, hash_key in [('model_file', 'model_sha256'),
                                   ('probability_file', 'probability_sha256')]:
            assert sha(REPO/row[file_key]) == row[hash_key]
    return result, audit


def figures(result):
    folder = ROOT/'figures'
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), layout='constrained')
    for ax, metric, title in zip(axes, ['mask_F1', 'IoU'],
                                  ['Macro-F1 классификации, %', 'IoU сегментации, %']):
        for position, arch in enumerate(ARMS):
            values = [100*row['selected'][metric] for row in result['rows']
                      if row['architecture'] == arch]
            assert len(values) == 3
            mean = float(np.mean(values))
            ax.bar(position, mean, color=COLORS[arch], width=.62, alpha=.86)
            ax.scatter(position+np.array([-.12, 0, .12]), values, color=COLORS[arch],
                       edgecolor='white', linewidth=.7, s=33, zorder=3)
            ax.annotate(f'{mean:.2f}', (position, mean), xytext=(0, 7),
                        textcoords='offset points', ha='center', fontsize=10)
        ax.set_xticks(range(3), [SHORT[arch] for arch in ARMS])
        ax.set_ylim(0, 105)
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Прежняя VAL: 27 масок, общий вспомогательный надзор\n'
                 'Столбцы — среднее; точки — три отдельные инициализации', fontsize=11)
    fig.savefig(folder/'untied_control_quality.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout='constrained')
    for ax, metric, title in zip(axes, ['mask_F1', 'IoU'],
                                  ['Macro-F1 классификации, %', 'IoU сегментации, %']):
        for arch in ARMS:
            selected = [row for row in result['rows'] if row['architecture'] == arch]
            curves = np.array([[100*score[metric] for score in table_rows(row)] for row in selected])
            thresholds = np.array([score['threshold'] for score in table_rows(selected[0])])
            ax.plot(thresholds, curves.mean(0), color=COLORS[arch], label=SHORT[arch], linewidth=2)
        ax.axvline(.5, color='#454545', linestyle='--', linewidth=1, alpha=.7)
        ax.set_xlabel('Порог бинаризации маски')
        ax.set_ylabel(title)
        ax.set_xlim(.1, .9)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle('Чувствительность к порогу: среднее по трём seed на прежней VAL\n'
                 'Описательная диагностика, без смены основного критерия', fontsize=11)
    fig.savefig(folder/'untied_control_thresholds.png', dpi=180)
    plt.close(fig)


def means(rows, mode):
    return {arch: {metric: float(np.mean([row[mode][metric] for row in rows
                                         if row['architecture'] == arch]))
                   for metric in ['mask_F1', 'IoU']} for arch in ARMS}


def contrast_rows(summary):
    lines = ['| Разность | ΔMacro-F1, п.п. | ΔIoU, п.п. |', '|---|---:|---:|']
    for arch in ['untied24_single', 'plain12_C4pooled']:
        f1 = summary['group6']['mask_F1']-summary[arch]['mask_F1']
        iou = summary['group6']['IoU']-summary[arch]['IoU']
        lines.append(f'| GroupCNN − {SHORT[arch]} | {pp(f1)} | {pp(iou)} |')
    return '\n'.join(lines)


def main():
    result, audit = validate()
    figures(result)
    primary, fixed = means(result['rows'], 'selected'), means(result['rows'], 'fixed05')
    gate = result['primary_development_gate']
    admission = result['confirmation_admission']
    passed = bool(gate['passed'])
    conclusion = 'пройден' if passed else 'не пройден'
    admission_conclusion = 'пройден' if admission['passed'] else 'не пройден'
    delta = primary['group6']['mask_F1']-primary['untied24_single']['mask_F1']
    delta_iou = primary['group6']['IoU']-primary['untied24_single']['IoU']
    lines = ['# Контроль связывания весов внутри CNN', '',
        f'3 ноября 2026 года. Заданный описательный критерий GroupCNN против '
        f'Untied24 {conclusion}: ΔMacro-F1 {pp(delta)} п.п., ΔIoU {pp(delta_iou)} п.п. '
        'Выполнены три новых обучения контроля с независимыми весами; '
        'шесть прежних состояний GroupCNN и Plain12 C4 использованы повторно. '
        f'Отдельный допуск к закрытому подтверждению {admission_conclusion}.', '',
        'Это исследование механизма на прежней многократно использованной VAL: '
        '695 фрагментов двух кадров. Выбор порога и оценка производятся на этой '
        'же VAL. Результат является описательным и не заменяет закрытого '
        'подтверждения. Прежние закрытые кадры и их предсказания в этот опыт '
        'не включались.', '',
        '## Что именно изменено', '',
        'Новый Untied24 имеет те же 24 канала, последовательность свёрток '
        '3×3 с dilation 1/2/4, GELU и выход 1×1, что материализованная GroupCNN. '
        'Используется один прямой проход без внешнего усреднения поворотов. '
        'Его начальные коэффициенты получены развёртыванием архивного начального '
        'состояния GroupCNN, до прежнего обучения. Численное совпадение '
        'стартовой функции проверено до запуска.', '',
        'После развёртывания все коэффициенты Untied24 обучаются независимо, '
        'включая первоначально совпадавшие повёрнутые фильтры и коэффициенты '
        'выходного слоя. GroupCNN сохраняет связывание. Это сравнение при '
        'одинаковой ширине и MAC; число независимых параметров различается '
        'как следствие снятия ограничений. Оно не является сравнением сетей '
        'с одинаковым числом свободных параметров.', '',
        '| Модель | Свободных параметров | MAC / фрагмент | Маршрут |',
        '|---|---:|---:|---|']
    for arch in ARMS:
        route = '4 обратно совмещённых поворота' if arch == 'plain12_C4pooled' else '1 прямой проход'
        lines.append(f'| {NAMES[arch]} | {PARAMETERS[arch]} | {MACS[arch]} | {route} |')
    lines += ['',
        'Прежний узкий Plain12 C4 сохранён как отдельный контроль при близком '
        'числе параметров и вычислений. Он обучался через четырёхвидовое '
        'усреднение вероятностей, а не только получил его при оценке.', '',
        '## Общие условия обучения', '',
        'Бюджет — 27 dense-масок, по три на геометрический класс; 1 152 weak-примера '
        'имеют только классовую метку. Seed: 5701, 5721, 5757. Во всех моделях '
        'прежний вспомогательный надзор одинаков: 400 шагов BCE по маскам, затем '
        '1 600 шагов BCE и неизменённой geometry_loss с весом 1. '
        'AdamW, lr=0,0005, weight decay=0,0001, dense batch 8 и weak batch 8. '
        'Последовательности батчей совпадают с соответствующими прежними Group-fit.', '',
        'Начальная функция совпадает, но параметризация и последующая оптимизация '
        'различаются: у одного оператора коэффициенты связаны, у другого свободны. '
        'Веса прежних обученных сетей не использовались для инициализации контроля. '
        'Три новых обучения выполнены полностью независимо от промежуточных метрик.', '',
        '## Результаты при выбранных VAL-порогах', '',
        'Для каждого состояния сохранена общая сетка 81 значения 0,10…0,90 '
        'с шагом 0,01 и прежнее правило выбора: Macro-F1, IoU, близость к 0,5, '
        'меньший порог. Macro-F1 относится к девяти классам из бинарной маски, '
        'IoU — к сегментации металла. Среднее по трём seed не является ансамблем.', '',
        '| Модель | Macro-F1, % | IoU, % |', '|---|---:|---:|']
    for arch in ARMS:
        lines.append(f"| {NAMES[arch]} | {percent(primary[arch]['mask_F1'])} | {percent(primary[arch]['IoU'])} |")
    lines += ['', contrast_rows(primary), '',
              '![Три модели и отдельные инициализации](figures/untied_control_quality.png)', '',
              '## Заранее установленный критерий', '',
              '| Условие | Исход |', '|---|---|']
    labels = {'mean_F1': 'Средний ΔMacro-F1 Group−Untied ≥ +0,5 п.п.',
              'seed_F1': 'ΔMacro-F1 каждого seed ≥ −1 п.п.',
              'per_seed_F1': 'ΔMacro-F1 каждого seed ≥ −1 п.п.',
              'mean_IoU': 'Средний ΔIoU Group−Untied ≥ −0,2 п.п.',
              'same_MACs': 'MAC Group и Untied совпадают',
              'matched_MACs': 'MAC Group и Untied совпадают',
              'initial_function_match': 'Начальная функция совпадает с установленным допуском',
              'initial_parity': 'Начальная функция совпадает с установленным допуском'}
    for key, value in gate['guards'].items():
        lines.append(f"| {labels.get(key, key)} | {'PASS' if value else 'FAIL'} |")
    lines += ['',
        'Этот gate относится к прежней VAL. Его прохождение само по себе '
        'не доказывает перенос или статистически значимое преимущество. '
        'Непрохождение сохраняется и не заменяется более удобным порогом.', '',
        '## Отдельный допуск к закрытому подтверждению', '',
        f'Дополнительный фильтр {admission_conclusion}. Для перехода к новым '
        'кадрам одновременно требуются основной gate, средний ΔF1 Group−Untied '
        'при общем пороге 0,5 ≥0 и средний ΔIoU при этом пороге ≥−0,2 п.п. '
        'Это заранее установленное дополнительное ограничение, которое может '
        'закрыть допуск, но не может спасти основной отрицательный результат.', '',
        '| Дополнительное условие | Исход |', '|---|---|']
    admission_labels = {'primary_gate': 'Основной описательный критерий пройден',
                        'fixed05_F1': 'Средний ΔF1 при общем пороге 0,5 ≥0',
                        'fixed05_IoU': 'Средний ΔIoU при общем пороге 0,5 ≥−0,2 п.п.'}
    for key, value in admission['guards'].items():
        lines.append(f"| {admission_labels[key]} | {'PASS' if value else 'FAIL'} |")
    lines += ['',
        'При отсутствии допуска новые кадры для этого варианта не загружаются. '
        'Направление Group−Untied не меняется по результатам на победившую модель. '
        'При прохождении допуска следующий этап требует отдельного протокола '
        'до просмотра ещё не использованных кадров.', '',
        '## Чувствительность к порогу', '',
        'Общий порог 0,5 и полные кривые — описательная диагностика. '
        'Они показывают зависимость результата от калибровки и не заменяют '
        'первичный критерий; значения при 0,5 также входят в отдельный '
        'заранее заданный фильтр допуска выше.', '',
        contrast_rows(fixed), '',
        '![Кривые Macro-F1 и IoU по 81 порогу](figures/untied_control_thresholds.png)', '',
        'Кривые усредняют три значения качества при каждом общем пороге. '
        'Это отличается от среднего качества после выбора индивидуального '
        'лучшего порога каждого состояния.', '',
        '## Все девять состояний', '',
        '| Модель | Seed | VAL-порог | Macro-F1, % | IoU, % | F1 при 0,5, % | IoU при 0,5, % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in sorted(result['rows'], key=lambda v: (ARMS.index(v['architecture']), v['seed'])):
        score, half = row['selected'], row['fixed05']
        lines.append(f"| {SHORT[row['architecture']]} | {row['seed']} | {score['threshold']:.2f} | "
                     f"{percent(score['mask_F1'])} | {percent(score['IoU'])} | "
                     f"{percent(half['mask_F1'])} | {percent(half['IoU'])} |")
    lines += ['', '## Проверка и воспроизведение', '',
        f"Независимый аудит — {audit['checks_passed']}/{audit['checks_total']}. "
        'Отчёт проверяет соответствие аудированных входов, текущего RESULT и '
        'LOCK перед построением графиков. Сохранены начальные/конечные веса, '
        'пробы градиента, вероятности, все пороги, маски, классы и прогресс.', '',
        'Данные — SEM металлизации M2 одной физической ИС. Классы получены '
        'из SVG-масок геометрическим правилом; они не являются экспертными '
        'типами производственных дефектов или электрическими измерениями. '
        'Межчиповый перенос этим опытом не проверялся. Известная конструкция '
        'C4-свёртки не заявляется новым оператором.', '',
        '[Протокол](PROTOCOL.md) · [Конфигурация](CONFIG.json) · '
        '[Фиксация](LOCK.json) · [Обучение](runner.py) · [Результат](RESULT.json) · '
        '[Аудит](AUDIT.json) · [Прежняя архитектура](../sem_arch_groupcnn_20261102/model.py) · '
        '[Зафиксированный основной вариант](../sem_model_freeze_20261103/MODEL_CARD.md).', '',
        'Повторное построение отчёта без обучения:', '', '```bash',
        '.venv/bin/python -B gipoteza/sem_arch_untied_control_20261103/report.py', '```', '']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))
    (ROOT/'README.md').write_text('# Контроль связывания весов внутри CNN\n\n'
        f'Описательный критерий Group−Untied {conclusion}: ΔMacro-F1 {pp(delta)} п.п. '
        'Три новых обучения при 27 масках; шесть прежних состояний использованы повторно. '
        f'Отдельный допуск к закрытой проверке {admission_conclusion}.\n\n'
        '[Отчёт и графики](REPORT.md) · [Результат](RESULT.json) · [Аудит](AUDIT.json) · '
        '[Протокол](PROTOCOL.md) · [Обучение](runner.py).\n\n'
        'Одинаковые начальная функция, ширина и MAC; 2 671 свободный параметр '
        'у Group против 10 681 у Untied. Прежняя адаптивная VAL, без нового '
        'независимого подтверждения. Кривые описательные; показатели общего порога '
        '0,5 дополнительно ограничивают допуск и не могут спасти основной исход.\n')
    names = ['RESULT.json', 'AUDIT.json', 'LOCK.json', 'PREFLIGHT.json', 'PREFLIGHT_AUDIT.json',
             'CONFIG.json', 'PROTOCOL.md', 'common.py', 'runner.py', 'audit.py', 'report.py']
    inputs = [ROOT/name for name in names if (ROOT/name).is_file()]
    outputs = [ROOT/name for name in ['REPORT.md', 'README.md',
               'figures/untied_control_quality.png', 'figures/untied_control_thresholds.png']]
    manifest = dict(scientific_inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs},
                    outputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in outputs},
                    scope='DESCRIPTIVE_OLD_VAL_WEIGHT_SHARING_CONTROL_NO_FIT_OR_NEW_TEST')
    (ROOT/'REPORT_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(report=str(ROOT/'REPORT.md'), gate_passed=passed,
                          new_fits=3, reused_fits=6, audit_checks=audit['checks_total'])))


if __name__ == '__main__': main()
