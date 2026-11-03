"""Render the frozen, descriptive 2x2 ablation without changing its results."""
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
ARCHS = ['plain12_C4pooled', 'group6']
OBJECTIVES = ['BCE', 'BCE_geometry']
NAMES = {'plain12_C4pooled': 'Обычная CNN + C4', 'group6': 'Групповая CNN'}
REGIMES = {'BCE': 'BCE', 'BCE_geometry': 'BCE + дополнительный надзор'}
CONTRASTS = {
    'architecture_BCE': 'Архитектура при BCE: Group − Plain',
    'architecture_auxiliary': 'Архитектура при BCE + aux: Group − Plain',
    'auxiliary_group': 'Дополнительный надзор у Group',
    'auxiliary_plain': 'Дополнительный надзор у Plain',
    'interaction': 'Взаимодействие: Δархитектуры aux − Δархитектуры BCE',
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pp(value):
    return f'{100 * value:+.3f}'


def percent(value):
    return f'{100 * value:.3f}'


def contrast_table(summary):
    lines = ['| Сравнение | ΔMacro-F1, п.п. | ΔIoU, п.п. |',
             '|---|---:|---:|']
    for key, name in CONTRASTS.items():
        row = summary['contrasts'][key]['mean']
        lines.append(f"| {name} | {pp(row['mask_F1'])} | {pp(row['IoU'])} |")
    return '\n'.join(lines)


def figures(result):
    folder = ROOT / 'figures'
    folder.mkdir(exist_ok=True)
    colors = ['#6a7a89', '#197d86']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    for ax, metric, title in zip(axes, ['mask_F1', 'IoU'],
                                  ['Macro-F1 классификации, %', 'IoU сегментации, %']):
        for j, arch in enumerate(ARCHS):
            x = np.arange(2) + (j - .5) * .32
            means = [100 * result['primary']['cells'][obj][arch]['mean'][metric]
                     for obj in OBJECTIVES]
            ax.bar(x, means, width=.29, color=colors[j], label=NAMES[arch], alpha=.85)
            for k, obj in enumerate(OBJECTIVES):
                values = [100 * row['selected'][metric] for row in result['rows']
                          if row['objective'] == obj and row['architecture'] == arch]
                jitter = np.linspace(-.085, .085, len(values))
                ax.scatter(x[k] + jitter, values, color=colors[j], edgecolor='white',
                           linewidth=.5, s=23, zorder=3)
                ax.annotate(f'{means[k]:.2f}', (x[k], means[k]),
                            xytext=(0, 6), textcoords='offset points', ha='center', fontsize=9)
        ax.set_xticks([0, 1], ['BCE', 'BCE + aux'])
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle('Архитектура × дополнительное обучение: прежняя VAL\n'
                 'Столбцы — среднее; точки — шесть состояний budget × seed', fontsize=11)
    fig.savefig(folder / 'factorial_quality.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    labels = ['Архитектура при BCE', 'Архитектура при BCE + aux',
              'Доп. надзор: Group', 'Доп. надзор: Plain', 'Взаимодействие']
    for j, metric in enumerate(['mask_F1', 'IoU']):
        values = [100 * result['primary']['contrasts'][key]['mean'][metric]
                  for key in CONTRASTS]
        y = np.arange(5) + (j - .5) * .28
        ax.barh(y, values, height=.25, color=colors[j],
                label='Macro-F1' if metric == 'mask_F1' else 'IoU')
        for position, value in zip(y, values):
            ax.annotate(f'{value:+.2f}', (value, position),
                        xytext=(4 if value >= 0 else -4, 0), textcoords='offset points',
                        va='center', ha='left' if value >= 0 else 'right', fontsize=8)
    ax.axvline(0, color='black', linewidth=.8)
    ax.set_yticks(range(5), labels)
    ax.invert_yaxis()
    ax.set_xlabel('Разность качества, процентные пункты')
    ax.set_title('Описательные эффекты факторов на прежней VAL')
    ax.grid(axis='x', alpha=.2)
    ax.margins(x=.2)
    ax.legend()
    fig.savefig(folder / 'factorial_effects.png', dpi=180)
    plt.close(fig)


def main():
    result = read(ROOT / 'RESULT.json')
    assert result['status'] == 'COMPLETE_FACTORIAL_DEVELOPMENT'
    assert (result['new_fits'], result['inherited_fits'], len(result['rows'])) == (12, 12, 24)
    audit = read(ROOT / 'AUDIT.json')
    assert audit['status'].startswith('PASS_')
    assert audit['checks_passed'] == audit['checks_total']
    for name, expected in audit['input_sha256'].items():
        assert sha(REPO / name) == expected, f'Stale audit input: {name}'
    for name in ['RESULT.json', 'LOCK.json']:
        assert str((ROOT / name).relative_to(REPO)) in audit['input_sha256']
    figures(result)
    gate = result['primary_development_gate']
    effect = result['primary']['contrasts']['architecture_BCE']['mean']
    conclusion = ('Заданный описательный критерий архитектурного преимущества без '
                  'дополнительного надзора пройден.' if gate['passed'] else
                  'Заданный описательный критерий архитектурного преимущества без '
                  'дополнительного надзора не пройден.')
    lines = [
        '# Абляция архитектуры и дополнительного обучения', '',
        '3 ноября 2026 года. Все 12 новых обучений закончены на MPS; '
        '24 000 новых обновлений оптимизатора. Для режима с дополнительным '
        'надзором повторно использованы 12 прежних обучений без изменения весов.', '',
        f"{conclusion} При BCE средний ΔMacro-F1 Group−Plain равен {pp(effect['mask_F1'])} п.п., "
        f"ΔIoU — {pp(effect['IoU'])} п.п. Все исходы сохранены независимо от этого критерия.", '',
        'Это анализ компонентов на прежней, многократно использованной VAL '
        '(695 фрагментов двух кадров), с выбором порога и оценкой на этой же VAL. '
        'Он не является новым независимым подтверждением. Закрытые 16 кадров '
        'новым обучением и оценкой не читались.', '',
        '## Что сравнивается', '',
        'GroupCNN: 2 671 независимый параметр, 1 527 552 MAC на изображение. '
        'Контроль Plain12: 2 749 параметров, 1 562 112 MAC; он обучается и '
        'оценивается через среднее вероятностей четырёх совмещённых поворотов. '
        'У Group поворотные связи находятся внутри свёрточных слоёв.', '',
        'BCE использует только 18 или 27 полных масок. BCE + aux дополнительно '
        'использует прежнюю geometry_loss по 1 152 классовым меткам. Поэтому '
        'разность этих режимов оценивает весь пакет дополнительного надзора, '
        'а не только формулу лосса при одинаковой информации в метках. '
        'Внутри каждого режима обе архитектуры имеют одинаковый доступ к разметке.', '',
        'Три seed: 5701, 5721, 5757; два бюджета масок: 18 и 27. '
        'AdamW, 2 000 шагов, lr=0,0005, weight decay=0,0001, dense batch=8. '
        'Начальные веса и последовательности батчей внутри пар режимов совпадают. '
        'После первых 400 шагов у BCE-only сохраняется прямой проход dense8+weak8 '
        'для одинаковой формы вычислений, но слабые выходы имеют нулевой градиент. '
        'Сами слабые метки в BCE-only не используются.', '',
        '## Основные результаты', '',
        'Порог маски выбирается из общей фиксированной сетки 81 значения '
        '0,10…0,90 по прежнему правилу. Ниже — среднее индивидуальных метрик '
        'по трём seed и двум бюджетам; это не ансамбль моделей. '
        'Macro-F1 относится к девяти геометрическим классам из маски; IoU — '
        'к сегментации металла.', '',
        '| Режим | Модель | Macro-F1, % | IoU, % |', '|---|---|---:|---:|',
    ]
    for obj in OBJECTIVES:
        for arch in ARCHS:
            mean = result['primary']['cells'][obj][arch]['mean']
            lines.append(f"| {REGIMES[obj]} | {NAMES[arch]} | {percent(mean['mask_F1'])} | {percent(mean['IoU'])} |")
    positive_pairs = sum(row['mask_F1'] > 0 for row in
                         result['primary']['contrasts']['architecture_BCE']['per_budget_seed'])
    lines += ['', '![Четыре ячейки эксперимента](figures/factorial_quality.png)', '',
              contrast_table(result['primary']), '',
              f'При обычном BCE Macro-F1 Group выше контроля в {positive_pairs} из 6 '
              'пар бюджет × seed. Это поддерживает самостоятельную полезность '
              'архитектурного изменения на прежней VAL. Средний IoU при BCE '
              f"изменился на {pp(effect['IoU'])} п.п.; улучшения всех метрик одновременно нет.", '',
              'Дополнительный надзор в среднем не повысил Macro-F1 на этой VAL: '
              f"его эффект у Group {pp(result['primary']['contrasts']['auxiliary_group']['mean']['mask_F1'])} п.п., "
              f"у Plain {pp(result['primary']['contrasts']['auxiliary_plain']['mean']['mask_F1'])} п.п. "
              'Поэтому самостоятельная польза этого компонента здесь не показана. '
              'Прежняя закрытая проверка сравнивала архитектуры при включённом aux; '
              'более высокую VAL-оценку BCE нельзя автоматически переносить на закрытые кадры '
              'или объявлять независимым основанием для замены основного режима.', '',
              '![Эффекты факторов](figures/factorial_effects.png)', '',
              'Положительное взаимодействие означает, что архитектурная разность '
              'сильнее при дополнительном надзоре. Отрицательное — что она сильнее '
              'при BCE. Это описательная разность разностей, без утверждения '
              'статистической значимости.', '',
              '## Проверка заданного критерия', '',
              '| Условие | Исход |', '|---|---|']
    labels = {'mean_F1': 'Средний ΔF1 ≥ +0,5 п.п.',
              'budget_F1': 'ΔF1 каждого бюджета ≥ −0,25 п.п.',
              'mean_IoU': 'Средний ΔIoU ≥ −0,2 п.п.',
              'matched_parameters': 'Разница свободных параметров ≤ 3%',
              'matched_MACs': 'Разница MAC ≤ 3%'}
    for key, value in gate['guards'].items():
        lines.append(f"| {labels[key]} | {'PASS' if value else 'FAIL'} |")
    lines += ['', '## Чувствительность к порогу: общее значение 0,5', '',
              contrast_table(result['fixed05_diagnostic']), '',
              'Фиксированный порог — вторичная диагностика. Он не заменяет '
              'основной критерий и не используется для смены исхода.', '',
              '## Все 24 состояния', '',
              '| Режим | Модель | Масок | Seed | Порог | Macro-F1, % | IoU, % |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in sorted(result['rows'], key=lambda v: (v['objective'], v['architecture'], v['budget'], v['seed'])):
        selected = row['selected']
        lines.append(f"| {REGIMES[row['objective']]} | {NAMES[row['architecture']]} | {row['budget']} | "
                     f"{row['seed']} | {selected['threshold']:.2f} | {percent(selected['mask_F1'])} | {percent(selected['IoU'])} |")
    lines += ['', '## Проверка и воспроизведение', '',
              f"Независимый аудит: {audit['checks_passed']}/{audit['checks_total']}. "
              'Сохранены начальные и конечные веса, материализованные GroupCNN, '
              'пробы градиентов, вероятности, все пороги и маски, прогресс и хеши. '
              'Доверительные интервалы по адаптивной VAL не строились.', '',
              '[Протокол](PROTOCOL.md) · [Конфигурация](CONFIG.json) · '
              '[Фиксация входов](LOCK.json) · [Скрипт обучения](runner.py) · '
              '[Результат JSON](RESULT.json) · [Независимый аудит](AUDIT.json).', '',
              'Прежняя модель с дополнительным надзором и её закрытый результат '
              'зафиксированы отдельно: [паспорт модели](../sem_model_freeze_20261103/MODEL_CARD.md). '
              'Новая абляция не переписывает исход той закрытой проверки.', '',
              'Классы — геометрические состояния металлизации одной ИС, вычисленные '
              'из SVG-масок; это не экспертные типы производственных дефектов. '
              'Известная групповая свёртка не заявляется новым оператором.', '',
              'Повторное построение этого отчёта без обучения:', '',
              '```bash', '.venv/bin/python -B gipoteza/sem_arch_loss_factorial_20261103/report.py', '```', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    (ROOT / 'README.md').write_text(
        '# Абляция архитектуры и дополнительного надзора\n\n'
        f"12 новых BCE-only обучений завершены; 12 прежних BCE+geometry состояний использованы повторно. {conclusion}\n\n"
        '[Отчёт и графики](REPORT.md) · [Результаты](RESULT.json) · [Аудит](AUDIT.json) · '
        '[Протокол](PROTOCOL.md) · [Обучение](runner.py).\n\n'
        'Оценка описательная, на прежней VAL; закрытые 16 кадров не использовались. '
        'Дополнительный лосс также добавляет слабые метки, поэтому эффект этого режима '
        'не является эффектом одной формулы при одинаковой информации в разметке.\n')
    inputs = [ROOT / name for name in ['RESULT.json', 'AUDIT.json', 'LOCK.json', 'PREFLIGHT.json',
                                       'PREFLIGHT_AUDIT.json', 'audit.py', 'PROTOCOL.md', 'CONFIG.json',
                                       'common.py', 'runner.py', 'report.py']]
    outputs = [ROOT / 'REPORT.md', ROOT / 'README.md', ROOT / 'figures/factorial_quality.png',
               ROOT / 'figures/factorial_effects.png']
    manifest = dict(scientific_inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs},
                    outputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in outputs},
                    scope='DESCRIPTIVE_OLD_VAL_FACTORIAL_REPORT_NO_FITTING')
    (ROOT / 'REPORT_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(report=str(ROOT / 'REPORT.md'), development_gate=gate['passed'],
                          new_fits=12, inherited_fits=12)))


if __name__ == '__main__':
    main()
