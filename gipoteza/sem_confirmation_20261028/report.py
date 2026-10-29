"""Publish all pilot comparisons and figures without changing model outputs."""
from pathlib import Path
import json
import os
import re
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/sem_physics_mpl_20261028')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from prepare import sha, save

ROOT = Path(__file__).resolve().parent


def fmt(x):
    return f'{100*x:.3f}'.replace('.', ',')


def signed(x):
    return f'{100*x:+.3f}'.replace('.', ',')


def main():
    r = json.loads((ROOT / 'RESULT.json').read_text())
    a = json.loads((ROOT / 'AUDIT.json').read_text())
    assert a['status'] == 'PASS_AUDIT'
    t = json.loads((ROOT / 'TRAINING_COMPLETE.json').read_text())
    manifest = json.loads((ROOT / 'DATA_MANIFEST.json').read_text())
    labels = {'base': 'Исходный BCE', 'replay': 'BCE: дообучение', 'soft': 'Непрерывная физика',
              'head9': 'Голова: 9 классов', 'head4': 'Голова: 4 пороговых цели'}
    order = ['base', 'replay', 'head9', 'head4', 'soft']
    colours = ['#8693A5', '#A8B2BF', '#8064A2', '#C18A36', '#087F8C']
    figdir = ROOT / 'figures'; figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), sharey=True)
    for j, (family, colour) in enumerate(zip(order, colours)):
        metric = r['metrics'][family]
        axes[0].scatter(100*np.array(metric['per_seed_mask_F1']), j+np.linspace(-.07,.07,3), color=colour, alpha=.5, s=25)
        axes[0].scatter([100*metric['mask_F1']], [j], marker='D', color=colour, s=60)
        axes[0].annotate(fmt(metric['mask_F1']), (100*metric['mask_F1'], j), xytext=(7,7), textcoords='offset points', color=colour)
        axes[1].scatter([100*metric['IoU']], [j], marker='D', color=colour, s=60)
        axes[1].annotate(fmt(metric['IoU']), (100*metric['IoU'], j), xytext=(7,7), textcoords='offset points', color=colour)
    for ax, title in zip(axes, ['Класс из бинарной маски: Macro-F1, %', 'Сегментация: средний IoU, %']):
        ax.set_yticks(range(5), [labels[k] for k in order]); ax.set_title(title, loc='left', fontsize=11)
        ax.grid(axis='x', alpha=.16); ax.margins(x=.3); ax.set_axisbelow(True)
    axes[0].set_ylim(4.5, -.5)
    fig.suptitle('Одинаковые слабые метки: новые кадры того же ИС', x=.03, ha='left', fontsize=14, weight='bold')
    fig.text(.03, .015, f"{r['test_examples']} фрагментов; 8 кадров; 3 seed. Головам доступны те же изображения и классовые метки.", fontsize=9, color='#52606D')
    fig.tight_layout(rect=(0,.05,1,.93), w_pad=3)
    for ext in ['png', 'svg']: fig.savefig(figdir/f'comparison.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    families = ['base', 'replay', 'head9', 'head4']
    fig, ax = plt.subplots(figsize=(9.5,4.1))
    for j, family in enumerate(families):
        c = r['soft_minus_control'][family]; lo,hi = np.array(c['CI95'])*100; p = c['delta']*100
        ax.plot([lo,hi],[j,j], color='#087F8C', linewidth=2.3)
        ax.scatter([p],[j], color='#087F8C', s=45, zorder=3)
        ax.annotate(signed(c['delta']), (hi,j), xytext=(7,-3), textcoords='offset points', fontsize=10)
    ax.axvline(0,color='#8993A0',linestyle='--'); ax.axvline(1,color='#C18A36',linestyle=':',label='Минимальный выигрыш: 1 п.п.')
    ax.set_yticks(range(4),[labels[k] for k in families]); ax.set_ylim(3.5,-.5)
    ax.set_xlabel('Непрерывная физика минус контроль, п.п. Macro-F1')
    ax.set_title('Парные сравнения: 95%-интервалы по кадрам и seed',loc='left',pad=16)
    ax.margins(x=.2); ax.grid(axis='x',alpha=.15); ax.legend(frameon=False,fontsize=9,loc='lower right')
    fig.tight_layout()
    for ext in ['png','svg']: fig.savefig(figdir/f'paired_effects.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    table = ['| Метод | Macro-F1 из маски, % | IoU, % | Прямой выход головы, % |','|---|---:|---:|---:|']
    for k in order:
        m=r['metrics'][k]
        table.append(f"| {labels[k]} | {fmt(m['mask_F1'])} | {fmt(m['IoU'])} | {fmt(m['direct_head_F1']) if 'direct_head_F1' in m else '—'} |")
    comparisons=['| Контроль | Разность F1, п.п. | 95%-интервал, п.п. | Критерий этого сравнения |','|---|---:|---|---|']
    for k in families:
        c=r['soft_minus_control'][k]; passed=all(c['gates'].values())
        comparisons.append(f"| {labels[k]} | {signed(c['delta'])} | [{signed(c['CI95'][0])}; {signed(c['CI95'][1])}] | {'PASS' if passed else 'FAIL'} |")
    best=r['selection']['primary_head']
    verdict='Пилот поддержал преимущество непрерывной регуляризации при одинаковых дополнительных метках.' if r['status']=='PASS_PILOT' else 'Заранее заданный критерий преимущества непрерывной регуляризации в этом пилоте не выполнен.'
    text=f'''# Сравнение геометрической регуляризации с одинаковым дополнительным надзором

28 октября 2026 года. **{r['status']}**. Технический аудит: **{a['status']}**.

{verdict}

## Что проверялось

Одинаковые54 плотные маски,1152 дополнительных SEM с геометрическими классами,
одинаковый сегментатор и число его обновлений. Сравниваются непрерывная физика,
обычное BCE-дообучение и две обучаемые классификационные головы. Основная
метрика всех вариантов — класс, вычисленный из бинарной сегментации.
Прямой класс головы показан отдельно.

TEST: **{r['test_examples']} фрагментов восьми ранее не использовавшихся кадров**:
{', '.join(map(str,r['test_frames']))}. Три новые инициализации:
{', '.join(map(str,r['seeds']))}. Это один физический чип, а не восемь
независимых изделий. Точные D4-мотивы и полутоновые фрагменты не пересекаются
между частями; по краям удалено512 нативных пикселей.

## Результаты

{chr(10).join(table)}

![Сравнение качества](figures/comparison.png)

Критерий: выигрыш soft минимум1п.п. и положительная нижняя граница95%-интервала
против replay и выбранной по VAL головы; IoU не хуже более2п.п. Каждый критерий
обязателен. Голова выбрана **до TEST**: **{best}**. Интервалы —4000 парных
crossed-bootstrap-повторов с независимым пересэмплированием кадров и seed.

{chr(10).join(comparisons)}

![Парные сравнения](figures/paired_effects.png)

Исходный BCE не дообучался; остальные варианты получили по2000 дополнительных
шагов из одинаковых прогретых весов соответствующего seed. У каждой головы
есть дополнительные параметры и200 head-only обновлений с замороженным
backbone; равенство ёмкости или полного времени вычислений не заявляется.
Все сравнения сообщаются, а не только выбранное главное.

## Выбор параметров и ограничения

Для каждого семейства вес выбран из0.1/0.4/1.0 по VAL на двух разработочных
seed. Выбранные веса: soft={r['selection']['chosen']['soft']['weight']},
head9={r['selection']['chosen']['head9']['weight']},
head4={r['selection']['chosen']['head4']['weight']}. У двух семейств голов
совокупный бюджет поиска больше, чем у одного soft. TEST не выбирал
параметры, эпоху, тип головы или итоговый seed.

Головы и их веса выбирались по качеству класса из сегментации. Поэтому
вторичные числа прямых голов не заменяют сравнение с отдельно настроенным
классификатором, оптимизированным именно для прямого выхода.

Первая проверка допуска данных остановилась: редкий VAL-класс имел31 вместо
заданных32 мотивов. До подготовки TEST и обучения опубликована
[поправка минимума до30](AMENDMENT_01.md); все31 примера сохранены. TEST,
методы и критерий выигрыша не менялись. Исходный FAIL_DATA_GATE сохранён
в [первой попытке](attempt1_data_gate/RESULT.json). Это ограничивает статус
опыта разработочным пилотом.

При первом запуске выявлена ошибка совместного переноса dtype/device MPS↔CPU.
Исправлен порядок преобразований в отдельном адаптере; аналитическая потеря
и CPU-градиент совпали точно, прямой и обратный MPS-проходы проверены.
Исходная попытка сохранена в attempt2_mps_transfer/. Новое обучение
перезапущено с исходных фиксированных seed, до открытия TEST.

Классы рассчитываются по двумерной резисторной модели SVG-геометрии.
Измеренной электрической проводимости, типов производственных отказов
и независимого другого чипа в этом опыте нет. Результат ничего не добавляет
к проверке мирового приоритета и не гарантирует публикационную новизну.

## Время и аудит

Финальный исправный запуск обучения: **{t['seconds']/60:.2f} мин**,44000обновлений
backbone,33сохранённых checkpoint. Оценка: **{r['seconds']:.2f} с**.
Это не включает загрузку и написание кода.

Аудит проверил SHA источников/данных/весов, разбиение и D4-ключи,
пересчёт Macro-F1 через sklearn, все IoU из сохранённых масок, независимый
bootstrap через веса групп, декодирование порядковой головы, CPU/MPS-вывод
и {a['independent_sparse_class_cases']} классов другим разреженным решателем.
Максимальное расхождение вероятностей CPU/MPS: {a['CPU_MPS_max_probability_error']:.3g};
bootstrap: {a['independent_bootstrap_max_error']:.3g}.
Технический PASS не подменяет научный критерий.

## Файлы

- [Протокол версии2](PROTOCOL.md), [поправка](AMENDMENT_01.md).
- [Данные и разбиение](DATA_MANIFEST.json), [замороженные входы](FROZEN.json).
- [Выбор по VAL](SELECTION.json), [все разработочные оценки](DEV_RESULTS.json).
- [Результат](RESULT.json), [аудит](AUDIT.json), [веса и их SHA](CHECKPOINT_HASHES.json).
- [Код подготовки](prepare.py), [обучения](train.py), [MPS-адаптер](physics_adapter.py),
  [оценки](evaluate.py), [аудита](audit.py), [графиков и отчёта](report.py).

Сохранены DATA.npz, PREDICTIONS.npz с полными бинарными масками/классами,
BOOTSTRAP.npz, исходные PNG/SVG и все веса models/. Текущий документ гипотезы
и старые результаты не переписывались.
'''
    text = re.sub(r'(?<=[А-Яа-яёЁ])(?=\d)|(?<=\d)(?=[А-Яа-яёЁ])', ' ', text)
    (ROOT/'REPORT.md').write_text(text)
    save(ROOT/'REPORT_MANIFEST.json',{'report_sha256':sha(ROOT/'REPORT.md'), 'result_sha256':sha(ROOT/'RESULT.json'),
         'audit_sha256':sha(ROOT/'AUDIT.json'),'figures_sha256':{str(p.relative_to(ROOT)):sha(p) for p in figdir.iterdir()},
         'scope':'Report and plots from fixed outputs; no refitting or selection changes.'})
    print(json.dumps({'status':r['status'],'audit':a['status'],'report':str(ROOT/'REPORT.md')},ensure_ascii=False))


if __name__=='__main__':main()
