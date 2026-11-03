"""Report a completed independent confirmation; no model, data or lock writes."""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/sem_bce_confirmation_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CELLS = ['BCE_group', 'BCE_plain', 'aux_group', 'aux_plain']
NAMES = {'BCE_group': 'Групповая CNN / BCE', 'BCE_plain': 'Обычная CNN C4 / BCE',
         'aux_group': 'Групповая CNN / BCE + aux', 'aux_plain': 'Обычная CNN C4 / BCE + aux'}
CONTRASTS = {'architecture_BCE': 'Архитектура при BCE: Group − Plain',
             'architecture_auxiliary': 'Архитектура при BCE + aux: Group − Plain',
             'auxiliary_group': 'Дополнительный надзор у Group',
             'auxiliary_plain': 'Дополнительный надзор у Plain',
             'interaction': 'Взаимодействие двух факторов'}


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def percent(value): return f'{100 * value:.3f}'
def pp(value): return f'{100 * value:+.3f}'


def validate():
    result, audit = read(ROOT/'RESULT.json'), read(ROOT/'AUDIT.json')
    assert result['status'] in ['PASS_CONFIRMATION', 'FAIL_CONFIRMATION']
    assert result['fit_count'] == result['optimizer_updates'] == 0
    assert result['source_model_count'] == result['evaluated_routes'] == 24 and len(result['rows']) == 24
    assert result['test_examples'] == 6144 and len(result['test_frames']) == 24
    assert audit['status'].startswith('PASS') and audit['checks_passed'] == audit['checks_total']
    assert all(audit['checks'].values())
    for name, expected in audit['input_sha256'].items():
        path = Path(name) if Path(name).is_absolute() else REPO/name
        assert sha(path) == expected, f'Independent audit has a changed input: {name}'
    for name in ['RESULT.json','SELECTION.json','MODEL_LOCK.json','EVAL.npz','PREDICTIONS.npz','BOOTSTRAP.npz']:
        key = str((ROOT/name).relative_to(REPO))
        assert key in audit['input_sha256'], f'Audit does not bind current {name}'
    for name, key in [('SELECTION.json','selection_sha256'),('EVALUATION_FROZEN.json','evaluation_frozen_sha256'),
                      ('PREDICTIONS.npz','predictions_sha256'),('BOOTSTRAP.npz','bootstrap_sha256')]:
        assert sha(ROOT/name) == result[key]
    assert result['confirmed'] == result['analyses']['primary_fine']['primary_goal']['confirmed']
    selection, plan, index = [read(ROOT/name) for name in ['SELECTION.json','SOURCE_PLAN.json','EVAL_INDEX.json']]
    assert len(selection['rows']) == 24 and len({(v['cell'],v['budget'],v['seed']) for v in selection['rows']}) == 24
    assert selection['source_plan_sha256'] == sha(ROOT/'SOURCE_PLAN.json')
    assert plan['test_frames'] == result['test_frames'] and plan['excluded_frame_count'] == 110
    assert len(plan['prior_manifests_sha256']) == 7
    assert index['examples'] == 6144 and all(index['gates'].values()) and index['EVAL_sha256'] == sha(ROOT/'EVAL.npz')
    assert result['analyses']['fixed05']['primary_goal']['confirmed'] is False
    assert 'CI95' not in result['analyses']['fixed05']['primary_goal']
    return result,audit,selection,plan,index


def figures(result):
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    goal=result['analyses']['primary_fine']['primary_goal']
    delta=100*goal['mean_F1_delta'];lo,hi=np.array(goal['CI95'])*100
    fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
    color='#197d86' if goal['confirmed'] else '#b66530'
    ax.hlines(0,lo,hi,color=color,linewidth=2.2)
    ax.vlines([lo,hi],-.065,.065,color=color,linewidth=2.2)
    ax.plot([delta],[0],'o',markersize=8,color=color,label='Наблюдаемый прирост и 95%-й интервал')
    ax.axvline(0,color='#45515a',linewidth=1.2,label='Нулевая разность; нижняя граница должна быть > 0')
    ax.axvline(.5,color='#b66530',linestyle='--',linewidth=1.2,label='Порог наблюдаемого прироста +0,5 п.п.')
    ax.annotate(f'{delta:+.2f} п.п.; 95% [{lo:+.2f}; {hi:+.2f}]',
                (delta,0),xytext=(0,20),textcoords='offset points',ha='center',fontsize=10)
    ax.set_yticks([0],['Group − Plain\nпри BCE'])
    ax.set_ylim(-.6,.65)
    low=min(lo,0.,.5,delta);high=max(hi,0.,.5,delta);pad=max(.3,.17*(high-low))
    ax.set_xlim(low-pad,high+pad)
    ax.set_xlabel('Разность Macro-F1 классификации, процентные пункты')
    ax.set_title('Закрытая проверка: 24 новых кадра той же ИС')
    ax.grid(axis='x',alpha=.2)
    ax.legend(loc='lower left',fontsize=8)
    fig.savefig(folder/'confirmation_primary_interval.png',dpi=180)
    plt.close(fig)

    colors=['#6a7a89','#197d86'];fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    primary=result['analyses']['primary_fine']
    for ax,metric,title in zip(axes,['mask_F1','IoU'],['Macro-F1 классификации, %','IoU сегментации, %']):
        for j,family in enumerate(['plain','group']):
            x=np.arange(2)+(j-.5)*.32
            cells=[prefix+'_'+family for prefix in ['BCE','aux']]
            means=[100*primary['cells'][cell]['mean'][metric] for cell in cells]
            ax.bar(x,means,width=.29,color=colors[j],alpha=.85,
                   label='Обычная CNN C4' if family=='plain' else 'Групповая CNN')
            for k,cell in enumerate(cells):
                points=[100*row['metrics']['primary_fine'][metric] for row in result['rows'] if row['cell']==cell]
                assert len(points)==6
                ax.scatter(x[k]+np.linspace(-.085,.085,6),points,color=colors[j],edgecolor='white',linewidth=.5,s=24,zorder=3)
                ax.annotate(f'{means[k]:.2f}',(x[k],means[k]),xytext=(0,6),textcoords='offset points',ha='center',fontsize=9)
        ax.set_xticks([0,1],['BCE','BCE + aux']);ax.set_ylim(0,105);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend(loc='lower left',fontsize=8)
    fig.suptitle('Четыре фиксированных сочетания на 6 144 новых фрагментах\n'
                 'Столбцы — среднее; точки — отдельные состояния бюджет × seed',fontsize=11)
    fig.savefig(folder/'confirmation_factorial_quality.png',dpi=180)
    plt.close(fig)


def contrast_table(summary):
    lines=['| Описательное сравнение | ΔMacro-F1, п.п. | ΔIoU, п.п. |','|---|---:|---:|']
    for name,title in CONTRASTS.items():
        effect=summary['contrasts'][name]['mean']
        lines.append(f"| {title} | {pp(effect['mask_F1'])} | {pp(effect['IoU'])} |")
    return '\n'.join(lines)


def main():
    result,audit,selection,plan,index=validate();figures(result)
    config=read(ROOT/'CONFIG.json');data_audit=read(ROOT/'DATA_AUDIT.json')
    primary=result['analyses']['primary_fine'];goal=primary['primary_goal']
    conclusion='подтверждена' if goal['confirmed'] else 'не подтверждена'
    interval=goal['CI95']
    lines=['# Закрытая проверка архитектурного преимущества при обычном BCE','',
        f'3 ноября 2026 года. Основная гипотеза {conclusion}: GroupCNN против обычной Plain12 C4 '
        'при одинаковом режиме BCE, близком числе независимых параметров и MAC. '
        'Проверены 6 144 фрагмента из 24 ранее неиспользованных кадров одной ИС.', '',
        f"Macro-F1: Group {percent(goal['group_F1'])}% против Plain {percent(goal['reference_F1'])}%; "
        f"наблюдаемая разность {pp(goal['mean_F1_delta'])} п.п., 95%-й интервал "
        f"[{pp(interval[0])}; {pp(interval[1])}] п.п. "
        f"Средний ΔIoU {pp(goal['IoU_delta'])} п.п.", '',
        'Новых обучений и обновлений оптимизатора — 0. Использованы 24 заранее '
        'зафиксированных конечных состояния: четыре сочетания архитектуры/надзора × '
        'два бюджета масок × три seed. Веса и прежние VAL-пороги закреплены до '
        'загрузки новых изображений. Ни один порог на новой выборке не подбирался.', '',
        '## Единственная подтверждающая проверка', '',
        'Первичное сравнение — GroupCNN против Plain12 C4 при обычном BCE. '
        'Практический порог +0,5 п.п. относится к наблюдаемому среднему приросту. '
        'Дополнительно нижняя граница 95%-го интервала должна быть строго больше нуля. '
        'Это разные условия; порог точки нельзя представлять как доказанный '
        'минимальный популяционный прирост +0,5 п.п.', '',
        '| Условие | Исход |','|---|---|']
    guard_names={'mean_F1':'Наблюдаемый средний ΔF1 ≥ +0,5 п.п.',
        'CI_lower':'Нижняя граница 95%-го интервала строго > 0',
        'budget_guard':'ΔF1 каждого бюджета ≥ −0,25 п.п.',
        'IoU_guard':'Средний ΔIoU ≥ −0,2 п.п.',
        'parameter_match':'Расхождение свободных параметров ≤ 3%',
        'MAC_match':'Расхождение MAC ≤ 3%',
        'development_eligible':'Предшествующая абляция и независимый аудит пройдены'}
    for name,passed in goal['guards'].items():
        lines.append(f"| {guard_names[name]} | {'PASS' if passed else 'FAIL'} |")
    lines+=['','![Первичная разность и интервал](figures/confirmation_primary_interval.png)','',
        'Остальные четыре факторных контраста и общий порог 0,5 являются '
        'описательными. Для них не строились подтверждающие интервалы или p-value; '
        'они не могут заменить исход основной проверки.', '',
        '## Четыре фиксированных сочетания', '',
        'В таблице — среднее индивидуальных метрик по трём seed и двум бюджетам. '
        'Это качество отдельных моделей, усреднённое для сравнения, а не результат ансамбля. '
        'Историческое поле `mask_F1` обозначает Macro-F1 девяти геометрических классов '
        'из бинарной маски; IoU отдельно измеряет сегментацию металла.', '',
        '| Сочетание | Macro-F1, % | IoU, % |','|---|---:|---:|']
    for cell in CELLS:
        mean=primary['cells'][cell]['mean']
        lines.append(f"| {NAMES[cell]} | {percent(mean['mask_F1'])} | {percent(mean['IoU'])} |")
    lines+=['','![Четыре ячейки и отдельные состояния](figures/confirmation_factorial_quality.png)','',
        contrast_table(primary),'',
        'В режиме BCE используются только 18 либо 27 полных масок. Режим BCE + aux '
        'дополнительно использует 1 152 классовые метки и прежнюю geometry_loss. '
        'Поэтому эффект aux — эффект всего дополнительного надзора, а не одной '
        'формулы лосса при одинаковой информации в разметке. Архитектурные сравнения '
        'внутри каждого режима используют одинаковую разметку.', '',
        '## Описательная чувствительность: общий порог 0,5', '',
        contrast_table(result['analyses']['fixed05']), '',
        'Общий порог 0,5 применяется к тем же вероятностям без нового обучения. '
        'Результат не меняет первичный исход при прежних зафиксированных VAL-порогах.', '',
        '## Все 24 состояния и фиксированные пороги', '',
        '| Сочетание | Масок | Seed | Порог | Primary F1, % | Primary IoU, % | F1 при 0,5, % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    locked={(row['cell'],row['budget'],row['seed']):row for row in selection['rows']}
    for row in result['rows']:
        identity=(row['cell'],row['budget'],row['seed']);threshold=locked[identity]['primary_fine_threshold']
        score=row['metrics']['primary_fine'];secondary=row['metrics']['fixed05']
        assert threshold==score['threshold'] and secondary['threshold']==.5
        lines.append(f"| {NAMES[row['cell']]} | {row['budget']} | {row['seed']} | {threshold:.2f} | "
            f"{percent(score['mask_F1'])} | {percent(score['IoU'])} | {percent(secondary['mask_F1'])} |")
    lines+=['','## Новые данные и происхождение','',
        'Кадры выбраны детерминированно по SHA-256 до загрузки. Исключены 110 '
        'кадров из прежних планов и данных среди 321 доступного M2. Ранжирование '
        '256 фрагментов каждого нового кадра использует SHA исходных интенсивностей, '
        'без маски, класса или качества предсказания. Фиксированный набор не заменялся '
        'по результатам проверки покрытия.', '',
        'На стадии формирования кандидатов маски используются для удаления '
        'однородных фрагментов и точных геометрических повторов. Утверждение '
        'об отсутствии маски и класса относится именно к последующему ранжированию, '
        'а не ко всей подготовке набора.', '',
        'По семи прежним манифестам исключены точные D4-повторы исходных '
        'интенсивностей и бинарных масок; такие повторы исключены и внутри нового набора. '
        'Используется прежняя подготовка: отступ 512 нативных пикселей, '
        'усреднение изображения 4×4, маска по доле металла ≥0,5, '
        'фрагменты 12×12 с шагом 6. Это не независимая геометрическая регистрация '
        'полей съёмки в координатах микросхемы.', '',
        'Фиксированные кадры: `'+str(plan['test_frames'])+'`.', '',
        '| Геометрический класс | Фрагментов | Различных кадров |','|---|---:|---:|']
    for cls,count in enumerate(index['class_counts']):
        lines.append(f"| {cls} | {count} | {len(index['class_frames'][str(cls)])} |")
    lines+=['',
        'У каждого класса требуется ≥8 фрагментов и ≥3 кадров. Вне EVAL остаются '
        'другие фрагменты тех же кадров; они не являются ещё одним независимым тестом.', '',
        '## Статистика и техническая проверка','',
        'Для единственного первичного контраста выполнены 10 000 парных '
        'crossed-bootstrap повторов: три seed и 24 кадра выбираются с возвращением. '
        'Одни индексы используются обеими CNN и обоими бюджетами. Confusion сначала '
        'суммируется по выбранным кадрам отдельно для каждого выбранного seed; '
        'затем рассчитывается Macro-F1 и усредняется по seed и бюджетам. '
        'Confusion разных seed не объединяются до вычисления F1. '
        'Использованы процентильные квантили 0,025 и 0,975.', '',
        'Интервал является приближённой оценкой в заданной схеме пересэмплирования. '
        'Всего три инициализации ограничивают оценку вариации обучения; '
        'тысячи зависимых фрагментов не считаются независимыми экспериментами.', '',
        f"Независимый аудит данных — {data_audit['checks_passed']}/{data_audit['checks_total']}; "
        f"независимый аудит результата — {audit['checks_passed']}/{audit['checks_total']}. "
        'Отчёт проверяет хеши аудированных входов перед построением графиков. '
        f"Оценка и статистика заняли {result['seconds']:.2f} с; "
        'это длительность данного прогона, а не измерение полной задержки приложения.', '',
        '## Границы и связь с прежними опытами','',
        'Все новые поля относятся к одной физической ИС и слою металлизации M2. '
        'Классы вычисляются из SVG-масок геометрическим правилом. Это не экспертные '
        'метки типов производственных дефектов, электрические измерения или '
        'измеренный выход годных. Межчиповый перенос не проверялся. '
        'Сама C4-групповая свёртка является известным оператором.', '',
        '[Прежняя абляция на VAL](../sem_arch_loss_factorial_20261103/REPORT.md) '
        'сохраняет свои исходные числа. [Зафиксированный вариант с aux]'
        '(../sem_model_freeze_20261103/MODEL_CARD.md) и '
        '[его прежнее закрытое подтверждение](../sem_arch_groupcnn_confirmation_20261102/REPORT.md) '
        'также сохранены без изменения. Исход новой проверки не переписывает предыдущие результаты.', '',
        '[Протокол](PROTOCOL.md) · [Конфигурация](CONFIG.json) · [Выбор и фиксация](selection.py) · '
        '[Веса и пороги](MODEL_LOCK.json) · [План данных](SOURCE_PLAN.json) · '
        '[Оценка](evaluate.py) · [Статистика](stats.py) · [Результат JSON](RESULT.json) · '
        '[Аудит данных](DATA_AUDIT.json) · [Итоговый аудит](AUDIT.json).', '',
        'Повторное построение отчёта без обучения и предсказаний:', '',
        '```bash','.venv/bin/python -B gipoteza/sem_arch_bce_confirmation_20261103/report.py','```','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))
    (ROOT/'README.md').write_text('# Закрытая проверка архитектуры при BCE\n\n'
        f"Основная гипотеза {conclusion}. 6 144 новых фрагмента из 24 кадров той же ИС; "
        '24 заранее фиксированных состояния, 0 новых обучений.\n\n'
        f"Group {percent(goal['group_F1'])}% против Plain {percent(goal['reference_F1'])}% Macro-F1; "
        f"Δ{pp(goal['mean_F1_delta'])} п.п., 95% [{pp(interval[0])}; {pp(interval[1])}] п.п.\n\n"
        '[Отчёт и графики](REPORT.md) · [Результат](RESULT.json) · [Аудит](AUDIT.json) · '
        '[Протокол](PROTOCOL.md).\n\n'
        'Четыре сочетания архитектуры/надзора оценены на одних новых данных. '
        'Только BCE Group−Plain является подтверждающим сравнением; прочие '
        'контрасты и общий порог 0,5 описательные. Классы — геометрические состояния '
        'металлизации, вычисленные из SVG-масок.\n')
    names=['RESULT.json','AUDIT.json','SELECTION.json','MODEL_LOCK.json','SOURCE_PLAN.json','CONFIG.json',
        'PROTOCOL.md','DATA_AUDIT.json','DATA_MANIFEST.json','EVAL_INDEX.json','EVAL.npz','PREDICTIONS.npz',
        'BOOTSTRAP.npz','EVALUATION_FROZEN.json','OLD_VAL_PREFLIGHT.json','OLD_VAL_PREFLIGHT.npz',
        'common.py','selection.py','evaluate.py','prepare.py','stats.py','audit.py','report.py']
    names += [name for name in ['PRE_LOCK_AUDIT.json','PRE_DOWNLOAD_AUDIT.json'] if (ROOT/name).exists()]
    outputs=['REPORT.md','README.md','figures/confirmation_primary_interval.png','figures/confirmation_factorial_quality.png']
    manifest=dict(scientific_inputs_sha256={name:sha(ROOT/name) for name in names},
        outputs_sha256={name:sha(ROOT/name) for name in outputs},
        scope='LOCKED_NEW_FRAME_BCE_ARCHITECTURE_CONFIRMATION_REPORT_NO_FIT_OR_INFERENCE')
    (ROOT/'REPORT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(report=str(ROOT/'REPORT.md'),status=result['status'],confirmed=goal['confirmed'],
        audit_checks=audit['checks_total'],scientific_inputs=len(names),outputs=len(outputs))),flush=True)


if __name__=='__main__':main()
