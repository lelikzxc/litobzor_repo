"""Descriptive, reproducible analysis of immutable calibrated predictions."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCE = ROOT.parent / 'sem_arch_calibrated_confirmation_20261103'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARCHITECTURES = ['group6', 'untied24_single']
SEEDS = [5701, 5721, 5757]
EXAMPLE_SEED = 5701
EXAMPLE_PREFIX = 'SEM-C4-ERROR-EXAMPLES-20261103/'
CATEGORY_NAMES = {
    'improvement': 'Исправление', 'regression': 'Ухудшение',
    'both_wrong': 'Обе сети ошиблись', 'both_correct': 'Обе сети верны',
}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def relative(path): return str(Path(path).resolve().relative_to(REPO))
def save(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def input_paths():
    names = ['PREDICTIONS.npz', 'EVAL.npz', 'EVAL_INDEX.json', 'RESULT.json',
             'AUDIT.json', 'DATA_AUDIT.json', 'DATA_MANIFEST.json', 'MODEL_LOCK.json']
    paths = [SOURCE / name for name in names]
    frames = sorted({r['frame'] for r in read(SOURCE / 'EVAL_INDEX.json')['records']})
    return paths + [SOURCE / 'sources' / f'sem{frame:04d}.png' for frame in frames]


def freeze():
    sources = {relative(path): sha(path) for path in input_paths()}
    lock = dict(scope='POSTHOC_DESCRIPTIVE_ERROR_ANALYSIS_NO_NEW_TEST',
                source_sha256=sources, script_sha256=sha(__file__),
                protocol_sha256=sha(ROOT / 'PROTOCOL.md'),
                architectures=ARCHITECTURES, seeds=SEEDS, example_seed=EXAMPLE_SEED,
                example_prefix=EXAMPLE_PREFIX, example_count_each_category=2,
                new_fits=0, new_thresholds=0)
    path = ROOT / 'LOCK.json'
    if path.exists(): assert read(path) == lock, 'Inputs or analysis changed; preserve earlier outputs'
    else: save(path, lock)
    return lock


def confusion(truth, prediction, n=9):
    return np.bincount(n * truth + prediction, minlength=n*n).reshape(n, n)


def class_metrics(cm):
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(len(tp)), where=cm.sum(0)>0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(len(tp)), where=cm.sum(1)>0)
    f1 = np.divide(2*tp, cm.sum(0)+cm.sum(1), out=np.zeros(len(tp)), where=cm.sum(0)+cm.sum(1)>0)
    return dict(precision=precision, recall=recall, F1=f1)


def direction_errors(truth, prediction):
    result = {}
    for name, y, p in [('LR', truth//3, prediction//3), ('TB', truth%3, prediction%3)]:
        zero = y == 0; connected = ~zero
        false = (zero & (p > 0)).sum(); missed = (connected & (p == 0)).sum()
        grade = (connected & (p > 0) & (p != y)).sum()
        result[name] = dict(false_connection=int(false), missed_connection=int(missed),
                            wrong_nonzero_grade=int(grade), absent_count=int(zero.sum()),
                            connected_count=int(connected.sum()),
                            false_connection_rate=float(false/zero.sum()),
                            missed_connection_rate=float(missed/connected.sum()),
                            wrong_nonzero_grade_rate=float(grade/connected.sum()),
                            confusion=confusion(y, p, 3).tolist())
    return result


def plot_classwise(metrics, support):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={'width_ratios': [2.2, 1]})
    y = np.arange(9)
    for a, label in enumerate(['GroupCNN', 'Untied24']):
        axes[0].barh(y+(a-.5)*.34, metrics[a, :, :].mean(0)*100, height=.32, label=label,
                     color=['#2176ae', '#dd8c36'][a])
    axes[0].set(yticks=y, yticklabels=[str(i) for i in y], xlabel='F1 класса, %; среднее по 3 инициализациям',
                ylabel='Класс y = 3·LR + TB', xlim=(0, 100))
    axes[0].legend(loc='lower right'); axes[0].grid(axis='x', alpha=.22); axes[0].set_axisbelow(True)
    axes[1].barh(y, support, color='#777777')
    axes[1].set(yticks=y, yticklabels=[str(i) for i in y], xlabel='Фрагментов', title='Поддержка классов')
    for i, value in enumerate(support): axes[1].text(value+25, i, str(value), va='center', fontsize=9)
    axes[1].set_xlim(0, support.max()*1.2)
    for ax in axes: ax.invert_yaxis()
    fig.suptitle('Классовое качество и дисбаланс: 6 144 фрагмента, 24 кадра M2')
    fig.tight_layout(); fig.savefig(ROOT/'figures/classwise_F1_support.png', dpi=180); plt.close(fig)


def plot_confusions(cms):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True, layout='constrained')
    for a, label in enumerate(['GroupCNN', 'Untied24']):
        cm = cms[a].mean(0)
        row_normalized = cm/cm.sum(1, keepdims=True)
        im = axes[a].imshow(row_normalized, vmin=0, vmax=1, cmap='Blues', interpolation='nearest')
        axes[a].set(xticks=range(9), yticks=range(9), xlabel='Предсказанный класс', title=label)
        for row in range(9):
            for col in range(9):
                value = 100*row_normalized[row,col]
                axes[a].text(col,row,f'{value:.0f}', ha='center', va='center', fontsize=8,
                             color='white' if value>=60 else '#222222')
    axes[0].set_ylabel('Истинный класс')
    fig.colorbar(im, ax=axes, shrink=.8, label='Доля внутри истинного класса')
    fig.suptitle('Матрицы ошибок: среднее по 3 инициализациям, числа в %')
    fig.savefig(ROOT/'figures/confusion_matrices.png', dpi=180); plt.close(fig)


def example_selection(records, truth, pred):
    s = SEEDS.index(EXAMPLE_SEED); gc = pred[0,s] == truth; uc = pred[1,s] == truth
    masks = dict(improvement=gc & ~uc, regression=~gc & uc, both_wrong=~gc & ~uc,
                 both_correct=gc & uc)
    result = []
    for category, mask in masks.items():
        candidates = np.flatnonzero(mask).tolist()
        candidates.sort(key=lambda i: hashlib.sha256((EXAMPLE_PREFIX+category+'/'+records[i]['raw_D4_sha256']).encode()).hexdigest())
        assert len(candidates)>=2
        for i in candidates[:2]:
            result.append(dict(category=category, evaluation_index=i, seed=EXAMPLE_SEED,
                               **records[i], truth=int(truth[i]),
                               GroupCNN_class=int(pred[0,s,i]), Untied24_class=int(pred[1,s,i]),
                               selection_sha256=hashlib.sha256((EXAMPLE_PREFIX+category+'/'+records[i]['raw_D4_sha256']).encode()).hexdigest()))
    assert len({r['evaluation_index'] for r in result}) == 8
    return result


def plot_examples(examples, target, masks, x):
    native_checks = []
    images = {}
    for name, selected in [('paired_changes', examples[:4]), ('shared_outcomes', examples[4:])]:
        fig, axes = plt.subplots(4, 4, figsize=(12, 11.8))
        fig.subplots_adjust(left=.10, right=.98, bottom=.025, top=.90, hspace=.25, wspace=.12)
        for row, record in enumerate(selected):
            i = record['evaluation_index']; frame = record['frame']
            if frame not in images:
                with Image.open(SOURCE/'sources'/f'sem{frame:04d}.png') as image:
                    images[frame] = np.array(image)
            native = images[frame][record['y']:record['y']+48,record['x']:record['x']+48]
            reconstructed = native.reshape(12,4,12,4).mean((1,3))/255
            error = float(np.max(np.abs(reconstructed-x[i,0])))
            assert native.shape==(48,48) and error<1e-7
            native_checks.append(dict(index=i, frame=frame, maximum_meanpool_error=error))
            views = [native, target[i], masks[1,0,i], masks[0,0,i]]
            titles = ['SEM 48×48', f'Разметка: класс {record["truth"]}',
                      f'Untied24: класс {record["Untied24_class"]}', f'GroupCNN: класс {record["GroupCNN_class"]}']
            for col, (view, title) in enumerate(zip(views, titles)):
                axes[row,col].imshow(view, cmap='gray', vmin=0, vmax=255 if col==0 else 1, interpolation='nearest')
                axes[row,col].set_title(title, fontsize=10)
                axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            axes[row,0].set_ylabel(f'{CATEGORY_NAMES[record["category"]]}\nкадр {frame}, индекс {i}', fontsize=9)
        fig.suptitle('Фиксированные примеры, инициализация 5701; маски 12×12\nСветлое — металл; нативное поле 48×48 пикселей', fontsize=12, y=.97)
        fig.savefig(ROOT/'figures'/f'examples_{name}.png', dpi=180); plt.close(fig)
    return native_checks


def report(result):
    rows = result['classwise']; pair = result['paired']; d = result['directional']
    f1 = result['Macro_F1']
    lines = ['# Разбор ошибок GroupCNN и Untied24', '',
        'Описательный разбор сохранённого закрытого опыта от 3 ноября 2026 года: '
        '6 144 фрагмента, 24 кадра металлизации M2 одной ИС, три инициализации. '
        'Новых обучений и подбора порогов нет. Используется финальная калибровка, зафиксированная до новых кадров.', '',
        f'Средняя Macro-F1: GroupCNN {f1["GroupCNN"]*100:.3f}%, Untied24 {f1["Untied24"]*100:.3f}%, '
        f'разность {(f1["GroupCNN"]-f1["Untied24"])*100:+.3f} п.п. Это среднее трёх отдельных F1, а не F1 ансамбля.', '',
        '## Какие классы различаются', '',
        'Класс задаётся как `y = 3·LR + TB`: в каждом направлении 0 означает отсутствие пути между границами маски, '
        '1 и 2 — наличие пути с нормированной геометрической проводимостью ниже и не ниже 0,5. '
        'Например, класс 0 — (0,0), класс 3 — (1,0), класс 8 — (2,2). Это геометрические состояния маски.', '',
        '| Класс (LR, TB) | Фрагментов | F1 Group, % | F1 Untied, % | ΔF1, п.п. |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        k=row['class']; lines.append(f'| {k} ({k//3}, {k%3}) | {row["support"]} | {row["GroupCNN"]["F1"]*100:.2f} | '
            f'{row["Untied24"]["F1"]*100:.2f} | {row["delta_F1"]*100:+.2f} |')
    lines += ['', 'F1 каждого класса сначала вычислена отдельно каждой инициализации, затем усреднена. '
              'Precision, recall и исходные матрицы также сохранены в [RESULT.json](RESULT.json). '
              'Редкие классы имеют только 79–208 фрагментов: большая разность для одного из них требует осторожной трактовки.', '',
              '![F1 классов и поддержка](figures/classwise_F1_support.png)', '',
              '![Матрицы ошибок](figures/confusion_matrices.png)', '',
              '## Парные исправления и ухудшения', '',
              '| Инициализация | Исправлений Group | Ухудшений Group | Обе ошиблись | Обе верны |',
              '|---|---:|---:|---:|---:|']
    for row in pair['per_seed']:
        lines.append(f'| {row["seed"]} | {row["improvement"]} | {row["regression"]} | {row["both_wrong"]} | {row["both_correct"]} |')
    t=pair['totals']
    lines += [f'| Сумма по трём | {t["improvement"]} | {t["regression"]} | {t["both_wrong"]} | {t["both_correct"]} |', '',
              f'Сумма содержит 18 432 пары «фрагмент × инициализация» с повторным использованием тех же 6 144 изображений. '
              f'Исправлений больше ухудшений на {t["improvement"]-t["regression"]}; это разность числа верных ответов, '
              'её нельзя приравнивать к разности Macro-F1. Категория «обе ошиблись» включает как одинаковые, так и разные ошибочные классы.', '',
              '## Какие ошибки геометрии возникают', '',
              'Числа ниже суммируют ошибки по трём инициализациям. Ложное соединение — истинное состояние 0, '
              'предсказанное 1/2; пропущенное — истинное 1/2, предсказанное 0. Перепутанная градация — 1↔2 '
              'при сохранённом наличии пути. Для ложных соединений знаменатель — число истинно несоединённых пар, '
              'для двух остальных ошибок — число истинно соединённых пар.', '',
              '| Направление / ошибка | Group | Untied | Δ числа ошибок Group−Untied |',
              '|---|---:|---:|---:|']
    for direction in ['LR','TB']:
        for key,label in [('false_connection','Ложное соединение'),('missed_connection','Пропущенное соединение'),('wrong_nonzero_grade','Градация 1↔2')]:
            g,u=d['GroupCNN'][direction][key],d['Untied24'][direction][key]
            lines.append(f'| {direction}: {label} | {g} | {u} | {g-u:+d} |')
    lines += ['', 'Ошибка связи между границами маски не равнозначна реальному замыканию или обрыву цепи: '
              'электрические измерения и экспертные типы производственных дефектов здесь отсутствуют. '
              'По этим частотам нельзя установить, что именно вызвало выигрыш архитектуры.', '',
              '## Примеры для ВКР', '',
              'Для инициализации 5701 выбраны по два примера каждой категории по фиксированному SHA-ранжированию, '
              'описанному в [протоколе](PROTOCOL.md). Показаны улучшения, ухудшения и общие исходы; '
              'примеры не ранжировались по величине выигрыша или уверенности. Маски 12×12 увеличены без интерполяции. '
              'В левом столбце исходный SEM-фрагмент 48×48, из которого сеть получает усреднённое изображение 12×12.', '',
              '![Исправления и ухудшения](figures/examples_paired_changes.png)', '',
              '![Общие ошибки и верные ответы](figures/examples_shared_outcomes.png)', '',
              'Координаты и индексы всех восьми примеров сохранены в [EXAMPLES.json](EXAMPLES.json). '
              'Числовая проверка восстановила сетевой вход из каждого нативного фрагмента с ошибкой меньше 10⁻⁷.', '',
              '## Границы интерпретации', '',
              'Это разбор уже полученного результата, а не новый независимый тест. Классы выведены из SVG-масок; '
              'данные относятся к одной ИС и слою M2. Пул фрагментов отфильтрован с использованием истинных масок '
              '(однородность и точные повторы), хотя последующее SHA-ранжирование отбора не использовало класс или маску. '
              'Дисбаланс классов и зависимость фрагментов внутри кадров сохраняются. Новые интервалы, значимость, '
              'причинный механизм и перенос на другую ИС не заявляются.', '',
              'Повторное воспроизведение без обучения и MPS:', '',
              '```bash', '.venv/bin/python -B gipoteza/sem_arch_error_analysis_20261103/analyze.py', '```', '',
              '[Исходный закрытый отчёт](../sem_arch_calibrated_confirmation_20261103/REPORT.md) · '
              '[Результаты](RESULT.json) · [Проверки](CHECKS.json) · [Зафиксированные входы](LOCK.json).', '']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))


def main():
    lock=freeze(); (ROOT/'figures').mkdir(exist_ok=True)
    source_result=read(SOURCE/'RESULT.json'); audit=read(SOURCE/'AUDIT.json')
    assert source_result['confirmed'] and all(audit['checks'].values())
    with np.load(SOURCE/'PREDICTIONS.npz',allow_pickle=False) as z:
        assert z['modes'][0]=='final_calibrated' and z['architectures'][:2].tolist()==ARCHITECTURES and z['seeds'].tolist()==SEEDS
        pred=z['mask_predictions'][0,:2].copy(); truth=z['truth'].copy(); target=z['target'].copy()
        packed=z['packed_masks'][0,:2].copy(); iou=z['IoU'][0,:2].copy(); probability=z['probability'][:2].copy()
        thresholds=z['thresholds'][0,:2].copy(); frames=z['frames'].copy()
    masks=np.unpackbits(packed,axis=-1).reshape(2,3,6144,12,12).astype(bool)
    assert np.array_equal(masks,probability>=thresholds[:,:,None,None,None])
    with np.load(SOURCE/'EVAL.npz',allow_pickle=False) as z:
        x=z['X'].copy(); assert np.array_equal(z['label'],truth) and np.array_equal(z['Y'][:,0],target)
        assert np.array_equal(z['frame'],frames)
    records=read(SOURCE/'EVAL_INDEX.json')['records']; assert len(records)==len(truth)==6144
    assert all(r['frame']==int(frames[i]) and r['label']==int(truth[i]) for i,r in enumerate(records))
    support=np.bincount(truth,minlength=9)
    cms=np.array([[confusion(truth,pred[a,s]) for s in range(3)] for a in range(2)])
    metrics=np.array([[class_metrics(cms[a,s])['F1'] for s in range(3)] for a in range(2)])
    macro=metrics.mean((1,2))
    names=['GroupCNN','Untied24']
    classwise=[]
    for k in range(9):
        row=dict(class_index=k, **{'class':k},support=int(support[k]),delta_F1=float(metrics[0,:,k].mean()-metrics[1,:,k].mean()))
        for a,name in enumerate(names):
            row[name]={key:float(np.mean([class_metrics(cms[a,s])[key][k] for s in range(3)])) for key in ['precision','recall','F1']}
        classwise.append(row)
    paired=[]
    for s,seed in enumerate(SEEDS):
        gc=pred[0,s]==truth;uc=pred[1,s]==truth
        paired.append(dict(seed=seed,improvement=int((gc&~uc).sum()),regression=int((~gc&uc).sum()),
                           both_wrong=int((~gc&~uc).sum()),both_correct=int((gc&uc).sum())))
    totals={key:sum(row[key] for row in paired) for key in ['improvement','regression','both_wrong','both_correct']}
    assert sum(totals.values())==3*len(truth)
    directional={name:{d:{key:sum(direction_errors(truth,pred[a,s])[d][key] for s in range(3))
        for key in ['false_connection','missed_connection','wrong_nonzero_grade','absent_count','connected_count']}
        for d in ['LR','TB']} for a,name in enumerate(names)}
    for name in names:
        for direction in ['LR','TB']:
            v=directional[name][direction]
            v.update(false_connection_rate=v['false_connection']/v['absent_count'],
                     missed_connection_rate=v['missed_connection']/v['connected_count'],
                     wrong_nonzero_grade_rate=v['wrong_nonzero_grade']/v['connected_count'])
    examples=example_selection(records,truth,pred)
    plot_classwise(metrics,support);plot_confusions(cms)
    reconstruction=plot_examples(examples,target,masks,x)
    per_seed_directional={name:[dict(seed=seed,directions=direction_errors(truth,pred[a,s])) for s,seed in enumerate(SEEDS)] for a,name in enumerate(names)}
    result=dict(status='COMPLETE_DESCRIPTIVE_ERROR_ANALYSIS',new_fits=0,new_thresholds=0,new_independent_test=False,
                source_lock_sha256=sha(ROOT/'LOCK.json'),examples=6144,frames=24,seeds=SEEDS,
                Macro_F1=dict(zip(names,macro.tolist())),classwise=classwise,
                confusion_per_seed=cms.tolist(),class_F1_per_seed=metrics.tolist(),
                paired=dict(per_seed=paired,totals=totals,unit='patch_seed_pair_not_independent_image'),
                directional=directional,directional_per_seed=per_seed_directional,
                IoU_mean=dict(zip(names,iou.mean((1,2)).tolist())),
                classwise_estimand='metric_per_seed_then_mean_no_pooled_confusion_F1')
    save(ROOT/'RESULT.json',result);save(ROOT/'EXAMPLES.json',dict(rule=EXAMPLE_PREFIX,example_seed=EXAMPLE_SEED,records=examples))
    report(result)
    historical_rows=source_result['rows']
    for a,architecture in enumerate(ARCHITECTURES):
        for s,seed in enumerate(SEEDS):
            row=next(r for r in historical_rows if r['architecture']==architecture and r['seed']==seed)
            assert abs(metrics[a,s].mean()-row['metrics']['final_calibrated']['mask_F1'])<1e-12
    checks=dict(source_closed_confirmation_and_audit_pass=True,probability_threshold_masks_reconstructed=True,
                truth_frames_records_match=True,classwise_means_match_original_scores=True,
                paired_partition_exact=True,native48_to12_reconstruction=True,
                all_four_example_categories_present=True,no_new_fit_or_threshold=True,
                source_bindings_unchanged=all(sha(REPO/p)==v for p,v in lock['source_sha256'].items()))
    assert all(checks.values())
    outputs=[ROOT/'RESULT.json',ROOT/'EXAMPLES.json',ROOT/'REPORT.md']+sorted((ROOT/'figures').glob('*.png'))
    save(ROOT/'CHECKS.json',dict(status='PASS_ERROR_ANALYSIS_CHECKS',checks=checks,checks_passed=len(checks),
                              native_reconstruction=reconstruction,outputs_sha256={relative(p):sha(p) for p in outputs}))
    print(json.dumps(dict(status=result['status'],Macro_F1=result['Macro_F1'],paired=totals,directional=directional,checks=len(checks)),ensure_ascii=False))


if __name__=='__main__': main()
