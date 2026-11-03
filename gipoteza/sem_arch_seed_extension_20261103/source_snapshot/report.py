"""Read-only reporting of the frozen two-seed extension; no inference or fitting."""
import common as c

def main():
    r=c.read(c.ROOT/'RESULT.json');primary=r['analyses']['final_calibrated']
    lines=['# Дополнительные инициализации GroupCNN и Untied24','',
        f"Статус: `{r['status']}`. Четыре новых обучения, две дополнительные пары seed 5801 и 5821. Вместе с прежними — пять пар. Это репликация по инициализациям на прежних 6144 участках 24 кадров одной ИС; новая независимая тестовая когорта не использовалась.",'',
        '| Группа seed | Group Macro-F1, % | Untied Macro-F1, % | ΔF1, п.п. | ΔAP, п.п. | ΔIoU, п.п. |',
        '|---|---:|---:|---:|---:|---:|']
    for name,title in [('old3','Прежние 3'),('new2','Новые 2'),('all5','Все 5')]:
        v=primary[name];d=v['group_minus_untied']['mean'];cells=v['cells']
        lines.append(f"| {title} | {100*cells['group6']['mean']['mask_F1']:.4f} | {100*cells['untied24_single']['mean']['mask_F1']:.4f} | {100*d['mask_F1']:+.4f} | {100*d['mean_frame_AP']:+.4f} | {100*d['IoU']:+.4f} |")
    lines += ['', '| Seed | ΔF1, п.п. | ΔAP, п.п. | ΔIoU, п.п. |','|---|---:|---:|---:|']
    for row in primary['all5']['group_minus_untied']['per_seed']:
        lines.append(f"| {row['seed']} | {100*row['mask_F1']:+.4f} | {100*row['mean_frame_AP']:+.4f} | {100*row['IoU']:+.4f} |")
    a=primary['all5'];f,a_ci=a['CI97_5_F1'],a['CI97_5_AP']
    lines+=['',f"Приближённые описательные центральные 97.5% интервалы всех пяти seed: ΔF1 [{100*f[0]:+.4f}; {100*f[1]:+.4f}] п.п.; ΔAP [{100*a_ci[0]:+.4f}; {100*a_ci[1]:+.4f}] п.п. Общие парные пересэмплирования seed и кадров, 10000 повторов; поправка на два основных исхода.",
        '',f"Критерии новых двух seed отдельно: `{primary['new2']['descriptive_point_guards']}`.",
        f"Составные описательные критерии всех пяти: `{primary['all5']['descriptive_composite_guards']}`.",'',
        'Новые seed показываются отдельно: положительное среднее всех пяти не заменяет неудачную репликацию новых двух. Исходный закрытый результат сохранён. Модели и процедура калибровки не менялись. Все четыре порога выбраны по тем же 512 классам старых DEV-кадров до новых тестовых предсказаний. Маски и IoU не использовались в выборе порога.',
        '', 'Диагностика при пороге 0.5:', '']
    for name in ['new2','all5']:
        d=r['analyses']['fixed05'][name]['group_minus_untied']['mean'];lines.append(f"- {name}: ΔF1 {100*d['mask_F1']:+.4f} п.п., ΔIoU {100*d['IoU']:+.4f} п.п.")
    lines+=['','Перенос на другую ИС, реальные измеренные электрические дефекты, новизна C4-оператора и превосходство при любом пороге этим экспериментом не установлены. Низкая детализация 12×12 и геометрические классы, полученные из масок, ограничивают область вывода.',
        '', 'Воспроизведение: `runner.py --preflight cpu`, `runner.py --freeze`, независимый `audit.py --pre`, `runner.py --preflight mps`, `runner.py --run`, `evaluate.py --run --device mps`, `audit.py`, `report.py`. Каждая стадия сохраняет веса, вероятности, калибровку, хеши источников и результаты. Повторный запуск в той же папке блокируется.']
    (c.ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (c.ROOT/'README.md').write_text('# Проверка двух дополнительных инициализаций\n\n[Отчёт](REPORT.md) · [Протокол](PROTOCOL.md) · [Результат](RESULT.json) · [Независимый аудит](AUDIT.json)\n')
if __name__=='__main__':main()
