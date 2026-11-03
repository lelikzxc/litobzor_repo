# Обучаемый CNN-блок объединения четырёх видов

Новая обучаемая часть модели — общий для четырёх видов CNN-блок `Conv1×1(29→8), GELU, Conv1×1(8→1)` с **249 параметрами**. Он вычисляет пространственные score; softmax по видам задаёт веса выпуклой суммы вероятностей. Исходная CNN с 10 681 параметром заморожена; всего у learned-модели 10 930 параметров. Нулевой последний слой даёт точное начальное совпадение с равномерным C4. На предсказании нужны четыре прохода исходной CNN и малый общий scorer. C4 и обучаемое объединение видов — известные общие принципы; новизна предметной конструкции здесь не объявляется доказанной.

29 входов scorer: 24 согласованных признака вида, его вероятность, равномерное среднее, SEM и две условные карты. Learned_plain получает вероятность/неопределённость; learned_joint — совместные DtN-карты из равномерного C4. Карты отделены от градиента; нормировка log1p100, центрирование/RMS, стабилизатор 1e−6. Внутри источника plain/joint имеют одинаковые начальные параметры и батчи. Общие 1×1-операции сохраняют согласованность перестановки видов; C4-эквивариантность проверяется отдельно от качества.

## Поиск и учёт обучений

Локальных попыток 1; завершено 1; технических ошибок 0; незавершённых 0. Сохранённых оценочных строк 24; новых обучений **16**, обновлений 8,000. Два замороженных выхода не считаются обучениями. Строки сводного расширения повторно не считаются.

Предыдущий поиск и контроли содержат 56 уникальных обучений; вместе с этим этапом **72**. Старые источники и результаты сохраняются: [полный журнал прежних архитектурных попыток](../sem_arch_orbit_pool_20261102/REPORT.md).

## Общее обучение и сильные контроли

Прежний лосс: BCE итоговой плотной маски и geometry_loss с весом 1 на слабых метках. 500 шагов AdamW, batch 8 dense + 8 weak, weight_decay 0,0001. У learned обучаются только 249 параметров с lr=0,0005. Uniform_adapted обучает все 10 681 параметр CNN 500 шагов с lr=0,0001 через равномерный C4 и тот же лосс. Stable logsumexp-логит смеси — численная реализация прежнего лосса. Плотные бюджеты 18/27, розыгрыш 5303, 1152 weak и 695 VAL прежние; weak-маски оптимизатору не передаются. Кеш допустим только для замороженных CNN; адаптируемые контроли пересчитывают проход. Single_adapted получает те же 500 шагов с lr=0,0001 и теми же батчами через один вид; оценивается также через C4. Uniform_adapted использует 4 CNN-прохода на обучающее изображение, single_adapted — 1; число шагов одинаково, вычислительная стоимость обучения различается. Этот контроль отделяет обучение через C4 от одного дополнительного дообучения CNN.

A проверяется отдельно для каждого learned-варианта против **всех трёх** uniform_frozen, uniform_adapted и single_adapted, отдельно в high_lr и low_lr. Все шесть сравнений должны пройти: средний ΔF1 ≥0,5 п.п.; ΔF1 каждого бюджета ≥−0,25 п.п.; средний ΔIoU ≥−0,2 п.п. B — learned_joint против learned_plain в обеих семьях, с требованием A joint. Source_identity публикуется как дополнительный исход, а не достаточный контроль для A. Все шесть выходов используют одну сетку 81 порога 0,10…0,90 с шагом 0,01: F1, IoU, близость к 0,5, меньший порог. Бюджеты равновесны. При расширении нужны одновременно все три и новые два seed отдельно.

**D, отдельная проверка обучения CNN:** uniform_adapted против обоих uniform_frozen и single_adapted, с теми же условиями, в обеих семьях. D относится к изменённым весам CNN при обучении через C4; его прохождение не подтверждает пользу scorer с 249 параметрами и не изменяет A/B. Критерий D и single-view контроль записаны до первого обучения.

| Набор seed | Семья | Identity F1, % | Frozen C4 F1, % | Learned plain F1, % | Learned joint F1, % | Adapted C4 F1, % | Single-adapted F1, % | A plain / joint | B | D |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| screen: [5501] | high_lr | 84.46 | 85.82 | 85.92 | 85.92 | 85.97 | 85.71 | FAIL / FAIL | FAIL | FAIL |
| screen: [5501] | low_lr | 84.63 | 85.72 | 85.90 | 85.85 | 86.05 | 85.80 | FAIL / FAIL | FAIL | FAIL |

Совместно обе семьи: A plain **FAIL**, A joint **FAIL**, B **FAIL**; отдельно D **FAIL**.

![Все выходы и раздельные сравнения с сильными контролями](figures/learned_orbit_validation.png)

Это точечные средние адаптивной разработки на повторно используемой VAL. Независимые интервалы подтверждения не рассчитываются; средние всех трёх и новых двух seed при расширении перекрываются по данным.

![D: обучение CNN через C4 и matched single-view контроль](figures/uniform_training_validation.png)

D публикуется отдельно от A/B обучаемого scorer. На рисунке средние ΔF1; ограничения по бюджетам и IoU приведены ниже.

## Все сохранённые попытки

### fusion_screen_s5501_9aea0b2f17

[FROZEN.json](trials/fusion_screen_s5501_9aea0b2f17/FROZEN.json).

`COMPLETE_DEVELOPMENT_TRIAL`; screen; seed [5501]; обучений 16, оценочных строк 24, обновлений 8,000; полное время 177.31 с. [RESULT.json](trials/fusion_screen_s5501_9aea0b2f17/RESULT.json), SHA-256 `ea9beefc77f1a071eaad1cfdc23cd42ee7e6fa2ddea7c89b7df3f797c15d5603`.

| Семья | Бюджет | Seed | Вариант | Обучаемых параметров | Шагов | Порог | F1, % | IoU, % | Zero ΔF1, п.п. | Roll ΔF1, п.п. |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| high_lr | 18 | 5501 | source_identity | 0 | 0 | 0.34 | 84.64 | 89.23 | — | — |
| high_lr | 18 | 5501 | uniform_frozen | 0 | 0 | 0.51 | 86.62 | 90.22 | — | — |
| high_lr | 18 | 5501 | learned_plain | 249 | 500 | 0.39 | 86.92 | 90.00 | +0.00 | +0.00 |
| high_lr | 18 | 5501 | learned_joint | 249 | 500 | 0.39 | 86.92 | 90.01 | +0.00 | +0.00 |
| high_lr | 18 | 5501 | uniform_adapted | 10681 | 500 | 0.36 | 87.08 | 90.13 | — | — |
| high_lr | 18 | 5501 | single_adapted | 10681 | 500 | 0.49 | 86.42 | 90.30 | — | — |
| high_lr | 27 | 5501 | source_identity | 0 | 0 | 0.25 | 84.29 | 89.43 | — | — |
| high_lr | 27 | 5501 | uniform_frozen | 0 | 0 | 0.48 | 85.02 | 90.31 | — | — |
| high_lr | 27 | 5501 | learned_plain | 249 | 500 | 0.48 | 84.92 | 90.26 | +0.00 | +0.00 |
| high_lr | 27 | 5501 | learned_joint | 249 | 500 | 0.48 | 84.92 | 90.25 | +0.00 | +0.00 |
| high_lr | 27 | 5501 | uniform_adapted | 10681 | 500 | 0.47 | 84.87 | 90.32 | — | — |
| high_lr | 27 | 5501 | single_adapted | 10681 | 500 | 0.49 | 85.01 | 90.37 | — | — |
| low_lr | 18 | 5501 | source_identity | 0 | 0 | 0.28 | 84.87 | 89.29 | — | — |
| low_lr | 18 | 5501 | uniform_frozen | 0 | 0 | 0.40 | 86.12 | 90.03 | — | — |
| low_lr | 18 | 5501 | learned_plain | 249 | 500 | 0.42 | 86.56 | 90.16 | -0.23 | -0.55 |
| low_lr | 18 | 5501 | learned_joint | 249 | 500 | 0.41 | 86.34 | 90.15 | -0.31 | -0.12 |
| low_lr | 18 | 5501 | uniform_adapted | 10681 | 500 | 0.36 | 86.81 | 90.13 | — | — |
| low_lr | 18 | 5501 | single_adapted | 10681 | 500 | 0.41 | 86.25 | 90.10 | — | — |
| low_lr | 27 | 5501 | source_identity | 0 | 0 | 0.28 | 84.39 | 89.67 | — | — |
| low_lr | 27 | 5501 | uniform_frozen | 0 | 0 | 0.33 | 85.31 | 90.24 | — | — |
| low_lr | 27 | 5501 | learned_plain | 249 | 500 | 0.44 | 85.23 | 90.35 | +0.12 | +0.12 |
| low_lr | 27 | 5501 | learned_joint | 249 | 500 | 0.44 | 85.36 | 90.34 | +0.00 | +0.00 |
| low_lr | 27 | 5501 | uniform_adapted | 10681 | 500 | 0.30 | 85.30 | 90.27 | — | — |
| low_lr | 27 | 5501 | single_adapted | 10681 | 500 | 0.46 | 85.36 | 90.37 | — | — |

**только эта попытка: [5501]**

| Семья | Сравнение | ΔF1, п.п. | ΔIoU, п.п. | ΔF1 бюджеты 18 / 27, п.п. | Критерий |
|---|---|---:|---:|---|---|
| high_lr | A learned_plain − uniform_frozen | +0.10 | -0.13 | +0.31 / -0.10 | FAIL |
| high_lr | A learned_plain − uniform_adapted | -0.05 | -0.09 | -0.15 / +0.05 | FAIL |
| high_lr | A learned_plain − single_adapted | +0.21 | -0.21 | +0.51 / -0.09 | FAIL |
| high_lr | A learned_joint − uniform_frozen | +0.10 | -0.13 | +0.31 / -0.10 | FAIL |
| high_lr | A learned_joint − uniform_adapted | -0.05 | -0.09 | -0.15 / +0.05 | FAIL |
| high_lr | A learned_joint − single_adapted | +0.21 | -0.20 | +0.51 / -0.09 | FAIL |
| high_lr | B learned_joint − learned_plain, с A joint | +0.00 | +0.00 | +0.00 / +0.00 | FAIL |
| high_lr | D uniform_adapted − uniform_frozen | +0.16 | -0.04 | +0.46 / -0.15 | FAIL |
| high_lr | D uniform_adapted − single_adapted | +0.26 | -0.11 | +0.66 / -0.14 | FAIL |
| low_lr | A learned_plain − uniform_frozen | +0.18 | +0.12 | +0.44 / -0.08 | FAIL |
| low_lr | A learned_plain − uniform_adapted | -0.16 | +0.05 | -0.25 / -0.06 | FAIL |
| low_lr | A learned_plain − single_adapted | +0.09 | +0.02 | +0.31 / -0.13 | FAIL |
| low_lr | A learned_joint − uniform_frozen | +0.13 | +0.11 | +0.22 / +0.05 | FAIL |
| low_lr | A learned_joint − uniform_adapted | -0.20 | +0.05 | -0.47 / +0.06 | FAIL |
| low_lr | A learned_joint − single_adapted | +0.05 | +0.01 | +0.10 / -0.01 | FAIL |
| low_lr | B learned_joint − learned_plain, с A joint | -0.05 | -0.01 | -0.21 / +0.12 | FAIL |
| low_lr | D uniform_adapted − uniform_frozen | +0.34 | +0.07 | +0.69 / -0.02 | FAIL |
| low_lr | D uniform_adapted − single_adapted | +0.25 | -0.03 | +0.57 / -0.07 | FAIL |

Независимый CPU-аудит: `PASS_AUDIT`, 27 / 27. [AUDIT.json](trials/fusion_screen_s5501_9aea0b2f17/AUDIT.json).

Примечание технического аудита: The compensated exact-uniform-initialization probability expression and stable logsumexp logit are mathematically identical. Float32 arithmetic produced a4.17e-7 parity gap and2.38e-7 range excess; an8-epsilon arithmetic budget replaces the preliminary3e-7 audit bound. Initial failed audit is preserved. Float64 identity is checked independently; no model, data, protocol, scientific threshold or gate changed.

Предыдущая техническая проверка: [AUDIT_PREVIOUS_09e3034bdc61.json](trials/fusion_screen_s5501_9aea0b2f17/AUDIT_PREVIOUS_09e3034bdc61.json), `FAIL_AUDIT`. Результаты обучения сохранены.

## Границы вывода

В этом этапе используются только прежние TRAIN/VAL. Он не добавляет независимой проверки learned-модели. Прохождение A без B не позволяет приписывать выигрыш физическим картам; вмешательства Zero/roll используют выбранный actual-порог и проверяют чувствительность, которая сама по себе не доказывает пользу.

Технический PASS относится к воспроизведению реализации и чисел; критерии A/B/D отдельно проверяют пользу соответствующих изменений.

Предыдущая отложенная проверка **равномерного C4**: `FAIL_PILOT`; [неизменный отчёт](../sem_arch_orbit_confirmation_20261102/REPORT.md). Это отдельная уже завершённая проверка на восьми полях той же ИС; её исход не переименовывается в результат нового scorer. Для нового обучаемого блока потребуется отдельное закрытое подтверждение. Перенос между изделиями, экспертная типология дефектов и новизна требуют отдельных оснований.

[PROTOCOL.md](PROTOCOL.md) · [CONFIG.json](CONFIG.json) · [LEDGER.json](LEDGER.json) · [model.py](model.py) · [runner.py](runner.py) · [audit.py](audit.py).
