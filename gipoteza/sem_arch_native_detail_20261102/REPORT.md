# Исходная детализация SEM и отдельная CNN-ветвь

Пятый этап последовательной разработки на прежних TRAIN/VAL. Меняется модель при предсказании; прежние BCE и geometry_loss с весом 1 остаются общими. Нового независимого TEST нет. Критерии разработки не доказывают новизну или статистическое превосходство.

К визуальному входу FiLM `[F3, coarse, SEM]` добавлены 24 канала отдельной CNN-ветви: Conv(1→8, 3×3, stride 2, pad 1), GELU, Conv(8→24, 3×3, stride 2, pad 1), GELU. Она переводит 48×48 в 24 канала 12×12; визуальный вход FiLM меняется 26→50 каналов. Новые входные ядра FiLM нулевые при старте, исходные ядра сохраняют начальные значения. Во всех трёх вариантах FiLM имеет 8 993 параметра, новая ветвь — 1 832, новый блок — 10 825, исходная CNN — 10 681, всего 21 506.

Lowres_plain подаёт в новую ветвь прежний SEM12×12, растянутый nearest-neighbor до 48×48. Native_plain использует исходный crop48×48 при обычных картах вероятности/неопределённости. Native_joint использует тот же crop48×48 с картами совместной пиксельной правки. Архитектуры, число параметров, инициализация и батчи совпадают; контроль lowres_plain отделяет исходные детали от новой ёмкости и входного размера. Исходная CNN продолжает получать прежний X12.

## Привязка исходных участков

[NATIVE_LINEAGE.json](NATIVE_LINEAGE.json), SHA-256 `a836033ec739e26d3e29c8a866f643f2a9ef6b97da72aa3d590f788c7c9e904a`, фиксирует 1 901 участок 48×48 из кадров 0–7 по координатам прежних записей TRAIN/VAL. Число точных усреднений 4×4 до прежнего X12: 1901; максимальная ошибка 0. Архив [NATIVE_TRAIN_VAL.npz](NATIVE_TRAIN_VAL.npz), SHA-256 `56fe8d1d0077ef4bb46a14f8faf4fcd679b167e43bf308a8bc2020fd59379f73`. Роли, индексы, классы, маски и поле обзора прежние; это не новые независимые кадры. Для новой ветви native = `5*(uint8/255−0.2)`, lowres = `nearest_x4(5*(X12−0.2))`.

Амплитуда `4*tanh(residual)`, первые 100 шагов только уточнитель, затем 400 шагов совместной адаптации. AdamW: lr CNN 0,0001, lr уточнителя 0,0005, weight_decay 0,0001; батч 8 плотных + 8 слабых. Карты и признаки пересчитываются по текущей CNN; карты отделены от градиента. Начальные параметры и батчи трёх вариантов совпадают.

Независимая проверка исходных участков: `PASS_NATIVE_DATA_AUDIT`, 7 / 7. [NATIVE_AUDIT.json](NATIVE_AUDIT.json). Технический PASS подтверждает привязку данных и не заменяет A/B/C.

## Объём поиска и общие контроли

Локально начато попыток 1; завершено 1; технических ошибок 0; незавершённых 0. Сохранено 6 новых обучений уточнителей, 3,000 обновлений. CNN-контроли внутри этих попыток не переучиваются.

Четыре предыдущих этапа содержат 8 попыток и 46 уникальных обучений с общим высокоскоростным CNN-контролем. Уникальных обучений вместе с текущим этапом: **52**. Дополнительных low-lr контролей вне предыдущих попыток: 0. Копии low_lr_controls/high_lr_controls и AGGREGATE_RESULT не считаются новыми обучениями.

Общий CNN lr=0,0005: [RESULT.json](../sem_arch_iterative_20261102/baseline_control/RESULT.json), SHA-256 `51772b1444ca376d4e89b406e45e6350910dbb232fd05cfcc9f5443913e571ee`. CNN lr=0,0001 берётся из явно указанных сохранённых источников:

- [../sem_arch_joint_adaptation_20261102/trials/film_screen_s5501_5e8c45e83c/RESULT.json](../sem_arch_joint_adaptation_20261102/trials/film_screen_s5501_5e8c45e83c/RESULT.json), SHA-256 `8b308fb9f2edb70517b5a87cb67522a4acb6b2f31be5fa166d77d878f5cb7a8e`.

Сравниваются точные пары **(бюджет, seed)**; проверены SHA исходных CNN, последовательности батчей и 500 шагов, а также совпадение сохранённых копий контролей. Контроль другого seed не подставляется при отсутствии пары.

## Раздельные критерии A/B/C

A для каждого из lowres_plain/native_plain/native_joint требует выигрыша относительно **каждой** из двух дообученных CNN. B сравнивает native_joint с native_plain и требует прохождения A native_joint. C сравнивает native_plain с lowres_plain. Вывод о пользе исходных деталей требует одновременно C и A native_plain. Числа A/B/C и условие вывода о детализации публикуются раздельно. Для каждого сравнения: средний ΔF1 ≥0,5 п.п.; ΔF1 каждого бюджета ≥−0,25 п.п.; средний ΔIoU ≥−0,2 п.п. Бюджеты 18/27 имеют равный вес. При расширении публикуются все три и новые два seed отдельно. Прежние критерии предыдущих этапов сохраняются без пересмотра.

| Набор seed | Low-lr CNN F1, % | High-lr CNN F1, % | Lowres plain F1, % | Native plain F1, % | Native joint F1, % | A lowres / native plain / native joint | B | C / вывод о деталях |
|---|---:|---:|---:|---:|---:|---|---|---|
| screen: [5501]: [5501] | 84.372 | 84.193 | 84.007 | 84.006 | 84.008 | FAIL / FAIL / FAIL | FAIL | FAIL / FAIL |

![Средние VAL и раздельные сравнения](figures/native_detail_validation.png)

Точечные средние без интервалов независимого подтверждения. При наличии расширения средние всех трёх и новых двух seed перекрываются по данным.

## Все сохранённые попытки этого этапа

### film_native_detail_screen_s5501_6c7f8b71e7

[FROZEN.json](trials/film_native_detail_screen_s5501_6c7f8b71e7/FROZEN.json).

`COMPLETE_DEVELOPMENT_TRIAL`; screen; seed [5501]; 3,000 обновлений; 78.60 с. [RESULT.json](trials/film_native_detail_screen_s5501_6c7f8b71e7/RESULT.json), SHA-256 `00872ee41e9ec042151fce9fa644091ce28232a1437f70c59a098a7f356875d5`.

| Бюджет | Seed | Вариант / источник | Порог VAL | Macro-F1, % | IoU, % | Шагов CNN / нового блока | Zero ΔF1, п.п. | Roll ΔF1, п.п. |
|---:|---:|---|---:|---:|---:|---|---:|---:|
| 18 | 5501 | lowres_plain / upsampled_X12 | 0.45 | 83.581 | 89.385 | 400 / 500 | +0.000 | +0.313 |
| 18 | 5501 | native_plain / native48 | 0.45 | 83.581 | 89.390 | 400 / 500 | +0.000 | +0.313 |
| 18 | 5501 | native_joint / native48 | 0.30 | 83.707 | 89.367 | 400 / 500 | +0.037 | -0.169 |
| 27 | 5501 | lowres_plain / upsampled_X12 | 0.30 | 84.432 | 89.698 | 400 / 500 | -0.114 | +0.116 |
| 27 | 5501 | native_plain / native48 | 0.30 | 84.430 | 89.706 | 400 / 500 | -0.112 | -0.003 |
| 27 | 5501 | native_joint / native48 | 0.30 | 84.309 | 89.650 | 400 / 500 | +0.000 | +0.000 |

**только эта попытка: [5501]**

| Сравнение | ΔF1, п.п. | ΔIoU, п.п. | ΔF1 по бюджетам 18 / 27, п.п. | Критерий |
|---|---:|---:|---|---|
| A lowres_plain − CNN lr=0,0001 | -0.366 | +0.025 | -0.949 / +0.218 | FAIL |
| A lowres_plain − CNN lr=0,0005 | -0.186 | +0.209 | -0.963 / +0.591 | FAIL |
| A native_plain − CNN lr=0,0001 | -0.367 | +0.031 | -0.949 / +0.215 | FAIL |
| A native_plain − CNN lr=0,0005 | -0.187 | +0.215 | -0.963 / +0.588 | FAIL |
| A native_joint − CNN lr=0,0001 | -0.364 | -0.008 | -0.823 / +0.094 | FAIL |
| A native_joint − CNN lr=0,0005 | -0.185 | +0.176 | -0.837 / +0.467 | FAIL |
| B native_joint − native_plain, с требованием A native_joint | +0.003 | -0.039 | +0.126 / -0.121 | FAIL |
| C native_plain − lowres_plain | -0.001 | +0.007 | +0.000 / -0.002 | FAIL |

Условие вывода о детализации (C и A native_plain): **FAIL**.

Независимый CPU-аудит: `PASS_AUDIT`; проверок 44 / 44; [AUDIT.json](trials/film_native_detail_screen_s5501_6c7f8b71e7/AUDIT.json).

## Предыдущие попытки адаптивного поиска

Все предыдущие отрицательные варианты остаются частью поиска. У этапов 1/2 A сравнивал с замороженной CNN; у этапов 3/4 — с двумя дообученными CNN. Эти критерии не смешиваются.

| Этап / попытка | Seed | Сохранённых обучений | F1 каждого варианта, % | Прежний A каждого варианта | B / C |
|---|---|---:|---|---|---|
| [sem_arch_iterative_20261102 / direct_screen_s5501_fc50a59fc2](../sem_arch_iterative_20261102/trials/direct_screen_s5501_fc50a59fc2/RESULT.json) | [5501] | 4 | plain: 83.906; joint: 83.990 | plain: FAIL; joint: FAIL | FAIL / — |
| [sem_arch_iterative_20261102 / split_screen_s5501_fc50a59fc2](../sem_arch_iterative_20261102/trials/split_screen_s5501_fc50a59fc2/RESULT.json) | [5501] | 4 | plain: 83.789; joint: 83.952 | plain: FAIL; joint: FAIL | FAIL / — |
| [sem_arch_iterative_20261102 / film_screen_s5501_fc50a59fc2](../sem_arch_iterative_20261102/trials/film_screen_s5501_fc50a59fc2/RESULT.json) | [5501] | 4 | plain: 83.986; joint: 84.105 | plain: FAIL; joint: FAIL | FAIL / — |
| [sem_arch_iterative_20261102 / gated_screen_s5501_fc50a59fc2](../sem_arch_iterative_20261102/trials/gated_screen_s5501_fc50a59fc2/RESULT.json) | [5501] | 4 | plain: 83.664; joint: 83.554 | plain: FAIL; joint: FAIL | FAIL / — |
| [sem_arch_amplitude4_20261102 / film_screen_s5501_0939dc15b2](../sem_arch_amplitude4_20261102/trials/film_screen_s5501_0939dc15b2/RESULT.json) | [5501] | 4 | plain: 84.040; joint: 84.269 | plain: FAIL; joint: PASS | FAIL / — |
| [sem_arch_amplitude4_20261102 / film_replicate_s5521-5557_0939dc15b2](../sem_arch_amplitude4_20261102/trials/film_replicate_s5521-5557_0939dc15b2/RESULT.json) | [5521, 5557] | 8 | plain: 83.116; joint: 83.302 | plain: PASS; joint: PASS | FAIL / — |
| [sem_arch_joint_adaptation_20261102 / film_screen_s5501_5e8c45e83c](../sem_arch_joint_adaptation_20261102/trials/film_screen_s5501_5e8c45e83c/RESULT.json) | [5501] | 6 | continued_base_low_lr: 84.372; plain: 84.336; joint: 83.981 | plain: FAIL; joint: FAIL | FAIL / — |
| [sem_arch_early_skip_20261102 / film_early_skip_screen_s5501_db118a321a](../sem_arch_early_skip_20261102/trials/film_early_skip_screen_s5501_db118a321a/RESULT.json) | [5501] | 6 | plain: 84.119; joint: 84.282; late_joint: 84.265 | plain: FAIL; joint: FAIL; late_joint: FAIL | FAIL / FAIL |

## Границы вывода

Те же TRAIN/VAL: пул 54 плотных масок, 1152 слабых примера, 695 VAL, бюджеты 18/27, розыгрыш 5303. Слабые маски оптимизатору не передаются. Девять классов рассчитаны условной решёткой. Порог каждой модели выбирается на VAL из [0,30; 0,40; 0,45; 0,50; 0,55; 0,60; 0,70] по F1, затем IoU, близости к 0,5 и меньшему порогу. Zero/roll используют фиксированный actual-порог.

При B FAIL добавочная польза физических карт по заданному критерию не установлена. При провале условия C и A native_plain выигрыш нельзя приписывать исходной детализации. Чувствительность к картам не доказывает их пользу. Для независимого подтверждения нужны заранее зафиксированные модель/критерии и закрытые новые данные, не использованные в EDA, разработке или прежних тестах. Такой проверки здесь нет. Перенос на другие изделия и новизна рассматриваются отдельно.

[PROTOCOL.md](PROTOCOL.md) · [CONFIG.json](CONFIG.json) · [LEDGER.json](LEDGER.json) · [model.py](model.py) · [runner.py](runner.py) · [audit.py](audit.py).
