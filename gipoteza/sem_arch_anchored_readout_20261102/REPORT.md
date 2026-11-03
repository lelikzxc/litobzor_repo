# Обучаемая CNN-поправка к общему физическому readout

У всех четырёх вариантов один физический anchor `a=4·tanh(old_ordinal_logits/4)`. CNN/MLP предсказывают четыре поправки `δ`, итог `b=a+δ`. Для каждого направления три score равны `[0,b0,b0+b1]`. Их softmax дают совместное внешнее произведение внутри вида; после LR/TB-перестановки при нечётном повороте усредняются четыре распределения. Итоговая каноническая смесь не обязана факторизоваться. Класс — argmax9.

Visual_cnn и physics_cnn имеют по **5 372** обучаемых параметра, всего 16 053 с замороженной CNN 10 681. Обе головы используют Conv3×3(26→16), FiLM от четырёх скаляров, Conv3×3(16→8), AvgPool3×3 и Linear(72→4). Последний слой нулевой. Physics_mlp: Linear(4→32), GELU, Linear(32→4), **292** параметра, всего 10 973. Calibrator: `b=exp(u)·a+v`, **8** параметров, всего 10 689; `u=v=0` при старте. Все варианты начинают с одного точно заданного soft-prior.

**Общий anchor физический и у visual_cnn.** Поэтому B сравнивает дополнительное физическое FiLM-conditioning с визуальными статистиками поверх общего anchor. Это не абляция наличия всей физики. MLP и calibrator — обученные непространственные контроли того же anchor. C_spatial проверяет добавочную пользу CNN относительно обоих; число параметров и доступ к пространственным признакам при этом различаются.

## Общий надзор и независимые критерии

Все четыре readout обучаются обычной CE итоговых девяти канонических вероятностей: 1 000 шагов AdamW, lr=0,001, weight_decay=0,0001, batch32. Прежние классы доступны одинаково всем головам; TRAIN — dense18/27 и 1152 weak, 1170/1179 примеров. Пиксельные маски не входят в оптимизатор. Encoder заморожен и ранее обучен с BCE+geometry_loss. Новая исследуемая функция потерь не предлагается. C4-сегментация, маски и IoU побайтно проверяются отдельно от классового argmax.

A_pipeline каждого readout требует среднего ΔF1 ≥0,5 п.п. и ΔF1 каждого бюджета ≥−0,25 п.п. против всех четырёх прежних readout: uniform_frozen, uniform_adapted, single_adapted, fixed_softordinal; отдельно high_lr/low_lr. A против фиксированного правила включает дополнительный классовый надзор и не является чистой атрибуцией эффекта слоя.

B: physics_cnn против visual_cnn, с обязательным A physics_cnn. C_spatial каждой CNN требует её A и тех же приростов над **обоими** обученными physics_mlp и calibrator в обеих LR-семьях. **CNN_SUCCESS** — хотя бы одна CNN прошла C_spatial. Прохождение A только MLP или calibrator не является успехом CNN. Новые seed разрешены лишь после независимого аудита PASS и CNN_SUCCESS; на расширении нужны одновременно все три и новые два seed.

У голов фиксированный argmax без отдельной VAL-калибровки решений; старые масочные контроли имеют пороги, выбранные по Macro-F1 на VAL. Поэтому отрицательный исход относится к этому протоколу сравнения, а не доказывает бесполезность всех возможных калибровок свободных голов. Между четырьмя обучаемыми readout CE, батчи и политика argmax общие.

Начато попыток 2, завершено 1, технических ошибок 1, незавершённых 0. Новых обучений **16**, обновлений 16,000. Прежних обучений 108, новых внешних контролей 0; уникальных обучений вместе **124**. Ссылки на контроли и агрегаты повторно не считаются.

## Сохранённая техническая остановка и восстановление

Первая попытка закончилась AssertionError перед первым low-LR обучением: сравнивались вероятности полного батча695 и отдельного батча16 с требованием побайтного равенства. Она успела сохранить 8 high-LR обучений и 8000 обновлений. Исходная попытка, FAILURE, model.py/runner.py, веса и вероятности сохранены.

[MPS_ROUNDING_CAPTURE.json](MPS_ROUNDING_CAPTURE.json): одинаковые score и воспроизведение полного сохранённого массива; разность probability 1.1920929e-07, log-probability 1.9073486e-06; argmax одинаков. Это воспроизведённое float32-округление при разном размере батча, а не ошибка обучения или изменение критерия качества.

Восстановление [anchored_screen_s5501_recovery_bd6f174c3c](trials/anchored_screen_s5501_recovery_bd6f174c3c/RECOVERY_MANIFEST.json) унаследовало 8 high-LR обучений по SHA и выполнило 8 новых low-LR обучений. Новый recovery.py сравнивает инициализацию при одинаковом layout16, записывая разность с695 в прежнем бюджете8ε: абсолютном для probability и относительном для log-probability. Обучающие функции, данные, параметры оптимизатора, батчи и A/B/C_spatial не менялись. [RECOVERY_PROOF.json](trials/anchored_screen_s5501_recovery_bd6f174c3c/RECOVERY_PROOF.json) проверяет неизменность прежних артефактов и отсутствие повторного high-LR fit.

Итоговые таблицы используют полный результат16 моделей. Учёт включает 16 различных выполненных обучений и 8 ссылок на уже выполненные обучения; эти ссылки не добавляются к числу fit. Обе записи запуска относятся к одной архитектуре и одним критериям, восстановление не объявляется новой научной гипотезой.

## screen: seed [5501]

**CNN_SUCCESS: FAIL.** C_spatial visual_cnn FAIL; physics_cnn FAIL. B дополнительного conditioning FAIL.

| Семья | Visual CNN F1, % | Physics CNN F1, % | Physics MLP F1, % | Calibrator F1, % | A visual / physics / MLP / calib | C visual / physics |
|---|---:|---:|---:|---:|---|---|
| high_lr | 85.85 | 86.29 | 85.62 | 86.05 | FAIL / FAIL / FAIL / FAIL | FAIL / FAIL |
| low_lr | 85.15 | 86.53 | 86.02 | 86.54 | FAIL / FAIL / FAIL / FAIL | FAIL / FAIL |

| Семья | Frozen readout F1, % | C4-adapted F1, % | Single-adapted F1, % | Fixed softordinal F1, % |
|---|---:|---:|---:|---:|
| high_lr | 85.82 | 85.97 | 85.71 | 85.95 |
| low_lr | 85.72 | 86.05 | 85.80 | 86.13 |

| Семья | Голова | Неизменный IoU маски, % | Class/mask несогласие, % |
|---|---|---:|---:|
| high_lr | visual_cnn | 90.26 | 8.85 |
| high_lr | physics_cnn | 90.26 | 8.63 |
| high_lr | physics_mlp | 90.26 | 7.41 |
| high_lr | calibrator | 90.26 | 7.27 |
| low_lr | visual_cnn | 90.13 | 6.04 |
| low_lr | physics_cnn | 90.13 | 6.62 |
| low_lr | physics_mlp | 90.13 | 5.18 |
| low_lr | calibrator | 90.13 | 5.32 |

Заранее заданное условие расширения CNN по seed не выполнено. Положительный A непространственного readout, если он есть, это условие не заменяет.

![Отдельные pipeline, spatial и conditioning сравнения](figures/anchored_classification.png)

Минимум A взят по четырём фиксированным readout. C_spatial дополнительно требует A. Пунктир +0,5 п.п. относится к среднему приросту; бюджетные ограничения сохранены в таблицах. Это точечные результаты многократно использованной VAL, без независимых интервалов подтверждения.

## Инициализация anchor и равенства

Нулевая поправка всех голов даёт один и тот же soft-prior побайтно. Его argmax не объявляется тождественным hard fixed_softordinal: прежнее правило использует `g>=g0` и `g>=0.5`, голова сохраняет обычный argmax с первым индексом при равенстве. Пороговые совпадения и score/rounding ties диагностируются отдельно. Допуск 1e−6 используется только для обозначения близкого score-tie, без смены порогов, классов или критериев.

[INITIAL_PRIOR_DIAGNOSTICS.json](trials/anchored_screen_s5501_recovery_bd6f174c3c/INITIAL_PRIOR_DIAGNOSTICS.json):

| Семья | Бюджет | Seed | Примеров | Начальных class-различий | Exact g-ties | Score/rounding ties | Проверенная parity вне ties |
|---|---:|---:|---:|---:|---:|---:|---|
| high_lr | 18 | 5501 | 695 | 0 | 0 | 0 | PASS |
| high_lr | 27 | 5501 | 695 | 0 | 0 | 0 | PASS |
| low_lr | 18 | 5501 | 695 | 0 | 0 | 0 | PASS |
| low_lr | 27 | 5501 | 695 | 0 | 0 | 0 | PASS |

## Все сохранённые попытки

### anchored_screen_s5501_128a2db5f3

[FROZEN.json](trials/anchored_screen_s5501_128a2db5f3/FROZEN.json).

Техническая ошибка `AssertionError`: . [FAILURE.json](trials/anchored_screen_s5501_128a2db5f3/FAILURE.json). Технический исход отделён от критерия качества.

Сохранённых обучений 8.

### anchored_screen_s5501_recovery_bd6f174c3c

[FROZEN.json](trials/anchored_screen_s5501_recovery_bd6f174c3c/FROZEN.json).

`COMPLETE_CLASSIFICATION_DEVELOPMENT`; screen; seed [5501]; в результате 16 моделей: 8 унаследованы, 8 обучены здесь; новых обновлений 8,000; 73.67 с. [RESULT.json](trials/anchored_screen_s5501_recovery_bd6f174c3c/RESULT.json), SHA-256 `b2779ee722a64154f09b175fdb537a80073cb87e6dc3ea6f097d3dcaf205296f`.

| Семья | Бюджет | Seed | Голова | Параметров | Class F1, % | Accuracy, % | Неизменный IoU, % | Class/mask несогласие, % |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| high_lr | 18 | 5501 | visual_cnn | 5372 | 85.98 | 86.33 | 90.22 | 8.49 |
| high_lr | 18 | 5501 | physics_cnn | 5372 | 85.84 | 86.33 | 90.22 | 8.63 |
| high_lr | 18 | 5501 | physics_mlp | 292 | 86.12 | 86.62 | 90.22 | 7.77 |
| high_lr | 18 | 5501 | calibrator | 8 | 86.78 | 87.19 | 90.22 | 7.63 |
| high_lr | 27 | 5501 | visual_cnn | 5372 | 85.72 | 86.04 | 90.31 | 9.21 |
| high_lr | 27 | 5501 | physics_cnn | 5372 | 86.74 | 87.05 | 90.31 | 8.63 |
| high_lr | 27 | 5501 | physics_mlp | 292 | 85.11 | 85.76 | 90.31 | 7.05 |
| high_lr | 27 | 5501 | calibrator | 8 | 85.31 | 85.90 | 90.31 | 6.91 |
| low_lr | 18 | 5501 | visual_cnn | 5372 | 84.42 | 84.75 | 90.03 | 6.33 |
| low_lr | 18 | 5501 | physics_cnn | 5372 | 86.33 | 86.62 | 90.03 | 7.91 |
| low_lr | 18 | 5501 | physics_mlp | 292 | 85.98 | 86.62 | 90.03 | 4.60 |
| low_lr | 18 | 5501 | calibrator | 8 | 86.61 | 87.05 | 90.03 | 5.04 |
| low_lr | 27 | 5501 | visual_cnn | 5372 | 85.87 | 86.04 | 90.24 | 5.76 |
| low_lr | 27 | 5501 | physics_cnn | 5372 | 86.73 | 87.05 | 90.24 | 5.32 |
| low_lr | 27 | 5501 | physics_mlp | 292 | 86.05 | 86.62 | 90.24 | 5.76 |
| low_lr | 27 | 5501 | calibrator | 8 | 86.48 | 86.91 | 90.24 | 5.61 |

**только эта попытка: [5501]**

| Семья | Сравнение | ΔF1, п.п. | ΔF1 бюджеты 18 / 27, п.п. | Критерий |
|---|---|---:|---|---|
| high_lr | A_pipeline visual_cnn − uniform_frozen | +0.03 | -0.64 / +0.71 | FAIL |
| high_lr | A_pipeline visual_cnn − uniform_adapted | -0.12 | -1.10 / +0.85 | FAIL |
| high_lr | A_pipeline visual_cnn − single_adapted | +0.14 | -0.44 / +0.71 | FAIL |
| high_lr | A_pipeline visual_cnn − fixed_softordinal | -0.10 | -0.14 / -0.07 | FAIL |
| high_lr | A_pipeline physics_cnn − uniform_frozen | +0.47 | -0.77 / +1.72 | FAIL |
| high_lr | A_pipeline physics_cnn − uniform_adapted | +0.32 | -1.23 / +1.87 | FAIL |
| high_lr | A_pipeline physics_cnn − single_adapted | +0.58 | -0.57 / +1.73 | FAIL |
| high_lr | A_pipeline physics_cnn − fixed_softordinal | +0.34 | -0.27 / +0.95 | FAIL |
| high_lr | A_pipeline physics_mlp − uniform_frozen | -0.20 | -0.49 / +0.10 | FAIL |
| high_lr | A_pipeline physics_mlp − uniform_adapted | -0.36 | -0.96 / +0.24 | FAIL |
| high_lr | A_pipeline physics_mlp − single_adapted | -0.10 | -0.30 / +0.10 | FAIL |
| high_lr | A_pipeline physics_mlp − fixed_softordinal | -0.33 | +0.01 / -0.68 | FAIL |
| high_lr | A_pipeline calibrator − uniform_frozen | +0.23 | +0.16 / +0.30 | FAIL |
| high_lr | A_pipeline calibrator − uniform_adapted | +0.07 | -0.30 / +0.44 | FAIL |
| high_lr | A_pipeline calibrator − single_adapted | +0.33 | +0.36 / +0.30 | FAIL |
| high_lr | A_pipeline calibrator − fixed_softordinal | +0.09 | +0.66 / -0.48 | FAIL |
| high_lr | B дополнительного context: physics_cnn − visual_cnn, с A | +0.44 | -0.14 / +1.01 | FAIL |
| high_lr | C_spatial visual_cnn − physics_mlp, с A | +0.23 | -0.14 / +0.61 | FAIL |
| high_lr | C_spatial visual_cnn − calibrator, с A | -0.19 | -0.80 / +0.41 | FAIL |
| high_lr | C_spatial physics_cnn − physics_mlp, с A | +0.67 | -0.28 / +1.62 | FAIL |
| high_lr | C_spatial physics_cnn − calibrator, с A | +0.24 | -0.93 / +1.42 | FAIL |
| low_lr | A_pipeline visual_cnn − uniform_frozen | -0.57 | -1.70 / +0.56 | FAIL |
| low_lr | A_pipeline visual_cnn − uniform_adapted | -0.91 | -2.39 / +0.58 | FAIL |
| low_lr | A_pipeline visual_cnn − single_adapted | -0.65 | -1.82 / +0.51 | FAIL |
| low_lr | A_pipeline visual_cnn − fixed_softordinal | -0.98 | -1.67 / -0.29 | FAIL |
| low_lr | A_pipeline physics_cnn − uniform_frozen | +0.81 | +0.20 / +1.42 | PASS |
| low_lr | A_pipeline physics_cnn − uniform_adapted | +0.47 | -0.49 / +1.43 | FAIL |
| low_lr | A_pipeline physics_cnn − single_adapted | +0.72 | +0.08 / +1.36 | PASS |
| low_lr | A_pipeline physics_cnn − fixed_softordinal | +0.39 | +0.23 / +0.56 | FAIL |
| low_lr | A_pipeline physics_mlp − uniform_frozen | +0.30 | -0.14 / +0.74 | FAIL |
| low_lr | A_pipeline physics_mlp − uniform_adapted | -0.04 | -0.83 / +0.75 | FAIL |
| low_lr | A_pipeline physics_mlp − single_adapted | +0.21 | -0.26 / +0.69 | FAIL |
| low_lr | A_pipeline physics_mlp − fixed_softordinal | -0.12 | -0.11 / -0.12 | FAIL |
| low_lr | A_pipeline calibrator − uniform_frozen | +0.83 | +0.48 / +1.17 | PASS |
| low_lr | A_pipeline calibrator − uniform_adapted | +0.49 | -0.21 / +1.18 | FAIL |
| low_lr | A_pipeline calibrator − single_adapted | +0.74 | +0.36 / +1.12 | PASS |
| low_lr | A_pipeline calibrator − fixed_softordinal | +0.41 | +0.51 / +0.31 | FAIL |
| low_lr | B дополнительного context: physics_cnn − visual_cnn, с A | +1.38 | +1.90 / +0.85 | FAIL |
| low_lr | C_spatial visual_cnn − physics_mlp, с A | -0.87 | -1.56 / -0.18 | FAIL |
| low_lr | C_spatial visual_cnn − calibrator, с A | -1.39 | -2.18 / -0.61 | FAIL |
| low_lr | C_spatial physics_cnn − physics_mlp, с A | +0.51 | +0.34 / +0.68 | PASS |
| low_lr | C_spatial physics_cnn − calibrator, с A | -0.02 | -0.28 / +0.25 | FAIL |

Независимый CPU-аудит: `PASS_AUDIT`, 42 / 42. [AUDIT.json](trials/anchored_screen_s5501_recovery_bd6f174c3c/AUDIT.json).

Примечание аудита: Log-domain parity uses an8-float32-epsilon relative budget plus an independent probability-space check. Anchor corrections form cumulative directional scores and each joint9 distribution before C4 pooling. Exact threshold ties have explicit argmax/inclusive-rule diagnostics; no scientific thresholds are changed.

## Пределы вывода

Это старые TRAIN/VAL, без нового закрытого результата для anchored-модели. Классы производны от SVG-масок и не являются экспертной типологией дефектов или измеренным электрическим состоянием. Физический anchor вычислен из предсказания. Class-argmax может не совпасть с классом неизменной маски; классификационный прирост не является ростом сегментации. Технический PASS отделён от A/B/C_spatial и CNN_SUCCESS.

[Вся предыдущая серия, 108 обучений и сохранённые отрицательные исходы](../sem_arch_directional_readout_20261102/REPORT.md). Цепочка сохраняет исход FAIL_PILOT прежнего равномерного C4 на восьми новых полях той же ИС. Он не переопределяется как проверка новой головы. Для выбранного нового CNN потребуется отдельное закрытое подтверждение. Известные anchor/residual/C4-принципы не объявляются новым открытием; перенос между изделиями и готовность для Q1 не установлены.

[PROTOCOL.md](PROTOCOL.md) · [CONFIG.json](CONFIG.json) · [LEDGER.json](LEDGER.json) · [model.py](model.py) · [runner.py](runner.py) · [audit.py](audit.py).
