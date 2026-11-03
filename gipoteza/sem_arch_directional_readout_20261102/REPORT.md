# Направленная факторизация классификационного выхода CNN

**Единственное изменение относительно flat9 — выход последнего слоя:** шесть логитов `[LR0, LR1, LR2, TB0, TB1, TB2]` вместо девяти свободных. Внутри каждого вида две трёхклассовые softmax задают `P_view(h,v) = P_LR(h)·P_TB(v)`, класс `3h+v`. После обратной перестановки классов усредняются четыре таких совместных распределения. **Итоговая смесь видов не обязана факторизоваться.** Усреднение двух маргиналей с последующим перемножением было бы другой моделью. Итоговый класс — argmax канонических девяти вероятностей, а не класс пороговой маски. Сегментационная C4-карта исходной CNN заморожена и проверяется побайтно; рост классификационного F1 не является ростом IoU или улучшением маски. Класс головы может не совпадать с геометрическим классом её маски; это измеряется отдельно.

Visual_cnn и physics_cnn имеют одинаковые **5 518** обучаемых параметров: Conv3×3(26→16), GELU; FiLM(4→32); Conv3×3(16→8), GELU; AvgPool3×3; Linear(72→6). С замороженной CNN 10 681 это 16 199 параметров. Вход26 — признаки24, C4-вероятность и SEM. Общие веса головы обрабатывают четыре вида. При поворотах 90/270 градусов классовые вероятности переставляются `[0,3,6,1,4,7,2,5,8]`, меняя LR/TB; затем усредняются вероятности, не номера классов. Проверяется C4-ковариантность с перестановкой меток.

Physics_cnn получает четыре tanh(ordinal_logits/4) от предсказанных непрерывных gLR/gTB по заострённой C4-карте μ⁴/(μ⁴+(1−μ)⁴). Visual_cnn получает четыре визуальные статистики в сопоставимом диапазоне: LR/TB-границы, среднюю вероятность и неопределённость. Physics_mlp — Linear(4→32), GELU, Linear(32→6), **358** параметров, всего 11 039; он получает тот же физический контекст. C сравнивает CNN со значительно меньшим MLP и изменяет также доступ к пространственным признакам.

## Надзор, лосс и границы сравнения

Новые головы обучаются заново; обученные flat9-параметры не используются как warm-start. Источники, признаки, контексты, классы и батчи сохранены. Для всех трёх голов используется та же **стандартная CE** канонического девятиклассового распределения, 1 000 шагов AdamW, lr=0,001, weight_decay=0,0001, batch32. Совокупный TRAIN — dense18/27 и 1152 weak, 1170/1179 примеров; одинаковые классы и батчи доступны всем головам. Пиксельные маски ни dense, ни weak не передаются оптимизатору. Encoder заморожен; его прежние веса обучены с BCE+geometry_loss. Это архитектурная факторизация выхода под общей CE; две отдельные направленные CE или новая штрафная функция не вводятся.

A_pipeline против фиксированного readout включает также дополнительное обучение головы по классам. Он не изолирует чистую замену одного слоя. B сравнивает две одинаковые CNN под одинаковой CE и проверяет физический контекст. C сравнивает physics_cnn с physics_mlp при той же CE; ёмкость и пространственный вход этих моделей различаются.

Начато попыток 1, завершено 1, технических ошибок 0, незавершённых 0. Новых обучений голов **12**, обновлений 12,000. С прежними 96 и дополнительными внешними контролями 0 уникальных обучений **108**. Сводные результаты и сохранённые контроли повторно не считаются.

## Критерии классификационного тракта

A_pipeline требует ΔMacro-F1 ≥0,5 п.п. в среднем и ΔF1 каждого бюджета ≥−0,25 п.п. против каждого из четырёх readout: uniform_frozen, uniform_adapted, single_adapted и fixed_softordinal(g0,0,5), отдельно в high_lr/low_lr. Старые масочные readout сохраняют свои выбранные по 81-точечной VAL-сетке пороги. У новых классовых голов порога нет: argmax9. IoU не включён как мнимый новый выигрыш; маска обязана остаться неизменной. B: physics_cnn против visual_cnn и A physics_cnn. C: physics_cnn против physics_mlp и A physics_cnn. A для MLP не устанавливает пользу пространственной CNN. При расширении отдельно нужны условия на всех трёх и новых двух seed. Повторы разрешены только после независимого аудита PASS и хотя бы одного A_pipeline на исходном seed; B/C уже требуют A. Если все A отрицательны, новые seed в этой серии не добавляются ради поиска положительного среднего. Нужные парные контроли должны существовать заранее.

## screen: seed [5501]

| Семья | Visual CNN F1, % | Physics CNN F1, % | Physics MLP F1, % | A visual / physics / MLP | B | C |
|---|---:|---:|---:|---|---|---|
| high_lr | 78.53 | 83.94 | 85.63 | FAIL / FAIL / FAIL | FAIL | FAIL |
| low_lr | 77.37 | 85.30 | 86.12 | FAIL / FAIL / FAIL | FAIL | FAIL |

Обе семьи: A visual_cnn **FAIL**; A physics_cnn **FAIL**; A physics_mlp **FAIL**; B **FAIL**; C **FAIL**.

| Семья | Frozen mask-readout F1, % | C4-adapted mask-readout F1, % | Single-adapted mask-readout F1, % | Fixed softordinal F1, % |
|---|---:|---:|---:|---:|
| high_lr | 85.82 | 85.97 | 85.71 | 85.95 |
| low_lr | 85.72 | 86.05 | 85.80 | 86.13 |

| Семья | Голова | Неизменный IoU маски, % | Несогласие class-argmax / class-of-mask, % |
|---|---|---:|---:|
| high_lr | visual_cnn | 90.26 | 17.84 |
| high_lr | physics_cnn | 90.26 | 11.37 |
| high_lr | physics_mlp | 90.26 | 7.55 |
| low_lr | visual_cnn | 90.13 | 17.19 |
| low_lr | physics_cnn | 90.13 | 10.14 |
| low_lr | physics_mlp | 90.13 | 5.11 |

Заранее заданное условие расширения по seed не выполнено: ни одна голова не прошла A_pipeline. Скрининг не объявляется подтверждённым эффектом.

high_lr: численные условия сравнения physics_cnn с visual_cnn пройдены (ΔF1 +5.40 п.п.), но полный B остаётся FAIL, поскольку обязательный A_pipeline physics_cnn не пройден. Разница внутри пары CNN и выигрыш над сильными readout — разные сравнения.

low_lr: численные условия сравнения physics_cnn с visual_cnn пройдены (ΔF1 +7.93 п.п.), но полный B остаётся FAIL, поскольку обязательный A_pipeline physics_cnn не пройден. Разница внутри пары CNN и выигрыш над сильными readout — разные сравнения.

## Историческое сопоставление с flat9

Это описательные разности по точно совпадающим source_lr/бюджету/seed/голове при тех же источниках, CE и числе шагов. Flat9 сохраняется историческим ориентиром и не добавляется к A/B/C после результата. CNN-голова меняется 5737→5518 параметров, MLP457→358: изменение относится к выходной структуре.

| Набор seed | Семья | Голова | Flat9 F1, % | Направленный F1, % | ΔF1, п.п. |
|---|---|---|---:|---:|---:|
| screen | high_lr | visual_cnn | 75.47 | 78.53 | +3.06 |
| screen | high_lr | physics_cnn | 84.85 | 83.94 | -0.91 |
| screen | high_lr | physics_mlp | 85.32 | 85.63 | +0.31 |
| screen | low_lr | visual_cnn | 75.79 | 77.37 | +1.58 |
| screen | low_lr | physics_cnn | 86.11 | 85.30 | -0.81 |
| screen | low_lr | physics_mlp | 85.96 | 86.12 | +0.16 |

![Направленный классовый выход и отдельно неизменная сегментация](figures/directional_classification_and_frozen_mask.png)

Слева/сверху — классификационные метрики; IoU показан отдельно. Минимальный прирост A берётся по четырём контролям. Пунктир +0,5 п.п. относится к среднему F1, бюджетные ограничения приведены ниже. Это повторная VAL без независимых интервалов подтверждения.

## Все сохранённые попытки

### directional_screen_s5501_2b0ea20c34

[FROZEN.json](trials/directional_screen_s5501_2b0ea20c34/FROZEN.json).

`COMPLETE_CLASSIFICATION_DEVELOPMENT`; screen; seed [5501]; обучений 12, обновлений 12,000; новых оценок 12; старых масочных контролей 12, фиксированных softordinal 4; 123.31 с. [RESULT.json](trials/directional_screen_s5501_2b0ea20c34/RESULT.json), SHA-256 `94cf5560bda0adaac8274ec6d34886ffd4ba2c68cba65a10d60194163b1c8360`.

| Семья | Бюджет | Seed | Голова | Параметров головы | Class F1, % | Accuracy, % | Неизменный mask IoU, % | Class/mask несогласие, % |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| high_lr | 18 | 5501 | visual_cnn | 5518 | 79.20 | 79.28 | 90.22 | 16.55 |
| high_lr | 18 | 5501 | physics_cnn | 5518 | 84.68 | 84.75 | 90.22 | 9.78 |
| high_lr | 18 | 5501 | physics_mlp | 358 | 86.12 | 86.62 | 90.22 | 7.77 |
| high_lr | 27 | 5501 | visual_cnn | 5518 | 77.87 | 77.84 | 90.31 | 19.14 |
| high_lr | 27 | 5501 | physics_cnn | 5518 | 83.20 | 83.88 | 90.31 | 12.95 |
| high_lr | 27 | 5501 | physics_mlp | 358 | 85.13 | 85.76 | 90.31 | 7.34 |
| low_lr | 18 | 5501 | visual_cnn | 5518 | 76.43 | 76.69 | 90.03 | 15.97 |
| low_lr | 18 | 5501 | physics_cnn | 5518 | 85.55 | 85.61 | 90.03 | 10.22 |
| low_lr | 18 | 5501 | physics_mlp | 358 | 86.39 | 86.91 | 90.03 | 4.60 |
| low_lr | 27 | 5501 | visual_cnn | 5518 | 78.31 | 78.27 | 90.24 | 18.42 |
| low_lr | 27 | 5501 | physics_cnn | 5518 | 85.05 | 85.61 | 90.24 | 10.07 |
| low_lr | 27 | 5501 | physics_mlp | 358 | 85.85 | 86.47 | 90.24 | 5.61 |

**только эта попытка: [5501]**

| Семья | Сравнение классификации | ΔF1, п.п. | ΔF1 бюджеты 18 / 27, п.п. | Критерий |
|---|---|---:|---|---|
| high_lr | A_pipeline visual_cnn − uniform_frozen | -7.28 | -7.42 / -7.15 | FAIL |
| high_lr | A_pipeline visual_cnn − uniform_adapted | -7.44 | -7.88 / -7.00 | FAIL |
| high_lr | A_pipeline visual_cnn − single_adapted | -7.18 | -7.22 / -7.14 | FAIL |
| high_lr | A_pipeline visual_cnn − fixed_softordinal | -7.42 | -6.92 / -7.92 | FAIL |
| high_lr | A_pipeline physics_cnn − uniform_frozen | -1.88 | -1.94 / -1.82 | FAIL |
| high_lr | A_pipeline physics_cnn − uniform_adapted | -2.04 | -2.40 / -1.67 | FAIL |
| high_lr | A_pipeline physics_cnn − single_adapted | -1.78 | -1.74 / -1.82 | FAIL |
| high_lr | A_pipeline physics_cnn − fixed_softordinal | -2.02 | -1.44 / -2.59 | FAIL |
| high_lr | A_pipeline physics_mlp − uniform_frozen | -0.19 | -0.49 / +0.11 | FAIL |
| high_lr | A_pipeline physics_mlp − uniform_adapted | -0.35 | -0.96 / +0.26 | FAIL |
| high_lr | A_pipeline physics_mlp − single_adapted | -0.09 | -0.30 / +0.12 | FAIL |
| high_lr | A_pipeline physics_mlp − fixed_softordinal | -0.33 | +0.01 / -0.66 | FAIL |
| high_lr | B physics_cnn − visual_cnn, с A | +5.40 | +5.48 / +5.32 | FAIL |
| high_lr | C physics_cnn − physics_mlp, с A | -1.69 | -1.45 / -1.93 | FAIL |
| low_lr | A_pipeline visual_cnn − uniform_frozen | -8.35 | -9.69 / -7.01 | FAIL |
| low_lr | A_pipeline visual_cnn − uniform_adapted | -8.69 | -10.38 / -6.99 | FAIL |
| low_lr | A_pipeline visual_cnn − single_adapted | -8.44 | -9.82 / -7.06 | FAIL |
| low_lr | A_pipeline visual_cnn − fixed_softordinal | -8.76 | -9.67 / -7.86 | FAIL |
| low_lr | A_pipeline physics_cnn − uniform_frozen | -0.42 | -0.58 / -0.27 | FAIL |
| low_lr | A_pipeline physics_cnn − uniform_adapted | -0.76 | -1.27 / -0.25 | FAIL |
| low_lr | A_pipeline physics_cnn − single_adapted | -0.51 | -0.70 / -0.32 | FAIL |
| low_lr | A_pipeline physics_cnn − fixed_softordinal | -0.84 | -0.55 / -1.12 | FAIL |
| low_lr | A_pipeline physics_mlp − uniform_frozen | +0.40 | +0.26 / +0.54 | FAIL |
| low_lr | A_pipeline physics_mlp − uniform_adapted | +0.06 | -0.43 / +0.55 | FAIL |
| low_lr | A_pipeline physics_mlp − single_adapted | +0.31 | +0.14 / +0.49 | FAIL |
| low_lr | A_pipeline physics_mlp − fixed_softordinal | -0.01 | +0.29 / -0.32 | FAIL |
| low_lr | B physics_cnn − visual_cnn, с A | +7.93 | +9.12 / +6.74 | FAIL |
| low_lr | C physics_cnn − physics_mlp, с A | -0.82 | -0.84 / -0.80 | FAIL |

Независимый CPU-аудит: `PASS_AUDIT`, 31 / 31. [AUDIT.json](trials/directional_screen_s5501_2b0ea20c34/AUDIT.json).

Примечание технического аудита: Log-domain parity uses an8-float32-epsilon relative budget plus an independent probability-space check. Directional marginals form each joint9 distribution before C4 pooling; the final mixture is not required to factorize.

## Пределы вывода и прежний поиск

Истинные девять классов прежние и производны от SVG-масок; они не являются экспертными типами дефектов или измеренным электрическим состоянием. Непрерывные физические признаки получены из предсказания и не назначают истинные метки. Выигрыш class-argmax может происходить при неизменной ошибочной маске; это не улучшение сегментации. B — matched-CNN сравнение физического и визуального контекстов, C меняет также параметры/пространственный вход. Технический PASS подтверждает воспроизведение чисел и неизменность маски, а критерии A/B/C отдельно проверяют качество классификации.

[Прежний flat9-выход, 96 обучений и ссылки на сохранённый C4 FAIL_PILOT](../sem_arch_classification_readout_20261102/REPORT.md). Эта серия использует только старые TRAIN/VAL. Независимого подтверждения новой классификационной головы ещё нет; прежний закрытый исход другой модели не переопределяется. Для нового выбранного тракта требуется отдельная закрытая проверка. Перенос между изделиями, новизна и готовность для Q1 не установлены.

[PROTOCOL.md](PROTOCOL.md) · [CONFIG.json](CONFIG.json) · [LEDGER.json](LEDGER.json) · [model.py](model.py) · [runner.py](runner.py) · [audit.py](audit.py).
