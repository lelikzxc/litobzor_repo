# Прямая классификационная CNN-ветвь при замороженной сегментации

**Изменился смысл классового выхода:** новая голова предсказывает девять вероятностей, итог — `argmax`, а не класс, вычисленный только из пороговой маски. Сегментационная C4-карта исходной CNN заморожена и проверяется побайтно; рост классификационного F1 не является ростом IoU или улучшением маски. Класс головы может не совпадать с геометрическим классом её маски; это измеряется отдельно.

Visual_cnn и physics_cnn имеют одинаковые **5 737** обучаемых параметров: Conv3×3(26→16), GELU; FiLM(4→32); Conv3×3(16→8), GELU; AvgPool3×3; Linear(72→9). С замороженной CNN 10 681 это 16 418 параметров. Вход26 — признаки24, C4-вероятность и SEM. Общие веса головы обрабатывают четыре вида. При поворотах 90/270 градусов классовые вероятности переставляются `[0,3,6,1,4,7,2,5,8]`, меняя LR/TB; затем усредняются вероятности, не номера классов. Проверяется C4-ковариантность с перестановкой меток.

Physics_cnn получает четыре tanh(ordinal_logits/4) от предсказанных непрерывных gLR/gTB по заострённой C4-карте μ⁴/(μ⁴+(1−μ)⁴). Visual_cnn получает четыре визуальные статистики в сопоставимом диапазоне: LR/TB-границы, среднюю вероятность и неопределённость. Physics_mlp — Linear(4→32), GELU, Linear(32→9), **457** параметров, всего 11 138; он получает тот же физический контекст. C сравнивает CNN со значительно меньшим MLP и изменяет также доступ к пространственным признакам.

## Надзор, лосс и границы сравнения

Для всех трёх голов используется одна **стандартная CE** канонического девятиклассового распределения, 1 000 шагов AdamW, lr=0,001, weight_decay=0,0001, batch32. Совокупный TRAIN — dense18/27 и 1152 weak, 1170/1179 примеров; одинаковые классы и батчи доступны всем головам. Пиксельные маски ни dense, ни weak не передаются оптимизатору. Encoder заморожен; его прежние веса обучены с BCE+geometry_loss. Это исследование классификационной архитектуры под общей CE, без новой предлагаемой функции потерь.

A_pipeline против фиксированного readout включает также дополнительное обучение головы по классам. Он не изолирует чистую замену одного слоя. B сравнивает две одинаковые CNN под одинаковой CE и проверяет физический контекст. C сравнивает physics_cnn с physics_mlp при той же CE; ёмкость и пространственный вход этих моделей различаются.

Начато попыток 1, завершено 1, технических ошибок 0, незавершённых 0. Новых обучений голов **12**, обновлений 12,000. С прежними 84 и дополнительными внешними контролями 0 уникальных обучений **96**. Сводные результаты и сохранённые контроли повторно не считаются.

## Критерии классификационного тракта

A_pipeline требует ΔMacro-F1 ≥0,5 п.п. в среднем и ΔF1 каждого бюджета ≥−0,25 п.п. против каждого из четырёх readout: uniform_frozen, uniform_adapted, single_adapted и fixed_softordinal(g0,0,5), отдельно в high_lr/low_lr. Старые масочные readout сохраняют свои выбранные по 81-точечной VAL-сетке пороги. У новых классовых голов порога нет: argmax9. IoU не включён как мнимый новый выигрыш; маска обязана остаться неизменной. B: physics_cnn против visual_cnn и A physics_cnn. C: physics_cnn против physics_mlp и A physics_cnn. A для MLP не устанавливает пользу пространственной CNN. При расширении отдельно нужны условия на всех трёх и новых двух seed.

## screen: seed [5501]

| Семья | Visual CNN F1, % | Physics CNN F1, % | Physics MLP F1, % | A visual / physics / MLP | B | C |
|---|---:|---:|---:|---|---|---|
| high_lr | 75.47 | 84.85 | 85.32 | FAIL / FAIL / FAIL | FAIL | FAIL |
| low_lr | 75.79 | 86.11 | 85.96 | FAIL / FAIL / FAIL | FAIL | FAIL |

Обе семьи: A visual_cnn **FAIL**; A physics_cnn **FAIL**; A physics_mlp **FAIL**; B **FAIL**; C **FAIL**.

| Семья | Frozen mask-readout F1, % | C4-adapted mask-readout F1, % | Single-adapted mask-readout F1, % | Fixed softordinal F1, % |
|---|---:|---:|---:|---:|
| high_lr | 85.82 | 85.97 | 85.71 | 85.95 |
| low_lr | 85.72 | 86.05 | 85.80 | 86.13 |

| Семья | Голова | Неизменный IoU маски, % | Несогласие class-argmax / class-of-mask, % |
|---|---|---:|---:|
| high_lr | visual_cnn | 90.26 | 20.22 |
| high_lr | physics_cnn | 90.26 | 12.01 |
| high_lr | physics_mlp | 90.26 | 7.70 |
| low_lr | visual_cnn | 90.13 | 17.84 |
| low_lr | physics_cnn | 90.13 | 9.93 |
| low_lr | physics_mlp | 90.13 | 5.18 |

high_lr: численные условия сравнения physics_cnn с visual_cnn пройдены (ΔF1 +9.37 п.п.), но полный B остаётся FAIL, поскольку обязательный A_pipeline physics_cnn не пройден. Разница внутри пары CNN и выигрыш над сильными readout — разные сравнения.

low_lr: численные условия сравнения physics_cnn с visual_cnn пройдены (ΔF1 +10.31 п.п.), но полный B остаётся FAIL, поскольку обязательный A_pipeline physics_cnn не пройден. Разница внутри пары CNN и выигрыш над сильными readout — разные сравнения.

![Классификационный результат и отдельно неизменная сегментация](figures/classification_and_frozen_mask.png)

Слева/сверху — классификационные метрики; IoU показан отдельно. Минимальный прирост A берётся по четырём контролям. Пунктир +0,5 п.п. относится к среднему F1, бюджетные ограничения приведены ниже. Это повторная VAL без независимых интервалов подтверждения.

## Все сохранённые попытки

### classhead_screen_s5501_b456aa22ce

[FROZEN.json](trials/classhead_screen_s5501_b456aa22ce/FROZEN.json).

`COMPLETE_CLASSIFICATION_DEVELOPMENT`; screen; seed [5501]; обучений 12, обновлений 12,000; новых оценок 12; старых масочных контролей 12, фиксированных softordinal 4; 113.16 с. [RESULT.json](trials/classhead_screen_s5501_b456aa22ce/RESULT.json), SHA-256 `e18087349f392aa77dd5c09c5831726d15df4024032e2f9af71caff81035f631`.

| Семья | Бюджет | Seed | Голова | Параметров головы | Class F1, % | Accuracy, % | Неизменный mask IoU, % | Class/mask несогласие, % |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| high_lr | 18 | 5501 | visual_cnn | 5737 | 74.86 | 75.25 | 90.22 | 20.43 |
| high_lr | 18 | 5501 | physics_cnn | 5737 | 86.09 | 86.47 | 90.22 | 9.35 |
| high_lr | 18 | 5501 | physics_mlp | 457 | 85.86 | 86.47 | 90.22 | 7.91 |
| high_lr | 27 | 5501 | visual_cnn | 5737 | 76.09 | 76.55 | 90.31 | 20.00 |
| high_lr | 27 | 5501 | physics_cnn | 5737 | 83.60 | 84.03 | 90.31 | 14.68 |
| high_lr | 27 | 5501 | physics_mlp | 457 | 84.77 | 85.47 | 90.31 | 7.48 |
| low_lr | 18 | 5501 | visual_cnn | 5737 | 75.16 | 75.68 | 90.03 | 17.99 |
| low_lr | 18 | 5501 | physics_cnn | 5737 | 86.85 | 87.34 | 90.03 | 7.34 |
| low_lr | 18 | 5501 | physics_mlp | 457 | 85.96 | 86.62 | 90.03 | 4.89 |
| low_lr | 27 | 5501 | visual_cnn | 5737 | 76.42 | 76.69 | 90.24 | 17.70 |
| low_lr | 27 | 5501 | physics_cnn | 5737 | 85.36 | 85.76 | 90.24 | 12.52 |
| low_lr | 27 | 5501 | physics_mlp | 457 | 85.96 | 86.62 | 90.24 | 5.47 |

**только эта попытка: [5501]**

| Семья | Сравнение классификации | ΔF1, п.п. | ΔF1 бюджеты 18 / 27, п.п. | Критерий |
|---|---|---:|---|---|
| high_lr | A_pipeline visual_cnn − uniform_frozen | -10.34 | -11.76 / -8.93 | FAIL |
| high_lr | A_pipeline visual_cnn − uniform_adapted | -10.50 | -12.22 / -8.78 | FAIL |
| high_lr | A_pipeline visual_cnn − single_adapted | -10.24 | -11.56 / -8.92 | FAIL |
| high_lr | A_pipeline visual_cnn − fixed_softordinal | -10.48 | -11.26 / -9.70 | FAIL |
| high_lr | A_pipeline physics_cnn − uniform_frozen | -0.97 | -0.53 / -1.41 | FAIL |
| high_lr | A_pipeline physics_cnn − uniform_adapted | -1.13 | -0.99 / -1.27 | FAIL |
| high_lr | A_pipeline physics_cnn − single_adapted | -0.87 | -0.33 / -1.41 | FAIL |
| high_lr | A_pipeline physics_cnn − fixed_softordinal | -1.11 | -0.03 / -2.19 | FAIL |
| high_lr | A_pipeline physics_mlp − uniform_frozen | -0.50 | -0.75 / -0.24 | FAIL |
| high_lr | A_pipeline physics_mlp − uniform_adapted | -0.66 | -1.21 / -0.10 | FAIL |
| high_lr | A_pipeline physics_mlp − single_adapted | -0.40 | -0.55 / -0.24 | FAIL |
| high_lr | A_pipeline physics_mlp − fixed_softordinal | -0.63 | -0.25 / -1.02 | FAIL |
| high_lr | B physics_cnn − visual_cnn, с A | +9.37 | +11.23 / +7.51 | FAIL |
| high_lr | C physics_cnn − physics_mlp, с A | -0.47 | +0.22 / -1.17 | FAIL |
| low_lr | A_pipeline visual_cnn − uniform_frozen | -9.93 | -10.96 / -8.89 | FAIL |
| low_lr | A_pipeline visual_cnn − uniform_adapted | -10.26 | -11.66 / -8.87 | FAIL |
| low_lr | A_pipeline visual_cnn − single_adapted | -10.01 | -11.09 / -8.94 | FAIL |
| low_lr | A_pipeline visual_cnn − fixed_softordinal | -10.34 | -10.94 / -9.74 | FAIL |
| low_lr | A_pipeline physics_cnn − uniform_frozen | +0.39 | +0.73 / +0.05 | FAIL |
| low_lr | A_pipeline physics_cnn − uniform_adapted | +0.05 | +0.04 / +0.06 | FAIL |
| low_lr | A_pipeline physics_cnn − single_adapted | +0.30 | +0.60 / -0.00 | FAIL |
| low_lr | A_pipeline physics_cnn − fixed_softordinal | -0.03 | +0.75 / -0.81 | FAIL |
| low_lr | A_pipeline physics_mlp − uniform_frozen | +0.24 | -0.16 / +0.65 | FAIL |
| low_lr | A_pipeline physics_mlp − uniform_adapted | -0.09 | -0.85 / +0.67 | FAIL |
| low_lr | A_pipeline physics_mlp − single_adapted | +0.16 | -0.29 / +0.60 | FAIL |
| low_lr | A_pipeline physics_mlp − fixed_softordinal | -0.17 | -0.14 / -0.20 | FAIL |
| low_lr | B physics_cnn − visual_cnn, с A | +10.31 | +11.69 / +8.94 | FAIL |
| low_lr | C physics_cnn − physics_mlp, с A | +0.14 | +0.89 / -0.60 | FAIL |

Независимый CPU-аудит: `PASS_AUDIT`, 28 / 28. [AUDIT.json](trials/classhead_screen_s5501_b456aa22ce/AUDIT.json).

Примечание технического аудита: The initial audit used an absolute3e-6 bound on log probabilities; large-magnitude negative logs had one-ULP CPU/MPS differences up to7.63e-6. The preserved initial audit is replaced by an8-float32-epsilon relative log budget and independent probability-space comparison. No source, trained model, data, metric, covariance tolerance, protocol or gate changed.

Предыдущая техническая проверка [AUDIT_PREVIOUS_c2de9d945d1f.json](trials/classhead_screen_s5501_b456aa22ce/AUDIT_PREVIOUS_c2de9d945d1f.json): `FAIL_AUDIT`. Исходные обучения сохранены.

## Пределы вывода и прежний поиск

Истинные девять классов прежние и производны от SVG-масок; они не являются экспертными типами дефектов или измеренным электрическим состоянием. Непрерывные физические признаки получены из предсказания и не назначают истинные метки. Выигрыш class-argmax может происходить при неизменной ошибочной маске; это не улучшение сегментации. B — matched-CNN сравнение физического и визуального контекстов, C меняет также параметры/пространственный вход. Технический PASS подтверждает воспроизведение чисел и неизменность маски, а критерии A/B/C отдельно проверяют качество классификации.

[Предыдущие попытки, 84 обучения и ссылки на сохранённый C4 FAIL_PILOT](../sem_arch_orbit_residual_20261102/REPORT.md). Эта серия использует только старые TRAIN/VAL. Независимого подтверждения новой классификационной головы ещё нет; прежний закрытый исход другой модели не переопределяется. Для нового выбранного тракта требуется отдельная закрытая проверка. Перенос между изделиями, новизна и готовность для Q1 не установлены.

[PROTOCOL.md](PROTOCOL.md) · [CONFIG.json](CONFIG.json) · [LEDGER.json](LEDGER.json) · [model.py](model.py) · [runner.py](runner.py) · [audit.py](audit.py).
