# Исследование сегментации и классификации для микроэлектроники

Текущий [список из 11 гипотез для ВКР](gipoteza/11_gipotez_dlya_vkr.md) включает H7.1 и ранжирован по пригодности, экспериментальным результатам и ограниченному учёту новизны.

Основные материалы находятся в [gipoteza/](gipoteza/README.md):

- H7.1: [постановка](gipoteza/H7_1.md), [код, данные и веса](gipoteza/h7_readout_20261025/README.md).
- Wafer maps: [протокол](gipoteza/diploma_selection_20261027/NEW_MINI_PROTOCOL.md), [результаты](gipoteza/diploma_selection_20261027/wafer_analysis/RESULT.json), [аудит](gipoteza/diploma_selection_20261027/wafer_analysis/INDEPENDENT_AUDIT.json).
- SEM и EP2-Core: [исторические отчёты, протоколы и код](gipoteza/retained_evidence_20261027/README.md). Данные и веса ранних опытов были удалены до нынешнего отбора; полный повтор требует их восстановления.
- [Исследовательский контекст](full_wafer_research_context_updated.md), [литературный разбор](gipoteza/literatura_20261019.md), исходная литературная таблица `litobzor_corrected.xlsx`.
- [Журнал очистки и сохранённые зависимости](gipoteza/cleanup_11_20261027/README.md).

## Базовые методы сравнения

Сохранены [RadonCNN](papers/radon_cnn/README.md) и [PolarCNN](papers/polar_cnn/README.md), общая библиотека `common/` и их принятые веса. Они нужны как готовые внешние сравнения для wafer-гипотез, а не дополнительные гипотезы списка.

- [Принятый RadonCNN](papers/radon_cnn/results/accepted_mps_20261008.json): веса и протоколы в `checkpoints/radon_cnn_mps_full_20261008/`.
- [Принятый PolarCNN](papers/polar_cnn/results/accepted_mps_20261011.json): пакет в `checkpoints/polar_cnn_accepted_mps_20261011/`.
- [Историческая таблица воспроизведения](docs/reproduction_results.md) сохранена. Числа прежних разбиений и разных наборов классов нельзя непосредственно сравнивать с новым групповым тестом.

Для доступных базовых методов используются `train.py` и `evaluate.py`:

```bash
.venv/bin/python train.py --help
.venv/bin/python evaluate.py --help
```

Исходный WM-811K сохранён целиком в `datasets/`. Рабочее окружение `.venv` и Git-история сохранены. Новые обучения при ранжировании и очистке не выполнялись.
