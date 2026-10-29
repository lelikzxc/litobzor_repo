# Проверка дополнительного классового надзора на SEM, 28 октября

Главный вопрос — помогает ли резисторная регуляризация относительно обучаемой
головы, получающей те же дополнительные изображения и классовые метки.

Текущий ход выполнения:

```bash
cat gipoteza/sem_confirmation_20261028/PROGRESS.json
tail -f gipoteza/sem_confirmation_20261028/progress.jsonl
```

После завершения: [полный отчёт](REPORT.md), [числа](RESULT.json),
[независимый аудит](AUDIT.json), [протокол](PROTOCOL.md).

Сохранены 16 исходных SEM-кадров и SVG, подготовленные DATA.npz,
DATA_MANIFEST.json с источниками/SHA/разбиением, веса models/ и все прогнозы.
Тестовые кадры выбраны до обучения; параметры и тип основного контроля —
по VAL. Первоначальный допуск VAL и техническая ошибка переноса MPS/CPU
сохранены отдельно; [поправка](AMENDMENT_01.md) опубликована до обучения.

Команды, использованные для последовательного запуска:

```bash
.venv/bin/python -B gipoteza/sem_confirmation_20261028/prepare.py download
.venv/bin/python -B gipoteza/sem_confirmation_20261028/prepare.py prepare
.venv/bin/python -B gipoteza/sem_confirmation_20261028/train.py
.venv/bin/python -B gipoteza/sem_confirmation_20261028/evaluate.py
.venv/bin/python -B gipoteza/sem_confirmation_20261028/audit.py
.venv/bin/python -B gipoteza/sem_confirmation_20261028/report.py
```

Загрузка требует сети, обучение/оценка — MPS; в песочнице эти возможности
могут быть недоступны. На хосте Mac они используются непосредственно.
Ограничения потоков CPU задавались OPENBLAS_NUM_THREADS=2,
OMP_NUM_THREADS=2,VECLIB_MAXIMUM_THREADS=2; PYTHONDONTWRITEBYTECODE=1.

Скрипты не перезаписывают завершённые данные, обучение или единственную
итоговую оценку. Для нового запуска нужен отдельный рабочий каталог с
копией кода/протокола и соответствующими зависимостями, а не удаление
сохранённых frozen manifests. Старый документ gipoteza.md не менялся.
