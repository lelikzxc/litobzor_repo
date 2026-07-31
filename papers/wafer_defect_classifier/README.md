# Wafer Defect Classifier

Классификация дефектов пластин (wafer maps) на основе пайплайна из двух этапов:
1. **Сегментация дефектов** — выделение дефектных областей на карте пластины
2. **Классификация типов дефектов** — определение типа дефекта (9 классов)

## Архитектура

### SegmentationModel
- **Encoder**: 3× Conv2d (1→32→64→128) + ReLU + MaxPool
- **Decoder**: 3× ConvTranspose2d (128→64→32→1) + ReLU + Sigmoid
- **Loss**: BCELoss

### ClassificationModel
- **Encoder**: 3× Conv2d (1→32→64→128) + ReLU + MaxPool
- **Classifier**: Flatten → Linear(128×16×16 → 256) → ReLU → Dropout(0.5) → Linear(256 → 9)
- **Loss**: CrossEntropyLoss + class weights

## Предобработка

1. Downsampling класса 'none' до 6000 образцов
2. Окно 3×3 с conditional filling (удаление изолированных дефектных пикселей)
3. Маскирование: выделение дефектных пикселей в бинарную маску

## Использование

```bash
# Полный пайплайн
python papers/wafer_defect_classifier/train.py --full

# Только предобработка
python papers/wafer_defect_classifier/train.py --preprocessing

# Только сегментация
python papers/wafer_defect_classifier/train.py --segmentation

# Только классификация
python papers/wafer_defect_classifier/train.py --classification

# С указанием количества эпох
python papers/wafer_defect_classifier/train.py --segmentation --epochs 50

# Предобработка отдельно
python papers/wafer_defect_classifier/preprocessing.py \
    --labels data/labels.csv \
    --images data/images_original \
    --output data
```

## Структура

```
papers/wafer_defect_classifier/
├── __init__.py
├── config.yaml              # Корневой конфиг
├── train.py                 # Точка входа для обучения
├── evaluate.py              # Функции оценки и метрики
├── preprocessing.py         # Предобработка данных
├── demo.py                  # Демонстрация
├── README.md
├── configs/
│   └── config.yaml          # Полная конфигурация
├── data_utils/
│   ├── __init__.py
│   └── dataset.py           # Датасеты
├── models/
│   ├── __init__.py          # SegmentationModel, ClassificationModel
├── utils/
│   └── __init__.py
└── tests/
    └── __init__.py
```

## Результаты

Метрики классификации (на тестовом наборе, stratified split):
- Accuracy: ~0.87
- Weighted F1: ~0.87

(см. pipeline (3).ipynb для детального анализа)