# Отчёты по проекту bottleDetect

Полный пайплайн работы над дипломной работой: от исследования классификации до финальной detection-системы.

## Навигация

| Документ | Содержание |
|---|---|
| [01_pipeline.md](01_pipeline.md) | Хронология проекта: эволюция задачи и архитектурных решений |
| [02_classification_comparison.md](02_classification_comparison.md) | Сравнительное исследование 7 YOLO-классификаторов |
| [03_detection_final.md](03_detection_final.md) | Обучение финальной YOLOv8n detection модели |
| [04_inference_system.md](04_inference_system.md) | Нативное приложение, стабилизация детекции |

## Структура

```
reports/
├── README.md                          ← вы здесь
├── 01_pipeline.md                     ← эволюция проекта
├── 02_classification_comparison.md    ← сравнение 7 cls-моделей
├── 03_detection_final.md              ← финальная detection-модель
├── 04_inference_system.md             ← приложение + стабилизация
├── classification_benchmarks.csv      ← сводная таблица метрик cls
├── figures/
│   ├── classification/                ← графики 7 cls-моделей (args, results, confusion)
│   ├── detection/                     ← графики финальной detection-модели
│   └── comparison/                    ← сравнительные графики
└── logs/
    ├── train_all.log                  ← обучение 4 cls-моделей (n, s)
    ├── train_extra.log                ← обучение 3 cls-моделей (m, l)
    ├── train_det.log                  ← первая попытка detection (nvrtc crash)
    └── train_det2.log                 ← успешное detection-обучение
```

## Итоговые числа

### Classification (промежуточное исследование, 7 моделей)

| Показатель | Значение |
|---|---|
| Модели | yolov8n/s/m/l-cls, yolo11n/s/m-cls |
| Датасет | 702 train / 100 val / 50 test, 4 класса |
| Лучшая | **yolov8n-cls** — 100% Top-1, 320 FPS CPU, 2.97 MB |
| Время обучения всех 7 | ~10 минут (RTX 5080) |

### Detection (финальная система)

| Показатель | Значение |
|---|---|
| Модель | **yolov8n_det** |
| Датасет | 1161 train / 204 val, 6 классов с bbox |
| mAP@0.5 | **0.984** |
| mAP@0.5:0.95 | 0.622 |
| Precision / Recall | 0.961 / 0.953 |
| CPU FPS (@ 640) | 33.5 |
| Размер весов | 6.23 MB |
| Время обучения (100 эпох) | ~9 минут (RTX 5080) |

### Оборудование

- **Обучение:** сервер с NVIDIA RTX 5080 Laptop GPU (16 GB VRAM), CUDA 13.0
- **Инференс (замеры):** Apple Silicon CPU (M-series), macOS
- **Развёртывание:** нативное приложение PySide6

## Read me first

Начни с [01_pipeline.md](01_pipeline.md) — там общая картина и обоснование каждого решения.
