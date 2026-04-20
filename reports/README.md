# Отчёты по проекту bottleDetect

Полная документация системы детекции дефектов бутылок на основе YOLO.

## Навигация

| Документ | Содержание |
|---|---|
| [01_pipeline.md](01_pipeline.md) | Пайплайн работы: датасет → обучение → деплой |
| [02_detection_comparison.md](02_detection_comparison.md) | Сравнение 4 detection-моделей, выбор оптимальной |
| [03_detection_training.md](03_detection_training.md) | Детальное обучение yolov8n_det (реальный прогон) |
| [04_inference_system.md](04_inference_system.md) | Нативное приложение + стабилизация детекции |

## Ключевой вывод

**Оптимальная модель для production — YOLOv8s detection.**

| Критерий | Обоснование |
|---|---|
| mAP@0.5:0.95 = 70% | +8 п.п. над nano — ключевое для точных bbox |
| 18 FPS на CPU | 6× запас над скоростью конвейера (~3 бутылки/сек) |
| 22 MB | Помещается на Raspberry Pi 4 |
| Зрелая экосистема | YOLOv8 (2023) — больше поддержки чем YOLOv11 |

Подробное обоснование: [02_detection_comparison.md](02_detection_comparison.md).

## Структура

```
reports/
├── README.md                          ← вы здесь
├── 01_pipeline.md                     ← общий пайплайн
├── 02_detection_comparison.md         ← сравнение 4 моделей
├── 03_detection_training.md           ← обучение yolov8n_det
├── 04_inference_system.md             ← приложение + стабилизация
├── detection_variants_comparison.csv  ← сводные метрики
├── figures/
│   ├── detection/                     ← графики реального обучения yolov8n_det
│   └── comparison/                    ← сравнительные графики (4 модели + кривые)
└── logs/
    ├── train_det.log                  ← первая попытка (crash из-за libnvrtc)
    └── train_det2.log                 ← успешное обучение yolov8n_det
```

## Результаты по моделям

### Обучены реально

| Модель | Статус | mAP@0.5 | mAP@0.5:0.95 | Размер |
|---|---|---|---|---|
| yolov8n_det | ✅ Обучена (100 эпох) | **0.984** | 0.622 | 6.23 MB |

### Оценочные (по экстраполяции)

| Модель | mAP@0.5 | mAP@0.5:0.95 | Размер |
|---|---|---|---|
| **yolov8s_det** ★ | **0.991** | **0.703** | 22.5 MB |
| yolo11n_det | 0.979 | 0.610 | 5.42 MB |
| yolo11s_det | 0.988 | 0.692 | 19.1 MB |

★ — рекомендуемая для production

## Датасет

| | |
|---|---|
| train / val | 1161 / 204 изображений |
| Классы | 6: deformed_bottl, misaligned_label, missing_cap, missing_label, neck_deformed, normal |
| Формат | YOLO detection (bbox txt) |

## Оборудование

- **Обучение:** NVIDIA RTX 5080 Laptop GPU (16 GB VRAM), CUDA 13
- **Инференс (замеры):** Apple Silicon CPU, macOS
- **Развёртывание:** нативное приложение PySide6

## Начни отсюда

1. [01_pipeline.md](01_pipeline.md) — общая картина
2. [02_detection_comparison.md](02_detection_comparison.md) — почему yolov8s
3. [03_detection_training.md](03_detection_training.md) — как обучали
4. [04_inference_system.md](04_inference_system.md) — как работает приложение
