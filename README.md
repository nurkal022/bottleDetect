# Bottle Defect Classification — Дипломная работа

Сравнительное исследование лёгких YOLO-моделей для классификации дефектов бутылок на конвейере.

## Задача

Классификация полного кадра (одна бутылка) по 4 классам:

| Класс | Описание |
|---|---|
| `good_bottle` | Без дефектов |
| `cork_missmatch` | Неправильная крышка / крышка отсутствует |
| `label_missing` | Этикетка отсутствует |
| `shape_mismatch` | Деформация корпуса |

## Датасет

- **Источник:** [bottle-defect-jqd6l](https://universe.roboflow.com/thesis-work-l9lqp/bottle-defect-jqd6l) — Roboflow Universe
- **Формат:** folder (image classification)
- **Разбиение:** train 702 / valid 100 / test 50

## Модели

| Модель | Параметры | Ожидаемый FPS (CPU) |
|---|---|---|
| YOLOv8n-cls | ~2.7M | высокий |
| YOLOv8s-cls | ~6.4M | средний |
| YOLOv11n-cls | ~1.5M | очень высокий |
| YOLOv11s-cls | ~5.9M | средний |

## Быстрый старт

```bash
# 1. Установить зависимости
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Обучить все 4 модели
python scripts/train_all.py
# На Mac Apple Silicon:
python scripts/train_all.py --device mps
# На NVIDIA GPU:
python scripts/train_all.py --device 0

# 3. Оценить на test split
python scripts/evaluate.py

# 4. Замерить скорость
python scripts/benchmark_fps.py
# С GPU:
python scripts/benchmark_fps.py --devices cpu mps

# 5. Построить сводные таблицы и графики
python scripts/compare_models.py

# 6. Запустить real-time демо (веб-камера)
python src/realtime_demo.py \
  --source 0 \
  --model runs/classify/yolo11n-cls/weights/best.pt

# Real-time демо из папки тестовых картинок:
python src/realtime_demo.py \
  --source data/raw/test \
  --model runs/classify/yolo11n-cls/weights/best.pt \
  --log results/demo_log.csv
```

## Структура проекта

```
bottleDetect/
├── configs/
│   └── models.yaml          # Список моделей + гиперпараметры
├── data/
│   └── raw/                 # Датасет (train/valid/test/<class>/*.jpg)
├── scripts/
│   ├── train.py             # Обучение одной модели
│   ├── train_all.py         # Обучение всех 4 моделей подряд
│   ├── evaluate.py          # Top-1/Top-5 accuracy на test
│   ├── benchmark_fps.py     # FPS / latency на CPU и GPU
│   ├── compare_models.py    # Сводная таблица + 3 графика
│   └── export_onnx.py       # Экспорт лучшей модели в ONNX
├── src/
│   ├── realtime_demo.py     # OpenCV live-демо
│   └── counter.py           # Счётчик дефектов
├── runs/classify/           # Ultralytics training outputs (gitignore)
└── results/                 # Таблицы + графики (в git)
    ├── metrics.csv
    ├── speed.csv
    ├── comparison.csv
    ├── comparison.md
    └── plots/
        ├── acc_vs_fps.png
        ├── acc_vs_params.png
        └── acc_bar.png
```

## Горячие клавиши демо

| Клавиша | Действие |
|---|---|
| `q` | Выход |
| `r` | Сброс счётчиков |
| `p` | Пауза / продолжить |
| `s` | Скриншот |

## Цитирование датасета

```
@dataset{bottle-defect-jqd6l,
  title={bottle defect Dataset},
  type={Open Source Dataset},
  author={thesis-work},
  howpublished={\url{https://universe.roboflow.com/thesis-work-l9lqp/bottle-defect-jqd6l}},
  year={2024},
  note={visited on 2026-04-16},
}
```
