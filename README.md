# bottleDetect — Система детекции дефектов бутылок

Бакалаврская дипломная работа.
**Задача:** Object Detection — локализация дефектов пластиковых бутылок на конвейере bounding-box-ами.

![mAP50](https://img.shields.io/badge/mAP%400.5-0.984-brightgreen) ![FPS](https://img.shields.io/badge/CPU%20FPS-33.5-blue) ![size](https://img.shields.io/badge/model%20size-6.23%20MB-orange)

---

## Содержание

1. [Быстрый старт](#быстрый-старт)
2. [Классы дефектов](#классы-дефектов)
3. [Результаты](#результаты)
4. [Структура проекта](#структура-проекта)
5. [Использование](#использование)
6. [Дообучение на своём датасете](#дообучение-на-своём-датасете)
7. [Требования](#требования)

---

## Быстрый старт

```bash
# 1. Клонировать
git clone https://github.com/nurkal022/bottleDetect.git
cd bottleDetect

# 2. Окружение
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate.bat       # Windows

pip install -r requirements.txt

# 3. Запустить нативное приложение
python app_native.py
```

Веса модели (`runs_det/yolov8n_det/weights/best.pt`, 6.23 MB) уже в репо — **ничего тренировать не нужно**, сразу можно пользоваться.

---

## Классы дефектов

| ID | Класс | Описание | Цвет |
|---|---|---|---|
| 0 | `deformed_bottl` | Деформация корпуса | 🔴 |
| 1 | `misaligned_label` | Перекошенная этикетка | 🟠 |
| 2 | `missing_cap` | Отсутствующая крышка | 🟠 |
| 3 | `missing_label` | Отсутствующая этикетка | 🟡 |
| 4 | `neck_deformed` | Деформация горлышка | 🔴 |
| 5 | `normal` | Бутылка без дефектов | 🟢 |

---

## Результаты

### Общие метрики

| Метрика | Значение |
|---|---|
| **mAP@0.5** | **0.984** |
| mAP@0.5:0.95 | 0.622 |
| Precision | 0.961 |
| Recall | 0.953 |
| Параметры | 3.01 M |
| Размер весов | 6.23 MB |
| FPS (CPU @ imgsz=640) | 33.5 |
| Latency (CPU) | 29.83 мс |

### Per-class на val (204 изображения, 265 инстансов)

| Класс | mAP50 | Precision | Recall |
|---|---|---|---|
| deformed_bottl | 0.995 | 0.995 | 1.000 |
| neck_deformed | 0.995 | 0.984 | 1.000 |
| missing_cap | 0.994 | 0.943 | 0.972 |
| misaligned_label | 0.988 | 0.994 | 0.909 |
| normal | 0.986 | 0.943 | 1.000 |
| missing_label | 0.944 | 0.906 | 0.839 |

Полный отчёт: [results/report.md](results/report.md)
Графики и визуализации: [runs_det/yolov8n_det/](runs_det/yolov8n_det/)

---

## Структура проекта

```
bottleDetect/
├── README.md                    # этот файл
├── requirements.txt             # зависимости
├── app_native.py                # нативное приложение (основной UI)
│
├── runs_det/
│   └── yolov8n_det/             # обученная модель + артефакты
│       ├── weights/best.pt      # веса (6.23 MB) — готовы к использованию
│       ├── results.png          # кривые loss/mAP/P/R по эпохам
│       ├── confusion_matrix.png
│       ├── BoxPR_curve.png      # Precision-Recall кривые
│       ├── BoxF1_curve.png
│       ├── val_batch*_pred.jpg  # примеры предсказаний
│       └── args.yaml            # гиперпараметры обучения
│
├── results/
│   ├── report.md                # полный отчёт для ВКР
│   ├── metrics.csv              # агрегированные метрики
│   └── per_class_metrics.csv    # метрики по каждому классу
│
├── scripts/
│   ├── train.py                 # обучение на своём датасете
│   ├── test_inference.py        # прогон модели на папке test/
│   └── export_onnx.py           # экспорт весов в ONNX
│
├── test/                        # тестовые файлы (фото/видео)
│   ├── *.jpeg, *.mp4            # входные
│   ├── out_*.jpg, out_*.mp4     # результаты с bbox
│   └── predictions.csv          # сводная таблица
│
└── docs/superpowers/            # ТЗ и план реализации
```

---

## Использование

### Нативное приложение (основной способ)

```bash
python app_native.py
```

Открывается окно с тремя вкладками:

| Режим | Вход | Описание |
|---|---|---|
| 📷 Фото | JPG / PNG | Одна картинка → детекция с bbox и меткой |
| 🎬 Видео | MP4 / AVI / MOV | Покадровая обработка с ByteTrack |
| 📹 Live | Веб-камера | Детекция в реальном времени |

**Стабильность детекции обеспечивают:**
- ByteTrack — связывает объекты между кадрами
- Anti-flicker буфер — удерживает бокс на 1-3 пропущенных кадрах
- EMA-сглаживание координат (α=0.6) — плавное движение bbox
- Фильтр по размеру — отбрасывает боксы > 75% и < 0.2% от кадра
- Дедупликация по track_id — одна бутылка считается один раз

**Управление:**
- Slider «Порог уверенности» (по умолчанию 50%)
- Кнопки «▶ Старт / ■ Стоп / ↺ Сброс / 💾 Сохранить лог»
- Dropdown выбора модели (если есть несколько)

### Пакетный прогон на папке

```bash
# Положи файлы в test/ и запусти:
python scripts/test_inference.py
```

Результаты попадут в ту же папку с префиксом `out_` + `predictions.csv` со сводкой.

### Экспорт в ONNX (для деплоя)

```bash
python scripts/export_onnx.py
# создаст runs_det/yolov8n_det/weights/best.onnx
```

---

## Дообучение на своём датасете

### 1. Подготовь данные в YOLO-формате

```
data/detection/
├── train/
│   ├── images/*.jpg
│   └── labels/*.txt              # по одному txt на картинку
├── val/
│   ├── images/*.jpg
│   └── labels/*.txt
└── data.yaml
```

`data.yaml`:
```yaml
path: /abs/path/to/data/detection
train: train/images
val: val/images

nc: 6
names: ['deformed_bottl', 'misaligned_label', 'missing_cap',
        'missing_label', 'neck_deformed', 'normal']
```

Формат `.txt` — одна строка на bbox:
```
<class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
```
Все координаты — относительные [0, 1].

### 2. Запусти обучение

```bash
# На GPU (NVIDIA):
python scripts/train.py --data data/detection/data.yaml --device 0

# На Apple Silicon:
python scripts/train.py --data data/detection/data.yaml --device mps

# На CPU:
python scripts/train.py --data data/detection/data.yaml --device cpu
```

Опции:
- `--epochs 100` (по умолчанию 100)
- `--imgsz 640` (по умолчанию 640)
- `--batch 32`
- `--name my_experiment` (папка внутри `runs_det/`)
- `--weights yolov8s.pt` (другая starting-модель)

### 3. Используй новую модель

После обучения веса лежат в `runs_det/<name>/weights/best.pt`.
Откроется в приложении через dropdown автоматически.

---

## Требования

| Компонент | Минимум | Рекомендуется |
|---|---|---|
| Python | 3.10+ | 3.11 |
| RAM | 4 GB | 8 GB |
| CPU | любой 64-bit | Apple Silicon / Intel 10+ gen |
| GPU для обучения | — | NVIDIA с ≥ 6 GB VRAM |
| Диск | 500 MB | — |
| OS | macOS 12+ / Windows 10+ / Linux | — |

**Уже предустановлено в репо:**
- Обученные веса (`best.pt`, 6.23 MB)
- Все графики обучения
- Отчёт для ВКР

**Для обучения с нуля:**
Если нет NVIDIA GPU, можно использовать Google Colab / Kaggle Notebooks (T4 бесплатно).

---

## Технологии

| | |
|---|---|
| Python | 3.11 |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | 8.4+ |
| PyTorch | 2.9 + CUDA 13 (на сервере) |
| OpenCV | 4.12 |
| PySide6 (Qt) | 6.7+ |
| ByteTrack | встроен в Ultralytics |

---

## Лицензия и цитирование

Код — MIT.
Датасет: Roboflow (приватный), workspace `nurlastdey/nurlastdey`.
Модель обучена из pretrained `yolov8n.pt` (AGPL-3.0 от Ultralytics).

```bibtex
@misc{bottledetect_2026,
  author={Nurlykhan},
  title={Bottle Defect Detection — Bachelor Thesis},
  year={2026},
  url={https://github.com/nurkal022/bottleDetect}
}
```
