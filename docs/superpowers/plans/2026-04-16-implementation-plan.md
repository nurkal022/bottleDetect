# План реализации

## Этап 1 — Scaffolding (первое, что уже сделано ночью)
- [x] Создать структуру директорий
- [x] `requirements.txt`, `.gitignore`, `.env.example`
- [x] Конфиги: `configs/dataset.yaml`, `configs/models.yaml`, `configs/class_mapping.yaml`
- [x] README с пошаговой инструкцией

## Этап 2 — Данные
1. Получить бесплатный ключ Roboflow: https://app.roboflow.com/settings/api
2. `export ROBOFLOW_API_KEY=...` (или положить в `.env`)
3. `python scripts/download_data.py` — качает spark-intelligence + thesis-work в `data/raw/`
4. `python scripts/prepare_data.py` — ремап классов и split в `data/processed/`
5. Проверить `data.yaml` (создастся автоматически)

## Этап 3 — Обучение
1. `python scripts/train_all.py --epochs 100 --imgsz 640`
2. Все 4 модели обучаются последовательно в `runs/train/<model_name>/`
3. Лучшие веса → `runs/train/<model_name>/weights/best.pt`

## Этап 4 — Оценка и сравнение
1. `python scripts/evaluate.py` — считает mAP/P/R для каждой модели на test split
2. `python scripts/benchmark_fps.py` — FPS на CPU и GPU
3. `python scripts/compare_models.py` — собирает `results/comparison.csv` + графики

## Этап 5 — OOD-тест
1. `python scripts/evaluate.py --model runs/train/<best>/weights/best.pt --data data/ood/data.yaml`
2. Результат в `results/ood.md`

## Этап 6 — Real-time демо
1. `python src/realtime_demo.py --source 0 --model runs/train/<best>/weights/best.pt`
   (source: `0` = webcam, путь к видео, путь к папке)
2. Экспорт ONNX: `python scripts/export_onnx.py --model <path>`

## Этап 7 — Thesis material
- Скопировать таблицы из `results/comparison.md` в текст ВКР.
- Скопировать графики из `results/plots/` в ВКР.
- Confusion matrix для лучшей модели — `runs/train/<best>/confusion_matrix.png`.
