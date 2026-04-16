#!/bin/bash
# Полный пайплайн: обучение → оценка → графики
# Запуск на GPU-машине: bash run_all.sh
# С конкретным GPU: DEVICE=0 bash run_all.sh

set -e

DEVICE=${DEVICE:-0}       # по умолчанию первый GPU
EPOCHS=${EPOCHS:-50}

echo "============================================="
echo " Bottle Defect Classification — Full Pipeline"
echo " Device: $DEVICE   Epochs: $EPOCHS"
echo "============================================="

# 1. Обучение
echo -e "\n[1/4] Обучение всех моделей..."
python scripts/train_all.py --device "$DEVICE" --epochs "$EPOCHS"

# 2. Оценка на test split
echo -e "\n[2/4] Оценка точности на test split..."
python scripts/evaluate.py --device "$DEVICE"

# 3. Замер скорости (CPU + GPU)
echo -e "\n[3/4] Бенчмарк скорости..."
python scripts/benchmark_fps.py --devices cpu "$DEVICE"

# 4. Сравнительные таблицы и графики
echo -e "\n[4/4] Построение графиков..."
python scripts/compare_models.py

echo -e "\n============================================="
echo " Готово! Результаты: results/"
echo " Графики: results/plots/*.png"
echo " Таблица: results/comparison.md"
echo "============================================="
