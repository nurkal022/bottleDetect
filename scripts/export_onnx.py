"""
Экспорт обученной модели в ONNX (для деплоя на edge-устройства).

По умолчанию экспортирует нашу detection-модель:
    python scripts/export_onnx.py
Либо другую:
    python scripts/export_onnx.py --model path/to/best.pt
"""
from __future__ import annotations

import argparse
import sys

from ultralytics import YOLO


def main() -> int:
    from pathlib import Path
    default_model = Path(__file__).resolve().parent.parent / \
                    "runs_det/yolov8n_det/weights/best.pt"

    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(default_model))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true")
    args = p.parse_args()

    model = YOLO(args.model)
    path = model.export(format="onnx", imgsz=args.imgsz, half=args.half, dynamic=False, simplify=True)
    print(f"Exported: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
