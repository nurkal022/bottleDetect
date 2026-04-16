"""
Экспорт лучшей модели в ONNX.
python scripts/export_onnx.py --model runs/train/yolo11n/weights/best.pt
"""
from __future__ import annotations

import argparse
import sys

from ultralytics import YOLO


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true")
    args = p.parse_args()

    model = YOLO(args.model)
    path = model.export(format="onnx", imgsz=args.imgsz, half=args.half, dynamic=False, simplify=True)
    print(f"Exported: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
