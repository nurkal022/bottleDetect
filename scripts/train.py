"""
Обучение YOLOv8n detection на датасете bottle-defect.

Использование:
    python scripts/train.py --data data/detection/data.yaml
    python scripts/train.py --data <path> --epochs 100 --device 0

Датасет должен быть в YOLO-формате:
    data/detection/
      train/images/*.jpg
      train/labels/*.txt
      val/images/*.jpg
      val/labels/*.txt
      data.yaml   ← с ключами path, train, val, nc, names
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to data.yaml")
    p.add_argument("--weights", default="yolov8n.pt",
                   help="начальные веса (скачаются автоматически)")
    p.add_argument("--name", default="yolov8n_det")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--device", default=None,
                   help="cpu / 0 (GPU) / mps (Apple Silicon)")
    args = p.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        optimizer="AdamW",
        lr0=args.lr0,
        seed=42,
        device=args.device,
        project=str(ROOT / "runs_det"),
        name=args.name,
        exist_ok=True,
    )
    print(f"\nDone. Weights: runs_det/{args.name}/weights/best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
