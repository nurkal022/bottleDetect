"""
Обучение одной YOLO-cls модели.
Пример: python scripts/train.py --weights yolov8n-cls.pt --name yolov8n-cls
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
MODELS_CFG = ROOT / "configs" / "models.yaml"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="например yolov8n-cls.pt")
    parser.add_argument("--name", required=True, help="имя run'а (папка в runs/classify/)")
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", default=None, help="cpu / 0 / mps")
    args = parser.parse_args()

    with open(MODELS_CFG) as f:
        cfg = yaml.safe_load(f)["train"]

    epochs = args.epochs or cfg["epochs"]
    imgsz = args.imgsz or cfg["imgsz"]
    batch = args.batch if args.batch is not None else cfg["batch"]

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=cfg["patience"],
        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"],
        seed=cfg["seed"],
        project=str(ROOT / "runs" / "classify"),
        name=args.name,
        exist_ok=True,
        device=args.device,
    )
    print(f"\nDone: {results.save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
