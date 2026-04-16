"""
Оценка всех обученных YOLO-cls моделей на test split.
Метрики: top-1 accuracy, top-5 accuracy, per-class accuracy (через confusion matrix).
Выход: results/metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs" / "classify"
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"


def eval_one(weights: Path, data: Path, split: str, device: str | None, imgsz: int) -> dict:
    model = YOLO(str(weights))
    metrics = model.val(data=str(data), split=split, device=device, imgsz=imgsz, verbose=False)
    return {
        "top1_acc": round(float(metrics.top1), 4),
        "top5_acc": round(float(metrics.top5), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--out", default=str(RESULTS_DIR / "metrics.csv"))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for run_dir in sorted(RUNS_DIR.iterdir()):
        weights = run_dir / "weights" / "best.pt"
        if not weights.exists():
            continue
        print(f"\n--- {run_dir.name} ---")
        try:
            m = eval_one(weights, Path(args.data), args.split, args.device, args.imgsz)
            rows.append({"model": run_dir.name, **m})
            print(m)
        except Exception as e:
            print(f"ERROR: {e}")

    if rows:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
