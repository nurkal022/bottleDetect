"""
Измерение скорости инференса YOLO-cls моделей.
Выход: results/speed.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs" / "classify"
RESULTS_DIR = ROOT / "results"


def get_model_meta(weights: Path) -> dict:
    """Параметры и размер весов."""
    import torch
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    mdl = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        n_params = sum(p.numel() for p in mdl.parameters())
    except AttributeError:
        n_params = 0
    return {
        "params_M": round(n_params / 1e6, 3),
        "weights_MB": round(weights.stat().st_size / 1e6, 2),
    }


def measure_fps(weights: Path, device: str, imgsz: int, n_warmup: int, n_iters: int) -> dict:
    model = YOLO(str(weights))
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    for _ in range(n_warmup):
        model.predict(dummy, device=device, imgsz=imgsz, verbose=False)

    latencies = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        model.predict(dummy, device=device, imgsz=imgsz, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    mean_ms = sum(latencies) / len(latencies)
    return {
        "device": device,
        "fps": round(1000.0 / mean_ms, 1),
        "latency_ms_mean": round(mean_ms, 2),
        "latency_ms_p95": round(latencies[int(len(latencies) * 0.95)], 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", default=["cpu"],
                        help="например: cpu mps 0")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", default=str(RESULTS_DIR / "speed.csv"))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for run_dir in sorted(RUNS_DIR.iterdir()):
        weights = run_dir / "weights" / "best.pt"
        if not weights.exists():
            continue
        meta = get_model_meta(weights)
        for dev in args.devices:
            try:
                print(f"\n--- {run_dir.name} @ {dev} ---")
                spd = measure_fps(weights, dev, args.imgsz, args.warmup, args.iters)
                rows.append({"model": run_dir.name, **meta, **spd})
                print(spd)
            except Exception as e:
                print(f"FAIL: {e}")

    if rows:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
