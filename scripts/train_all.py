"""
Последовательно обучает все YOLO-cls модели из configs/models.yaml.
Пример: python scripts/train_all.py
        python scripts/train_all.py --device mps
        python scripts/train_all.py --device 0   (NVIDIA GPU)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_CFG = ROOT / "configs" / "models.yaml"
TRAIN_SCRIPT = ROOT / "scripts" / "train.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()

    with open(MODELS_CFG) as f:
        models = yaml.safe_load(f)["models"]

    for m in models:
        print(f"\n{'=' * 60}\nTRAINING  {m['name']}\n{'=' * 60}")
        cmd = [sys.executable, str(TRAIN_SCRIPT),
               "--weights", m["weights"], "--name", m["name"]]
        if args.device is not None:
            cmd += ["--device", str(args.device)]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        if args.batch:
            cmd += ["--batch", str(args.batch)]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"FAILED: {m['name']} (rc={rc}) — продолжаю дальше")

    print("\nВсе модели обучены. Дальше: python scripts/evaluate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
