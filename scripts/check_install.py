"""
Проверка корректности установки проекта.
Запуск: python scripts/check_install.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check(condition: bool, label: str) -> bool:
    mark = "✓" if condition else "✗"
    print(f"  {mark} {label}")
    return condition


def main() -> int:
    print("=== Проверка установки bottleDetect ===\n")
    ok = True

    # 1. Python
    print("Python:")
    ok &= check(sys.version_info >= (3, 10),
                f"Python {sys.version.split()[0]} (нужен 3.10+)")

    # 2. Пакеты
    print("\nПакеты:")
    packages = {
        "ultralytics": "ultralytics",
        "cv2": "opencv-python",
        "PySide6": "PySide6",
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
    }
    for mod, pkg in packages.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            check(True, f"{pkg} ({ver})")
        except ImportError:
            ok &= check(False, f"{pkg} — не установлен (pip install {pkg})")

    # 3. Модель
    print("\nМодель:")
    weights = ROOT / "runs_det" / "yolov8n_det" / "weights" / "best.pt"
    ok &= check(weights.exists(), f"Веса: {weights.relative_to(ROOT)}")

    if weights.exists():
        try:
            from ultralytics import YOLO
            m = YOLO(str(weights))
            names = list(m.names.values()) if isinstance(m.names, dict) else list(m.names)
            task = getattr(m, "task", "?")
            check(True, f"Модель загружается (task={task}, {len(names)} классов)")
            check(names == ["deformed_bottl", "misaligned_label", "missing_cap",
                            "missing_label", "neck_deformed", "normal"],
                  f"Классы: {names}")
        except Exception as e:
            ok = check(False, f"Не удалось загрузить: {e}") and ok

    # 4. Структура
    print("\nСтруктура:")
    for p in ["app_native.py", "scripts/test_inference.py",
              "scripts/train.py", "results/report.md"]:
        ok &= check((ROOT / p).exists(), p)

    # 5. GPU (опционально)
    print("\nGPU (опционально, для обучения):")
    try:
        import torch
        if torch.cuda.is_available():
            check(True, f"CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            check(True, "Apple Silicon MPS доступен")
        else:
            check(True, "GPU нет — будет работать на CPU (33 FPS)")
    except Exception:
        pass

    print("\n" + ("=== ✅ Всё ОК — запускай: python app_native.py ==="
                  if ok else
                  "=== ✗ Есть проблемы — см. выше ==="))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
