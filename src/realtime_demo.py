"""
Real-time классификация дефектов бутылок с веб-камеры/видео.

Модель получает кадр целиком → предсказывает класс дефекта.
Поверх кадра рисуется: predicted class, confidence, FPS, счётчики классов.

Примеры:
  python src/realtime_demo.py --source 0 --model runs/classify/yolo11n-cls/weights/best.pt
  python src/realtime_demo.py --source video.mp4 --model <path> --save out.mp4

Управление:
  q — выход   r — сброс счётчиков   s — скриншот   p — пауза
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

# Цвет бокса оверлея для каждого класса
CLASS_COLORS = {
    "good_bottle":    (50, 200, 50),    # зелёный
    "cork_missmatch": (0, 100, 255),    # оранжевый
    "label_missing":  (255, 200, 0),    # жёлтый
    "shape_mismatch": (0, 0, 220),      # красный
}
DEFAULT_COLOR = (180, 180, 180)


def draw_overlay(frame, cls_name: str, conf: float, fps: float, counts: Counter) -> None:
    h, w = frame.shape[:2]
    color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

    # Рамка вокруг кадра — цветная по классу
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 4)

    # Блок слева сверху: класс + уверенность
    label = f"{cls_name}  {conf * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.rectangle(frame, (0, 0), (tw + 16, th + 16), color, -1)
    cv2.putText(frame, label, (8, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Блок справа сверху: FPS
    fps_txt = f"FPS: {fps:.1f}"
    (fw, fh), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (w - fw - 16, 0), (w, fh + 12), (0, 0, 0), -1)
    cv2.putText(frame, fps_txt, (w - fw - 8, fh + 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Счётчики снизу
    pad, lh = 8, 22
    lines = [f"{k}: {v}" for k, v in sorted(counts.items())]
    lines.append(f"total: {sum(counts.values())}")
    box_h = pad * 2 + lh * len(lines)
    cv2.rectangle(frame, (0, h - box_h), (220, h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        y = h - box_h + pad + (i + 1) * lh - 4
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True, help="0 / video.mp4 / folder/")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="минимальная уверенность для счётчика")
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--save", default=None, help="путь для сохранения видео")
    parser.add_argument("--log", default=None, help="путь для CSV-лога")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.model)
    class_names: dict = model.names  # {0: 'cork_missmatch', 1: 'good_bottle', ...}

    # Открываем источник
    source = args.source
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        src_type = "stream"
        frames_dir = None
    else:
        p = Path(source)
        if p.is_dir():
            frames_dir = sorted([x for x in p.iterdir()
                                 if x.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            src_type = "folder"
            cap = None
        else:
            cap = cv2.VideoCapture(str(p))
            src_type = "video"
            frames_dir = None

    writer = None
    log_file = None
    log_csv = None
    if args.log:
        log_file = open(args.log, "w", newline="")
        log_csv = csv.writer(log_file)
        log_csv.writerow(["timestamp", "frame_idx", "class", "confidence"])

    counts: Counter = Counter()
    idx = 0
    paused = False
    t_last = time.perf_counter()
    fps_ema = 0.0
    cls_name, conf_val = "—", 0.0

    try:
        while True:
            if not paused:
                # Читаем кадр
                if src_type == "folder":
                    if idx >= len(frames_dir):
                        break
                    frame = cv2.imread(str(frames_dir[idx]))
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break

                # Инференс
                res = model.predict(frame, device=args.device, imgsz=args.imgsz,
                                    verbose=False)[0]
                top_idx = int(res.probs.top1)
                top_conf = float(res.probs.top1conf)
                cls_name = class_names[top_idx]
                conf_val = top_conf

                if top_conf >= args.conf:
                    counts[cls_name] += 1

                if log_csv:
                    log_csv.writerow([datetime.utcnow().isoformat(), idx,
                                      cls_name, f"{top_conf:.3f}"])

                # FPS
                now = time.perf_counter()
                dt = now - t_last
                t_last = now
                fps_ema = (1.0 / dt) if fps_ema == 0 else 0.9 * fps_ema + 0.1 / max(dt, 1e-6)

                annotated = frame.copy()
                draw_overlay(annotated, cls_name, conf_val, fps_ema, counts)

                if args.save and writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        args.save, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
                if writer:
                    writer.write(annotated)

                if not args.no_show:
                    cv2.imshow("Bottle Defect Classifier", annotated)

                idx += 1

            key = cv2.waitKey(1 if not paused else 50) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                counts.clear()
            elif key == ord("p"):
                paused = not paused
            elif key == ord("s"):
                snap = ROOT / f"screenshot_{int(time.time())}.png"
                cv2.imwrite(str(snap), annotated)
                print(f"Saved {snap}")
    finally:
        if cap:
            cap.release()
        if writer:
            writer.release()
        if log_file:
            log_file.close()
        cv2.destroyAllWindows()

    print("\n=== Итого детекций ===")
    for cls, cnt in sorted(counts.items()):
        print(f"  {cls}: {cnt}")
    print(f"  total: {sum(counts.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
