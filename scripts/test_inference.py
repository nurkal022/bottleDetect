"""
Прогон обученной модели YOLOv8n detection по всем файлам в test/.
Сохраняет результаты (аннотированные фото и видео) в ту же папку с префиксом out_.
Создаёт test/predictions.csv с детекциями по каждому файлу.

Запуск: python scripts/test_inference.py
"""
from __future__ import annotations

import csv
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "test"
MODEL_PATH = ROOT / "runs_det" / "yolov8n_det" / "weights" / "best.pt"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

CONF_THRESH = 0.5          # выше = меньше ложных
MAX_BBOX_AREA_RATIO = 0.75  # боксы > 75% кадра — выбрасываем
MIN_BBOX_AREA_RATIO = 0.002
BBOX_EMA_ALPHA = 0.6        # сглаживание координат между кадрами

CLASS_COLORS = {
    "normal":           (50,  205,  50),
    "deformed_bottl":   (30,   30, 220),
    "neck_deformed":    (30,   30, 220),
    "missing_cap":      (30,  100, 255),
    "missing_label":    (0,  215, 255),
    "misaligned_label": (0,  165, 255),
}
DEFAULT_COLOR = (140, 140, 140)


def annotate(frame, detections, names):
    out = frame.copy()
    for cls_id, conf, (x1, y1, x2, y2) in detections:
        name = names[cls_id]
        color = CLASS_COLORS.get(name, DEFAULT_COLOR)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = f"{name} {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x1, max(0, y1-th-10)), (x1+tw+10, y1), color, -1)
        cv2.putText(out, label, (x1+5, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def names_list(model):
    names = model.names
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def _filter_by_area(dets, frame_shape):
    h, w = frame_shape[:2]
    area = float(h * w)
    out = []
    for d in dets:
        _, _, (x1, y1, x2, y2) = d[:3]
        a = max(0.0, (x2 - x1) * (y2 - y1))
        r = a / area
        if MIN_BBOX_AREA_RATIO <= r <= MAX_BBOX_AREA_RATIO:
            out.append(d)
    return out


def process_image(model, names, path: Path, out_dir: Path) -> dict:
    img = cv2.imread(str(path))
    if img is None:
        return {"file": path.name, "type": "image", "error": "cannot read"}

    t0 = time.perf_counter()
    res = model.predict(img, imgsz=640, conf=CONF_THRESH, iou=0.5,
                        verbose=False, device="cpu")[0]
    dt_ms = (time.perf_counter() - t0) * 1000

    dets = []
    if res.boxes is not None and len(res.boxes) > 0:
        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            dets.append((cls_id, conf, tuple(xyxy)))
    dets = _filter_by_area(dets, img.shape)

    annotated = annotate(img, dets, names)
    out_path = out_dir / f"out_{path.stem}.jpg"
    cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

    counts = Counter()
    for cls_id, conf, _ in dets:
        counts[names[cls_id]] += 1

    return {
        "file": path.name,
        "type": "image",
        "detections": len(dets),
        "classes": ", ".join(f"{k}:{v}" for k, v in counts.items()) or "—",
        "latency_ms": round(dt_ms, 1),
        "output": out_path.name,
    }


def process_video(model, names, path: Path, out_dir: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"file": path.name, "type": "video", "error": "cannot open"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = out_dir / f"out_{path.stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    counts_unique = Counter()
    seen_ids = set()
    smoothed: dict[int, list[float]] = {}  # track_id -> smoothed xyxy
    frames_done = 0
    t_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_area = float(frame.shape[0] * frame.shape[1])

        # Track для уникальных объектов
        res = model.track(frame, imgsz=640, conf=CONF_THRESH, iou=0.5,
                          persist=True, tracker="bytetrack.yaml",
                          verbose=False, device="cpu")[0]

        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            ids = res.boxes.id.int().cpu().tolist() \
                if res.boxes.id is not None else [-1] * len(res.boxes)
            for i in range(len(res.boxes)):
                cls_id = int(res.boxes.cls[i])
                conf = float(res.boxes.conf[i])
                raw = res.boxes.xyxy[i].cpu().numpy().astype(float).tolist()
                x1, y1, x2, y2 = raw
                a = max(0.0, (x2 - x1) * (y2 - y1))
                r = a / frame_area
                if not (MIN_BBOX_AREA_RATIO <= r <= MAX_BBOX_AREA_RATIO):
                    continue
                tid = ids[i]
                # EMA сглаживание координат по track_id
                if tid >= 0 and tid in smoothed:
                    alpha = BBOX_EMA_ALPHA
                    raw = [alpha * p + (1 - alpha) * c
                           for p, c in zip(smoothed[tid], raw)]
                if tid >= 0:
                    smoothed[tid] = raw
                xyxy = tuple(int(round(v)) for v in raw)
                dets.append((cls_id, conf, xyxy))
                if tid >= 0:
                    key = (tid, cls_id)
                    if key not in seen_ids:
                        seen_ids.add(key)
                        counts_unique[names[cls_id]] += 1

        annotated = annotate(frame, dets, names)
        writer.write(annotated)
        frames_done += 1
        if frames_done % 100 == 0:
            print(f"    {frames_done}/{total} frames", flush=True)

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t_start

    return {
        "file": path.name,
        "type": "video",
        "frames": frames_done,
        "unique_objects": sum(counts_unique.values()),
        "classes": ", ".join(f"{k}:{v}" for k, v in counts_unique.items()) or "—",
        "elapsed_s": round(elapsed, 1),
        "avg_fps": round(frames_done / max(elapsed, 1e-6), 1),
        "output": out_path.name,
    }


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"Модель не найдена: {MODEL_PATH}")
        return 1
    if not TEST_DIR.exists():
        print(f"Папка test/ не найдена")
        return 1

    print(f"Загружаю модель: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    names = names_list(model)
    print(f"Классы: {names}")

    files = sorted([f for f in TEST_DIR.iterdir()
                    if f.is_file() and not f.name.startswith("out_")
                    and f.suffix.lower() in IMG_EXT | VID_EXT])
    print(f"Найдено файлов для обработки: {len(files)}\n")

    rows = []
    for f in files:
        print(f"→ {f.name}")
        if f.suffix.lower() in IMG_EXT:
            r = process_image(model, names, f, TEST_DIR)
        else:
            r = process_video(model, names, f, TEST_DIR)
        rows.append(r)
        for k, v in r.items():
            if k != "file":
                print(f"    {k}: {v}")
        print()

    # CSV
    if rows:
        all_keys = sorted({k for r in rows for k in r.keys()})
        csv_path = TEST_DIR / "predictions.csv"
        with open(csv_path, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=all_keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nСводка: {csv_path}")

    print("\nГотово. Результаты в test/ с префиксом out_")
    return 0


if __name__ == "__main__":
    sys.exit(main())
