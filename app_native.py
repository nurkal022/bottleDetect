"""
Bottle Defect Classifier — нативное десктопное приложение.
Требования: PySide6, ultralytics, opencv-python

Запуск:
    python app_native.py
"""
from __future__ import annotations

import csv
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import (Qt, QThread, Signal, QSize, QTimer, QMutex,
                             QMutexLocker)
from PySide6.QtGui import (QImage, QPixmap, QPalette, QColor, QFont,
                            QIcon, QAction)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QFileDialog, QSizePolicy,
    QFrame, QProgressBar, QStatusBar, QSplitter, QScrollArea,
    QButtonGroup, QRadioButton, QGroupBox, QTextEdit, QMessageBox,
)
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RUNS_DIR_CLS = ROOT / "runs" / "classify"
RUNS_DIR_DET = ROOT / "runs_det"
LOG_DIR = ROOT / "results"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Detection classes (main task now) + legacy classification classes
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    # Detection (yolov8_bottle dataset)
    "normal":           (50,  205,  50),
    "deformed_bottl":   (220,  30,  30),
    "neck_deformed":    (220,  30,  30),
    "missing_cap":      (255, 100,  30),
    "missing_label":    (255, 215,   0),
    "misaligned_label": (255, 165,   0),
    # Legacy classification
    "good_bottle":      (50,  205,  50),
    "cork_missmatch":   (255, 100,  30),
    "label_missing":    (255, 215,   0),
    "shape_mismatch":   (220,  30,  30),
}
DEFAULT_COLOR = (140, 140, 140)

QSS_DARK = """
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'SF Pro Text', sans-serif;
    font-size: 13px;
}
QMainWindow { background-color: #181825; }
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 6px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 14px;
    color: #cdd6f4;
    min-height: 28px;
}
QPushButton:hover  { background-color: #45475a; border-color: #89b4fa; }
QPushButton:pressed { background-color: #1e1e2e; }
QPushButton:disabled { color: #585b70; border-color: #313244; }
QPushButton#btn_primary {
    background-color: #89b4fa;
    color: #1e1e2e;
    font-weight: bold;
    border: none;
}
QPushButton#btn_primary:hover  { background-color: #b4befe; }
QPushButton#btn_primary:pressed { background-color: #74c7ec; }
QPushButton#btn_danger {
    background-color: #f38ba8;
    color: #1e1e2e;
    font-weight: bold;
    border: none;
}
QPushButton#btn_danger:hover  { background-color: #eba0ac; }
QComboBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 28px;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #313244;
    border: 1px solid #45475a;
    selection-background-color: #45475a;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #45475a;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    width: 14px; height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal { background: #89b4fa; border-radius: 2px; }
QRadioButton { spacing: 6px; }
QRadioButton::indicator {
    width: 14px; height: 14px;
    border-radius: 7px;
    border: 2px solid #45475a;
    background: #1e1e2e;
}
QRadioButton::indicator:checked {
    background: #89b4fa;
    border-color: #89b4fa;
}
QLabel#video_display {
    background-color: #11111b;
    border: 1px solid #313244;
    border-radius: 8px;
}
QLabel#class_badge {
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 18px;
    font-weight: bold;
}
QProgressBar {
    background-color: #313244;
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
}
QProgressBar::chunk { border-radius: 3px; }
QStatusBar { background-color: #181825; border-top: 1px solid #313244; }
QFrame#separator { background-color: #45475a; max-height: 1px; }
QScrollArea { border: none; }
QTextEdit {
    background-color: #11111b;
    border: 1px solid #313244;
    border-radius: 6px;
    font-family: monospace;
    font-size: 11px;
    color: #a6e3a1;
}
"""


# ─── Inference worker ──────────────────────────────────────────────────────────

class InferenceWorker(QThread):
    frame_ready   = Signal(np.ndarray, str, float, float)  # frame, cls, conf, fps
    stats_updated = Signal(dict)                           # counts dict
    error         = Signal(str)
    finished_work = Signal()

    def __init__(self, model_path: str, source, conf_thresh: float = 0.5,
                 device: str = "cpu"):
        super().__init__()
        self.model_path  = model_path
        self.source      = source       # int | str | Path
        self.conf_thresh = conf_thresh
        self.device      = device
        self._stop       = False
        self._mutex      = QMutex()
        self.counts: Counter = Counter()
        self.total_frames = 0

    def stop(self):
        with QMutexLocker(self._mutex):
            self._stop = True

    def run(self):
        try:
            model = YOLO(self.model_path)
            class_names: dict = model.names
        except Exception as e:
            self.error.emit(f"Не удалось загрузить модель: {e}")
            return

        # Источник: папка с фото, видео, вебкам
        if isinstance(self.source, list):
            self._run_folder(model, class_names)
        else:
            self._run_stream(model, class_names)

        self.finished_work.emit()

    MAX_BBOX_AREA_RATIO = 0.75  # отбрасываем боксы покрывающие > 75% кадра
    MIN_BBOX_AREA_RATIO = 0.002  # и очень мелкие (< 0.2%)
    BBOX_EMA_ALPHA = 0.6         # сглаживание координат между кадрами (0=жёстко, 1=плавно)

    def _is_detection(self, model) -> bool:
        return getattr(model, "task", "") == "detect"

    def _infer(self, model, frame, use_tracking: bool = False,
               smoothed_boxes: dict | None = None):
        """Returns list of detections: [(cls_id, conf, (x1,y1,x2,y2), track_id), ...]"""
        if self._is_detection(model):
            imgsz = 640  # как при обучении — точнее bbox
            if use_tracking:
                res = model.track(frame, imgsz=imgsz, conf=self.conf_thresh,
                                  iou=0.5, persist=True, tracker="bytetrack.yaml",
                                  verbose=False, device=self.device)[0]
            else:
                res = model.predict(frame, imgsz=imgsz, conf=self.conf_thresh,
                                    iou=0.5, verbose=False, device=self.device)[0]
            dets = []
            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None \
                      else [-1] * len(boxes)
                h, w = frame.shape[:2]
                frame_area = float(h * w)
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(float).tolist()
                    x1, y1, x2, y2 = xyxy
                    area = max(0.0, (x2 - x1) * (y2 - y1))
                    # Фильтр: слишком большие или слишком маленькие
                    if area / frame_area > self.MAX_BBOX_AREA_RATIO:
                        continue
                    if area / frame_area < self.MIN_BBOX_AREA_RATIO:
                        continue
                    tid = ids[i]
                    # EMA-сглаживание координат для стабильных трек-ID
                    if smoothed_boxes is not None and tid >= 0 and tid in smoothed_boxes:
                        a = self.BBOX_EMA_ALPHA
                        px = smoothed_boxes[tid]
                        xyxy = [a * p + (1 - a) * c for p, c in zip(px, xyxy)]
                    if smoothed_boxes is not None and tid >= 0:
                        smoothed_boxes[tid] = xyxy
                    dets.append((cls_id, conf,
                                 tuple(int(round(v)) for v in xyxy), tid))
            return dets
        else:
            res = model.predict(frame, imgsz=224, verbose=False,
                                device=self.device)[0]
            cls_id = int(res.probs.top1)
            conf = float(res.probs.top1conf)
            h, w = frame.shape[:2]
            return [(cls_id, conf, (0, 0, w, h), -1)]

    def _annotate(self, frame, detections, class_names):
        out = frame.copy()
        for det in detections:
            cls_id, conf, (x1, y1, x2, y2), tid = det
            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, max(0, y1-th-8)), (x1+tw+8, y1), color, -1)
            cv2.putText(out, label, (x1+4, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        return out

    def _names_list(self, model) -> list:
        names = model.names
        if isinstance(names, dict):
            return [names[i] for i in sorted(names)]
        return list(names)

    def _run_stream(self, model, class_names):
        names = self._names_list(model)
        is_det = self._is_detection(model)
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error.emit(f"Не удалось открыть источник: {self.source}")
            return

        # Downscale большие кадры — быстрее и стабильнее
        target_w = 960

        fps_buf = []
        seen_ids: set[tuple[int, int]] = set()  # (track_id, cls_id) — уникальные
        last_dets: list = []  # для smoothing/fallback между кадрами
        miss_counter = 0
        smoothed_boxes: dict[int, list[float]] = {}  # track_id -> [x1,y1,x2,y2] EMA

        while True:
            with QMutexLocker(self._mutex):
                if self._stop:
                    break
            ok, frame = cap.read()
            if not ok:
                break

            # Downscale
            h0, w0 = frame.shape[:2]
            if w0 > target_w:
                scale = target_w / w0
                frame = cv2.resize(frame, (target_w, int(h0 * scale)),
                                   interpolation=cv2.INTER_LINEAR)

            t_start = time.perf_counter()
            dets = self._infer(model, frame, use_tracking=is_det,
                               smoothed_boxes=smoothed_boxes if is_det else None)
            t_inf = time.perf_counter() - t_start

            # Anti-flicker: если на 1-2 кадрах объекты не найдены — показываем предыдущие
            if not dets and last_dets and miss_counter < 3:
                dets = last_dets
                miss_counter += 1
            else:
                if dets:
                    last_dets = dets
                    miss_counter = 0
                else:
                    last_dets = []

            # top detection — для бейджа
            if dets:
                top = max(dets, key=lambda d: d[1])
                top_name = names[top[0]]
                top_conf = top[1]
            else:
                top_name, top_conf = "—", 0.0

            # Счётчики: по уникальным track_id (если трекинг есть)
            for cls_id, conf, _, tid in dets:
                if conf < self.conf_thresh:
                    continue
                if is_det and tid >= 0:
                    key = (tid, cls_id)
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                self.counts[names[cls_id]] += 1
            self.total_frames += 1

            annotated = self._annotate(frame, dets, names)

            fps_buf.append(1.0 / max(t_inf, 1e-6))
            if len(fps_buf) > 30:
                fps_buf.pop(0)
            fps = sum(fps_buf) / len(fps_buf)

            self.frame_ready.emit(annotated, top_name, top_conf, fps)
            self.stats_updated.emit(dict(self.counts))

        cap.release()

    def _run_folder(self, model, class_names):
        names = self._names_list(model)
        for p in self.source:
            with QMutexLocker(self._mutex):
                if self._stop:
                    break
            frame = cv2.imread(str(p))
            if frame is None:
                continue

            t_start = time.perf_counter()
            dets = self._infer(model, frame, use_tracking=False)
            fps = 1.0 / max(time.perf_counter() - t_start, 1e-6)

            if dets:
                top = max(dets, key=lambda d: d[1])
                top_name, top_conf = names[top[0]], top[1]
            else:
                top_name, top_conf = "—", 0.0

            for cls_id, conf, _, _ in dets:
                if conf >= self.conf_thresh:
                    self.counts[names[cls_id]] += 1
            self.total_frames += 1

            annotated = self._annotate(frame, dets, names)
            self.frame_ready.emit(annotated, top_name, top_conf, fps)
            self.stats_updated.emit(dict(self.counts))
            self.msleep(50)


# ─── Виджет: класс-бейдж ──────────────────────────────────────────────────────

class ClassBadge(QLabel):
    _BG = {
        "normal":           "#32c832",
        "deformed_bottl":   "#dc1e1e",
        "neck_deformed":    "#c71e3a",
        "missing_cap":      "#ff6420",
        "missing_label":    "#ffd700",
        "misaligned_label": "#ffa500",
        "good_bottle":      "#32c832",
        "cork_missmatch":   "#ff6420",
        "label_missing":    "#ffd700",
        "shape_mismatch":   "#dc1e1e",
        "—":                "#585b70",
    }
    _FG = {
        "normal":           "#0a2a0a",
        "deformed_bottl":   "#1a0000",
        "neck_deformed":    "#1a0000",
        "missing_cap":      "#1a0800",
        "missing_label":    "#1a1600",
        "misaligned_label": "#1a0e00",
        "good_bottle":      "#0a2a0a",
        "cork_missmatch":   "#1a0800",
        "label_missing":    "#1a1600",
        "shape_mismatch":   "#1a0000",
        "—":                "#cdd6f4",
    }

    def __init__(self, parent=None):
        super().__init__("—", parent)
        self.setObjectName("class_badge")
        self.setAlignment(Qt.AlignCenter)
        self._update("—", 0.0)

    def update_class(self, cls_name: str, conf: float):
        self._update(cls_name, conf)

    def _update(self, cls_name: str, conf: float):
        bg = self._BG.get(cls_name, self._BG["—"])
        fg = self._FG.get(cls_name, self._FG["—"])
        conf_str = f"  {conf*100:.1f}%" if conf > 0 else ""
        self.setText(cls_name + conf_str)
        self.setStyleSheet(
            f"background-color:{bg}; color:{fg}; border-radius:8px;"
            f" padding:8px 16px; font-size:17px; font-weight:bold;"
        )


# ─── Виджет: строка счётчика ──────────────────────────────────────────────────

class CountRow(QWidget):
    _COLORS = {
        # Detection classes
        "normal":           "#32c832",
        "deformed_bottl":   "#dc1e1e",
        "neck_deformed":    "#c71e3a",
        "missing_cap":      "#ff6420",
        "missing_label":    "#ffd700",
        "misaligned_label": "#ffa500",
        # Legacy cls
        "good_bottle":      "#32c832",
        "cork_missmatch":   "#ff6420",
        "label_missing":    "#ffd700",
        "shape_mismatch":   "#dc1e1e",
    }

    def __init__(self, cls_name: str, parent=None):
        super().__init__(parent)
        self.cls_name = cls_name
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)

        top = QHBoxLayout()
        self.lbl_name = QLabel(cls_name)
        self.lbl_name.setStyleSheet("font-size:11px; color:#a6adc8;")
        self.lbl_count = QLabel("0")
        self.lbl_count.setStyleSheet("font-size:14px; font-weight:bold; color:#cdd6f4;")
        top.addWidget(self.lbl_name)
        top.addStretch()
        top.addWidget(self.lbl_count)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        color = self._COLORS.get(cls_name, "#89b4fa")
        self.bar.setStyleSheet(
            f"QProgressBar::chunk {{ background: {color}; }}"
            "QProgressBar { background:#313244; border:none; border-radius:3px; height:5px; }"
        )

        layout.addLayout(top)
        layout.addWidget(self.bar)

    def update_count(self, count: int, total: int):
        self.lbl_count.setText(str(count))
        pct = int((count / total * 100)) if total > 0 else 0
        self.bar.setValue(pct)


# ─── Главное окно ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    # Detection classes from yolov8_bottle dataset
    CLASS_NAMES = ["deformed_bottl", "misaligned_label", "missing_cap",
                   "missing_label", "neck_deformed", "normal"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bottle Defect Classifier")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 780)

        self._worker: InferenceWorker | None = None
        self._log_rows: list[dict] = []
        self._last_cls = "—"
        self._last_conf = 0.0
        self._last_fps = 0.0
        self._current_source = None   # int | str | list[Path]

        self._build_ui()
        self._refresh_models()
        self._set_status("Выберите модель и режим")

    # ── Build UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # splitter = left | center | right
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([220, 680, 240])
        splitter.setHandleWidth(4)

        self.setStatusBar(QStatusBar())

    # ── Left panel ──────────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(220)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Model selection
        grp_model = QGroupBox("Модель")
        vl = QVBoxLayout(grp_model)
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        vl.addWidget(self.model_combo)
        self.lbl_model_info = QLabel("—")
        self.lbl_model_info.setStyleSheet("color:#a6adc8; font-size:11px;")
        self.lbl_model_info.setWordWrap(True)
        vl.addWidget(self.lbl_model_info)
        layout.addWidget(grp_model)

        # Mode
        grp_mode = QGroupBox("Режим")
        vl2 = QVBoxLayout(grp_mode)
        self.bg_mode = QButtonGroup(self)
        for idx, (label, icon_char) in enumerate([
            ("Фото", "📷"), ("Видео", "🎬"), ("Live (камера)", "📹")
        ]):
            rb = QRadioButton(f"  {icon_char}  {label}")
            self.bg_mode.addButton(rb, idx)
            vl2.addWidget(rb)
        self.bg_mode.button(0).setChecked(True)
        layout.addWidget(grp_mode)

        # Confidence threshold
        grp_conf = QGroupBox("Порог уверенности")
        vl3 = QVBoxLayout(grp_conf)
        conf_row = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 95)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(
            lambda v: self.lbl_conf.setText(f"{v}%")
        )
        self.lbl_conf = QLabel("50%")
        self.lbl_conf.setFixedWidth(36)
        conf_row.addWidget(self.conf_slider)
        conf_row.addWidget(self.lbl_conf)
        vl3.addLayout(conf_row)
        layout.addWidget(grp_conf)

        # Actions
        grp_actions = QGroupBox("Управление")
        vl4 = QVBoxLayout(grp_actions)
        vl4.setSpacing(6)

        self.btn_open = QPushButton("Открыть файл / папку")
        self.btn_open.clicked.connect(self._on_open)

        self.btn_start = QPushButton("▶  Старт")
        self.btn_start.setObjectName("btn_primary")
        self.btn_start.clicked.connect(self._on_start)

        self.btn_stop = QPushButton("■  Стоп")
        self.btn_stop.setObjectName("btn_danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)

        self.btn_reset = QPushButton("↺  Сброс счётчиков")
        self.btn_reset.clicked.connect(self._on_reset)

        self.btn_save_log = QPushButton("💾  Сохранить лог")
        self.btn_save_log.clicked.connect(self._on_save_log)

        for btn in [self.btn_open, self.btn_start, self.btn_stop,
                    self.btn_reset, self.btn_save_log]:
            vl4.addWidget(btn)

        layout.addWidget(grp_actions)
        layout.addStretch()

        # Version info
        lbl_ver = QLabel("YOLOv8 / YOLOv11\nbottleDetect v1.0")
        lbl_ver.setStyleSheet("color:#585b70; font-size:10px;")
        lbl_ver.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_ver)

        return w

    # ── Center panel ─────────────────────────────────────────────────────────

    def _build_center_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Video display
        self.video_label = QLabel()
        self.video_label.setObjectName("video_display")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(400, 300)
        self._show_placeholder()
        layout.addWidget(self.video_label, stretch=1)

        # Class badge + FPS row
        bottom = QHBoxLayout()
        self.badge = ClassBadge()
        self.badge.setMinimumWidth(260)

        fps_widget = QWidget()
        fps_layout = QVBoxLayout(fps_widget)
        fps_layout.setContentsMargins(12, 4, 12, 4)
        self.lbl_fps_big = QLabel("— FPS")
        self.lbl_fps_big.setStyleSheet(
            "font-size:28px; font-weight:bold; color:#89b4fa;"
        )
        self.lbl_fps_big.setAlignment(Qt.AlignCenter)
        self.lbl_device_info = QLabel("")
        self.lbl_device_info.setStyleSheet("color:#a6adc8; font-size:11px;")
        self.lbl_device_info.setAlignment(Qt.AlignCenter)
        fps_layout.addWidget(self.lbl_fps_big)
        fps_layout.addWidget(self.lbl_device_info)

        bottom.addWidget(self.badge, stretch=2)
        bottom.addStretch()
        bottom.addWidget(fps_widget, stretch=1)
        layout.addLayout(bottom)

        return w

    # ── Right panel ──────────────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(240)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Defect counters
        grp_counts = QGroupBox("Счётчики")
        vl = QVBoxLayout(grp_counts)
        vl.setSpacing(8)
        self.count_rows: dict[str, CountRow] = {}
        for cls in self.CLASS_NAMES:
            row = CountRow(cls)
            self.count_rows[cls] = row
            vl.addWidget(row)
        layout.addWidget(grp_counts)

        # Summary
        grp_sum = QGroupBox("Итого")
        vl2 = QVBoxLayout(grp_sum)

        def _stat_row(label, attr):
            hl = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet("color:#a6adc8;")
            val = QLabel("0")
            val.setStyleSheet("font-weight:bold; color:#cdd6f4;")
            val.setAlignment(Qt.AlignRight)
            hl.addWidget(lbl)
            hl.addWidget(val)
            setattr(self, attr, val)
            return hl

        vl2.addLayout(_stat_row("Кадров:", "lbl_total_frames"))
        vl2.addLayout(_stat_row("Обнаружено:", "lbl_total_detected"))
        vl2.addLayout(_stat_row("Дефектов:", "lbl_total_defects"))

        defect_rate_row = QHBoxLayout()
        dr_lbl = QLabel("Доля дефектов:")
        dr_lbl.setStyleSheet("color:#a6adc8;")
        self.lbl_defect_rate = QLabel("0%")
        self.lbl_defect_rate.setStyleSheet(
            "font-weight:bold; color:#f38ba8; font-size:15px;"
        )
        self.lbl_defect_rate.setAlignment(Qt.AlignRight)
        defect_rate_row.addWidget(dr_lbl)
        defect_rate_row.addWidget(self.lbl_defect_rate)
        vl2.addLayout(defect_rate_row)
        layout.addWidget(grp_sum)

        # Log console
        grp_log = QGroupBox("Лог")
        vl3 = QVBoxLayout(grp_log)
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(150)
        vl3.addWidget(self.log_console)
        layout.addWidget(grp_log)

        layout.addStretch()
        return w

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _show_placeholder(self):
        self.video_label.clear()
        self.video_label.setText(
            "Нет изображения\n\nВыберите модель и режим,\nзатем нажмите Старт"
        )
        self.video_label.setStyleSheet(
            "color:#585b70; font-size:14px; background:#11111b;"
            " border:1px solid #313244; border-radius:8px;"
        )

    def _set_status(self, msg: str):
        self.statusBar().showMessage(msg)

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_console.append(f"[{ts}] {msg}")

    def _refresh_models(self):
        self.model_combo.clear()
        # Detection runs (priority)
        if RUNS_DIR_DET.exists():
            for d in sorted(RUNS_DIR_DET.iterdir()):
                w = d / "weights" / "best.pt"
                if w.exists():
                    self.model_combo.addItem(f"{d.name} (detect)", userData=str(w))
        # Classification runs
        if RUNS_DIR_CLS.exists():
            for d in sorted(RUNS_DIR_CLS.iterdir()):
                w = d / "weights" / "best.pt"
                if w.exists():
                    self.model_combo.addItem(f"{d.name} (cls)", userData=str(w))
        if self.model_combo.count() == 0:
            self.model_combo.addItem("(нет обученных моделей)")

    def _current_model_path(self) -> str | None:
        data = self.model_combo.currentData()
        return data

    def _on_model_changed(self, name: str):
        path = self._current_model_path()
        if not path:
            return
        try:
            m = YOLO(path)
            n = sum(p.numel() for p in m.model.parameters())
            size = Path(path).stat().st_size / 1e6
            self.lbl_model_info.setText(
                f"Params: {n/1e6:.2f}M\nSize: {size:.1f} MB"
            )
        except Exception:
            self.lbl_model_info.setText("")

    # ── Slot: Open ────────────────────────────────────────────────────────────

    def _on_open(self):
        mode = self.bg_mode.checkedId()
        if mode == 0:  # photo
            path, _ = QFileDialog.getOpenFileName(
                self, "Открыть изображение", str(ROOT / "data"),
                "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
            )
            if path:
                self._current_source = path
                self._preview_image(path)
                self._log(f"Фото: {Path(path).name}")
        elif mode == 1:  # video
            path, _ = QFileDialog.getOpenFileName(
                self, "Открыть видео", str(ROOT),
                "Videos (*.mp4 *.avi *.mov *.mkv *.webm)"
            )
            if path:
                self._current_source = path
                self._log(f"Видео: {Path(path).name}")
        elif mode == 2:  # live
            self._current_source = 0
            self._log("Источник: веб-камера (0)")

    def _preview_image(self, path: str):
        img = cv2.imread(path)
        if img is not None:
            self._display_frame(img)

    # ── Slot: Start ───────────────────────────────────────────────────────────

    def _on_start(self):
        model_path = self._current_model_path()
        if not model_path:
            QMessageBox.warning(self, "Ошибка", "Выберите модель")
            return

        mode = self.bg_mode.checkedId()

        if mode == 2:  # live — источник камера
            self._current_source = 0
        elif self._current_source is None:
            QMessageBox.warning(
                self, "Ошибка",
                "Нажмите «Открыть файл» и выберите изображение или видео"
            )
            return

        if mode == 0:  # single image — just infer once
            self._infer_single(model_path, self._current_source)
            return

        # Video or live → start worker
        conf = self.conf_slider.value() / 100
        device = "cpu"  # для переносимости; можно добавить GPU toggle

        self._worker = InferenceWorker(
            model_path=model_path,
            source=self._current_source,
            conf_thresh=conf,
            device=device,
        )
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.stats_updated.connect(self._on_stats)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished_work.connect(self._on_worker_done)
        self._worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_status(
            f"▶  {['Фото','Видео','Live'][mode]} — {self.model_combo.currentText()}"
        )
        self._log(f"Запуск в режиме: {['Фото','Видео','Live'][mode]}")
        self.lbl_device_info.setText(f"device: {device}")

    def _infer_single(self, model_path: str, source: str):
        try:
            model = YOLO(model_path)
            img = cv2.imread(source)
            if img is None:
                QMessageBox.warning(self, "Ошибка", "Не удалось прочитать файл")
                return

            is_detect = getattr(model, "task", "") == "detect"
            names = model.names
            if isinstance(names, dict):
                names_list = [names[i] for i in sorted(names)]
            else:
                names_list = list(names)

            t0 = time.perf_counter()
            conf_t = self.conf_slider.value() / 100
            if is_detect:
                res = model.predict(img, imgsz=640, conf=conf_t, verbose=False)[0]
                dets = []
                if res.boxes is not None and len(res.boxes) > 0:
                    for box in res.boxes:
                        cid = int(box.cls[0])
                        cf = float(box.conf[0])
                        xy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        dets.append((cid, cf, tuple(xy)))
            else:
                res = model.predict(img, imgsz=224, verbose=False)[0]
                cid = int(res.probs.top1)
                cf = float(res.probs.top1conf)
                h, w = img.shape[:2]
                dets = [(cid, cf, (0, 0, w, h))]
            fps = 1.0 / (time.perf_counter() - t0)

            # annotate
            annotated = img.copy()
            for cid, cf, (x1, y1, x2, y2) in dets:
                cls_name = names_list[cid]
                color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                lbl_txt = f"{cls_name} {cf*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(lbl_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, max(0, y1-th-8)), (x1+tw+8, y1), color, -1)
                cv2.putText(annotated, lbl_txt, (x1+4, y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

            if dets:
                top = max(dets, key=lambda d: d[1])
                top_name = names_list[top[0]]
                top_conf = top[1]
            else:
                top_name, top_conf = "—", 0.0

            self._on_frame(annotated, top_name, top_conf, fps)
            counts = Counter()
            for cid, cf, _ in dets:
                counts[names_list[cid]] += 1
            self._on_stats(dict(counts))
            self._log(f"Найдено объектов: {len(dets)}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    # ── Slot: Stop ────────────────────────────────────────────────────────────

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_status("⏹  Остановлено")

    def _on_reset(self):
        if self._worker:
            self._worker.counts.clear()
            self._worker.total_frames = 0
        for row in self.count_rows.values():
            row.update_count(0, 1)
        for lbl in [self.lbl_total_frames, self.lbl_total_detected,
                    self.lbl_total_defects]:
            lbl.setText("0")
        self.lbl_defect_rate.setText("0%")
        self._log_rows.clear()
        self._log("Счётчики сброшены")

    def _on_save_log(self):
        if not self._log_rows:
            QMessageBox.information(self, "Лог", "Нет данных для сохранения")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить лог", str(LOG_DIR / "log.csv"), "CSV (*.csv)"
        )
        if path:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._log_rows[0].keys())
                writer.writeheader()
                writer.writerows(self._log_rows)
            self._log(f"Лог сохранён: {path}")

    # ── Worker slots ─────────────────────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray, cls_name: str, conf: float, fps: float):
        self._last_cls  = cls_name
        self._last_conf = conf
        self._last_fps  = fps

        self._display_frame(frame)
        self.badge.update_class(cls_name, conf)
        self.lbl_fps_big.setText(f"{fps:.0f} FPS")

        self._log_rows.append({
            "timestamp": datetime.utcnow().isoformat(),
            "class": cls_name,
            "confidence": round(conf, 4),
            "fps": round(fps, 1),
        })

    def _on_stats(self, counts: dict):
        total = sum(counts.values())
        defects = sum(v for k, v in counts.items() if k not in ("good_bottle", "normal"))
        for cls, row in self.count_rows.items():
            row.update_count(counts.get(cls, 0), max(total, 1))

        self.lbl_total_frames.setText(str(total))
        self.lbl_total_detected.setText(str(total))
        self.lbl_total_defects.setText(str(defects))
        rate = defects / total * 100 if total > 0 else 0
        self.lbl_defect_rate.setText(f"{rate:.1f}%")

    def _on_worker_error(self, msg: str):
        QMessageBox.critical(self, "Ошибка", msg)
        self._on_worker_done()

    def _on_worker_done(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_status("✅  Готово")
        self._log("Завершено")

    # ── Display ───────────────────────────────────────────────────────────────

    def _display_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        px = QPixmap.fromImage(qi)
        lbl_w = self.video_label.width()
        lbl_h = self.video_label.height()
        self.video_label.setPixmap(
            px.scaled(lbl_w, lbl_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.video_label.setStyleSheet(
            "background:#11111b; border:1px solid #313244; border-radius:8px;"
        )

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        event.accept()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Bottle Defect Classifier")
    app.setStyleSheet(QSS_DARK)

    # Dark palette fallback
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#1e1e2e"))
    palette.setColor(QPalette.WindowText, QColor("#cdd6f4"))
    palette.setColor(QPalette.Base, QColor("#11111b"))
    palette.setColor(QPalette.AlternateBase, QColor("#313244"))
    palette.setColor(QPalette.Button, QColor("#313244"))
    palette.setColor(QPalette.ButtonText, QColor("#cdd6f4"))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
