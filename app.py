"""
Gradio-интерфейс для классификации дефектов бутылок.
Поддерживает режимы: фото, видео, веб-камера.

Запуск:
    python app.py
    python app.py --share          # публичная ссылка
    python app.py --port 7860
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs" / "classify"

# Цвета для классов
CLASS_COLORS = {
    "good_bottle":    "#32C832",
    "cork_missmatch": "#FF6432",
    "label_missing":  "#FFD700",
    "shape_mismatch": "#DC1414",
}
DEFAULT_COLOR = "#888888"


# ─── Загрузка доступных моделей ──────────────────────────────────────────────

def get_available_models() -> dict[str, Path]:
    models = {}
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir()):
            w = d / "weights" / "best.pt"
            if w.exists():
                models[d.name] = w
    return models


# ─── Инференс ─────────────────────────────────────────────────────────────────

def predict_image(model: YOLO, img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    res = model.predict(img_bgr, imgsz=224, verbose=False)[0]
    cls_id = int(res.probs.top1)
    conf = float(res.probs.top1conf)
    cls_name = model.names[cls_id]

    # Аннотируем кадр
    color_hex = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
    r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
    color = (b, g, r)  # BGR для OpenCV

    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, 5)
    label = f"{cls_name}  {conf * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    cv2.rectangle(out, (0, 0), (tw + 20, th + 20), color, -1)
    cv2.putText(out, label, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA)

    probs = {model.names[i]: round(float(p), 4)
             for i, p in enumerate(res.probs.data.tolist())}
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), probs


# ─── Хэндлеры вкладок ─────────────────────────────────────────────────────────

def run_image(model_name: str, image: np.ndarray):
    models = get_available_models()
    if model_name not in models:
        return None, {"error": "модель не найдена"}
    if image is None:
        return None, {"error": "загрузите изображение"}
    model = YOLO(str(models[model_name]))
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated, probs = predict_image(model, img_bgr)
    top = max(probs, key=probs.get)
    label = f"**{top}** — {probs[top]*100:.1f}%"
    return annotated, label, probs


def run_video(model_name: str, video_path: str):
    models = get_available_models()
    if model_name not in models or not video_path:
        return None
    model = YOLO(str(models[model_name]))

    cap = cv2.VideoCapture(video_path)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = str(ROOT / "results" / "demo_output.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated_rgb, _ = predict_image(model, frame)
        writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))

    cap.release()
    writer.release()
    return out_path


def run_webcam(model_name: str, frame: np.ndarray):
    """Обрабатывает один кадр с веб-камеры (Gradio streaming)."""
    models = get_available_models()
    if model_name not in models or frame is None:
        return frame
    model = YOLO(str(models[model_name]))
    img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated, _ = predict_image(model, img_bgr)
    return annotated


# ─── Сборка интерфейса ─────────────────────────────────────────────────────────

def build_ui():
    model_list = list(get_available_models().keys())
    if not model_list:
        model_list = ["модели не найдены — запусти train_all.py"]

    css = """
    .class-good    { color: #32C832; font-weight: bold; }
    .class-cork    { color: #FF6432; font-weight: bold; }
    .class-label   { color: #FFD700; font-weight: bold; }
    .class-shape   { color: #DC1414; font-weight: bold; }
    .title { text-align: center; font-size: 1.6em; font-weight: bold; margin-bottom: 4px; }
    .subtitle { text-align: center; color: #888; margin-bottom: 16px; }
    """

    with gr.Blocks(title="Bottle Defect Classifier", theme=gr.themes.Soft(), css=css) as demo:
        gr.HTML("<div class='title'>Bottle Defect Classifier</div>")
        gr.HTML("<div class='subtitle'>Лёгкая система компьютерного зрения для контроля дефектов на конвейере</div>")

        with gr.Row():
            model_dd = gr.Dropdown(
                choices=model_list,
                value=model_list[0] if model_list else None,
                label="Модель",
                scale=1,
            )
            gr.HTML("""
            <div style='padding:8px; font-size:0.85em; color:#666; line-height:1.6'>
              <b>Классы:</b><br>
              🟢 good_bottle — без дефекта<br>
              🟠 cork_missmatch — крышка<br>
              🟡 label_missing — этикетка<br>
              🔴 shape_mismatch — деформация
            </div>""", scale=1)

        with gr.Tabs():
            # ── Вкладка: Фото ──────────────────────────────────────────────
            with gr.Tab("📷 Фото"):
                with gr.Row():
                    img_in  = gr.Image(label="Загрузить изображение", type="numpy", scale=1)
                    img_out = gr.Image(label="Результат", scale=1)
                with gr.Row():
                    img_label = gr.Markdown("Загрузите изображение и нажмите Classify")
                    img_probs = gr.Label(label="Вероятности классов", num_top_classes=4)
                btn_img = gr.Button("Classify", variant="primary")
                btn_img.click(
                    fn=run_image,
                    inputs=[model_dd, img_in],
                    outputs=[img_out, img_label, img_probs],
                )
                gr.Examples(
                    examples=_collect_examples(),
                    inputs=img_in,
                    label="Примеры из test set",
                )

            # ── Вкладка: Видео ─────────────────────────────────────────────
            with gr.Tab("🎬 Видео"):
                with gr.Row():
                    vid_in  = gr.Video(label="Загрузить видео", scale=1)
                    vid_out = gr.Video(label="Результат с разметкой", scale=1)
                btn_vid = gr.Button("Обработать видео", variant="primary")
                btn_vid.click(
                    fn=run_video,
                    inputs=[model_dd, vid_in],
                    outputs=[vid_out],
                )
                gr.Markdown("""
> Видео обрабатывается покадрово — каждый кадр классифицируется.
> Результат сохраняется в `results/demo_output.mp4`.
                """)

            # ── Вкладка: Веб-камера ────────────────────────────────────────
            with gr.Tab("📹 Live (веб-камера)"):
                gr.Markdown("Нажми **Start** для запуска live-классификации с камеры.")
                webcam = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="Камера",
                    type="numpy",
                )
                webcam_out = gr.Image(label="Классифицированный кадр", type="numpy")
                webcam.stream(
                    fn=run_webcam,
                    inputs=[model_dd, webcam],
                    outputs=[webcam_out],
                )

            # ── Вкладка: Сравнение моделей ────────────────────────────────
            with gr.Tab("📊 Сравнение моделей"):
                _build_comparison_tab()

    return demo


def _collect_examples() -> list:
    test_dir = ROOT / "data" / "raw" / "test"
    examples = []
    if test_dir.exists():
        for cls_dir in sorted(test_dir.iterdir()):
            if cls_dir.is_dir():
                imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
                if imgs:
                    examples.append([str(imgs[0])])
    return examples[:8]


def _build_comparison_tab():
    comp_md = ROOT / "results" / "comparison.md"
    plots_dir = ROOT / "results" / "plots"

    if comp_md.exists():
        gr.Markdown(comp_md.read_text())
    else:
        gr.Markdown("_Запусти `python scripts/compare_models.py` для генерации таблицы._")

    plot_files = sorted(plots_dir.glob("*.png")) if plots_dir.exists() else []
    if plot_files:
        with gr.Row():
            for pf in plot_files[:3]:
                gr.Image(str(pf), label=pf.stem, show_download_button=True)


# ─── Точка входа ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server", default="127.0.0.1")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
