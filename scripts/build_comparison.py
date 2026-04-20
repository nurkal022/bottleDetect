"""
Генерирует сравнительные графики и таблицы из:
- reports/figures/classification/*/results.csv   (7 cls моделей)
- runs_det/yolov8n_det/results.csv               (detection, финальная)

Выход: reports/figures/comparison/ + reports/metrics_all.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLS_DIR = ROOT / "reports" / "figures" / "classification"
DET_DIR = ROOT / "runs_det" / "yolov8n_det"
OUT_DIR = ROOT / "reports" / "figures" / "comparison"
REPORTS_DIR = ROOT / "reports"

# Из ранее собранных benchmark результатов на сервере (CSV сохранены в report.md)
CLS_PERF = {
    # model: {params_M, weights_MB, cpu_fps, gpu_fps, top1_acc}
    "yolov8n-cls": {"params_M": 1.44, "weights_MB": 2.97,  "cpu_fps": 320.7, "gpu_fps": 885.1, "top1_acc": 1.00},
    "yolov8s-cls": {"params_M": 5.09, "weights_MB": 10.26, "cpu_fps": 161.7, "gpu_fps": 878.5, "top1_acc": 1.00},
    "yolov8m-cls": {"params_M": 15.78,"weights_MB": 31.69, "cpu_fps":  65.4, "gpu_fps": 575.3, "top1_acc": 1.00},
    "yolov8l-cls": {"params_M": 36.21,"weights_MB": 72.59, "cpu_fps":  32.0, "gpu_fps": 377.9, "top1_acc": 1.00},
    "yolo11n-cls": {"params_M": 1.54, "weights_MB": 3.19,  "cpu_fps": 252.9, "gpu_fps": 683.8, "top1_acc": 1.00},
    "yolo11s-cls": {"params_M": 5.45, "weights_MB": 11.03, "cpu_fps": 131.0, "gpu_fps": 669.7, "top1_acc": 1.00},
    "yolo11m-cls": {"params_M": 10.36,"weights_MB": 20.88, "cpu_fps":  62.3, "gpu_fps": 566.9, "top1_acc": 1.00},
}

COLORS = {
    "yolov8n-cls": "#4C72B0", "yolov8s-cls": "#5781b8", "yolov8m-cls": "#6794c2", "yolov8l-cls": "#7cadd1",
    "yolo11n-cls": "#55A868", "yolo11s-cls": "#6cbc7c", "yolo11m-cls": "#8fd6a1",
}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame.from_dict(CLS_PERF, orient="index").reset_index()
    df.rename(columns={"index": "model"}, inplace=True)
    df["family"] = df["model"].apply(lambda m: "YOLOv8" if m.startswith("yolov8") else "YOLOv11")
    df.to_csv(REPORTS_DIR / "classification_benchmarks.csv", index=False)

    # ── Plot 1: Accuracy vs FPS (CPU) ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for _, r in df.iterrows():
        c = COLORS.get(r["model"], "gray")
        ax.scatter(r["cpu_fps"], r["top1_acc"]*100, s=160, color=c,
                   label=r["model"], edgecolor="black", zorder=3)
        ax.annotate(r["model"], (r["cpu_fps"], r["top1_acc"]*100),
                    fontsize=8, xytext=(8, 6), textcoords="offset points")
    ax.set_xlabel("CPU FPS (изображений/сек)", fontsize=11)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax.set_title("Classification: Точность vs Скорость (CPU)", fontsize=13)
    ax.set_ylim(98, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cls_acc_vs_fps_cpu.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Params vs FPS ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for _, r in df.iterrows():
        c = COLORS.get(r["model"], "gray")
        ax.scatter(r["params_M"], r["cpu_fps"], s=160, color=c,
                   label=r["model"], edgecolor="black", zorder=3)
        ax.annotate(r["model"], (r["params_M"], r["cpu_fps"]),
                    fontsize=8, xytext=(8, 6), textcoords="offset points")
    ax.set_xlabel("Параметры модели (M)", fontsize=11)
    ax.set_ylabel("CPU FPS", fontsize=11)
    ax.set_title("Classification: Параметры vs Скорость", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cls_params_vs_fps.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Weights size bar ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sorted_df = df.sort_values("weights_MB")
    colors = [COLORS.get(m, "gray") for m in sorted_df["model"]]
    bars = ax.bar(sorted_df["model"], sorted_df["weights_MB"], color=colors, edgecolor="black")
    ax.bar_label(bars, fmt="%.1f MB", padding=3, fontsize=9)
    ax.set_ylabel("Размер весов (MB)", fontsize=11)
    ax.set_title("Compactness: размер best.pt", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cls_size_bar.png", dpi=150)
    plt.close(fig)

    # ── Plot 4: CPU vs GPU FPS для всех ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    ax.bar([i - 0.2 for i in x], df["cpu_fps"], 0.4, label="CPU", color="#4C72B0")
    ax.bar([i + 0.2 for i in x], df["gpu_fps"], 0.4, label="GPU (RTX 5080)", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"], rotation=30, ha="right")
    ax.set_ylabel("FPS", fontsize=11)
    ax.set_title("Classification: CPU vs GPU скорость", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cls_cpu_vs_gpu.png", dpi=150)
    plt.close(fig)

    # ── Plot 5: Кривые обучения detection ─────────────────────────────
    det_csv = DET_DIR / "results.csv"
    if det_csv.exists():
        ddf = pd.read_csv(det_csv)
        ddf.columns = [c.strip() for c in ddf.columns]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        # Loss
        axes[0].plot(ddf["epoch"], ddf["train/box_loss"], label="box_loss (train)", color="#C44E52")
        axes[0].plot(ddf["epoch"], ddf["train/cls_loss"], label="cls_loss (train)", color="#4C72B0")
        axes[0].plot(ddf["epoch"], ddf["val/box_loss"], "--", label="box_loss (val)", color="#C44E52", alpha=0.6)
        axes[0].plot(ddf["epoch"], ddf["val/cls_loss"], "--", label="cls_loss (val)", color="#4C72B0", alpha=0.6)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("Detection: Кривые потерь"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
        # mAP
        axes[1].plot(ddf["epoch"], ddf["metrics/mAP50(B)"], label="mAP@0.5", color="#55A868", linewidth=2)
        axes[1].plot(ddf["epoch"], ddf["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", color="#DD8452", linewidth=2)
        axes[1].plot(ddf["epoch"], ddf["metrics/precision(B)"], label="Precision", color="#4C72B0", alpha=0.7)
        axes[1].plot(ddf["epoch"], ddf["metrics/recall(B)"], label="Recall", color="#C44E52", alpha=0.7)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Metric")
        axes[1].set_title("Detection: Метрики качества по эпохам"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
        axes[1].set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "det_training_curves.png", dpi=150)
        plt.close(fig)

    # ── Plot 6: Per-class detection mAP ───────────────────────────────
    per_class_csv = REPORTS_DIR.parent / "results" / "per_class_metrics.csv"
    if per_class_csv.exists():
        pc = pd.read_csv(per_class_csv)
        pc = pc[pc["class_name"] != "ALL"]
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(pc))
        ax.bar([i - 0.2 for i in x], pc["mAP50"], 0.4, label="mAP@0.5", color="#55A868")
        ax.bar([i + 0.2 for i in x], pc["mAP50-95"], 0.4, label="mAP@0.5:0.95", color="#DD8452")
        ax.set_xticks(list(x))
        ax.set_xticklabels(pc["class_name"], rotation=25, ha="right")
        ax.set_ylabel("mAP")
        ax.set_title("Detection: Per-class mAP (yolov8n_det)", fontsize=13)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "det_per_class_map.png", dpi=150)
        plt.close(fig)

    print("OK — сохранены графики:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
