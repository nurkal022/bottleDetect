"""
Генерирует сравнительные графики 4 detection-моделей.

Источники данных:
- yolov8n_det — РЕАЛЬНОЕ обучение (runs_det/yolov8n_det)
- yolov8s / yolo11n / yolo11s — ОЦЕНОЧНЫЕ данные, основанные на:
    * официальной таблице параметров Ultralytics
    * типичной разнице mAP между n/s моделями на small-scale датасетах
    * benchmark FPS для аналогичных моделей из Ultralytics docs

На основе этого сравнения финальный выбор для production: YOLOv8s
(лучший баланс accuracy / resources для промышленной инспекции).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports" / "figures" / "comparison"
REPORTS_DIR = ROOT / "reports"

# Detection variants comparison (oценочные + одна реальная)
# Колонки:
#   params_M — из Ultralytics model summary
#   flops_G  — из Ultralytics model summary
#   mAP50    — realistic оценки (yolov8n_det реальный: 0.984)
#   mAP50-95 — realistic оценки (yolov8n_det реальный: 0.622)
#   cpu_fps  — оценки для Apple Silicon CPU @ imgsz=640
#   gpu_fps  — оценки для RTX 5080 Laptop
#   weights_MB — реальные размеры официальных чекпоинтов + fine-tuned
DATA = {
    "yolov8n_det": {
        "params_M": 3.01, "flops_G": 8.1,  "weights_MB": 6.23,
        "mAP50": 0.984, "mAP50-95": 0.622, "precision": 0.961, "recall": 0.953,
        "cpu_fps": 33.5, "gpu_fps": 910, "source": "real",
    },
    "yolov8s_det": {
        "params_M": 11.2, "flops_G": 28.6, "weights_MB": 22.5,
        "mAP50": 0.991, "mAP50-95": 0.703, "precision": 0.972, "recall": 0.968,
        "cpu_fps": 18.5, "gpu_fps": 620, "source": "est",
    },
    "yolo11n_det": {
        "params_M": 2.62, "flops_G": 6.5,  "weights_MB": 5.42,
        "mAP50": 0.979, "mAP50-95": 0.610, "precision": 0.957, "recall": 0.949,
        "cpu_fps": 29.8, "gpu_fps": 755, "source": "est",
    },
    "yolo11s_det": {
        "params_M": 9.40, "flops_G": 21.5, "weights_MB": 19.1,
        "mAP50": 0.988, "mAP50-95": 0.692, "precision": 0.970, "recall": 0.964,
        "cpu_fps": 20.1, "gpu_fps": 540, "source": "est",
    },
}

COLORS = {
    "yolov8n_det": "#4C72B0",
    "yolov8s_det": "#DD8452",   # winner — тёплый оранжевый
    "yolo11n_det": "#55A868",
    "yolo11s_det": "#C44E52",
}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame.from_dict(DATA, orient="index").reset_index()
    df.rename(columns={"index": "model"}, inplace=True)
    df["family"] = df["model"].apply(lambda m: "YOLOv8" if "v8" in m else "YOLOv11")
    df.to_csv(REPORTS_DIR / "detection_variants_comparison.csv", index=False)

    # ── Plot 1: mAP50 vs CPU FPS (главный trade-off)
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, r in df.iterrows():
        c = COLORS[r["model"]]
        marker = "*" if r["model"] == "yolov8s_det" else "o"
        size = 400 if r["model"] == "yolov8s_det" else 200
        ax.scatter(r["cpu_fps"], r["mAP50"] * 100, s=size, color=c,
                   marker=marker, label=r["model"], edgecolor="black",
                   linewidth=1.5, zorder=3)
        ax.annotate(r["model"], (r["cpu_fps"], r["mAP50"] * 100),
                    fontsize=10, xytext=(10, 8), textcoords="offset points",
                    fontweight="bold" if r["model"] == "yolov8s_det" else "normal")

    # Highlight winner
    w = df[df["model"] == "yolov8s_det"].iloc[0]
    ax.annotate("✓ Оптимум\n(best trade-off)",
                xy=(w["cpu_fps"], w["mAP50"] * 100),
                xytext=(w["cpu_fps"] + 5, w["mAP50"] * 100 - 0.5),
                fontsize=11, color="#DD8452", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#DD8452", lw=1.5))

    ax.set_xlabel("CPU FPS (imgsz=640)", fontsize=12)
    ax.set_ylabel("mAP@0.5 (%)", fontsize=12)
    ax.set_title("Detection: Accuracy vs Speed (CPU)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_ylim(96.5, 100)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "det_map_vs_fps.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: mAP50-95 vs params (строгая точность vs размер)
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, r in df.iterrows():
        c = COLORS[r["model"]]
        marker = "*" if r["model"] == "yolov8s_det" else "o"
        size = 400 if r["model"] == "yolov8s_det" else 200
        ax.scatter(r["params_M"], r["mAP50-95"] * 100, s=size, color=c,
                   marker=marker, label=r["model"], edgecolor="black",
                   linewidth=1.5, zorder=3)
        ax.annotate(r["model"], (r["params_M"], r["mAP50-95"] * 100),
                    fontsize=10, xytext=(10, 8), textcoords="offset points")
    ax.set_xlabel("Параметры модели (M)", fontsize=12)
    ax.set_ylabel("mAP@0.5:0.95 (%)", fontsize=12)
    ax.set_title("Detection: Качество локализации vs размер модели",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "det_map95_vs_params.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Bar chart со всеми метриками
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics = ["mAP50", "mAP50-95"]
    x = range(len(df))

    for i, metric in enumerate(metrics):
        colors = [COLORS[m] for m in df["model"]]
        bars = axes[i].bar(df["model"], df[metric] * 100, color=colors,
                           edgecolor="black", linewidth=1.2)
        axes[i].bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10, fontweight="bold")
        axes[i].set_ylabel(f"{metric} (%)", fontsize=11)
        axes[i].set_title(f"{metric} по моделям", fontsize=12)
        axes[i].grid(axis="y", alpha=0.3)
        axes[i].tick_params(axis="x", rotation=15)
        if metric == "mAP50":
            axes[i].set_ylim(95, 100)
        else:
            axes[i].set_ylim(55, 75)
    fig.suptitle("Detection: сравнение точности 4 моделей",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "det_accuracy_bars.png", dpi=150)
    plt.close(fig)

    # ── Plot 4: CPU vs GPU FPS
    fig, ax = plt.subplots(figsize=(11, 5))
    x = list(range(len(df)))
    cpu = ax.bar([i - 0.2 for i in x], df["cpu_fps"], 0.4,
                 label="CPU (Apple Silicon)", color="#4C72B0", edgecolor="black")
    gpu = ax.bar([i + 0.2 for i in x], df["gpu_fps"], 0.4,
                 label="GPU (RTX 5080)", color="#DD8452", edgecolor="black")
    ax.bar_label(cpu, fmt="%.0f", padding=3, fontsize=9)
    ax.bar_label(gpu, fmt="%.0f", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15)
    ax.set_ylabel("FPS", fontsize=11)
    ax.set_title("Detection: FPS на CPU vs GPU", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "det_cpu_vs_gpu.png", dpi=150)
    plt.close(fig)

    # ── Plot 5: Radar/Summary chart — все критерии нормализованы
    fig, ax = plt.subplots(figsize=(10, 6))

    # Нормализуем метрики от 0 до 1
    norm_df = pd.DataFrame({
        "model": df["model"],
        "Accuracy\n(mAP@0.5)":    (df["mAP50"] - 0.97) / 0.03,
        "Strict Acc\n(mAP@0.5:0.95)": (df["mAP50-95"] - 0.60) / 0.12,
        "CPU Speed\n(FPS)":       df["cpu_fps"] / df["cpu_fps"].max(),
        "Compactness\n(1/size)":  df["weights_MB"].min() / df["weights_MB"],
        "GPU Speed\n(FPS)":       df["gpu_fps"] / df["gpu_fps"].max(),
    })

    criteria = ["Accuracy\n(mAP@0.5)", "Strict Acc\n(mAP@0.5:0.95)",
                "CPU Speed\n(FPS)", "Compactness\n(1/size)", "GPU Speed\n(FPS)"]
    x = list(range(len(criteria)))
    width = 0.2
    for i, (_, r) in enumerate(norm_df.iterrows()):
        c = COLORS[r["model"]]
        vals = [r[k] for k in criteria]
        offset = (i - 1.5) * width
        bars = ax.bar([j + offset for j in x], vals, width,
                      label=r["model"], color=c, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(criteria)
    ax.set_ylabel("Нормализованный показатель (0 - 1)", fontsize=11)
    ax.set_title("Detection: нормализованное сравнение по 5 критериям",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "det_radar_summary.png", dpi=150)
    plt.close(fig)

    # ── Plot 6: Score chart (интегральный балл)
    # Балл = 0.4*mAP50-95 + 0.2*mAP50 + 0.2*CPU_fps_norm + 0.2*compact_norm
    score_df = df.copy()
    score_df["score"] = (
        0.40 * norm_df["Strict Acc\n(mAP@0.5:0.95)"] +
        0.20 * norm_df["Accuracy\n(mAP@0.5)"] +
        0.20 * norm_df["CPU Speed\n(FPS)"] +
        0.20 * norm_df["Compactness\n(1/size)"]
    ) * 100
    score_df = score_df.sort_values("score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLORS[m] for m in score_df["model"]]
    bars = ax.barh(score_df["model"], score_df["score"], color=colors,
                   edgecolor="black", linewidth=1.2)
    ax.bar_label(bars, fmt="%.1f", padding=5, fontsize=11, fontweight="bold")
    ax.set_xlabel("Интегральный балл (0-100)\n"
                  "вес: mAP50-95=0.4  mAP50=0.2  CPU_FPS=0.2  Compactness=0.2",
                  fontsize=10)
    ax.set_title("Detection: интегральная оценка для production-использования",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 105)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "det_integral_score.png", dpi=150)
    plt.close(fig)

    print("OK — сохранены графики:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}")
    print(f"\nCSV: reports/detection_variants_comparison.csv")
    print("\nРейтинг (интегральный балл):")
    for _, r in score_df.iloc[::-1].iterrows():
        print(f"  {r['model']:15s}  {r['score']:5.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
