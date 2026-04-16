"""
Объединяет results/metrics.csv + results/speed.csv.
Строит графики для ВКР.
Выходы:
  results/comparison.csv
  results/comparison.md
  results/plots/acc_vs_fps.png
  results/plots/acc_vs_params.png
  results/plots/acc_vs_size.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Порядок моделей для читаемых графиков
MODEL_ORDER = ["yolov8n-cls", "yolov8s-cls", "yolo11n-cls", "yolo11s-cls"]
COLORS = {"yolov8n-cls": "#4C72B0", "yolov8s-cls": "#DD8452",
          "yolo11n-cls": "#55A868", "yolo11s-cls": "#C44E52"}


def main() -> int:
    metrics_f = RESULTS_DIR / "metrics.csv"
    speed_f = RESULTS_DIR / "speed.csv"
    if not metrics_f.exists() or not speed_f.exists():
        print("ERROR: запусти сначала evaluate.py и benchmark_fps.py")
        return 1

    metrics = pd.read_csv(metrics_f)
    speed = pd.read_csv(speed_f)
    merged = speed.merge(metrics, on="model", how="left")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(RESULTS_DIR / "comparison.csv", index=False)

    # Markdown-таблица
    md_lines = ["# Сравнение YOLO-cls моделей\n\n"]
    md_lines.append(merged.to_markdown(index=False))
    md_lines.append("\n")
    (RESULTS_DIR / "comparison.md").write_text("\n".join(md_lines))

    # Берём только CPU (или первое устройство)
    device_val = merged["device"].iloc[0]
    df = merged[merged["device"] == device_val].copy()

    def annotate(ax, df, x_col, y_col):
        for _, r in df.iterrows():
            ax.annotate(r["model"], (r[x_col], r[y_col]),
                        fontsize=8, xytext=(6, 4), textcoords="offset points")

    # Plot 1: top-1 accuracy vs FPS
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, r in df.iterrows():
        c = COLORS.get(r["model"], "gray")
        ax.scatter(r["fps"], r["top1_acc"], s=120, color=c, label=r["model"], zorder=3)
    annotate(ax, df, "fps", "top1_acc")
    ax.set_xlabel("FPS (изображений/сек)")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title(f"Точность vs Скорость ({device_val})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "acc_vs_fps.png", dpi=150)
    plt.close(fig)

    # Plot 2: top-1 accuracy vs params
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for _, r in df.iterrows():
        c = COLORS.get(r["model"], "gray")
        ax2.scatter(r["params_M"], r["top1_acc"], s=120, color=c, label=r["model"], zorder=3)
    annotate(ax2, df, "params_M", "top1_acc")
    ax2.set_xlabel("Параметры (M)")
    ax2.set_ylabel("Top-1 Accuracy")
    ax2.set_title("Точность vs Размер модели")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "acc_vs_params.png", dpi=150)
    plt.close(fig2)

    # Plot 3: bar chart — top-1 per model
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    bars = ax3.bar(df["model"], df["top1_acc"] * 100,
                   color=[COLORS.get(m, "gray") for m in df["model"]])
    ax3.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax3.set_ylabel("Top-1 Accuracy, %")
    ax3.set_title("Точность классификации по моделям")
    ax3.set_ylim(0, 110)
    ax3.grid(axis="y", alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(PLOTS_DIR / "acc_bar.png", dpi=150)
    plt.close(fig3)

    print("OK: results/comparison.csv, comparison.md")
    print(f"    plots: acc_vs_fps.png, acc_vs_params.png, acc_bar.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
