from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCES = [
    ("Text-Mean", "outputs/baselines_seed2026/summary.csv", "text"),
    ("Text-BiGRU", "outputs/exp001_text_bigru/summary.csv", "text_bigru"),
    ("+Audio concat", "outputs/exp002_pair_bigru/summary.csv", "text_audio_bigru"),
    ("+Vision concat", "outputs/exp002_pair_bigru/summary.csv", "text_vision_bigru"),
    ("+Audio gate", "outputs/exp003_text_audio_gate/summary.csv", "text_audio_gated_bigru"),
    ("Dropout 0.5", "outputs/exp004_text_bigru_drop05/summary.csv", "text_bigru"),
    ("Sqrt class weight", "outputs/exp007_class_weight_sqrt/summary.csv", "text_bigru"),
    ("MSE loss", "outputs/exp008_regression_mse/summary.csv", "text_bigru"),
]


def main() -> None:
    rows = []
    for label, relative_path, model in SOURCES:
        frame = pd.read_csv(ROOT / relative_path)
        row = frame.loc[(frame["model"] == model) & (frame["split"] == "valid")].iloc[0]
        rows.append(
            {
                "方案": label,
                "Accuracy": row["accuracy"],
                "Macro-F1": row["macro_f1"],
                "MAE": row["mae"],
                "Pearson": row["pearson"],
                "Selection score": row["selection_score"],
            }
        )
    result = pd.DataFrame(rows)
    output_dir = ROOT / "outputs/day1_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "validation_comparison.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    settings = [
        ("Macro-F1", True, "Macro-F1 (higher is better)"),
        ("Pearson", True, "Pearson (higher is better)"),
        ("MAE", False, "MAE (lower is better)"),
    ]
    colors = ["#4C78A8" if label != "Text-BiGRU" else "#E45756" for label in result["方案"]]
    for ax, (metric, higher, title) in zip(axes, settings):
        bars = ax.barh(result["方案"], result[metric], color=colors)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        for bar, value in zip(bars, result[metric]):
            ax.text(value + 0.004, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
        if not higher:
            ax.text(0.98, 0.02, "lower is better", transform=ax.transAxes, ha="right", fontsize=8)
    fig.suptitle("Day-1 validation comparison (seed=2026)", fontsize=14)
    fig.savefig(output_dir / "validation_comparison.png", dpi=180)
    plt.close(fig)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
