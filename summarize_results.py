from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a compact baseline comparison.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/baselines_seed2026/summary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.summary.resolve()
    frame = pd.read_csv(summary_path)
    test = frame.loc[frame["split"] == "test"].copy()
    order = ["text", "audio", "vision", "fusion"]
    test["model"] = pd.Categorical(test["model"], categories=order, ordered=True)
    test = test.sort_values("model")

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    metrics = [
        ("accuracy", "Accuracy", False),
        ("macro_f1", "Macro-F1", False),
        ("pearson", "Pearson correlation", False),
        ("mae", "MAE (lower is better)", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (column, title, lower_is_better) in zip(axes.flat, metrics):
        values = test[column].to_numpy()
        bars = ax.bar(test["model"].astype(str), values, color=colors)
        ax.set_title(title)
        ax.set_ylim(0, max(1.0, float(values.max()) * 1.18))
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        if lower_is_better:
            ax.text(0.98, 0.92, "lower is better", transform=ax.transAxes, ha="right", fontsize=9)

    fig.suptitle("Four minimal baselines on the official test split", fontsize=15)
    output_path = summary_path.parent / "baseline_comparison.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
