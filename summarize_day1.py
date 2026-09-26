from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from plot_style import apply_chinese_style


apply_chinese_style()


ROOT = Path(__file__).resolve().parent
SOURCES = [
    ("文本均值池化", "outputs/baselines_seed2026/summary.csv", "text"),
    ("文本双向GRU", "outputs/exp001_text_bigru/summary.csv", "text_bigru"),
    ("文本+语音拼接", "outputs/exp002_pair_bigru/summary.csv", "text_audio_bigru"),
    ("文本+视觉拼接", "outputs/exp002_pair_bigru/summary.csv", "text_vision_bigru"),
    ("文本+语音门控", "outputs/exp003_text_audio_gate/summary.csv", "text_audio_gated_bigru"),
    ("丢弃率0.5", "outputs/exp004_text_bigru_drop05/summary.csv", "text_bigru"),
    ("类别权重开方", "outputs/exp007_class_weight_sqrt/summary.csv", "text_bigru"),
    ("均方误差损失", "outputs/exp008_regression_mse/summary.csv", "text_bigru"),
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
        ("Macro-F1", True, "Macro-F1（越高越好）"),
        ("Pearson", True, "Pearson（越高越好）"),
        ("MAE", False, "MAE（越低越好）"),
    ]
    colors = ["#4C78A8" if label != "文本双向GRU" else "#E45756" for label in result["方案"]]
    for ax, (metric, higher, title) in zip(axes, settings):
        bars = ax.barh(result["方案"], result[metric], color=colors)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        for bar, value in zip(bars, result[metric]):
            ax.text(value + 0.004, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
        if not higher:
            ax.text(0.98, 0.02, "数值越低越好", transform=ax.transAxes, ha="right", fontsize=8)
    fig.suptitle("第1天验证集方案对比（随机种子2026）", fontsize=14)
    fig.legend(
        handles=[Patch(color="#E45756", label="文本双向GRU"), Patch(color="#4C78A8", label="其他方案")],
        loc="lower center",
        ncol=2,
        title="颜色说明",
    )
    fig.savefig(output_dir / "validation_comparison.png", dpi=180)
    plt.close(fig)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
