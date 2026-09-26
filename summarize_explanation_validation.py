from __future__ import annotations

"""汇总EXP-023多尺度解释验证，生成统计检验和中文图。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from plot_style import apply_chinese_style


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS = {
    2: PROJECT_ROOT / "outputs" / "exp023_explanation_validation_block2",
    5: PROJECT_ROOT / "outputs" / "exp023_explanation_validation",
    10: PROJECT_ROOT / "outputs" / "exp023_explanation_validation_block10",
}


def configure_chinese_font() -> None:
    apply_chinese_style()
    for path in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break


def bootstrap_mean_ci(values: np.ndarray, seed: int = 2026) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(5000, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    output_dir = PROJECT_ROOT / "reports" / "question3_stage_b"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    tables = {}
    summaries = {}
    for block_size, run_dir in RUNS.items():
        table = pd.read_csv(run_dir / "sample_explanations.csv")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        tables[block_size] = table
        summaries[block_size] = summary
        class_difference = (
            table["classification_top_deletion_delta"]
            - table["classification_random_deletion_delta"]
        ).to_numpy()
        reg_difference = (
            table["regression_top_deletion_abs_delta"]
            - table["regression_random_deletion_abs_delta"]
        ).to_numpy()
        class_ci = bootstrap_mean_ci(class_difference)
        reg_ci = bootstrap_mean_ci(reg_difference)
        comparison_rows.append(
            {
                "block_size": block_size,
                "classification_top_minus_random_deletion": float(class_difference.mean()),
                "classification_difference_ci95_low": class_ci[0],
                "classification_difference_ci95_high": class_ci[1],
                "classification_wilcoxon_p": float(
                    wilcoxon(class_difference, alternative="greater").pvalue
                ),
                "classification_top_minus_random_retention": summary[
                    "classification_fidelity"
                ]["top_minus_random_retention"],
                "classification_top1_seed_agreement": summary["seed_stability"][
                    "classification_top1_pairwise_agreement"
                ],
                "regression_top_minus_random_deletion": float(reg_difference.mean()),
                "regression_difference_ci95_low": reg_ci[0],
                "regression_difference_ci95_high": reg_ci[1],
                "regression_wilcoxon_p": float(
                    wilcoxon(reg_difference, alternative="greater").pvalue
                ),
                "regression_top1_seed_agreement": summary["seed_stability"][
                    "regression_top1_pairwise_agreement"
                ],
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values("block_size")
    comparison.to_csv(output_dir / "block_size_comparison.csv", index=False, encoding="utf-8-sig")

    selected = tables[5]
    class_primary = selected["classification_top_modality"].value_counts().to_dict()
    reg_primary = tables[10]["regression_top_modality"].value_counts().to_dict()
    modality_stats = {}
    for modality in ("text", "audio", "vision"):
        values = selected[f"classification_{modality}_ablation_delta"]
        modality_stats[f"classification_{modality}"] = {
            "signed_mean": float(values.mean()),
            "absolute_mean": float(values.abs().mean()),
            "positive_ratio": float((values > 0).mean()),
        }
    for modality in ("text", "audio"):
        values = tables[10][f"regression_{modality}_ablation_signed_delta"]
        modality_stats[f"regression_{modality}"] = {
            "signed_mean": float(values.mean()),
            "absolute_mean": float(values.abs().mean()),
            "positive_ratio": float((values > 0).mean()),
        }

    configure_chinese_font()
    x = np.arange(len(comparison))
    widths = comparison["block_size"].astype(str).tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    axes[0].bar(
        x - 0.18,
        [summaries[size]["classification_fidelity"]["top_deletion_probability_drop_mean"] for size in comparison.block_size],
        width=0.36,
        label="最高贡献证据删除",
    )
    axes[0].bar(
        x + 0.18,
        [summaries[size]["classification_fidelity"]["random_deletion_probability_drop_mean"] for size in comparison.block_size],
        width=0.36,
        label="随机删除",
    )
    axes[0].set_xticks(x, widths)
    axes[0].set_xlabel("每个连续块的位置数")
    axes[0].set_ylabel("分类目标概率下降")
    axes[0].set_title("删除忠实性")
    axes[0].legend()

    axes[1].plot(
        widths,
        comparison["classification_top1_seed_agreement"],
        "o-",
        label="分类",
    )
    axes[1].plot(
        widths,
        comparison["regression_top1_seed_agreement"],
        "s--",
        label="回归",
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("每个连续块的位置数")
    axes[1].set_ylabel("三种子首要证据两两一致率")
    axes[1].set_title("解释稳定性")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    categories = ["文本", "音频", "视觉"]
    axes[2].bar(
        categories,
        [class_primary.get("text", 0), class_primary.get("audio", 0), class_primary.get("vision", 0)],
        color=["#2563EB", "#D97706", "#16A34A"],
    )
    axes[2].set_ylabel("验证样本数")
    axes[2].set_title("分类首要局部证据模态（5位置块）")
    for index, value in enumerate(
        [class_primary.get("text", 0), class_primary.get("audio", 0), class_primary.get("vision", 0)]
    ):
        axes[2].text(index, value + 8, str(value), ha="center")
    fig.savefig(output_dir / "解释忠实性与稳定性.png", dpi=200)
    plt.close(fig)

    final_summary = {
        "validation_sample_count": len(selected),
        "selected_protocol": {
            "classification_block_size": 5,
            "classification_top_k": 2,
            "regression_block_size": 10,
            "regression_top_k": 1,
            "reason": "分类兼顾局部精度与种子稳定性；回归采用更粗粒度以提高忠实性和稳定性。",
        },
        "classification_primary_modality_counts": {
            str(key): int(value) for key, value in class_primary.items()
        },
        "regression_primary_modality_counts": {
            str(key): int(value) for key, value in reg_primary.items()
        },
        "modality_ablation_statistics": modality_stats,
        "classification_selected_metrics": summaries[5]["classification_fidelity"],
        "regression_selected_metrics": summaries[10]["regression_fidelity"],
        "selected_seed_stability": {
            "classification": summaries[5]["seed_stability"][
                "classification_top1_pairwise_agreement"
            ],
            "regression": summaries[10]["seed_stability"][
                "regression_top1_pairwise_agreement"
            ],
        },
        "interpretation": (
            "文本是分类主证据；音频和视觉是小幅反事实纠错来源。"
            "弱模态可以报告贡献大小与方向，但不得强行写成主要参考模态。"
        ),
        "attachment4_used_for_method_selection": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
