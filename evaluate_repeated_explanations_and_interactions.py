from __future__ import annotations

"""重复随机解释对照、分层忠实性与文本条件下音视频交互评估。"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import apply_chinese_style
import numpy as np
import torch
from scipy.stats import wilcoxon

from evaluate_explanation_methods import (
    CLASS_MODALITIES,
    REG_MODALITIES,
    CLASSIFICATION_PATHS,
    REGRESSION_PATHS,
    apply_selected_blocks,
    classification_outputs,
    load_model,
    mask_modality,
    regression_outputs,
    valid_block_matrix,
)

import sys


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mosei.data import MoseiDataModule, find_aligned_pickle
from mosei.utils import move_batch


CLASS_BLOCK_SIZE = 5
CLASS_TOP_K = 2
REG_BLOCK_SIZE = 10
REG_TOP_K = 1
RANDOM_REPEATS = 30
BOOTSTRAP_REPEATS = 5000
RANDOM_SEED = 2026
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp029_repeated_explanations_interactions"
CLASS_BLOCKS = (
    PROJECT_ROOT / "outputs" / "exp023_explanation_validation" / "block_importance.csv"
)
REG_BLOCKS = (
    PROJECT_ROOT
    / "outputs"
    / "exp023_explanation_validation_block10"
    / "block_importance.csv"
)
CLASS_SAMPLES = (
    PROJECT_ROOT
    / "outputs"
    / "exp023_explanation_validation"
    / "sample_explanations.csv"
)
REG_SAMPLES = (
    PROJECT_ROOT
    / "outputs"
    / "exp023_explanation_validation_block10"
    / "sample_explanations.csv"
)


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def top_selections(
    path: Path,
    task: str,
    modalities: tuple[str, ...],
    blocks_per_modality: int,
    top_k: int,
    absolute: bool,
) -> dict[str, list[int]]:
    grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
    modality_index = {name: index for index, name in enumerate(modalities)}
    for row in read_rows(path):
        if row["task"] != task or row["available"].lower() != "true":
            continue
        score = float(row["importance"])
        rank_score = abs(score) if absolute else score
        flat = modality_index[row["modality"]] * blocks_per_modality + int(
            row["block_index"]
        )
        grouped[row["id"]].append((rank_score, flat))
    return {
        sample_id: [flat for _, flat in sorted(values, reverse=True)[:top_k]]
        for sample_id, values in grouped.items()
    }


def random_selections(
    valid: np.ndarray, top_k: int, rng: np.random.Generator
) -> list[list[int]]:
    result = []
    for sample_valid in valid:
        candidates = np.flatnonzero(sample_valid)
        count = min(top_k, len(candidates))
        selected = rng.choice(candidates, size=count, replace=False) if count else []
        result.append([int(value) for value in selected])
    return result


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    means = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    for repeat in range(BOOTSTRAP_REPEATS):
        sample = rng.integers(0, len(values), size=len(values))
        means[repeat] = values[sample].mean()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_figure(rows: list[dict], path: Path) -> None:
    apply_chinese_style()
    class_top = np.asarray([row["classification_top_deletion"] for row in rows])
    class_random = np.asarray([row["classification_random_deletion_mean"] for row in rows])
    reg_top = np.asarray([row["regression_top_deletion"] for row in rows])
    reg_random = np.asarray([row["regression_random_deletion_mean"] for row in rows])
    interaction = np.asarray([row["classification_audio_visual_interaction"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].boxplot([class_top, class_random], tick_labels=["最高贡献证据", "30次随机均值"], showfliers=False)
    axes[0].set_title("分类删除忠实性")
    axes[0].set_ylabel("目标类别概率下降")
    axes[1].boxplot([reg_top, reg_random], tick_labels=["最高贡献证据", "30次随机均值"], showfliers=False)
    axes[1].set_title("回归删除忠实性")
    axes[1].set_ylabel("预测值绝对变化")
    axes[2].hist(interaction, bins=35, color="#2563EB", alpha=0.8, label="样本分布")
    axes[2].axvline(0, color="#64748B", ls="--", lw=1, label="零交互基准")
    axes[2].set_title("文本条件下音视频交互")
    axes[2].set_xlabel("二阶反事实交互值")
    axes[2].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("重复随机解释验证与模态交互", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_models = [load_model(path, device) for path in CLASSIFICATION_PATHS]
    reg_models = [load_model(path, device) for path in REGRESSION_PATHS]
    module = MoseiDataModule(find_aligned_pickle(PROJECT_ROOT), batch_size=128)
    rng = np.random.default_rng(RANDOM_SEED)

    class_top = top_selections(
        CLASS_BLOCKS,
        "classification",
        CLASS_MODALITIES,
        50 // CLASS_BLOCK_SIZE,
        CLASS_TOP_K,
        absolute=False,
    )
    reg_top = top_selections(
        REG_BLOCKS,
        "regression",
        REG_MODALITIES,
        50 // REG_BLOCK_SIZE,
        REG_TOP_K,
        absolute=True,
    )
    class_sample_rows = {row["id"]: row for row in read_rows(CLASS_SAMPLES)}
    reg_sample_rows = {row["id"]: row for row in read_rows(REG_SAMPLES)}

    output_rows = []
    for batch_number, raw_batch in enumerate(module.loader("valid", shuffle=False), start=1):
        ids = [str(value) for value in raw_batch["id"]]
        batch = move_batch(raw_batch, device)
        base_prob, _, base_vote = classification_outputs(class_models, batch)
        base_mean_prob = base_prob.mean(dim=0)
        target = base_mean_prob.gather(1, base_vote.unsqueeze(1)).squeeze(1)
        base_reg = regression_outputs(reg_models, batch).mean(dim=0)

        class_valid = valid_block_matrix(batch, CLASS_MODALITIES, CLASS_BLOCK_SIZE)
        reg_valid = valid_block_matrix(batch, REG_MODALITIES, REG_BLOCK_SIZE)
        class_random_delete = np.zeros((RANDOM_REPEATS, len(ids)), dtype=np.float64)
        class_random_retain = np.zeros((RANDOM_REPEATS, len(ids)), dtype=np.float64)
        reg_random_delete = np.zeros((RANDOM_REPEATS, len(ids)), dtype=np.float64)
        for repeat in range(RANDOM_REPEATS):
            class_selected = random_selections(class_valid, CLASS_TOP_K, rng)
            reg_selected = random_selections(reg_valid, REG_TOP_K, rng)
            deleted_prob, _, _ = classification_outputs(
                class_models,
                apply_selected_blocks(
                    batch,
                    class_selected,
                    CLASS_MODALITIES,
                    CLASS_BLOCK_SIZE,
                    retain_only=False,
                ),
            )
            retained_prob, _, _ = classification_outputs(
                class_models,
                apply_selected_blocks(
                    batch,
                    class_selected,
                    CLASS_MODALITIES,
                    CLASS_BLOCK_SIZE,
                    retain_only=True,
                ),
            )
            deleted_target = deleted_prob.mean(dim=0).gather(
                1, base_vote.unsqueeze(1)
            ).squeeze(1)
            retained_target = retained_prob.mean(dim=0).gather(
                1, base_vote.unsqueeze(1)
            ).squeeze(1)
            class_random_delete[repeat] = (target - deleted_target).cpu().numpy()
            class_random_retain[repeat] = retained_target.cpu().numpy()

            deleted_reg = regression_outputs(
                reg_models,
                apply_selected_blocks(
                    batch,
                    reg_selected,
                    REG_MODALITIES,
                    REG_BLOCK_SIZE,
                    retain_only=False,
                ),
            ).mean(dim=0)
            reg_random_delete[repeat] = (base_reg - deleted_reg).abs().cpu().numpy()

        no_audio_prob, _, _ = classification_outputs(
            class_models, mask_modality(batch, "audio")
        )
        no_vision_prob, _, _ = classification_outputs(
            class_models, mask_modality(batch, "vision")
        )
        no_av_prob, _, _ = classification_outputs(
            class_models, mask_modality(mask_modality(batch, "audio"), "vision")
        )
        gather = base_vote.unsqueeze(1)
        no_audio_target = no_audio_prob.mean(dim=0).gather(1, gather).squeeze(1)
        no_vision_target = no_vision_prob.mean(dim=0).gather(1, gather).squeeze(1)
        no_av_target = no_av_prob.mean(dim=0).gather(1, gather).squeeze(1)
        interaction = (
            target - no_vision_target - no_audio_target + no_av_target
        ).cpu().numpy()

        for index, sample_id in enumerate(ids):
            class_row = class_sample_rows[sample_id]
            reg_row = reg_sample_rows[sample_id]
            output_rows.append(
                {
                    "id": sample_id,
                    "classification_true": int(class_row["classification_true"]),
                    "classification_pred": int(class_row["classification_pred"]),
                    "classification_correct": class_row["classification_correct"].lower() == "true",
                    "classification_top_deletion": float(class_row["classification_top_deletion_delta"]),
                    "classification_random_deletion_mean": float(class_random_delete[:, index].mean()),
                    "classification_top_minus_random_deletion": float(class_row["classification_top_deletion_delta"]) - float(class_random_delete[:, index].mean()),
                    "classification_top_retention": float(class_row["classification_top_retained_probability"]),
                    "classification_random_retention_mean": float(class_random_retain[:, index].mean()),
                    "classification_top_minus_random_retention": float(class_row["classification_top_retained_probability"]) - float(class_random_retain[:, index].mean()),
                    "regression_top_deletion": float(reg_row["regression_top_deletion_abs_delta"]),
                    "regression_random_deletion_mean": float(reg_random_delete[:, index].mean()),
                    "regression_top_minus_random_deletion": float(reg_row["regression_top_deletion_abs_delta"]) - float(reg_random_delete[:, index].mean()),
                    "classification_audio_visual_interaction": float(interaction[index]),
                    "classification_audio_visual_interaction_abs": float(abs(interaction[index])),
                }
            )
        print(f"batch {batch_number}: 完成{len(output_rows)}/728条")

    class_delete_gap = np.asarray(
        [row["classification_top_minus_random_deletion"] for row in output_rows]
    )
    class_retain_gap = np.asarray(
        [row["classification_top_minus_random_retention"] for row in output_rows]
    )
    reg_delete_gap = np.asarray(
        [row["regression_top_minus_random_deletion"] for row in output_rows]
    )
    interaction = np.asarray(
        [row["classification_audio_visual_interaction"] for row in output_rows]
    )
    bootstrap_rng = np.random.default_rng(RANDOM_SEED + 1)

    strata_rows = []
    strata = [("全部", np.ones(len(output_rows), dtype=bool))]
    correct = np.asarray([row["classification_correct"] for row in output_rows])
    strata.extend([("分类正确", correct), ("分类错误", ~correct)])
    true_class = np.asarray([row["classification_true"] for row in output_rows])
    strata.extend(
        [(name, true_class == value) for value, name in enumerate(("负向", "中性", "正向"))]
    )
    for name, selected in strata:
        strata_rows.append(
            {
                "stratum": name,
                "count": int(selected.sum()),
                "classification_deletion_gap_mean": float(class_delete_gap[selected].mean()),
                "classification_retention_gap_mean": float(class_retain_gap[selected].mean()),
                "regression_deletion_gap_mean": float(reg_delete_gap[selected].mean()),
                "audio_visual_interaction_abs_mean": float(np.abs(interaction[selected]).mean()),
            }
        )

    summary = {
        "protocol": {
            "split": "official_valid_only",
            "sample_count": len(output_rows),
            "random_repeats_per_sample": RANDOM_REPEATS,
            "classification_block_size": CLASS_BLOCK_SIZE,
            "classification_top_k": CLASS_TOP_K,
            "regression_block_size": REG_BLOCK_SIZE,
            "regression_top_k": REG_TOP_K,
            "test_accessed": False,
        },
        "classification_fidelity": {
            "top_deletion_mean": float(np.mean([row["classification_top_deletion"] for row in output_rows])),
            "random_deletion_mean": float(np.mean([row["classification_random_deletion_mean"] for row in output_rows])),
            "top_minus_random_deletion_mean": float(class_delete_gap.mean()),
            "top_minus_random_deletion_ci95": bootstrap_interval(class_delete_gap, bootstrap_rng),
            "wilcoxon_greater_p": float(wilcoxon(class_delete_gap, alternative="greater").pvalue),
            "top_retention_mean": float(np.mean([row["classification_top_retention"] for row in output_rows])),
            "random_retention_mean": float(np.mean([row["classification_random_retention_mean"] for row in output_rows])),
            "top_minus_random_retention_mean": float(class_retain_gap.mean()),
            "top_minus_random_retention_ci95": bootstrap_interval(class_retain_gap, bootstrap_rng),
        },
        "regression_fidelity": {
            "top_deletion_mean": float(np.mean([row["regression_top_deletion"] for row in output_rows])),
            "random_deletion_mean": float(np.mean([row["regression_random_deletion_mean"] for row in output_rows])),
            "top_minus_random_deletion_mean": float(reg_delete_gap.mean()),
            "top_minus_random_deletion_ci95": bootstrap_interval(reg_delete_gap, bootstrap_rng),
            "wilcoxon_greater_p": float(wilcoxon(reg_delete_gap, alternative="greater").pvalue),
        },
        "audio_visual_interaction_given_text": {
            "definition": "p(T,A,V)-p(T,A,0)-p(T,0,V)+p(T,0,0)",
            "signed_mean": float(interaction.mean()),
            "signed_mean_ci95": bootstrap_interval(interaction, bootstrap_rng),
            "absolute_mean": float(np.abs(interaction).mean()),
            "positive_ratio": float(np.mean(interaction > 0)),
            "negative_ratio": float(np.mean(interaction < 0)),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "per_sample.csv", output_rows)
    write_csv(OUTPUT_DIR / "stratified_summary.csv", strata_rows)
    create_figure(output_rows, OUTPUT_DIR / "重复随机解释与模态交互.png")
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
