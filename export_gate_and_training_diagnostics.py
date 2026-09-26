"""导出定型模型的弱模态门控响应与三随机种子训练轨迹。

只读取官方 valid、既有冻结权重和历史日志，不训练模型、不访问 test。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import apply_chinese_style
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mosei.data import MoseiDataModule, find_aligned_pickle
from mosei.models import build_model
from mosei.utils import move_batch


CLASSIFICATION_RUNS = {
    2026: PROJECT_ROOT / "outputs" / "exp020c_output_residual" / "fusion_output_residual_bigru",
    2027: PROJECT_ROOT / "outputs" / "exp020c_output_residual_seed2027" / "fusion_output_residual_bigru",
    2028: PROJECT_ROOT / "outputs" / "exp020c_output_residual_seed2028" / "fusion_output_residual_bigru",
}
REGRESSION_RUNS = {
    seed: PROJECT_ROOT / "outputs" / f"exp014_text_audio_seed{seed}" / "text_audio_bigru"
    for seed in (2026, 2027, 2028)
}
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp031_gate_and_training_diagnostics"
MODALITIES = ("audio", "vision")
MODALITY_NAMES = {"audio": "语音", "vision": "视觉"}
SCALE = 0.25
QUANTILES = np.linspace(0.0, 1.0, 6)
QUANTILE_LABELS = ["Q1（低）", "Q2", "Q3", "Q4", "Q5（高）"]


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        checkpoint["model_name"], checkpoint["hidden_dim"], checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.fusion.audio_scale = SCALE
    model.fusion.vision_scale = SCALE
    return model.eval()


def availability(batch: dict, modality: str) -> tuple[np.ndarray, np.ndarray]:
    # 与模型内部完全相同的定义：有效步比例与文本有效区间内最长连续缺口比例。
    descriptors, _ = build_availability(batch, modality)
    return (
        descriptors[:, 0].detach().cpu().numpy(),
        descriptors[:, 1].detach().cpu().numpy(),
    )


def build_availability(batch: dict, modality: str) -> tuple[torch.Tensor, torch.Tensor]:
    text_mask = batch["text_mask"].bool()
    modality_mask = batch[f"{modality}_mask"].bool() & text_mask
    text_length = text_mask.sum(dim=1).clamp_min(1).to(torch.float32)
    coverage = modality_mask.sum(dim=1).to(torch.float32) / text_length
    run = torch.zeros_like(text_length)
    longest = torch.zeros_like(text_length)
    for position in range(text_mask.shape[1]):
        missing = text_mask[:, position] & ~modality_mask[:, position]
        run = torch.where(missing, run + 1.0, torch.zeros_like(run))
        longest = torch.maximum(longest, run)
    longest_gap_ratio = longest / text_length
    return torch.stack([coverage, longest_gap_ratio], dim=-1).clamp(0.0, 1.0), coverage


@torch.inference_mode()
def export_gates() -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(path / "best.pt", device) for path in CLASSIFICATION_RUNS.values()]
    module = MoseiDataModule(find_aligned_pickle(PROJECT_ROOT), batch_size=128)
    rows = []
    for raw_batch in module.loader("valid", shuffle=False):
        ids = [str(value) for value in raw_batch["id"]]
        batch = move_batch(raw_batch, device)
        coverage_gap = {
            modality: availability(batch, modality) for modality in MODALITIES
        }
        seed_values = {modality: [] for modality in MODALITIES}
        for seed, model in zip(CLASSIFICATION_RUNS, models):
            output = model(batch)
            embeddings = {
                modality: model.encoders[modality](
                    batch[modality], batch[f"{modality}_mask"]
                )
                for modality in MODALITIES
            }
            for modality_index, modality in enumerate(MODALITIES):
                learned_gate = output["task_fusion_weights"][:, modality_index, 0]
                correction = model.fusion.classification_corrections[modality](
                    embeddings[modality]
                )
                scaled_correction = correction * SCALE
                effective_residual = learned_gate.unsqueeze(-1) * scaled_correction
                seed_values[modality].append(
                    {
                        "seed": seed,
                        "gate": learned_gate.cpu().numpy(),
                        "effective_gate": (learned_gate * SCALE).cpu().numpy(),
                        "residual_potential_l2": scaled_correction.norm(dim=-1).cpu().numpy(),
                        "effective_residual_l2": effective_residual.norm(dim=-1).cpu().numpy(),
                    }
                )
        for modality in MODALITIES:
            coverage, gap = coverage_gap[modality]
            for index, sample_id in enumerate(ids):
                rows.append(
                    {
                        "id": sample_id,
                        "modality": modality,
                        "coverage": float(coverage[index]),
                        "longest_gap_ratio": float(gap[index]),
                        "learned_gate_mean": float(
                            np.mean([item["gate"][index] for item in seed_values[modality]])
                        ),
                        "effective_gate_025_mean": float(
                            np.mean(
                                [item["effective_gate"][index] for item in seed_values[modality]]
                            )
                        ),
                        "scaled_residual_potential_l2_mean": float(
                            np.mean(
                                [item["residual_potential_l2"][index] for item in seed_values[modality]]
                            )
                        ),
                        "effective_residual_l2_mean": float(
                            np.mean(
                                [item["effective_residual_l2"][index] for item in seed_values[modality]]
                            )
                        ),
                    }
                )
    return rows


def bin_gate_rows(rows: list[dict]) -> list[dict]:
    result = []
    for modality in MODALITIES:
        modality_rows = [row for row in rows if row["modality"] == modality]
        for variable in ("coverage", "longest_gap_ratio"):
            all_values = np.asarray([row[variable] for row in modality_rows])
            edges = np.quantile(all_values, QUANTILES)
            for bin_index, label in enumerate(QUANTILE_LABELS):
                lower, upper = edges[bin_index], edges[bin_index + 1]
                selected = [
                    row
                    for row in modality_rows
                    if lower <= row[variable]
                    and (
                        row[variable] <= upper
                        if bin_index == len(QUANTILE_LABELS) - 1
                        else row[variable] < upper
                    )
                ]
                if not selected:
                    continue
                current = {
                    "modality": modality,
                    "availability_variable": variable,
                    "bin_index": bin_index,
                    "bin_label": label,
                    "bin_lower": float(lower),
                    "bin_upper": float(upper),
                    "count": len(selected),
                    "availability_mean": float(np.mean([row[variable] for row in selected])),
                }
                for metric in ("effective_gate_025_mean", "effective_residual_l2_mean"):
                    values = np.asarray([row[metric] for row in selected], dtype=np.float64)
                    standard_error = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
                    current[f"{metric}_mean"] = float(values.mean())
                    current[f"{metric}_ci95_low"] = float(values.mean() - 1.96 * standard_error)
                    current[f"{metric}_ci95_high"] = float(values.mean() + 1.96 * standard_error)
                result.append(current)
    return result


def plot_gate_response(rows: list[dict], path: Path) -> None:
    apply_chinese_style()
    colors = {"audio": "#2563EB", "vision": "#E76F51"}
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    specifications = [
        ("coverage", "模态覆盖率分箱"),
        ("longest_gap_ratio", "最长连续缺口比例分箱"),
    ]
    metrics = [
        ("effective_gate_025_mean", "有效门控系数 $0.25g_m$"),
        ("effective_residual_l2_mean", "最终分类输出残差幅度（L2范数）"),
    ]
    for column, (variable, x_label) in enumerate(specifications):
        for row_index, (metric, y_label) in enumerate(metrics):
            axis = axes[row_index, column]
            for modality in MODALITIES:
                selected = sorted(
                    [
                        row
                        for row in rows
                        if row["modality"] == modality
                        and row["availability_variable"] == variable
                    ],
                    key=lambda row: row["bin_index"],
                )
                x = np.asarray([row["bin_index"] for row in selected])
                y = np.asarray([row[f"{metric}_mean"] for row in selected])
                low = np.asarray([row[f"{metric}_ci95_low"] for row in selected])
                high = np.asarray([row[f"{metric}_ci95_high"] for row in selected])
                axis.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2,
                    color=colors[modality],
                    label=MODALITY_NAMES[modality],
                )
                axis.fill_between(x, low, high, color=colors[modality], alpha=0.14)
            axis.set_xticks(range(len(QUANTILE_LABELS)), QUANTILE_LABELS)
            axis.set_xlabel(f"{x_label}（各模态内部五分位）")
            axis.set_ylabel(y_label)
            axis.grid(alpha=0.25)
            axis.legend(title="弱模态", frameon=True, loc="best")
    figure.suptitle("置信门控随弱模态可用性变化的响应", fontsize=16)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def export_training_histories() -> tuple[list[dict], list[dict]]:
    rows, summaries = [], []
    groups = (("classification", CLASSIFICATION_RUNS), ("regression", REGRESSION_RUNS))
    for task, runs in groups:
        for seed, directory in runs.items():
            history = read_json(directory / "history.json")
            summary = read_json(directory / "summary.json")
            for item in history:
                rows.append({"task": task, "seed": seed, **item})
            summaries.append(
                {
                    "task": task,
                    "seed": seed,
                    "selected_epoch": int(summary["best_epoch"]),
                    "early_stop_epoch": int(history[-1]["epoch"]),
                    "selected_macro_f1": float(summary["valid"]["macro_f1"]),
                    "selected_mae": float(summary["valid"]["mae"]),
                    "selected_pearson": float(summary["valid"]["pearson"]),
                }
            )
    return rows, summaries


def plot_training_histories(rows: list[dict], summaries: list[dict], path: Path) -> None:
    apply_chinese_style()
    colors = {2026: "#2563EB", 2027: "#E76F51", 2028: "#2A9D8F"}
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), sharex="col")
    for column, task in enumerate(("classification", "regression")):
        metric = "valid_macro_f1" if task == "classification" else "valid_mae"
        y_label = "验证集 Macro-F1" if task == "classification" else "验证集 MAE"
        title = "三模态分类专家" if task == "classification" else "文本+语音回归专家"
        for seed in colors:
            selected_rows = [
                row for row in rows if row["task"] == task and row["seed"] == seed
            ]
            epochs = np.asarray([row["epoch"] for row in selected_rows])
            values = np.asarray([row[metric] for row in selected_rows])
            learning_rates = np.asarray([row["learning_rate"] for row in selected_rows])
            summary = next(
                row for row in summaries if row["task"] == task and row["seed"] == seed
            )
            selected_epoch = summary["selected_epoch"]
            selected_value = values[np.flatnonzero(epochs == selected_epoch)[0]]
            axes[0, column].plot(
                epochs, values, marker="o", markersize=3.5, color=colors[seed], label=f"随机种子{seed}"
            )
            axes[0, column].scatter(
                [selected_epoch], [selected_value], marker="*", s=150, color=colors[seed], zorder=5
            )
            axes[0, column].axvline(
                summary["early_stop_epoch"], color=colors[seed], linestyle=":", alpha=0.55
            )
            axes[1, column].step(
                epochs,
                learning_rates,
                where="post",
                color=colors[seed],
                linewidth=2,
                label=f"随机种子{seed}",
            )
        axes[0, column].set_title(title)
        axes[0, column].set_ylabel(y_label)
        axes[0, column].grid(alpha=0.25)
        axes[1, column].set_yscale("log")
        axes[1, column].set_ylabel("学习率（对数轴）")
        axes[1, column].set_xlabel("训练轮次")
        axes[1, column].grid(alpha=0.25)
        seed_handles = [
            Line2D([0], [0], color=colors[seed], marker="o", linewidth=2, label=f"随机种子{seed}")
            for seed in colors
        ]
        semantic_handles = [
            Line2D([0], [0], color="#111827", marker="*", linestyle="None", markersize=11, label="选模轮次"),
            Line2D([0], [0], color="#111827", linestyle=":", linewidth=1.5, label="早停轮次"),
        ]
        axes[0, column].legend(
            handles=seed_handles + semantic_handles,
            title="随机种子与标记",
            ncol=2,
            fontsize=8.5,
            loc="best",
        )
        axes[1, column].legend(title="随机种子", ncol=3, fontsize=8.5, loc="best")
    figure.text(
        0.5,
        0.015,
        "星号：验证选择轮次；同色竖虚线：早停轮次",
        ha="center",
        fontsize=10,
    )
    figure.suptitle("多随机种子训练、学习率调度与早停位置", fontsize=16)
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def correlation_summary(rows: list[dict]) -> list[dict]:
    result = []
    for modality in MODALITIES:
        selected = [row for row in rows if row["modality"] == modality]
        for x_name in ("coverage", "longest_gap_ratio"):
            for y_name in ("effective_gate_025_mean", "effective_residual_l2_mean"):
                statistic = spearmanr(
                    [row[x_name] for row in selected], [row[y_name] for row in selected]
                )
                result.append(
                    {
                        "modality": modality,
                        "x": x_name,
                        "y": y_name,
                        "spearman": float(statistic.statistic),
                        "p_value": float(statistic.pvalue),
                    }
                )
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gate_rows = export_gates()
    binned_rows = bin_gate_rows(gate_rows)
    correlations = correlation_summary(gate_rows)
    history_rows, history_summaries = export_training_histories()
    write_csv(OUTPUT_DIR / "门控逐样本数据.csv", gate_rows)
    write_csv(OUTPUT_DIR / "门控分箱统计.csv", binned_rows)
    write_csv(OUTPUT_DIR / "门控相关性统计.csv", correlations)
    write_csv(OUTPUT_DIR / "三种子训练轨迹.csv", history_rows)
    write_csv(OUTPUT_DIR / "选模与早停轮次.csv", history_summaries)
    plot_gate_response(binned_rows, OUTPUT_DIR / "图6-3_置信门控响应曲线.png")
    plot_training_histories(
        history_rows, history_summaries, OUTPUT_DIR / "图6-4_多随机种子训练与早停.png"
    )
    summary = {
        "protocol": {
            "gate_split": "official_valid_only",
            "test_accessed": False,
            "classification_seeds": list(CLASSIFICATION_RUNS),
            "regression_seeds": list(REGRESSION_RUNS),
            "weak_modality_scale": SCALE,
            "training_reexecuted": False,
            "source": "frozen checkpoints and complete history.json logs",
        },
        "gate_correlations": correlations,
        "training_runs": history_summaries,
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
