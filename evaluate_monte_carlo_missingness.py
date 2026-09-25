from __future__ import annotations

"""在官方验证集上进行随机连续块缺失的蒙特卡洛鲁棒性评估。"""

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mosei.data import MoseiDataModule, find_aligned_pickle
from mosei.metrics import compute_metrics
from mosei.models import build_model
from mosei.utils import move_batch


CLASSIFICATION_PATHS = [
    PROJECT_ROOT / "outputs" / run / "fusion_output_residual_bigru" / "best.pt"
    for run in (
        "exp020c_output_residual",
        "exp020c_output_residual_seed2027",
        "exp020c_output_residual_seed2028",
    )
]
REGRESSION_PATHS = [
    PROJECT_ROOT
    / "outputs"
    / f"exp014_text_audio_seed{seed}"
    / "text_audio_bigru"
    / "best.pt"
    for seed in (2026, 2027, 2028)
]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp027_monte_carlo_missingness"
RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
MODALITY_GROUPS = (("audio",), ("vision",), ("audio", "vision"))
REPEATS = 20
AUDIO_SCALE = 0.25
VISION_SCALE = 0.25


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        checkpoint["model_name"], checkpoint["hidden_dim"], checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    if checkpoint["model_name"] == "fusion_output_residual_bigru":
        model.fusion.audio_scale = AUDIO_SCALE
        model.fusion.vision_scale = VISION_SCALE
    return model.eval()


def majority_vote(predictions: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            int(np.argmax(np.bincount(column.astype(int), minlength=3)))
            for column in predictions.T
        ]
    )


def random_contiguous_mask(
    batch: Dict[str, object],
    modalities: Sequence[str],
    ratio: float,
    rng: np.random.Generator,
) -> tuple[Dict[str, object], int, int]:
    changed = dict(batch)
    text_mask = batch["text_mask"]
    starts_and_ends = []
    for index in range(text_mask.shape[0]):
        valid_positions = torch.nonzero(text_mask[index], as_tuple=False).flatten()
        length = int(valid_positions[-1].item() + 1) if len(valid_positions) else 0
        if length == 0:
            starts_and_ends.append((0, 0))
            continue
        width = max(1, min(length, math.ceil(length * ratio)))
        start = int(rng.integers(0, length - width + 1))
        starts_and_ends.append((start, start + width))

    requested = removed = 0
    for modality in modalities:
        values = batch[modality].clone()
        mask = batch[f"{modality}_mask"].clone()
        for index, (start, end) in enumerate(starts_and_ends):
            requested += end - start
            removed += int(mask[index, start:end].sum().item())
            values[index, start:end] = 0.0
            mask[index, start:end] = False
        changed[modality] = values
        changed[f"{modality}_mask"] = mask
    return changed, requested, removed


@torch.inference_mode()
def evaluate_once(
    class_models: Sequence[torch.nn.Module],
    reg_models: Sequence[torch.nn.Module],
    module: MoseiDataModule,
    modalities: Sequence[str],
    ratio: float,
    random_seed: int,
    device: torch.device,
) -> dict:
    rng = np.random.default_rng(random_seed)
    true_cls, true_reg = [], []
    class_parts = [[] for _ in class_models]
    reg_parts = [[] for _ in reg_models]
    requested = removed = 0
    for raw_batch in module.loader("valid", shuffle=False):
        changed, current_requested, current_removed = random_contiguous_mask(
            raw_batch, modalities, ratio, rng
        )
        requested += current_requested
        removed += current_removed
        true_cls.append(changed["classification_label"].numpy())
        true_reg.append(changed["regression_label"].numpy())
        batch = move_batch(changed, device)
        for index, model in enumerate(class_models):
            class_parts[index].append(
                model(batch)["classification_logits"].argmax(dim=-1).cpu().numpy()
            )
        for index, model in enumerate(reg_models):
            reg_parts[index].append(
                model(batch)["regression"].clamp(-3.0, 3.0).cpu().numpy()
            )
    y_cls = np.concatenate(true_cls)
    y_reg = np.concatenate(true_reg)
    class_pred = majority_vote(
        np.asarray([np.concatenate(parts) for parts in class_parts])
    )
    reg_pred = np.mean(
        np.asarray([np.concatenate(parts) for parts in reg_parts]), axis=0
    )
    metrics = compute_metrics(y_cls, class_pred, y_reg, reg_pred)
    return {
        "missing_modalities": "+".join(modalities),
        "requested_ratio": ratio,
        "random_seed": random_seed,
        "removed_available_fraction": float(removed / requested) if requested else 0.0,
        **{
            key: metrics[key]
            for key in ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson")
        },
    }


def clean_metrics(
    class_models: Sequence[torch.nn.Module],
    reg_models: Sequence[torch.nn.Module],
    module: MoseiDataModule,
    device: torch.device,
) -> dict:
    true_cls, true_reg = [], []
    class_parts = [[] for _ in class_models]
    reg_parts = [[] for _ in reg_models]
    with torch.inference_mode():
        for raw_batch in module.loader("valid", shuffle=False):
            true_cls.append(raw_batch["classification_label"].numpy())
            true_reg.append(raw_batch["regression_label"].numpy())
            batch = move_batch(raw_batch, device)
            for index, model in enumerate(class_models):
                class_parts[index].append(
                    model(batch)["classification_logits"].argmax(dim=-1).cpu().numpy()
                )
            for index, model in enumerate(reg_models):
                reg_parts[index].append(
                    model(batch)["regression"].clamp(-3.0, 3.0).cpu().numpy()
                )
    metrics = compute_metrics(
        np.concatenate(true_cls),
        majority_vote(np.asarray([np.concatenate(parts) for parts in class_parts])),
        np.concatenate(true_reg),
        np.mean(np.asarray([np.concatenate(parts) for parts in reg_parts]), axis=0),
    )
    return {key: metrics[key] for key in ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson")}


def interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_figure(summary_rows: list[dict], clean: dict, path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    labels = {"audio": "语音缺失", "vision": "视觉缺失", "audio+vision": "语音+视觉同步缺失"}
    colors = {"audio": "#2563EB", "vision": "#F59E0B", "audio+vision": "#DC2626"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for modality in labels:
        rows = [row for row in summary_rows if row["missing_modalities"] == modality]
        x = np.asarray([row["requested_ratio"] for row in rows]) * 100
        for axis, metric, clean_value in zip(
            axes,
            ("macro_f1_mean", "mae_mean", "pearson_mean"),
            (clean["macro_f1"], clean["mae"], clean["pearson"]),
        ):
            y = np.asarray([row[metric] for row in rows])
            low = np.asarray([row[metric.replace("_mean", "_ci95_low")] for row in rows])
            high = np.asarray([row[metric.replace("_mean", "_ci95_high")] for row in rows])
            axis.plot(x, y, marker="o", lw=1.8, color=colors[modality], label=labels[modality])
            axis.fill_between(x, low, high, color=colors[modality], alpha=0.12)
            axis.axhline(clean_value, color="#64748B", lw=0.8, ls="--")
    axes[0].set_title("Macro-F1")
    axes[1].set_title("MAE")
    axes[2].set_title("Pearson")
    for axis in axes:
        axis.set_xlabel("随机连续缺失比例（%）")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("指标值")
    axes[0].legend(fontsize=8)
    fig.suptitle("随机连续块缺失下的验证集鲁棒性响应", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_models = [load_model(path, device) for path in CLASSIFICATION_PATHS]
    reg_models = [load_model(path, device) for path in REGRESSION_PATHS]
    module = MoseiDataModule(find_aligned_pickle(PROJECT_ROOT), batch_size=128)
    clean = clean_metrics(class_models, reg_models, module, device)
    raw_rows = []
    for modalities in MODALITY_GROUPS:
        for ratio in RATIOS:
            for repeat in range(REPEATS):
                seed = 2026000 + repeat + int(ratio * 1000) + 10000 * len(modalities)
                row = evaluate_once(
                    class_models, reg_models, module, modalities, ratio, seed, device
                )
                row["repeat"] = repeat
                raw_rows.append(row)
            print(f"{'+'.join(modalities)} ratio={ratio:.1f}: 完成{REPEATS}次")

    summary_rows = []
    for modalities in MODALITY_GROUPS:
        name = "+".join(modalities)
        for ratio in RATIOS:
            selected = [
                row
                for row in raw_rows
                if row["missing_modalities"] == name
                and row["requested_ratio"] == ratio
            ]
            summary = {
                "missing_modalities": name,
                "requested_ratio": ratio,
                "repeats": len(selected),
                "removed_available_fraction_mean": float(
                    np.mean([row["removed_available_fraction"] for row in selected])
                ),
            }
            for metric in ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson"):
                values = np.asarray([row[metric] for row in selected])
                low, high = interval(values)
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_std"] = float(values.std(ddof=1))
                summary[f"{metric}_ci95_low"] = low
                summary[f"{metric}_ci95_high"] = high
            summary["macro_f1_drop_from_clean"] = clean["macro_f1"] - summary["macro_f1_mean"]
            summary["mae_increase_from_clean"] = summary["mae_mean"] - clean["mae"]
            summary_rows.append(summary)

    auc_rows = []
    ratio_array = np.asarray(RATIOS)
    for modalities in MODALITY_GROUPS:
        name = "+".join(modalities)
        rows = [row for row in summary_rows if row["missing_modalities"] == name]
        auc_rows.append(
            {
                "missing_modalities": name,
                "macro_f1_mean_over_ratios": float(np.mean([row["macro_f1_mean"] for row in rows])),
                "macro_f1_normalized_auc": float(
                    np.trapz([row["macro_f1_mean"] for row in rows], ratio_array)
                    / (ratio_array[-1] - ratio_array[0])
                    / clean["macro_f1"]
                ),
                "mae_mean_over_ratios": float(np.mean([row["mae_mean"] for row in rows])),
                "pearson_mean_over_ratios": float(np.mean([row["pearson_mean"] for row in rows])),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "raw_runs.csv", raw_rows)
    write_csv(OUTPUT_DIR / "summary.csv", summary_rows)
    write_csv(OUTPUT_DIR / "robustness_auc.csv", auc_rows)
    create_figure(summary_rows, clean, OUTPUT_DIR / "连续缺失鲁棒性响应.png")
    result = {
        "protocol": {
            "split": "official_valid_only",
            "random_block_repeats": REPEATS,
            "ratios": list(RATIOS),
            "test_accessed": False,
            "attachment3_masks_used_for_selection": False,
        },
        "clean": clean,
        "robustness_auc": auc_rows,
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
