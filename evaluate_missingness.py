from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

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
from plot_style import apply_chinese_style


apply_chinese_style()

MODEL_DISPLAY_NAMES = {
    "fusion": "三模态均值融合",
    "text_bigru": "文本双向GRU",
    "text_audio_bigru": "文本+语音双向GRU",
    "fusion_uncertainty_bigru": "不确定性感知三模态融合",
    "fusion_task_gated_bigru": "任务特异门控三模态融合",
    "fusion_output_residual_bigru": "输出残差三模态融合",
}


DEFAULT_CHECKPOINTS = (
    PROJECT_ROOT / "outputs" / "baselines_seed2026" / "fusion" / "best.pt",
    PROJECT_ROOT / "outputs" / "exp001_text_bigru" / "text_bigru" / "best.pt",
    PROJECT_ROOT / "outputs" / "exp002_pair_bigru" / "text_audio_bigru" / "best.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证集局部模态缺失压力测试")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "exp009_missing_stress")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def scenarios() -> List[dict]:
    result = [{"scenario": "clean", "modalities": (), "ratio": 0.0, "location": "none"}]
    for modalities, name in (("audio", "A"), ("vision", "V"), (("audio", "vision"), "AV")):
        modalities = (modalities,) if isinstance(modalities, str) else modalities
        for ratio in (0.1, 0.3, 0.5):
            for location in ("front", "middle", "back"):
                result.append(
                    {
                        "scenario": f"{name}_{location}_{int(ratio * 100)}",
                        "modalities": modalities,
                        "ratio": ratio,
                        "location": location,
                    }
                )
        result.append(
            {
                "scenario": f"{name}_all",
                "modalities": modalities,
                "ratio": 1.0,
                "location": "all",
            }
        )
    return result


def block_bounds(length: int, ratio: float, location: str) -> tuple[int, int]:
    width = max(1, min(length, math.ceil(length * ratio)))
    if location == "all":
        start = 0
    elif location == "front":
        start = 0
    elif location == "back":
        start = length - width
    else:
        start = (length - width) // 2
    return start, start + width


def mask_batch(batch: Dict[str, object], scenario: dict) -> tuple[Dict[str, object], int, int]:
    if not scenario["modalities"]:
        return batch, 0, 0
    changed = dict(batch)
    text_mask = batch["text_mask"]
    requested_steps = 0
    removed_available_steps = 0
    for modality in scenario["modalities"]:
        values = batch[modality].clone()
        mask = batch[f"{modality}_mask"].clone()
        for index in range(values.shape[0]):
            valid_positions = torch.nonzero(text_mask[index], as_tuple=False).flatten()
            length = int(valid_positions[-1].item() + 1) if len(valid_positions) else 0
            if length == 0:
                continue
            start, end = block_bounds(length, scenario["ratio"], scenario["location"])
            requested_steps += end - start
            removed_available_steps += int(mask[index, start:end].sum().item())
            values[index, start:end] = 0.0
            mask[index, start:end] = False
        changed[modality] = values
        changed[f"{modality}_mask"] = mask
    return changed, requested_steps, removed_available_steps


@torch.inference_mode()
def evaluate_scenario(
    model: torch.nn.Module,
    loader: Iterable[Dict[str, object]],
    scenario: dict,
    device: torch.device,
) -> dict:
    model.eval()
    true_cls, pred_cls, true_reg, pred_reg = [], [], [], []
    requested_steps = removed_steps = 0
    weight_sum = uncertainty_sum = None
    diagnostic_samples = 0
    for batch in loader:
        batch, requested, removed = mask_batch(batch, scenario)
        requested_steps += requested
        removed_steps += removed
        batch = move_batch(batch, device)
        outputs = model(batch)
        if "fusion_weights" in outputs:
            current_weights = outputs["fusion_weights"].detach().sum(dim=0).cpu()
            current_uncertainty = outputs["modality_uncertainty"].detach().sum(dim=0).cpu()
            weight_sum = current_weights if weight_sum is None else weight_sum + current_weights
            uncertainty_sum = current_uncertainty if uncertainty_sum is None else uncertainty_sum + current_uncertainty
            diagnostic_samples += outputs["fusion_weights"].shape[0]
        true_cls.append(batch["classification_label"].cpu().numpy())
        pred_cls.append(outputs["classification_logits"].argmax(dim=-1).cpu().numpy())
        true_reg.append(batch["regression_label"].cpu().numpy())
        pred_reg.append(outputs["regression"].clamp(-3.0, 3.0).cpu().numpy())
    metrics = compute_metrics(
        np.concatenate(true_cls),
        np.concatenate(pred_cls),
        np.concatenate(true_reg),
        np.concatenate(pred_reg),
    )
    diagnostics = {}
    for index, modality in enumerate(("text", "audio", "vision")):
        diagnostics[f"weight_{modality}"] = (
            float(weight_sum[index] / diagnostic_samples) if weight_sum is not None else float("nan")
        )
        diagnostics[f"uncertainty_{modality}"] = (
            float(uncertainty_sum[index] / diagnostic_samples) if uncertainty_sum is not None else float("nan")
        )
    return {
        "scenario": scenario["scenario"],
        "missing_modalities": "+".join(scenario["modalities"]) or "none",
        "requested_ratio": scenario["ratio"],
        "location": scenario["location"],
        "requested_steps": requested_steps,
        "removed_available_steps": removed_steps,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "mae": metrics["mae"],
        "pearson": metrics["pearson"],
        "selection_score": metrics["selection_score"],
        **diagnostics,
    }


def save_csv(rows: List[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: List[dict], path: Path) -> None:
    selected = [row for row in rows if row["missing_modalities"] == "audio+vision" and row["location"] == "middle"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model_name in dict.fromkeys(row["model"] for row in selected):
        model_rows = sorted((row for row in selected if row["model"] == model_name), key=lambda row: row["requested_ratio"])
        x = [row["requested_ratio"] for row in model_rows]
        model_label = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        axes[0].plot(x, [row["macro_f1"] for row in model_rows], "o-", label=model_label)
        axes[1].plot(x, [row["mae"] for row in model_rows], "o-", label=model_label)
    axes[0].set_ylabel("Macro-F1")
    axes[1].set_ylabel("MAE")
    for ax in axes:
        ax.set_xlabel("语音与视觉中部联合缺失比例")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    data_path = args.data or find_aligned_pickle(PROJECT_ROOT)
    module = MoseiDataModule(data_path, batch_size=args.batch_size, pin_memory=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for checkpoint_path in args.checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model(checkpoint["model_name"], checkpoint["hidden_dim"], checkpoint["dropout"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Evaluating {checkpoint['model_name']}: {checkpoint_path}")
        for scenario in scenarios():
            row = evaluate_scenario(model, module.loader("valid", shuffle=False), scenario, device)
            row = {"model": checkpoint["model_name"], **row}
            rows.append(row)
            print(f"  {row['scenario']:15s} F1={row['macro_f1']:.4f} MAE={row['mae']:.4f} r={row['pearson']:.4f}")
    save_csv(rows, args.output_dir / "metrics.csv")
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    save_plot(rows, args.output_dir / "joint_middle_degradation.png")
    print(f"Saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
