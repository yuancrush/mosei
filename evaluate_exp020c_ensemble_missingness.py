"""评估EXP-020C正式任务专家集成在局部缺失场景下的鲁棒性。"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate_missingness import mask_batch, scenarios
from mosei.data import MoseiDataModule, find_aligned_pickle
from mosei.metrics import compute_metrics
from mosei.models import build_model
from mosei.utils import move_batch


CLASSIFICATION_PATHS = [
    PROJECT_ROOT
    / "outputs"
    / run_name
    / "fusion_output_residual_bigru"
    / "best.pt"
    for run_name in (
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp021_ensemble_missing_stress"
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
    voted = []
    for column in predictions.T:
        voted.append(int(np.argmax(np.bincount(column.astype(int), minlength=3))))
    return np.asarray(voted)


@torch.inference_mode()
def evaluate_scenario(
    classification_models: list[torch.nn.Module],
    regression_models: list[torch.nn.Module],
    module: MoseiDataModule,
    scenario: dict,
    device: torch.device,
) -> dict:
    true_cls, true_reg = [], []
    class_by_seed = [[] for _ in classification_models]
    regression_by_seed = [[] for _ in regression_models]
    requested_steps = removed_steps = 0
    for batch in module.loader("valid", shuffle=False):
        changed, requested, removed = mask_batch(batch, scenario)
        requested_steps += requested
        removed_steps += removed
        true_cls.append(changed["classification_label"].numpy())
        true_reg.append(changed["regression_label"].numpy())
        changed = move_batch(changed, device)
        for index, model in enumerate(classification_models):
            output = model(changed)
            class_by_seed[index].append(
                output["classification_logits"].argmax(dim=-1).cpu().numpy()
            )
        for index, model in enumerate(regression_models):
            output = model(changed)
            regression_by_seed[index].append(
                output["regression"].clamp(-3.0, 3.0).cpu().numpy()
            )

    y_cls = np.concatenate(true_cls)
    y_reg = np.concatenate(true_reg)
    classification = majority_vote(
        np.asarray([np.concatenate(parts) for parts in class_by_seed])
    )
    regression = np.mean(
        np.asarray([np.concatenate(parts) for parts in regression_by_seed]), axis=0
    )
    metrics = compute_metrics(y_cls, classification, y_reg, regression)
    return {
        "scenario": scenario["scenario"],
        "missing_modalities": "+".join(scenario["modalities"]) or "none",
        "requested_ratio": scenario["ratio"],
        "location": scenario["location"],
        "removed_available_ratio": (
            float(removed_steps / requested_steps) if requested_steps else 0.0
        ),
        **{
            key: metrics[key]
            for key in (
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "mae",
                "pearson",
                "selection_score",
            )
        },
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = MoseiDataModule(
        find_aligned_pickle(PROJECT_ROOT), batch_size=128, pin_memory=device.type == "cuda"
    )
    classification_models = [load_model(path, device) for path in CLASSIFICATION_PATHS]
    regression_models = [load_model(path, device) for path in REGRESSION_PATHS]
    rows = []
    for scenario in scenarios():
        row = evaluate_scenario(
            classification_models, regression_models, module, scenario, device
        )
        rows.append(row)
        print(
            f"{row['scenario']:15s} F1={row['macro_f1']:.4f} "
            f"MAE={row['mae']:.4f} r={row['pearson']:.4f}"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
