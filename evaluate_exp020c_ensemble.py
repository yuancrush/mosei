"""汇总EXP-020C三种子分类投票，并复用已冻结的Text+Audio回归专家。"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mosei.metrics import compute_metrics
from mosei.data import MoseiDataModule, find_aligned_pickle
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
    / "valid_predictions.json"
    for seed in (2026, 2027, 2028)
]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp020c_multiseed_experts"
AUDIO_SCALE = 0.25
VISION_SCALE = 0.25


def read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def majority_vote(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    voted, agreement = [], []
    for column in predictions.T:
        counts = np.bincount(column.astype(int), minlength=3)
        voted.append(int(np.argmax(counts)))
        agreement.append(float(np.max(counts) / len(column)))
    return np.asarray(voted), np.asarray(agreement)


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        checkpoint["model_name"], checkpoint["hidden_dim"], checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.fusion.audio_scale = AUDIO_SCALE
    model.fusion.vision_scale = VISION_SCALE
    return model.eval()


def main() -> None:
    regression_runs = [read(path) for path in REGRESSION_PATHS]
    first = regression_runs[0]
    for run in regression_runs[1:]:
        if first["ids"] != run["ids"]:
            raise ValueError("验证样本顺序不一致")
        if first["classification_true"] != run["classification_true"]:
            raise ValueError("分类真值不一致")
        if not np.allclose(first["regression_true"], run["regression_true"]):
            raise ValueError("回归真值不一致")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(path, device) for path in CLASSIFICATION_PATHS]
    module = MoseiDataModule(find_aligned_pickle(PROJECT_ROOT), batch_size=128)
    classification_parts = [[] for _ in models]
    with torch.inference_mode():
        for batch in module.loader("valid", shuffle=False):
            batch = move_batch(batch, device)
            for index, model in enumerate(models):
                classification_parts[index].append(
                    model(batch)["classification_logits"].argmax(dim=-1).cpu().numpy()
                )
    classification, agreement = majority_vote(
        np.asarray([np.concatenate(parts) for parts in classification_parts])
    )
    regression = np.mean(
        [run["regression_pred"] for run in regression_runs], axis=0
    )
    true_cls = np.asarray(first["classification_true"])
    true_reg = np.asarray(first["regression_true"])
    metrics = compute_metrics(true_cls, classification, true_reg, regression)
    summary = {
        "classification": "fusion_output_residual_bigru_majority_3seed",
        "audio_scale": AUDIO_SCALE,
        "vision_scale": VISION_SCALE,
        "regression": "text_audio_regression_mean_3seed",
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
        "mean_vote_agreement": float(np.mean(agreement)),
        "unanimous_ratio": float(np.mean(agreement == 1.0)),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (OUTPUT_DIR / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    with (OUTPUT_DIR / "validation_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "classification_true",
                "classification_pred",
                "vote_agreement",
                "regression_true",
                "regression_pred",
            ]
        )
        writer.writerows(
            zip(
                first["ids"],
                true_cls,
                classification,
                agreement,
                true_reg,
                regression,
            )
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
