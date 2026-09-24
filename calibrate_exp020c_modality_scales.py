"""在验证集上校准EXP-020C音频/视觉残差的全局缩放系数。"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mosei.data import MoseiDataModule, find_aligned_pickle
from mosei.metrics import compute_metrics
from mosei.models import build_model
from mosei.utils import move_batch


CHECKPOINTS = [
    PROJECT_ROOT / "outputs" / run / "fusion_output_residual_bigru" / "best.pt"
    for run in (
        "exp020c_output_residual",
        "exp020c_output_residual_seed2027",
        "exp020c_output_residual_seed2028",
    )
]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp020c_scale_calibration"
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_models(device: torch.device) -> list[torch.nn.Module]:
    models = []
    for path in CHECKPOINTS:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = build_model(
            checkpoint["model_name"], checkpoint["hidden_dim"], checkpoint["dropout"]
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        models.append(model.eval())
    return models


def vote(predictions: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            int(np.argmax(np.bincount(column.astype(int), minlength=3)))
            for column in predictions.T
        ]
    )


@torch.inference_mode()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device)
    module = MoseiDataModule(find_aligned_pickle(PROJECT_ROOT), batch_size=128)
    rows = []
    for audio_scale in SCALES:
        for vision_scale in SCALES:
            for model in models:
                model.fusion.audio_scale = audio_scale
                model.fusion.vision_scale = vision_scale
            true, predictions = [], [[] for _ in models]
            for batch in module.loader("valid", shuffle=False):
                true.append(batch["classification_label"].numpy())
                batch = move_batch(batch, device)
                for index, model in enumerate(models):
                    predictions[index].append(
                        model(batch)["classification_logits"].argmax(dim=-1).cpu().numpy()
                    )
            y_true = np.concatenate(true)
            y_pred = vote(
                np.asarray([np.concatenate(parts) for parts in predictions])
            )
            metrics = compute_metrics(y_true, y_pred, y_true.astype(float), y_pred.astype(float))
            rows.append(
                {
                    "audio_scale": audio_scale,
                    "vision_scale": vision_scale,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                }
            )
    rows.sort(key=lambda row: (row["macro_f1"], row["accuracy"]), reverse=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "grid.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT_DIR / "best.json").open("w", encoding="utf-8") as handle:
        json.dump(rows[0], handle, ensure_ascii=False, indent=2)
    print(json.dumps(rows[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
