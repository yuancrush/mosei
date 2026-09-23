from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mosei.metrics import compute_metrics


DEFAULT_CLASSIFICATION = (
    PROJECT_ROOT / "outputs" / "exp001_text_bigru" / "text_bigru" / "valid_predictions.json"
)
DEFAULT_REGRESSION = {
    "text_bigru": DEFAULT_CLASSIFICATION,
    "text_audio_bigru": PROJECT_ROOT
    / "outputs"
    / "exp002_pair_bigru"
    / "text_audio_bigru"
    / "valid_predictions.json",
    "uncertainty_aug": PROJECT_ROOT
    / "outputs"
    / "exp011_uncertainty_aug"
    / "fusion_uncertainty_bigru"
    / "valid_predictions.json",
}


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="组合分类专家与回归专家的验证集预测")
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "exp012_task_experts")
    args = parser.parse_args()
    classification = load(args.classification)
    rows = []
    for name, regression_path in DEFAULT_REGRESSION.items():
        regression = load(regression_path)
        if classification["ids"] != regression["ids"]:
            raise ValueError(f"{name}与分类专家的样本顺序不一致")
        if classification["classification_true"] != regression["classification_true"]:
            raise ValueError(f"{name}与分类专家的分类真值不一致")
        if not np.allclose(classification["regression_true"], regression["regression_true"]):
            raise ValueError(f"{name}与分类专家的回归真值不一致")
        metrics = compute_metrics(
            np.asarray(classification["classification_true"]),
            np.asarray(classification["classification_pred"]),
            np.asarray(regression["regression_true"]),
            np.asarray(regression["regression_pred"]),
        )
        rows.append(
            {
                "classification_expert": "text_bigru",
                "regression_expert": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "mae": metrics["mae"],
                "pearson": metrics["pearson"],
                "selection_score": metrics["selection_score"],
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"text_bigru + {row['regression_expert']}: "
            f"F1={row['macro_f1']:.4f} MAE={row['mae']:.4f} "
            f"r={row['pearson']:.4f} score={row['selection_score']:.4f}"
        )


if __name__ == "__main__":
    main()
