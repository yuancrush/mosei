from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mosei.metrics import compute_metrics


TEXT_PATHS = [
    PROJECT_ROOT / "outputs" / "exp001_text_bigru" / "text_bigru" / "valid_predictions.json",
    PROJECT_ROOT / "outputs" / "exp005_seed2027" / "text_bigru" / "valid_predictions.json",
    PROJECT_ROOT / "outputs" / "exp005_seed2028" / "text_bigru" / "valid_predictions.json",
]
TEXT_AUDIO_PATHS = [
    PROJECT_ROOT / "outputs" / f"exp014_text_audio_seed{seed}" / "text_audio_bigru" / "valid_predictions.json"
    for seed in (2026, 2027, 2028)
]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp015_multiseed_experts"


def read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def assert_aligned(runs: list[dict]) -> None:
    first = runs[0]
    for run in runs[1:]:
        for key in ("ids", "classification_true", "regression_true"):
            if key == "regression_true":
                if not np.allclose(first[key], run[key]):
                    raise ValueError("回归真值不一致")
            elif first[key] != run[key]:
                raise ValueError(f"{key}不一致")


def majority_vote(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    voted, agreement = [], []
    for column in predictions.T:
        counts = np.bincount(column.astype(int), minlength=3)
        voted.append(int(np.argmax(counts)))
        agreement.append(float(np.max(counts) / len(column)))
    return np.asarray(voted), np.asarray(agreement)


def main() -> None:
    text_runs = [read(path) for path in TEXT_PATHS]
    audio_runs = [read(path) for path in TEXT_AUDIO_PATHS]
    assert_aligned(text_runs + audio_runs)
    classification, agreement = majority_vote(
        np.asarray([run["classification_pred"] for run in text_runs])
    )
    true_cls = np.asarray(text_runs[0]["classification_true"])
    true_reg = np.asarray(text_runs[0]["regression_true"])
    candidates = {
        "text_regression_mean": np.mean(
            [run["regression_pred"] for run in text_runs], axis=0
        ),
        "text_audio_regression_mean": np.mean(
            [run["regression_pred"] for run in audio_runs], axis=0
        ),
    }
    summaries = []
    for name, regression in candidates.items():
        metrics = compute_metrics(true_cls, classification, true_reg, regression)
        summaries.append(
            {
                "classification": "text_bigru_majority_3seed",
                "regression": name,
                **{
                    key: metrics[key]
                    for key in (
                        "accuracy", "macro_f1", "weighted_f1", "mae", "pearson", "selection_score"
                    )
                },
                "mean_vote_agreement": float(np.mean(agreement)),
                "unanimous_ratio": float(np.mean(agreement == 1.0)),
            }
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)
    with (OUTPUT_DIR / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    chosen_regression = candidates["text_audio_regression_mean"]
    with (OUTPUT_DIR / "validation_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["id", "classification_true", "classification_pred", "vote_agreement", "regression_true", "regression_pred"]
        )
        writer.writerows(
            zip(
                text_runs[0]["ids"], true_cls, classification, agreement,
                true_reg, chosen_regression,
            )
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
