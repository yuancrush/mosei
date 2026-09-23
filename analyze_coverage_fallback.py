from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mosei.metrics import compute_metrics


TEXT_RUNS = [
    PROJECT_ROOT / "outputs" / "exp001_text_bigru" / "text_bigru" / "valid_predictions.json",
    PROJECT_ROOT / "outputs" / "exp005_seed2027" / "text_bigru" / "valid_predictions.json",
    PROJECT_ROOT / "outputs" / "exp005_seed2028" / "text_bigru" / "valid_predictions.json",
]
AUDIO_RUNS = [
    PROJECT_ROOT / "outputs" / f"exp014_text_audio_seed{seed}" / "text_audio_bigru" / "valid_predictions.json"
    for seed in (2026, 2027, 2028)
]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp016_attachment3_predictions"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    aligned_candidates = list(PROJECT_ROOT.parent.glob("**/aligned_50.pkl"))
    if len(aligned_candidates) != 1:
        raise FileNotFoundError("aligned_50.pkl必须唯一")
    with aligned_candidates[0].open("rb") as handle:
        valid = pickle.load(handle)["valid"]
    text_mask = np.asarray(valid["text_bert"])[:, 1, :] > 0
    audio_mask = text_mask & np.any(np.asarray(valid["audio"]) != 0, axis=-1)
    coverage = audio_mask.sum(axis=1) / text_mask.sum(axis=1)

    text_runs = [read_json(path) for path in TEXT_RUNS]
    audio_runs = [read_json(path) for path in AUDIO_RUNS]
    text_regression = np.mean([run["regression_pred"] for run in text_runs], axis=0)
    audio_regression = np.mean([run["regression_pred"] for run in audio_runs], axis=0)
    votes = np.asarray([run["classification_pred"] for run in text_runs])
    classification = np.asarray(
        [np.bincount(column, minlength=3).argmax() for column in votes.T]
    )
    true_cls = np.asarray(text_runs[0]["classification_true"])
    true_reg = np.asarray(text_runs[0]["regression_true"])

    rows = []
    for threshold in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01):
        use_text = coverage < threshold
        prediction = np.where(use_text, text_regression, audio_regression)
        metrics = compute_metrics(true_cls, classification, true_reg, prediction)
        rows.append(
            {
                "coverage_threshold": threshold,
                "fallback_samples": int(use_text.sum()),
                "macro_f1": metrics["macro_f1"],
                "mae": metrics["mae"],
                "pearson": metrics["pearson"],
                "selection_score": metrics["selection_score"],
            }
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "coverage_fallback_validation.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    correlation = spearmanr(coverage, np.abs(audio_regression - true_reg))
    summary = {
        "validation_audio_coverage_quantiles": np.quantile(
            coverage, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
        ).tolist(),
        "coverage_vs_absolute_error_spearman": {
            "rho": float(correlation.statistic),
            "pvalue": float(correlation.pvalue),
        },
        "best_tested_threshold_by_selection_score": max(
            rows, key=lambda row: row["selection_score"]
        ),
        "caution": "阈值0.8仅覆盖42个验证样本，且附件3覆盖率分布更低，当前只作为候选规则。",
    }
    with (OUTPUT_DIR / "coverage_fallback_analysis.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"thresholds": rows, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
