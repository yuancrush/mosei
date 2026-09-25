from __future__ import annotations

"""评估最终集成在官方验证集上的概率校准与选择性风险。"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parent
CALIBRATION_INPUTS = (
    PROJECT_ROOT
    / "outputs"
    / "exp026_final_calibration"
    / "validation_calibration_inputs.npz"
)
REGRESSION_INPUTS = (
    PROJECT_ROOT
    / "outputs"
    / "exp017_ensemble_calibration"
    / "validation_ensemble_inputs.npz"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp028_uncertainty_calibration"
N_BINS = 10
RANDOM_SEED = 2026


def majority_vote(seed_predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    voted, agreement = [], []
    for column in seed_predictions.T:
        counts = np.bincount(column.astype(int), minlength=3)
        voted.append(int(np.argmax(counts)))
        agreement.append(float(np.max(counts) / len(column)))
    return np.asarray(voted), np.asarray(agreement)


def probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(logits / temperature, axis=-1).mean(axis=0)


def multiclass_nll(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0))))


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    target = np.eye(probs.shape[1])[y_true]
    return float(np.mean(np.sum((probs - target) ** 2, axis=1)))


def reliability_bins(
    y_true: np.ndarray, prediction: np.ndarray, confidence: np.ndarray
) -> tuple[list[dict], float]:
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    rows, ece = [], 0.0
    correct = prediction == y_true
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (confidence >= left) & (
            confidence <= right if index == N_BINS - 1 else confidence < right
        )
        count = int(selected.sum())
        if count:
            accuracy = float(correct[selected].mean())
            mean_confidence = float(confidence[selected].mean())
            ece += count / len(y_true) * abs(accuracy - mean_confidence)
        else:
            accuracy = mean_confidence = float("nan")
        rows.append(
            {
                "bin_left": left,
                "bin_right": right,
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
        )
    return rows, float(ece)


def choose_temperature(y_true: np.ndarray, logits: np.ndarray, indices: np.ndarray) -> float:
    result = minimize_scalar(
        lambda value: multiclass_nll(y_true[indices], probabilities(logits[:, indices], value)),
        bounds=(0.2, 5.0),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x)


def risk_coverage_rows(
    classification_error: np.ndarray,
    classification_uncertainty: np.ndarray,
    regression_error: np.ndarray,
    regression_uncertainty: np.ndarray,
) -> list[dict]:
    rows = []
    coverages = np.linspace(0.1, 1.0, 19)
    class_order = np.argsort(classification_uncertainty)
    reg_order = np.argsort(regression_uncertainty)
    for coverage in coverages:
        count = max(1, int(np.ceil(len(class_order) * coverage)))
        rows.append(
            {
                "coverage": float(coverage),
                "classification_error_rate": float(classification_error[class_order[:count]].mean()),
                "regression_mae": float(regression_error[reg_order[:count]].mean()),
            }
        )
    return rows


def area(rows: list[dict], key: str) -> float:
    x = np.asarray([row["coverage"] for row in rows])
    y = np.asarray([row[key] for row in rows])
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_figure(
    uncalibrated_bins: list[dict],
    calibrated_bins: list[dict],
    risk_rows: list[dict],
    path: Path,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].plot([0, 1], [0, 1], "--", color="#64748B", lw=1)
    for rows, label, color in (
        (uncalibrated_bins, "校准前", "#DC2626"),
        (calibrated_bins, "温度校准后", "#2563EB"),
    ):
        valid = [row for row in rows if row["count"] > 0]
        axes[0].plot(
            [row["mean_confidence"] for row in valid],
            [row["accuracy"] for row in valid],
            marker="o",
            color=color,
            label=label,
        )
    axes[0].set(xlabel="平均置信度", ylabel="实际准确率", title="分类可靠性图", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(fontsize=8)

    coverage = np.asarray([row["coverage"] for row in risk_rows]) * 100
    axes[1].plot(
        coverage,
        [row["classification_error_rate"] for row in risk_rows],
        marker="o",
        color="#7C3AED",
    )
    axes[1].set(xlabel="覆盖率（%）", ylabel="错误率", title="分类风险—覆盖曲线")
    axes[2].plot(
        coverage,
        [row["regression_mae"] for row in risk_rows],
        marker="o",
        color="#059669",
    )
    axes[2].set(xlabel="覆盖率（%）", ylabel="MAE", title="回归风险—覆盖曲线")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("最终集成的不确定性校准与选择性风险", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = np.load(CALIBRATION_INPUTS)
    regression = np.load(REGRESSION_INPUTS)
    y_cls = data["classification_true"].astype(int)
    logits = data["classification_logits"].astype(np.float64)
    y_reg = data["regression_true"].astype(np.float64)
    reg_by_seed = regression["text_audio_regression"].astype(np.float64)
    reg_pred = reg_by_seed.mean(axis=0)
    reg_std = reg_by_seed.std(axis=0)

    seed_prediction = logits.argmax(axis=-1)
    voted, vote_agreement = majority_vote(seed_prediction)
    uncalibrated_prob = probabilities(logits, 1.0)

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    crossfit_prob = np.zeros_like(uncalibrated_prob)
    fold_temperatures = []
    all_indices = np.arange(len(y_cls))
    for train_index, held_index in splitter.split(all_indices, y_cls):
        temperature = choose_temperature(y_cls, logits, train_index)
        fold_temperatures.append(temperature)
        crossfit_prob[held_index] = probabilities(logits[:, held_index], temperature)
    final_temperature = choose_temperature(y_cls, logits, all_indices)

    uncalibrated_confidence = uncalibrated_prob[np.arange(len(y_cls)), voted]
    calibrated_confidence = crossfit_prob[np.arange(len(y_cls)), voted]
    uncalibrated_bins, uncalibrated_ece = reliability_bins(
        y_cls, voted, uncalibrated_confidence
    )
    calibrated_bins, calibrated_ece = reliability_bins(
        y_cls, voted, calibrated_confidence
    )

    entropy = -np.sum(
        crossfit_prob * np.log(np.clip(crossfit_prob, 1e-12, 1.0)), axis=1
    )
    seed_probabilities = softmax(logits / np.mean(fold_temperatures), axis=-1)
    expected_entropy = -np.mean(
        np.sum(seed_probabilities * np.log(np.clip(seed_probabilities, 1e-12, 1.0)), axis=2),
        axis=0,
    )
    mutual_information = entropy - expected_entropy
    class_error = (voted != y_cls).astype(float)
    reg_error = np.abs(reg_pred - y_reg)
    risk_rows = risk_coverage_rows(class_error, entropy, reg_error, reg_std)

    vote_rows = []
    for agreement in sorted(np.unique(vote_agreement)):
        selected = vote_agreement == agreement
        vote_rows.append(
            {
                "vote_agreement": float(agreement),
                "count": int(selected.sum()),
                "accuracy": float(1.0 - class_error[selected].mean()),
                "mean_entropy": float(entropy[selected].mean()),
            }
        )

    summary = {
        "protocol": {
            "split": "official_valid_only",
            "sample_count": len(y_cls),
            "temperature_crossfit_folds": 5,
            "test_accessed": False,
        },
        "temperature": {
            "full_validation": final_temperature,
            "fold_values": fold_temperatures,
        },
        "classification_calibration": {
            "uncalibrated_nll": multiclass_nll(y_cls, uncalibrated_prob),
            "crossfit_calibrated_nll": multiclass_nll(y_cls, crossfit_prob),
            "uncalibrated_brier": brier_score(y_cls, uncalibrated_prob),
            "crossfit_calibrated_brier": brier_score(y_cls, crossfit_prob),
            "uncalibrated_ece": uncalibrated_ece,
            "crossfit_calibrated_ece": calibrated_ece,
            "entropy_error_auroc": float(roc_auc_score(class_error, entropy)),
            "mutual_information_error_auroc": float(
                roc_auc_score(class_error, mutual_information)
            ),
            "vote_disagreement_error_auroc": float(
                roc_auc_score(class_error, 1.0 - vote_agreement)
            ),
            "aurc_entropy": area(risk_rows, "classification_error_rate"),
        },
        "regression_uncertainty": {
            "seed_std_absolute_error_spearman": float(
                spearmanr(reg_std, reg_error).statistic
            ),
            "seed_std_absolute_error_spearman_p": float(
                spearmanr(reg_std, reg_error).pvalue
            ),
            "aurc_seed_std": area(risk_rows, "regression_mae"),
            "overall_mae": float(reg_error.mean()),
        },
        "vote_agreement_strata": vote_rows,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "reliability_uncalibrated.csv", uncalibrated_bins)
    write_csv(OUTPUT_DIR / "reliability_calibrated.csv", calibrated_bins)
    write_csv(OUTPUT_DIR / "risk_coverage.csv", risk_rows)
    write_csv(OUTPUT_DIR / "vote_agreement_strata.csv", vote_rows)
    create_figure(
        uncalibrated_bins,
        calibrated_bins,
        risk_rows,
        OUTPUT_DIR / "不确定性校准与风险覆盖.png",
    )
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
