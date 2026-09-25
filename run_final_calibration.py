from __future__ import annotations

"""只在官方验证集上进行低维分类/回归校准并给出统计晋级判断。

分类只调整一个中性类logit偏置；回归只调整文本与文本+音频专家的线性配比。
脚本不创建test loader，避免在最终冻结前意外读取测试集。
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mosei.data import MoseiDataModule, find_aligned_pickle
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
REGRESSION_INPUTS = (
    PROJECT_ROOT
    / "outputs"
    / "exp017_ensemble_calibration"
    / "validation_ensemble_inputs.npz"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp026_final_calibration"
AUDIO_SCALE = 0.25
VISION_SCALE = 0.25
NEUTRAL_GRID = np.round(np.linspace(0.0, 1.5, 31), 6)
REGRESSION_ALPHA_GRID = np.round(np.linspace(0.0, 1.0, 21), 6)
BOOTSTRAP_REPEATS = 5000
RANDOM_SEED = 2026


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        checkpoint["model_name"], checkpoint["hidden_dim"], checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.fusion.audio_scale = AUDIO_SCALE
    model.fusion.vision_scale = VISION_SCALE
    return model.eval()


def majority_vote(seed_predictions: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            int(np.argmax(np.bincount(column.astype(int), minlength=3)))
            for column in seed_predictions.T
        ]
    )


def predict_with_neutral_penalty(logits: np.ndarray, penalty: float) -> np.ndarray:
    adjusted = logits.copy()
    adjusted[:, :, 1] -= penalty
    return majority_vote(adjusted.argmax(axis=-1))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "pearson": float(pearsonr(y_true, y_pred).statistic),
    }


def choose_penalty(y_true: np.ndarray, logits: np.ndarray, indices: np.ndarray) -> float:
    scored = []
    for penalty in NEUTRAL_GRID:
        pred = predict_with_neutral_penalty(logits[:, indices], float(penalty))
        score = f1_score(y_true[indices], pred, average="macro", zero_division=0)
        scored.append((float(score), -float(penalty), float(penalty)))
    return max(scored)[2]


def choose_alpha(
    y_true: np.ndarray,
    text_pred: np.ndarray,
    text_audio_pred: np.ndarray,
    indices: np.ndarray,
) -> float:
    scored = []
    for alpha in REGRESSION_ALPHA_GRID:
        pred = (1.0 - alpha) * text_pred[indices] + alpha * text_audio_pred[indices]
        mae = float(np.mean(np.abs(y_true[indices] - pred)))
        pearson = float(pearsonr(y_true[indices], pred).statistic)
        # 主目标为MAE；极小的MAE并列时优先Pearson，再优先更简单的0.5附近配比。
        scored.append((-mae, pearson, -abs(float(alpha) - 0.5), float(alpha)))
    return max(scored)[3]


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


@torch.inference_mode()
def collect_validation_logits() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(path, device) for path in CLASSIFICATION_PATHS]
    module = MoseiDataModule(find_aligned_pickle(PROJECT_ROOT), batch_size=128)
    logits_parts = [[] for _ in models]
    labels, ids = [], []
    for raw_batch in module.loader("valid", shuffle=False):
        labels.append(raw_batch["classification_label"].numpy())
        ids.extend(str(value) for value in raw_batch["id"])
        batch = move_batch(raw_batch, device)
        for index, model in enumerate(models):
            logits_parts[index].append(
                model(batch)["classification_logits"].cpu().numpy()
            )
    return (
        np.asarray(ids),
        np.concatenate(labels),
        np.asarray([np.concatenate(parts) for parts in logits_parts]),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ids, y_cls, logits = collect_validation_logits()
    regression = np.load(REGRESSION_INPUTS)
    if not np.array_equal(ids.astype(str), regression["ids"].astype(str)):
        raise ValueError("分类与回归验证样本顺序不一致")
    if not np.array_equal(y_cls, regression["classification_true"]):
        raise ValueError("验证集分类真值不一致")

    y_reg = regression["regression_true"].astype(np.float64)
    text_reg = regression["text_regression"].mean(axis=0).astype(np.float64)
    text_audio_reg = regression["text_audio_regression"].mean(axis=0).astype(np.float64)

    baseline_cls = predict_with_neutral_penalty(logits, 0.0)
    baseline_reg = text_audio_reg
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    crossfit_cls = np.empty_like(y_cls)
    crossfit_reg = np.empty_like(y_reg)
    fold_rows = []
    fold_penalties, fold_alphas = [], []
    all_indices = np.arange(len(y_cls))
    for fold, (train_index, held_index) in enumerate(
        splitter.split(all_indices, y_cls), start=1
    ):
        penalty = choose_penalty(y_cls, logits, train_index)
        alpha = choose_alpha(y_reg, text_reg, text_audio_reg, train_index)
        fold_penalties.append(penalty)
        fold_alphas.append(alpha)
        crossfit_cls[held_index] = predict_with_neutral_penalty(
            logits[:, held_index], penalty
        )
        crossfit_reg[held_index] = (
            (1.0 - alpha) * text_reg[held_index]
            + alpha * text_audio_reg[held_index]
        )
        fold_cls = classification_metrics(y_cls[held_index], crossfit_cls[held_index])
        fold_reg = regression_metrics(y_reg[held_index], crossfit_reg[held_index])
        fold_rows.append(
            {
                "fold": fold,
                "held_count": len(held_index),
                "neutral_penalty": penalty,
                "regression_text_audio_alpha": alpha,
                **fold_cls,
                **fold_reg,
            }
        )

    final_penalty = choose_penalty(y_cls, logits, all_indices)
    final_alpha = choose_alpha(y_reg, text_reg, text_audio_reg, all_indices)
    calibrated_cls = predict_with_neutral_penalty(logits, final_penalty)
    calibrated_reg = (1.0 - final_alpha) * text_reg + final_alpha * text_audio_reg

    class_grid_rows = []
    for penalty in NEUTRAL_GRID:
        metrics = classification_metrics(
            y_cls, predict_with_neutral_penalty(logits, float(penalty))
        )
        class_grid_rows.append({"neutral_penalty": float(penalty), **metrics})
    regression_grid_rows = []
    for alpha in REGRESSION_ALPHA_GRID:
        pred = (1.0 - alpha) * text_reg + alpha * text_audio_reg
        regression_grid_rows.append(
            {"text_audio_alpha": float(alpha), **regression_metrics(y_reg, pred)}
        )

    rng = np.random.default_rng(RANDOM_SEED)
    class_differences = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    mae_differences = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    pearson_differences = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    for repeat in range(BOOTSTRAP_REPEATS):
        sample = rng.integers(0, len(y_cls), size=len(y_cls))
        class_differences[repeat] = f1_score(
            y_cls[sample], crossfit_cls[sample], average="macro", zero_division=0
        ) - f1_score(
            y_cls[sample], baseline_cls[sample], average="macro", zero_division=0
        )
        mae_differences[repeat] = np.mean(
            np.abs(y_reg[sample] - crossfit_reg[sample])
            - np.abs(y_reg[sample] - baseline_reg[sample])
        )
        pearson_differences[repeat] = pearsonr(
            y_reg[sample], crossfit_reg[sample]
        ).statistic - pearsonr(y_reg[sample], baseline_reg[sample]).statistic

    baseline_cls_metrics = classification_metrics(y_cls, baseline_cls)
    crossfit_cls_metrics = classification_metrics(y_cls, crossfit_cls)
    calibrated_cls_metrics = classification_metrics(y_cls, calibrated_cls)
    baseline_reg_metrics = regression_metrics(y_reg, baseline_reg)
    crossfit_reg_metrics = regression_metrics(y_reg, crossfit_reg)
    calibrated_reg_metrics = regression_metrics(y_reg, calibrated_reg)

    classification_accept = bool(
        crossfit_cls_metrics["macro_f1"] - baseline_cls_metrics["macro_f1"] >= 0.003
        and percentile_interval(class_differences)[0] > 0.0
    )
    regression_accept = bool(
        crossfit_reg_metrics["mae"] < baseline_reg_metrics["mae"]
        and percentile_interval(mae_differences)[1] < 0.0
        and percentile_interval(pearson_differences)[0] >= -0.002
    )

    summary = {
        "protocol": {
            "split": "official_valid_only",
            "sample_count": len(y_cls),
            "folds": 5,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "test_accessed": False,
        },
        "classification": {
            "baseline": baseline_cls_metrics,
            "cross_fitted": crossfit_cls_metrics,
            "full_validation_selected": calibrated_cls_metrics,
            "selected_neutral_penalty": final_penalty,
            "fold_selected_penalties": fold_penalties,
            "crossfit_macro_f1_difference": crossfit_cls_metrics["macro_f1"]
            - baseline_cls_metrics["macro_f1"],
            "bootstrap_macro_f1_difference_ci95": percentile_interval(
                class_differences
            ),
            "accept_update": classification_accept,
        },
        "regression": {
            "baseline": baseline_reg_metrics,
            "cross_fitted": crossfit_reg_metrics,
            "full_validation_selected": calibrated_reg_metrics,
            "selected_text_audio_alpha": final_alpha,
            "fold_selected_alphas": fold_alphas,
            "crossfit_mae_difference": crossfit_reg_metrics["mae"]
            - baseline_reg_metrics["mae"],
            "crossfit_pearson_difference": crossfit_reg_metrics["pearson"]
            - baseline_reg_metrics["pearson"],
            "bootstrap_mae_difference_ci95": percentile_interval(mae_differences),
            "bootstrap_pearson_difference_ci95": percentile_interval(
                pearson_differences
            ),
            "accept_update": regression_accept,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "classification_grid.csv", class_grid_rows)
    write_csv(OUTPUT_DIR / "regression_grid.csv", regression_grid_rows)
    write_csv(OUTPUT_DIR / "crossfit_folds.csv", fold_rows)
    write_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        [
            {
                "id": sample_id,
                "classification_true": int(cls_true),
                "classification_baseline": int(cls_base),
                "classification_crossfit": int(cls_crossfit),
                "classification_calibrated": int(cls_calibrated),
                "regression_true": float(reg_true),
                "regression_baseline": float(reg_base),
                "regression_crossfit": float(reg_crossfit),
                "regression_calibrated": float(reg_calibrated),
            }
            for sample_id, cls_true, cls_base, cls_crossfit, cls_calibrated,
            reg_true, reg_base, reg_crossfit, reg_calibrated in zip(
                ids,
                y_cls,
                baseline_cls,
                crossfit_cls,
                calibrated_cls,
                y_reg,
                baseline_reg,
                crossfit_reg,
                calibrated_reg,
            )
        ],
    )
    np.savez_compressed(
        OUTPUT_DIR / "validation_calibration_inputs.npz",
        ids=ids,
        classification_true=y_cls,
        classification_logits=logits,
        classification_baseline=baseline_cls,
        classification_crossfit=crossfit_cls,
        classification_calibrated=calibrated_cls,
        regression_true=y_reg,
        regression_text=text_reg,
        regression_text_audio=text_audio_reg,
        regression_baseline=baseline_reg,
        regression_crossfit=crossfit_reg,
        regression_calibrated=calibrated_reg,
    )
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
