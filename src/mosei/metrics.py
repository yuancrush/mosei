from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error


def compute_metrics(
    classification_true: np.ndarray,
    classification_pred: np.ndarray,
    regression_true: np.ndarray,
    regression_pred: np.ndarray,
) -> Dict[str, object]:
    if np.std(regression_true) < 1e-12 or np.std(regression_pred) < 1e-12:
        pearson = 0.0
    else:
        pearson = float(pearsonr(regression_true, regression_pred).statistic)
    result: Dict[str, object] = {
        "accuracy": float(accuracy_score(classification_true, classification_pred)),
        "macro_f1": float(
            f1_score(classification_true, classification_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                classification_true,
                classification_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "mae": float(mean_absolute_error(regression_true, regression_pred)),
        "pearson": pearson,
        "confusion_matrix": confusion_matrix(
            classification_true, classification_pred, labels=[0, 1, 2]
        ).tolist(),
    }
    result["selection_score"] = float(
        result["macro_f1"] + result["pearson"] - result["mae"] / 3.0
    )
    return result
