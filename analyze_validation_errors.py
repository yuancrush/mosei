from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


ROOT = Path(__file__).resolve().parent
PREDICTIONS = {
    2026: ROOT / "outputs/exp001_text_bigru/text_bigru/valid_predictions.json",
    2027: ROOT / "outputs/exp005_seed2027/text_bigru/valid_predictions.json",
    2028: ROOT / "outputs/exp005_seed2028/text_bigru/valid_predictions.json",
}
CLASS_NAMES = ["Negative", "Neutral", "Positive"]


def load_predictions(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    output_dir = ROOT / "outputs/exp006_error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    class_rows = []
    regression_rows = []
    prediction_frames = []

    for seed, path in PREDICTIONS.items():
        data = load_predictions(path)
        y_true = np.asarray(data["classification_true"])
        y_pred = np.asarray(data["classification_pred"])
        r_true = np.asarray(data["regression_true"], dtype=float)
        r_pred = np.asarray(data["regression_pred"], dtype=float)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], zero_division=0
        )
        for label, name in enumerate(CLASS_NAMES):
            class_rows.append(
                {
                    "seed": seed,
                    "class": name,
                    "precision": precision[label],
                    "recall": recall[label],
                    "f1": f1[label],
                    "support": support[label],
                }
            )
            mask = y_true == label
            regression_rows.append(
                {
                    "seed": seed,
                    "class": name,
                    "mae": np.abs(r_true[mask] - r_pred[mask]).mean(),
                    "bias_pred_minus_true": (r_pred[mask] - r_true[mask]).mean(),
                    "support": int(mask.sum()),
                }
            )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "id": data["ids"],
                    "true_class": y_true,
                    "true_regression": r_true,
                    f"pred_class_{seed}": y_pred,
                    f"pred_regression_{seed}": r_pred,
                }
            )
        )

    class_frame = pd.DataFrame(class_rows)
    regression_frame = pd.DataFrame(regression_rows)
    class_frame.to_csv(output_dir / "per_class_by_seed.csv", index=False, encoding="utf-8-sig")
    regression_frame.to_csv(
        output_dir / "regression_by_class_and_seed.csv", index=False, encoding="utf-8-sig"
    )
    class_summary = class_frame.groupby("class")[["precision", "recall", "f1"]].agg(["mean", "std"])
    class_summary.columns = [f"{metric}_{stat}" for metric, stat in class_summary.columns]
    class_summary.reset_index().to_csv(
        output_dir / "per_class_summary.csv", index=False, encoding="utf-8-sig"
    )
    regression_summary = regression_frame.groupby("class")[["mae", "bias_pred_minus_true"]].agg(["mean", "std"])
    regression_summary.columns = [f"{metric}_{stat}" for metric, stat in regression_summary.columns]
    regression_summary.reset_index().to_csv(
        output_dir / "regression_by_class_summary.csv", index=False, encoding="utf-8-sig"
    )

    merged = prediction_frames[0]
    for frame in prediction_frames[1:]:
        merged = merged.merge(frame.drop(columns=["true_class", "true_regression"]), on="id", validate="one_to_one")
    class_columns = [f"pred_class_{seed}" for seed in PREDICTIONS]
    reg_columns = [f"pred_regression_{seed}" for seed in PREDICTIONS]
    merged["wrong_seed_count"] = sum(merged[column] != merged["true_class"] for column in class_columns)
    merged["mean_regression_prediction"] = merged[reg_columns].mean(axis=1)
    merged["ensemble_absolute_error"] = np.abs(
        merged["mean_regression_prediction"] - merged["true_regression"]
    )
    merged.sort_values(
        ["wrong_seed_count", "ensemble_absolute_error"], ascending=False
    ).to_csv(output_dir / "hard_cases.csv", index=False, encoding="utf-8-sig")

    print("分类分项指标（三种子均值与样本标准差）：")
    print(class_summary.to_string())
    print("\n按真实类别划分的回归误差：")
    print(regression_summary.to_string())
    print("\n三个种子都分类错误的样本数：", int((merged["wrong_seed_count"] == 3).sum()))


if __name__ == "__main__":
    main()
