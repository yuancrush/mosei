"""对官方 test 做唯一一次冻结评估。

模型、尺度与温度均由验证集实验事先冻结。本脚本在输出目录留下完成标记，
若标记已存在会拒绝再次运行，避免无意中反复查看 test 并据此调参。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import apply_chinese_style
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mosei.data import MoseiDataModule, find_aligned_pickle
from mosei.models import build_model
from mosei.utils import move_batch


CLASSIFICATION_PATHS = [
    PROJECT_ROOT / "outputs" / run_name / "fusion_output_residual_bigru" / "best.pt"
    for run_name in (
        "exp020c_output_residual",
        "exp020c_output_residual_seed2027",
        "exp020c_output_residual_seed2028",
    )
]
REGRESSION_PATHS = [
    PROJECT_ROOT / "outputs" / f"exp014_text_audio_seed{seed}" / "text_audio_bigru" / "best.pt"
    for seed in (2026, 2027, 2028)
]
CALIBRATION_SUMMARY = PROJECT_ROOT / "outputs" / "exp028_uncertainty_calibration" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp030_final_test_once"
COMPLETE_MARKER = OUTPUT_DIR / "RUN_COMPLETE.json"
CLASS_NAMES = ["负向", "中性", "正向"]
AUDIO_SCALE = 0.25
VISION_SCALE = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="唯一一次官方 test 冻结评估")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="只检查冻结配置与文件，不读取数据、不运行 test",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path: Path, device: torch.device, classification: bool) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        checkpoint["model_name"], checkpoint["hidden_dim"], checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    if classification:
        model.fusion.audio_scale = AUDIO_SCALE
        model.fusion.vision_scale = VISION_SCALE
    return model.eval()


def majority_vote(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    voted, agreement = [], []
    for column in predictions.T:
        counts = np.bincount(column.astype(int), minlength=3)
        voted.append(int(np.argmax(counts)))
        agreement.append(float(np.max(counts) / len(column)))
    return np.asarray(voted), np.asarray(agreement)


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    prediction = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = prediction == y_true
    total = len(y_true)
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        chosen = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if index == 0:
            chosen |= confidence == 0.0
        if chosen.any():
            value += chosen.sum() / total * abs(correct[chosen].mean() - confidence[chosen].mean())
    return float(value)


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    apply_chinese_style()
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    figure, axis = plt.subplots(figsize=(5.8, 5.0))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks(range(3), CLASS_NAMES)
    axis.set_yticks(range(3), CLASS_NAMES)
    axis.set_xlabel("预测类别")
    axis.set_ylabel("真实类别")
    axis.set_title("官方测试集混淆矩阵")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def preflight() -> None:
    required = [*CLASSIFICATION_PATHS, *REGRESSION_PATHS, CALIBRATION_SUMMARY]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"冻结输入缺失：{missing}")
    with CALIBRATION_SUMMARY.open(encoding="utf-8") as handle:
        temperature = float(json.load(handle)["temperature"]["full_validation"])
    print(
        json.dumps(
            {
                "status": "preflight_passed",
                "data_accessed": False,
                "test_accessed": False,
                "classification_checkpoints": len(CLASSIFICATION_PATHS),
                "regression_checkpoints": len(REGRESSION_PATHS),
                "audio_scale": AUDIO_SCALE,
                "vision_scale": VISION_SCALE,
                "temperature": temperature,
                "complete_marker_exists": COMPLETE_MARKER.exists(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.preflight:
        preflight()
        return
    if COMPLETE_MARKER.exists():
        raise RuntimeError(
            f"官方 test 已完成过一次，拒绝重复运行。结果标记：{COMPLETE_MARKER}"
        )
    preflight()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with CALIBRATION_SUMMARY.open(encoding="utf-8") as handle:
        temperature = float(json.load(handle)["temperature"]["full_validation"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classification_models = [load_model(path, device, True) for path in CLASSIFICATION_PATHS]
    regression_models = [load_model(path, device, False) for path in REGRESSION_PATHS]
    module = MoseiDataModule(find_aligned_pickle(PROJECT_ROOT), batch_size=128)

    ids, y_cls_parts, y_reg_parts = [], [], []
    logits_parts = [[] for _ in classification_models]
    regression_parts = [[] for _ in regression_models]
    for batch in module.loader("test", shuffle=False):
        ids.extend(batch["id"])
        y_cls_parts.append(batch["classification_label"].numpy())
        y_reg_parts.append(batch["regression_label"].numpy())
        device_batch = move_batch(batch, device)
        for index, model in enumerate(classification_models):
            logits_parts[index].append(model(device_batch)["classification_logits"].cpu().numpy())
        for index, model in enumerate(regression_models):
            regression_parts[index].append(
                model(device_batch)["regression"].clamp(-3.0, 3.0).cpu().numpy()
            )

    y_cls = np.concatenate(y_cls_parts).astype(int)
    y_reg = np.concatenate(y_reg_parts).astype(np.float64)
    logits = np.asarray([np.concatenate(parts) for parts in logits_parts])
    seed_predictions = logits.argmax(axis=2)
    class_prediction, vote_agreement = majority_vote(seed_predictions)
    scaled_logits = logits / temperature
    scaled_logits = scaled_logits - scaled_logits.max(axis=2, keepdims=True)
    exp_logits = np.exp(scaled_logits)
    probabilities = (exp_logits / exp_logits.sum(axis=2, keepdims=True)).mean(axis=0)
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)

    regressions = np.asarray([np.concatenate(parts) for parts in regression_parts])
    regression_prediction = regressions.mean(axis=0)
    regression_std = regressions.std(axis=0, ddof=1)
    absolute_error = np.abs(y_reg - regression_prediction)

    report = classification_report(
        y_cls,
        class_prediction,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    errors = (class_prediction != y_cls).astype(int)
    entropy_auroc = float(roc_auc_score(errors, entropy)) if np.unique(errors).size == 2 else None
    reg_spearman = spearmanr(regression_std, absolute_error)
    summary = {
        "protocol": {
            "split": "official_test",
            "sample_count": len(ids),
            "run_count": 1,
            "selection_completed_before_test": True,
            "prediction_rule_changed_after_test": False,
        },
        "frozen_configuration": {
            "classification": "EXP020C三模态输出残差BiGRU，3种子多数投票",
            "audio_scale": AUDIO_SCALE,
            "vision_scale": VISION_SCALE,
            "regression": "Text+Audio-BiGRU，3种子算术平均",
            "uncertainty_temperature": temperature,
            "temperature_changes_predicted_class": False,
        },
        "classification": {
            "accuracy": float(accuracy_score(y_cls, class_prediction)),
            "macro_f1": float(f1_score(y_cls, class_prediction, average="macro")),
            "weighted_f1": float(f1_score(y_cls, class_prediction, average="weighted")),
            "per_class": {name: report[name] for name in CLASS_NAMES},
            "confusion_matrix": confusion_matrix(y_cls, class_prediction, labels=[0, 1, 2]).tolist(),
        },
        "classification_uncertainty": {
            "nll_temperature_scaled": float(log_loss(y_cls, probabilities, labels=[0, 1, 2])),
            "brier_temperature_scaled": float(
                np.mean(np.sum((probabilities - np.eye(3)[y_cls]) ** 2, axis=1))
            ),
            "ece_10bin_temperature_scaled": expected_calibration_error(y_cls, probabilities),
            "entropy_error_auroc": entropy_auroc,
            "mean_vote_agreement": float(vote_agreement.mean()),
            "unanimous_ratio": float(np.mean(vote_agreement == 1.0)),
        },
        "regression": {
            "mae": float(mean_absolute_error(y_reg, regression_prediction)),
            "pearson": float(pearsonr(y_reg, regression_prediction).statistic),
        },
        "regression_uncertainty": {
            "seed_std_error_spearman": float(reg_spearman.statistic),
            "seed_std_error_spearman_p": float(reg_spearman.pvalue),
            "mean_seed_std": float(regression_std.mean()),
        },
        "checkpoint_sha256": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256(path)
            for path in [*CLASSIFICATION_PATHS, *REGRESSION_PATHS]
        },
    }

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (OUTPUT_DIR / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id", "classification_true", "classification_pred", "vote_agreement",
                "prob_negative", "prob_neutral", "prob_positive", "entropy",
                "regression_true", "regression_pred", "regression_seed_std", "absolute_error",
            ]
        )
        writer.writerows(
            zip(
                ids, y_cls, class_prediction, vote_agreement,
                probabilities[:, 0], probabilities[:, 1], probabilities[:, 2], entropy,
                y_reg, regression_prediction, regression_std, absolute_error,
            )
        )
    save_confusion(y_cls, class_prediction, OUTPUT_DIR / "官方测试集混淆矩阵.png")
    marker = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "official_test_run_count": 1,
        "summary_file": "summary.json",
        "rerun_guard": "若此文件存在，脚本拒绝再次运行",
    }
    with COMPLETE_MARKER.open("w", encoding="utf-8") as handle:
        json.dump(marker, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
