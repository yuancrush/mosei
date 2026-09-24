from __future__ import annotations

"""EXP-017：重建验证集概率并比较分类软投票与回归专家加权。"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mosei.data import MoseiDataModule, find_aligned_pickle
from mosei.metrics import compute_metrics
from mosei.models import build_model
from mosei.utils import move_batch


TEXT_CHECKPOINTS = [
    ROOT / "outputs/exp001_text_bigru/text_bigru/best.pt",
    ROOT / "outputs/exp005_seed2027/text_bigru/best.pt",
    ROOT / "outputs/exp005_seed2028/text_bigru/best.pt",
]
TEXT_AUDIO_CHECKPOINTS = [
    ROOT / f"outputs/exp014_text_audio_seed{seed}/text_audio_bigru/best.pt"
    for seed in (2026, 2027, 2028)
]
OUTPUT_DIR = ROOT / "outputs/exp017_ensemble_calibration"


@torch.inference_mode()
def predict_checkpoint(
    checkpoint_path: Path,
    module: MoseiDataModule,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(
        checkpoint["model_name"],
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    ids: list[str] = []
    logits: list[np.ndarray] = []
    regression: list[np.ndarray] = []
    true_cls: list[np.ndarray] = []
    true_reg: list[np.ndarray] = []
    for batch in module.loader("valid", shuffle=False):
        ids.extend(batch["id"])
        true_cls.append(batch["classification_label"].numpy())
        true_reg.append(batch["regression_label"].numpy())
        outputs = model(move_batch(batch, device))
        logits.append(outputs["classification_logits"].cpu().numpy())
        regression.append(outputs["regression"].clamp(-3.0, 3.0).cpu().numpy())
    return {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "seed": int(checkpoint["seed"]),
        "ids": ids,
        "logits": np.concatenate(logits),
        "regression": np.concatenate(regression),
        "true_cls": np.concatenate(true_cls),
        "true_reg": np.concatenate(true_reg),
    }


def assert_aligned(runs: list[dict[str, object]]) -> None:
    first = runs[0]
    for run in runs[1:]:
        if first["ids"] != run["ids"]:
            raise ValueError("验证集样本顺序不一致")
        if not np.array_equal(first["true_cls"], run["true_cls"]):
            raise ValueError("分类真值不一致")
        if not np.allclose(first["true_reg"], run["true_reg"]):
            raise ValueError("回归真值不一致")


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = MoseiDataModule(
        find_aligned_pickle(ROOT),
        batch_size=128,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    text_runs = [predict_checkpoint(path, module, device) for path in TEXT_CHECKPOINTS]
    audio_runs = [predict_checkpoint(path, module, device) for path in TEXT_AUDIO_CHECKPOINTS]
    assert_aligned(text_runs + audio_runs)

    logits = np.stack([run["logits"] for run in text_runs])
    probabilities = softmax(logits)
    hard = np.stack([values.argmax(axis=-1) for values in logits])
    majority = np.asarray([
        np.bincount(column, minlength=3).argmax() for column in hard.T
    ])
    classification_candidates = {
        "硬投票": majority,
        "概率算术平均": probabilities.mean(axis=0).argmax(axis=-1),
        "对数概率平均": np.log(np.clip(probabilities, 1e-8, 1.0)).mean(axis=0).argmax(axis=-1),
    }
    text_regression = np.mean([run["regression"] for run in text_runs], axis=0)
    audio_regression = np.mean([run["regression"] for run in audio_runs], axis=0)
    regression_candidates = {
        f"文本权重{1-alpha:.2f}_文本音频权重{alpha:.2f}": (
            (1 - alpha) * text_regression + alpha * audio_regression
        )
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0)
    }
    true_cls = np.asarray(text_runs[0]["true_cls"])
    true_reg = np.asarray(text_runs[0]["true_reg"])
    rows: list[dict[str, object]] = []
    for cls_name, cls_pred in classification_candidates.items():
        for reg_name, reg_pred in regression_candidates.items():
            metrics = compute_metrics(true_cls, cls_pred, true_reg, reg_pred)
            rows.append({
                "classification_strategy": cls_name,
                "regression_strategy": reg_name,
                **{key: float(metrics[key]) for key in (
                    "accuracy", "macro_f1", "weighted_f1", "mae", "pearson", "selection_score"
                )},
            })
    best_cls = max(
        rows,
        key=lambda row: (row["macro_f1"], row["accuracy"]),
    )["classification_strategy"]
    cls_rows = [row for row in rows if row["classification_strategy"] == best_cls]
    # 分类固定后，以工程一直使用的综合选择分数决定回归权重；
    # 它同时奖励Pearson并惩罚MAE，避免只追逐其中一个指标。
    best = max(cls_rows, key=lambda row: row["selection_score"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "protocol": "仅使用官方valid；附件3/4未参与选择",
        "text_checkpoints": [run["checkpoint"] for run in text_runs],
        "text_audio_checkpoints": [run["checkpoint"] for run in audio_runs],
        "candidate_count": len(rows),
        "best": best,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        OUTPUT_DIR / "validation_ensemble_inputs.npz",
        ids=np.asarray(text_runs[0]["ids"]),
        classification_true=true_cls,
        regression_true=true_reg,
        text_logits=logits,
        text_probabilities=probabilities,
        text_regression=np.stack([run["regression"] for run in text_runs]),
        text_audio_regression=np.stack([run["regression"] for run in audio_runs]),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
