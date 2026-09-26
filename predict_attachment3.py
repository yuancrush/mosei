from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mosei.data import MoseiDataModule, find_aligned_pickle
from mosei.models import build_model


FEATURE_PATH = (
    PROJECT_ROOT / "outputs" / "exp013_attachment3_preparation" / "attachment3_aligned_features.pkl"
)
TEXT_CHECKPOINTS = [
    PROJECT_ROOT / "outputs" / "exp001_text_bigru" / "text_bigru" / "best.pt",
    PROJECT_ROOT / "outputs" / "exp005_seed2027" / "text_bigru" / "best.pt",
    PROJECT_ROOT / "outputs" / "exp005_seed2028" / "text_bigru" / "best.pt",
]
TEXT_AUDIO_CHECKPOINTS = [
    PROJECT_ROOT / "outputs" / f"exp014_text_audio_seed{seed}" / "text_audio_bigru" / "best.pt"
    for seed in (2026, 2027, 2028)
]
OUTPUT_RESIDUAL_CHECKPOINTS = [
    PROJECT_ROOT
    / "outputs"
    / run_name
    / "fusion_output_residual_bigru"
    / "best.pt"
    for run_name in (
        "exp020c_output_residual",
        "exp020c_output_residual_seed2027",
        "exp020c_output_residual_seed2028",
    )
]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp020c_attachment3_predictions"
CLASS_NAMES = {0: "负向", 1: "中性", 2: "正向"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对附件3执行冻结模型推理")
    parser.add_argument(
        "--classification-preset",
        choices=("text", "output_residual"),
        default="output_residual",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def load_trusted_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        checkpoint["model_name"], checkpoint["hidden_dim"], checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval()


def majority_vote(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    voted, agreement = [], []
    for column in predictions.T:
        counts = np.bincount(column.astype(int), minlength=3)
        voted.append(int(np.argmax(counts)))
        agreement.append(float(np.max(counts) / len(column)))
    return np.asarray(voted), np.asarray(agreement)


def build_batch(features: dict, module: MoseiDataModule, device: torch.device) -> dict:
    text_mask = np.asarray(features["text_bert"])[:, 1, :] > 0
    batch = {"text_mask": torch.as_tensor(text_mask, dtype=torch.bool)}
    for modality in ("text", "audio", "vision"):
        values = np.asarray(features[modality], dtype=np.float32)
        mask = text_mask & np.isfinite(values).all(axis=-1) & np.any(values != 0, axis=-1)
        if modality in module.stats:
            values = (values - module.stats[modality].mean) / module.stats[modality].std
        values = np.where(mask[..., None], values, 0.0)
        batch[modality] = torch.as_tensor(values, dtype=torch.float32)
        batch[f"{modality}_mask"] = torch.as_tensor(mask, dtype=torch.bool)
    return {key: value.to(device) for key, value in batch.items()}


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not FEATURE_PATH.exists():
        raise FileNotFoundError("请先运行 prepare_attachment3.py 生成本地特征缓存")
    features = load_trusted_pickle(FEATURE_PATH)
    module = MoseiDataModule(find_aligned_pickle(PROJECT_ROOT), batch_size=64)
    batch = build_batch(features, module, device)

    class_predictions = []
    class_probabilities = []
    classification_paths = (
        OUTPUT_RESIDUAL_CHECKPOINTS
        if args.classification_preset == "output_residual"
        else TEXT_CHECKPOINTS
    )
    for path in classification_paths:
        model = load_model(path, device)
        if args.classification_preset == "output_residual":
            model.fusion.audio_scale = 0.25
            model.fusion.vision_scale = 0.25
        output = model(batch)
        probabilities = torch.softmax(output["classification_logits"], dim=-1).cpu().numpy()
        class_probabilities.append(probabilities)
        class_predictions.append(np.argmax(probabilities, axis=-1))
    classification, vote_agreement = majority_vote(np.asarray(class_predictions))
    mean_probability = np.mean(class_probabilities, axis=0)

    regressions = []
    for path in TEXT_AUDIO_CHECKPOINTS:
        output = load_model(path, device)(batch)
        regressions.append(output["regression"].clamp(-3.0, 3.0).cpu().numpy())
    regressions = np.asarray(regressions)
    regression_mean = regressions.mean(axis=0)
    regression_std = regressions.std(axis=0, ddof=1)

    text_extent = batch["text_mask"].sum(dim=1).clamp_min(1).cpu().numpy()
    audio_coverage = batch["audio_mask"].sum(dim=1).cpu().numpy() / text_extent
    vision_coverage = batch["vision_mask"].sum(dim=1).cpu().numpy() / text_extent
    rows = []
    for index, sample_id in enumerate(features["id"]):
        rows.append(
            {
                "sample_id": sample_id,
                "classification_id": int(classification[index]),
                "classification_name": CLASS_NAMES[int(classification[index])],
                "vote_agreement": float(vote_agreement[index]),
                "prob_negative": float(mean_probability[index, 0]),
                "prob_neutral": float(mean_probability[index, 1]),
                "prob_positive": float(mean_probability[index, 2]),
                "regression_score": float(regression_mean[index]),
                "regression_seed_std": float(regression_std[index]),
                "audio_coverage": float(audio_coverage[index]),
                "vision_coverage": float(vision_coverage[index]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "sample_count": len(rows),
        "ground_truth_available": False,
        "use_as_training_pseudo_labels": False,
        "classification_method": (
            "3-seed tri-modal output-residual BiGRU majority vote"
            if args.classification_preset == "output_residual"
            else "3-seed Text-BiGRU majority vote"
        ),
        "classification_audio_scale": (
            0.25 if args.classification_preset == "output_residual" else None
        ),
        "classification_vision_scale": (
            0.25 if args.classification_preset == "output_residual" else None
        ),
        "regression_method": "3-seed Text+Audio-BiGRU arithmetic mean",
        "class_mapping": CLASS_NAMES,
        "classification_counts": {
            CLASS_NAMES[class_id]: int(np.sum(classification == class_id))
            for class_id in (0, 1, 2)
        },
        "mean_vote_agreement": float(np.mean(vote_agreement)),
        "unanimous_ratio": float(np.mean(vote_agreement == 1.0)),
        "mean_regression_seed_std": float(np.mean(regression_std)),
        "mean_audio_coverage": float(np.mean(audio_coverage)),
        "mean_vision_coverage": float(np.mean(vision_coverage)),
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
