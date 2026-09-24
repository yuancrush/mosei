from __future__ import annotations

"""用冻结模型与冻结解释协议生成附件4的正式预测和局部证据。"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate_explanation_methods import (
    AUDIO_SCALE,
    CLASSIFICATION_PATHS,
    CLASS_MODALITIES,
    REGRESSION_PATHS,
    REG_MODALITIES,
    VISION_SCALE,
    classification_outputs,
    load_model,
    mask_block,
    mask_modality,
    regression_outputs,
)
from mosei.data import MoseiDataModule, find_aligned_pickle


sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)

CLASS_NAMES = {0: "负向", 1: "中性", 2: "正向"}


def find_attachment4_aligned(root: Path) -> Path:
    candidates = [
        path
        for path in root.parent.glob("**/附件4-可解释专项视频样本与特征文件/对齐版本")
        if path.is_dir()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(f"附件4对齐版应唯一，实际为{len(candidates)}")
    return candidates[0]


def load_trusted_pickle(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    return obj["test"] if "test" in obj and isinstance(obj["test"], Mapping) else obj


def build_batch(
    samples: Sequence[Mapping[str, Any]],
    module: MoseiDataModule,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    text_bert = np.stack([np.asarray(sample["text_bert"]) for sample in samples])
    text_mask = text_bert[:, 1, :] > 0
    batch: Dict[str, torch.Tensor] = {
        "text_mask": torch.as_tensor(text_mask, dtype=torch.bool)
    }
    for modality in CLASS_MODALITIES:
        values = np.stack(
            [np.asarray(sample[modality], dtype=np.float32) for sample in samples]
        )
        mask = text_mask & np.isfinite(values).all(axis=-1) & np.any(values != 0, axis=-1)
        if modality in module.stats:
            values = (values - module.stats[modality].mean) / module.stats[modality].std
        values = np.where(mask[..., None], values, 0.0)
        batch[modality] = torch.as_tensor(values, dtype=torch.float32)
        batch[f"{modality}_mask"] = torch.as_tensor(mask, dtype=torch.bool)
    return {key: value.to(device) for key, value in batch.items()}


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def vote_agreement(seed_predictions: torch.Tensor) -> np.ndarray:
    values = seed_predictions.cpu().numpy()
    return np.asarray(
        [np.bincount(values[:, index], minlength=3).max() / values.shape[0] for index in range(values.shape[1])]
    )


def top_seed_agreement(seed_scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    results = []
    for sample_index in range(seed_scores.shape[1]):
        candidates = np.flatnonzero(valid[sample_index])
        if not len(candidates):
            results.append(0.0)
            continue
        choices = [
            int(candidates[np.argmax(seed_scores[seed, sample_index, candidates])])
            for seed in range(seed_scores.shape[0])
        ]
        results.append(np.bincount(choices).max() / len(choices))
    return np.asarray(results)


def block_validity(
    batch: Dict[str, torch.Tensor], modalities: Sequence[str], block_size: int
) -> np.ndarray:
    columns = []
    for modality in modalities:
        for start in range(0, 50, block_size):
            columns.append(
                batch[f"{modality}_mask"][:, start : start + block_size]
                .any(dim=1)
                .cpu()
                .numpy()
            )
    return np.stack(columns, axis=1)


def evidence_text_and_time(
    token_map: pd.DataFrame,
    sample: str,
    start: int,
    end: int,
    allow_exact_time: bool,
) -> dict:
    selected = token_map.loc[
        (token_map["sample"] == sample)
        & (token_map["token_index"] >= start)
        & (token_map["token_index"] < end)
        & (token_map["word_index"] >= 0)
    ].copy()
    words = []
    for word in selected["word"].fillna("").astype(str):
        if word and (not words or words[-1] != word):
            words.append(word)
    if selected.empty or not allow_exact_time:
        return {
            "evidence_words": " ".join(words),
            "start_seconds": "",
            "end_seconds": "",
            "mean_alignment_score": "",
        }
    return {
        "evidence_words": " ".join(words),
        "start_seconds": float(selected["start_seconds"].min()),
        "end_seconds": float(selected["end_seconds"].max()),
        "mean_alignment_score": float(selected["alignment_score"].mean()),
    }


def extract_frame(video: Path, seconds: float, output: Path) -> bool:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return False
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV在Windows下直接写含中文的绝对路径可能静默失败，改用编码后写字节。
    encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not encoded:
        return False
    buffer.tofile(str(output))
    return output.exists() and output.stat().st_size > 0


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="附件4冻结预测与反事实解释")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "exp024_attachment4_explanations",
    )
    args = parser.parse_args()
    data_dir = find_attachment4_aligned(PROJECT_ROOT)
    paths = sorted(data_dir.glob("*.pkl"))
    samples = [load_trusted_pickle(path) for path in paths]
    sample_ids = [path.stem for path in paths]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = MoseiDataModule(find_aligned_pickle(PROJECT_ROOT), batch_size=64)
    batch = build_batch(samples, module, device)
    class_models = [load_model(path, device) for path in CLASSIFICATION_PATHS]
    reg_models = [load_model(path, device) for path in REGRESSION_PATHS]

    base_class_prob, base_seed_pred, base_vote = classification_outputs(class_models, batch)
    base_mean_prob = base_class_prob.mean(dim=0)
    class_target_prob = base_mean_prob.gather(1, base_vote.unsqueeze(1)).squeeze(1)
    base_reg_seed = regression_outputs(reg_models, batch).clamp(-3.0, 3.0)
    base_reg = base_reg_seed.mean(dim=0)

    class_modality_delta = {}
    for modality in CLASS_MODALITIES:
        masked_prob, _, _ = classification_outputs(class_models, mask_modality(batch, modality))
        class_modality_delta[modality] = (
            class_target_prob
            - masked_prob.mean(dim=0).gather(1, base_vote.unsqueeze(1)).squeeze(1)
        ).cpu().numpy()
    reg_modality_delta = {}
    for modality in REG_MODALITIES:
        masked_reg = regression_outputs(reg_models, mask_modality(batch, modality)).mean(dim=0)
        reg_modality_delta[modality] = (base_reg - masked_reg).cpu().numpy()

    # 分类使用验证集冻结的5位置块。
    class_block_size = 5
    class_blocks = 50 // class_block_size
    class_seed_scores = np.zeros((3, len(samples), 3, class_blocks), dtype=np.float32)
    for modality_index, modality in enumerate(CLASS_MODALITIES):
        for block_index, start in enumerate(range(0, 50, class_block_size)):
            changed = mask_block(batch, modality, start, start + class_block_size)
            masked_prob, _, _ = classification_outputs(class_models, changed)
            gather = base_vote.view(1, -1, 1).expand(3, -1, 1)
            class_seed_scores[:, :, modality_index, block_index] = (
                base_class_prob.gather(2, gather).squeeze(2)
                - masked_prob.gather(2, gather).squeeze(2)
            ).cpu().numpy()
    class_scores = class_seed_scores.mean(axis=0)
    class_flat = class_scores.reshape(len(samples), -1)
    class_seed_flat = class_seed_scores.reshape(3, len(samples), -1)
    class_valid = block_validity(batch, CLASS_MODALITIES, class_block_size)

    # 回归使用验证集冻结的10位置块。
    reg_block_size = 10
    reg_blocks = 50 // reg_block_size
    reg_seed_signed = np.zeros((3, len(samples), 2, reg_blocks), dtype=np.float32)
    for modality_index, modality in enumerate(REG_MODALITIES):
        for block_index, start in enumerate(range(0, 50, reg_block_size)):
            changed = mask_block(batch, modality, start, start + reg_block_size)
            masked_reg = regression_outputs(reg_models, changed).clamp(-3.0, 3.0)
            reg_seed_signed[:, :, modality_index, block_index] = (
                base_reg_seed - masked_reg
            ).cpu().numpy()
    reg_scores = reg_seed_signed.mean(axis=0)
    reg_flat_abs = np.abs(reg_scores.reshape(len(samples), -1))
    reg_seed_flat_abs = np.abs(reg_seed_signed.reshape(3, len(samples), -1))
    reg_valid = block_validity(batch, REG_MODALITIES, reg_block_size)

    class_top_flat = np.asarray(
        [
            np.flatnonzero(valid)[np.argmax(scores[np.flatnonzero(valid)])]
            for scores, valid in zip(class_flat, class_valid)
        ]
    )
    reg_top_flat = np.asarray(
        [
            np.flatnonzero(valid)[np.argmax(scores[np.flatnonzero(valid)])]
            for scores, valid in zip(reg_flat_abs, reg_valid)
        ]
    )
    class_explanation_agreement = top_seed_agreement(class_seed_flat, class_valid)
    reg_explanation_agreement = top_seed_agreement(reg_seed_flat_abs, reg_valid)

    token_map = pd.read_csv(
        PROJECT_ROOT / "outputs" / "question3_stage_a_timeline" / "token_time_map.csv",
        dtype={"sample": str},
    )
    policy = pd.read_csv(
        PROJECT_ROOT / "reports" / "question3_stage_a" / "evidence_policy.csv",
        dtype={"sample": str},
    ).set_index("sample")

    prediction_rows = []
    evidence_rows = []
    frames_dir = args.output_dir / "keyframes"
    class_vote_agreement = vote_agreement(base_seed_pred)
    text_extent = batch["text_mask"].sum(dim=1).clamp_min(1).cpu().numpy()
    audio_coverage = batch["audio_mask"].sum(dim=1).cpu().numpy() / text_extent
    vision_coverage = batch["vision_mask"].sum(dim=1).cpu().numpy() / text_extent

    for index, sample_id in enumerate(sample_ids):
        class_deltas = np.asarray(
            [class_modality_delta[modality][index] for modality in CLASS_MODALITIES]
        )
        class_strength = np.abs(class_deltas)
        class_strength /= max(class_strength.sum(), 1e-12)
        reg_deltas = np.asarray(
            [reg_modality_delta[modality][index] for modality in REG_MODALITIES]
        )
        reg_strength = np.abs(reg_deltas)
        reg_strength /= max(reg_strength.sum(), 1e-12)
        class_primary = CLASS_MODALITIES[int(np.argmax(class_strength))]
        reg_primary = REG_MODALITIES[int(np.argmax(reg_strength))]
        class_id = int(base_vote[index].item())
        regression_value = float(base_reg[index].item())
        polarity_regression_conflict = bool(
            (class_id == 0 and regression_value > 0)
            or (class_id == 2 and regression_value < 0)
        )
        prediction_rows.append(
            {
                "sample": sample_id,
                "raw_text": str(samples[index]["raw_text"]),
                "classification_id": class_id,
                "classification_name": CLASS_NAMES[class_id],
                "classification_vote_agreement": float(class_vote_agreement[index]),
                "prob_negative": float(base_mean_prob[index, 0].item()),
                "prob_neutral": float(base_mean_prob[index, 1].item()),
                "prob_positive": float(base_mean_prob[index, 2].item()),
                "regression_score": regression_value,
                "emotion_intensity_abs": abs(regression_value),
                "regression_seed_std": float(base_reg_seed[:, index].std(unbiased=True).item()),
                "classification_regression_polarity_conflict": polarity_regression_conflict,
                "classification_primary_modality": class_primary,
                "classification_text_strength": float(class_strength[0]),
                "classification_audio_strength": float(class_strength[1]),
                "classification_vision_strength": float(class_strength[2]),
                "classification_text_signed_effect": float(class_deltas[0]),
                "classification_audio_signed_effect": float(class_deltas[1]),
                "classification_vision_signed_effect": float(class_deltas[2]),
                "regression_primary_modality": reg_primary,
                "regression_text_strength": float(reg_strength[0]),
                "regression_audio_strength": float(reg_strength[1]),
                "regression_text_signed_effect": float(reg_deltas[0]),
                "regression_audio_signed_effect": float(reg_deltas[1]),
                "classification_evidence_seed_agreement": float(
                    class_explanation_agreement[index]
                ),
                "regression_evidence_seed_agreement": float(reg_explanation_agreement[index]),
                "audio_coverage": float(audio_coverage[index]),
                "vision_coverage": float(vision_coverage[index]),
                "stage_a_restrictions": str(policy.loc[sample_id, "required_restrictions"]),
            }
        )

        allow_exact_time = (
            policy.loc[sample_id, "text_word_time_reliability"]
            != "不可作逐词精确定位"
        )
        model_visual_usable = bool(
            policy.loc[sample_id, "model_visual_contribution_usable"]
        )
        for modality_index, modality in enumerate(CLASS_MODALITIES):
            valid_indices = np.flatnonzero(
                class_valid[
                    index,
                    modality_index * class_blocks : (modality_index + 1) * class_blocks,
                ]
            )
            if not len(valid_indices) or (modality == "vision" and not model_visual_usable):
                continue
            local_scores = class_scores[index, modality_index]
            block_index = int(valid_indices[np.argmax(local_scores[valid_indices])])
            start = block_index * class_block_size
            end = start + class_block_size
            mapped = evidence_text_and_time(
                token_map, sample_id, start, end, allow_exact_time
            )
            frame_file = ""
            if modality == "vision" and mapped["start_seconds"] != "":
                midpoint = (mapped["start_seconds"] + mapped["end_seconds"]) / 2
                frame_path = frames_dir / f"{sample_id}_classification_vision.jpg"
                if extract_frame(data_dir / "videos" / f"{sample_id}.mp4", midpoint, frame_path):
                    frame_file = str(frame_path.relative_to(PROJECT_ROOT))
            evidence_rows.append(
                {
                    "sample": sample_id,
                    "task": "classification",
                    "modality": modality,
                    "block_index": block_index,
                    "start_position": start,
                    "end_position_exclusive": end,
                    "importance_signed": float(local_scores[block_index]),
                    "importance_abs": float(abs(local_scores[block_index])),
                    "seed_std": float(
                        np.std(class_seed_scores[:, index, modality_index, block_index])
                    ),
                    **mapped,
                    "keyframe_file": frame_file,
                    "evidence_restriction": str(policy.loc[sample_id, "required_restrictions"]),
                }
            )

        for modality_index, modality in enumerate(REG_MODALITIES):
            valid_indices = np.flatnonzero(
                reg_valid[
                    index,
                    modality_index * reg_blocks : (modality_index + 1) * reg_blocks,
                ]
            )
            if not len(valid_indices):
                continue
            local_scores = reg_scores[index, modality_index]
            block_index = int(valid_indices[np.argmax(np.abs(local_scores[valid_indices]))])
            start = block_index * reg_block_size
            end = start + reg_block_size
            mapped = evidence_text_and_time(
                token_map, sample_id, start, end, allow_exact_time
            )
            evidence_rows.append(
                {
                    "sample": sample_id,
                    "task": "regression",
                    "modality": modality,
                    "block_index": block_index,
                    "start_position": start,
                    "end_position_exclusive": end,
                    "importance_signed": float(local_scores[block_index]),
                    "importance_abs": float(abs(local_scores[block_index])),
                    "seed_std": float(
                        np.std(reg_seed_signed[:, index, modality_index, block_index])
                    ),
                    **mapped,
                    "keyframe_file": "",
                    "evidence_restriction": str(policy.loc[sample_id, "required_restrictions"]),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "predictions_and_contributions.csv", prediction_rows)
    write_csv(args.output_dir / "local_evidence.csv", evidence_rows)
    summary = {
        "sample_count": len(samples),
        "ground_truth_available": False,
        "attachment4_used_for_method_selection": False,
        "classification_method": "3-seed tri-modal output-residual majority vote",
        "classification_scales": {"audio": AUDIO_SCALE, "vision": VISION_SCALE},
        "regression_method": "3-seed Text+Audio mean",
        "classification_counts": {
            CLASS_NAMES[class_id]: int((base_vote.cpu().numpy() == class_id).sum())
            for class_id in CLASS_NAMES
        },
        "classification_primary_modality_counts": {
            modality: sum(
                row["classification_primary_modality"] == modality
                for row in prediction_rows
            )
            for modality in CLASS_MODALITIES
        },
        "regression_primary_modality_counts": {
            modality: sum(
                row["regression_primary_modality"] == modality
                for row in prediction_rows
            )
            for modality in REG_MODALITIES
        },
        "mean_vote_agreement": float(class_vote_agreement.mean()),
        "unanimous_classification_count": int((class_vote_agreement == 1.0).sum()),
        "mean_regression_seed_std": float(base_reg_seed.std(dim=0, unbiased=True).mean().item()),
        "classification_regression_polarity_conflict_samples": [
            row["sample"]
            for row in prediction_rows
            if row["classification_regression_polarity_conflict"]
        ],
        "mean_classification_evidence_seed_agreement": float(
            class_explanation_agreement.mean()
        ),
        "mean_regression_evidence_seed_agreement": float(reg_explanation_agreement.mean()),
        "keyframe_count": len(list(frames_dir.glob("*.jpg"))),
        "special_handling": {
            "15": "不输出逐词精确时间",
            "13": "不输出正式模型视觉贡献关键帧",
            "07_18": "只解释进入50-token接口的范围",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
