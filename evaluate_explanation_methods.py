from __future__ import annotations

"""在附件2验证集评估问题3的反事实解释与时间块遮挡解释。

解释算法只在有标签验证集上选择。附件4不参与阈值、块宽或方法选择。
分类采用问题2冻结的三种子三模态输出残差集成；回归采用冻结的三种子
Text+Audio专家均值。主要证据是输出反事实变化，不把注意力权重直接当结论。
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


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
REGRESSION_PATHS = [
    PROJECT_ROOT
    / "outputs"
    / f"exp014_text_audio_seed{seed}"
    / "text_audio_bigru"
    / "best.pt"
    for seed in (2026, 2027, 2028)
]
CLASS_MODALITIES = ("text", "audio", "vision")
REG_MODALITIES = ("text", "audio")
AUDIO_SCALE = 0.25
VISION_SCALE = 0.25


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        checkpoint["model_name"], checkpoint["hidden_dim"], checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(
        checkpoint["model_state"], strict=not checkpoint.get("partial_state", False)
    )
    if checkpoint["model_name"] == "fusion_output_residual_bigru":
        model.fusion.audio_scale = AUDIO_SCALE
        model.fusion.vision_scale = VISION_SCALE
    return model.eval()


def majority_vote(seed_predictions: torch.Tensor) -> torch.Tensor:
    # seed_predictions: [seed, batch]
    return torch.mode(seed_predictions, dim=0).values


@torch.inference_mode()
def classification_outputs(
    models: Sequence[torch.nn.Module], batch: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = torch.stack(
        [torch.softmax(model(batch)["classification_logits"], dim=-1) for model in models]
    )
    seed_predictions = probabilities.argmax(dim=-1)
    voted = majority_vote(seed_predictions)
    return probabilities, seed_predictions, voted


@torch.inference_mode()
def regression_outputs(
    models: Sequence[torch.nn.Module], batch: Dict[str, torch.Tensor]
) -> torch.Tensor:
    return torch.stack([model(batch)["regression"] for model in models])


def mask_block(
    batch: Dict[str, torch.Tensor], modality: str, start: int, end: int
) -> Dict[str, torch.Tensor]:
    changed = dict(batch)
    changed[f"{modality}_mask"] = batch[f"{modality}_mask"].clone()
    changed[f"{modality}_mask"][:, start:end] = False
    changed[modality] = batch[modality].clone()
    changed[modality][:, start:end] = 0.0
    return changed


def mask_modality(
    batch: Dict[str, torch.Tensor], modality: str
) -> Dict[str, torch.Tensor]:
    return mask_block(batch, modality, 0, batch[modality].shape[1])


def apply_selected_blocks(
    batch: Dict[str, torch.Tensor],
    selections: Sequence[Sequence[int]],
    modalities: Sequence[str],
    block_size: int,
    retain_only: bool,
) -> Dict[str, torch.Tensor]:
    changed = dict(batch)
    for modality in modalities:
        changed[f"{modality}_mask"] = (
            torch.zeros_like(batch[f"{modality}_mask"])
            if retain_only
            else batch[f"{modality}_mask"].clone()
        )
        changed[modality] = batch[modality].clone()
        if retain_only:
            changed[modality].zero_()
    modality_count = len(modalities)
    blocks_per_modality = batch[modalities[0]].shape[1] // block_size
    for sample_index, selected in enumerate(selections):
        for flat_index in selected:
            modality_index = int(flat_index) // blocks_per_modality
            block_index = int(flat_index) % blocks_per_modality
            if modality_index >= modality_count:
                continue
            modality = modalities[modality_index]
            start = block_index * block_size
            end = start + block_size
            if retain_only:
                original_mask = batch[f"{modality}_mask"][sample_index, start:end]
                changed[f"{modality}_mask"][sample_index, start:end] = original_mask
                changed[modality][sample_index, start:end] = batch[modality][
                    sample_index, start:end
                ]
            else:
                changed[f"{modality}_mask"][sample_index, start:end] = False
                changed[modality][sample_index, start:end] = 0.0
    return changed


def valid_block_matrix(
    batch: Dict[str, torch.Tensor], modalities: Sequence[str], block_size: int
) -> np.ndarray:
    available = []
    length = batch[modalities[0]].shape[1]
    for modality in modalities:
        for start in range(0, length, block_size):
            available.append(
                batch[f"{modality}_mask"][:, start : start + block_size]
                .any(dim=1)
                .cpu()
                .numpy()
            )
    return np.stack(available, axis=1)


def select_top_and_random(
    scores: np.ndarray,
    valid: np.ndarray,
    top_k: int,
    rng: np.random.Generator,
) -> Tuple[List[List[int]], List[List[int]]]:
    top, random = [], []
    for sample_scores, sample_valid in zip(scores, valid):
        candidates = np.flatnonzero(sample_valid)
        count = min(top_k, len(candidates))
        ranked = candidates[np.argsort(sample_scores[candidates])[::-1]][:count]
        sampled = rng.choice(candidates, size=count, replace=False) if count else []
        top.append([int(value) for value in ranked])
        random.append([int(value) for value in sampled])
    return top, random


def mean_pairwise_top1_agreement(seed_scores: np.ndarray, valid: np.ndarray) -> float:
    # seed_scores: [seed, sample, flat_block]
    agreements = []
    for sample_index, sample_valid in enumerate(valid):
        candidates = np.flatnonzero(sample_valid)
        if not len(candidates):
            continue
        top_indices = [
            int(candidates[np.argmax(seed_scores[seed, sample_index, candidates])])
            for seed in range(seed_scores.shape[0])
        ]
        pairs = [
            top_indices[left] == top_indices[right]
            for left in range(len(top_indices))
            for right in range(left + 1, len(top_indices))
        ]
        agreements.append(float(np.mean(pairs)))
    return float(np.mean(agreements))


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="验证问题3反事实与时间遮挡解释")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "exp023_explanation_validation",
    )
    args = parser.parse_args()
    if 50 % args.block_size:
        raise ValueError("block-size必须整除50")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_models = [load_model(path, device) for path in CLASSIFICATION_PATHS]
    reg_models = [load_model(path, device) for path in REGRESSION_PATHS]
    module = MoseiDataModule(
        find_aligned_pickle(PROJECT_ROOT), batch_size=args.batch_size
    )
    rng = np.random.default_rng(args.random_seed)
    blocks_per_modality = 50 // args.block_size

    sample_rows: List[dict] = []
    block_rows: List[dict] = []
    class_top_deletion, class_random_deletion = [], []
    class_top_retention, class_random_retention = [], []
    reg_top_deletion, reg_random_deletion = [], []
    class_seed_top_agreements = []
    reg_seed_top_agreements = []

    for batch_number, raw_batch in enumerate(module.loader("valid", shuffle=False), start=1):
        batch = move_batch(raw_batch, device)
        batch_size = len(raw_batch["id"])
        base_class_prob, base_seed_pred, base_vote = classification_outputs(
            class_models, batch
        )
        base_ensemble_prob = base_class_prob.mean(dim=0)
        target_prob = base_ensemble_prob.gather(1, base_vote.unsqueeze(1)).squeeze(1)
        base_reg = regression_outputs(reg_models, batch)
        base_reg_mean = base_reg.mean(dim=0)

        class_modality_delta: Dict[str, np.ndarray] = {}
        reg_modality_signed: Dict[str, np.ndarray] = {}
        for modality in CLASS_MODALITIES:
            masked_prob, _, _ = classification_outputs(
                class_models, mask_modality(batch, modality)
            )
            masked_target = masked_prob.mean(dim=0).gather(
                1, base_vote.unsqueeze(1)
            ).squeeze(1)
            class_modality_delta[modality] = (target_prob - masked_target).cpu().numpy()
        for modality in REG_MODALITIES:
            masked_reg = regression_outputs(reg_models, mask_modality(batch, modality))
            reg_modality_signed[modality] = (
                base_reg_mean - masked_reg.mean(dim=0)
            ).cpu().numpy()

        class_scores = np.zeros(
            (batch_size, len(CLASS_MODALITIES), blocks_per_modality), dtype=np.float32
        )
        class_seed_scores = np.zeros(
            (
                len(class_models),
                batch_size,
                len(CLASS_MODALITIES),
                blocks_per_modality,
            ),
            dtype=np.float32,
        )
        reg_signed_scores = np.zeros(
            (batch_size, len(REG_MODALITIES), blocks_per_modality), dtype=np.float32
        )
        reg_seed_abs_scores = np.zeros(
            (
                len(reg_models),
                batch_size,
                len(REG_MODALITIES),
                blocks_per_modality,
            ),
            dtype=np.float32,
        )

        for modality_index, modality in enumerate(CLASS_MODALITIES):
            for block_index, start in enumerate(range(0, 50, args.block_size)):
                changed = mask_block(batch, modality, start, start + args.block_size)
                masked_prob, _, _ = classification_outputs(class_models, changed)
                gather_index = base_vote.view(1, -1, 1).expand(
                    len(class_models), -1, 1
                )
                base_seed_target = base_class_prob.gather(2, gather_index).squeeze(2)
                masked_seed_target = masked_prob.gather(2, gather_index).squeeze(2)
                seed_delta = base_seed_target - masked_seed_target
                class_seed_scores[:, :, modality_index, block_index] = (
                    seed_delta.cpu().numpy()
                )
                class_scores[:, modality_index, block_index] = (
                    seed_delta.mean(dim=0).cpu().numpy()
                )

        for modality_index, modality in enumerate(REG_MODALITIES):
            for block_index, start in enumerate(range(0, 50, args.block_size)):
                changed = mask_block(batch, modality, start, start + args.block_size)
                masked_reg = regression_outputs(reg_models, changed)
                seed_signed = base_reg - masked_reg
                reg_seed_abs_scores[:, :, modality_index, block_index] = (
                    seed_signed.abs().cpu().numpy()
                )
                reg_signed_scores[:, modality_index, block_index] = (
                    seed_signed.mean(dim=0).cpu().numpy()
                )

        class_flat = class_scores.reshape(batch_size, -1)
        class_seed_flat = class_seed_scores.reshape(len(class_models), batch_size, -1)
        reg_flat_abs = np.abs(reg_signed_scores.reshape(batch_size, -1))
        reg_seed_flat = reg_seed_abs_scores.reshape(len(reg_models), batch_size, -1)
        class_valid = valid_block_matrix(batch, CLASS_MODALITIES, args.block_size)
        reg_valid = valid_block_matrix(batch, REG_MODALITIES, args.block_size)
        class_top, class_random = select_top_and_random(
            class_flat, class_valid, args.top_k, rng
        )
        reg_top, reg_random = select_top_and_random(
            reg_flat_abs, reg_valid, args.top_k, rng
        )
        class_seed_top_agreements.append(
            mean_pairwise_top1_agreement(class_seed_flat, class_valid)
        )
        reg_seed_top_agreements.append(
            mean_pairwise_top1_agreement(reg_seed_flat, reg_valid)
        )

        top_deleted_prob, _, _ = classification_outputs(
            class_models,
            apply_selected_blocks(
                batch, class_top, CLASS_MODALITIES, args.block_size, retain_only=False
            ),
        )
        random_deleted_prob, _, _ = classification_outputs(
            class_models,
            apply_selected_blocks(
                batch, class_random, CLASS_MODALITIES, args.block_size, retain_only=False
            ),
        )
        top_retained_prob, _, _ = classification_outputs(
            class_models,
            apply_selected_blocks(
                batch, class_top, CLASS_MODALITIES, args.block_size, retain_only=True
            ),
        )
        random_retained_prob, _, _ = classification_outputs(
            class_models,
            apply_selected_blocks(
                batch, class_random, CLASS_MODALITIES, args.block_size, retain_only=True
            ),
        )
        gather = base_vote.unsqueeze(1)
        top_deleted_target = top_deleted_prob.mean(dim=0).gather(1, gather).squeeze(1)
        random_deleted_target = random_deleted_prob.mean(dim=0).gather(1, gather).squeeze(1)
        top_retained_target = top_retained_prob.mean(dim=0).gather(1, gather).squeeze(1)
        random_retained_target = random_retained_prob.mean(dim=0).gather(1, gather).squeeze(1)

        top_deleted_reg = regression_outputs(
            reg_models,
            apply_selected_blocks(
                batch, reg_top, REG_MODALITIES, args.block_size, retain_only=False
            ),
        ).mean(dim=0)
        random_deleted_reg = regression_outputs(
            reg_models,
            apply_selected_blocks(
                batch, reg_random, REG_MODALITIES, args.block_size, retain_only=False
            ),
        ).mean(dim=0)

        class_top_batch = (target_prob - top_deleted_target).cpu().numpy()
        class_random_batch = (target_prob - random_deleted_target).cpu().numpy()
        class_top_retained_batch = top_retained_target.cpu().numpy()
        class_random_retained_batch = random_retained_target.cpu().numpy()
        reg_top_batch = (base_reg_mean - top_deleted_reg).abs().cpu().numpy()
        reg_random_batch = (base_reg_mean - random_deleted_reg).abs().cpu().numpy()
        class_top_deletion.extend(class_top_batch.tolist())
        class_random_deletion.extend(class_random_batch.tolist())
        class_top_retention.extend(class_top_retained_batch.tolist())
        class_random_retention.extend(class_random_retained_batch.tolist())
        reg_top_deletion.extend(reg_top_batch.tolist())
        reg_random_deletion.extend(reg_random_batch.tolist())

        true_cls = batch["classification_label"].cpu().numpy()
        true_reg = batch["regression_label"].cpu().numpy()
        ids = [str(value) for value in raw_batch["id"]]
        for sample_index, sample_id in enumerate(ids):
            top_class_flat = class_top[sample_index][0] if class_top[sample_index] else -1
            top_reg_flat = reg_top[sample_index][0] if reg_top[sample_index] else -1
            sample_rows.append(
                {
                    "id": sample_id,
                    "classification_true": int(true_cls[sample_index]),
                    "classification_pred": int(base_vote[sample_index].item()),
                    "classification_correct": bool(
                        true_cls[sample_index] == base_vote[sample_index].item()
                    ),
                    "classification_target_probability": float(
                        target_prob[sample_index].item()
                    ),
                    "regression_true": float(true_reg[sample_index]),
                    "regression_pred": float(base_reg_mean[sample_index].item()),
                    **{
                        f"classification_{modality}_ablation_delta": float(
                            class_modality_delta[modality][sample_index]
                        )
                        for modality in CLASS_MODALITIES
                    },
                    **{
                        f"regression_{modality}_ablation_signed_delta": float(
                            reg_modality_signed[modality][sample_index]
                        )
                        for modality in REG_MODALITIES
                    },
                    "classification_top_modality": (
                        CLASS_MODALITIES[top_class_flat // blocks_per_modality]
                        if top_class_flat >= 0
                        else ""
                    ),
                    "classification_top_block": (
                        top_class_flat % blocks_per_modality if top_class_flat >= 0 else -1
                    ),
                    "classification_top_deletion_delta": float(
                        class_top_batch[sample_index]
                    ),
                    "classification_random_deletion_delta": float(
                        class_random_batch[sample_index]
                    ),
                    "classification_top_retained_probability": float(
                        class_top_retained_batch[sample_index]
                    ),
                    "classification_random_retained_probability": float(
                        class_random_retained_batch[sample_index]
                    ),
                    "regression_top_modality": (
                        REG_MODALITIES[top_reg_flat // blocks_per_modality]
                        if top_reg_flat >= 0
                        else ""
                    ),
                    "regression_top_block": (
                        top_reg_flat % blocks_per_modality if top_reg_flat >= 0 else -1
                    ),
                    "regression_top_deletion_abs_delta": float(
                        reg_top_batch[sample_index]
                    ),
                    "regression_random_deletion_abs_delta": float(
                        reg_random_batch[sample_index]
                    ),
                }
            )

            for modality_index, modality in enumerate(CLASS_MODALITIES):
                for block_index in range(blocks_per_modality):
                    block_rows.append(
                        {
                            "id": sample_id,
                            "task": "classification",
                            "modality": modality,
                            "block_index": block_index,
                            "start_position": block_index * args.block_size,
                            "end_position_exclusive": (block_index + 1) * args.block_size,
                            "available": bool(
                                class_valid[
                                    sample_index,
                                    modality_index * blocks_per_modality + block_index,
                                ]
                            ),
                            "importance": float(
                                class_scores[sample_index, modality_index, block_index]
                            ),
                            "seed_std": float(
                                np.std(
                                    class_seed_scores[
                                        :, sample_index, modality_index, block_index
                                    ]
                                )
                            ),
                        }
                    )
            for modality_index, modality in enumerate(REG_MODALITIES):
                for block_index in range(blocks_per_modality):
                    block_rows.append(
                        {
                            "id": sample_id,
                            "task": "regression",
                            "modality": modality,
                            "block_index": block_index,
                            "start_position": block_index * args.block_size,
                            "end_position_exclusive": (block_index + 1) * args.block_size,
                            "available": bool(
                                reg_valid[
                                    sample_index,
                                    modality_index * blocks_per_modality + block_index,
                                ]
                            ),
                            "importance": float(
                                reg_signed_scores[
                                    sample_index, modality_index, block_index
                                ]
                            ),
                            "seed_std": float(
                                np.std(
                                    reg_seed_abs_scores[
                                        :, sample_index, modality_index, block_index
                                    ]
                                )
                            ),
                        }
                    )
        print(f"batch {batch_number}: 已解释{len(sample_rows)}/728条验证样本")

    sample_table = sample_rows
    class_top_array = np.asarray(class_top_deletion)
    class_random_array = np.asarray(class_random_deletion)
    class_top_retention_array = np.asarray(class_top_retention)
    class_random_retention_array = np.asarray(class_random_retention)
    reg_top_array = np.asarray(reg_top_deletion)
    reg_random_array = np.asarray(reg_random_deletion)

    summary = {
        "protocol": {
            "split": "official_valid_only",
            "sample_count": len(sample_rows),
            "block_size_positions": args.block_size,
            "top_k_blocks": args.top_k,
            "random_seed": args.random_seed,
            "attachment4_used_for_selection": False,
        },
        "classification_fidelity": {
            "top_deletion_probability_drop_mean": float(class_top_array.mean()),
            "random_deletion_probability_drop_mean": float(class_random_array.mean()),
            "top_minus_random_deletion": float(
                class_top_array.mean() - class_random_array.mean()
            ),
            "top_deletion_better_ratio": float(
                np.mean(class_top_array > class_random_array)
            ),
            "top_retained_probability_mean": float(class_top_retention_array.mean()),
            "random_retained_probability_mean": float(
                class_random_retention_array.mean()
            ),
            "top_minus_random_retention": float(
                class_top_retention_array.mean() - class_random_retention_array.mean()
            ),
        },
        "regression_fidelity": {
            "top_deletion_abs_change_mean": float(reg_top_array.mean()),
            "random_deletion_abs_change_mean": float(reg_random_array.mean()),
            "top_minus_random_deletion": float(
                reg_top_array.mean() - reg_random_array.mean()
            ),
            "top_deletion_better_ratio": float(
                np.mean(reg_top_array > reg_random_array)
            ),
        },
        "seed_stability": {
            "classification_top1_pairwise_agreement": float(
                np.mean(class_seed_top_agreements)
            ),
            "regression_top1_pairwise_agreement": float(
                np.mean(reg_seed_top_agreements)
            ),
        },
        "mean_modality_ablation": {
            key: float(np.mean([row[key] for row in sample_rows]))
            for key in sample_rows[0]
            if "ablation" in key
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "sample_explanations.csv", sample_rows)
    write_csv(args.output_dir / "block_importance.csv", block_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
