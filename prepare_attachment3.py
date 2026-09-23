from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModel


PROJECT_ROOT = Path(__file__).resolve().parent


def find_unique(pattern: str) -> Path:
    candidates = list(PROJECT_ROOT.parent.glob(pattern))
    if len(candidates) != 1:
        raise FileNotFoundError(f"{pattern} 应唯一匹配，实际为 {len(candidates)}: {candidates}")
    return candidates[0]


def load_trusted_pickle(path: Path):
    # 仅用于竞赛官方数据文件。
    with path.open("rb") as handle:
        return pickle.load(handle)


@torch.inference_mode()
def encode_text_bert(
    text_bert: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    for start in range(0, len(text_bert), batch_size):
        encoded = torch.as_tensor(text_bert[start : start + batch_size], dtype=torch.long)
        encoded = encoded.to(device)
        hidden = model(
            input_ids=encoded[:, 0],
            attention_mask=encoded[:, 1],
            token_type_ids=encoded[:, 2],
        ).last_hidden_state
        outputs.append(hidden.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def reconstruction_metrics(
    provided: np.ndarray,
    reconstructed: np.ndarray,
    text_mask: np.ndarray,
) -> Dict[str, float]:
    expected = provided[text_mask]
    actual = reconstructed[text_mask]
    difference = expected - actual
    cosine = np.sum(expected * actual, axis=1) / (
        np.linalg.norm(expected, axis=1) * np.linalg.norm(actual, axis=1) + 1e-12
    )
    return {
        "effective_vectors": int(len(expected)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "mae": float(np.mean(np.abs(difference))),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "cosine_mean": float(np.mean(cosine)),
        "cosine_min": float(np.min(cosine)),
        "pearson_flat": float(np.corrcoef(expected.ravel(), actual.ravel())[0, 1]),
    }


def label_semantics(data: dict) -> dict:
    result = {}
    for split_name in ("train", "valid", "test"):
        split = data[split_name]
        cls = np.asarray(split["classification_labels"], dtype=np.int64)
        reg = np.asarray(split["regression_labels"], dtype=np.float32)
        expected_cls = np.where(reg < 0, 0, np.where(reg == 0, 1, 2))
        if not np.array_equal(cls, expected_cls):
            raise ValueError(f"{split_name} 三分类标签与回归符号不一致")
        result[split_name] = {
            "negative_regression_class_0": int(np.sum((reg < 0) & (cls == 0))),
            "zero_regression_neutral_class_1": int(np.sum((reg == 0) & (cls == 1))),
            "positive_regression_class_2": int(np.sum((reg > 0) & (cls == 2))),
        }
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="验证并重建附件3的768维BERT文本特征")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "exp013_attachment3_preparation"
    )
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    aligned_path = find_unique("**/aligned_50.pkl")
    attachment_dir = find_unique("**/附件3-模态缺失特征样本/对齐版本")
    aligned = load_trusted_pickle(aligned_path)
    model = AutoModel.from_pretrained(
        args.model, local_files_only=not args.allow_download
    ).eval().to(device)

    valid_text_bert = np.asarray(aligned["valid"]["text_bert"])
    valid_reconstructed = encode_text_bert(
        valid_text_bert, model, device, args.batch_size
    )
    valid_mask = valid_text_bert[:, 1, :] > 0
    validation = reconstruction_metrics(
        np.asarray(aligned["valid"]["text"]), valid_reconstructed, valid_mask
    )

    files = sorted(attachment_dir.glob("*.pkl"))
    samples = [load_trusted_pickle(path)["test"] for path in files]
    text_bert = np.concatenate([np.asarray(sample["text_bert"]) for sample in samples])
    attachment3 = {
        "id": [path.stem for path in files],
        "text_bert": text_bert.astype(np.float32),
        "text": encode_text_bert(text_bert, model, device, args.batch_size),
        "audio": np.concatenate([np.asarray(sample["audio"]) for sample in samples]).astype(np.float32),
        "vision": np.concatenate([np.asarray(sample["vision"]) for sample in samples]).astype(np.float32),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "attachment3_aligned_features.pkl"
    with feature_path.open("wb") as handle:
        pickle.dump(attachment3, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "bert_model": args.model,
        "device": str(device),
        "validation_split_reconstruction": validation,
        "label_semantics": label_semantics(aligned),
        "attachment3": {
            "sample_count": len(files),
            "ids": attachment3["id"],
            "shapes": {
                key: list(value.shape)
                for key, value in attachment3.items()
                if isinstance(value, np.ndarray)
            },
            "generated_feature_file": str(feature_path.resolve()),
            "sha256": sha256(feature_path),
            "contains_labels": False,
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
