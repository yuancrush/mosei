from __future__ import annotations

"""在附件1与附件2重合的18条样本上做独立结构核验。

附件2使用不同的音频/视觉提取器，因此不能比较特征值；本脚本只比较文本
token接口及非零时序支持区间，避免把“维度相同”误当成“特征相同”。
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = left | right
    return float((left & right).sum() / union.sum()) if union.any() else 1.0


def main() -> None:
    manifest = pd.read_csv(ROOT / "reports/question1_audit/manifest.csv")
    manifest = manifest.loc[manifest["reference_match"]].copy()
    candidates = list(ROOT.parent.glob("**/附件2-数据集特征文件/aligned_50.pkl"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"aligned_50.pkl数量应为1，实际为{len(candidates)}")
    with candidates[0].open("rb") as handle:
        reference = pickle.load(handle)
    index = {
        str(sample_id): (split, position)
        for split, values in reference.items()
        for position, sample_id in enumerate(values["id"])
    }

    records = []
    for row in manifest.itertuples(index=False):
        split, position = index[row.sample_id]
        values = reference[split]
        path = ROOT / "outputs/question1_features/samples" / f"{row.video_id}__{int(row.clip_id)}.npz"
        with np.load(path) as generated:
            generated_bert = generated["text_bert"]
            valid = generated_bert[1].astype(bool)
            reference_bert = values["text_bert"][position]
            reference_audio_support = np.any(values["audio"][position] != 0, axis=1)
            reference_vision_support = np.any(values["vision"][position] != 0, axis=1)
            generated_audio_support = generated["audio_mask"]
            generated_vision_support = generated["vision_mask"]
            text_rmse = float(np.sqrt(np.mean(
                (generated["text"][valid] - values["text"][position][valid]) ** 2
            )))
        records.append({
            "sample_id": row.sample_id,
            "reference_split": split,
            "token_ids_equal": bool(np.array_equal(generated_bert[0], reference_bert[0])),
            "attention_mask_equal": bool(np.array_equal(generated_bert[1], reference_bert[1])),
            "token_type_ids_equal": bool(np.array_equal(generated_bert[2], reference_bert[2])),
            "text_feature_rmse": text_rmse,
            "audio_support_jaccard": jaccard(generated_audio_support, reference_audio_support),
            "vision_support_jaccard": jaccard(generated_vision_support, reference_vision_support),
        })

    output_dir = ROOT / "reports/question1_reference_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    summary = {
        "overlap_count": len(table),
        "token_ids_exact_count": int(table["token_ids_equal"].sum()),
        "attention_mask_exact_count": int(table["attention_mask_equal"].sum()),
        "token_type_ids_exact_count": int(table["token_type_ids_equal"].sum()),
        "mean_text_feature_rmse": float(table["text_feature_rmse"].mean()),
        "mean_audio_support_jaccard": float(table["audio_support_jaccard"].mean()),
        "mean_vision_support_jaccard": float(table["vision_support_jaccard"].mean()),
        "note": "音频/视觉提取器不同，Jaccard仅作时序支持诊断，不要求等于1。",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
