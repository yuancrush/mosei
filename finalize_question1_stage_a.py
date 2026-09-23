from __future__ import annotations

"""冻结问题1产物：恢复BERT padding隐状态，并补充各模态有效时长。

该脚本只更新 ``outputs/question1_features`` 中的派生产物，不修改官方附件。
重复执行是安全的；每个NPZ均先写临时文件再原子替换。
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent
MAX_LENGTH = 50


def safe_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def effective_duration(intervals: np.ndarray, mask: np.ndarray | None = None) -> float:
    lengths = np.maximum(intervals[:, 1] - intervals[:, 0], 0)
    if mask is not None:
        lengths = lengths[np.asarray(mask, dtype=bool)]
    return float(lengths.sum())


def main() -> None:
    feature_root = ROOT / "outputs/question1_features"
    log_path = feature_root / "extraction_log.csv"
    manifest = pd.read_csv(ROOT / "reports/question1_audit/manifest.csv")
    log = pd.read_csv(log_path)
    manifest_by_id = manifest.set_index("sample_id")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased", use_fast=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        "bert-base-uncased", local_files_only=True
    ).to(device).eval()

    padding_counts: list[int] = []
    padding_nonzero_counts: list[int] = []
    for index, row in log.iterrows():
        path = feature_root / str(row["output_file"])
        with np.load(path) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        text = str(arrays["raw_text"].item())
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = model(**{key: value.to(device) for key, value in encoded.items()})
            text_features = hidden.last_hidden_state[0].cpu().numpy().astype(np.float32)
        arrays["text"] = text_features

        intervals = arrays["word_intervals"]
        sample_id = str(row["sample_id"])
        video_duration = float(manifest_by_id.loc[sample_id, "duration_s"])
        durations = {
            "video_duration_s": video_duration,
            "text_effective_duration_s": effective_duration(intervals),
            "audio_effective_duration_s": effective_duration(intervals, arrays["word_audio_mask"]),
            "visual_effective_duration_s": effective_duration(intervals, arrays["word_vision_mask"]),
        }
        for name, value in durations.items():
            arrays[name] = np.asarray(value, dtype=np.float32)
            log.loc[index, name] = value

        attention = arrays["text_bert"][1].astype(bool)
        padding_counts.append(int((~attention).sum()))
        padding_nonzero_counts.append(int(np.any(text_features[~attention] != 0, axis=1).sum()))
        safe_save_npz(path, arrays)

    log.to_csv(log_path, index=False, encoding="utf-8-sig")
    summary_path = feature_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = {
        "sample_count": int(len(log)),
        "text_padding_policy": "保留BERT原始隐状态；由text_bert[1]排除padding",
        "samples_with_padding": int(sum(value > 0 for value in padding_counts)),
        "samples_without_padding": int(sum(value == 0 for value in padding_counts)),
        "all_padding_positions_preserve_bert_output": bool(all(
            nonzero == count
            for count, nonzero in zip(padding_counts, padding_nonzero_counts)
            if count > 0
        )),
        "mean_video_duration_s": float(log["video_duration_s"].mean()),
        "mean_text_effective_duration_s": float(log["text_effective_duration_s"].mean()),
        "mean_audio_effective_duration_s": float(log["audio_effective_duration_s"].mean()),
        "mean_visual_effective_duration_s": float(log["visual_effective_duration_s"].mean()),
    }
    summary.update(result)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_dir = ROOT / "reports/question1_stage_a"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "finalization_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
