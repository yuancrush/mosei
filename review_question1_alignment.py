from __future__ import annotations

"""用独立Whisper ASR复核低MMS强制对齐置信度样本。

ASR结果只用于质量分层，不替换官方转写，也不把无标签数据用于情感训练。
"""

import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import pipeline

from extract_question1_features import PROJECT_ROOT, extract_wav, find_unique


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower().replace("’", "'"))


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_word in enumerate(left, start=1):
        current = [i]
        for j, right_word in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + int(left_word != right_word),
            ))
        previous = current
    return previous[-1]


def safe_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    model_name = "openai/whisper-base.en"
    feature_root = PROJECT_ROOT / "outputs/question1_features"
    log = pd.read_csv(feature_root / "extraction_log.csv")
    low = log.loc[log["mean_alignment_score"] < 0.6].copy()
    manifest = pd.read_csv(PROJECT_ROOT / "reports/question1_audit/manifest.csv")
    manifest_by_id = manifest.set_index("sample_id")
    data_dir = find_unique(
        "**/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条",
        "附件1视频目录",
    )
    output_dir = PROJECT_ROOT / "reports/question1_stage_a"
    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "低对齐置信度样本ASR复核.csv"
    cached: dict[str, dict[str, object]] = {}
    if review_path.exists():
        cached_table = pd.read_csv(review_path)
        cached = {
            str(row["样本编号"]): row.to_dict()
            for _, row in cached_table.iterrows()
        }

    missing_ids = [str(value) for value in low["sample_id"] if str(value) not in cached]
    recognizer = None
    if missing_ids:
        device = 0 if torch.cuda.is_available() else -1
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        recognizer = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            torch_dtype=dtype,
        )

    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="q1_asr_review_") as temp_dir:
        for row in low.itertuples(index=False):
            if row.sample_id in cached:
                records.append(cached[row.sample_id])
                continue
            manifest_row = manifest_by_id.loc[row.sample_id]
            wav_path = Path(temp_dir) / f"{len(records):03d}.wav"
            extract_wav(data_dir / Path(manifest_row["relative_path"]), wav_path, 16000)
            waveform, sample_rate = sf.read(wav_path, dtype="float32")
            assert recognizer is not None
            result = recognizer(
                {"array": waveform, "sampling_rate": sample_rate},
            )
            hypothesis = str(result["text"]).strip()
            reference_words = normalize_words(str(manifest_row["text"]))
            hypothesis_words = normalize_words(hypothesis)
            wer = edit_distance(reference_words, hypothesis_words) / max(len(reference_words), 1)
            path = feature_root / str(row.output_file)
            with np.load(path) as data:
                intervals = data["word_intervals"]
                temporal_valid = bool(
                    np.all(intervals[:, 0] <= intervals[:, 1])
                    and np.all(intervals[1:, 0] >= intervals[:-1, 0])
                    and intervals[0, 0] >= -1e-6
                    and intervals[-1, 1] <= float(manifest_row["duration_s"]) + 0.05
                )
            if wer <= 0.4:
                asr_level = "高度支持官方转写"
            elif wer <= 0.7:
                asr_level = "基本支持官方转写"
            else:
                asr_level = "ASR差异较大，保留低置信标记"
            records.append({
                "样本编号": row.sample_id,
                "MMS平均对齐置信度": float(row.mean_alignment_score),
                "官方转写": str(manifest_row["text"]),
                "Whisper转写": hypothesis,
                "词错误率WER": wer,
                "时间区间结构合法": temporal_valid,
                "独立ASR复核结论": asr_level,
                "最终处理": "保留官方转写和MMS时间区间；保留质量标记，不用于调参",
            })
            print(json.dumps(records[-1], ensure_ascii=False))

    table = pd.DataFrame(records)
    table.to_csv(review_path, index=False, encoding="utf-8-sig")
    counts = table["独立ASR复核结论"].value_counts().to_dict()
    summary = {
        "reviewed_count": int(len(table)),
        "asr_model": model_name,
        "mean_wer": float(table["词错误率WER"].mean()),
        "median_wer": float(table["词错误率WER"].median()),
        "all_temporal_structures_valid": bool(table["时间区间结构合法"].all()),
        "review_level_counts": {str(key): int(value) for key, value in counts.items()},
        "interpretation": "独立ASR只用于复核分层；高WER不自动判定官方转写或强制对齐错误。",
    }
    (output_dir / "alignment_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    review_by_id = table.set_index("样本编号")
    log["alignment_review_method"] = "MMS自动阈值"
    log["independent_asr_wer"] = np.nan
    log["alignment_reliability"] = "高：MMS平均置信度不低于0.6"
    log["time_mapping_usable"] = True
    for index, row in log.loc[log["mean_alignment_score"] < 0.6].iterrows():
        reviewed = review_by_id.loc[row["sample_id"]]
        wer = float(reviewed["词错误率WER"])
        supported = wer <= 0.7
        log.loc[index, "alignment_review_method"] = "MMS低分+Whisper独立ASR"
        log.loc[index, "independent_asr_wer"] = wer
        log.loc[index, "alignment_reliability"] = (
            "中：独立ASR支持官方转写" if supported
            else "低：独立ASR与官方转写差异较大"
        )
        log.loc[index, "time_mapping_usable"] = supported

    log.to_csv(feature_root / "extraction_log.csv", index=False, encoding="utf-8-sig")
    for row in log.itertuples(index=False):
        path = feature_root / str(row.output_file)
        with np.load(path) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        arrays["alignment_reliability"] = np.asarray(str(row.alignment_reliability))
        arrays["time_mapping_usable"] = np.asarray(bool(row.time_mapping_usable))
        arrays["independent_asr_wer"] = np.asarray(
            float(row.independent_asr_wer), dtype=np.float32
        )
        safe_save_npz(path, arrays)

    summary["high_mms_count"] = int((log["mean_alignment_score"] >= 0.6).sum())
    summary["independent_asr_supported_count"] = int(
        ((log["mean_alignment_score"] < 0.6) & log["time_mapping_usable"]).sum()
    )
    summary["low_reliability_count"] = int((~log["time_mapping_usable"]).sum())
    (output_dir / "alignment_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
