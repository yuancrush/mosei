from __future__ import annotations

"""将附件4的BERT token位置映射到原视频的词级物理时间。

复用问题1已验证的MMS强制对齐方法。输出只建立解释证据索引，不生成情感标签，
不使用附件4调参。对低置信度词必须保留分数，阶段B/C不得把它包装成高精度证据。
"""

import argparse
import csv
import json
import pickle
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torchaudio
from transformers import AutoTokenizer

from extract_question1_features import clean_word, extract_wav, force_align_words


PROJECT_ROOT = Path(__file__).resolve().parent
MAX_LENGTH = 50

# 附件4由NumPy 2.x序列化，兼容当前NumPy 1.26竞赛环境。
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)


def find_attachment4(root: Path) -> tuple[Path, Path]:
    aligned = [
        path
        for path in root.parent.glob("**/附件4-可解释专项视频样本与特征文件/对齐版本")
        if path.is_dir()
    ]
    unaligned = [
        path
        for path in root.parent.glob("**/附件4-可解释专项视频样本与特征文件/未对齐版本")
        if path.is_dir()
    ]
    if len(aligned) != 1 or len(unaligned) != 1:
        raise FileNotFoundError(
            f"附件4目录应唯一，对齐版={len(aligned)}，未对齐版={len(unaligned)}"
        )
    return aligned[0], unaligned[0]


def load_trusted_pickle(path: Path) -> Mapping[str, Any]:
    # 仅用于竞赛官方pickle。
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if "test" in obj and isinstance(obj["test"], Mapping):
        return obj["test"]
    return obj


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def raw_word_spans(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    pieces = [(match.group(), match.span()) for match in re.finditer(r"\S+", text)]
    kept = [(word, span) for word, span in pieces if clean_word(word)]
    return [word for word, _ in kept], [span for _, span in kept]


def token_to_word(
    token_start: int,
    token_end: int,
    word_spans: list[tuple[int, int]],
) -> int:
    if token_start == token_end:
        return -1
    return next(
        (
            index
            for index, (word_start, word_end) in enumerate(word_spans)
            if token_start < word_end and token_end > word_start
        ),
        -1,
    )


def unaligned_time_rows(
    sample: str,
    modality: str,
    length: int,
    feature_rate_hz: float,
    audio_duration: float,
) -> list[dict]:
    """按附件未对齐特征的固定帧率建立原音轨秒级索引。"""
    rows = []
    for index in range(length):
        start = index / feature_rate_hz
        end = min((index + 1) / feature_rate_hz, audio_duration)
        rows.append(
            {
                "sample": sample,
                "modality": modality,
                "feature_index": index,
                "start_seconds": round(start, 6),
                "end_seconds": round(end, 6),
                "center_seconds": round((start + end) / 2, 6),
                "feature_rate_hz": feature_rate_hz,
                "evidence_basis": "fixed_rate_from_unaligned_feature_interface",
            }
        )
    return rows


def robust_unaligned_length(sample: Mapping[str, Any], modality: str) -> tuple[int, int]:
    """返回修正后的有效范围及官方length字段，兼容13/16号视觉长度元数据异常。"""
    metadata_length = int(np.asarray(sample[f"{modality}_lengths"]).item())
    values = np.asarray(sample[modality])
    nonzero = np.isfinite(values).all(axis=-1) & np.any(values != 0, axis=-1)
    indices = np.flatnonzero(nonzero)
    nonzero_extent = int(indices[-1] + 1) if len(indices) else 0
    return max(metadata_length, nonzero_extent), metadata_length


def main() -> None:
    parser = argparse.ArgumentParser(description="附件4词级强制对齐与token时间映射")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--unaligned-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "question3_stage_a_timeline",
    )
    args = parser.parse_args()
    discovered_aligned, discovered_unaligned = find_attachment4(PROJECT_ROOT)
    data_dir = args.data_dir or discovered_aligned
    unaligned_dir = args.unaligned_dir or discovered_unaligned
    files = sorted(data_dir.glob("*.pkl"))
    if len(files) != 20:
        raise ValueError(f"附件4预期20条样本，实际为{len(files)}")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    bundle = torchaudio.pipelines.MMS_FA
    alignment_model = bundle.get_model(with_star=True).to(device).eval()
    alignment_tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    bert_tokenizer = AutoTokenizer.from_pretrained(
        args.bert_model, use_fast=True, local_files_only=True
    )

    word_rows: list[dict] = []
    token_rows: list[dict] = []
    unaligned_rows: list[dict] = []
    sample_rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="mosei_q3_timeline_") as temporary:
        temporary_dir = Path(temporary)
        for order, path in enumerate(files, start=1):
            sample = load_trusted_pickle(path)
            unaligned_sample = load_trusted_pickle(unaligned_dir / path.name)
            text = str(sample["raw_text"])
            words, word_spans = raw_word_spans(text)
            wav_path = temporary_dir / f"{path.stem}.wav"
            extract_wav(data_dir / "videos" / f"{path.stem}.mp4", wav_path, bundle.sample_rate)
            waveform, sample_rate = torchaudio.load(wav_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            intervals, scores = force_align_words(
                waveform,
                alignment_model,
                alignment_tokenizer,
                aligner,
                words,
                sample_rate,
                device,
            )

            encoded = bert_tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
                return_offsets_mapping=True,
            )
            provided = np.asarray(sample["text_bert"])
            regenerated = np.stack(
                [
                    encoded["input_ids"],
                    encoded["attention_mask"],
                    encoded.get("token_type_ids", [0] * MAX_LENGTH),
                ]
            )
            bert_interface_match = bool(np.array_equal(provided, regenerated))
            tokens = bert_tokenizer.convert_ids_to_tokens(encoded["input_ids"])
            aligned_audio = np.asarray(sample["audio"])
            aligned_vision = np.asarray(sample["vision"])
            audio_available = np.isfinite(aligned_audio).all(axis=-1) & np.any(
                aligned_audio != 0, axis=-1
            )
            vision_available = np.isfinite(aligned_vision).all(axis=-1) & np.any(
                aligned_vision != 0, axis=-1
            )

            for word_index, (word, interval, score, char_span) in enumerate(
                zip(words, intervals, scores, word_spans)
            ):
                word_rows.append(
                    {
                        "sample": path.stem,
                        "word_index": word_index,
                        "word": word,
                        "normalized_word": clean_word(word),
                        "char_start": char_span[0],
                        "char_end": char_span[1],
                        "start_seconds": round(float(interval[0]), 6),
                        "end_seconds": round(float(interval[1]), 6),
                        "duration_seconds": round(float(interval[1] - interval[0]), 6),
                        "alignment_score": round(float(score), 6),
                        "low_confidence_below_0_5": bool(score < 0.5),
                    }
                )

            mapped_word_indices = set()
            for token_index, (token, token_id, mask, offset) in enumerate(
                zip(
                    tokens,
                    encoded["input_ids"],
                    encoded["attention_mask"],
                    encoded["offset_mapping"],
                )
            ):
                word_index = token_to_word(offset[0], offset[1], word_spans)
                if word_index >= 0:
                    mapped_word_indices.add(word_index)
                    start_seconds = float(intervals[word_index, 0])
                    end_seconds = float(intervals[word_index, 1])
                    score = float(scores[word_index])
                    evidence = "forced_alignment_word_interval"
                else:
                    start_seconds = float("nan")
                    end_seconds = float("nan")
                    score = float("nan")
                    evidence = "special_or_padding_no_time_evidence"
                token_rows.append(
                    {
                        "sample": path.stem,
                        "token_index": token_index,
                        "token": token,
                        "token_id": int(token_id),
                        "attention_mask": int(mask),
                        "char_start": int(offset[0]),
                        "char_end": int(offset[1]),
                        "word_index": word_index,
                        "word": words[word_index] if word_index >= 0 else "",
                        "start_seconds": "" if word_index < 0 else round(start_seconds, 6),
                        "end_seconds": "" if word_index < 0 else round(end_seconds, 6),
                        "alignment_score": "" if word_index < 0 else round(score, 6),
                        "aligned_audio_available": bool(audio_available[token_index]),
                        "aligned_vision_available": bool(vision_available[token_index]),
                        "evidence_basis": evidence,
                    }
                )

            audio_duration = waveform.shape[-1] / sample_rate
            unaligned_audio_length, audio_length_metadata = robust_unaligned_length(
                unaligned_sample, "audio"
            )
            unaligned_vision_length, vision_length_metadata = robust_unaligned_length(
                unaligned_sample, "vision"
            )
            unaligned_rows.extend(
                unaligned_time_rows(
                    path.stem, "audio", unaligned_audio_length, 20.0, audio_duration
                )
            )
            unaligned_rows.extend(
                unaligned_time_rows(
                    path.stem, "vision", unaligned_vision_length, 15.0, audio_duration
                )
            )

            sample_rows.append(
                {
                    "sample": path.stem,
                    "word_count": len(words),
                    "mapped_word_count_within_50": len(mapped_word_indices),
                    "truncated_word_count": len(words) - len(mapped_word_indices),
                    "effective_bert_tokens": int(np.sum(encoded["attention_mask"])),
                    "bert_interface_exact_match": bert_interface_match,
                    "audio_duration_seconds": round(audio_duration, 6),
                    "unaligned_audio_length": unaligned_audio_length,
                    "unaligned_vision_length": unaligned_vision_length,
                    "audio_length_metadata": audio_length_metadata,
                    "vision_length_metadata": vision_length_metadata,
                    "vision_length_metadata_consistent": (
                        vision_length_metadata == unaligned_vision_length
                    ),
                    "audio_length_rate_hz": round(unaligned_audio_length / audio_duration, 6),
                    "vision_length_rate_hz": round(unaligned_vision_length / audio_duration, 6),
                    "first_word_start_seconds": round(float(intervals[0, 0]), 6),
                    "last_word_end_seconds": round(float(intervals[-1, 1]), 6),
                    "mean_alignment_score": round(float(scores.mean()), 6),
                    "min_alignment_score": round(float(scores.min()), 6),
                    "low_confidence_word_count": int(np.sum(scores < 0.5)),
                    "low_confidence_word_ratio": round(float(np.mean(scores < 0.5)), 6),
                }
            )
            print(
                f"[{order:02d}/{len(files)}] {path.stem}: words={len(words)}, "
                f"mean_score={scores.mean():.3f}, bert_match={bert_interface_match}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "samples.csv", sample_rows)
    write_csv(args.output_dir / "word_intervals.csv", word_rows)
    write_csv(args.output_dir / "token_time_map.csv", token_rows)
    write_csv(args.output_dir / "unaligned_feature_time_map.csv", unaligned_rows)

    summary = {
        "sample_count": len(sample_rows),
        "device": str(device),
        "alignment_model": "torchaudio.pipelines.MMS_FA",
        "bert_tokenizer": args.bert_model,
        "bert_interface_exact_match_count": sum(
            row["bert_interface_exact_match"] for row in sample_rows
        ),
        "total_word_count": sum(row["word_count"] for row in sample_rows),
        "mapped_word_count_within_50": sum(
            row["mapped_word_count_within_50"] for row in sample_rows
        ),
        "truncated_sample_count": sum(row["truncated_word_count"] > 0 for row in sample_rows),
        "truncated_word_count": sum(row["truncated_word_count"] for row in sample_rows),
        "mean_of_sample_alignment_scores": float(
            np.mean([row["mean_alignment_score"] for row in sample_rows])
        ),
        "samples_mean_score_below_0_6": [
            row["sample"] for row in sample_rows if row["mean_alignment_score"] < 0.6
        ],
        "low_confidence_word_count": sum(
            row["low_confidence_word_count"] for row in sample_rows
        ),
        "unaligned_feature_time_mapping": {
            "audio_feature_rate_hz": 20.0,
            "vision_feature_rate_hz": 15.0,
            "audio_observed_length_rate_median": float(
                np.median([row["audio_length_rate_hz"] for row in sample_rows])
            ),
            "vision_observed_length_rate_median_excluding_length_1": float(
                np.median(
                    [
                        row["vision_length_rate_hz"]
                        for row in sample_rows
                        if row["unaligned_vision_length"] > 1
                    ]
                )
            ),
            "time_origin": "视频/音轨起点0秒",
            "vision_length_metadata_inconsistent_samples": [
                row["sample"]
                for row in sample_rows
                if not row["vision_length_metadata_consistent"]
            ],
            "warning": (
                "第13、16条vision_lengths=1但后续存在非零帧，已按最后非零位置修正有效范围；"
                "其中13号对齐版视觉仍为全零，不能声称视觉影响了正式对齐模型。"
            ),
        },
        "time_mapping_definition": (
            "每个有效BERT WordPiece通过原文字符区间指向一个官方文本词，再使用MMS强制对齐"
            "得到该词在原视频音轨中的[start,end)秒区间；[CLS]/[SEP]/padding不分配伪时间。"
        ),
        "writing_constraint": (
            "低置信度词需保留alignment_score；50-token截断之外的词不能由对齐版模型定位，"
            "但完整word_intervals仍保留供原始证据展示。"
        ),
        "contains_labels": False,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
