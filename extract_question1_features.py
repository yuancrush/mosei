from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import imageio_ffmpeg
import numpy as np
import opensmile
import pandas as pd
import torch
import torchaudio
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
MAX_LENGTH = 50

# 35个固定的面部几何距离，均用双眼外眼角距离归一化。
# 与附件2的FACET 35维不是同一特征系，不应混用权重。
VISUAL_PAIRS: tuple[tuple[str, int, int], ...] = (
    ("left_eye_open_1", 159, 145),
    ("left_eye_open_2", 158, 153),
    ("left_eye_open_3", 160, 144),
    ("left_eye_width", 33, 133),
    ("right_eye_open_1", 386, 374),
    ("right_eye_open_2", 385, 380),
    ("right_eye_open_3", 387, 373),
    ("right_eye_width", 362, 263),
    ("left_brow_eye_inner", 107, 159),
    ("left_brow_eye_mid", 105, 158),
    ("left_brow_eye_outer", 70, 160),
    ("right_brow_eye_inner", 336, 386),
    ("right_brow_eye_mid", 334, 385),
    ("right_brow_eye_outer", 300, 387),
    ("mouth_outer_width", 61, 291),
    ("mouth_inner_width", 78, 308),
    ("mouth_outer_open", 0, 17),
    ("mouth_inner_open", 13, 14),
    ("upper_lip_left", 61, 13),
    ("upper_lip_right", 291, 13),
    ("lower_lip_left", 61, 14),
    ("lower_lip_right", 291, 14),
    ("mouth_left_to_nose", 61, 1),
    ("mouth_right_to_nose", 291, 1),
    ("mouth_left_to_chin", 61, 152),
    ("mouth_right_to_chin", 291, 152),
    ("philtrum", 2, 0),
    ("lower_lip_to_chin", 17, 152),
    ("nose_width", 49, 279),
    ("nose_length", 168, 2),
    ("left_cheek", 234, 1),
    ("right_cheek", 454, 1),
    ("face_height", 10, 152),
    ("jaw_width", 172, 397),
    ("nose_to_chin", 1, 152),
)


def find_unique(pattern: str, description: str) -> Path:
    candidates = list(PROJECT_ROOT.parent.glob(pattern))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"{description}数量应为1，实际为{len(candidates)}: {candidates}"
        )
    return candidates[0]


def integer_to_english(value: int) -> str:
    ones = (
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen",
    )
    tens = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
    if value < 20:
        return ones[value]
    if value < 100:
        return tens[value // 10] + (ones[value % 10] if value % 10 else "")
    if value < 1000:
        return ones[value // 100] + "hundred" + (
            integer_to_english(value % 100) if value % 100 else ""
        )
    if value < 1_000_000:
        return integer_to_english(value // 1000) + "thousand" + (
            integer_to_english(value % 1000) if value % 1000 else ""
        )
    return "".join(ones[int(digit)] for digit in str(value))


def clean_word(word: str) -> str:
    text = word.lower().replace("’", "'").replace("‘", "'").replace(",", "")
    text = re.sub(r"\d+", lambda match: integer_to_english(int(match.group())), text)
    return re.sub(r"[^a-z']", "", text)


def extract_wav(video_path: Path, wav_path: Path, sample_rate: int = 16000) -> None:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(command, check=True)


def force_align_words(
    waveform: torch.Tensor,
    model: torch.nn.Module,
    tokenizer: Any,
    aligner: Any,
    words: list[str],
    sample_rate: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = [clean_word(word) for word in words]
    if any(not word for word in normalized):
        bad = [words[index] for index, word in enumerate(normalized) if not word]
        raise ValueError(f"转写中存在无法对齐的纯符号token: {bad}")
    tokens = tokenizer(normalized)
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
    spans = aligner(emission[0].cpu(), tokens)
    samples_per_frame = waveform.shape[-1] / emission.shape[1]
    intervals = np.zeros((len(words), 2), dtype=np.float32)
    scores = np.zeros(len(words), dtype=np.float32)
    for index, word_spans in enumerate(spans):
        intervals[index, 0] = word_spans[0].start * samples_per_frame / sample_rate
        intervals[index, 1] = word_spans[-1].end * samples_per_frame / sample_rate
        scores[index] = float(np.mean([span.score for span in word_spans]))
    return intervals, scores


def aggregate_frames(
    values: np.ndarray, times: np.ndarray, intervals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros((len(intervals), values.shape[1]), dtype=np.float32)
    available = np.zeros(len(intervals), dtype=bool)
    for index, (start, end) in enumerate(intervals):
        selected = (times >= start) & (times < end)
        if selected.any():
            output[index] = np.nan_to_num(values[selected].mean(axis=0))
            available[index] = True
    return output, available


def extract_audio_words(
    wav_path: Path, intervals: np.ndarray, smile: opensmile.Smile
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = smile.process_file(str(wav_path))
    starts = frame.index.get_level_values("start").total_seconds().to_numpy()
    ends = frame.index.get_level_values("end").total_seconds().to_numpy()
    times = (starts + ends) / 2
    values = frame.to_numpy(dtype=np.float32)
    aggregated, mask = aggregate_frames(values, times, intervals)
    return aggregated, mask, [str(column) for column in frame.columns]


def face_geometry(landmarks: Any) -> np.ndarray:
    points = np.asarray([[point.x, point.y] for point in landmarks.landmark], dtype=np.float32)
    scale = float(np.linalg.norm(points[33] - points[263]))
    scale = max(scale, 1e-6)
    return np.asarray(
        [np.linalg.norm(points[left] - points[right]) / scale for _, left, right in VISUAL_PAIRS],
        dtype=np.float32,
    )


def extract_visual_words(
    video_path: Path, intervals: np.ndarray, sample_fps: float = 10.0
) -> tuple[np.ndarray, np.ndarray, float]:
    import mediapipe as mp

    capture = cv2.VideoCapture(str(video_path))
    native_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not capture.isOpened() or native_fps <= 0:
        raise RuntimeError(f"无法读取视频: {video_path}")
    stride = max(1, round(native_fps / sample_fps))
    values: list[np.ndarray] = []
    times: list[float] = []
    sampled_frames = 0
    detected_frames = 0
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                sampled_frames += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)
                if result.multi_face_landmarks:
                    values.append(face_geometry(result.multi_face_landmarks[0]))
                    times.append(frame_index / native_fps)
                    detected_frames += 1
            frame_index += 1
    capture.release()
    if values:
        aggregated, mask = aggregate_frames(
            np.stack(values), np.asarray(times, dtype=np.float32), intervals
        )
    else:
        aggregated = np.zeros((len(intervals), len(VISUAL_PAIRS)), dtype=np.float32)
        mask = np.zeros(len(intervals), dtype=bool)
    detection_ratio = detected_frames / sampled_frames if sampled_frames else 0.0
    return aggregated, mask, detection_ratio


def encode_text_and_align(
    text: str,
    words: list[str],
    intervals: np.ndarray,
    audio_words: np.ndarray,
    audio_mask: np.ndarray,
    visual_words: np.ndarray,
    visual_mask: np.ndarray,
    bert_tokenizer: Any,
    bert_model: torch.nn.Module,
    device: torch.device,
) -> dict[str, np.ndarray]:
    encoding = bert_tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)
    model_input = {key: value.to(device) for key, value in encoding.items()}
    with torch.inference_mode():
        full_text_features = bert_model(**model_input).last_hidden_state[0].cpu().numpy()

    text_features = np.zeros((MAX_LENGTH, full_text_features.shape[1]), dtype=np.float32)
    copied_length = min(MAX_LENGTH, len(word_ids))
    text_features[:copied_length] = full_text_features[:copied_length]
    word_text = np.zeros((len(words), full_text_features.shape[1]), dtype=np.float32)
    word_counts = np.zeros(len(words), dtype=np.int16)
    for token_index, word_index in enumerate(word_ids):
        if word_index is not None:
            word_text[word_index] += full_text_features[token_index]
            word_counts[word_index] += 1
    word_text /= np.maximum(word_counts[:, None], 1)

    audio = np.zeros((MAX_LENGTH, audio_words.shape[1]), dtype=np.float32)
    vision = np.zeros((MAX_LENGTH, visual_words.shape[1]), dtype=np.float32)
    token_intervals = np.zeros((MAX_LENGTH, 2), dtype=np.float32)
    token_audio_mask = np.zeros(MAX_LENGTH, dtype=bool)
    token_visual_mask = np.zeros(MAX_LENGTH, dtype=bool)
    word_index_array = np.full(MAX_LENGTH, -1, dtype=np.int16)
    for token_index, word_index in enumerate(word_ids[:MAX_LENGTH]):
        if word_index is None:
            continue
        audio[token_index] = audio_words[word_index]
        vision[token_index] = visual_words[word_index]
        token_intervals[token_index] = intervals[word_index]
        token_audio_mask[token_index] = audio_mask[word_index]
        token_visual_mask[token_index] = visual_mask[word_index]
        word_index_array[token_index] = word_index

    input_ids = np.zeros(MAX_LENGTH, dtype=np.int64)
    attention_mask = np.zeros(MAX_LENGTH, dtype=np.int64)
    token_type_ids = np.zeros(MAX_LENGTH, dtype=np.int64)
    input_ids[:copied_length] = encoding["input_ids"][0, :copied_length].numpy()
    attention_mask[:copied_length] = encoding["attention_mask"][0, :copied_length].numpy()
    if "token_type_ids" in encoding:
        token_type_ids[:copied_length] = encoding["token_type_ids"][0, :copied_length].numpy()
    text_bert = np.stack(
        [input_ids, attention_mask, token_type_ids]
    )
    return {
        "text": text_features.astype(np.float32),
        "word_text": word_text,
        "text_bert": text_bert,
        "audio": audio,
        "vision": vision,
        "token_intervals": token_intervals,
        "token_word_index": word_index_array,
        "audio_mask": token_audio_mask,
        "vision_mask": token_visual_mask,
        "raw_text": np.asarray(text),
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="附件1词级时间对齐与三模态特征提取")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "reports/question1_audit/manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/question1_features")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--visual-fps", type=float, default=10.0)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.sample_id:
        manifest = manifest[manifest["sample_id"].isin(args.sample_id)]
    if args.limit is not None:
        manifest = manifest.head(args.limit)
    if manifest.empty:
        raise ValueError("未选中任何样本")

    data_dir = find_unique(
        "**/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条", "附件1视频目录"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alignment_bundle = torchaudio.pipelines.MMS_FA
    # 保留star类以吸收非转写语音、片头片尾和噪声，避免把非文本片段
    # 强行分配给相邻词。
    alignment_model = alignment_bundle.get_model(with_star=True).to(device).eval()
    alignment_tokenizer = alignment_bundle.get_tokenizer()
    aligner = alignment_bundle.get_aligner()
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    feature_dir = args.output_dir / "samples"
    feature_dir.mkdir(parents=True, exist_ok=True)
    log_rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        sample_id = str(row["sample_id"])
        start_time = time.perf_counter()
        video_path = data_dir / Path(str(row["relative_path"]))
        raw_words = str(row["text"]).split()
        words = [word for word in raw_words if clean_word(word)]
        safe_name = f'{row["video_id"]}__{int(row["clip_id"])}'
        with tempfile.TemporaryDirectory(prefix="mosei_q1_") as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            extract_wav(video_path, wav_path, alignment_bundle.sample_rate)
            waveform, sample_rate = torchaudio.load(wav_path)
            intervals, alignment_scores = force_align_words(
                waveform,
                alignment_model,
                alignment_tokenizer,
                aligner,
                words,
                sample_rate,
                device,
            )
            audio_words, audio_word_mask, audio_names = extract_audio_words(
                wav_path, intervals, smile
            )
        vision_words, vision_word_mask, face_detection_ratio = extract_visual_words(
            video_path, intervals, sample_fps=args.visual_fps
        )
        arrays = encode_text_and_align(
            str(row["text"]),
            words,
            intervals,
            audio_words,
            audio_word_mask,
            vision_words,
            vision_word_mask,
            bert_tokenizer,
            bert_model,
            device,
        )
        np.savez_compressed(
            feature_dir / f"{safe_name}.npz",
            **arrays,
            sample_id=np.asarray(sample_id),
            words=np.asarray(words),
            normalized_words=np.asarray([clean_word(word) for word in words]),
            word_intervals=intervals,
            alignment_scores=alignment_scores,
            word_audio=audio_words,
            word_vision=vision_words,
            word_audio_mask=audio_word_mask,
            word_vision_mask=vision_word_mask,
            audio_feature_names=np.asarray(audio_names),
            visual_feature_names=np.asarray([name for name, _, _ in VISUAL_PAIRS]),
            regression_label=np.asarray(float(row["regression_label"]), dtype=np.float32),
            annotation=np.asarray(str(row["annotation"])),
        )
        retained_word_count = len(set(arrays["token_word_index"][arrays["token_word_index"] >= 0].tolist()))
        low_alignment_ratio = float((alignment_scores < 0.5).mean())
        quality_status = "review" if (
            float(alignment_scores.mean()) < 0.6
            or low_alignment_ratio > 0.25
            or face_detection_ratio < 0.7
            or float(vision_word_mask.mean()) < 0.7
            or retained_word_count < len(words)
        ) else "pass"
        log_rows.append(
            {
                "sample_id": sample_id,
                "output_file": f"samples/{safe_name}.npz",
                "word_count": len(words),
                "retained_token_count": int((arrays["token_word_index"] >= 0).sum()),
                "retained_word_count": retained_word_count,
                "truncated_word_count": len(words) - retained_word_count,
                "mean_alignment_score": float(alignment_scores.mean()),
                "min_alignment_score": float(alignment_scores.min()),
                "low_alignment_word_ratio": low_alignment_ratio,
                "aligned_start_s": float(intervals[0, 0]),
                "aligned_end_s": float(intervals[-1, 1]),
                "audio_word_coverage": float(audio_word_mask.mean()),
                "visual_word_coverage": float(vision_word_mask.mean()),
                "face_frame_detection_ratio": face_detection_ratio,
                "elapsed_s": time.perf_counter() - start_time,
                "status": quality_status,
            }
        )
        print(json.dumps(log_rows[-1], ensure_ascii=False))

    write_csv(args.output_dir / "extraction_log.csv", log_rows)
    summary = {
        "sample_count": len(log_rows),
        "device": str(device),
        "alignment_model": "torchaudio.pipelines.MMS_FA",
        "text_model": "bert-base-uncased",
        "audio_feature_set": "openSMILE eGeMAPSv02 LowLevelDescriptors",
        "audio_dimension": len(audio_names),
        "visual_feature_set": "MediaPipe FaceMesh 35 normalized geometric distances",
        "visual_dimension": len(VISUAL_PAIRS),
        "max_token_length": MAX_LENGTH,
        "mean_alignment_score": float(
            np.mean([row["mean_alignment_score"] for row in log_rows])
        ),
        "mean_audio_word_coverage": float(
            np.mean([row["audio_word_coverage"] for row in log_rows])
        ),
        "mean_visual_word_coverage": float(
            np.mean([row["visual_word_coverage"] for row in log_rows])
        ),
        "mean_face_frame_detection_ratio": float(
            np.mean([row["face_frame_detection_ratio"] for row in log_rows])
        ),
        "quality_pass_count": sum(row["status"] == "pass" for row in log_rows),
        "quality_review_count": sum(row["status"] == "review" for row in log_rows),
        "truncated_sample_count": sum(row["truncated_word_count"] > 0 for row in log_rows),
        "total_elapsed_s": float(sum(row["elapsed_s"] for row in log_rows)),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
