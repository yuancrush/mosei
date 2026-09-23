from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent


def find_unique(root: Path, pattern: str, description: str) -> Path:
    candidates = list(root.parent.glob(pattern))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"{description}数量应为1，实际为{len(candidates)}: {candidates}"
        )
    return candidates[0]


def normalize_text(value: Any) -> str:
    text = str(value).replace("’", "'").replace("‘", "'").lower().strip()
    return re.sub(r"\s+", " ", text)


def parse_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    numerator, denominator = value.split("/", maxsplit=1)
    denominator_value = float(denominator)
    return float(numerator) / denominator_value if denominator_value else None


def probe_media(path: Path, ffprobe_exe: Path) -> dict[str, Any]:
    command = [
        str(ffprobe_exe),
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,width,height,avg_frame_rate,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
    duration = float(payload.get("format", {}).get("duration", 0.0))
    return {
        "duration_s": duration,
        "file_size_bytes": int(payload.get("format", {}).get("size", path.stat().st_size)),
        "video_codec": video.get("codec_name"),
        "width": video.get("width"),
        "height": video.get("height"),
        "fps": parse_rate(video.get("avg_frame_rate")),
        "audio_codec": audio.get("codec_name"),
        "audio_sample_rate": int(audio["sample_rate"]) if audio.get("sample_rate") else None,
        "audio_channels": audio.get("channels"),
        "has_video_stream": bool(video),
        "has_audio_stream": bool(audio),
    }


def find_ffprobe() -> Path:
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    ffprobe = ffmpeg.with_name("ffprobe.exe")
    if ffprobe.exists():
        return ffprobe
    # imageio-ffmpeg的二进制文件有时只附带ffmpeg；此时用ffmpeg的
    # JSON信息能力不足，明确报错而不猜测媒体参数。
    raise FileNotFoundError(
        f"未找到ffprobe.exe；已找到imageio-ffmpeg的ffmpeg: {ffmpeg}"
    )


def has_audio_stream(path: Path, ffmpeg_exe: Path) -> bool:
    command = [
        str(ffmpeg_exe),
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-frames:a",
        "1",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True)
    return result.returncode == 0


def opencv_probe(path: Path, ffmpeg_exe: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV无法打开视频: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return {
        "duration_s": frame_count / fps if fps > 0 else None,
        "file_size_bytes": path.stat().st_size,
        "video_codec": None,
        "width": width,
        "height": height,
        "fps": fps if fps > 0 else None,
        "audio_codec": None,
        "audio_sample_rate": None,
        "audio_channels": None,
        "has_video_stream": frame_count > 0,
        "has_audio_stream": has_audio_stream(path, ffmpeg_exe),
    }


def load_reference(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as handle:
        dataset = pickle.load(handle)  # 仅读取赛题官方pickle。
    result: dict[str, dict[str, Any]] = {}
    for split, subset in dataset.items():
        for index, sample_id in enumerate(subset["id"]):
            result[str(sample_id)] = {
                "reference_split": split,
                "reference_text": str(subset["raw_text"][index]),
                "reference_regression_label": float(subset["regression_labels"][index]),
                "reference_classification_label": int(
                    subset["classification_labels"][index]
                ),
            }
    return result


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, Any]], label_rows: int) -> dict[str, Any]:
    durations = np.asarray([row["duration_s"] for row in rows], dtype=float)
    class_counts = Counter(row["annotation"] for row in rows)
    resolutions = Counter(f'{row["width"]}x{row["height"]}' for row in rows)
    return {
        "label_row_count": label_rows,
        "video_file_count": len(rows),
        "unique_sample_id_count": len({row["sample_id"] for row in rows}),
        "label_video_exact_match_count": sum(row["label_match"] for row in rows),
        "reference_match_count": sum(row["reference_match"] for row in rows),
        "reference_text_exact_match_count": sum(
            row["reference_text_match"] for row in rows
        ),
        "reference_label_close_count": sum(
            row["reference_label_match"] for row in rows
        ),
        "annotation_sign_consistent_count": sum(
            row["annotation_sign_consistent"] for row in rows
        ),
        "has_video_stream_count": sum(bool(row["has_video_stream"]) for row in rows),
        "has_audio_stream_count": sum(row["has_audio_stream"] is True for row in rows),
        "duration_s": {
            "min": float(durations.min()),
            "mean": float(durations.mean()),
            "median": float(np.median(durations)),
            "max": float(durations.max()),
            "total": float(durations.sum()),
        },
        "annotation_distribution": dict(class_counts),
        "resolution_distribution": dict(resolutions),
        "all_checks_passed": all(
            row["label_match"]
            and row["annotation_sign_consistent"]
            and bool(row["has_video_stream"])
            and bool(row["has_audio_stream"])
            for row in rows
        )
        and len(rows) == label_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="审计附件1的100条原始视频、标签与附件2参照")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--label-file", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "question1_audit",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or find_unique(
        PROJECT_ROOT, "**/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条", "附件1视频目录"
    )
    label_file = args.label_file or data_dir / "label-100.xlsx"
    reference_file = args.reference or find_unique(
        PROJECT_ROOT, "**/附件2-数据集特征文件/aligned_50.pkl", "附件2对齐特征"
    )

    labels = pd.read_excel(label_file, sheet_name="label")
    required_columns = {"video_id", "clip_id", "text", "label", "annotation"}
    missing_columns = required_columns.difference(labels.columns)
    if missing_columns:
        raise ValueError(f"标签表缺少字段: {sorted(missing_columns)}")
    labels["video_id"] = labels["video_id"].astype(str)
    labels["clip_id"] = labels["clip_id"].astype(int)
    labels["sample_id"] = labels["video_id"] + "$_$" + labels["clip_id"].astype(str)
    if labels["sample_id"].duplicated().any():
        duplicates = labels.loc[labels["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"标签表sample_id重复: {duplicates}")
    label_map = labels.set_index("sample_id").to_dict("index")
    reference = load_reference(reference_file)

    videos = sorted(data_dir.glob("*/*.mp4"))
    if not videos:
        raise FileNotFoundError(f"未在{data_dir}下发现mp4文件")
    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    try:
        ffprobe_exe = find_ffprobe()
    except FileNotFoundError:
        ffprobe_exe = None

    rows: list[dict[str, Any]] = []
    for path in videos:
        video_id = path.parent.name
        clip_id = int(path.stem)
        sample_id = f"{video_id}$_${clip_id}"
        label = label_map.get(sample_id)
        ref = reference.get(sample_id)
        if label is None:
            raise ValueError(f"视频在label-100.xlsx中无对应记录: {sample_id}")
        media = (
            probe_media(path, ffprobe_exe)
            if ffprobe_exe
            else opencv_probe(path, ffmpeg_exe)
        )
        regression_label = float(label["label"])
        annotation = str(label["annotation"])
        expected_annotation = (
            "Positive" if regression_label > 0 else "Negative" if regression_label < 0 else "Neutral"
        )
        row = {
            "sample_id": sample_id,
            "video_id": video_id,
            "clip_id": clip_id,
            "relative_path": path.relative_to(data_dir).as_posix(),
            "text": str(label["text"]),
            "regression_label": regression_label,
            "annotation": annotation,
            "label_match": True,
            "annotation_sign_consistent": annotation.casefold()
            == expected_annotation.casefold(),
            **media,
            "reference_match": ref is not None,
            "reference_split": ref["reference_split"] if ref else None,
            "reference_text_match": bool(
                ref and normalize_text(label["text"]) == normalize_text(ref["reference_text"])
            ),
            "reference_label_match": bool(
                ref
                and np.isclose(
                    regression_label, ref["reference_regression_label"], atol=1e-6
                )
            ),
            "reference_classification_label": (
                ref["reference_classification_label"] if ref else None
            ),
        }
        rows.append(row)

    video_ids = {row["sample_id"] for row in rows}
    missing_videos = sorted(set(label_map).difference(video_ids))
    extra_videos = sorted(video_ids.difference(label_map))
    summary = build_summary(rows, len(labels))
    summary.update(
        {
            "data_dir": str(data_dir.resolve()),
            "label_file": str(label_file.resolve()),
            "reference_file": str(reference_file.resolve()),
            "media_probe": "ffprobe" if ffprobe_exe else "opencv_fallback",
            "missing_video_for_label": missing_videos,
            "extra_video_without_label": extra_videos,
        }
    )
    summary["all_checks_passed"] = bool(
        summary["all_checks_passed"] and not missing_videos and not extra_videos
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "manifest.csv", rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
