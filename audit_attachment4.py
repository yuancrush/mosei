from __future__ import annotations

"""审计附件4，并建立可解释性实验所需的原始视频时间轴索引。

本脚本只读取竞赛官方 pickle/video，不运行模型，也不产生预测。输出包括：
1. 对齐版与未对齐版的字段、形状和配对一致性；
2. 视频时长、帧率、帧数和两套视频是否为同一文件；
3. 50位置对齐特征及变长特征到原始视频秒数的确定性索引表。

注意：对齐版的50个位置是文本token/词位接口，并非天然的等长视频分段。
在尚未进行语音强制对齐前，脚本仅给出可逆的“归一化位置映射”，并明确标记
其证据等级，禁止把它写成逐词精确时间戳。
"""

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import cv2
import numpy as np

# 附件4由NumPy 2.x序列化；比赛环境仍使用NumPy 1.26时需提供只读反序列化别名。
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)


PROJECT_ROOT = Path(__file__).resolve().parent


def find_attachment4(root: Path) -> Tuple[Path, Path]:
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
            f"附件4目录应各唯一匹配一次，实际对齐版={len(aligned)}，未对齐版={len(unaligned)}"
        )
    return aligned[0], unaligned[0]


def load_trusted_pickle(path: Path) -> Mapping[str, Any]:
    # 仅用于竞赛官方数据文件；pickle不可用于来源不明的文件。
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, Mapping):
        raise TypeError(f"{path} 顶层类型不是字典: {type(obj)}")
    if "test" in obj and isinstance(obj["test"], Mapping):
        return obj["test"]
    return obj


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unwrap_singleton(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape[:1] == (1,):
        return value[0]
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def scalar_text(value: Any) -> str:
    value = unwrap_singleton(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def field_schema(sample: Mapping[str, Any]) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    for key, value in sample.items():
        if isinstance(value, str):
            result[key] = {"kind": "text", "preview": value[:120]}
            continue
        array = np.asarray(value)
        item = {"kind": "array", "shape": list(array.shape), "dtype": str(array.dtype)}
        if array.size and np.issubdtype(array.dtype, np.number):
            item.update(
                {
                    "finite": bool(np.isfinite(array).all()),
                    "zero_ratio": float(np.mean(array == 0)),
                }
            )
        result[key] = item
    return result


def sequence(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim >= 3 and array.shape[0] == 1:
        array = array[0]
    return array


def effective_length(sample: Mapping[str, Any], modality: str) -> int:
    values = sequence(sample[modality])
    data_extent = 0
    if values.ndim >= 2:
        valid = np.isfinite(values).all(axis=-1) & np.any(values != 0, axis=-1)
        indices = np.flatnonzero(valid)
        data_extent = int(indices[-1] + 1) if len(indices) else 0
    length_key = f"{modality}_lengths"
    if length_key in sample:
        raw = np.asarray(sample[length_key]).reshape(-1)
        if raw.size:
            # 附件4的13/16号vision_lengths错误地记录为1，但后续仍有非零帧。
            # 采用“元数据长度与最后非零位置的较大值”避免静默截掉真实特征。
            return max(0, min(max(int(raw[0]), data_extent), len(values)))
    if modality == "text" and "text_bert" in sample:
        text_bert = sequence(sample["text_bert"])
        if text_bert.ndim == 2 and text_bert.shape[0] == 3:
            return int(np.sum(text_bert[1] > 0))
    if values.ndim >= 2:
        return data_extent
    return len(values)


def video_metadata(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    duration = frame_count / fps if fps > 0 else float("nan")
    capture.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def values_equal(left: Any, right: Any) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        return bool(np.allclose(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="附件4数据审计与原始视频时间轴索引")
    parser.add_argument("--aligned-dir", type=Path, default=None)
    parser.add_argument("--unaligned-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "question3_stage_a",
    )
    args = parser.parse_args()
    discovered_aligned, discovered_unaligned = find_attachment4(PROJECT_ROOT)
    aligned_dir = args.aligned_dir or discovered_aligned
    unaligned_dir = args.unaligned_dir or discovered_unaligned

    aligned_files = sorted(aligned_dir.glob("*.pkl"))
    unaligned_files = sorted(unaligned_dir.glob("*.pkl"))
    aligned_names = {path.stem for path in aligned_files}
    unaligned_names = {path.stem for path in unaligned_files}
    paired_names = sorted(aligned_names & unaligned_names)
    if not paired_names:
        raise FileNotFoundError("附件4没有可配对的对齐/未对齐pickle")

    sample_rows: List[dict] = []
    schemas: Dict[str, Dict[str, dict]] = {"aligned": {}, "unaligned": {}}
    shared_field_equal_counts: Counter[str] = Counter()
    identical_video_count = 0

    for name in paired_names:
        aligned_path = aligned_dir / f"{name}.pkl"
        unaligned_path = unaligned_dir / f"{name}.pkl"
        aligned = load_trusted_pickle(aligned_path)
        unaligned = load_trusted_pickle(unaligned_path)
        if not schemas["aligned"]:
            schemas["aligned"] = field_schema(aligned)
            schemas["unaligned"] = field_schema(unaligned)

        aligned_video = video_metadata(aligned_dir / "videos" / f"{name}.mp4")
        unaligned_video = video_metadata(unaligned_dir / "videos" / f"{name}.mp4")
        video_identical = aligned_video["sha256"] == unaligned_video["sha256"]
        identical_video_count += int(video_identical)

        shared_fields = sorted(set(aligned) & set(unaligned))
        shared_equal = {key: values_equal(aligned[key], unaligned[key]) for key in shared_fields}
        shared_field_equal_counts.update(key for key, equal in shared_equal.items() if equal)

        raw_text = ""
        for source in (aligned, unaligned):
            if "raw_text" in source:
                raw_text = scalar_text(source["raw_text"])
                break
        sample_id = name
        for source in (aligned, unaligned):
            if "id" in source:
                sample_id = scalar_text(source["id"])
                break

        text_length = effective_length(aligned, "text") if "text" in aligned else 0
        audio_aligned_length = effective_length(aligned, "audio")
        vision_aligned_length = effective_length(aligned, "vision")
        audio_unaligned_length = effective_length(unaligned, "audio")
        vision_unaligned_length = effective_length(unaligned, "vision")
        unaligned_audio_values = sequence(unaligned["audio"])
        unaligned_vision_values = sequence(unaligned["vision"])
        unaligned_audio_length_metadata = int(np.asarray(unaligned["audio_lengths"]).item())
        unaligned_vision_length_metadata = int(np.asarray(unaligned["vision_lengths"]).item())
        unaligned_audio_nonzero_rows = int(
            np.sum(np.any(unaligned_audio_values != 0, axis=-1))
        )
        unaligned_vision_nonzero_rows = int(
            np.sum(np.any(unaligned_vision_values != 0, axis=-1))
        )
        duration = aligned_video["duration_seconds"]

        row = {
            "sample": name,
            "id": sample_id,
            "raw_text": raw_text,
            "video_duration_seconds": round(duration, 6),
            "fps": aligned_video["fps"],
            "frame_count": aligned_video["frame_count"],
            "width": aligned_video["width"],
            "height": aligned_video["height"],
            "paired_video_byte_identical": video_identical,
            "aligned_text_effective_length": text_length,
            "aligned_audio_effective_length": audio_aligned_length,
            "aligned_vision_effective_length": vision_aligned_length,
            "unaligned_audio_effective_length": audio_unaligned_length,
            "unaligned_vision_effective_length": vision_unaligned_length,
            "unaligned_audio_length_metadata": unaligned_audio_length_metadata,
            "unaligned_vision_length_metadata": unaligned_vision_length_metadata,
            "unaligned_vision_length_metadata_consistent": (
                unaligned_vision_length_metadata == vision_unaligned_length
            ),
            "unaligned_audio_nonzero_rows": unaligned_audio_nonzero_rows,
            "unaligned_vision_nonzero_rows": unaligned_vision_nonzero_rows,
            "shared_fields": "|".join(shared_fields),
            "shared_equal_fields": "|".join(key for key in shared_fields if shared_equal[key]),
        }
        sample_rows.append(row)

    durations = np.asarray([row["video_duration_seconds"] for row in sample_rows])
    summary = {
        "data_directories": {
            "aligned": str(aligned_dir.resolve()),
            "unaligned": str(unaligned_dir.resolve()),
        },
        "file_counts": {
            "aligned_pickle": len(aligned_files),
            "unaligned_pickle": len(unaligned_files),
            "paired_samples": len(paired_names),
            "aligned_video": len(list((aligned_dir / "videos").glob("*.mp4"))),
            "unaligned_video": len(list((unaligned_dir / "videos").glob("*.mp4"))),
        },
        "paired_name_sets_equal": aligned_names == unaligned_names,
        "paired_videos_byte_identical": identical_video_count,
        "video_duration_seconds": {
            "min": float(durations.min()),
            "mean": float(durations.mean()),
            "median": float(np.median(durations)),
            "max": float(durations.max()),
        },
        "schemas_first_sample": schemas,
        "shared_field_equal_counts": dict(shared_field_equal_counts),
        "effective_length": {
            key: {
                "min": int(min(row[key] for row in sample_rows)),
                "median": float(np.median([row[key] for row in sample_rows])),
                "max": int(max(row[key] for row in sample_rows)),
            }
            for key in (
                "aligned_text_effective_length",
                "aligned_audio_effective_length",
                "aligned_vision_effective_length",
                "unaligned_audio_effective_length",
                "unaligned_vision_effective_length",
            )
        },
        "time_mapping_status": {
            "physical_video_timeline": "established_from_frame_count_and_fps",
            "feature_positions": "see outputs/question3_stage_a_timeline",
            "warning": (
                "音轨时长可能短于视频流时长，禁止按视频总时长线性拉伸特征位置；"
                "必须使用align_attachment4_timeline.py生成的强制对齐/固定帧率映射。"
            ),
        },
        "contains_labels": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "samples.csv", sample_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
