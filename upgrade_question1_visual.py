from __future__ import annotations

"""重算第一问视觉特征，并安全更新已有NPZ与提取日志。

用途：原始整帧 FaceMesh 对小脸、侧脸较敏感。本脚本复用已生成的词级时间区间，
只重算视觉分支，因此不会重复计算文本BERT、语音特征或强制对齐。
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from extract_question1_features import (
    PROJECT_ROOT,
    extract_visual_words,
    find_unique,
)


def update_fixed_visual(data: dict[str, np.ndarray], word_visual: np.ndarray,
                        word_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """由变长词级视觉特征重建50位置兼容层。"""
    token_word_index = data["token_word_index"]
    visual = np.zeros((len(token_word_index), word_visual.shape[1]), dtype=np.float32)
    visual_mask = np.zeros(len(token_word_index), dtype=bool)
    valid = token_word_index >= 0
    indices = token_word_index[valid].astype(int)
    visual[valid] = word_visual[indices]
    visual_mask[valid] = word_mask[indices]
    return visual, visual_mask


def safe_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """先写同目录临时文件再原子替换，避免中断时损坏正式特征。"""
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def status_from_row(row: pd.Series) -> str:
    is_warning = (
        row["mean_alignment_score"] < 0.6
        or row["low_alignment_word_ratio"] > 0.25
        or row["face_frame_detection_ratio"] < 0.7
        or row["visual_word_coverage"] < 0.7
        or row["truncated_word_count"] > 0
    )
    return "review" if is_warning else "pass"


def main() -> None:
    parser = argparse.ArgumentParser(description="升级第一问小脸/侧脸视觉提取")
    parser.add_argument(
        "--manifest", type=Path,
        default=PROJECT_ROOT / "reports/question1_audit/manifest.csv",
    )
    parser.add_argument(
        "--feature-dir", type=Path,
        default=PROJECT_ROOT / "outputs/question1_features",
    )
    parser.add_argument("--visual-fps", type=float, default=10.0)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    log_path = args.feature_dir / "extraction_log.csv"
    log = pd.read_csv(log_path)
    before = {
        "mean_visual_word_coverage": float(log["visual_word_coverage"].mean()),
        "mean_face_frame_detection_ratio": float(log["face_frame_detection_ratio"].mean()),
        "zero_face_detection_count": int((log["face_frame_detection_ratio"] == 0).sum()),
        "quality_warning_count": int((log["status"] == "review").sum()),
    }
    data_dir = find_unique(
        "**/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条",
        "附件1视频目录",
    )

    for index, row in log.iterrows():
        manifest_row = manifest.loc[manifest["sample_id"] == row["sample_id"]].iloc[0]
        feature_path = args.feature_dir / str(row["output_file"])
        with np.load(feature_path) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        word_visual, word_mask, detection_ratio, fallback_ratio = extract_visual_words(
            data_dir / Path(manifest_row["relative_path"]),
            arrays["word_intervals"],
            sample_fps=args.visual_fps,
        )
        visual, visual_mask = update_fixed_visual(arrays, word_visual, word_mask)
        arrays.update(
            word_vision=word_visual,
            word_vision_mask=word_mask,
            vision=visual,
            vision_mask=visual_mask,
        )
        safe_save_npz(feature_path, arrays)

        log.loc[index, "visual_word_coverage"] = float(word_mask.mean())
        log.loc[index, "face_frame_detection_ratio"] = detection_ratio
        log.loc[index, "face_crop_fallback_ratio"] = fallback_ratio
        log.loc[index, "status"] = status_from_row(log.loc[index])
        print(json.dumps({
            "sample_id": row["sample_id"],
            "visual_word_coverage": float(word_mask.mean()),
            "face_frame_detection_ratio": detection_ratio,
            "face_crop_fallback_ratio": fallback_ratio,
        }, ensure_ascii=False))

    log.to_csv(log_path, index=False, encoding="utf-8-sig")
    after = {
        "mean_visual_word_coverage": float(log["visual_word_coverage"].mean()),
        "mean_face_frame_detection_ratio": float(log["face_frame_detection_ratio"].mean()),
        "mean_face_crop_fallback_ratio": float(log["face_crop_fallback_ratio"].mean()),
        "zero_face_detection_count": int((log["face_frame_detection_ratio"] == 0).sum()),
        "quality_warning_count": int((log["status"] == "review").sum()),
    }
    comparison = {"before": before, "after": after}
    with (args.feature_dir / "visual_upgrade_comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(comparison, handle, ensure_ascii=False, indent=2)

    summary_path = args.feature_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        visual_feature_set=(
            "MediaPipe FaceMesh 35 normalized geometric distances; "
            "Haar face-crop fallback for small/profile faces"
        ),
        mean_visual_word_coverage=after["mean_visual_word_coverage"],
        mean_face_frame_detection_ratio=after["mean_face_frame_detection_ratio"],
        mean_face_crop_fallback_ratio=after["mean_face_crop_fallback_ratio"],
        quality_pass_count=int((log["status"] == "pass").sum()),
        quality_review_count=after["quality_warning_count"],
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
