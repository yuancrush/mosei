from __future__ import annotations

"""汇总问题3阶段A审计结果，生成证据策略表和中文质量图。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent


def configure_chinese_font() -> None:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)
    if font_path is not None:
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    audit_dir = PROJECT_ROOT / "reports" / "question3_stage_a"
    timeline_dir = PROJECT_ROOT / "outputs" / "question3_stage_a_timeline"
    audit = pd.read_csv(audit_dir / "samples.csv", dtype={"sample": str, "id": str})
    timeline = pd.read_csv(timeline_dir / "samples.csv", dtype={"sample": str})
    review_path = audit_dir / "low_alignment_asr_review.csv"
    review = pd.read_csv(review_path, dtype={"sample": str}) if review_path.exists() else pd.DataFrame()
    words = pd.read_csv(timeline_dir / "word_intervals.csv", dtype={"sample": str})
    tokens = pd.read_csv(timeline_dir / "token_time_map.csv", dtype={"sample": str})
    feature_times = pd.read_csv(
        timeline_dir / "unaligned_feature_time_map.csv", dtype={"sample": str}
    )
    merged = audit.merge(timeline, on="sample", validate="one_to_one")
    unusable = set(
        review.loc[~review["word_time_mapping_usable"], "sample"].astype(str)
        if not review.empty
        else []
    )

    rows = []
    for row in merged.itertuples(index=False):
        if row.sample in unusable:
            text_time_level = "不可作逐词精确定位"
        elif row.mean_alignment_score >= 0.8:
            text_time_level = "高"
        else:
            text_time_level = "中"
        visual_usable = (
            row.unaligned_vision_nonzero_rows > 0
            and row.aligned_vision_effective_length > 0
        )
        if row.truncated_word_count > 0:
            token_scope = f"仅前50个BERT位置；另有{row.truncated_word_count}词未进入模型接口"
        else:
            token_scope = "完整官方文本进入50位置接口"
        restrictions = []
        if row.sample in unusable:
            restrictions.append("不输出逐词秒级结论；改用粗粒度音视频区间")
        if not visual_usable:
            restrictions.append("正式模型视觉贡献不可用；关键帧只能作原视频描述，不能声称影响预测")
        if row.truncated_word_count > 0:
            restrictions.append("截断词只能展示原文，不能声称影响模型预测")
        rows.append(
            {
                "sample": row.sample,
                "text_word_time_reliability": text_time_level,
                "mms_mean_score": row.mean_alignment_score,
                "text_model_scope": token_scope,
                "model_visual_contribution_usable": bool(visual_usable),
                "audio_evidence_usable": row.unaligned_audio_length > 1,
                "bert_interface_exact_match": bool(row.bert_interface_exact_match),
                "required_restrictions": "；".join(restrictions) if restrictions else "无额外降级",
            }
        )
    policy = pd.DataFrame(rows)
    policy.to_csv(audit_dir / "evidence_policy.csv", index=False, encoding="utf-8-sig")

    configure_chinese_font()
    labels = merged["sample"].tolist()
    x = np.arange(len(labels))
    scores = merged["mean_alignment_score"].to_numpy()
    colors = ["#DC2626" if sample in unusable else "#2563EB" for sample in labels]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].bar(x, scores, color=colors)
    axes[0].axhline(0.6, color="#D97706", linestyle="--", label="自动复核阈值0.6")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("MMS平均对齐置信度")
    axes[0].set_title("附件4词级时间映射质量（红色为独立ASR复核后降级）")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    video_duration = merged["video_duration_seconds"].to_numpy()
    audio_duration = merged["audio_duration_seconds"].to_numpy()
    audio_duration_by_sample = merged.set_index("sample")["audio_duration_seconds"]
    validation_checks = {
        "sample_count_is_20": len(merged) == 20,
        "each_sample_has_50_token_rows": bool(
            (tokens.groupby("sample").size() == 50).all()
        ),
        "word_intervals_ordered": bool(
            (words["start_seconds"] <= words["end_seconds"]).all()
        ),
        "word_intervals_within_audio": bool(
            all(
                group["end_seconds"].max()
                <= audio_duration_by_sample.loc[sample] + 1e-4
                for sample, group in words.groupby("sample")
            )
        ),
        "unaligned_feature_intervals_within_audio": bool(
            all(
                group["end_seconds"].max()
                <= audio_duration_by_sample.loc[sample] + 1e-4
                for sample, group in feature_times.groupby("sample")
            )
        ),
    }
    if not all(validation_checks.values()):
        raise RuntimeError(f"问题3阶段A结构校验失败: {validation_checks}")
    axes[1].plot(x, video_duration, "o-", label="视频流时长")
    axes[1].plot(x, audio_duration, "s--", label="音轨时长")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("附件4样本编号")
    axes[1].set_ylabel("时长（秒）")
    axes[1].set_title("视频流与音轨时长不同：解释时间轴以音轨0秒为共同原点")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.savefig(audit_dir / "附件4时间轴与证据质量.png", dpi=200)
    plt.close(fig)

    summary = {
        "sample_count": len(policy),
        "bert_interface_exact_match_count": int(policy["bert_interface_exact_match"].sum()),
        "text_word_time_reliability_counts": {
            str(key): int(value)
            for key, value in policy["text_word_time_reliability"].value_counts().items()
        },
        "model_visual_contribution_unusable_samples": policy.loc[
            ~policy["model_visual_contribution_usable"], "sample"
        ].tolist(),
        "vision_length_metadata_inconsistent_samples": merged.loc[
            ~merged["unaligned_vision_length_metadata_consistent"].astype(bool), "sample"
        ].tolist(),
        "model_text_truncated_samples": merged.loc[
            merged["truncated_word_count"] > 0, "sample"
        ].tolist(),
        "word_time_unusable_samples": sorted(unusable),
        "video_audio_duration_gap_over_0_1s_count": int(
            np.sum(np.abs(video_duration - audio_duration) > 0.1)
        ),
        "maximum_video_audio_duration_gap_seconds": float(
            np.max(np.abs(video_duration - audio_duration))
        ),
        "row_counts": {
            "word_intervals": len(words),
            "token_time_map": len(tokens),
            "unaligned_feature_time_map": len(feature_times),
        },
        "validation_checks": validation_checks,
        "stage_a_conclusion": (
            "20条均建立物理时间索引；15号逐词时间证据降级，13号正式模型视觉证据禁用，"
            "13/16号视觉长度元数据已纠错，07/18号明确50-token模型解释边界。"
        ),
    }
    (audit_dir / "stage_a_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
