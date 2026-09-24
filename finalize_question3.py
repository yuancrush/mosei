from __future__ import annotations

"""把问题3机器输出整理为写作人员可直接使用的中文表格。"""

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "outputs" / "exp024_attachment4_explanations"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "question3_stage_c"
MODALITY_NAMES = {"text": "文本", "audio": "音频", "vision": "视觉"}
TASK_NAMES = {"classification": "极性分类", "regression": "强度回归"}


def main() -> None:
    predictions = pd.read_csv(
        SOURCE_DIR / "predictions_and_contributions.csv", dtype={"sample": str}
    )
    evidence = pd.read_csv(SOURCE_DIR / "local_evidence.csv", dtype={"sample": str})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = pd.DataFrame(
        {
            "编号": predictions["sample"],
            "预测极性": predictions["classification_name"],
            "连续情感值": predictions["regression_score"].round(4),
            "情感强度绝对值": predictions["emotion_intensity_abs"].round(4),
            "分类投票一致率": predictions["classification_vote_agreement"].round(4),
            "分类主要参考模态": predictions["classification_primary_modality"].map(
                MODALITY_NAMES
            ),
            "分类文本作用程度": predictions["classification_text_strength"].round(4),
            "分类音频作用程度": predictions["classification_audio_strength"].round(4),
            "分类视觉作用程度": predictions["classification_vision_strength"].round(4),
            "回归主要参考模态": predictions["regression_primary_modality"].map(
                MODALITY_NAMES
            ),
            "回归文本作用程度": predictions["regression_text_strength"].round(4),
            "回归音频作用程度": predictions["regression_audio_strength"].round(4),
            "回归种子标准差": predictions["regression_seed_std"].round(4),
            "分类证据位置一致率": predictions[
                "classification_evidence_seed_agreement"
            ].round(4),
            "回归证据位置一致率": predictions[
                "regression_evidence_seed_agreement"
            ].round(4),
            "分类与连续值极性冲突": predictions[
                "classification_regression_polarity_conflict"
            ],
            "证据使用限制": predictions["stage_a_restrictions"],
        }
    )
    result.to_csv(
        OUTPUT_DIR / "附件4正式预测与解释摘要.csv", index=False, encoding="utf-8-sig"
    )

    local = pd.DataFrame(
        {
            "编号": evidence["sample"],
            "任务": evidence["task"].map(TASK_NAMES),
            "模态": evidence["modality"].map(MODALITY_NAMES),
            "位置块起点": evidence["start_position"],
            "位置块终点_不含": evidence["end_position_exclusive"],
            "反事实作用方向": evidence["importance_signed"].round(6),
            "反事实作用大小": evidence["importance_abs"].round(6),
            "种子标准差": evidence["seed_std"].round(6),
            "对应官方文本": evidence["evidence_words"],
            "开始秒": evidence["start_seconds"],
            "结束秒": evidence["end_seconds"],
            "时间对齐平均分": evidence["mean_alignment_score"],
            "关键帧文件": evidence["keyframe_file"],
            "证据限制": evidence["evidence_restriction"],
        }
    )
    local.to_csv(OUTPUT_DIR / "附件4局部证据摘要.csv", index=False, encoding="utf-8-sig")

    validation = {
        "prediction_rows": len(result),
        "local_evidence_rows": len(local),
        "visual_keyframe_rows": int(local["关键帧文件"].fillna("").ne("").sum()),
        "classification_counts": result["预测极性"].value_counts().to_dict(),
        "classification_regression_conflict_samples": result.loc[
            result["分类与连续值极性冲突"], "编号"
        ].tolist(),
        "special_restriction_samples": result.loc[
            result["证据使用限制"] != "无额外降级", "编号"
        ].tolist(),
        "ground_truth_available": False,
        "warning": "附件4无标签；本表是冻结模型推理，不能据此报告准确率或回灌训练。",
    }
    (OUTPUT_DIR / "finalization_summary.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
