"""按白名单生成唯一最终提交附件，并执行数量、体积与禁用项检查。"""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUBMISSION_DIR = ROOT / "submission"
README = SUBMISSION_DIR / "README_提交说明.md"
ZIP_PATH = SUBMISSION_DIR / "研究生数学建模E题_最终提交附件.zip"
MANIFEST_PATH = SUBMISSION_DIR / "最终附件_内容清单.csv"
SUMMARY_PATH = SUBMISSION_DIR / "package_summary.json"
MAX_BYTES = 50 * 1024 * 1024

CLASSIFICATION_WEIGHTS = [
    ROOT / "outputs" / run / "fusion_output_residual_bigru" / "best.pt"
    for run in (
        "exp020c_output_residual",
        "exp020c_output_residual_seed2027",
        "exp020c_output_residual_seed2028",
    )
]
REGRESSION_WEIGHTS = [
    ROOT / "outputs" / f"exp014_text_audio_seed{seed}" / "text_audio_bigru" / "best.pt"
    for seed in (2026, 2027, 2028)
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def collect_files() -> list[tuple[Path, str, str]]:
    files: list[tuple[Path, str, str]] = []

    def add(path: Path, archive: str, role: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"白名单文件缺失：{path}")
        files.append((path, archive.replace("\\", "/"), role))

    def add_tree(directory: Path, prefix: str, role: str, pattern: str = "*") -> None:
        for path in sorted(directory.rglob(pattern)):
            if path.is_file():
                add(path, f"{prefix}/{path.relative_to(directory).as_posix()}", role)

    add(README, "00_提交说明.md", "总说明")

    feature_dir = ROOT / "outputs" / "question1_features"
    feature_files = sorted((feature_dir / "samples").glob("*.npz"))
    if len(feature_files) != 100:
        raise ValueError(f"问题1特征应为100条，实际{len(feature_files)}条")
    for path in feature_files:
        add(path, f"01_问题1特征与对齐/features/{path.name}", "问题1三模态特征")
    add(feature_dir / "extraction_log.csv", "01_问题1特征与对齐/extraction_log.csv", "问题1提取审计")
    add(feature_dir / "summary.json", "01_问题1特征与对齐/feature_summary.json", "问题1提取审计")
    add(
        feature_dir / "visual_upgrade_comparison.json",
        "01_问题1特征与对齐/visual_upgrade_comparison.json",
        "问题1提取审计",
    )
    q1_submission = ROOT / "outputs" / "question1_submission"
    for name in (
        "问题1_全量特征汇总.csv",
        "问题1_复现说明.md",
        "read_question1_npz.py",
        "典型样本对齐验证.png",
        "零人脸检出样本诊断.png",
    ):
        add(q1_submission / name, f"01_问题1特征与对齐/{name}", "问题1正式交付")
    add_tree(ROOT / "outputs" / "question1_quality", "01_问题1特征与对齐/质量审计", "问题1质量审计")
    add_tree(ROOT / "reports" / "question1_stage_a", "01_问题1特征与对齐/人工复核", "问题1人工复核")

    attachment3 = ROOT / "outputs" / "exp020c_attachment3_predictions"
    prediction3 = attachment3 / "predictions.csv"
    if count_csv_rows(prediction3) != 30:
        raise ValueError("附件3正式预测必须恰好30条")
    add(prediction3, "02_问题2预测与模型/附件3正式预测.csv", "问题2正式预测")
    add(attachment3 / "metadata.json", "02_问题2预测与模型/附件3预测说明.json", "问题2正式预测")
    for index, path in enumerate(CLASSIFICATION_WEIGHTS, start=1):
        add(path, f"02_问题2预测与模型/weights/classification_seed{2025 + index}.pt", "冻结分类权重")
    for index, path in enumerate(REGRESSION_WEIGHTS, start=1):
        add(path, f"02_问题2预测与模型/weights/regression_seed{2025 + index}.pt", "冻结回归权重")
    for directory, prefix in (
        (ROOT / "outputs" / "exp020c_multiseed_experts", "正式验证"),
        (ROOT / "outputs" / "exp021_ensemble_missing_stress", "连续缺失压力"),
        (ROOT / "outputs" / "exp027_monte_carlo_missingness", "蒙特卡洛连续缺失"),
        (ROOT / "outputs" / "exp028_uncertainty_calibration", "不确定性"),
        (ROOT / "outputs" / "exp031_gate_and_training_diagnostics", "门控与训练轨迹"),
    ):
        add_tree(directory, f"02_问题2预测与模型/{prefix}", f"问题2{prefix}")
    add(ROOT / "outputs" / "exp030_final_test_once" / "summary.json", "02_问题2预测与模型/唯一官方test结果.json", "最终泛化结果")
    add(ROOT / "outputs" / "exp030_final_test_once" / "官方测试集混淆矩阵.png", "02_问题2预测与模型/官方测试集混淆矩阵.png", "最终泛化结果")

    q3_summary = ROOT / "reports" / "question3_stage_c" / "附件4正式预测与解释摘要.csv"
    if count_csv_rows(q3_summary) != 20:
        raise ValueError("附件4正式预测必须恰好20条")
    add_tree(ROOT / "reports" / "question3_stage_c", "03_问题3预测与解释/正式报告", "问题3正式报告")
    add_tree(ROOT / "outputs" / "exp024_attachment4_explanations", "03_问题3预测与解释/模型输出", "问题3正式模型输出")
    add_tree(ROOT / "outputs" / "question3_stage_a_timeline", "03_问题3预测与解释/时间映射", "问题3时间映射")
    add_tree(ROOT / "outputs" / "exp029_repeated_explanations_interactions", "03_问题3预测与解释/解释验证", "问题3解释验证")

    source_files = [
        "plot_style.py",
        "audit_attachment1.py",
        "extract_question1_features.py",
        "upgrade_question1_visual.py",
        "finalize_question1_stage_a.py",
        "review_question1_alignment.py",
        "analyze_question1_quality.py",
        "validate_question1_reference.py",
        "prepare_question1_submission.py",
        "audit_data.py",
        "audit_missingness.py",
        "train_baselines.py",
        "prepare_attachment3.py",
        "predict_attachment3.py",
        "evaluate_exp020c_ensemble.py",
        "evaluate_monte_carlo_missingness.py",
        "evaluate_uncertainty_calibration.py",
        "export_gate_and_training_diagnostics.py",
        "audit_attachment4.py",
        "align_attachment4_timeline.py",
        "review_attachment4_alignment.py",
        "evaluate_explanation_methods.py",
        "evaluate_repeated_explanations_and_interactions.py",
        "explain_attachment4.py",
        "finalize_question3.py",
        "visualize_attachment4_explanations.py",
    ]
    for name in source_files:
        add(ROOT / name, f"04_复现源码/{name}", "复现源码")
    add_tree(ROOT / "src" / "mosei", "04_复现源码/src/mosei", "核心模型源码", "*.py")
    add(ROOT / "environment-mosei.yml", "04_复现源码/environment-mosei.yml", "环境配置")

    experiment_names = [
        "Q1-STAGE3_最终冻结与独立ASR复核.md",
        "EXP-020C_三模态输出残差纠错.md",
        "EXP-021_正式集成缺失压力与不确定性.md",
        "EXP-023_三路互证解释.md",
        "EXP-024_附件4正式解释.md",
        "EXP-025_问题3冻结.md",
        "EXP-026_最终轻量校准.md",
        "EXP-027_蒙特卡洛连续缺失统计.md",
        "EXP-028_不确定性校准与风险覆盖.md",
        "EXP-029_重复随机解释与模态交互.md",
        "EXP-030_官方测试唯一评估.md",
        "EXP-031_门控响应与多种子训练轨迹.md",
        "EXP-032_最终附件白名单与清理.md",
        "EXP-033_全项目图表中文化与样式统一.md",
    ]
    for name in experiment_names:
        add(ROOT / "experiments" / name, f"05_实验说明/{name}", "最终实验说明")
    handoff_names = [
        "README.md",
        "Q1_2026-09-25_典型图重构与文档纠偏交接.md",
        "Q2_2026-09-26_门控响应与训练轨迹交接.md",
        "Q3_2026-09-25_重复随机解释与模态交互交接.md",
        "FINAL_2026-09-26_唯一提交附件交接.md",
        "FINAL_2026-09-26_全项目图表中文化交接.md",
    ]
    for name in handoff_names:
        add(
            ROOT / "reports" / "writing_handoffs" / name,
            f"05_实验说明/最新写作交接/{name}",
            "最新写作交接",
        )
    add(
        ROOT / "reports" / "figure_audit" / "术语与样式规范.md",
        "05_实验说明/图表术语与样式规范.md",
        "图表规范",
    )

    forbidden = ("smoke", "exp016", "候选", "pseudo")
    for _, archive, _ in files:
        if any(token.lower() in archive.lower() for token in forbidden):
            raise ValueError(f"压缩包含禁用候选项：{archive}")
    archives = [archive for _, archive, _ in files]
    if len(archives) != len(set(archives)):
        raise ValueError("压缩包内存在重名路径")
    return files


def main() -> None:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    files = collect_files()
    manifest = []
    for source, archive, role in files:
        manifest.append(
            {
                "archive_path": archive,
                "role": role,
                "size_bytes": source.stat().st_size,
                "sha256": sha256(source),
                "source_path": source.relative_to(ROOT).as_posix(),
            }
        )
    with MANIFEST_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, target, _ in files:
            archive.write(source, target)
        archive.write(MANIFEST_PATH, "00_最终附件_内容清单.csv")

    size = ZIP_PATH.stat().st_size
    if size > MAX_BYTES:
        ZIP_PATH.unlink()
        raise ValueError(f"最终附件{size / 1024**2:.2f}MB，超过50MB限制")
    summary = {
        "package": ZIP_PATH.name,
        "file_count": len(files) + 1,
        "question1_feature_count": 100,
        "attachment3_prediction_count": 30,
        "attachment4_prediction_count": 20,
        "classification_weight_count": len(CLASSIFICATION_WEIGHTS),
        "regression_weight_count": len(REGRESSION_WEIGHTS),
        "size_bytes": size,
        "size_mb": size / 1024**2,
        "under_50mb": True,
        "sha256": sha256(ZIP_PATH),
        "excluded": [
            "原始赛事数据",
            "smoke输出",
            "EXP-016旧附件3预测",
            "淘汰的BERT微调权重",
            "历史候选模型权重",
            "官方test逐样本预测",
        ],
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
