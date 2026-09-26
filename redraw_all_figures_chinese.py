"""从已有实验结果重绘项目内全部科研图，不重新训练或访问官方测试集。"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from evaluate_final_test_once import save_confusion
from evaluate_missingness import save_plot as save_missingness_plot
from evaluate_monte_carlo_missingness import create_figure as create_missingness_figure
from evaluate_repeated_explanations_and_interactions import create_figure as create_interaction_figure
from evaluate_uncertainty_calibration import create_figure as create_uncertainty_figure
from export_gate_and_training_diagnostics import plot_gate_response, plot_training_histories
from plot_style import apply_chinese_style
from prepare_question1_submission import (
    create_typical_figure,
    create_zero_face_contact_sheet,
    find_attachment1,
)
from train_baselines import save_confusion_matrix, save_history


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_NAMES = {
    "audio": "语音均值池化",
    "fusion": "三模态均值融合",
    "text": "文本均值池化",
    "vision": "视觉均值池化",
    "text_bigru": "文本双向GRU",
    "text_audio_bigru": "文本+语音双向GRU",
    "text_vision_bigru": "文本+视觉双向GRU",
    "text_audio_gated_bigru": "文本锚定语音门控融合",
    "fusion_uncertainty_bigru": "不确定性感知三模态融合",
    "text_bert_finetune": "BERT文本微调",
    "fusion_gated_bigru": "共享门控三模态融合",
    "fusion_task_gated_bigru": "任务特异门控三模态融合",
    "fusion_output_residual_bigru": "输出残差三模态融合",
}


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_rows(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict(orient="records")


def redraw_histories() -> int:
    count = 0
    for path in sorted((PROJECT_ROOT / "outputs").glob("**/history.json")):
        history = json.loads(path.read_text(encoding="utf-8"))
        model_name = MODEL_NAMES.get(path.parent.name, path.parent.name)
        save_history(history, path.with_name("training_history.png"), f"{model_name}训练过程")
        count += 1
    return count


def redraw_historical_confusions() -> int:
    count = 0
    baseline_root = PROJECT_ROOT / "outputs" / "baselines_seed2026"
    for model in ("text", "audio", "vision", "fusion"):
        path = baseline_root / model / "test_predictions.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        save_confusion_matrix(
            metrics,
            path.with_name("test_confusion_matrix.png"),
            f"{MODEL_NAMES[model]}测试集混淆矩阵",
        )
        count += 1
    final_predictions = pd.read_csv(
        PROJECT_ROOT / "outputs" / "exp030_final_test_once" / "predictions.csv"
    )
    save_confusion(
        final_predictions["classification_true"].to_numpy(),
        final_predictions["classification_pred"].to_numpy(),
        PROJECT_ROOT / "outputs" / "exp030_final_test_once" / "官方测试集混淆矩阵.png",
    )
    return count + 1


def redraw_missingness_stress() -> int:
    count = 0
    for path in sorted((PROJECT_ROOT / "outputs").glob("**/joint_middle_degradation.png")):
        rows = numeric_rows(path.with_name("metrics.csv"))
        save_missingness_plot(rows, path)
        count += 1
    return count


def redraw_final_diagnostics() -> int:
    exp027 = PROJECT_ROOT / "outputs" / "exp027_monte_carlo_missingness"
    clean = json.loads((exp027 / "summary.json").read_text(encoding="utf-8"))["clean"]
    create_missingness_figure(
        numeric_rows(exp027 / "summary.csv"), clean, exp027 / "连续缺失鲁棒性响应.png"
    )

    exp028 = PROJECT_ROOT / "outputs" / "exp028_uncertainty_calibration"
    create_uncertainty_figure(
        numeric_rows(exp028 / "reliability_uncalibrated.csv"),
        numeric_rows(exp028 / "reliability_calibrated.csv"),
        numeric_rows(exp028 / "risk_coverage.csv"),
        exp028 / "不确定性校准与风险覆盖.png",
    )

    exp029 = PROJECT_ROOT / "outputs" / "exp029_repeated_explanations_interactions"
    create_interaction_figure(
        numeric_rows(exp029 / "per_sample.csv"), exp029 / "重复随机解释与模态交互.png"
    )

    exp031 = PROJECT_ROOT / "outputs" / "exp031_gate_and_training_diagnostics"
    plot_gate_response(
        numeric_rows(exp031 / "门控分箱统计.csv"), exp031 / "图6-3_置信门控响应曲线.png"
    )
    plot_training_histories(
        numeric_rows(exp031 / "三种子训练轨迹.csv"),
        numeric_rows(exp031 / "选模与早停轮次.csv"),
        exp031 / "图6-4_多随机种子训练与早停.png",
    )
    return 5


def redraw_question1_submission() -> int:
    """仅重绘问题1两张论文图，不生成阶段压缩包。"""

    manifest = pd.read_csv(PROJECT_ROOT / "reports" / "question1_audit" / "manifest.csv")
    feature_root = PROJECT_ROOT / "outputs" / "question1_features"
    log = pd.read_csv(feature_root / "extraction_log.csv")
    output_dir = PROJECT_ROOT / "outputs" / "question1_submission"
    create_typical_figure(
        "-aqamKhZ1Ec$_$0",
        manifest,
        log,
        feature_root / "samples",
        find_attachment1(),
        output_dir / "典型样本对齐验证.png",
    )
    create_zero_face_contact_sheet(
        manifest, log, find_attachment1(), output_dir / "零人脸检出样本诊断.png"
    )
    return 2


def run_plot_only_scripts() -> None:
    scripts = [
        "summarize_results.py",
        "summarize_day1.py",
        "audit_missingness.py",
        "analyze_question1_quality.py",
        "summarize_question3_stage_a.py",
        "summarize_explanation_validation.py",
        "visualize_attachment4_explanations.py",
    ]
    for script in scripts:
        subprocess.run([sys.executable, str(PROJECT_ROOT / script)], cwd=PROJECT_ROOT, check=True)


def write_figure_manifest() -> int:
    """记录全部PNG的尺寸与位置，供写作人员选图和复核。"""

    rows = []
    for base in (PROJECT_ROOT / "outputs", PROJECT_ROOT / "reports"):
        for path in sorted(base.glob("**/*.png")):
            with Image.open(path) as image:
                width, height = image.size
            rows.append(
                {
                    "相对路径": path.relative_to(PROJECT_ROOT).as_posix(),
                    "宽度_像素": width,
                    "高度_像素": height,
                    "文件大小_字节": path.stat().st_size,
                    "处理口径": "Python代码重绘",
                }
            )
    output_dir = PROJECT_ROOT / "reports" / "figure_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "全项目图表清单.csv", index=False, encoding="utf-8-sig")
    return len(rows)


def main() -> None:
    apply_chinese_style()
    history_count = redraw_histories()
    confusion_count = redraw_historical_confusions()
    stress_count = redraw_missingness_stress()
    diagnostic_count = redraw_final_diagnostics()
    question1_submission_count = redraw_question1_submission()
    run_plot_only_scripts()
    image_count = write_figure_manifest()
    print(
        json.dumps(
            {
                "training_history_redrawn": history_count,
                "confusion_matrix_redrawn": confusion_count,
                "missingness_stress_redrawn": stress_count,
                "final_diagnostic_redrawn": diagnostic_count,
                "question1_submission_redrawn": question1_submission_count,
                "project_png_count": image_count,
                "official_test_rerun": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
