from __future__ import annotations

"""生成附件4正式解释的中文总览图与典型案例图。"""

import json
import textwrap
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp024_attachment4_explanations"
REPORT_DIR = PROJECT_ROOT / "reports" / "question3_stage_c"


def configure_chinese_font() -> None:
    for path in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取关键帧: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> None:
    configure_chinese_font()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(
        OUTPUT_DIR / "predictions_and_contributions.csv", dtype={"sample": str}
    )
    evidence = pd.read_csv(OUTPUT_DIR / "local_evidence.csv", dtype={"sample": str})
    x = np.arange(len(predictions))

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)
    bottom = np.zeros(len(predictions))
    for column, label, color in (
        ("classification_text_strength", "文本", "#2563EB"),
        ("classification_audio_strength", "音频", "#D97706"),
        ("classification_vision_strength", "视觉", "#16A34A"),
    ):
        values = predictions[column].to_numpy()
        axes[0].bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[0].set_xticks(x, predictions["sample"])
    axes[0].set_ylabel("归一化反事实作用程度")
    axes[0].set_title("附件4三模态分类作用程度（绝对变化归一化）")
    axes[0].legend(ncol=3)

    class_colors = predictions["classification_id"].map(
        {0: "#DC2626", 1: "#64748B", 2: "#16A34A"}
    )
    axes[1].bar(x, predictions["regression_score"], color=class_colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, predictions["sample"])
    axes[1].set_ylabel("预测情感强度")
    axes[1].set_title("极性与强度输出（红/灰/绿分别为负向/中性/正向）")
    axes[1].grid(axis="y", alpha=0.2)

    axes[2].plot(
        x,
        predictions["classification_vote_agreement"],
        "o-",
        label="分类投票一致率",
    )
    axes[2].plot(
        x,
        predictions["classification_evidence_seed_agreement"],
        "s--",
        label="分类证据位置一致率",
    )
    axes[2].plot(
        x,
        predictions["regression_evidence_seed_agreement"],
        "^:",
        label="回归证据位置一致率",
    )
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xticks(x, predictions["sample"])
    axes[2].set_xlabel("附件4样本编号")
    axes[2].set_ylabel("一致率")
    axes[2].set_title("预测与解释的多种子稳定性")
    axes[2].legend(ncol=3)
    axes[2].grid(alpha=0.2)
    fig.savefig(REPORT_DIR / "附件4预测贡献与稳定性总览.png", dpi=200)
    plt.close(fig)

    # 典型案例由预先声明的科学角色选取，不按“图好看”选择：稳定文本、音频抑制、视觉冲突。
    cases = [
        ("09", "稳定文本主导的负向样本"),
        ("03", "音频抑制作用较强的中性样本"),
        ("19", "视觉抑制且分类/回归冲突样本"),
    ]
    fig, axes = plt.subplots(len(cases), 3, figsize=(16, 12), constrained_layout=True)
    for row_index, (sample, role) in enumerate(cases):
        prediction = predictions.loc[predictions["sample"] == sample].iloc[0]
        strengths = [
            prediction["classification_text_strength"],
            prediction["classification_audio_strength"],
            prediction["classification_vision_strength"],
        ]
        signed = [
            prediction["classification_text_signed_effect"],
            prediction["classification_audio_signed_effect"],
            prediction["classification_vision_signed_effect"],
        ]
        axes[row_index, 0].bar(
            ["文本", "音频", "视觉"], strengths, color=["#2563EB", "#D97706", "#16A34A"]
        )
        axes[row_index, 0].set_ylim(0, 1)
        axes[row_index, 0].set_ylabel("作用程度")
        axes[row_index, 0].set_title(f"{sample}号：{role}")
        for column, value in enumerate(signed):
            axes[row_index, 0].text(
                column,
                strengths[column] + 0.04,
                f"方向{value:+.3f}",
                ha="center",
                fontsize=9,
            )

        probabilities = [
            prediction["prob_negative"],
            prediction["prob_neutral"],
            prediction["prob_positive"],
        ]
        axes[row_index, 1].bar(
            ["负向", "中性", "正向"], probabilities, color=["#DC2626", "#64748B", "#16A34A"]
        )
        axes[row_index, 1].set_ylim(0, 1)
        axes[row_index, 1].set_title(
            f"预测：{prediction['classification_name']}；强度{prediction['regression_score']:.3f}"
        )
        axes[row_index, 1].set_ylabel("平均概率")

        visual = evidence.loc[
            (evidence["sample"] == sample)
            & (evidence["task"] == "classification")
            & (evidence["modality"] == "vision")
        ].iloc[0]
        frame_path = PROJECT_ROOT / str(visual["keyframe_file"])
        axes[row_index, 2].imshow(read_image(frame_path))
        axes[row_index, 2].axis("off")
        time_text = f"{visual['start_seconds']:.2f}—{visual['end_seconds']:.2f}秒"
        words = " ".join(textwrap.wrap(str(visual["evidence_words"]), width=28))
        axes[row_index, 2].set_title(
            f"视觉候选区间：{time_text}\n对应文本：{words}\n"
            f"视觉反事实方向：{prediction['classification_vision_signed_effect']:+.4f}",
            fontsize=10,
        )
    fig.savefig(REPORT_DIR / "附件4典型解释案例.png", dpi=200)
    plt.close(fig)

    summary = {
        "overview_figure": "附件4预测贡献与稳定性总览.png",
        "case_figure": "附件4典型解释案例.png",
        "case_selection": {
            "09": "稳定文本主导",
            "03": "验证音频抑制/纠偏",
            "19": "验证视觉抑制及任务冲突",
        },
        "note": "英文只来自附件官方原文证据，图题、坐标、图例均为中文。",
    }
    (REPORT_DIR / "visualization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
