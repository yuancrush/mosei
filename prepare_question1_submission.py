from __future__ import annotations

"""生成第一问论文及附件所需的全量汇总表、典型样本图和可复现提交包。

该脚本只读取原始视频和 `outputs/question1_features`，不修改官方数据。
"""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import soundfile as sf

from extract_question1_features import PROJECT_ROOT, extract_wav


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def find_attachment1() -> Path:
    candidates = list(
        PROJECT_ROOT.parent.glob(
            "**/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条"
        )
    )
    if len(candidates) != 1:
        raise FileNotFoundError(f"附件1视频目录数量应为1，实际为{len(candidates)}")
    return candidates[0]


def quality_label(status: str) -> str:
    return "高置信自动通过" if status == "pass" else "质量预警，已保留掩码与日志"


def build_full_summary(
    manifest: pd.DataFrame, log: pd.DataFrame, feature_dir: Path
) -> pd.DataFrame:
    records = []
    combined = manifest.merge(log, on="sample_id", validate="one_to_one")
    for row in combined.itertuples(index=False):
        data = np.load(feature_dir / Path(row.output_file).name)
        valid_token_count = int(data["text_bert"][1].sum())
        common = {
            "样本编号": row.sample_id,
            "原始视频相对路径": row.relative_path,
            "原始有效时长_秒": round(float(row.duration_s), 6),
            "词数": int(row.word_count),
            "对齐轴": "MMS强制对齐得到的词级物理时间区间",
            "50位置层有效token数": valid_token_count,
            "填充规则": "[CLS]/[SEP]/padding的音视频为0；无有效观测时为0并保留mask",
            "平均对齐置信度": round(float(row.mean_alignment_score), 6),
            "质量状态": quality_label(row.status),
            "特征文件": row.output_file,
        }
        records.extend(
            [
                {
                    **common,
                    "模态": "文本",
                    "特征定义": "bert-base-uncased最后一层隐状态",
                    "特征维度": 768,
                    "变长词级形状": f"{row.word_count}×768",
                    "50位置层形状": "50×768",
                    "原始帧/位置粒度": "WordPiece，词级为子词均值",
                    "有效覆盖率": 1.0,
                },
                {
                    **common,
                    "模态": "语音",
                    "特征定义": "openSMILE eGeMAPSv02的25维帧级LLD",
                    "特征维度": 25,
                    "变长词级形状": f"{row.word_count}×25",
                    "50位置层形状": "50×25",
                    "原始帧/位置粒度": "openSMILE帧，按词区间均值聚合",
                    "有效覆盖率": round(float(row.audio_word_coverage), 6),
                },
                {
                    **common,
                    "模态": "视觉",
                    "特征定义": "MediaPipe FaceMesh的35维归一化面部几何距离；小脸/侧脸采用人脸框裁剪回退",
                    "特征维度": 35,
                    "变长词级形状": f"{row.word_count}×35",
                    "50位置层形状": "50×35",
                    "原始帧/位置粒度": "10 FPS采样，按词区间均值聚合",
                    "有效覆盖率": round(float(row.visual_word_coverage), 6),
                },
            ]
        )
    return pd.DataFrame(records)


def read_video_frame(video_path: Path, time_s: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_s) * 1000)
    ok, frame = capture.read()
    capture.release()
    if ok:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 少数mp4的索引使OpenCV随机定位失败，使用同一环境中的FFmpeg解码单帧。
    for candidate_time in (time_s, time_s * 0.5, 0.1, 0.0):
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v", "error", "-i", str(video_path), "-ss", f"{max(0.0, candidate_time):.6f}",
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
        ]
        result = subprocess.run(command, capture_output=True)
        if result.returncode == 0 and result.stdout:
            decoded = cv2.imdecode(
                np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if decoded is not None:
                return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    raise RuntimeError(f"无法读取视频帧: {video_path} @ {time_s:.3f}s")


def standardized_energy(values: np.ndarray) -> np.ndarray:
    energy = np.linalg.norm(values - values.mean(axis=0, keepdims=True), axis=1)
    std = float(energy.std())
    return (energy - energy.mean()) / std if std > 1e-8 else np.zeros_like(energy)


def create_typical_figure(
    sample_id: str,
    manifest: pd.DataFrame,
    log: pd.DataFrame,
    feature_dir: Path,
    data_dir: Path,
    output_path: Path,
) -> None:
    row = manifest.loc[manifest["sample_id"] == sample_id].iloc[0]
    log_row = log.loc[log["sample_id"] == sample_id].iloc[0]
    data = np.load(feature_dir / Path(log_row["output_file"]).name)
    words = data["words"]
    intervals = data["word_intervals"]
    video_path = data_dir / Path(row["relative_path"])
    selected = np.unique(np.linspace(0, len(words) - 1, 5).round().astype(int))

    with tempfile.TemporaryDirectory(prefix="q1_figure_") as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        extract_wav(video_path, wav_path, 16000)
        waveform, sample_rate = sf.read(wav_path)

    fig = plt.figure(figsize=(15, 9))
    grid = GridSpec(3, 5, figure=fig, height_ratios=[2.2, 1.2, 1.1], hspace=0.38)
    for column, word_index in enumerate(selected):
        midpoint = float(intervals[word_index].mean())
        axis = fig.add_subplot(grid[0, column])
        axis.imshow(read_video_frame(video_path, midpoint))
        axis.set_title(
            f"{words[word_index]}\n{intervals[word_index, 0]:.2f}–{intervals[word_index, 1]:.2f}秒",
            fontsize=10,
        )
        axis.axis("off")

    audio_axis = fig.add_subplot(grid[1, :])
    times = np.arange(len(waveform)) / sample_rate
    step = max(1, len(times) // 12000)
    audio_axis.plot(times[::step], waveform[::step], color="#334155", linewidth=0.55)
    for word_index in selected:
        start, end = intervals[word_index]
        audio_axis.axvspan(start, end, color="#2563EB", alpha=0.22)
        audio_axis.text((start + end) / 2, 0.92, str(words[word_index]), rotation=35,
                        transform=audio_axis.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    audio_axis.set(xlabel="物理时间（秒）", ylabel="归一化幅值", title="转写词区间与原始语音波形")
    audio_axis.set_xlim(0, float(row["duration_s"]))

    feature_axis = fig.add_subplot(grid[2, :])
    energies = np.vstack(
        [
            standardized_energy(data["word_text"]),
            standardized_energy(data["word_audio"]),
            standardized_energy(data["word_vision"]),
        ]
    )
    image = feature_axis.imshow(energies, aspect="auto", cmap="RdYlBu_r", vmin=-2, vmax=2)
    feature_axis.set_yticks([0, 1, 2], ["文本", "语音", "视觉"])
    feature_axis.set_xticks(np.arange(len(words)), words, rotation=55, ha="right", fontsize=8)
    feature_axis.set_title("各词区间三模态特征相对变化强度（行内标准化）")
    fig.colorbar(image, ax=feature_axis, fraction=0.02, pad=0.01, label="标准化变化强度")
    fig.suptitle(
        f"典型样本的文本—语音—视觉词级对齐\n{sample_id.replace('$_$', ' / ')}",
        fontsize=16,
    )
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.06, right=0.96)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_zero_face_contact_sheet(
    manifest: pd.DataFrame, log: pd.DataFrame, data_dir: Path, output_path: Path
) -> None:
    zero_face = log.loc[log["face_frame_detection_ratio"] == 0]
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.5))
    for axis, (_, log_row) in zip(axes.flat, zero_face.iterrows()):
        row = manifest.loc[manifest["sample_id"] == log_row["sample_id"]].iloc[0]
        frame = read_video_frame(
            data_dir / Path(row["relative_path"]), float(row["duration_s"]) / 2
        )
        axis.imshow(frame)
        axis.set_title(str(log_row["sample_id"]).replace("$_$", " / "), fontsize=9)
        axis.axis("off")
    for axis in axes.flat[len(zero_face):]:
        axis.axis("off")
    fig.suptitle("人脸检出率为0的样本中帧诊断", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def package_readme() -> str:
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchaudio", "transformers", "opensmile", "mediapipe", "opencv-contrib-python", "numpy")
    }
    return f"""# 问题1可复现提交包

## 内容

- `features/`：附件1全部100条样本的npz特征。
- `问题1_全量特征汇总.csv`：按样本和模态展开的300行可追溯索引。
- `extraction_log.csv`：对齐置信度、模态覆盖率和质量预警。
- `read_question1_npz.py`：读取示例。
- `source/`：数据审计、特征提取、质量分析和环境配置。

## 核心版本

{json.dumps(versions, ensure_ascii=False, indent=2)}

## 复现

```powershell
conda env create -f environment-mosei.yml
conda activate mosei
python audit_attachment1.py
python extract_question1_features.py --output-dir outputs/question1_features
python upgrade_question1_visual.py
python analyze_question1_quality.py
python validate_question1_reference.py
python prepare_question1_submission.py
```

首次运行会下载 `torchaudio.pipelines.MMS_FA` 和 `bert-base-uncased`预训练权重。它们只用于特征提取与对齐，未引入外部情感数据。

## 时间轴

每个词的 `word_intervals[j]=[s_j,e_j)` 对应原视频的秒级物理时间。语音和视觉帧按该区间取均值。完整变长词级特征保存在 `word_*`字段；50位置是为与附件2建模接口保持一致的兼容层。
"""


def reader_script() -> str:
    return '''"""读取问题1单个样本特征的最小示例。"""
from pathlib import Path
import numpy as np

sample_path = next((Path(__file__).parent / "features").glob("*.npz"))
data = np.load(sample_path)
print("sample_id:", data["sample_id"].item())
print("word-level:", data["word_text"].shape, data["word_audio"].shape, data["word_vision"].shape)
print("aligned-50:", data["text"].shape, data["audio"].shape, data["vision"].shape)
print("first words and intervals:")
for word, interval in zip(data["words"][:5], data["word_intervals"][:5]):
    print(str(word), interval.tolist())
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_zip(
    output_path: Path, submission_dir: Path, feature_root: Path
) -> None:
    source_files = [
        PROJECT_ROOT / "audit_attachment1.py",
        PROJECT_ROOT / "extract_question1_features.py",
        PROJECT_ROOT / "upgrade_question1_visual.py",
        PROJECT_ROOT / "analyze_question1_quality.py",
        PROJECT_ROOT / "validate_question1_reference.py",
        PROJECT_ROOT / "prepare_question1_submission.py",
        PROJECT_ROOT / "environment-mosei.yml",
    ]
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted((feature_root / "samples").glob("*.npz")):
            archive.write(path, Path("问题1可复现提交包/features") / path.name)
        archive.write(feature_root / "extraction_log.csv", "问题1可复现提交包/extraction_log.csv")
        archive.write(feature_root / "summary.json", "问题1可复现提交包/feature_summary.json")
        archive.write(
            feature_root / "visual_upgrade_comparison.json",
            "问题1可复现提交包/visual_upgrade_comparison.json",
        )
        for name in (
            "问题1_全量特征汇总.csv",
            "问题1_复现说明.md",
            "read_question1_npz.py",
            "典型样本对齐验证.png",
            "零人脸检出样本诊断.png",
        ):
            archive.write(submission_dir / name, Path("问题1可复现提交包") / name)
        archive.write(
            PROJECT_ROOT / "outputs/question1_quality/summary.json",
            "问题1可复现提交包/quality_summary.json",
        )
        archive.write(
            PROJECT_ROOT / "reports/question1_reference_validation/summary.json",
            "问题1可复现提交包/reference_validation_summary.json",
        )
        archive.write(
            PROJECT_ROOT / "reports/question1_reference_validation/comparison.csv",
            "问题1可复现提交包/reference_validation_comparison.csv",
        )
        archive.write(
            PROJECT_ROOT / "reports/问题1完整求解与写作材料.md",
            "问题1可复现提交包/问题1完整求解与写作材料.md",
        )
        for path in source_files:
            archive.write(path, Path("问题1可复现提交包/source") / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成问题1全量汇总、典型图和提交包")
    parser.add_argument("--typical-sample", default="-UacrmKiTn4$_$10")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/question1_submission")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = find_attachment1()
    feature_root = PROJECT_ROOT / "outputs/question1_features"
    feature_dir = feature_root / "samples"
    manifest = pd.read_csv(PROJECT_ROOT / "reports/question1_audit/manifest.csv")
    log = pd.read_csv(feature_root / "extraction_log.csv")

    summary = build_full_summary(manifest, log, feature_dir)
    summary.to_csv(args.output_dir / "问题1_全量特征汇总.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "问题1_复现说明.md").write_text(package_readme(), encoding="utf-8")
    (args.output_dir / "read_question1_npz.py").write_text(reader_script(), encoding="utf-8")
    create_typical_figure(
        args.typical_sample,
        manifest,
        log,
        feature_dir,
        data_dir,
        args.output_dir / "典型样本对齐验证.png",
    )
    create_zero_face_contact_sheet(
        manifest, log, data_dir, args.output_dir / "零人脸检出样本诊断.png"
    )

    zip_path = args.output_dir / "问题1_可复现提交包.zip"
    create_zip(zip_path, args.output_dir, feature_root)
    package_summary = {
        "sample_count": int(log.shape[0]),
        "modality_summary_rows": int(summary.shape[0]),
        "typical_sample": args.typical_sample,
        "zip_size_bytes": zip_path.stat().st_size,
        "zip_size_mb": zip_path.stat().st_size / 1024**2,
        "under_50mb": zip_path.stat().st_size <= 50 * 1024**2,
        "sha256": sha256(zip_path),
    }
    (args.output_dir / "package_summary.json").write_text(
        json.dumps(package_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(package_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
