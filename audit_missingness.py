from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_style import apply_chinese_style


apply_chinese_style()


PROJECT_ROOT = Path(__file__).resolve().parent


def find_attachment3(root: Path) -> Path:
    candidates = [
        path
        for path in root.parent.glob("**/附件3-模态缺失特征样本/对齐版本")
        if path.is_dir()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(f"附件3对齐版本目录数量应为1，实际为{len(candidates)}: {candidates}")
    return candidates[0]


def find_reference_aligned(root: Path) -> Path:
    candidates = list(root.parent.glob("**/aligned_50.pkl"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"aligned_50.pkl数量应为1，实际为{len(candidates)}")
    return candidates[0]


def contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """返回布尔掩码中True连续区间，采用[start, end)下标。"""
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def load_masks(path: Path) -> Dict[str, np.ndarray]:
    # 赛题官方pickle为可信输入；不要对来源不明的pickle复用此读取方式。
    with path.open("rb") as handle:
        sample = pickle.load(handle)["test"]
    return {
        "text": np.asarray(sample["text_bert"])[0, 1] > 0,
        "audio": np.any(np.asarray(sample["audio"])[0] != 0, axis=-1),
        "vision": np.any(np.asarray(sample["vision"])[0] != 0, axis=-1),
    }


def rows_for_sample(path: Path) -> Tuple[dict, List[dict]]:
    masks = load_masks(path)
    union = np.logical_or.reduce(list(masks.values()))
    effective_length = int(np.flatnonzero(union)[-1] + 1) if union.any() else 0
    summary = {"sample": path.stem, "effective_length": effective_length}
    intervals: List[dict] = []
    for modality, available in masks.items():
        missing = ~available[:effective_length]
        runs = contiguous_runs(missing)
        summary[f"{modality}_missing_steps"] = int(missing.sum())
        summary[f"{modality}_missing_ratio"] = (
            float(missing.mean()) if effective_length else 0.0
        )
        summary[f"{modality}_run_count"] = len(runs)
        for run_index, (start, end) in enumerate(runs, start=1):
            intervals.append(
                {
                    "sample": path.stem,
                    "modality": modality,
                    "run_index": run_index,
                    "start": start,
                    "end_exclusive": end,
                    "duration": end - start,
                    "relative_start": start / effective_length,
                    "relative_end": end / effective_length,
                }
            )
    audio_missing = ~masks["audio"][:effective_length]
    vision_missing = ~masks["vision"][:effective_length]
    summary["audio_vision_same_mask"] = bool(np.array_equal(audio_missing, vision_missing))
    summary["joint_missing_steps"] = int(np.logical_and(audio_missing, vision_missing).sum())
    summary["union_missing_steps"] = int(np.logical_or(audio_missing, vision_missing).sum())
    return summary, intervals


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: List[dict], path: Path) -> None:
    x = np.arange(1, len(rows) + 1)
    audio = np.array([row["audio_missing_ratio"] for row in rows])
    vision = np.array([row["vision_missing_ratio"] for row in rows])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, audio, "o-", label="语音")
    ax.plot(x, vision, "s--", label="视觉")
    ax.axhline(audio.mean(), color="tab:blue", alpha=0.35, linestyle=":")
    ax.set_xlabel("样本编号")
    ax.set_ylabel("有效序列内的缺失比例")
    ax.set_ylim(0, 1)
    ax.set_title("附件3局部模态缺失比例")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="审计附件3对齐样本的局部模态缺失模式")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "attachment3_audit"
    )
    args = parser.parse_args()
    data_dir = args.data_dir or find_attachment3(PROJECT_ROOT)
    files = sorted(data_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"未在{data_dir}发现pkl文件")

    samples: List[dict] = []
    intervals: List[dict] = []
    for path in files:
        sample_row, sample_intervals = rows_for_sample(path)
        samples.append(sample_row)
        intervals.extend(sample_intervals)

    reference_path = find_reference_aligned(PROJECT_ROOT)
    with reference_path.open("rb") as handle:
        reference = pickle.load(handle)["valid"]
    reference_text_mask = np.asarray(reference["text_bert"])[:, 1, :] > 0
    reference_missing = {}
    for modality in ("audio", "vision"):
        available = np.any(np.asarray(reference[modality]) != 0, axis=-1)
        ratios = []
        for index, text_mask in enumerate(reference_text_mask):
            length = int(np.flatnonzero(text_mask)[-1] + 1)
            ratios.append(float((~available[index, :length]).mean()))
        reference_missing[modality] = float(np.mean(ratios))

    summary = {
        "data_dir": str(data_dir.resolve()),
        "sample_count": len(samples),
        "effective_length": {
            "min": min(row["effective_length"] for row in samples),
            "mean": float(np.mean([row["effective_length"] for row in samples])),
            "max": max(row["effective_length"] for row in samples),
        },
        "mean_missing_ratio": {
            modality: float(np.mean([row[f"{modality}_missing_ratio"] for row in samples]))
            for modality in ("text", "audio", "vision")
        },
        "reference_valid_mean_zero_ratio": reference_missing,
        "excess_over_reference": {
            modality: float(
                np.mean([row[f"{modality}_missing_ratio"] for row in samples])
                - reference_missing[modality]
            )
            for modality in ("audio", "vision")
        },
        "samples_with_missing": {
            modality: sum(row[f"{modality}_missing_steps"] > 0 for row in samples)
            for modality in ("text", "audio", "vision")
        },
        "audio_vision_same_mask_count": sum(
            row["audio_vision_same_mask"] for row in samples
        ),
        "audio_missing_ratio_quantiles": np.quantile(
            [row["audio_missing_ratio"] for row in samples], [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
        "audio_run_count_distribution": dict(
            Counter(str(row["audio_run_count"]) for row in samples)
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "samples.csv", samples)
    write_csv(args.output_dir / "intervals.csv", intervals)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    save_plot(samples, args.output_dir / "missing_ratio.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
