from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent


def validate_sample(path: Path) -> dict[str, bool]:
    data = np.load(path)
    word_count = len(data["words"])
    token_word_index = data["token_word_index"]
    nonword = token_word_index < 0
    intervals = data["word_intervals"]
    return {
        "fixed_shapes_valid": (
            data["text"].shape == (50, 768)
            and data["audio"].shape == (50, 25)
            and data["vision"].shape == (50, 35)
            and data["text_bert"].shape == (3, 50)
        ),
        "word_shapes_valid": (
            data["word_text"].shape == (word_count, 768)
            and data["word_audio"].shape == (word_count, 25)
            and data["word_vision"].shape == (word_count, 35)
            and intervals.shape == (word_count, 2)
        ),
        "finite_values": all(
            np.isfinite(data[key]).all()
            for key in ("text", "audio", "vision", "word_text", "word_audio", "word_vision")
        ),
        "interval_order_valid": bool(
            np.all(intervals[:, 0] <= intervals[:, 1])
            and np.all(intervals[1:, 0] >= intervals[:-1, 0])
        ),
        "nonword_av_zero": bool(
            np.all(data["audio"][nonword] == 0) and np.all(data["vision"][nonword] == 0)
        ),
    }


def plot_overview(log: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(log["mean_alignment_score"], bins=np.linspace(0, 1, 21), color="#2563EB")
    axes[0, 0].axvline(0.6, color="#DC2626", linestyle="--")
    axes[0, 0].set(title="Mean forced-alignment score", xlabel="Score", ylabel="Samples")

    axes[0, 1].hist(log["face_frame_detection_ratio"], bins=np.linspace(0, 1, 21), color="#0F766E")
    axes[0, 1].axvline(0.7, color="#DC2626", linestyle="--")
    axes[0, 1].set(title="Face detection coverage", xlabel="Detected frame ratio", ylabel="Samples")

    colors = log["status"].map({"pass": "#16A34A", "review": "#DC2626"})
    axes[1, 0].scatter(log["word_count"], log["mean_alignment_score"], c=colors, alpha=0.75)
    axes[1, 0].set(title="Alignment by transcript length", xlabel="Words", ylabel="Mean score", ylim=(0, 1.02))

    cause_counts = {
        "Alignment mean": int((log["mean_alignment_score"] < 0.6).sum()),
        "Low-score words": int((log["low_alignment_word_ratio"] > 0.25).sum()),
        "Face detection": int((log["face_frame_detection_ratio"] < 0.7).sum()),
        "Visual coverage": int((log["visual_word_coverage"] < 0.7).sum()),
        "50-token truncation": int((log["truncated_word_count"] > 0).sum()),
    }
    axes[1, 1].barh(list(cause_counts), list(cause_counts.values()), color="#F59E0B")
    axes[1, 1].set(title="Review triggers (overlapping)", xlabel="Samples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_alignment_examples(log: pd.DataFrame, sample_dir: Path, output_path: Path) -> None:
    examples = [
        log.loc[log["mean_alignment_score"].idxmax()],
        log.loc[log["mean_alignment_score"].idxmin()],
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for axis, row in zip(axes, examples):
        data = np.load(sample_dir / Path(row["output_file"]).name)
        words = data["words"][:25]
        intervals = data["word_intervals"][:25]
        scores = data["alignment_scores"][:25]
        for index, (word, interval, score) in enumerate(zip(words, intervals, scores)):
            axis.barh(index, interval[1] - interval[0], left=interval[0], color=plt.cm.RdYlGn(float(score)))
            axis.text(interval[0], index, str(word), va="center", fontsize=7)
        axis.set_yticks([])
        axis.set_xlabel("Physical time (s)")
        display_id = str(row["sample_id"]).replace("$", r"\$")
        axis.set_title(f'{display_id} | mean score={row["mean_alignment_score"]:.3f}')
        axis.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="分析并验证第一问提取结果")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "outputs/question1_features")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/question1_quality")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log = pd.read_csv(args.input_dir / "extraction_log.csv")
    sample_dir = args.input_dir / "samples"
    validation_rows = []
    for row in log.itertuples(index=False):
        checks = validate_sample(sample_dir / Path(row.output_file).name)
        validation_rows.append({"sample_id": row.sample_id, **checks})
    validation = pd.DataFrame(validation_rows)
    merged = log.merge(validation, on="sample_id", validate="one_to_one")

    causes = {
        "low_mean_alignment": log["mean_alignment_score"] < 0.6,
        "many_low_score_words": log["low_alignment_word_ratio"] > 0.25,
        "low_face_detection": log["face_frame_detection_ratio"] < 0.7,
        "low_visual_word_coverage": log["visual_word_coverage"] < 0.7,
        "aligned50_truncation": log["truncated_word_count"] > 0,
    }
    for name, mask in causes.items():
        merged[name] = mask.to_numpy()
    review = merged[merged["status"] == "review"].copy()
    review.to_csv(args.output_dir / "review_samples.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(args.output_dir / "validation_checks.csv", index=False, encoding="utf-8-sig")

    summary = {
        "sample_count": len(log),
        "pass_count": int((log["status"] == "pass").sum()),
        "review_count": int((log["status"] == "review").sum()),
        "mean_alignment_score_quantiles": log["mean_alignment_score"].quantile([0, .25, .5, .75, 1]).tolist(),
        "visual_word_coverage_quantiles": log["visual_word_coverage"].quantile([0, .25, .5, .75, 1]).tolist(),
        "face_detection_ratio_quantiles": log["face_frame_detection_ratio"].quantile([0, .25, .5, .75, 1]).tolist(),
        "review_cause_counts": {name: int(mask.sum()) for name, mask in causes.items()},
        "zero_face_detection_count": int((log["face_frame_detection_ratio"] == 0).sum()),
        "all_fixed_shapes_valid": bool(validation["fixed_shapes_valid"].all()),
        "all_word_shapes_valid": bool(validation["word_shapes_valid"].all()),
        "all_values_finite": bool(validation["finite_values"].all()),
        "all_intervals_ordered": bool(validation["interval_order_valid"].all()),
        "all_nonword_av_zero": bool(validation["nonword_av_zero"].all()),
        "output_npz_count": len(list(sample_dir.glob("*.npz"))),
        "output_size_bytes": sum(path.stat().st_size for path in sample_dir.glob("*.npz")),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    plot_overview(log, args.output_dir / "quality_overview.png")
    plot_alignment_examples(log, sample_dir, args.output_dir / "alignment_examples.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
