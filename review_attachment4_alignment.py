from __future__ import annotations

"""用独立Whisper ASR复核附件4中MMS平均分低于0.6的时间轴。

复核只用于解释证据质量分层；不修改官方文本，不生成伪标签，也不参与模型调参。
"""

import argparse
import json
import pickle
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import pipeline

from align_attachment4_timeline import find_attachment4
from extract_question1_features import extract_wav


PROJECT_ROOT = Path(__file__).resolve().parent
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)


def load_trusted_pickle(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    return obj["test"] if "test" in obj and isinstance(obj["test"], Mapping) else obj


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower().replace("’", "'"))


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_word in enumerate(left, start=1):
        current = [i]
        for j, right_word in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + int(left_word != right_word),
                )
            )
        previous = current
    return previous[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="复核附件4低置信度强制对齐样本")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--model", default="openai/whisper-base.en")
    parser.add_argument(
        "--timeline-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "question3_stage_a_timeline",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "question3_stage_a",
    )
    args = parser.parse_args()
    aligned_dir, _ = find_attachment4(PROJECT_ROOT)
    timeline = pd.read_csv(args.timeline_dir / "samples.csv", dtype={"sample": str})
    low = timeline.loc[timeline["mean_alignment_score"] < args.threshold].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_path = args.output_dir / "low_alignment_asr_review.csv"
    if low.empty:
        pd.DataFrame().to_csv(review_path, index=False, encoding="utf-8-sig")
        return

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
        torch_dtype=dtype,
    )
    records = []
    with tempfile.TemporaryDirectory(prefix="mosei_q3_asr_") as temporary:
        for row in low.itertuples(index=False):
            name = str(row.sample).zfill(2)
            sample = load_trusted_pickle(aligned_dir / f"{name}.pkl")
            official = str(sample["raw_text"])
            wav_path = Path(temporary) / f"{name}.wav"
            extract_wav(aligned_dir / "videos" / f"{name}.mp4", wav_path, 16000)
            waveform, sample_rate = sf.read(wav_path, dtype="float32")
            result = recognizer({"array": waveform, "sampling_rate": sample_rate})
            hypothesis = str(result["text"]).strip()
            official_words = normalize_words(official)
            hypothesis_words = normalize_words(hypothesis)
            wer = edit_distance(official_words, hypothesis_words) / max(len(official_words), 1)
            if wer <= 0.4:
                conclusion = "独立ASR高度支持官方文本，可保留词级时间解释"
                usable = True
            elif wer <= 0.7:
                conclusion = "独立ASR基本支持官方文本，时间解释需标记中等可靠"
                usable = True
            else:
                conclusion = "独立ASR与官方文本差异较大，不输出逐词精确解释"
                usable = False
            records.append(
                {
                    "sample": name,
                    "mms_mean_alignment_score": float(row.mean_alignment_score),
                    "official_text": official,
                    "whisper_text": hypothesis,
                    "wer": wer,
                    "word_time_mapping_usable": usable,
                    "review_conclusion": conclusion,
                    "handling": "官方文本仍保留；仅降低证据时间定位等级，不用于调参",
                }
            )

    table = pd.DataFrame(records)
    table.to_csv(review_path, index=False, encoding="utf-8-sig")
    summary = {
        "threshold": args.threshold,
        "reviewed_count": len(records),
        "samples": [record["sample"] for record in records],
        "asr_model": args.model,
        "mean_wer": float(table["wer"].mean()),
        "word_time_mapping_usable_count": int(table["word_time_mapping_usable"].sum()),
        "unusable_samples": table.loc[
            ~table["word_time_mapping_usable"], "sample"
        ].tolist(),
        "constraint": "Whisper仅作独立质量复核，不替换官方文本、不产生情感伪标签。",
    }
    (args.output_dir / "alignment_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
