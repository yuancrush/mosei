from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mosei.data import MODALITY_DIMS, SPLITS, MoseiDataModule, find_aligned_pickle
from mosei.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit aligned CMU-MOSEI features")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "reports" / "data_audit.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data or find_aligned_pickle(PROJECT_ROOT)
    module = MoseiDataModule(data_path, batch_size=8, num_workers=0)
    report = {
        "data_path": str(data_path.resolve()),
        "split_sizes": module.split_sizes(),
        "feature_dimensions": MODALITY_DIMS,
        "splits": {},
        "train_statistics": {
            modality: stats.to_dict() for modality, stats in module.stats.items()
        },
    }

    videos = {}
    for split_name in SPLITS:
        split = module.data[split_name]
        labels = np.asarray(split["classification_labels"], dtype=int)
        regression = np.asarray(split["regression_labels"], dtype=float)
        ids = [str(value) for value in split["id"]]
        video_ids = {value.split("$_$")[0] for value in ids}
        videos[split_name] = video_ids
        dataset = module.datasets[split_name]
        report["splits"][split_name] = {
            "samples": len(ids),
            "unique_ids": len(set(ids)),
            "unique_video_ids": len(video_ids),
            "class_counts": np.bincount(labels, minlength=3).tolist(),
            "regression_min": float(regression.min()),
            "regression_max": float(regression.max()),
            "valid_step_fraction": {
                modality: float(dataset.masks[modality].mean())
                for modality in MODALITY_DIMS
            },
        }

    report["video_id_overlap"] = {
        "train_valid": len(videos["train"] & videos["valid"]),
        "train_test": len(videos["train"] & videos["test"]),
        "valid_test": len(videos["valid"] & videos["test"]),
    }
    batch = next(iter(module.loader("train", shuffle=False)))
    report["first_batch_shapes"] = {
        key: list(value.shape)
        for key, value in batch.items()
        if hasattr(value, "shape")
    }
    write_json(args.output, report)
    print(f"Audit passed: {args.output.resolve()}")
    print(f"Split sizes: {report['split_sizes']}")
    print(f"Video overlap: {report['video_id_overlap']}")
    print(f"Batch shapes: {report['first_batch_shapes']}")


if __name__ == "__main__":
    main()
