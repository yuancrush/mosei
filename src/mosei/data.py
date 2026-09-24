from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


MODALITY_DIMS = {"text": 768, "audio": 74, "vision": 35}
SPLITS = ("train", "valid", "test")


def find_aligned_pickle(project_root: Path) -> Path:
    candidates = list(project_root.parent.glob("**/aligned_50.pkl"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one aligned_50.pkl below {project_root.parent}, "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def load_pickle(path: Path):
    # Competition-provided pickle files are trusted inputs. Do not use this
    # helper for arbitrary pickle files from unknown sources.
    with path.open("rb") as handle:
        return pickle.load(handle)


def _text_mask(split: Mapping[str, object]) -> np.ndarray:
    text_bert = np.asarray(split["text_bert"])
    return text_bert[:, 1, :] > 0


def _modality_mask(values: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values).all(axis=-1)
    nonzero = np.any(values != 0, axis=-1)
    return text_mask & finite & nonzero


@dataclass(frozen=True)
class FeatureStats:
    mean: np.ndarray
    std: np.ndarray

    def to_dict(self) -> Dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def compute_feature_stats(
    values: np.ndarray,
    mask: np.ndarray,
    chunk_size: int = 256,
) -> FeatureStats:
    """Compute per-feature train statistics without flattening the full array."""
    total = np.zeros(values.shape[-1], dtype=np.float64)
    total_sq = np.zeros(values.shape[-1], dtype=np.float64)
    count = 0
    for start in range(0, len(values), chunk_size):
        chunk = np.asarray(values[start : start + chunk_size], dtype=np.float64)
        chunk_mask = mask[start : start + chunk_size]
        selected = chunk[chunk_mask]
        if selected.size == 0:
            continue
        total += selected.sum(axis=0)
        total_sq += np.square(selected).sum(axis=0)
        count += selected.shape[0]
    if count == 0:
        raise ValueError("No valid feature rows were found while computing statistics")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return FeatureStats(mean.astype(np.float32), std.astype(np.float32))


class AlignedMoseiDataset(Dataset):
    def __init__(
        self,
        split: Mapping[str, object],
        stats: Mapping[str, FeatureStats],
    ) -> None:
        self.split = split
        self.ids = [str(value) for value in split["id"]]
        self.text_mask = _text_mask(split)
        self.masks = {
            modality: _modality_mask(
                np.asarray(split[modality]), self.text_mask
            )
            for modality in MODALITY_DIMS
        }
        self.stats = stats

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        item: Dict[str, object] = {
            "id": self.ids[index],
            "text_input_ids": torch.as_tensor(
                np.asarray(self.split["text_bert"][index][0]), dtype=torch.long
            ),
            "text_token_type_ids": torch.as_tensor(
                np.asarray(self.split["text_bert"][index][2]), dtype=torch.long
            ),
            "classification_label": torch.tensor(
                int(self.split["classification_labels"][index]), dtype=torch.long
            ),
            "regression_label": torch.tensor(
                float(self.split["regression_labels"][index]), dtype=torch.float32
            ),
        }
        for modality in MODALITY_DIMS:
            values = torch.as_tensor(
                np.asarray(self.split[modality][index]), dtype=torch.float32
            )
            mask = torch.as_tensor(self.masks[modality][index], dtype=torch.bool)
            if modality in self.stats:
                mean = torch.from_numpy(self.stats[modality].mean)
                std = torch.from_numpy(self.stats[modality].std)
                values = (values - mean) / std
                values = values.masked_fill(~mask.unsqueeze(-1), 0.0)
            item[modality] = values
            item[f"{modality}_mask"] = mask
        return item


class MoseiDataModule:
    def __init__(
        self,
        pickle_path: Path,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        self.pickle_path = Path(pickle_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data = load_pickle(self.pickle_path)
        self._validate_structure()

        train_mask = _text_mask(self.data["train"])
        self.stats = {
            modality: compute_feature_stats(
                np.asarray(self.data["train"][modality]),
                _modality_mask(
                    np.asarray(self.data["train"][modality]), train_mask
                ),
            )
            for modality in ("audio", "vision")
        }
        self.datasets = {
            split: AlignedMoseiDataset(self.data[split], self.stats)
            for split in SPLITS
        }

    def _validate_structure(self) -> None:
        if tuple(self.data.keys()) != SPLITS:
            if set(self.data.keys()) != set(SPLITS):
                raise ValueError(f"Unexpected split keys: {self.data.keys()}")
        for split_name in SPLITS:
            split = self.data[split_name]
            n = len(split["id"])
            for modality, dim in MODALITY_DIMS.items():
                shape = np.asarray(split[modality]).shape
                if shape != (n, 50, dim):
                    raise ValueError(
                        f"{split_name}.{modality} has shape {shape}, "
                        f"expected {(n, 50, dim)}"
                    )
            if np.asarray(split["text_bert"]).shape != (n, 3, 50):
                raise ValueError(
                    f"{split_name}.text_bert has unexpected shape "
                    f"{np.asarray(split['text_bert']).shape}"
                )
            labels = np.asarray(split["classification_labels"])
            if not set(np.unique(labels)).issubset({0.0, 1.0, 2.0}):
                raise ValueError(f"Unexpected labels in {split_name}: {np.unique(labels)}")
            regression = np.asarray(split["regression_labels"])
            if not np.isfinite(regression).all():
                raise ValueError(f"Non-finite regression labels in {split_name}")

    def loader(self, split: str, shuffle: bool | None = None) -> DataLoader:
        if shuffle is None:
            shuffle = split == "train"
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def class_weights(self, power: float = 1.0) -> torch.Tensor:
        labels = np.asarray(self.data["train"]["classification_labels"], dtype=int)
        counts = np.bincount(labels, minlength=3).astype(np.float64)
        inverse = counts.sum() / (len(counts) * counts)
        weights = np.power(inverse, power)
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32)

    def split_sizes(self) -> Dict[str, int]:
        return {split: len(self.datasets[split]) for split in SPLITS}

    def iter_ids(self, split: str) -> Iterable[str]:
        return iter(self.datasets[split].ids)
