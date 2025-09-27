from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        denom = np.where(self.std == 0, 1.0, self.std)
        return (values - self.mean) / denom

    def invert(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


class SlidingWindowDataset(Dataset):
    """Create (input, target) windows for sequence models."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        input_length: int,
        forecast_horizon: int,
        stride: int = 1,
    ) -> None:
        if features.shape[0] != targets.shape[0]:
            raise ValueError("Features and targets must align on the first dimension.")
        self.features = features
        self.targets = targets
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride

        max_start = features.shape[0] - input_length - forecast_horizon + 1
        if max_start <= 0:
            raise ValueError("Not enough rows to build the requested windows.")
        self.offsets = np.arange(0, max_start, stride)

    def __len__(self) -> int:  # noqa: D401
        return self.offsets.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: D401
        start = int(self.offsets[index])
        x_slice = slice(start, start + self.input_length)
        y_slice = slice(start + self.input_length, start + self.input_length + self.forecast_horizon)
        inputs = torch.from_numpy(self.features[x_slice]).float()
        targets = torch.from_numpy(self.targets[y_slice]).float()
        return inputs, targets


def compute_normalization(values: np.ndarray) -> NormalizationStats:
    return NormalizationStats(mean=values.mean(axis=0), std=values.std(axis=0))


def load_orderbook_dataset(
    parquet_path: Path,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    input_length: int,
    forecast_horizon: int,
    stride: int = 1,
    normalize_targets: bool = True,
) -> Tuple[SlidingWindowDataset, NormalizationStats, NormalizationStats]:
    """Load parquet data and return a windowed dataset plus normalization stats."""

    all_columns: List[str] = sorted({*feature_columns, *target_columns, "timestamp"})
    table = pq.read_table(parquet_path, columns=all_columns)
    df = table.to_pandas()
    df = df.sort_values("timestamp").reset_index(drop=True)

    features = df[list(feature_columns)].to_numpy(dtype=np.float32)
    targets = df[list(target_columns)].to_numpy(dtype=np.float32)

    feature_stats = compute_normalization(features)
    norm_features = feature_stats.apply(features)

    if normalize_targets:
        target_stats = compute_normalization(targets)
        norm_targets = target_stats.apply(targets)
    else:
        target_stats = NormalizationStats(mean=np.zeros(targets.shape[1], dtype=np.float32), std=np.ones(targets.shape[1], dtype=np.float32))
        norm_targets = targets

    dataset = SlidingWindowDataset(
        features=norm_features,
        targets=norm_targets,
        input_length=input_length,
        forecast_horizon=forecast_horizon,
        stride=stride,
    )
    return dataset, feature_stats, target_stats
