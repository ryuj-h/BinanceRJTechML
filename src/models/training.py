from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from .datasets import NormalizationStats, SlidingWindowDataset, load_orderbook_dataset
from .tcn import TCNConfig, TCNForecaster, default_tcn_config


@dataclass(slots=True)
class TrainingConfig:
    parquet_path: Path
    feature_columns: Sequence[str]
    target_columns: Sequence[str]
    input_length: int = 100
    forecast_horizon: int = 10
    stride: int = 1
    batch_size: int = 256
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_split: float = 0.1
    normalize_targets: bool = True
    model_dir: Path = Path("models")
    checkpoint_name: str = "tcn_forecaster.pt"
    num_workers: int = 0
    seed: int = 2024
    device: Optional[str] = None


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_dataloaders(
    dataset: SlidingWindowDataset,
    batch_size: int,
    validation_split: float,
    num_workers: int,
    seed: int,
) -> Dict[str, DataLoader]:
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
    }
    if val_size > 0:
        loaders["val"] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loaders


def _stats_to_dict(stats: NormalizationStats) -> Dict[str, List[float]]:
    return {"mean": stats.mean.astype(float).tolist(), "std": stats.std.astype(float).tolist()}


def train_tcn_model(
    config: TrainingConfig,
    model_config: Optional[TCNConfig] = None,
) -> Dict[str, float]:
    _set_seeds(config.seed)

    dataset, feature_stats, target_stats = load_orderbook_dataset(
        parquet_path=config.parquet_path,
        feature_columns=config.feature_columns,
        target_columns=config.target_columns,
        input_length=config.input_length,
        forecast_horizon=config.forecast_horizon,
        stride=config.stride,
        normalize_targets=config.normalize_targets,
    )

    loaders = _make_dataloaders(
        dataset=dataset,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if model_config is None:
        model_config = default_tcn_config(
            input_dim=len(config.feature_columns),
            target_dim=len(config.target_columns),
        )
        model_config.forecast_horizon = config.forecast_horizon

    model = TCNForecaster(model_config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    model_dir = config.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / config.checkpoint_name

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        train_loader = loaders["train"]
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs} [train]", leave=False)
        for inputs, targets in train_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            running_loss += batch_loss * inputs.size(0)
            train_bar.set_postfix(loss=batch_loss)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)

        epoch_val_loss = float("nan")
        if "val" in loaders:
            model.eval()
            val_loader = loaders["val"]
            val_loss = 0.0
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{config.epochs} [val]", leave=False)
            with torch.no_grad():
                for inputs, targets in val_bar:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    batch_loss = loss.item()
                    val_loss += batch_loss * inputs.size(0)
                    val_bar.set_postfix(loss=batch_loss)
            epoch_val_loss = val_loss / len(val_loader.dataset)
            history["val_loss"].append(epoch_val_loss)
            improved = epoch_val_loss < best_val_loss
            if improved:
                best_val_loss = epoch_val_loss
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    model_config,
                    feature_stats,
                    target_stats,
                    history,
                )
        else:
            _save_checkpoint(
                checkpoint_path,
                model,
                model_config,
                feature_stats,
                target_stats,
                history,
            )

        if epoch_val_loss == epoch_val_loss:
            print(f"Epoch {epoch}/{config.epochs} - train_loss: {epoch_train_loss:.6f} val_loss: {epoch_val_loss:.6f}")
        else:
            print(f"Epoch {epoch}/{config.epochs} - train_loss: {epoch_train_loss:.6f} (no validation)")

    metrics = {
        "train_loss": history["train_loss"][-1],
        "val_loss": history["val_loss"][-1] if history["val_loss"] else float("nan"),
        "checkpoint_path": str(checkpoint_path),
    }
    return metrics


def _save_checkpoint(
    path: Path,
    model: TCNForecaster,
    model_config: TCNConfig,
    feature_stats: NormalizationStats,
    target_stats: NormalizationStats,
    history: Dict[str, List[float]],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": model_config.input_dim,
            "target_dim": model_config.target_dim,
            "forecast_horizon": model_config.forecast_horizon,
            "hidden_channels": list(model_config.hidden_channels),
            "kernel_size": model_config.kernel_size,
            "dropout": model_config.dropout,
        },
        "feature_stats": _stats_to_dict(feature_stats),
        "target_stats": _stats_to_dict(target_stats),
        "history": history,
    }
    torch.save(payload, path)
