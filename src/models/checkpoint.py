from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .datasets import NormalizationStats
from .tcn import TCNConfig, TCNForecaster


@dataclass(slots=True)
class TCNCheckpoint:
    model: TCNForecaster
    config: TCNConfig
    feature_stats: NormalizationStats
    target_stats: NormalizationStats
    history: Dict[str, List[float]]
    checkpoint_path: Path


def load_tcn_checkpoint(path: Path, device: Optional[str] = None) -> TCNCheckpoint:
    """Load a trained TCN checkpoint along with normalization statistics."""

    payload = torch.load(path, map_location=device or "cpu", weights_only=False)

    cfg_data = payload.get("model_config")
    if cfg_data is None:
        raise KeyError("Checkpoint missing 'model_config'.")

    config = TCNConfig(
        input_dim=cfg_data["input_dim"],
        target_dim=cfg_data["target_dim"],
        forecast_horizon=cfg_data.get("forecast_horizon", 1),
        hidden_channels=tuple(cfg_data.get("hidden_channels", (64, 64, 128))),
        kernel_size=cfg_data.get("kernel_size", 3),
        dropout=cfg_data.get("dropout", 0.1),
    )

    model = TCNForecaster(config)
    state_dict = payload.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint missing 'model_state_dict'.")
    model.load_state_dict(state_dict)
    device_to_use = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device_to_use)
    model.eval()

    feature_payload = payload.get("feature_stats")
    target_payload = payload.get("target_stats")
    if feature_payload is None or target_payload is None:
        raise KeyError("Checkpoint missing normalization statistics.")

    feature_stats = NormalizationStats(
        mean=np.array(feature_payload["mean"], dtype=np.float32),
        std=np.array(feature_payload["std"], dtype=np.float32),
    )
    target_stats = NormalizationStats(
        mean=np.array(target_payload["mean"], dtype=np.float32),
        std=np.array(target_payload["std"], dtype=np.float32),
    )

    history = payload.get("history", {})
    return TCNCheckpoint(
        model=model,
        config=config,
        feature_stats=feature_stats,
        target_stats=target_stats,
        history=history,
        checkpoint_path=Path(path),
    )