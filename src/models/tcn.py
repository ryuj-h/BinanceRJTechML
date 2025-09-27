from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """Remove padding added to keep temporal dimension constant."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    """Dilated causal convolutional block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.final_act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.act2(out)
        out = self.drop2(out)

        res = self.downsample(x)
        return self.final_act(out + res)


class TemporalConvNet(nn.Module):
    """Stacked dilated causal convolutions for sequence modeling."""

    def __init__(
        self,
        num_inputs: int,
        channel_sizes: Sequence[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_channels = num_inputs
        for i, num_channels in enumerate(channel_sizes):
            dilation = 2**i
            layers.append(
                TemporalBlock(
                    prev_channels,
                    num_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            prev_channels = num_channels
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.network(x)


@dataclass(slots=True)
class TCNConfig:
    input_dim: int
    target_dim: int
    forecast_horizon: int = 10
    hidden_channels: Sequence[int] = (64, 64, 128)
    kernel_size: int = 3
    dropout: float = 0.1


class TCNForecaster(nn.Module):
    """Multi-step forecaster using a TCN backbone."""

    def __init__(self, config: TCNConfig) -> None:
        super().__init__()
        self.config = config
        self.tcn = TemporalConvNet(
            num_inputs=config.input_dim,
            channel_sizes=config.hidden_channels,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )
        last_channels = config.hidden_channels[-1]
        self.head = nn.Linear(last_channels, config.target_dim * config.forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # x: [batch, seq_len, input_dim]
        x = x.transpose(1, 2)
        features = self.tcn(x)
        last_state = features[:, :, -1]
        out = self.head(last_state)
        return out.view(-1, self.config.forecast_horizon, self.config.target_dim)


def default_tcn_config(input_dim: int, target_dim: int) -> TCNConfig:
    return TCNConfig(input_dim=input_dim, target_dim=target_dim)
