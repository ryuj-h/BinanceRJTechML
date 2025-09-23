from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SinkConfig:
    """Configuration for how collected data is persisted."""

    format: str = "parquet"
    root_dir: Path = Path("data")
    flush_interval_seconds: float = 5.0
    max_batch_size: int = 1000

    def resolve_root(self) -> Path:
        path = Path(self.root_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass(slots=True)
class CollectorConfig:
    """High-level configuration for Binance data collection."""

    symbol: str = "BTCUSDT"
    depth_limit: int = 1000
    sink: SinkConfig = field(default_factory=SinkConfig)
    http_timeout: float = 10.0
    reconnect_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 60.0
    max_update_gap: int = 100

    @property
    def normalized_symbol(self) -> str:
        return self.symbol.upper()

    @property
    def stream_symbol(self) -> str:
        return self.symbol.lower()
