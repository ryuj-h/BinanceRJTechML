from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from binance_data.config import CollectorConfig, SinkConfig
from binance_data.orderbook import OrderBookCollector
from binance_data.schemas import orderbook_feature_schema, trades_schema
from binance_data.storage import JsonlSink, ParquetSink
from binance_data.trades import TradeCollector

LOGGER = logging.getLogger("collect_binance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Binance futures market data.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol (default: BTCUSDT)")
    parser.add_argument("--data-dir", default="data", help="Root directory for persisted data")
    parser.add_argument(
        "--sink-format",
        choices=["parquet", "jsonl"],
        default="parquet",
        help="File format used to store records",
    )
    parser.add_argument("--depth-limit", type=int, default=1000, help="REST snapshot depth limit")
    parser.add_argument("--flush-interval", type=float, default=5.0, help="Flush interval in seconds")
    parser.add_argument("--max-batch-size", type=int, default=1000, help="Records to buffer before flushing")
    parser.add_argument("--http-timeout", type=float, default=10.0, help="HTTP connect timeout in seconds")
    parser.add_argument("--backoff", type=float, default=2.0, help="Initial reconnect backoff")
    parser.add_argument("--max-backoff", type=float, default=60.0, help="Maximum reconnect backoff")
    parser.add_argument("--max-update-gap", type=int, default=100, help="Maximum tolerable gap in update IDs")
    parser.add_argument("--feature-levels-per-side", type=int, default=10, help="Depth levels per side to persist around the mid price")
    parser.add_argument("--feature-window-bps", type=float, default=50.0, help="Half-window around the mid price (in basis points) for depth aggregation")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def create_sink(sink_cfg: SinkConfig, dataset: str, schema: Any | None) -> ParquetSink | JsonlSink:
    root = sink_cfg.resolve_root()
    if sink_cfg.format == "parquet":
        if schema is None:
            raise ValueError("Parquet sink requires a schema")
        return ParquetSink(
            root_dir=root,
            dataset_name=dataset,
            schema=schema,
            flush_interval_seconds=sink_cfg.flush_interval_seconds,
            max_batch_size=sink_cfg.max_batch_size,
        )
    if sink_cfg.format == "jsonl":
        return JsonlSink(
            root_dir=root,
            dataset_name=dataset,
            flush_interval_seconds=sink_cfg.flush_interval_seconds,
            max_batch_size=sink_cfg.max_batch_size,
        )
    raise ValueError(f"Unsupported sink format: {sink_cfg.format}")


async def run_collectors(cfg: CollectorConfig) -> None:
    if cfg.sink.format == "parquet":
        orderbook_schema_obj = orderbook_feature_schema(cfg.feature_levels_per_side)
        trades_schema_obj = trades_schema()
    else:
        orderbook_schema_obj = None
        trades_schema_obj = None

    orderbook_sink = create_sink(cfg.sink, "orderbook_features", orderbook_schema_obj)
    trades_sink = create_sink(cfg.sink, "trades", trades_schema_obj)
    orderbook = OrderBookCollector(cfg, orderbook_sink)
    trades = TradeCollector(cfg, trades_sink)

    tasks = [
        asyncio.create_task(orderbook.run(), name="orderbook"),
        asyncio.create_task(trades.run(), name="trades"),
    ]

    stop_event = asyncio.Event()

    def _task_done(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            LOGGER.error("Collector %s stopped with error: %s", task.get_name(), exc)
        stop_event.set()

    for task in tasks:
        task.add_done_callback(_task_done)

    def _signal_handler(*_: object) -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            signal.signal(sig, lambda *_: _signal_handler())

    await stop_event.wait()
    for task in tasks:
        task.cancel()

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
            LOGGER.error("Collector finished with error: %s", result)

    await orderbook_sink.close()
    await trades_sink.close()


def main() -> None:
    args = parse_args()
    if sys.platform.startswith("win"):
        with suppress(Exception):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    configure_logging(args.log_level)

    sink_cfg = SinkConfig(
        format=args.sink_format,
        root_dir=Path(args.data_dir),
        flush_interval_seconds=args.flush_interval,
        max_batch_size=args.max_batch_size,
    )
    collector_cfg = CollectorConfig(
        symbol=args.symbol,
        depth_limit=args.depth_limit,
        sink=sink_cfg,
        http_timeout=args.http_timeout,
        reconnect_backoff_seconds=args.backoff,
        max_backoff_seconds=args.max_backoff,
        max_update_gap=args.max_update_gap,
        feature_levels_per_side=args.feature_levels_per_side,
        feature_window_bps=args.feature_window_bps,
    )

    try:
        asyncio.run(run_collectors(collector_cfg))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
