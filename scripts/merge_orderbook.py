#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from heapq import nlargest, nsmallest
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)

ORDERBOOK_COLUMNS = [
    "event_time",
    "symbol",
    "is_snapshot",
    "bids",
    "asks",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a regular time series from incremental Binance order book data."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/orderbook"),
        help="Root directory with raw order book parquet files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/orderbook_top20_250ms.parquet"),
        help="Destination parquet file for the merged series.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Optional symbol filter (e.g. BTCUSDT).",
    )
    parser.add_argument(
        "--top-levels",
        type=int,
        default=20,
        help="Number of price levels per side to keep in the output.",
    )
    parser.add_argument(
        "--frequency-ms",
        type=int,
        default=250,
        help="Sampling frequency in milliseconds.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Number of rows to buffer before flushing to disk.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (e.g. INFO, DEBUG).",
    )
    return parser.parse_args()


def iter_orderbook_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {root}")
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root}")
    return files


def iter_orderbook_events(files: Iterable[Path], symbol_filter: Optional[str]) -> Iterable[Dict[str, object]]:
    for path in files:
        pf = pq.ParquetFile(path)
        schema_names = pf.schema_arrow.names
        columns = [name for name in ORDERBOOK_COLUMNS if name in schema_names]
        required = {"event_time", "symbol", "is_snapshot"}
        missing = required.difference(columns)
        if missing:
            raise ValueError(f"File {path} missing required columns: {sorted(missing)}")
        for batch in pf.iter_batches(columns=columns):
            for row in batch.to_pylist():
                symbol = row.get("symbol")
                if symbol_filter and symbol != symbol_filter:
                    continue
                event_time = row.get("event_time")
                if event_time is None:
                    continue
                row["event_time"] = int(event_time)
                yield row


class OrderBookAggregator:
    def __init__(self, symbol: str, top_levels: int, frequency_ms: int) -> None:
        self.symbol = symbol
        self.top_levels = top_levels
        self.frequency_ms = frequency_ms
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.has_snapshot = False
        self.next_emit_time: Optional[int] = None
        self.last_event_time: Optional[int] = None

    def process(self, record: Dict[str, object]) -> List[Dict[str, object]]:
        event_time = int(record["event_time"])
        if record.get("is_snapshot"):
            self._load_snapshot(record)
        else:
            if not self.has_snapshot:
                LOGGER.debug("Ignoring update before snapshot for %s", self.symbol)
                return []
            self._apply_update(record)
        self.last_event_time = event_time
        if self.next_emit_time is None:
            self.next_emit_time = event_time
        emitted: List[Dict[str, object]] = []
        while self.next_emit_time is not None and event_time >= self.next_emit_time:
            emitted.append(self._serialize_state(self.next_emit_time))
            self.next_emit_time += self.frequency_ms
        return emitted

    def _load_snapshot(self, record: Dict[str, object]) -> None:
        self.bids = self._levels_to_map(record.get("bids"))
        self.asks = self._levels_to_map(record.get("asks"))
        self.has_snapshot = True
        self.next_emit_time = None

    def _apply_update(self, record: Dict[str, object]) -> None:
        self._apply_side(record.get("bids"), self.bids)
        self._apply_side(record.get("asks"), self.asks)

    def _levels_to_map(self, levels: Optional[Iterable[Dict[str, object]]]) -> Dict[float, float]:
        book: Dict[float, float] = {}
        if not levels:
            return book
        for level in levels:
            price = float(level["price"])
            qty = float(level["qty"])
            if qty > 0.0:
                book[price] = qty
        return book

    def _apply_side(self, levels: Optional[Iterable[Dict[str, object]]], book: Dict[float, float]) -> None:
        if not levels:
            return
        for level in levels:
            price = float(level["price"])
            qty = float(level["qty"])
            if qty <= 0.0:
                book.pop(price, None)
            else:
                book[price] = qty

    def _serialize_state(self, timestamp: int) -> Dict[str, object]:
        bids = self._top_levels(self.bids, is_bid_side=True)
        asks = self._top_levels(self.asks, is_bid_side=False)
        best_bid_price = bids[0][0] if bids else None
        best_bid_qty = bids[0][1] if bids else 0.0
        best_ask_price = asks[0][0] if asks else None
        best_ask_qty = asks[0][1] if asks else 0.0
        spread = None
        mid_price = None
        if best_bid_price is not None and best_ask_price is not None:
            spread = best_ask_price - best_bid_price
            mid_price = (best_bid_price + best_ask_price) / 2.0
        bid_vol = sum(qty for _, qty in bids)
        ask_vol = sum(qty for _, qty in asks)
        imbalance = 0.0
        denom = bid_vol + ask_vol
        if denom > 0.0:
            imbalance = (bid_vol - ask_vol) / denom
        row: Dict[str, object] = {
            "timestamp": timestamp,
            "symbol": self.symbol,
            "best_bid_price": best_bid_price,
            "best_bid_qty": best_bid_qty,
            "best_ask_price": best_ask_price,
            "best_ask_qty": best_ask_qty,
            "spread": spread,
            "mid_price": mid_price,
            "bid_volume_top": bid_vol,
            "ask_volume_top": ask_vol,
            "volume_imbalance": imbalance,
        }
        for idx in range(self.top_levels):
            bid_price, bid_qty = self._level_at(bids, idx)
            ask_price, ask_qty = self._level_at(asks, idx)
            row[f"bid_px_{idx + 1}"] = bid_price
            row[f"bid_qty_{idx + 1}"] = bid_qty
            row[f"ask_px_{idx + 1}"] = ask_price
            row[f"ask_qty_{idx + 1}"] = ask_qty
        return row

    def _top_levels(self, book: Dict[float, float], *, is_bid_side: bool) -> List[Tuple[float, float]]:
        if not book:
            return []
        items = ((price, qty) for price, qty in book.items() if qty > 0.0)
        if is_bid_side:
            selected = nlargest(self.top_levels, items, key=lambda item: item[0])
        else:
            selected = nsmallest(self.top_levels, items, key=lambda item: item[0])
        return [(float(price), float(qty)) for price, qty in selected]

    def _level_at(self, levels: List[Tuple[float, float]], index: int) -> Tuple[Optional[float], float]:
        if index < len(levels):
            price, qty = levels[index]
            return price, qty
        return None, 0.0


class ParquetSeriesWriter:
    def __init__(self, path: Path, schema: pa.Schema, batch_size: int) -> None:
        self.path = path
        self.schema = schema
        self.batch_size = batch_size
        self._buffer: List[Dict[str, object]] = []
        self._writer: Optional[pq.ParquetWriter] = None
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: Dict[str, object]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, self.schema)
        self._writer.write_table(table)
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def build_output_schema(top_levels: int) -> pa.Schema:
    fields = [
        pa.field("timestamp", pa.int64()),
        pa.field("symbol", pa.string()),
        pa.field("best_bid_price", pa.float64()),
        pa.field("best_bid_qty", pa.float64()),
        pa.field("best_ask_price", pa.float64()),
        pa.field("best_ask_qty", pa.float64()),
        pa.field("spread", pa.float64()),
        pa.field("mid_price", pa.float64()),
        pa.field("bid_volume_top", pa.float64()),
        pa.field("ask_volume_top", pa.float64()),
        pa.field("volume_imbalance", pa.float64()),
    ]
    for idx in range(1, top_levels + 1):
        fields.append(pa.field(f"bid_px_{idx}", pa.float64()))
        fields.append(pa.field(f"bid_qty_{idx}", pa.float64()))
        fields.append(pa.field(f"ask_px_{idx}", pa.float64()))
        fields.append(pa.field(f"ask_qty_{idx}", pa.float64()))
    return pa.schema(fields)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    files = iter_orderbook_files(args.source)
    schema = build_output_schema(args.top_levels)
    writer = ParquetSeriesWriter(args.output, schema, args.batch_size)
    aggregators: Dict[str, OrderBookAggregator] = {}
    total_events = 0
    total_rows = 0
    try:
        for record in iter_orderbook_events(files, args.symbol):
            symbol = str(record["symbol"])
            aggregator = aggregators.get(symbol)
            if aggregator is None:
                aggregator = OrderBookAggregator(symbol, args.top_levels, args.frequency_ms)
                aggregators[symbol] = aggregator
            rows = aggregator.process(record)
            total_events += 1
            for row in rows:
                writer.write(row)
                total_rows += 1
            if total_events and total_events % 50000 == 0:
                LOGGER.info("Processed %s events -> %s rows", total_events, total_rows)
    finally:
        writer.close()
    LOGGER.info(
        "Finished merging order book: %s events processed, %s rows written to %s",
        total_events,
        total_rows,
        args.output,
    )


if __name__ == "__main__":
    main()




