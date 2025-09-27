from __future__ import annotations

from functools import lru_cache


def _require_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
    except ImportError as exc:  # pragma: no cover - import-time guard
        raise RuntimeError(
            "pyarrow is required for Parquet operations. Install it with 'pip install pyarrow'."
        ) from exc
    return pa


@lru_cache(maxsize=1)
def orderbook_schema():
    pa = _require_pyarrow()
    level_struct = pa.struct(
        [
            ("price", pa.float64()),
            ("qty", pa.float64()),
        ]
    )
    return pa.schema(
        [
            ("event_type", pa.string()),
            ("event_time", pa.int64()),
            ("received_time", pa.timestamp("ms", tz="UTC")),
            ("symbol", pa.string()),
            ("first_update_id", pa.int64()),
            ("final_update_id", pa.int64()),
            ("prev_final_update_id", pa.int64()),
            ("is_snapshot", pa.bool_()),
            ("bids", pa.list_(level_struct)),
            ("asks", pa.list_(level_struct)),
        ]
    )


@lru_cache(maxsize=1)
def trades_schema():
    pa = _require_pyarrow()
    return pa.schema(
        [
            ("event_type", pa.string()),
            ("event_time", pa.int64()),
            ("trade_time", pa.int64()),
            ("received_time", pa.timestamp("ms", tz="UTC")),
            ("symbol", pa.string()),
            ("agg_id", pa.int64()),
            ("price", pa.float64()),
            ("qty", pa.float64()),
            ("first_trade_id", pa.int64()),
            ("last_trade_id", pa.int64()),
            ("is_buyer_maker", pa.bool_()),
        ]
    )


@lru_cache(maxsize=None)
def orderbook_feature_schema(levels_per_side: int):
    pa = _require_pyarrow()
    levels = max(1, int(levels_per_side))
    fields = [
        ("event_time", pa.int64()),
        ("received_time", pa.timestamp("ms", tz="UTC")),
        ("symbol", pa.string()),
        ("update_id", pa.int64()),
        ("mid_price", pa.float64()),
        ("spread", pa.float64()),
        ("spread_bps", pa.float64()),
        ("best_bid_price", pa.float64()),
        ("best_bid_qty", pa.float64()),
        ("best_ask_price", pa.float64()),
        ("best_ask_qty", pa.float64()),
        ("window_lower_price", pa.float64()),
        ("window_upper_price", pa.float64()),
        ("bid_window_depth", pa.float64()),
        ("ask_window_depth", pa.float64()),
        ("depth_imbalance", pa.float64()),
        ("feature_window_bps", pa.float64()),
    ]
    for idx in range(levels):
        fields.extend([
            (f"bid_px_{idx}", pa.float64()),
            (f"bid_qty_{idx}", pa.float64()),
            (f"ask_px_{idx}", pa.float64()),
            (f"ask_qty_{idx}", pa.float64()),
        ])
    return pa.schema(fields)
