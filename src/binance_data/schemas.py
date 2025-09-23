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
