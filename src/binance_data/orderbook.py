from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from .config import CollectorConfig
from .storage import AsyncDataSink

LOGGER = logging.getLogger(__name__)

DEPTH_SNAPSHOT_URL = "https://fapi.binance.com/fapi/v1/depth"
WEBSOCKET_BASE_URL = "wss://fstream.binance.com/ws"


class ResyncRequired(Exception):
    """Raised when the local order book falls out of sync."""


@dataclass(slots=True)
class OrderBookState:
    """Mutable in-memory order book used to derive training features."""

    last_update_id: int
    bids: Dict[float, float] = field(default_factory=dict)
    asks: Dict[float, float] = field(default_factory=dict)

    @classmethod
    def from_snapshot(cls, snapshot: Dict[str, Any]) -> "OrderBookState":
        return cls(
            last_update_id=snapshot["lastUpdateId"],
            bids=_build_level_map(snapshot.get("bids", [])),
            asks=_build_level_map(snapshot.get("asks", [])),
        )

    def apply_event(self, event: Dict[str, Any]) -> None:
        _apply_level_updates(self.bids, event.get("b", []))
        _apply_level_updates(self.asks, event.get("a", []))
        self.last_update_id = event["u"]


def _build_level_map(levels: List[List[str]]) -> Dict[float, float]:
    book: Dict[float, float] = {}
    for price_str, qty_str in levels:
        price = float(price_str)
        qty = float(qty_str)
        if qty > 0.0:
            book[price] = qty
    return book


def _apply_level_updates(level_map: Dict[float, float], updates: Optional[List[List[str]]]) -> None:
    if not updates:
        return
    for price_str, qty_str in updates:
        price = float(price_str)
        qty = float(qty_str)
        if qty <= 0.0:
            level_map.pop(price, None)
        else:
            level_map[price] = qty


class OrderBookFeatureBuilder:
    """Builds compact feature rows centred around the current mid price."""

    def __init__(self, levels_per_side: int, window_bps: float) -> None:
        self._levels_per_side = max(1, int(levels_per_side))
        self._window_bps = max(0.0, float(window_bps))

    def build(
        self,
        *,
        state: OrderBookState,
        symbol: str,
        event_time: int,
        received_time: datetime,
    ) -> Dict[str, Any]:
        bids_sorted = sorted(state.bids.items(), key=lambda item: item[0], reverse=True)
        asks_sorted = sorted(state.asks.items(), key=lambda item: item[0])

        best_bid_price, best_bid_qty = self._top_level(bids_sorted)
        best_ask_price, best_ask_qty = self._top_level(asks_sorted)

        mid_price = self._compute_mid(best_bid_price, best_ask_price)
        spread = self._compute_spread(best_bid_price, best_ask_price)
        spread_bps = self._compute_spread_bps(spread, mid_price)
        lower_bound, upper_bound = self._compute_window_bounds(mid_price)

        bid_window_depth = self._accumulate_window_depth(bids_sorted, lower_bound, True)
        ask_window_depth = self._accumulate_window_depth(asks_sorted, upper_bound, False)
        depth_total = bid_window_depth + ask_window_depth
        depth_imbalance = (
            (bid_window_depth - ask_window_depth) / depth_total
            if depth_total > 0.0
            else None
        )

        record: Dict[str, Any] = {
            "event_time": event_time,
            "received_time": received_time,
            "symbol": symbol,
            "update_id": state.last_update_id,
            "mid_price": mid_price,
            "spread": spread,
            "spread_bps": spread_bps,
            "best_bid_price": best_bid_price,
            "best_bid_qty": best_bid_qty,
            "best_ask_price": best_ask_price,
            "best_ask_qty": best_ask_qty,
            "window_lower_price": lower_bound,
            "window_upper_price": upper_bound,
            "bid_window_depth": bid_window_depth,
            "ask_window_depth": ask_window_depth,
            "depth_imbalance": depth_imbalance,
            "feature_window_bps": self._window_bps,
        }

        for idx in range(self._levels_per_side):
            bid_price, bid_qty = self._level_at(bids_sorted, idx)
            ask_price, ask_qty = self._level_at(asks_sorted, idx)
            record[f"bid_px_{idx}"] = bid_price
            record[f"bid_qty_{idx}"] = bid_qty
            record[f"ask_px_{idx}"] = ask_price
            record[f"ask_qty_{idx}"] = ask_qty

        return record

    @staticmethod
    def _top_level(levels: List[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float]]:
        if not levels:
            return None, None
        price, qty = levels[0]
        return price, qty

    @staticmethod
    def _level_at(levels: List[Tuple[float, float]], index: int) -> Tuple[Optional[float], Optional[float]]:
        if index < len(levels):
            price, qty = levels[index]
            return price, qty
        return None, None

    @staticmethod
    def _compute_mid(
        bid_price: Optional[float],
        ask_price: Optional[float],
    ) -> Optional[float]:
        if bid_price is None and ask_price is None:
            return None
        if bid_price is None:
            return ask_price
        if ask_price is None:
            return bid_price
        return (bid_price + ask_price) / 2.0

    @staticmethod
    def _compute_spread(
        bid_price: Optional[float],
        ask_price: Optional[float],
    ) -> Optional[float]:
        if bid_price is None or ask_price is None:
            return None
        return ask_price - bid_price

    @staticmethod
    def _compute_spread_bps(
        spread: Optional[float],
        mid_price: Optional[float],
    ) -> Optional[float]:
        if spread is None or mid_price is None or mid_price == 0.0:
            return None
        return (spread / mid_price) * 10_000.0

    def _compute_window_bounds(
        self, mid_price: Optional[float]
    ) -> Tuple[Optional[float], Optional[float]]:
        if mid_price is None or mid_price <= 0.0:
            return None, None
        window_ratio = self._window_bps / 10_000.0
        lower = mid_price * (1.0 - window_ratio)
        upper = mid_price * (1.0 + window_ratio)
        return lower, upper

    @staticmethod
    def _accumulate_window_depth(
        levels: List[Tuple[float, float]],
        bound: Optional[float],
        is_bid: bool,
    ) -> float:
        if bound is None:
            return sum(qty for _, qty in levels if qty > 0.0)
        depth = 0.0
        for price, qty in levels:
            if qty <= 0.0:
                continue
            if is_bid:
                if price < bound:
                    break
            else:
                if price > bound:
                    break
            depth += qty
        return depth


class OrderBookCollector:
    """Collects Binance futures order book data and emits ML-ready features."""

    def __init__(self, config: CollectorConfig, sink: AsyncDataSink) -> None:
        self._config = config
        self._sink = sink
        self._timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=config.http_timeout,
            sock_read=45,
        )
        self._feature_builder = OrderBookFeatureBuilder(
            levels_per_side=config.feature_levels_per_side,
            window_bps=config.feature_window_bps,
        )

    async def run(self) -> None:
        backoff = self._config.reconnect_backoff_seconds
        while True:
            try:
                await self._run_once()
                backoff = self._config.reconnect_backoff_seconds
            except asyncio.CancelledError:
                raise
            except ResyncRequired as exc:
                LOGGER.warning("Order book resync required: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._config.max_backoff_seconds)
            except Exception as exc:  # pragma: no cover - network resilience
                LOGGER.exception("Order book collector error: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._config.max_backoff_seconds)

    async def _run_once(self) -> None:
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            stream_url = f"{WEBSOCKET_BASE_URL}/{self._config.stream_symbol}@depth@100ms"
            queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
            LOGGER.info("Connecting order book stream %s", stream_url)
            async with session.ws_connect(stream_url, heartbeat=15) as ws:
                reader_task = asyncio.create_task(self._stream_reader(ws, queue))
                try:
                    snapshot = await self._fetch_snapshot(session)
                    state = OrderBookState.from_snapshot(snapshot)
                    await self._emit_state_features(state, origin="snapshot")
                    await self._drain_initial_events(queue, state)
                    await self._consume_updates(queue, state)
                finally:
                    reader_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await reader_task

    async def _stream_reader(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        queue: asyncio.Queue[Optional[Dict[str, Any]]],
    ) -> None:
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await queue.put(data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    LOGGER.error("Websocket error: %s", ws.exception())
                    break
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                    break
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Order book websocket reader failed: %s", exc)
        finally:
            await queue.put(None)

    async def _fetch_snapshot(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        params = {"symbol": self._config.normalized_symbol, "limit": self._config.depth_limit}
        LOGGER.info("Fetching depth snapshot for %s", self._config.normalized_symbol)
        async with session.get(DEPTH_SNAPSHOT_URL, params=params) as resp:
            resp.raise_for_status()
            payload = await resp.json()
            payload["lastUpdateId"] = int(payload["lastUpdateId"])
            return payload

    async def _emit_state_features(
        self,
        state: OrderBookState,
        *,
        origin: str,
        event_time: Optional[int] = None,
        received_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        received_ts = received_time or now
        event_ts = event_time if event_time is not None else int(received_ts.timestamp() * 1000)
        record = self._feature_builder.build(
            state=state,
            symbol=symbol or self._config.normalized_symbol,
            event_time=event_ts,
            received_time=received_ts,
        )
        await self._sink.write(record)
        LOGGER.debug(
            "Stored order book features (%s) for %s update_id=%s",
            origin,
            record["symbol"],
            state.last_update_id,
        )

    async def _drain_initial_events(
        self,
        queue: asyncio.Queue[Optional[Dict[str, Any]]],
        state: OrderBookState,
    ) -> None:
        while True:
            event = await queue.get()
            if event is None:
                raise ConnectionError("Websocket closed before synchronization completed")
            if "u" not in event or "U" not in event:
                continue
            event["u"] = int(event["u"])
            event["U"] = int(event["U"])
            if event.get("pu") is not None:
                event["pu"] = int(event["pu"])

            if event["u"] <= state.last_update_id:
                continue

            if event["U"] <= state.last_update_id + 1 <= event["u"]:
                prev_id = event.get("pu")
                if prev_id is not None and prev_id > state.last_update_id + 1:
                    LOGGER.warning(
                        "Depth init prev id ahead of local state (local=%s, prev=%s)",
                        state.last_update_id,
                        prev_id,
                    )
                await self._apply_event_and_emit(state, event)
                break

            init_gap = event["U"] - (state.last_update_id + 1)
            if init_gap > self._config.max_update_gap:
                LOGGER.warning(
                    "Large depth init gap detected (local=%s, first=%s, final=%s, prev=%s, gap=%s)",
                    state.last_update_id,
                    event["U"],
                    event["u"],
                    event.get("pu"),
                    init_gap,
                )
                raise ResyncRequired("Large gap in depth events during initialization")
            LOGGER.info(
                "Small depth init gap tolerated, accepting event (local=%s, first=%s, gap=%s)",
                state.last_update_id,
                event["U"],
                init_gap,
            )
            await self._apply_event_and_emit(state, event)
            break

    async def _consume_updates(
        self,
        queue: asyncio.Queue[Optional[Dict[str, Any]]],
        state: OrderBookState,
    ) -> None:
        while True:
            event = await queue.get()
            if event is None:
                raise ConnectionError("Order book stream closed")
            if "u" not in event or "U" not in event:
                continue
            event["u"] = int(event["u"])
            event["U"] = int(event["U"])
            if event.get("pu") is not None:
                event["pu"] = int(event["pu"])

            if event["u"] <= state.last_update_id:
                continue

            next_expected = state.last_update_id + 1
            prev_id = event.get("pu")

            if prev_id is not None:
                if prev_id == state.last_update_id and event["U"] > next_expected:
                    LOGGER.debug(
                        "Depth update bridged by prev id (local=%s, first=%s, final=%s, prev=%s)",
                        state.last_update_id,
                        event["U"],
                        event["u"],
                        prev_id,
                    )
                    await self._apply_event_and_emit(state, event)
                    continue
                if prev_id > next_expected:
                    prev_gap = prev_id - state.last_update_id
                    if prev_gap > self._config.max_update_gap:
                        LOGGER.warning(
                            "Large prev id gap detected (local=%s, prev=%s, gap=%s)",
                            state.last_update_id,
                            prev_id,
                            prev_gap,
                        )
                        raise ResyncRequired(
                            f"Large prev id gap {prev_gap} exceeds tolerance {self._config.max_update_gap}"
                        )
                    LOGGER.info(
                        "Small prev id gap tolerated (local=%s, prev=%s, gap=%s)",
                        state.last_update_id,
                        prev_id,
                        prev_gap,
                    )

            if event["U"] > next_expected:
                gap_size = event["U"] - next_expected
                if gap_size > self._config.max_update_gap:
                    LOGGER.warning(
                        "Large depth update gap detected (local=%s, first=%s, final=%s, prev=%s, gap=%s)",
                        state.last_update_id,
                        event["U"],
                        event["u"],
                        prev_id,
                        gap_size,
                    )
                    raise ResyncRequired("Large gap in depth updates; resync required")
                LOGGER.info(
                    "Small depth update gap tolerated (local=%s, first=%s, gap=%s)",
                    state.last_update_id,
                    event["U"],
                    gap_size,
                )

            if event["U"] <= next_expected <= event["u"]:
                await self._apply_event_and_emit(state, event)
                continue

            if event["U"] <= state.last_update_id:
                continue

            final_gap = event["U"] - state.last_update_id
            if final_gap > self._config.max_update_gap:
                LOGGER.warning(
                    "Large final depth update gap (local=%s, first=%s, final=%s, prev=%s, gap=%s)",
                    state.last_update_id,
                    event["U"],
                    event["u"],
                    prev_id,
                    final_gap,
                )
                raise ResyncRequired("Large final gap in depth events; resync required")
            LOGGER.info(
                "Accepting out-of-order update with small gap (local=%s, first=%s, gap=%s)",
                state.last_update_id,
                event["U"],
                final_gap,
            )
            await self._apply_event_and_emit(state, event)

    async def _apply_event_and_emit(self, state: OrderBookState, event: Dict[str, Any]) -> None:
        state.apply_event(event)
        received_time = datetime.now(timezone.utc)
        event_time = int(event.get("E", 0))
        if event_time <= 0:
            event_time = int(received_time.timestamp() * 1000)
        symbol = event.get("s", self._config.normalized_symbol)
        await self._emit_state_features(
            state,
            origin="depthUpdate",
            event_time=event_time,
            received_time=received_time,
            symbol=symbol,
        )
