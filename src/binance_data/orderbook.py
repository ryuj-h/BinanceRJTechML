from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from .config import CollectorConfig
from .storage import AsyncDataSink

LOGGER = logging.getLogger(__name__)

DEPTH_SNAPSHOT_URL = "https://fapi.binance.com/fapi/v1/depth"
WEBSOCKET_BASE_URL = "wss://fstream.binance.com/ws"


class ResyncRequired(Exception):
    """Raised when the local order book falls out of sync."""


def _convert_price_levels(levels: List[List[str]]) -> List[Dict[str, float]]:
    return [{"price": float(price), "qty": float(qty)} for price, qty in levels]


@dataclass(slots=True)
class OrderBookState:
    last_update_id: int


class OrderBookCollector:
    """Collects Binance futures order book snapshots and diff updates."""

    def __init__(self, config: CollectorConfig, sink: AsyncDataSink) -> None:
        self._config = config
        self._sink = sink
        self._timeout = aiohttp.ClientTimeout(total=None, sock_connect=config.http_timeout, sock_read=45)

    async def run(self) -> None:
        backoff = self._config.reconnect_backoff_seconds
        while True:
            try:
                await self._run_once()
                backoff = self._config.reconnect_backoff_seconds
            except asyncio.CancelledError:
                raise
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
                    state = OrderBookState(last_update_id=snapshot["lastUpdateId"])
                    await self._emit_snapshot(snapshot)
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

    async def _emit_snapshot(self, snapshot: Dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        record = {
            "event_type": "snapshot",
            "event_time": int(now.timestamp() * 1000),
            "received_time": now,
            "symbol": self._config.normalized_symbol,
            "first_update_id": snapshot["lastUpdateId"],
            "final_update_id": snapshot["lastUpdateId"],
            "prev_final_update_id": snapshot["lastUpdateId"] - 1,
            "is_snapshot": True,
            "bids": _convert_price_levels(snapshot.get("bids", [])),
            "asks": _convert_price_levels(snapshot.get("asks", [])),
        }
        await self._sink.write(record)
        LOGGER.info("Snapshot stored for %s (update %s)", self._config.normalized_symbol, snapshot["lastUpdateId"])

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
                await self._emit_update(event)
                state.last_update_id = event["u"]
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
            else:
                LOGGER.info(
                    "Small depth init gap tolerated, accepting event (local=%s, first=%s, gap=%s)",
                    state.last_update_id,
                    event["U"],
                    init_gap,
                )
                await self._emit_update(event)
                state.last_update_id = event["u"]
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

            if event["U"] > state.last_update_id + 1:
                gap_size = event["U"] - (state.last_update_id + 1)
                if gap_size > self._config.max_update_gap:
                    LOGGER.warning(
                        "Large depth update gap detected (local=%s, first=%s, final=%s, prev=%s, gap=%s)",
                        state.last_update_id,
                        event["U"],
                        event["u"],
                        event.get("pu"),
                        gap_size,
                    )
                    raise ResyncRequired("Large gap in depth updates; resync required")
                else:
                    LOGGER.info(
                        "Small depth update gap tolerated (local=%s, first=%s, gap=%s)",
                        state.last_update_id,
                        event["U"],
                        gap_size,
                    )

            if event["U"] <= state.last_update_id + 1 <= event["u"]:
                prev_id = event.get("pu")
                if prev_id is not None and prev_id > state.last_update_id + 1:
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
                    else:
                        LOGGER.info(
                            "Small prev id gap tolerated (local=%s, prev=%s, gap=%s)",
                            state.last_update_id,
                            prev_id,
                            prev_gap,
                        )
                await self._emit_update(event)
                state.last_update_id = event["u"]
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
                    event.get("pu"),
                    final_gap,
                )
                raise ResyncRequired("Large final gap in depth events; resync required")
            else:
                LOGGER.info(
                    "Accepting out-of-order update with small gap (local=%s, first=%s, gap=%s)",
                    state.last_update_id,
                    event["U"],
                    final_gap,
                )
                await self._emit_update(event)
                state.last_update_id = event["u"]

    async def _emit_update(self, event: Dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        record = {
            "event_type": event.get("e", "depthUpdate"),
            "event_time": int(event.get("E", 0)),
            "received_time": now,
            "symbol": event.get("s", self._config.normalized_symbol),
            "first_update_id": event.get("U", 0),
            "final_update_id": event.get("u", 0),
            "prev_final_update_id": event.get("pu", event.get("U", 0) - 1),
            "is_snapshot": False,
            "bids": _convert_price_levels(event.get("b", [])),
            "asks": _convert_price_levels(event.get("a", [])),
        }
        await self._sink.write(record)
