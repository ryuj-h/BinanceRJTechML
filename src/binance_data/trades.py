from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

import aiohttp

from .config import CollectorConfig
from .storage import AsyncDataSink

LOGGER = logging.getLogger(__name__)

WEBSOCKET_BASE_URL = "wss://fstream.binance.com/ws"


class TradeCollector:
    """Streams aggregated trade data from Binance futures."""

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
                LOGGER.exception("Trade collector error: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._config.max_backoff_seconds)

    async def _run_once(self) -> None:
        stream_url = f"{WEBSOCKET_BASE_URL}/{self._config.stream_symbol}@aggTrade"
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            LOGGER.info("Connecting trade stream %s", stream_url)
            async with session.ws_connect(stream_url, heartbeat=15) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        record = self._to_record(data)
                        await self._sink.write(record)
                    elif msg.type == aiohttp.WSMsgType.PING:
                        await ws.pong()
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError("Trade websocket encountered an error")
                    elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                        raise RuntimeError("Trade websocket closed")

    def _to_record(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        price = float(payload.get("p", 0.0))
        quantity = float(payload.get("q", 0.0))
        record = {
            "event_type": payload.get("e", "aggTrade"),
            "event_time": int(payload.get("E", 0)),
            "trade_time": int(payload.get("T", payload.get("E", 0))),
            "received_time": now,
            "symbol": payload.get("s", self._config.normalized_symbol),
            "agg_id": int(payload.get("a", 0)),
            "price": price,
            "qty": quantity,
            "first_trade_id": int(payload.get("f", 0)),
            "last_trade_id": int(payload.get("l", 0)),
            "is_buyer_maker": bool(payload.get("m", False)),
        }
        return record
