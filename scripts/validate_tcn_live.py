from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Tuple

import aiohttp
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from models.checkpoint import load_tcn_checkpoint

DEPTH_SNAPSHOT_URL = "https://fapi.binance.com/fapi/v1/depth"
WEBSOCKET_BASE_URL = "wss://fstream.binance.com/ws"
API_DEPTH_LIMITS = (5, 10, 20, 50, 100, 500, 1000, 5000)
LOGGER = logging.getLogger("validate_tcn_live")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained TCN model on live Binance depth data.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance futures symbol (e.g., BTCUSDT).")
    parser.add_argument("--input-length", type=int, default=100, help="Window size expected by the model.")
    parser.add_argument("--forecast-horizon", type=int, default=10, help="Forecast horizon expected by the model.")
    parser.add_argument("--depth-limit", type=int, default=25, help="Order book depth to maintain.")
    parser.add_argument("--report-interval", type=int, default=100, help="Ticks between metric reports.")
    parser.add_argument("--device", default=None, help="Device override (cuda/cpu). Defaults to checkpoint device.")
    return parser.parse_args()


@dataclass
class OrderBook:
    bids: Dict[float, float]
    asks: Dict[float, float]
    depth_limit: int

    def apply_snapshot(self, bids: Iterable[Tuple[float, float]], asks: Iterable[Tuple[float, float]]) -> None:
        self.bids = {price: qty for price, qty in bids if qty > 0}
        self.asks = {price: qty for price, qty in asks if qty > 0}

    def apply_diff(self, updates: Iterable[Tuple[float, float]], side: str) -> None:
        book = self.bids if side == "bid" else self.asks
        for price, qty in updates:
            if qty == 0:
                book.pop(price, None)
            else:
                book[price] = qty

    def top_levels(self, side: str) -> List[Tuple[float, float]]:
        book = self.bids if side == "bid" else self.asks
        reverse = side == "bid"
        sorted_levels = sorted(book.items(), key=lambda x: x[0], reverse=reverse)
        return sorted_levels[: self.depth_limit]

    def best_bid(self) -> Tuple[float, float]:
        levels = self.top_levels("bid")
        return levels[0] if levels else (np.nan, np.nan)

    def best_ask(self) -> Tuple[float, float]:
        levels = self.top_levels("ask")
        return levels[0] if levels else (np.nan, np.nan)

    def aggregate_volume(self) -> Tuple[float, float]:
        bid_vol = sum(qty for _, qty in self.top_levels("bid"))
        ask_vol = sum(qty for _, qty in self.top_levels("ask"))
        return bid_vol, ask_vol


def _select_api_limit(depth: int) -> int:
    for limit in API_DEPTH_LIMITS:
        if depth <= limit:
            return limit
    return API_DEPTH_LIMITS[-1]


async def fetch_snapshot(symbol: str, depth: int) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    api_limit = _select_api_limit(depth)
    params = {"symbol": symbol.upper(), "limit": api_limit}
    async with aiohttp.ClientSession() as session:
        async with session.get(DEPTH_SNAPSHOT_URL, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
    bids = [(float(p), float(q)) for p, q in data["bids"]]
    asks = [(float(p), float(q)) for p, q in data["asks"]]
    return bids, asks


def compute_features(book: OrderBook) -> Tuple[np.ndarray, float]:
    best_bid_price, best_bid_qty = book.best_bid()
    best_ask_price, best_ask_qty = book.best_ask()
    mid_price = (best_bid_price + best_ask_price) / 2 if np.isfinite(best_bid_price) and np.isfinite(best_ask_price) else np.nan
    spread = best_ask_price - best_bid_price if np.isfinite(best_bid_price) and np.isfinite(best_ask_price) else np.nan
    bid_vol, ask_vol = book.aggregate_volume()
    imbalance = 0.0
    if bid_vol + ask_vol > 0:
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    features = np.array(
        [
            best_bid_price,
            best_ask_price,
            mid_price,
            spread,
            bid_vol,
            ask_vol,
            imbalance,
        ],
        dtype=np.float32,
    )
    return features, float(mid_price)


async def run_validation(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_tcn_checkpoint(args.checkpoint, device=device)
    model = checkpoint.model
    feature_stats = checkpoint.feature_stats
    target_stats = checkpoint.target_stats
    if args.forecast_horizon != checkpoint.config.forecast_horizon:
        LOGGER.warning(
            "Adjusting forecast horizon from %s to checkpoint value %s",
            args.forecast_horizon,
            checkpoint.config.forecast_horizon,
        )
        args.forecast_horizon = checkpoint.config.forecast_horizon

    feature_buffer: Deque[np.ndarray] = deque(maxlen=args.input_length)
    target_history: Dict[int, float] = {}
    pending_predictions: Deque[Tuple[int, np.ndarray]] = deque()

    order_book = OrderBook(bids={}, asks={}, depth_limit=args.depth_limit)
    bids, asks = await fetch_snapshot(args.symbol, args.depth_limit)
    order_book.apply_snapshot(bids, asks)

    tick_index = 0
    mae_sum = np.zeros(args.forecast_horizon, dtype=np.float64)
    mae_count = np.zeros(args.forecast_horizon, dtype=np.int64)

    stream_url = f"{WEBSOCKET_BASE_URL}/{args.symbol.lower()}@depth@100ms"
    LOGGER.info("Connecting to %s", stream_url)

    session_timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=45)
    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        async with session.ws_connect(stream_url, heartbeat=15) as ws:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = msg.json()
                bid_updates = [(float(p), float(q)) for p, q in data.get("b", [])]
                ask_updates = [(float(p), float(q)) for p, q in data.get("a", [])]
                order_book.apply_diff(bid_updates, side="bid")
                order_book.apply_diff(ask_updates, side="ask")

                features, mid_price = compute_features(order_book)
                if not np.isfinite(mid_price):
                    continue

                feature_buffer.append(features)
                target_history[tick_index] = mid_price

                if len(feature_buffer) == args.input_length:
                    norm_features = feature_stats.apply(np.stack(feature_buffer, axis=0))
                    inputs = torch.from_numpy(norm_features).unsqueeze(0).to(device)
                    with torch.no_grad():
                        preds = model(inputs)
                    preds_np = preds.squeeze(0).cpu().numpy()
                    denorm_preds = target_stats.invert(preds_np)
                    pending_predictions.append((tick_index, denorm_preds))

                if pending_predictions:
                    updated_predictions: Deque[Tuple[int, np.ndarray]] = deque()
                    actual_value = target_history.get(tick_index)
                    is_valid_actual = actual_value is not None and np.isfinite(actual_value)
                    for start_tick, pred_vector in pending_predictions:
                        horizon_idx = tick_index - start_tick
                        if 1 <= horizon_idx <= args.forecast_horizon and is_valid_actual:
                            error = abs(pred_vector[horizon_idx - 1] - actual_value)
                            mae_sum[horizon_idx - 1] += error
                            mae_count[horizon_idx - 1] += 1
                        if tick_index - start_tick < args.forecast_horizon:
                            updated_predictions.append((start_tick, pred_vector))
                    pending_predictions = updated_predictions

                if tick_index and tick_index % args.report_interval == 0:
                    metrics = []
                    for h in range(args.forecast_horizon):
                        if mae_count[h] > 0:
                            metrics.append(f"h{h+1}: {mae_sum[h] / mae_count[h]:.4f}")
                    if metrics:
                        LOGGER.info("Tick %s - cumulative MAE %s", tick_index, ", ".join(metrics))

                tick_index += 1


def main() -> None:
    args = parse_args()
    asyncio.run(run_validation(args))


if __name__ == "__main__":
    main()
