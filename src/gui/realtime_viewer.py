from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import aiohttp
import dearpygui.dearpygui as dpg
import numpy as np
import torch

from models.checkpoint import TCNCheckpoint, load_tcn_checkpoint


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ViewerConfig:
    checkpoint_path: Path
    symbol: str = "BTCUSDT"
    input_length: int = 100
    forecast_horizon: int = 10
    depth_limit: int = 25
    max_points: int = 600
    refresh_interval: float = 0.1
    device: Optional[str] = None


@dataclass(slots=True)
class ChartUpdate:
    tick_index: int
    timestamp: float
    mid_price: float
    predictions: Optional[np.ndarray]


@dataclass(slots=True)
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
    api_limits = (5, 10, 20, 50, 100, 500, 1000, 5000)
    for limit in api_limits:
        if depth <= limit:
            return limit
    return api_limits[-1]


async def _fetch_snapshot(session: aiohttp.ClientSession, symbol: str, depth: int) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    api_limit = _select_api_limit(depth)
    params = {"symbol": symbol.upper(), "limit": api_limit}
    async with session.get("https://fapi.binance.com/fapi/v1/depth", params=params) as resp:
        resp.raise_for_status()
        payload = await resp.json()
    bids = [(float(price), float(qty)) for price, qty in payload["bids"]]
    asks = [(float(price), float(qty)) for price, qty in payload["asks"]]
    return bids, asks


def _compute_features(order_book: OrderBook) -> Tuple[np.ndarray, float]:
    best_bid_price, best_bid_qty = order_book.best_bid()
    best_ask_price, best_ask_qty = order_book.best_ask()
    has_prices = np.isfinite(best_bid_price) and np.isfinite(best_ask_price)
    mid_price = (best_bid_price + best_ask_price) / 2 if has_prices else np.nan
    spread = best_ask_price - best_bid_price if has_prices else np.nan
    bid_vol, ask_vol = order_book.aggregate_volume()
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


class RealTimeModelViewer:
    """Visualize live Binance data alongside model forecasts using Dear PyGui."""

    def __init__(self, config: ViewerConfig) -> None:
        self.config = config
        self.checkpoint: TCNCheckpoint = load_tcn_checkpoint(config.checkpoint_path, device=config.device)
        self._device = next(self.checkpoint.model.parameters()).device
        self._queue: Queue[ChartUpdate] = Queue()
        self._status_queue: Queue[str] = Queue()
        self._status_message = "Connecting to Binance..."
        self._stop_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stream_task: Optional[asyncio.Task[None]] = None
        self._data_thread = threading.Thread(target=self._run_streaming_loop, name="binance-stream", daemon=True)

        horizon = self.checkpoint.config.forecast_horizon
        self._actual_points: Deque[Tuple[float, float]] = deque(maxlen=config.max_points)
        self._prediction_points: Dict[int, Deque[Tuple[float, float]]] = {
            h: deque(maxlen=config.max_points) for h in range(1, horizon + 1)
        }
        self._pending_predictions: Deque[Tuple[int, float, np.ndarray]] = deque()
        self._mae_sum = np.zeros(horizon, dtype=np.float64)
        self._mae_count = np.zeros(horizon, dtype=np.int64)
        self._latest_tick = 0
        self._selected_horizon = min(config.forecast_horizon, horizon)

        self._status_text_id: Optional[int] = None
        self._horizon_combo_id: Optional[int] = None
        self._actual_series_id: Optional[int] = None
        self._forecast_series_id: Optional[int] = None
        self._plot_id: Optional[int] = None
        self._x_axis_id: Optional[int] = None
        self._y_axis_id: Optional[int] = None
        self._last_refresh = 0.0
        self._price_format = "%.2f"
        self._start_time: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._estimated_interval = 0.1

    def run(self) -> None:
        self._data_thread.start()
        self._build_ui()
        dpg.create_viewport(title=f"{self.config.symbol} Real-Time Viewer", width=1400, height=900)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("viewer_window", True)
        while dpg.is_dearpygui_running():
            self._on_frame()
            dpg.render_dearpygui_frame()
        self._stop_event.set()
        if self._loop and self._stream_task:
            self._loop.call_soon_threadsafe(self._stream_task.cancel)
        self._data_thread.join(timeout=5)
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        dpg.destroy_context()

    def _build_ui(self) -> None:
        dpg.create_context()
        dpg.set_exit_callback(self._on_exit)
        with dpg.window(tag="viewer_window", label="Real-Time Model Visualizer", width=1350, height=820):
            dpg.add_text(f"Symbol: {self.config.symbol}")
            dpg.add_text(f"Checkpoint: {self.checkpoint.checkpoint_path}")
            self._status_text_id = dpg.add_text(self._status_message)
            dpg.add_separator()
            horizon_items = [f"h={i}" for i in range(1, self.checkpoint.config.forecast_horizon + 1)]
            default_item = f"h={self._selected_horizon}"
            self._horizon_combo_id = dpg.add_combo(
                label="Forecast horizon", items=horizon_items, default_value=default_item, width=150, callback=self._on_horizon_change
            )
            dpg.add_text("- Scroll to zoom, drag with left mouse to pan, right-click for more options.")
            self._plot_id = dpg.add_plot(label="Mid Price vs Forecast", height=-1, width=-1)
            dpg.add_plot_legend(parent=self._plot_id)
            self._x_axis_id = dpg.add_plot_axis(dpg.mvXAxis, label="Time (UTC)", parent=self._plot_id)
            self._y_axis_id = dpg.add_plot_axis(dpg.mvYAxis, label="Price", parent=self._plot_id)
            dpg.set_axis_limits_auto(self._x_axis_id)
            self._actual_series_id = dpg.add_line_series([], [], label="Mid Price", parent=self._y_axis_id)
            self._forecast_series_id = dpg.add_line_series([], [], label=f"Forecast (h={self._selected_horizon})", parent=self._y_axis_id)
    def _on_exit(self) -> None:
        self._stop_event.set()
        self._status_message = "Shutting down viewer..."
        self._status_queue.put(self._status_message)

    def _on_horizon_change(self, sender: int, app_data: str, user_data: None) -> None:
        try:
            horizon = int(app_data.split("=")[1])
        except (IndexError, ValueError):
            return
        horizon = max(1, min(self.checkpoint.config.forecast_horizon, horizon))
        self._selected_horizon = horizon
        if self._forecast_series_id is not None:
            dpg.set_item_label(self._forecast_series_id, f"Forecast (h={horizon})")

    def _on_frame(self) -> None:
        if self._stop_event.is_set():
            return
        now = time.perf_counter()
        if now - self._last_refresh < self.config.refresh_interval:
            return
        self._last_refresh = now
        updated = False
        while True:
            try:
                update = self._queue.get_nowait()
            except Empty:
                break
            self._process_update(update)
            updated = True
        status_changed = False
        while True:
            try:
                status = self._status_queue.get_nowait()
            except Empty:
                break
            self._status_message = status
            status_changed = True
        if updated:
            self._refresh_plot()
        if updated or status_changed:
            self._update_status()

    def _process_update(self, update: ChartUpdate) -> None:
        self._latest_tick = update.tick_index
        timestamp = update.timestamp
        self._actual_points.append((timestamp, update.mid_price))
        if self._start_time is None:
            self._start_time = timestamp
        if self._last_timestamp is not None:
            delta = timestamp - self._last_timestamp
            if delta > 0:
                self._estimated_interval = 0.8 * self._estimated_interval + 0.2 * delta
        self._last_timestamp = timestamp

        if update.predictions is not None:
            preds = np.array(update.predictions, copy=True)
            self._pending_predictions.append((update.tick_index, timestamp, preds))
            target_values = preds[:, 0]
            for horizon_offset, value in enumerate(target_values, start=1):
                pred_time = timestamp + horizon_offset * self._estimated_interval
                self._prediction_points[horizon_offset].append((pred_time, float(value)))

        actual_value = update.mid_price
        remaining: Deque[Tuple[int, float, np.ndarray]] = deque()
        for start_tick, start_time, pred_matrix in self._pending_predictions:
            horizon_index = update.tick_index - start_tick
            if 1 <= horizon_index <= pred_matrix.shape[0]:
                predicted_value = float(pred_matrix[horizon_index - 1, 0])
                error = abs(predicted_value - actual_value)
                self._mae_sum[horizon_index - 1] += error
                self._mae_count[horizon_index - 1] += 1
            if update.tick_index - start_tick < pred_matrix.shape[0]:
                remaining.append((start_tick, start_time, pred_matrix))
        self._pending_predictions = remaining

    def _refresh_plot(self) -> None:
        if self._actual_series_id is None or self._forecast_series_id is None:
            return
        if not self._actual_points:
            return

        base_time = self._start_time or self._actual_points[0][0]
        actual_x = [point[0] - base_time for point in self._actual_points]
        actual_y = [point[1] for point in self._actual_points]
        dpg.set_value(self._actual_series_id, [actual_x, actual_y])

        horizon_points = self._prediction_points[self._selected_horizon]
        pred_x = [point[0] - base_time for point in horizon_points]
        pred_y = [point[1] for point in horizon_points]
        dpg.set_value(self._forecast_series_id, [pred_x, pred_y])

        if self._x_axis_id is not None:
            tick_pairs: List[List[Any]] = []
            if actual_x:
                span = actual_x[-1] - actual_x[0]
                if span <= 0:
                    label = datetime.fromtimestamp(base_time + actual_x[-1], tz=timezone.utc).strftime('%H:%M:%S')
                    tick_pairs.append([float(actual_x[-1]), label])
                else:
                    step = span / 4 if span > 0 else 1.0
                    for i in range(5):
                        value = float(actual_x[0] + step * i)
                        label_time = datetime.fromtimestamp(base_time + value, tz=timezone.utc)
                        tick_pairs.append([value, label_time.strftime('%H:%M:%S')])
            dpg.set_axis_limits_auto(self._x_axis_id)
            if tick_pairs:
                try:
                    dpg.set_axis_ticks(self._x_axis_id, tick_pairs)
                except SystemError as exc:
                    LOGGER.debug("Failed to set axis ticks: %s", exc, exc_info=True)
            else:
                dpg.set_axis_ticks(self._x_axis_id, [])

        if self._y_axis_id is not None:
            if actual_y or pred_y:
                combined = actual_y + pred_y
                y_min = min(combined)
                y_max = max(combined)
                if y_max > y_min:
                    margin = (y_max - y_min) * 0.05 or 1.0
                    dpg.set_axis_limits(self._y_axis_id, y_min - margin, y_max + margin)

    def _update_status(self) -> None:
        if self._status_text_id is None:
            return
        metrics: List[str] = []
        for index in range(self._selected_horizon):
            count = self._mae_count[index]
            if count > 0:
                mae = self._mae_sum[index] / count
                metrics.append(f"h{index + 1} MAE: {mae:.2f}")
        mae_text = " | ".join(metrics) if metrics else "Collecting samples..."
        time_text = ""
        if self._last_timestamp is not None:
            last_dt = datetime.fromtimestamp(self._last_timestamp, tz=timezone.utc)
            time_text = f"Last tick: {last_dt.strftime('%H:%M:%S')} UTC"
        parts: List[str] = []
        if self._status_message:
            parts.append(self._status_message)
        parts.append(f"Ticks: {self._latest_tick}")
        if mae_text:
            parts.append(mae_text)
        if time_text:
            parts.append(time_text)
        dpg.set_value(self._status_text_id, " | ".join(parts))

    def _run_streaming_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._stream_task = loop.create_task(self._stream_data())
        try:
            loop.run_until_complete(self._stream_task)
        except asyncio.CancelledError:
            pass
        finally:
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()
            try:
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            except Exception:
                pass
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _stream_data(self) -> None:
        session_timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=45)
        feature_buffer: Deque[np.ndarray] = deque(maxlen=self.config.input_length)
        order_book = OrderBook(bids={}, asks={}, depth_limit=self.config.depth_limit)
        while not self._stop_event.is_set():
            try:
                self._status_queue.put("Connecting to Binance snapshot...")
                async with aiohttp.ClientSession(timeout=session_timeout) as session:
                    bids, asks = await _fetch_snapshot(session, self.config.symbol, self.config.depth_limit)
                    order_book.apply_snapshot(bids, asks)
                    feature_buffer.clear()
                    tick_index = 0
                    stream_url = f"wss://fstream.binance.com/ws/{self.config.symbol.lower()}@depth@100ms"
                    async with session.ws_connect(stream_url, heartbeat=15) as ws:
                        self._status_queue.put("Connected. Streaming order book updates...")
                        async for message in ws:
                            if self._stop_event.is_set():
                                await ws.close()
                                return
                            if message.type != aiohttp.WSMsgType.TEXT:
                                continue
                            data = message.json()
                            bid_updates = [(float(price), float(qty)) for price, qty in data.get("b", [])]
                            ask_updates = [(float(price), float(qty)) for price, qty in data.get("a", [])]
                            order_book.apply_diff(bid_updates, side="bid")
                            order_book.apply_diff(ask_updates, side="ask")

                            features, mid_price = _compute_features(order_book)
                            if not np.isfinite(mid_price):
                                continue

                            feature_buffer.append(features)
                            predictions: Optional[np.ndarray] = None
                            if len(feature_buffer) == self.config.input_length:
                                predictions = self._infer(np.stack(feature_buffer, axis=0))
                            event_time = data.get("E")
                            timestamp = float(event_time) / 1000.0 if event_time is not None else time.time()
                            update = ChartUpdate(
                                tick_index=tick_index,
                                timestamp=timestamp,
                                mid_price=float(mid_price),
                                predictions=predictions,
                            )
                            self._queue.put(update)
                            tick_index += 1
            except asyncio.CancelledError:
                self._status_queue.put("Streaming cancelled. Shutting down...")
                return
            except Exception as exc:
                LOGGER.exception("Streaming error: %s", exc)
                self._status_queue.put(f"Streaming error: {exc}. Reconnecting...")
                await asyncio.sleep(3)

    def _infer(self, features: np.ndarray) -> np.ndarray:
        norm_features = self.checkpoint.feature_stats.apply(features)
        inputs = torch.from_numpy(norm_features).unsqueeze(0).to(self._device)
        with torch.no_grad():
            preds = self.checkpoint.model(inputs)
        preds_np = preds.squeeze(0).cpu().numpy()
        denorm = self.checkpoint.target_stats.invert(preds_np)
        return denorm


def run_viewer(config: ViewerConfig) -> None:
    viewer = RealTimeModelViewer(config)
    viewer.run()
