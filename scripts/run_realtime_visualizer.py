from pathlib import Path
import argparse
import logging
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gui.realtime_viewer import ViewerConfig, run_viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time visualization of Binance data with TCN forecasts.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained TCN checkpoint file.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol to stream from Binance (e.g., BTCUSDT).")
    parser.add_argument("--input-length", type=int, default=100, help="Sliding window length expected by the model.")
    parser.add_argument("--forecast-horizon", type=int, default=10, help="Forecast horizon to display.")
    parser.add_argument("--depth-limit", type=int, default=25, help="Order book depth to maintain.")
    parser.add_argument("--max-points", type=int, default=600, help="Max points to retain for plotting.")
    parser.add_argument("--refresh-interval", type=float, default=0.1, help="UI refresh interval in seconds.")
    parser.add_argument("--device", default=None, help="Torch device override (cuda/cpu). Defaults to checkpoint device.")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error", "critical"], help="Logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = ViewerConfig(
        checkpoint_path=args.checkpoint,
        symbol=args.symbol,
        input_length=args.input_length,
        forecast_horizon=args.forecast_horizon,
        depth_limit=args.depth_limit,
        max_points=args.max_points,
        refresh_interval=args.refresh_interval,
        device=args.device,
    )
    run_viewer(config)


if __name__ == "__main__":
    main()
