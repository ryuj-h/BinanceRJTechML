from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from models.training import TrainingConfig, train_tcn_model

DEFAULT_FEATURES = [
    "best_bid_price",
    "best_ask_price",
    "mid_price",
    "spread",
    "bid_volume_top",
    "ask_volume_top",
    "volume_imbalance",
]
DEFAULT_TARGETS = ["mid_price"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TCN forecaster on order book windows.")
    parser.add_argument("--parquet-path", type=Path, required=True, help="Path to the parquet file with order book snapshots.")
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES, help="Feature columns for the input window.")
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS, help="Target columns to predict across the forecast horizon.")
    parser.add_argument("--input-length", type=int, default=100, help="Number of ticks in the input window.")
    parser.add_argument("--forecast-horizon", type=int, default=10, help="Number of ticks to predict.")
    parser.add_argument("--stride", type=int, default=1, help="Stride between windows when creating the dataset.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size for training.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="L2 regularization factor.")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Fraction of windows reserved for validation.")
    parser.add_argument("--no-normalize-targets", action="store_true", help="Disable target normalization.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models"), help="Directory to store the trained model.")
    parser.add_argument("--checkpoint-name", default="tcn_forecaster.pt", help="Model checkpoint file name.")
    parser.add_argument("--device", default=None, help="Device override (e.g., cuda, cpu). Defaults to auto-detect.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = TrainingConfig(
        parquet_path=args.parquet_path,
        feature_columns=args.features,
        target_columns=args.targets,
        input_length=args.input_length,
        forecast_horizon=args.forecast_horizon,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_split=args.validation_split,
        normalize_targets=not args.no_normalize_targets,
        model_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )

    metrics = train_tcn_model(config)
    print("Training complete")
    print(f"Train loss: {metrics['train_loss']:.6f}")
    if metrics["val_loss"] == metrics["val_loss"]:
        print(f"Validation loss: {metrics['val_loss']:.6f}")
    else:
        print("Validation loss: N/A (no validation split)")
    print(f"Model saved to: {metrics['checkpoint_path']}")


if __name__ == "__main__":
    main()
