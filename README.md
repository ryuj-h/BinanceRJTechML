# Binance Futures Data Collector

Continuous Binance BTCUSDT futures order book and trade capture designed for machine learning workflows.

## Features
- Streams Binance USDT-margined futures depth updates at 100 ms cadence.
- Persists initial snapshots plus diff updates so the full book can be reconstructed offline.
- Captures aggregated trade events in parallel.
- Buffered disk writers (Parquet or JSONL) tuned for high-frequency data ingestion.
- Resilient reconnection loop for long-lived collection sessions.

## Requirements
- Python 3.10+
- `aiohttp`, `pyarrow`, `numpy`, `pandas`, `torch`, `tqdm`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python scripts/collect_binance.py \
    --symbol BTCUSDT \
    --sink-format parquet \
    --data-dir data \
    --flush-interval 5 \
    --max-batch-size 2000
```

The script runs until interrupted (Ctrl+C). Supported sink formats:

- `parquet` (recommended for analytics and ML pipelines)
- `jsonl` (line-delimited JSON for quick inspection or custom tooling)

## Data Layout

Files are partitioned by dataset and date:

```
data/
  orderbook/
    date=2025-09-22/
      orderbook_20250922T101512123456_ab12cd34.parquet
  trades/
    date=2025-09-22/
      trades_20250922T101500000000_a1b2c3d4.parquet
```

Order book rows include the applied diff updates plus the full snapshot that anchors the stream. Trade rows mirror Binance`s aggregated trade schema while adding a local wall-clock timestamp.

## TCN Training Pipeline

```bash
python scripts/train_tcn.py \
    --parquet-path data/processed/orderbook_all_top25_100ms_a.parquet \
    --features best_bid_price best_ask_price mid_price spread bid_volume_top ask_volume_top volume_imbalance \
    --targets mid_price \
    --input-length 100 \
    --forecast-horizon 10 \
    --epochs 25 \
    --batch-size 512
```

The script builds normalized sliding windows (100 ticks in, 10 ticks out) and trains a Temporal Convolutional Network forecaster. Model checkpoints and normalization stats land in `models/` by default. Override the feature or target lists to experiment with deeper ladder levels or custom engineered signals. Training uses tqdm progress bars so you can monitor epoch and batch completion.

## Extensibility

- Additional symbols: launch multiple instances or extend the CLI to accept a list of pairs.
- Alternative sinks: implement another `AsyncDataSink` (e.g., streaming database writer) and reuse the collectors.
- Feature generation: build downstream jobs that load Parquet datasets and derive features for model training.

## Notes

- Network interruptions trigger exponential backoff with automatic resubscription.
- If you switch sinks/formats, ensure downstream consumers handle the corresponding schema.

