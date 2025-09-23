# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python application for collecting real-time Binance BTCUSDT futures market data designed for machine learning workflows. It streams order book depth updates and aggregated trades, persisting them to Parquet or JSONL formats with date partitioning.

## Commands

### Running the Data Collector
```bash
python scripts/collect_binance.py --symbol BTCUSDT --sink-format parquet --data-dir data --flush-interval 5 --max-batch-size 2000
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Common Development Commands
- No test suite is present in this codebase
- No linting/formatting tools are configured
- No build process is defined

## Architecture

### Core Components

**Data Collection Pipeline:**
- `OrderBookCollector` (orderbook.py): Fetches REST snapshots then streams WebSocket depth updates, maintaining sequence integrity
- `TradeCollector` (trades.py): Streams aggregated trade data via WebSocket
- Both collectors run concurrently and handle reconnection with exponential backoff

**Storage Layer:**
- `AsyncDataSink` interface with two implementations:
  - `ParquetSink`: Buffered Parquet writer with PyArrow schemas for analytics
  - `JsonlSink`: Line-delimited JSON with daily file rotation
- Both sinks use configurable batch sizes and flush intervals

**Configuration:**
- `CollectorConfig`: Main configuration including symbol, timeouts, backoff settings
- `SinkConfig`: Storage configuration (format, directory, batch settings)

### Data Flow

1. Main script (`collect_binance.py`) creates collectors and sinks based on CLI args
2. OrderBook collector fetches initial snapshot via REST, then synchronizes with WebSocket stream
3. Trade collector connects directly to aggregated trade WebSocket
4. Both streams write to their respective sinks with buffering
5. Data is partitioned by date: `data/{dataset}/date=YYYY-MM-DD/`

### Key Integration Points

- **Schema Definitions**: `schemas.py` defines PyArrow schemas for both datasets
- **WebSocket Synchronization**: OrderBook collector implements Binance's recommended sync protocol (snapshot + diff updates)
- **Error Handling**: Network errors trigger exponential backoff reconnection loops
- **Signal Handling**: Graceful shutdown on SIGINT/SIGTERM with proper resource cleanup

### File Structure

- `src/binance_data/`: Core library modules
- `scripts/`: Entry point scripts
- `data/`: Default output directory (created at runtime)

The codebase follows async/await patterns throughout and is designed for long-running data collection sessions.