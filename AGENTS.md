# Repository Guidelines

## Project Structure & Module Organization
The collector logic lives in `src/binance_data/`, split into `orderbook.py`, `trades.py`, `storage.py`, and `config.py`. Scripts under `scripts/` (`collect_binance.py`, `collect_trades_only.py`) wrap those modules for CLI usage. Runtime artifacts land in `data/` (order book snapshots, trade batches) and `trades/` contains sample parquet outputs; keep generated data out of commits. Configuration defaults sit in `config.py`; extend them rather than inlining constants.

## Build, Test, and Development Commands
Create a virtual environment and install dependencies with `python -m venv .venv` followed by `.\.venv\Scripts\pip install -r requirements.txt`. Run the full collector locally via `python scripts/collect_binance.py --symbol BTCUSDT --sink-format parquet --data-dir data`. For lightweight trade-only capture use `python scripts/collect_trades_only.py --symbol BTCUSDT --data-dir trades`. Before opening a PR, execute `python -m compileall src` to catch syntax issues.

## Coding Style & Naming Conventions
Use PEP 8 with 4-space indentation and keep functions cohesive around a single concern. Maintain the existing type-hinted interfaces; prefer `dataclass(slots=True)` for immutable state buckets as seen in `OrderBookState`. Modules, packages, and files should remain `snake_case`; classes stay `PascalCase`. Loggers should be module-scoped via `logging.getLogger(__name__)`. Prefer dependency injection for sinks/configs to keep collectors testable.

## Testing Guidelines
Place tests in a top-level `tests/` package mirroring the `src/binance_data` layout (`tests/test_orderbook.py`, etc.). Use `pytest` with `pytest-asyncio` for coroutine collectors and mark integration runs that hit Binance as `slow`. Aim for regression coverage over reconnection logic, diff application, and schema serialization; mock network calls with recorded payloads. Run `pytest -q` before pushing changes.

## Commit & Pull Request Guidelines
Follow conventional short messages in the form `type: summary` (e.g., `fix: handle empty diff batch`), referencing linked issues when available. Each PR should summarize behavior changes, list manual test results, and document any schema-impacting adjustments. Include screenshots or sample filenames when altering data layout. Request review when CI or local `pytest` runs green.
