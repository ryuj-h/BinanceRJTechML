from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


LOGGER = logging.getLogger(__name__)


class AsyncDataSink(ABC):
    """Asynchronous interface for persisting structured records."""

    @abstractmethod
    async def write(self, record: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def flush(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError


class ParquetSink(AsyncDataSink):
    """Buffered Parquet writer that periodically flushes to disk."""

    def __init__(
        self,
        root_dir: Path,
        dataset_name: str,
        schema: Any,
        flush_interval_seconds: float = 5.0,
        max_batch_size: int = 1000,
    ) -> None:
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:  # pragma: no cover - import-time guard
            raise RuntimeError(
                "pyarrow is required for Parquet sinks. Install it with 'pip install pyarrow'."
            ) from exc

        self._pa = pa
        self._pq = pq
        self._schema = schema
        self._dataset_name = dataset_name
        self._root = Path(root_dir).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._flush_interval = flush_interval_seconds
        self._max_batch_size = max_batch_size
        self._buffer: List[Dict[str, Any]] = []
        self._last_flush = time.monotonic()
        self._lock = asyncio.Lock()

    async def write(self, record: Dict[str, Any]) -> None:
        async with self._lock:
            self._buffer.append(record)
            if self._should_flush_locked():
                await self._flush_locked()

    async def flush(self) -> None:
        async with self._lock:
            await self._flush_locked()

    async def close(self) -> None:
        await self.flush()

    def _should_flush_locked(self) -> bool:
        if len(self._buffer) >= self._max_batch_size:
            return True
        elapsed = time.monotonic() - self._last_flush
        return elapsed >= self._flush_interval

    async def _flush_locked(self) -> None:
        if not self._buffer:
            return
        records = self._buffer
        self._buffer = []
        self._last_flush = time.monotonic()
        await asyncio.to_thread(self._write_records, records)

    def _write_records(self, records: List[Dict[str, Any]]) -> None:
        timestamp = datetime.now(timezone.utc)
        partition_dir = self._root / self._dataset_name / f"date={timestamp:%Y-%m-%d}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        file_name = (
            f"{self._dataset_name}_{timestamp:%Y%m%dT%H%M%S%f}_{uuid.uuid4().hex[:8]}.parquet"
        )
        table = self._pa.Table.from_pylist(records, schema=self._schema)
        file_path = partition_dir / file_name
        self._pq.write_table(table, file_path)
        LOGGER.debug("Wrote %s records to %s", len(records), file_path)


class JsonlSink(AsyncDataSink):
    """Writable JSONL sink with daily file rotation."""

    def __init__(
        self,
        root_dir: Path,
        dataset_name: str,
        flush_interval_seconds: float = 5.0,
        max_batch_size: int = 1000,
        json_dumps: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> None:
        self._root = Path(root_dir).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._dataset_name = dataset_name
        self._flush_interval = flush_interval_seconds
        self._max_batch_size = max_batch_size
        self._json_dumps = json_dumps or self._default_dumps
        self._buffer: List[Dict[str, Any]] = []
        self._last_flush = time.monotonic()
        self._lock = asyncio.Lock()
        self._current_date_key: Optional[str] = None
        self._file_handle: Optional[Any] = None

    async def write(self, record: Dict[str, Any]) -> None:
        async with self._lock:
            self._buffer.append(record)
            if self._should_flush_locked():
                await self._flush_locked()

    async def flush(self) -> None:
        async with self._lock:
            await self._flush_locked()

    async def close(self) -> None:
        await self.flush()
        if self._file_handle:
            await asyncio.to_thread(self._file_handle.close)
            self._file_handle = None

    def _should_flush_locked(self) -> bool:
        if len(self._buffer) >= self._max_batch_size:
            return True
        elapsed = time.monotonic() - self._last_flush
        return elapsed >= self._flush_interval

    async def _flush_locked(self) -> None:
        if not self._buffer:
            return
        records = self._buffer
        self._buffer = []
        self._last_flush = time.monotonic()
        await asyncio.to_thread(self._write_lines, records)

    def _write_lines(self, records: Iterable[Dict[str, Any]]) -> None:
        timestamp = datetime.now(timezone.utc)
        date_key = f"date={timestamp:%Y-%m-%d}"
        if self._current_date_key != date_key or self._file_handle is None:
            self._open_file(timestamp, date_key)
        assert self._file_handle is not None
        for record in records:
            line = self._json_dumps(record)
            self._file_handle.write(line + "\n")
        self._file_handle.flush()

    def _open_file(self, timestamp: datetime, date_key: str) -> None:
        if self._file_handle is not None:
            self._file_handle.close()
        partition_dir = self._root / self._dataset_name / date_key
        partition_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{self._dataset_name}_{timestamp:%Y%m%d}.jsonl"
        self._file_handle = open(partition_dir / file_name, "a", encoding="utf-8")
        self._current_date_key = date_key

    @staticmethod
    def _default_dumps(record: Dict[str, Any]) -> str:
        def _encoder(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.astimezone(timezone.utc).isoformat()
            raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")

        return json.dumps(record, default=_encoder, ensure_ascii=False)
