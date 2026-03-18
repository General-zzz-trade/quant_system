"""Data store types -- minimal stubs for the data layer.

The data layer (data/) is NOT part of the default production runtime.
These types are used by data collectors, backfill scripts, and storage backends.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass(frozen=True, slots=True)
class Bar:
    """OHLCV bar data point."""

    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Optional[Decimal]
    symbol: str = ""
    exchange: str = ""


class TimeSeriesStore:
    """Parquet-based time series storage for bar data.

    Stores bars as parquet files partitioned by symbol under a root directory.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _symbol_path(self, symbol: str) -> Path:
        return self._root / f"{symbol}.parquet"

    def _fallback_path(self, symbol: str) -> Path:
        return self._root / f"{symbol}.jsonl"

    def write_bars(self, symbol: str, bars: Sequence[Bar]) -> None:
        """Write bars to parquet file for the given symbol."""
        pa, pq = _try_import_pyarrow()
        if pa is None or pq is None:
            self._write_bars_jsonl(symbol, bars)
            return

        records = [
            {
                "ts": b.ts,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": float(b.volume) if b.volume is not None else None,
                "symbol": b.symbol or symbol,
                "exchange": b.exchange,
            }
            for b in bars
        ]
        table = pa.Table.from_pylist(records)
        path = self._symbol_path(symbol)

        if path.exists():
            existing = pq.read_table(path)
            table = pa.concat_tables([existing, table])

        pq.write_table(table, path)

    def read_bars(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Bar]:
        """Read bars for a symbol, optionally filtered by time range."""
        path = self._symbol_path(symbol)
        fallback_path = self._fallback_path(symbol)
        if not path.exists() and not fallback_path.exists():
            return []

        pa, pq = _try_import_pyarrow()
        if path.exists():
            if pq is None:
                raise RuntimeError(
                    "TimeSeriesStore requires pyarrow to read existing parquet files"
                )
            table = pq.read_table(path)
            rows = table.to_pylist()
        else:
            rows = self._read_rows_jsonl(fallback_path)

        bars = []
        for r in rows:
            ts = r["ts"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            bars.append(
                Bar(
                    ts=ts,
                    open=Decimal(str(r["open"])),
                    high=Decimal(str(r["high"])),
                    low=Decimal(str(r["low"])),
                    close=Decimal(str(r["close"])),
                    volume=Decimal(str(r["volume"])) if r.get("volume") is not None else None,
                    symbol=r.get("symbol", symbol),
                    exchange=r.get("exchange", ""),
                )
            )
        return bars

    def list_symbols(self) -> List[str]:
        """List all symbols with stored data."""
        symbols = {p.stem for p in self._root.glob("*.parquet") if p.is_file()}
        symbols.update(p.stem for p in self._root.glob("*.jsonl") if p.is_file())
        return sorted(symbols)

    def _write_bars_jsonl(self, symbol: str, bars: Sequence[Bar]) -> None:
        path = self._fallback_path(symbol)
        with path.open("a", encoding="utf-8") as fh:
            for bar in bars:
                fh.write(json.dumps(_bar_to_json_record(bar, symbol), sort_keys=True))
                fh.write("\n")

    @staticmethod
    def _read_rows_jsonl(path: Path) -> List[dict]:
        rows: List[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


def _try_import_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        return None, None
    return pa, pq


def _bar_to_json_record(bar: Bar, symbol: str) -> dict:
    return {
        "ts": bar.ts.isoformat(),
        "open": str(bar.open),
        "high": str(bar.high),
        "low": str(bar.low),
        "close": str(bar.close),
        "volume": str(bar.volume) if bar.volume is not None else None,
        "symbol": bar.symbol or symbol,
        "exchange": bar.exchange,
    }
