"""Historical kline backfiller — batch download and gap fill from exchange REST APIs."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Protocol, Sequence

from data.store import Bar

logger = logging.getLogger(__name__)


class BarStoreProtocol(Protocol):
    """Minimal store interface required by the backfiller."""

    def read_bars(
        self,
        symbol: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> List[Bar]: ...

    def write_bars(self, symbol: str, bars: Sequence[Bar]) -> Any: ...


# Type alias for the exchange kline fetcher.
# Signature: (symbol, interval, start_ms, end_ms, limit) -> List[list]
KlineFetcher = Callable[[str, str, int, int, int], List[list]]


@dataclass(frozen=True, slots=True)
class BackfillConfig:
    """Configuration for historical backfill."""

    symbols: tuple[str, ...]
    interval: str = "1h"
    max_requests_per_minute: int = 1200
    batch_size: int = 1000


_INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}


def _interval_to_seconds(interval: str) -> int:
    """Convert interval string to seconds."""
    if interval in _INTERVAL_SECONDS:
        return _INTERVAL_SECONDS[interval]
    raise ValueError(f"unsupported interval: {interval}")


def _kline_to_bar(raw: list, symbol: str, exchange: str = "") -> Bar:
    """Convert a raw kline list [open_time, o, h, l, c, vol, ...] to a Bar."""
    ts = datetime.fromtimestamp(int(raw[0]) / 1000, tz=timezone.utc)
    return Bar(
        ts=ts,
        open=Decimal(str(raw[1])),
        high=Decimal(str(raw[2])),
        low=Decimal(str(raw[3])),
        close=Decimal(str(raw[4])),
        volume=Decimal(str(raw[5])) if len(raw) > 5 and raw[5] is not None else None,
        symbol=symbol,
        exchange=exchange,
    )


class HistoricalBackfiller:
    """Batch backfill historical klines from an exchange REST API.

    Auto-detects existing data to avoid re-downloading bars already stored.
    Respects rate limits by sleeping between request batches.
    """

    def __init__(
        self,
        *,
        fetch_klines: KlineFetcher,
        bar_store: Any,
        config: BackfillConfig,
        exchange: str = "",
    ) -> None:
        self._fetch = fetch_klines
        self._store = bar_store
        self._config = config
        self._exchange = exchange
        self._min_request_interval = 60.0 / max(config.max_requests_per_minute, 1)

    def backfill(self, symbol: str, start: datetime, end: datetime) -> int:
        """Backfill bars for *symbol* in the date range [start, end].

        Returns the count of new bars written.
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        interval_sec = _interval_to_seconds(self._config.interval)

        # Detect existing range to skip already-fetched data
        existing = self._store.read_bars(symbol, start=start, end=end)
        existing_ts = {b.ts for b in existing}

        logger.info(
            "Backfill %s [%s -> %s] interval=%s, existing=%d bars",
            symbol,
            start.isoformat(),
            end.isoformat(),
            self._config.interval,
            len(existing_ts),
        )

        batch_duration = timedelta(seconds=interval_sec * self._config.batch_size)
        cursor = start
        total_written = 0

        while cursor < end:
            batch_end = min(cursor + batch_duration, end)
            start_ms = int(cursor.timestamp() * 1000)
            end_ms = int(batch_end.timestamp() * 1000)

            request_start = time.monotonic()
            raw_klines = self._fetch(
                symbol,
                self._config.interval,
                start_ms,
                end_ms,
                self._config.batch_size,
            )

            if not raw_klines:
                cursor = batch_end
                self._rate_limit(request_start)
                continue

            bars: list[Bar] = []
            for kline in raw_klines:
                bar = _kline_to_bar(kline, symbol, self._exchange)
                if bar.ts not in existing_ts:
                    bars.append(bar)
                    existing_ts.add(bar.ts)

            if bars:
                self._store.write_bars(symbol, bars)
                total_written += len(bars)
                logger.debug(
                    "Wrote %d new bars for %s (%s -> %s)",
                    len(bars),
                    symbol,
                    cursor.isoformat(),
                    batch_end.isoformat(),
                )

            cursor = batch_end
            self._rate_limit(request_start)

        logger.info("Backfill %s complete: %d new bars", symbol, total_written)
        return total_written

    def backfill_all(self, start: datetime, end: datetime) -> Dict[str, int]:
        """Backfill all configured symbols. Returns mapping of symbol -> bars written."""
        results: Dict[str, int] = {}
        for symbol in self._config.symbols:
            results[symbol] = self.backfill(symbol, start, end)
        return results

    def _rate_limit(self, request_start: float) -> None:
        """Sleep if needed to respect rate limit."""
        elapsed = time.monotonic() - request_start
        sleep_time = self._min_request_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
