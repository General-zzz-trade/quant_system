"""Bar aggregator — up-samples 1m bars to higher timeframes (5m, 15m, 1h, 4h)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable, Dict, Optional, Sequence


@dataclass(frozen=True, slots=True)
class AggregatedBar:
    """OHLCV bar at a specific timeframe."""
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str
    symbol: str = ""


# Timeframe durations in minutes
TIMEFRAME_MINUTES: Dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


def _align_ts(ts: datetime, minutes: int) -> datetime:
    """Align timestamp to timeframe boundary."""
    epoch = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed_minutes = (ts - epoch).total_seconds() / 60
    aligned_minutes = int(elapsed_minutes // minutes) * minutes
    return epoch + timedelta(minutes=aligned_minutes)


class BarAggregator:
    """Aggregates 1-minute bars into higher timeframes.

    Usage:
        agg = BarAggregator(timeframes=["5m", "15m", "1h"])
        for bar_1m in stream:
            completed = agg.on_bar(bar_1m)
            # completed: dict of timeframe -> AggregatedBar (only when a bar closes)
    """

    def __init__(
        self,
        timeframes: Sequence[str] = ("5m", "15m", "1h"),
        symbol: str = "",
    ) -> None:
        self._symbol = symbol
        self._timeframes = list(timeframes)
        self._minutes = {tf: TIMEFRAME_MINUTES[tf] for tf in timeframes}

        # Current accumulating bar per timeframe
        self._current: Dict[str, _BarAccumulator] = {}
        for tf in timeframes:
            self._current[tf] = _BarAccumulator()

        # Last completed bar per timeframe
        self._last_completed: Dict[str, Optional[AggregatedBar]] = {tf: None for tf in timeframes}

    def on_bar(
        self,
        ts: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0,
    ) -> Dict[str, AggregatedBar]:
        """Feed a 1m bar. Returns dict of completed bars (if any timeframe closed)."""
        completed: Dict[str, AggregatedBar] = {}

        for tf in self._timeframes:
            mins = self._minutes[tf]
            aligned = _align_ts(ts, mins)
            acc = self._current[tf]

            # New bar period started — emit the previous accumulated bar
            if acc.start_ts is not None and aligned != acc.start_ts:
                bar = acc.to_bar(tf, self._symbol)
                if bar is not None:
                    completed[tf] = bar
                    self._last_completed[tf] = bar
                acc.reset()

            acc.update(aligned, open, high, low, close, volume)

        return completed

    def get_last(self, timeframe: str) -> Optional[AggregatedBar]:
        """Get the last completed bar for a timeframe."""
        return self._last_completed.get(timeframe)

    def get_current(self, timeframe: str) -> Optional[AggregatedBar]:
        """Get the current (incomplete) bar for a timeframe."""
        acc = self._current.get(timeframe)
        if acc is None:
            return None
        return acc.to_bar(timeframe, self._symbol)

    @property
    def timeframes(self) -> list[str]:
        return list(self._timeframes)


class _BarAccumulator:
    """Accumulates ticks/sub-bars into a single OHLCV bar."""

    __slots__ = ("start_ts", "open", "high", "low", "close", "volume")

    def __init__(self) -> None:
        self.start_ts: Optional[datetime] = None
        self.open: float = 0.0
        self.high: float = 0.0
        self.low: float = 0.0
        self.close: float = 0.0
        self.volume: float = 0.0

    def update(
        self,
        ts: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        if self.start_ts is None:
            self.start_ts = ts
            self.open = open
            self.high = high
            self.low = low
        else:
            self.high = max(self.high, high)
            self.low = min(self.low, low)
        self.close = close
        self.volume += volume

    def reset(self) -> None:
        self.start_ts = None
        self.open = 0.0
        self.high = 0.0
        self.low = 0.0
        self.close = 0.0
        self.volume = 0.0

    def to_bar(self, timeframe: str, symbol: str) -> Optional[AggregatedBar]:
        if self.start_ts is None:
            return None
        return AggregatedBar(
            ts=self.start_ts,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            timeframe=timeframe,
            symbol=symbol,
        )
