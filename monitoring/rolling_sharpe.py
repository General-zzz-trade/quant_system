# monitoring/rolling_sharpe.py
"""Lightweight per-symbol rolling Sharpe ratio tracker.

Updated bar-by-bar from the engine. Uses deque for O(1) append/evict
and minimal memory footprint.
"""
from __future__ import annotations

import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Optional


# Annualization factors: bars_per_year by interval
_BARS_PER_YEAR = {
    "1h": 8760,
    "4h": 2190,
    "15m": 35040,
    "1d": 365,
}


class RollingSharpeTracker:
    """Per-symbol rolling Sharpe ratio tracker."""

    def __init__(self, window: int = 720, interval: str = "1h") -> None:
        self._window = window
        self._interval = interval
        self._returns: Dict[str, deque] = {}  # symbol -> deque of returns
        # Running sums for O(1) Sharpe computation
        self._sum: Dict[str, float] = {}
        self._sum_sq: Dict[str, float] = {}

    def update(self, symbol: str, pnl_pct: float) -> None:
        """Record a bar's PnL percentage."""
        if symbol not in self._returns:
            self._returns[symbol] = deque(maxlen=self._window)
            self._sum[symbol] = 0.0
            self._sum_sq[symbol] = 0.0

        buf = self._returns[symbol]
        # If buffer is full, subtract the value being evicted
        if len(buf) == self._window:
            old = buf[0]
            self._sum[symbol] -= old
            self._sum_sq[symbol] -= old * old

        buf.append(pnl_pct)
        self._sum[symbol] += pnl_pct
        self._sum_sq[symbol] += pnl_pct * pnl_pct

    def sharpe(self, symbol: str) -> Optional[float]:
        """Rolling Sharpe for symbol (annualized). None if insufficient data (<30 bars)."""
        buf = self._returns.get(symbol)
        if buf is None or len(buf) < 30:
            return None

        n = len(buf)
        mean = self._sum[symbol] / n
        var = self._sum_sq[symbol] / n - mean * mean
        # Guard against negative variance from float drift
        if var <= 0:
            return 0.0 if mean <= 0 else float("inf")

        std = math.sqrt(var)
        raw_sharpe = mean / std
        bars_per_year = _BARS_PER_YEAR.get(self._interval, 8760)
        return raw_sharpe * math.sqrt(bars_per_year)

    def report(self) -> Dict[str, float]:
        """All symbols' current rolling Sharpe (only those with sufficient data)."""
        out: Dict[str, float] = {}
        for symbol in self._returns:
            s = self.sharpe(symbol)
            if s is not None:
                out[symbol] = round(s, 3)
        return out

    def status(self) -> Dict[str, str]:
        """GREEN/YELLOW/RED per symbol based on rolling Sharpe."""
        out: Dict[str, str] = {}
        for symbol in self._returns:
            s = self.sharpe(symbol)
            if s is None:
                out[symbol] = "WARMUP"
            elif s > 1.0:
                out[symbol] = "GREEN"
            elif s > 0:
                out[symbol] = "YELLOW"
            else:
                out[symbol] = "RED"
        return out


if __name__ == "__main__":
    # Load from ic_health.json if available, otherwise print usage
    ic_path = Path("ic_health.json")
    if ic_path.exists():
        data = json.loads(ic_path.read_text())
        print("=== Rolling Sharpe Report (from ic_health.json) ===")
        tracker = RollingSharpeTracker(window=720, interval="1h")
        # ic_health.json may contain per-symbol return series
        for symbol, info in data.items():
            returns = info if isinstance(info, list) else info.get("returns", [])
            for r in returns:
                tracker.update(symbol, float(r))
        report = tracker.report()
        statuses = tracker.status()
        for sym in sorted(report):
            print(f"  {sym}: Sharpe={report[sym]:.3f}  [{statuses.get(sym, '?')}]")
        if not report:
            print("  (no symbols with sufficient data)")
    else:
        print("Usage: python -m monitoring.rolling_sharpe")
        print()
        print("Reads ic_health.json from CWD if available.")
        print("Otherwise, use RollingSharpeTracker programmatically:")
        print()
        print("  from monitoring.rolling_sharpe import RollingSharpeTracker")
        print("  tracker = RollingSharpeTracker(window=720, interval='1h')")
        print("  tracker.update('BTCUSDT', 0.12)  # bar PnL %")
        print("  print(tracker.report())")
        print("  print(tracker.status())")
        sys.exit(0)
