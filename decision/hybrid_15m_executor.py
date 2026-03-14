"""Hybrid 1h Signal + 15m Execution — uses 1h model for direction, 15m bars for timing.

The core idea:
- 1h model gives the DIRECTION (long/short/flat via z-score)
- 15m bars give better TIMING (enter on pullbacks, exit on micro-reversals)

How it works:
1. Accumulate 15m bars → aggregate to 1h OHLCV every 4 bars
2. Compute 1h features on aggregated bars → run 1h ML model → z-score
3. On each 15m bar, check entry/exit conditions:
   - ENTRY: 1h signal says "go long" + 15m price pulls back to support
   - EXIT:  15m micro-reversal or trailing stop hit

Entry timing heuristics (15m level):
- Pullback entry: wait for 15m close below 15m MA5 before entering long
- Momentum entry: 15m RSI crosses above 30 (oversold bounce)
- Immediate entry: if signal is very strong (|z| > 2x deadzone), enter now

Exit timing heuristics (15m level):
- Micro trailing stop: tighter than 1h (e.g., 1.5% vs 2%)
- 15m momentum fade: 15m RSI > 70 while long → exit
- Signal reversal: 1h z-score reverses
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np


@dataclass
class _Bar15m:
    """Single 15m bar."""
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: int = 0


@dataclass
class _BarAccumulator:
    """Accumulates 15m bars into 1h bars."""
    buffer: List[_Bar15m] = field(default_factory=list)
    completed_1h: List[Dict[str, float]] = field(default_factory=list)

    def push(self, bar: _Bar15m) -> Optional[Dict[str, float]]:
        """Add a 15m bar. Returns aggregated 1h bar when 4 bars complete."""
        self.buffer.append(bar)
        if len(self.buffer) >= 4:
            agg = {
                "open": self.buffer[0].open,
                "high": max(b.high for b in self.buffer),
                "low": min(b.low for b in self.buffer),
                "close": self.buffer[-1].close,
                "volume": sum(b.volume for b in self.buffer),
                "open_time": self.buffer[0].timestamp,
            }
            self.completed_1h.append(agg)
            self.buffer.clear()
            return agg
        return None


class MicroTiming:
    """15m-level entry/exit timing signals.

    Computes fast indicators on 15m bars to find optimal
    entry/exit points within the 1h signal direction.
    """

    def __init__(self, ma_period: int = 5, rsi_period: int = 6):
        self._closes: Deque[float] = deque(maxlen=max(ma_period, rsi_period) + 5)
        self._ma_period = ma_period
        self._rsi_period = rsi_period

    def update(self, close: float) -> None:
        self._closes.append(close)

    @property
    def ready(self) -> bool:
        return len(self._closes) >= self._rsi_period + 1

    @property
    def ma(self) -> float:
        """Simple MA of recent 15m closes."""
        if len(self._closes) < self._ma_period:
            return self._closes[-1]
        return float(np.mean(list(self._closes)[-self._ma_period:]))

    @property
    def rsi(self) -> float:
        """Fast RSI on 15m bars."""
        if len(self._closes) < self._rsi_period + 1:
            return 50.0
        closes = list(self._closes)[-(self._rsi_period + 1):]
        deltas = [closes[i+1] - closes[i] for i in range(len(closes)-1)]
        gains = [max(d, 0) for d in deltas]
        losses = [max(-d, 0) for d in deltas]
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        if avg_loss < 1e-12:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @property
    def price_vs_ma(self) -> float:
        """Current price relative to MA (negative = below MA = pullback)."""
        ma = self.ma
        if ma < 1e-12:
            return 0.0
        return (self._closes[-1] - ma) / ma

    def is_pullback_entry(self, direction: int) -> bool:
        """Check if 15m shows a pullback in the signal direction.

        For long: price below MA (pulled back) + RSI not overbought
        For short: price above MA (bounced) + RSI not oversold
        """
        if not self.ready:
            return False
        if direction > 0:
            return self.price_vs_ma < -0.001 and self.rsi < 65
        elif direction < 0:
            return self.price_vs_ma > 0.001 and self.rsi > 35
        return False

    def is_momentum_entry(self, direction: int) -> bool:
        """Check if momentum aligns with signal direction.

        For long: RSI crossing up from oversold
        For short: RSI crossing down from overbought
        """
        if not self.ready:
            return False
        if direction > 0:
            return 25 < self.rsi < 50  # recovering from oversold
        elif direction < 0:
            return 50 < self.rsi < 75  # fading from overbought
        return False

    def should_exit_micro(self, direction: int) -> bool:
        """Check if 15m micro-structure suggests exit.

        For long: RSI > 80 (overbought) or price far above MA
        For short: RSI < 20 (oversold) or price far below MA
        """
        if not self.ready:
            return False
        if direction > 0:
            return self.rsi > 80 or self.price_vs_ma > 0.015
        elif direction < 0:
            return self.rsi < 20 or self.price_vs_ma < -0.015
        return False


class Hybrid15mExecutor:
    """1h signal + 15m execution hybrid strategy.

    Parameters
    ----------
    deadzone : float
        Z-score threshold for entry signal (from 1h model).
    min_hold_15m : int
        Minimum hold in 15m bars before exit allowed.
    max_hold_15m : int
        Maximum hold in 15m bars before forced exit.
    trailing_stop_pct : float
        Trailing stop percentage on 15m bars (tighter than 1h).
    strong_signal_mult : float
        If |z| > deadzone * this, skip timing and enter immediately.
    long_only : bool
        Only take long positions.
    """

    def __init__(
        self,
        deadzone: float = 0.3,
        min_hold_15m: int = 4,      # 1 hour
        max_hold_15m: int = 128,    # 32 hours
        trailing_stop_pct: float = 0.015,
        strong_signal_mult: float = 2.0,
        long_only: bool = False,
    ):
        self._deadzone = deadzone
        self._min_hold = min_hold_15m
        self._max_hold = max_hold_15m
        self._trailing_stop = trailing_stop_pct
        self._strong_mult = strong_signal_mult
        self._long_only = long_only

        self._timing = MicroTiming()
        self._position: int = 0      # +1 long, -1 short, 0 flat
        self._entry_bar: int = 0
        self._entry_price: float = 0.0
        self._peak_price: float = 0.0
        self._trough_price: float = float("inf")
        self._bar_count: int = 0

        # Pending signal from 1h model
        self._pending_direction: int = 0
        self._pending_z: float = 0.0
        self._wait_bars: int = 0      # bars waited for entry timing
        self._max_wait: int = 8       # max 15m bars to wait (2 hours)

    def on_15m_bar(
        self,
        bar: _Bar15m,
        z_1h: float,
    ) -> Optional[Dict[str, Any]]:
        """Process one 15m bar with 1h z-score signal.

        Parameters
        ----------
        bar : _Bar15m
            Current 15m OHLCV bar.
        z_1h : float
            Current 1h z-score (updated every 4 bars, held constant between).

        Returns
        -------
        dict or None
            Trade event if entry/exit triggered, else None.
            {"action": "entry"/"exit", "side": "long"/"short", "price": ..., "reason": ...}
        """
        self._bar_count += 1
        self._timing.update(bar.close)

        # ── Exit logic ──
        if self._position != 0:
            held = self._bar_count - self._entry_bar

            # Update trailing stop tracker
            if self._position > 0:
                self._peak_price = max(self._peak_price, bar.high)
            else:
                self._trough_price = min(self._trough_price, bar.low)

            # Check exits
            reason = self._check_exit(bar, z_1h, held)
            if reason:
                side = "short" if self._position > 0 else "long"
                result = {
                    "action": "exit",
                    "side": side,
                    "price": bar.close,
                    "reason": reason,
                    "held_bars": held,
                    "held_hours": held * 0.25,
                }
                self._position = 0
                self._pending_direction = 0
                self._wait_bars = 0
                return result

        # ── Entry logic ──
        if self._position == 0:
            # Determine direction from 1h signal
            direction = 0
            if z_1h > self._deadzone:
                direction = 1
            elif not self._long_only and z_1h < -self._deadzone:
                direction = -1

            if direction != 0:
                # Check if signal is strong enough for immediate entry
                is_strong = abs(z_1h) > self._deadzone * self._strong_mult

                if is_strong:
                    return self._enter(direction, bar.close, "strong_signal")

                # Set or update pending signal
                if direction != self._pending_direction:
                    self._pending_direction = direction
                    self._pending_z = z_1h
                    self._wait_bars = 0

                self._wait_bars += 1

                # Check timing conditions
                if self._timing.is_pullback_entry(direction):
                    return self._enter(direction, bar.close, "pullback_entry")

                if self._timing.is_momentum_entry(direction):
                    return self._enter(direction, bar.close, "momentum_entry")

                # Timeout: enter anyway if waited too long
                if self._wait_bars >= self._max_wait:
                    return self._enter(direction, bar.close, "timeout_entry")

            else:
                # No signal → clear pending
                self._pending_direction = 0
                self._wait_bars = 0

        return None

    def _enter(self, direction: int, price: float, reason: str) -> Dict[str, Any]:
        self._position = direction
        self._entry_bar = self._bar_count
        self._entry_price = price
        self._peak_price = price
        self._trough_price = price
        self._pending_direction = 0
        self._wait_bars = 0
        return {
            "action": "entry",
            "side": "long" if direction > 0 else "short",
            "price": price,
            "reason": reason,
        }

    def _check_exit(self, bar: _Bar15m, z_1h: float, held: int) -> Optional[str]:
        """Check all exit conditions. Returns reason string or None."""
        # Max hold
        if held >= self._max_hold:
            return "max_hold"

        if held < self._min_hold:
            return None

        # Trailing stop (15m level — tighter)
        if self._trailing_stop > 0:
            if self._position > 0:
                drawdown = (self._peak_price - bar.close) / self._peak_price
                if drawdown > self._trailing_stop:
                    return f"trailing_stop_{drawdown:.3f}"
            else:
                rally = (bar.close - self._trough_price) / self._trough_price
                if rally > self._trailing_stop:
                    return f"trailing_stop_{rally:.3f}"

        # 1h signal reversal
        if self._position * z_1h < -0.3:
            return "signal_reversal"

        # Deadzone fade (signal too weak)
        if abs(z_1h) < 0.15:
            return "deadzone_fade"

        # 15m micro-exit (RSI extremes)
        if self._timing.should_exit_micro(self._position):
            return "micro_exit"

        return None

    @property
    def position(self) -> int:
        return self._position

    def reset(self) -> None:
        self._position = 0
        self._pending_direction = 0
        self._wait_bars = 0
        self._bar_count = 0
