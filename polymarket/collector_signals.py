"""Signal computation helpers for Polymarket collector.

Extracted from collector.py: binary option fair value, volatility tracking, RSI tracking.
"""
from __future__ import annotations

import math
import statistics


def binary_call_fair_value(S: float, K: float, T_minutes: float, sigma_annual: float) -> float:
    """Fair value of a binary call option (digital call).

    Uses the Black-Scholes formula to compute the risk-neutral probability
    that the underlying finishes at or above the strike.

    Args:
        S: current price
        K: strike (window open price)
        T_minutes: time remaining in minutes
        sigma_annual: annualized volatility

    Returns:
        Probability that S_T >= K (0 to 1).
    """
    if T_minutes <= 0:
        return 1.0 if S >= K else 0.0
    if K <= 0 or S <= 0:
        return 0.5
    T = T_minutes / (365 * 24 * 60)
    d2 = (math.log(S / K) + (-0.5 * sigma_annual**2) * T) / (sigma_annual * math.sqrt(T))
    # norm.cdf(x) = 0.5 * (1 + erf(x / sqrt(2)))
    return 0.5 * (1.0 + math.erf(d2 / math.sqrt(2)))


class VolatilityTracker:
    """Track rolling 1-hour realized volatility from Binance 1-minute returns."""

    def __init__(self, window: int = 60):
        self._returns: list[float] = []
        self._window = window
        self._prev_price: float | None = None

    def update(self, price: float) -> None:
        """Update with a new price observation."""
        if self._prev_price is not None and self._prev_price > 0:
            ret = math.log(price / self._prev_price)
            self._returns.append(ret)
            if len(self._returns) > self._window:
                self._returns.pop(0)
        self._prev_price = price

    @property
    def sigma_annual(self) -> float:
        """Annualized volatility estimate.

        Falls back to 50% if fewer than 10 observations.
        """
        if len(self._returns) < 10:
            return 0.50  # default 50% annual vol
        std_1m = statistics.stdev(self._returns)
        return std_1m * math.sqrt(365 * 24 * 60)


class RSITracker:
    """Track rolling RSI(5) on 5-minute BTC closes for signal annotation."""

    def __init__(self, period: int = 5):
        self._period = period
        self._closes: list[float] = []
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self._rsi: float = 50.0

    def update(self, close: float) -> float:
        """Feed a 5-minute close price. Returns current RSI."""
        self._closes.append(close)
        n = len(self._closes)
        if n < 2:
            return 50.0

        change = self._closes[-1] - self._closes[-2]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        if n <= self._period + 1:
            # Initial SMA phase
            if n == self._period + 1:
                gains = [max(self._closes[i] - self._closes[i - 1], 0)
                         for i in range(1, n)]
                losses = [max(self._closes[i - 1] - self._closes[i], 0)
                          for i in range(1, n)]
                self._avg_gain = sum(gains[:self._period]) / self._period
                self._avg_loss = sum(losses[:self._period]) / self._period
            else:
                return 50.0
        else:
            self._avg_gain = (self._avg_gain * (self._period - 1) + gain) / self._period
            self._avg_loss = (self._avg_loss * (self._period - 1) + loss) / self._period

        if self._avg_loss < 1e-10:
            self._rsi = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            self._rsi = 100.0 - 100.0 / (1.0 + rs)
        return self._rsi

    @property
    def value(self) -> float:
        return self._rsi

    @property
    def signal(self) -> str:
        """Return 'up', 'down', or 'neutral' based on RSI thresholds."""
        if self._rsi < 25:
            return "up"
        elif self._rsi > 75:
            return "down"
        return "neutral"

    @property
    def bar_count(self) -> int:
        return len(self._closes)
