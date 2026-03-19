"""Exponential-moving-average volatility estimator from trade prices."""

from __future__ import annotations

import math


class VolEstimator:
    """EMA volatility from trade log-returns.

    Uses a simple EMA of squared log-returns, then sqrt to get vol.
    Window is controlled by *alpha* (default ~1/200 trades).
    """

    def __init__(self, alpha: float = 0.01, min_trades: int = 20) -> None:
        self._alpha = alpha
        self._min_trades = min_trades
        self._last_price: float | None = None
        self._ema_var: float = 0.0
        self._count: int = 0

    @property
    def ready(self) -> bool:
        return self._count >= self._min_trades

    @property
    def volatility(self) -> float:
        """Current annualised vol estimate (0.0 if not ready)."""
        if not self.ready:
            return 0.0
        return math.sqrt(max(self._ema_var, 0.0))

    def on_trade(self, price: float) -> float:
        """Update with a new trade price, return current vol."""
        if price <= 0:
            return self.volatility
        if self._last_price is not None and self._last_price > 0:
            log_ret = math.log(price / self._last_price)
            sq = log_ret * log_ret
            self._ema_var = (1 - self._alpha) * self._ema_var + self._alpha * sq
            self._count += 1
        self._last_price = price
        return self.volatility

    def reset(self) -> None:
        self._last_price = None
        self._ema_var = 0.0
        self._count = 0
