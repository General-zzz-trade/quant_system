"""Cross-asset correlation computer for portfolio risk monitoring.

Maintains a rolling window of log returns per symbol and computes
average pairwise correlation across the portfolio.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class CorrelationComputer:
    """Rolling correlation tracker across symbols.

    Call ``update(symbol, close)`` each bar per symbol.
    Query ``portfolio_avg_correlation(symbols)`` for current risk metric.
    """

    window: int = 60
    _returns: Dict[str, List[float]] = field(default_factory=dict, init=False)
    _last_prices: Dict[str, float] = field(default_factory=dict, init=False)

    def checkpoint(self) -> dict:
        """Serialize state for persistence."""
        return {
            "returns": {k: list(v) for k, v in self._returns.items()},
            "last_prices": dict(self._last_prices),
        }

    def restore(self, data: dict) -> None:
        """Restore state from checkpoint."""
        self._returns = {k: list(v) for k, v in data.get("returns", {}).items()}
        self._last_prices = {k: float(v) for k, v in data.get("last_prices", {}).items()}

    def update(self, symbol: str, close: float) -> None:
        """Record a new close price, compute log return."""
        prev = self._last_prices.get(symbol)
        self._last_prices[symbol] = close
        if prev is None or prev <= 0 or close <= 0:
            return
        ret = math.log(close / prev)
        buf = self._returns.setdefault(symbol, [])
        buf.append(ret)
        if len(buf) > self.window:
            buf[:] = buf[-self.window:]

    def portfolio_avg_correlation(self, symbols: Sequence[str]) -> Optional[float]:
        """Average pairwise correlation among held symbols.

        Returns None if fewer than 2 symbols have sufficient data.
        """
        active = [s for s in symbols if len(self._returns.get(s, [])) >= 2]
        if len(active) < 2:
            return None

        total = 0.0
        count = 0
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                corr = self._pearson(active[i], active[j])
                if corr is not None:
                    total += corr
                    count += 1

        return total / count if count > 0 else None

    def position_correlation(self, new_sym: str, existing: Sequence[str]) -> Optional[float]:
        """Average correlation of new_sym against each existing symbol."""
        if not existing or len(self._returns.get(new_sym, [])) < 2:
            return None

        total = 0.0
        count = 0
        for sym in existing:
            corr = self._pearson(new_sym, sym)
            if corr is not None:
                total += corr
                count += 1

        return total / count if count > 0 else None

    def _pearson(self, a: str, b: str) -> Optional[float]:
        """Pearson correlation between two return series."""
        ra = self._returns.get(a, [])
        rb = self._returns.get(b, [])
        n = min(len(ra), len(rb))
        if n < 2:
            return None
        xa = ra[-n:]
        xb = rb[-n:]
        return _pearson_corr(xa, xb)


def _pearson_corr(x: List[float], y: List[float]) -> Optional[float]:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    cov = 0.0
    vx = 0.0
    vy = 0.0
    for i in range(n):
        dx = x[i] - mx
        dy = y[i] - my
        cov += dx * dy
        vx += dx * dx
        vy += dy * dy
    denom = math.sqrt(vx * vy)
    if denom < 1e-15:
        return None
    return cov / denom


# --- Rust acceleration ---
try:
    from _quant_hotpath import RustCorrelationComputer  # noqa: F401
    _RUST_CORRELATION_AVAILABLE = True
except ImportError:
    _RUST_CORRELATION_AVAILABLE = False
