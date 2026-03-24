"""Cross-asset correlation computer — Rust-delegated via RustCorrelationComputer.

Maintains a rolling window of log returns per symbol and computes
average pairwise correlation across the portfolio.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from _quant_hotpath import RustCorrelationComputer


class CorrelationComputer:
    """Rolling correlation tracker — delegates to RustCorrelationComputer.

    Call ``update(symbol, close)`` each bar per symbol.
    Query ``portfolio_avg_correlation(symbols)`` for current risk metric.
    """

    def __init__(self, window: int = 60) -> None:
        self.window = window
        self._inner = RustCorrelationComputer(window=window)
        # Keep _returns accessor for correlation_gate.py data-sufficiency check
        self._returns: Dict[str, List[float]] = {}
        self._last_prices: Dict[str, float] = {}

    def checkpoint(self) -> dict:
        """Serialize state for persistence."""
        return self._inner.checkpoint()

    def restore(self, data: dict) -> None:
        """Restore state from checkpoint."""
        self._inner.restore(data)
        # Sync Python-side _returns and _last_prices from checkpoint
        self._returns = {k: list(v) for k, v in data.get("returns", {}).items()}
        self._last_prices = {k: float(v) for k, v in data.get("last_prices", {}).items()}

    def update(self, symbol: str, close: float) -> None:
        """Record a new close price, compute log return."""
        self._inner.update(symbol, close)
        # Maintain _returns for correlation_gate data-sufficiency check
        prev = self._last_prices.get(symbol)
        self._last_prices[symbol] = close
        if prev is not None and prev > 0 and close > 0:
            ret = math.log(close / prev)
            buf = self._returns.setdefault(symbol, [])
            buf.append(ret)
            if len(buf) > self.window:
                buf[:] = buf[-self.window:]

    def portfolio_avg_correlation(self, symbols: Sequence[str]) -> Optional[float]:
        """Average pairwise correlation among held symbols."""
        result = self._inner.avg_correlation(list(symbols))
        return result if result is not None else None

    def position_correlation(self, new_sym: str, existing: Sequence[str]) -> Optional[float]:
        """Average correlation of new_sym against each existing symbol."""
        result = self._inner.position_correlation(new_sym, list(existing))
        return result if result is not None else None
