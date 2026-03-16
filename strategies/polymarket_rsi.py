"""Polymarket RSI strategy -- wraps RSI5mStrategy in StrategyProtocol."""
from __future__ import annotations

from typing import Any, Dict

from polymarket.strategies.rsi_5m import RSI5mStrategy
from strategies.base import Signal


class PolymarketRSIStrategy:
    """StrategyProtocol wrapper around the RSI(5) mean-reversion strategy.

    Feeds 1-minute BTC/USDT closes to RSI5mStrategy and converts
    the resulting RSI5mSignal into a universal Signal.

    Parameters
    ----------
    period : int
        RSI lookback period (default 5).
    oversold : float
        RSI level below which we predict "up" (default 25.0).
    overbought : float
        RSI level above which we predict "down" (default 75.0).
    min_bars : int
        Minimum bars before generating signals (default 10).
    """

    name: str = "polymarket_rsi"
    version: str = "1.0"
    venue: str = "polymarket"
    timeframe: str = "5m"

    def __init__(
        self,
        period: int = 5,
        oversold: float = 25.0,
        overbought: float = 75.0,
        min_bars: int = 10,
    ) -> None:
        self._period = period
        self._oversold = oversold
        self._overbought = overbought
        self._min_bars = min_bars
        self._rsi = RSI5mStrategy(
            period=period,
            oversold=oversold,
            overbought=overbought,
            min_bars=min_bars,
        )

    def generate_signal(self, features: Dict[str, Any]) -> Signal:
        """Generate signal from features.

        Expected features:
            - "close": latest 1-minute BTC/USDT close price
            - "timestamp_ms": optional timestamp in milliseconds
        """
        close = features.get("close")
        if close is None:
            return Signal(direction=0, confidence=0.0,
                          meta={"reason": "no_close_price"})

        timestamp_ms = features.get("timestamp_ms", 0)
        result = self._rsi.update(float(close), int(timestamp_ms))

        if result is None:
            return Signal(
                direction=0,
                confidence=0.0,
                meta={"rsi": self._rsi.current_rsi, "bars": self._rsi.bar_count},
            )

        direction = 1 if result.direction == "up" else -1
        return Signal(
            direction=direction,
            confidence=result.confidence,
            meta={
                "rsi": result.rsi_value,
                "rsi_direction": result.direction,
                "timestamp_ms": result.timestamp_ms,
            },
        )

    def validate_config(self) -> bool:
        return (
            self._period > 0
            and 0.0 < self._oversold < self._overbought < 100.0
            and self._min_bars >= self._period
        )

    def describe(self) -> str:
        return (
            f"Polymarket RSI({self._period}) mean-reversion: "
            f"oversold<{self._oversold} -> Up, overbought>{self._overbought} -> Down. "
            f"Accuracy ~55%, 23/23 folds PASS."
        )

    def reset(self) -> None:
        """Reset internal RSI state."""
        self._rsi.reset()
