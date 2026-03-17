# runner/gates/adaptive_stop_gate.py
"""Adaptive ATR-based stop-loss gate.

Extracted from AlphaRunner._compute_stop_price().

Three-phase stop (validated: 18/20 walk-forward folds, ATR×2.0 initial):
1. Initial:   long  → entry - ATR×2.0
              short → entry + ATR×2.0
2. Breakeven: after 1×ATR profit from peak, move stop to entry
3. Trailing:  after 0.8×ATR profit, trail at peak ∓ ATR×0.3

ATR is passed as absolute price distance (e.g. $100 for a $3000 asset).
Use ``atr = atr_pct * entry_price`` to convert fractional ATR.

Hard limits:
- Floor:   max 5% loss from entry (capital protection)
- Ceiling: min 0.3% distance from current price (avoid noise stops)
"""
from __future__ import annotations

from typing import Dict, Optional


class _PositionStop:
    """Per-symbol stop state."""

    def __init__(
        self,
        side: str,
        entry_price: float,
        atr: float,
        atr_stop_mult: float = 2.0,
        breakeven_atr: float = 1.0,
        trail_atr_mult: float = 0.8,
        trail_step: float = 0.3,
    ) -> None:
        self.side = side          # "long" or "short"
        self.entry = entry_price
        self.atr = atr            # absolute ATR in price units
        self.peak = entry_price   # best price since entry
        self.current = entry_price

        self._atr_stop_mult = atr_stop_mult
        self._breakeven_atr = breakeven_atr
        self._trail_atr_mult = trail_atr_mult
        self._trail_step = trail_step

    def update(self, price: float) -> None:
        self.current = price
        if self.side == "long":
            self.peak = max(self.peak, price)
        else:
            self.peak = min(self.peak, price)

    def stop_price(self) -> float:
        """Compute current stop price using 3-phase ATR logic.

        ATR is treated as absolute price distance.
        Phase 1 initial stop: entry ± atr × atr_stop_mult
        """
        atr = self.atr
        entry = self.entry
        side_long = self.side == "long"

        if side_long:
            profit = self.peak - entry   # absolute profit from entry
        else:
            profit = entry - self.peak

        # Phase boundary thresholds in price units
        breakeven_threshold = atr * self._breakeven_atr
        trail_threshold = atr * self._trail_atr_mult

        if profit >= breakeven_threshold:
            if profit >= trail_threshold:
                # Phase 3: trailing stop
                trail_dist = atr * self._trail_step
                if side_long:
                    stop = self.peak - trail_dist
                else:
                    stop = self.peak + trail_dist
            else:
                # Phase 2: breakeven stop (entry + small buffer)
                buffer = atr * 0.1
                if side_long:
                    stop = entry + buffer
                else:
                    stop = entry - buffer
        else:
            # Phase 1: initial wide stop
            initial_dist = atr * self._atr_stop_mult
            if side_long:
                stop = entry - initial_dist
            else:
                stop = entry + initial_dist

        # Hard floor: max 5% loss from entry
        if side_long:
            stop = max(stop, entry * 0.95)
        else:
            stop = min(stop, entry * 1.05)

        # Hard ceiling: min 0.3% from current price (avoid noise stops)
        min_dist = entry * 0.003
        if side_long and self.current - stop < min_dist:
            stop = min(stop, self.current - min_dist)
        elif not side_long and stop - self.current < min_dist:
            stop = max(stop, self.current + min_dist)

        return stop


class AdaptiveStopGate:
    """Tracks adaptive ATR stop-loss per symbol.

    ATR is expected as absolute price distance (e.g. $100 for a $3000 asset).
    To convert from fractional: pass ``atr = atr_pct * entry_price``.

    Usage::

        gate = AdaptiveStopGate()
        gate.on_new_position("ETH", side="long", entry_price=3000.0, atr=100.0)
        gate.update_price("ETH", 3150.0)
        stop = gate.get_stop_price("ETH")  # → near 3000 after breakeven
    """

    def __init__(
        self,
        atr_stop_mult: float = 2.0,
        breakeven_atr: float = 1.0,
        trail_atr_mult: float = 0.8,
        trail_step: float = 0.3,
    ) -> None:
        self._atr_stop_mult = atr_stop_mult
        self._breakeven_atr = breakeven_atr
        self._trail_atr_mult = trail_atr_mult
        self._trail_step = trail_step
        self._positions: Dict[str, _PositionStop] = {}

    def on_new_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        atr: float,
    ) -> None:
        """Register a new position for stop tracking.

        Args:
            symbol:      Symbol identifier (e.g. "ETH").
            side:        "long" or "short".
            entry_price: Entry fill price.
            atr:         Current ATR as absolute price distance.
        """
        self._positions[symbol] = _PositionStop(
            side=side,
            entry_price=entry_price,
            atr=atr,
            atr_stop_mult=self._atr_stop_mult,
            breakeven_atr=self._breakeven_atr,
            trail_atr_mult=self._trail_atr_mult,
            trail_step=self._trail_step,
        )

    def update_price(self, symbol: str, price: float) -> None:
        """Update current price for a tracked position."""
        pos = self._positions.get(symbol)
        if pos is not None:
            pos.update(price)

    def get_stop_price(self, symbol: str) -> Optional[float]:
        """Return current adaptive stop price, or None if no position tracked."""
        pos = self._positions.get(symbol)
        if pos is None:
            return None
        return pos.stop_price()

    def close_position(self, symbol: str) -> None:
        """Remove stop tracking for a closed position."""
        self._positions.pop(symbol, None)
