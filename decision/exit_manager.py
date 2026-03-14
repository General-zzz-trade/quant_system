"""Exit Manager — trailing stop, z-score cap, time filter, signal-based exits.

All exit/entry gating logic in one place, parameterized by ExitConfig.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class _TrailingState:
    """Per-position trailing stop state."""
    entry_price: float = 0.0
    peak_price: float = 0.0
    entry_bar: int = 0
    direction: float = 0.0   # +1 long, -1 short


class ExitManager:
    """Manages exit decisions and entry gating.

    Parameters
    ----------
    config : ExitConfig
        Exit strategy configuration (from V11Config.exit).
    min_hold : int
        Minimum bars before any exit (from V11Config.min_hold).
    max_hold : int
        Maximum bars before forced exit (from V11Config.max_hold).
    """

    def __init__(self, config, min_hold: int = 12, max_hold: int = 96):
        from alpha.v11_config import ExitConfig
        self._config: ExitConfig = config
        self._min_hold = min_hold
        self._max_hold = max_hold
        self._positions: Dict[str, _TrailingState] = {}

    def on_entry(self, symbol: str, price: float, bar: int, direction: float) -> None:
        """Record a new position entry."""
        self._positions[symbol] = _TrailingState(
            entry_price=price,
            peak_price=price,
            entry_bar=bar,
            direction=direction,
        )

    def on_exit(self, symbol: str) -> None:
        """Clear position tracking on exit."""
        self._positions.pop(symbol, None)

    def update_price(self, symbol: str, price: float) -> None:
        """Update peak price for trailing stop tracking."""
        state = self._positions.get(symbol)
        if state is None:
            return
        if state.direction > 0:
            state.peak_price = max(state.peak_price, price)
        else:
            state.peak_price = min(state.peak_price, price)

    def check_exit(
        self,
        symbol: str,
        price: float,
        bar: int,
        z_score: float,
        position: float,
    ) -> Tuple[bool, str]:
        """Check if position should be exited.

        Returns (should_exit, reason).
        """
        state = self._positions.get(symbol)
        if state is None:
            return False, ""

        held = bar - state.entry_bar

        # 1. Max hold — always enforced
        if held >= self._max_hold:
            return True, f"max_hold={held}"

        # Must respect min_hold for everything except max_hold
        if held < self._min_hold:
            return False, ""

        # 2. Trailing stop (if enabled)
        cfg = self._config
        if cfg.trailing_stop_pct > 0:
            if state.direction > 0:
                # Long: exit if price dropped from peak
                drawdown = (state.peak_price - price) / state.peak_price
            else:
                # Short: exit if price rose from trough
                drawdown = (price - state.peak_price) / state.peak_price
            if drawdown >= cfg.trailing_stop_pct:
                return True, f"trailing_stop={drawdown:.4f}"

        # 3. Signal reversal
        if position * z_score < cfg.reversal_threshold:
            return True, f"reversal_z={z_score:.2f}"

        # 4. Deadzone fade (signal too weak to hold)
        if abs(z_score) < cfg.deadzone_fade:
            return True, f"deadzone_fade_z={z_score:.2f}"

        return False, ""

    def checkpoint(self) -> dict:
        """Serialize position tracking state for persistence."""
        return {
            sym: {
                "entry_price": s.entry_price,
                "peak_price": s.peak_price,
                "entry_bar": s.entry_bar,
                "direction": s.direction,
            }
            for sym, s in self._positions.items()
        }

    def restore(self, data: dict) -> None:
        """Restore position tracking state from checkpoint."""
        self._positions = {
            sym: _TrailingState(**vals) for sym, vals in data.items()
        }

    def allow_entry(self, z_score: float, hour_utc: Optional[int] = None) -> bool:
        """Check if entry is allowed (z-cap + time filter).

        Parameters
        ----------
        z_score : float
            Current z-score of prediction.
        hour_utc : int, optional
            Current hour in UTC (0-23).
        """
        cfg = self._config

        # Z-score cap: don't enter on extreme z-scores (likely noise)
        if cfg.zscore_cap > 0 and abs(z_score) > cfg.zscore_cap:
            return False

        # Time filter
        tf = cfg.time_filter
        if tf is not None and tf.enabled and hour_utc is not None:
            if hour_utc in tf.skip_hours_utc:
                return False

        return True
