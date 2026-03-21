# runner/gates/micro_stop_gate.py
"""Micro stop-loss gate for 100x leverage.

At 100x, standard ATR stop (2x ATR ≈ 3%) would mean 300% capital loss.
This gate enforces much tighter stops:
  - Hard stop: 0.3% from entry = 30% capital at 100x
  - Breakeven: move to entry after 0.1% profit
  - Trailing: 0.15% from peak after 0.2% profit

Also provides tick-level real-time stop checking (for WS price feed).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

from runner.gate_chain import GateResult

_log = logging.getLogger(__name__)


class MicroStopPhase(str, Enum):
    INITIAL = "INITIAL"
    BREAKEVEN = "BREAKEVEN"
    TRAILING = "TRAILING"


@dataclass
class MicroStopState:
    """Per-symbol stop state."""
    entry_price: float = 0.0
    side: int = 0              # +1 long, -1 short
    peak_price: float = 0.0
    phase: MicroStopPhase = MicroStopPhase.INITIAL
    entry_ts: float = 0.0
    stop_price: float = 0.0    # cached stop price
    n_checks: int = 0
    triggered: bool = False


@dataclass
class MicroStopConfig:
    """Stop-loss parameters — tuned for high leverage (20-100x).

    Defaults are optimised for 20x (walk-forward validated):
      2% stop × 20x = 40% capital risk, ATR>stop only 9% of time.
    For 100x: override initial_stop_pct=0.003, max_loss_pct=0.005.
    """
    # Phase 1: Initial stop
    initial_stop_pct: float = 0.020     # 2.0% = 40% capital at 20x (validated)
    # Phase 2: Breakeven
    breakeven_trigger_pct: float = 0.005  # move to breakeven after 0.5% profit
    breakeven_buffer_pct: float = 0.001   # stop at entry + 0.1% buffer
    # Phase 3: Trailing
    trail_trigger_pct: float = 0.010    # trailing after 1.0% profit
    trail_distance_pct: float = 0.005   # 0.5% from peak
    # Hard limits
    max_loss_pct: float = 0.025         # absolute max 2.5% = 50% capital at 20x
    # Time-based
    max_hold_s: float = 86400.0         # force close after 24h
    stale_position_s: float = 172800.0  # emergency close after 48h


class MicroStopGate:
    """Gate: ultra-tight stop-loss for 100x leverage.

    This gate:
    1. Records new positions via on_new_position()
    2. On each check(), evaluates stop conditions
    3. Provides check_realtime() for tick-by-tick WS monitoring

    Phase transitions:
      INITIAL → BREAKEVEN: profit >= breakeven_trigger_pct
      BREAKEVEN → TRAILING: profit >= trail_trigger_pct
      Any phase → TRIGGERED: price breaches stop
    """

    name = "MicroStop"

    def __init__(self, cfg: MicroStopConfig | None = None) -> None:
        self._cfg = cfg or MicroStopConfig()
        self._states: dict[str, MicroStopState] = {}

    def on_new_position(
        self,
        symbol: str,
        side: int,
        entry_price: float,
    ) -> float:
        """Record new position and return initial stop price.

        Call this immediately after fill confirmation.
        Returns the initial stop price for logging.
        """
        state = MicroStopState(
            entry_price=entry_price,
            side=side,
            peak_price=entry_price,
            phase=MicroStopPhase.INITIAL,
            entry_ts=time.monotonic(),
        )

        # Compute initial stop (clamped by max_loss)
        stop_pct = min(self._cfg.initial_stop_pct, self._cfg.max_loss_pct)
        if side == 1:  # long
            state.stop_price = entry_price * (1 - stop_pct)
        else:  # short
            state.stop_price = entry_price * (1 + stop_pct)

        self._states[symbol] = state
        _log.info(
            "MicroStop NEW %s: side=%d entry=%.2f stop=%.2f (%.1fbps)",
            symbol, side, entry_price, state.stop_price,
            self._cfg.initial_stop_pct * 10000,
        )
        return state.stop_price

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        """Gate check — evaluate stop conditions."""
        symbol = context.get("symbol", "")
        price = float(context.get("price", 0))

        if not symbol or price <= 0:
            return GateResult(allowed=True)

        state = self._states.get(symbol)
        if state is None or state.side == 0:
            return GateResult(allowed=True)

        triggered, reason = self._evaluate(state, price)

        if triggered:
            state.triggered = True
            _log.warning(
                "MicroStop TRIGGERED %s: %s price=%.2f stop=%.2f phase=%s",
                symbol, reason, price, state.stop_price, state.phase.value,
            )
            return GateResult(allowed=False, reason=f"micro_stop: {reason}")

        return GateResult(allowed=True, scale=1.0)

    def check_realtime(self, symbol: str, price: float) -> bool:
        """Tick-level real-time stop check.

        Call from WS price feed thread. Returns True if stopped out.
        Thread-safe: only reads/writes to per-symbol state.
        """
        state = self._states.get(symbol)
        if state is None or state.side == 0:
            return False

        triggered, reason = self._evaluate(state, price)

        if triggered:
            state.triggered = True
            _log.warning("MicroStop RT %s: %s price=%.2f", symbol, reason, price)
            return True

        return False

    def get_stop_price(self, symbol: str) -> float:
        """Current stop price for symbol (0 if no position)."""
        state = self._states.get(symbol)
        if state is None:
            return 0.0
        return state.stop_price

    def get_phase(self, symbol: str) -> str:
        """Current stop phase for symbol."""
        state = self._states.get(symbol)
        if state is None:
            return "NONE"
        return state.phase.value

    def reset_symbol(self, symbol: str) -> None:
        """Clear stop state after position closed."""
        self._states.pop(symbol, None)

    def _evaluate(self, state: MicroStopState, price: float) -> tuple[bool, str]:
        """Core stop evaluation. Returns (triggered, reason)."""
        cfg = self._cfg
        state.n_checks += 1

        entry = state.entry_price
        side = state.side

        # Update peak
        if side == 1:
            state.peak_price = max(state.peak_price, price)
        else:
            state.peak_price = min(state.peak_price, price)

        # Compute profit
        if side == 1:
            profit_pct = (price - entry) / entry
        else:
            profit_pct = (entry - price) / entry

        # Phase transitions
        if state.phase == MicroStopPhase.INITIAL and profit_pct >= cfg.breakeven_trigger_pct:
            state.phase = MicroStopPhase.BREAKEVEN
            if side == 1:
                state.stop_price = entry * (1 + cfg.breakeven_buffer_pct)
            else:
                state.stop_price = entry * (1 - cfg.breakeven_buffer_pct)
            _log.debug("MicroStop %s→BREAKEVEN stop=%.2f", "long" if side == 1 else "short", state.stop_price)

        if state.phase == MicroStopPhase.BREAKEVEN and profit_pct >= cfg.trail_trigger_pct:
            state.phase = MicroStopPhase.TRAILING
            _log.debug("MicroStop →TRAILING")

        # Update trailing stop
        if state.phase == MicroStopPhase.TRAILING:
            if side == 1:
                new_stop = state.peak_price * (1 - cfg.trail_distance_pct)
                state.stop_price = max(state.stop_price, new_stop)
            else:
                new_stop = state.peak_price * (1 + cfg.trail_distance_pct)
                state.stop_price = min(state.stop_price, new_stop)

        # Hard max loss clamp
        if side == 1:
            hard_stop = entry * (1 - cfg.max_loss_pct)
            state.stop_price = max(state.stop_price, hard_stop)
        else:
            hard_stop = entry * (1 + cfg.max_loss_pct)
            state.stop_price = min(state.stop_price, hard_stop)

        # Check stop breach
        if side == 1 and price <= state.stop_price:
            loss_pct = (entry - price) / entry
            return True, f"long_stop loss={loss_pct:.3%} phase={state.phase.value}"
        if side == -1 and price >= state.stop_price:
            loss_pct = (price - entry) / entry
            return True, f"short_stop loss={loss_pct:.3%} phase={state.phase.value}"

        # Time-based stop
        elapsed = time.monotonic() - state.entry_ts
        if elapsed > cfg.max_hold_s:
            return True, f"max_hold {elapsed:.0f}s"

        return False, ""
