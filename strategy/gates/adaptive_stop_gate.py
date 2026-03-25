# runner/gates/adaptive_stop_gate.py
"""AdaptiveStopGate — ATR 3-phase stop-loss gate for the GateChain.

Three-phase logic (ported from AlphaRunner._compute_stop_price):
  INITIAL   : stop = entry ± atr × atr_initial_mult   (default 2.0×)
  BREAKEVEN : stop = entry ± atr × 0.1 buffer         (after ≥ atr × breakeven_trigger profit)
  TRAILING  : stop = peak ∓ atr × atr_trailing_mult   (after ≥ atr × trail_trigger profit)

Hard limits applied on every check:
  • max 5% loss from entry  (floor/ceiling on stop)
  • min 0.3% distance from current price  (avoids noise stops)

State is maintained per symbol so multiple symbols can share one gate instance.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, Optional

from strategy.gates.types import GateResult

try:
    from _quant_hotpath import RustAdaptiveStopGate as _RustAdaptiveStopGate
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

logger = logging.getLogger(__name__)

# ── Phase enum ──────────────────────────────────────────────────────────────

class StopPhase(str, Enum):
    INITIAL = "INITIAL"
    BREAKEVEN = "BREAKEVEN"
    TRAILING = "TRAILING"


# ── Per-symbol stop state ────────────────────────────────────────────────────

@dataclass
class _SymbolState:
    entry_price: float = 0.0
    side: int = 0              # +1 long / -1 short / 0 flat
    peak_price: float = 0.0
    stop_phase: StopPhase = StopPhase.INITIAL
    atr_buffer: Deque[float] = field(default_factory=lambda: deque(maxlen=50))

    def reset(self) -> None:
        self.entry_price = 0.0
        self.side = 0
        self.peak_price = 0.0
        self.stop_phase = StopPhase.INITIAL

    def current_atr(self, fallback: float = 0.015) -> float:
        """Mean of last 14 TR values, or fallback if < 5 samples."""
        if len(self.atr_buffer) < 5:
            return fallback
        window = list(self.atr_buffer)[-14:]
        return sum(window) / len(window)

    def push_true_range(self, high: float, low: float, prev_close: float) -> None:
        """Compute and buffer true range as fraction of close."""
        if prev_close <= 0 or not math.isfinite(high) or not math.isfinite(low):
            return
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        atr_pct = tr / prev_close
        self.atr_buffer.append(atr_pct)


# ── Rust forwarding proxy ───────────────────────────────────────────────────

class _RustForwardingState(_SymbolState):
    """Thin proxy that forwards state mutations to Rust when Rust is active.

    Used only for the _get_state() path so that tests seeding ATR and
    manipulating state via gate._get_state(sym).x = ... also update Rust.
    """

    def __init__(self, delegate: _SymbolState, rust_gate: Any, symbol: str) -> None:
        object.__setattr__(self, "_delegate", delegate)
        object.__setattr__(self, "_rust_gate", rust_gate)
        object.__setattr__(self, "_symbol", symbol)

    def __getattr__(self, name: str) -> Any:  # type: ignore[override]
        return getattr(object.__getattribute__(self, "_delegate"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        delegate = object.__getattribute__(self, "_delegate")
        setattr(delegate, name, value)
        # Sync specific fields to Rust
        rust_gate = object.__getattribute__(self, "_rust_gate")
        symbol = object.__getattribute__(self, "_symbol")
        try:
            if name == "stop_phase":
                if hasattr(value, "value"):
                    rust_gate.set_phase(symbol, value.value)
                else:
                    rust_gate.set_phase(symbol, str(value))
            elif name == "peak_price":
                rust_gate.set_peak(symbol, float(value))
        except Exception:
            pass

    def push_true_range(self, high: float, low: float, prev_close: float) -> None:
        """Forward to both Python state and Rust buffer."""
        delegate = object.__getattribute__(self, "_delegate")
        delegate.push_true_range(high, low, prev_close)
        rust_gate = object.__getattribute__(self, "_rust_gate")
        symbol = object.__getattribute__(self, "_symbol")
        try:
            rust_gate.push_true_range(symbol, high, low, prev_close)
        except Exception:
            pass


# ── Gate ────────────────────────────────────────────────────────────────────

class AdaptiveStopGate:
    """Gate: ATR-based 3-phase adaptive stop-loss.

    On each ORDER event the gate:
      1. Ingests bar OHLC from ``context`` to maintain the ATR buffer and
         update the peak price.
      2. Computes the stop price from the current phase.
      3. Rejects the order (or any new order when a position is stopped out)
         if the current price breaches the stop.

    Context keys consumed (all optional; gate degrades gracefully):
      ``price``      — current market price (float)
      ``side``       — order side: "buy" / "sell"  (str)
      ``bar_high``   — bar high  (float)
      ``bar_low``    — bar low   (float)
      ``prev_close`` — previous bar close (float)

    Public helpers:
      ``on_new_position(symbol, side, entry_price)`` — called when a fill confirms
         a new position; resets state and records entry.
      ``check_stop(symbol, price)`` — real-time tick check; returns True if stopped.
      ``compute_stop_price(symbol, current_price)`` — returns the current stop
         price without triggering state changes.
    """

    name = "AdaptiveStop"

    # Phase transition thresholds (expressed as multiples of ATR)
    _BREAKEVEN_TRIGGER: float = 1.0   # profit ≥ 1×ATR → move to breakeven
    _TRAIL_TRIGGER: float = 0.8       # profit ≥ 0.8×ATR → move to trailing
    _BREAKEVEN_BUFFER: float = 0.1    # breakeven stop = entry ± 0.1×ATR

    # Hard limits (fractions)
    _MAX_LOSS_PCT: float = 0.05       # 5%
    _MIN_DIST_PCT: float = 0.003      # 0.3%

    def __init__(
        self,
        *,
        atr_initial_mult: float = 2.0,
        atr_breakeven_trigger: float = 1.0,
        atr_trailing_mult: float = 0.3,
        atr_fallback: float = 0.015,
    ) -> None:
        self._atr_initial_mult = atr_initial_mult
        self._breakeven_trigger = atr_breakeven_trigger
        self._trailing_mult = atr_trailing_mult
        self._atr_fallback = atr_fallback
        self._states: Dict[str, _SymbolState] = {}  # kept for Python fallback

        if _HAS_RUST:
            self._rust = _RustAdaptiveStopGate(
                atr_initial_mult=atr_initial_mult,
                atr_breakeven_trigger=atr_breakeven_trigger,
                atr_trailing_mult=atr_trailing_mult,
                atr_fallback=atr_fallback,
            )
            self._use_rust = True
        else:
            self._rust = None
            self._use_rust = False

    # ── Gate Protocol ───────────────────────────────────────────────────────

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        """Main gate entry-point called by GateChain."""
        sym: str = getattr(ev, "symbol", "") or context.get("symbol", "")
        if not sym:
            return GateResult(allowed=True)

        if self._use_rust:
            return self._check_rust(ev, sym, context)

        # ── Python fallback path ──────────────────────────────────────────
        state = self._get_state(sym)

        # ── Update ATR buffer from bar data ──────────────────────────────
        high: Optional[float] = context.get("bar_high")
        low: Optional[float] = context.get("bar_low")
        prev_close: Optional[float] = context.get("prev_close")
        if high is not None and low is not None and prev_close is not None:
            state.push_true_range(high, low, prev_close)

        # ── Current price ────────────────────────────────────────────────
        price: float = float(context.get("price") or 0.0)
        if price <= 0:
            price = float(getattr(ev, "price", 0.0) or 0.0)

        # ── No active position → check if this ORDER opens one ───────────
        if state.side == 0:
            # Allow the order; record entry once filled (caller must call
            # on_new_position after the fill).
            return GateResult(allowed=True)

        # ── Update peak price ────────────────────────────────────────────
        if price > 0:
            self._update_peak(state, price)

        # ── Check stop breach ────────────────────────────────────────────
        if price > 0 and self._is_stop_breached(state, price):
            logger.warning(
                "AdaptiveStop TRIGGERED for %s: price=%.4f stop=%.4f "
                "entry=%.4f phase=%s",
                sym, price, self.compute_stop_price(sym, price),
                state.entry_price, state.stop_phase.value,
            )
            state.reset()
            return GateResult(allowed=False, reason="stop_triggered")

        return GateResult(allowed=True)

    def _check_rust(self, ev: Any, sym: str, context: Dict[str, Any]) -> GateResult:
        """Rust-accelerated check path."""
        # Update ATR buffer from bar data (both Rust and Python for state inspection)
        high = context.get("bar_high")
        low = context.get("bar_low")
        prev_close = context.get("prev_close")
        if high is not None and low is not None and prev_close is not None:
            try:
                self._rust.push_true_range(sym, float(high), float(low), float(prev_close))
            except (ValueError, Exception) as e:
                logger.debug("AdaptiveStop: push_true_range rejected: %s", e)
            # Also update Python state so state inspection via _get_state() reflects ATR
            py_state = self._states.get(sym)
            if py_state is None:
                self._states[sym] = _SymbolState()
                py_state = self._states[sym]
            py_state.push_true_range(float(high), float(low), float(prev_close))

        # Current price
        price: float = float(context.get("price") or 0.0)
        if price <= 0:
            price = float(getattr(ev, "price", 0.0) or 0.0)

        # If no price, allow
        if price <= 0:
            return GateResult(allowed=True)

        # Rust check_stop returns False when side==0 (no active position)
        phase = self._rust.get_phase(sym)
        if self._rust.check_stop(sym, price):
            logger.warning(
                "AdaptiveStop TRIGGERED for %s: price=%.4f phase=%s",
                sym, price, phase,
            )
            return GateResult(allowed=False, reason="stop_triggered")

        return GateResult(allowed=True)

    # ── Public helpers ───────────────────────────────────────────────────────

    def on_new_position(
        self,
        symbol: str,
        side: int,
        entry_price: float,
    ) -> None:
        """Record a new position.  Call this after a fill is confirmed.

        Args:
            symbol: trading symbol (e.g. "ETHUSDT")
            side:   +1 for long, -1 for short
            entry_price: fill price
        """
        if self._use_rust:
            self._rust.on_new_position(symbol, side, entry_price)
        else:
            state = self._get_state(symbol)
            state.entry_price = entry_price
            state.side = side
            state.peak_price = entry_price
            state.stop_phase = StopPhase.INITIAL
        logger.debug(
            "AdaptiveStop NEW POSITION %s side=%d entry=%.4f",
            symbol, side, entry_price,
        )

    def check_stop(self, symbol: str, price: float) -> bool:
        """Real-time tick check.  Returns True (and resets state) if stopped out."""
        if not math.isfinite(price) or price <= 0:
            return False
        if self._use_rust:
            breached = self._rust.check_stop(symbol, price)
            if breached:
                logger.warning(
                    "AdaptiveStop TICK STOP %s: price=%.4f",
                    symbol, price,
                )
            return breached

        state = self._get_state(symbol)
        if state.side == 0 or state.entry_price <= 0:
            return False
        self._update_peak(state, price)
        if self._is_stop_breached(state, price):
            logger.warning(
                "AdaptiveStop TICK STOP %s: price=%.4f stop=%.4f phase=%s",
                symbol, price,
                self.compute_stop_price(symbol, price),
                state.stop_phase.value,
            )
            state.reset()
            return True
        return False

    def compute_stop_price(self, symbol: str, current_price: float) -> float:
        """Return the current stop price without mutating state.

        Safe to call repeatedly; does NOT update peak or phase.
        """
        if self._use_rust:
            return self._rust.compute_stop_price(symbol, current_price)

        state = self._get_state(symbol)
        if state.side == 0 or state.entry_price <= 0:
            return 0.0
        return self._compute_stop(state, current_price, mutate_phase=False)

    def reset_symbol(self, symbol: str) -> None:
        """Explicitly clear stop state for *symbol* (e.g. after position closed)."""
        if self._use_rust:
            self._rust.reset_symbol(symbol)
        elif symbol in self._states:
            self._states[symbol].reset()

    def get_phase(self, symbol: str) -> StopPhase:
        """Return the current stop phase for *symbol*."""
        if self._use_rust:
            phase_str = self._rust.get_phase(symbol)
            return StopPhase(phase_str)
        return self._get_state(symbol).stop_phase

    # ── Internal ─────────────────────────────────────────────────────────────

    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        if self._use_rust:
            # Return a proxy that forwards push_true_range to Rust as well
            return _RustForwardingState(self._states[symbol], self._rust, symbol)
        return self._states[symbol]

    def _update_peak(self, state: _SymbolState, price: float) -> None:
        if state.side > 0:
            state.peak_price = max(state.peak_price, price)
        else:
            state.peak_price = min(state.peak_price, price)

    def _is_stop_breached(self, state: _SymbolState, price: float) -> bool:
        stop = self._compute_stop(state, price, mutate_phase=True)
        if state.side > 0:
            return price <= stop
        return price >= stop

    def _compute_stop(
        self, state: _SymbolState, current_price: float, *, mutate_phase: bool
    ) -> float:
        """Core stop-price computation.

        When *mutate_phase* is True the phase is updated in place as thresholds
        are crossed (used on the hot path).  When False the phase is read-only
        (used by compute_stop_price / tests).
        """
        atr = state.current_atr(self._atr_fallback)
        side = state.side
        entry = state.entry_price
        peak = state.peak_price

        # Profit fraction from the best price seen since entry
        if side > 0:
            profit_pct = (peak - entry) / entry if entry > 0 else 0.0
        else:
            profit_pct = (entry - peak) / entry if entry > 0 else 0.0

        # ── Phase selection ──────────────────────────────────────────────
        # Phases are monotonically advancing: INITIAL → BREAKEVEN → TRAILING.
        # We re-evaluate from current profit each call to be resilient to
        # ATR changes, but never regress once TRAILING is reached.
        current_phase = state.stop_phase

        # Phase thresholds:
        #   TRAILING  : profit ≥ _TRAIL_TRIGGER × ATR  (0.8×, mirrors AlphaRunner)
        #   BREAKEVEN : profit ≥ breakeven_trigger × ATR (default 1.0×)
        # Note: _TRAIL_TRIGGER < breakeven_trigger means TRAILING fires before
        # BREAKEVEN would be reached from INITIAL — BREAKEVEN is therefore only
        # reachable if breakeven_trigger is configured lower than _TRAIL_TRIGGER.
        if profit_pct >= atr * self._TRAIL_TRIGGER:
            new_phase = StopPhase.TRAILING
        elif profit_pct >= atr * self._breakeven_trigger:
            new_phase = StopPhase.BREAKEVEN
        else:
            new_phase = StopPhase.INITIAL

        # Monotonic advance only
        _order = {StopPhase.INITIAL: 0, StopPhase.BREAKEVEN: 1, StopPhase.TRAILING: 2}
        if _order[new_phase] > _order[current_phase]:
            if mutate_phase:
                state.stop_phase = new_phase
            effective_phase = new_phase
        else:
            effective_phase = current_phase

        # ── Compute raw stop ─────────────────────────────────────────────
        if effective_phase == StopPhase.TRAILING:
            trail_dist = atr * self._trailing_mult
            if side > 0:
                stop = peak * (1 - trail_dist)
            else:
                stop = peak * (1 + trail_dist)

        elif effective_phase == StopPhase.BREAKEVEN:
            buffer = atr * self._BREAKEVEN_BUFFER
            if side > 0:
                stop = entry * (1 + buffer)
            else:
                stop = entry * (1 - buffer)

        else:  # INITIAL
            initial_dist = atr * self._atr_initial_mult
            if side > 0:
                stop = entry * (1 - initial_dist)
            else:
                stop = entry * (1 + initial_dist)

        # ── Hard limits ──────────────────────────────────────────────────
        # 1. Max loss 5%
        if side > 0:
            floor = entry * (1 - self._MAX_LOSS_PCT)
            stop = max(stop, floor)
        else:
            ceil_ = entry * (1 + self._MAX_LOSS_PCT)
            stop = min(stop, ceil_)

        # 2. Min distance 0.3% from current price (avoids noise stops).
        # Only applies when the stop is on the safe side of current price
        # (i.e. not already breached).  If the stop has already been crossed
        # we leave it as-is; _is_stop_breached() will catch the trigger.
        min_dist = current_price * self._MIN_DIST_PCT
        if side > 0 and stop < current_price:
            # Stop is below current price (correct for long) — enforce min gap
            if current_price - stop < min_dist:
                stop = current_price - min_dist
        elif side < 0 and stop > current_price:
            # Stop is above current price (correct for short) — enforce min gap
            if stop - current_price < min_dist:
                stop = current_price + min_dist

        return stop
