# tests/unit/runner/test_adaptive_stop_gate.py
"""Unit tests for AdaptiveStopGate — ATR 3-phase stop-loss gate.

Coverage:
  - INITIAL phase: stop at entry ± atr × 2.0
  - BREAKEVEN transition: after sufficient profit
  - TRAILING phase: stop follows peak
  - Hard limits: 5% max loss, 0.3% min distance
  - Short positions (symmetric to long)
  - GateResult rejection when stop triggered
  - check_stop() real-time tick interface
  - on_new_position() state initialization
  - compute_stop_price() read-only query
  - Phase monotonic advance (no regression)
"""
from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any, Dict

from strategy.gates.adaptive_stop_gate import AdaptiveStopGate, StopPhase


# ── Minimal ORDER event stub ─────────────────────────────────────────────────

@dataclass
class _Order:
    symbol: str
    price: float
    qty: float
    side: str = "buy"


def _ctx(price: float, **kwargs: Any) -> Dict[str, Any]:
    return {"price": price, **kwargs}


def _gate(
    initial_mult: float = 2.0,
    breakeven_trigger: float = 1.0,
    trailing_mult: float = 0.3,
    atr_fallback: float = 0.015,
) -> AdaptiveStopGate:
    return AdaptiveStopGate(
        atr_initial_mult=initial_mult,
        atr_breakeven_trigger=breakeven_trigger,
        atr_trailing_mult=trailing_mult,
        atr_fallback=atr_fallback,
    )


def _seed_atr(gate: AdaptiveStopGate, symbol: str, atr_pct: float, n: int = 20) -> None:
    """Seed a deterministic ATR value into the gate's ATR buffer."""
    state = gate._get_state(symbol)
    for _ in range(n):
        # high/low/prev_close that produce atr_pct as TR/close
        close = 1000.0
        high = close * (1 + atr_pct / 2)
        low = close * (1 - atr_pct / 2)
        state.push_true_range(high, low, close)


# ── INITIAL phase ─────────────────────────────────────────────────────────────

class TestInitialPhase:
    def test_long_stop_at_initial_distance(self):
        gate = _gate()
        sym = "ETHUSDT"
        entry = 2000.0
        atr = 0.02  # 2%
        _seed_atr(gate, sym, atr)
        gate.on_new_position(sym, +1, entry)

        stop = gate.compute_stop_price(sym, entry)
        expected = entry * (1 - atr * 2.0)
        assert abs(stop - expected) < 0.01, f"Expected ~{expected:.2f}, got {stop:.2f}"

    def test_short_stop_at_initial_distance(self):
        gate = _gate()
        sym = "ETHUSDT"
        entry = 2000.0
        atr = 0.02
        _seed_atr(gate, sym, atr)
        gate.on_new_position(sym, -1, entry)

        stop = gate.compute_stop_price(sym, entry)
        expected = entry * (1 + atr * 2.0)
        assert abs(stop - expected) < 0.01

    def test_long_not_triggered_above_stop(self):
        gate = _gate(atr_fallback=0.02)
        sym = "BTCUSDT"
        entry = 10000.0
        gate.on_new_position(sym, +1, entry)

        # Price above stop — should pass
        safe_price = entry * 0.99  # only 1% down; stop is at -3% (2×1.5% fallback)
        result = gate.check(
            _Order(sym, safe_price, 1.0),
            _ctx(safe_price),
        )
        assert result.allowed

    def test_long_triggered_below_stop(self):
        gate = _gate(atr_fallback=0.02)
        sym = "BTCUSDT"
        entry = 10000.0
        gate.on_new_position(sym, +1, entry)

        # Price well below initial stop (entry × (1 - 2×0.02) = 9600)
        breached_price = entry * 0.95
        result = gate.check(
            _Order(sym, breached_price, 1.0),
            _ctx(breached_price),
        )
        assert not result.allowed
        assert result.reason == "stop_triggered"

    def test_short_triggered_above_stop(self):
        gate = _gate(atr_fallback=0.02)
        sym = "BTCUSDT"
        entry = 10000.0
        gate.on_new_position(sym, -1, entry)

        # Price well above initial stop (entry × (1 + 2×0.02) = 10400)
        breached_price = entry * 1.05
        result = gate.check(
            _Order(sym, breached_price, 1.0),
            _ctx(breached_price),
        )
        assert not result.allowed
        assert result.reason == "stop_triggered"

    def test_phase_is_initial_on_entry(self):
        gate = _gate()
        sym = "ETHUSDT"
        gate.on_new_position(sym, +1, 1800.0)
        assert gate.get_phase(sym) == StopPhase.INITIAL


# ── BREAKEVEN transition ──────────────────────────────────────────────────────

class TestBreakevenTransition:
    def test_phase_advances_to_breakeven(self):
        # Configure breakeven_trigger=0.5 so it fires before trailing trigger (0.8).
        # This creates a valid 3-phase path: INITIAL → BREAKEVEN → TRAILING.
        gate = AdaptiveStopGate(
            atr_initial_mult=2.0,
            atr_breakeven_trigger=0.5,   # fires at 0.5×ATR profit
            atr_trailing_mult=0.3,
            atr_fallback=0.015,
        )
        sym = "ETHUSDT"
        entry = 2000.0
        atr = 0.02
        _seed_atr(gate, sym, atr)
        gate.on_new_position(sym, +1, entry)

        # Move price up by 0.6×ATR profit (> 0.5× breakeven, < 0.8× trail)
        profit_price = entry * (1 + atr * 0.6)
        gate._update_peak(gate._get_state(sym), profit_price)
        gate.check(_Order(sym, profit_price, 1.0), _ctx(profit_price))
        assert gate.get_phase(sym) == StopPhase.BREAKEVEN

    def test_breakeven_stop_above_entry_for_long(self):
        gate = _gate()
        sym = "ETHUSDT"
        entry = 2000.0
        atr = 0.02
        _seed_atr(gate, sym, atr)
        gate.on_new_position(sym, +1, entry)

        # Force breakeven phase directly (bypassing automatic trigger)
        profit_price = entry * (1 + atr * 0.6)
        gate._update_peak(gate._get_state(sym), profit_price)
        gate._get_state(sym).stop_phase = StopPhase.BREAKEVEN

        stop = gate.compute_stop_price(sym, profit_price)
        # Breakeven stop should be at entry + 0.1×ATR buffer
        expected = entry * (1 + atr * gate._BREAKEVEN_BUFFER)
        assert stop >= entry, "Breakeven stop must be at or above entry for long"
        assert abs(stop - expected) < 1.0

    def test_breakeven_stop_below_entry_for_short(self):
        gate = _gate()
        sym = "ETHUSDT"
        entry = 2000.0
        atr = 0.02
        _seed_atr(gate, sym, atr)
        gate.on_new_position(sym, -1, entry)

        profit_price = entry * (1 - atr * 0.6)
        gate._update_peak(gate._get_state(sym), profit_price)
        gate._get_state(sym).stop_phase = StopPhase.BREAKEVEN

        stop = gate.compute_stop_price(sym, profit_price)
        assert stop <= entry, "Breakeven stop must be at or below entry for short"


# ── TRAILING phase ───────────────────────────────────────────────────────────

class TestTrailingPhase:
    def test_phase_advances_to_trailing(self):
        gate = _gate()
        sym = "ETHUSDT"
        entry = 2000.0
        atr = 0.02
        _seed_atr(gate, sym, atr)
        gate.on_new_position(sym, +1, entry)

        # Move price up by > 0.8×ATR (trail trigger)
        profit_price = entry * (1 + atr * 0.9)
        gate.check(_Order(sym, profit_price, 1.0), _ctx(profit_price))
        assert gate.get_phase(sym) == StopPhase.TRAILING

    def test_trailing_stop_follows_peak_long(self):
        gate = _gate()
        sym = "ETHUSDT"
        entry = 2000.0
        atr = 0.02
        trail_mult = 0.3
        _seed_atr(gate, sym, atr)
        gate.on_new_position(sym, +1, entry)

        # Simulate price running up
        peak = entry * 1.05
        state = gate._get_state(sym)
        state.peak_price = peak
        state.stop_phase = StopPhase.TRAILING

        stop = gate.compute_stop_price(sym, peak)
        expected = peak * (1 - atr * trail_mult)
        assert abs(stop - expected) < 1.0

    def test_trailing_stop_follows_peak_short(self):
        gate = _gate()
        sym = "ETHUSDT"
        entry = 2000.0
        atr = 0.02
        trail_mult = 0.3
        _seed_atr(gate, sym, atr)
        gate.on_new_position(sym, -1, entry)

        peak = entry * 0.95  # lower = better for short
        state = gate._get_state(sym)
        state.peak_price = peak
        state.stop_phase = StopPhase.TRAILING

        stop = gate.compute_stop_price(sym, peak)
        expected = peak * (1 + atr * trail_mult)
        assert abs(stop - expected) < 1.0

    def test_trailing_stop_moves_up_as_price_rises(self):
        gate = _gate()
        sym = "ETHUSDT"
        entry = 2000.0
        atr = 0.02
        _seed_atr(gate, sym, atr)
        gate.on_new_position(sym, +1, entry)

        state = gate._get_state(sym)
        state.stop_phase = StopPhase.TRAILING

        # First peak
        peak1 = entry * 1.03
        state.peak_price = peak1
        stop1 = gate.compute_stop_price(sym, peak1)

        # Higher peak
        peak2 = entry * 1.06
        state.peak_price = peak2
        stop2 = gate.compute_stop_price(sym, peak2)

        assert stop2 > stop1, "Trailing stop should advance as price rises"

    def test_phase_does_not_regress(self):
        """Once TRAILING, a modest dip that does NOT breach the stop should keep TRAILING."""
        gate = _gate()
        sym = "ETHUSDT"
        entry = 2000.0
        atr = 0.02
        _seed_atr(gate, sym, atr)
        gate.on_new_position(sym, +1, entry)

        state = gate._get_state(sym)
        peak = entry * 1.10  # e.g. 2200
        state.peak_price = peak
        state.stop_phase = StopPhase.TRAILING

        # Trailing stop = peak × (1 - atr × trailing_mult) = 2200 × (1 - 0.006) ≈ 2186.8
        # Use a price slightly above the trailing stop (not a breach)
        trailing_stop = peak * (1 - atr * 0.3)
        safe_price = trailing_stop + 5.0  # just above the stop

        gate.check(_Order(sym, safe_price, 1.0), _ctx(safe_price))
        assert gate.get_phase(sym) == StopPhase.TRAILING


# ── Hard limits ───────────────────────────────────────────────────────────────

class TestHardLimits:
    def test_max_loss_5pct_long(self):
        """Stop must never allow more than 5% loss for long."""
        # Use tiny ATR so initial stop would be further than 5%
        gate = _gate(initial_mult=20.0, atr_fallback=0.30)  # 20×30% = 600% — huge
        sym = "ETHUSDT"
        entry = 2000.0
        _seed_atr(gate, sym, 0.30)
        gate.on_new_position(sym, +1, entry)

        stop = gate.compute_stop_price(sym, entry)
        min_stop = entry * 0.95
        assert stop >= min_stop, f"Stop {stop:.2f} allows > 5% loss (entry={entry})"

    def test_max_loss_5pct_short(self):
        gate = _gate(initial_mult=20.0, atr_fallback=0.30)
        sym = "ETHUSDT"
        entry = 2000.0
        _seed_atr(gate, sym, 0.30)
        gate.on_new_position(sym, -1, entry)

        stop = gate.compute_stop_price(sym, entry)
        max_stop = entry * 1.05
        assert stop <= max_stop, f"Stop {stop:.2f} allows > 5% loss for short"

    def test_min_distance_0_3pct_long(self):
        """Stop must stay ≥ 0.3% below current price for long."""
        # Force breakeven stop very tight by using near-zero ATR fallback
        gate = _gate(atr_fallback=0.0001, initial_mult=0.001)
        sym = "ETHUSDT"
        entry = 2000.0
        _seed_atr(gate, sym, 0.0001)
        gate.on_new_position(sym, +1, entry)

        current_price = entry * 1.001
        stop = gate.compute_stop_price(sym, current_price)
        min_dist = current_price * 0.003
        assert current_price - stop >= min_dist - 1e-6, (
            f"Stop {stop:.4f} too close to price {current_price:.4f} "
            f"(min dist={min_dist:.4f})"
        )

    def test_min_distance_0_3pct_short(self):
        gate = _gate(atr_fallback=0.0001, initial_mult=0.001)
        sym = "ETHUSDT"
        entry = 2000.0
        _seed_atr(gate, sym, 0.0001)
        gate.on_new_position(sym, -1, entry)

        current_price = entry * 0.999
        stop = gate.compute_stop_price(sym, current_price)
        min_dist = current_price * 0.003
        assert stop - current_price >= min_dist - 1e-6, (
            f"Stop {stop:.4f} too close to price {current_price:.4f} for short"
        )


# ── Short positions ───────────────────────────────────────────────────────────

class TestShortPositions:
    def test_short_initial_stop_above_entry(self):
        gate = _gate(atr_fallback=0.02)
        sym = "BTCUSDT"
        entry = 50000.0
        gate.on_new_position(sym, -1, entry)

        stop = gate.compute_stop_price(sym, entry)
        assert stop > entry, "Short initial stop must be above entry"

    def test_short_not_triggered_when_price_falls(self):
        gate = _gate(atr_fallback=0.02)
        sym = "BTCUSDT"
        entry = 50000.0
        gate.on_new_position(sym, -1, entry)

        lower_price = entry * 0.97  # price fell 3% — good for short
        result = gate.check(_Order(sym, lower_price, 1.0), _ctx(lower_price))
        assert result.allowed

    def test_short_triggered_when_price_rises_past_stop(self):
        gate = _gate(atr_fallback=0.02)
        sym = "BTCUSDT"
        entry = 50000.0
        gate.on_new_position(sym, -1, entry)

        # Initial stop = entry × (1 + 2×0.02) = 52000
        breached_price = entry * 1.05  # 52500 > 52000
        result = gate.check(_Order(sym, breached_price, 1.0), _ctx(breached_price))
        assert not result.allowed
        assert result.reason == "stop_triggered"


# ── check_stop() tick interface ───────────────────────────────────────────────

class TestCheckStopTick:
    def test_returns_false_when_flat(self):
        gate = _gate()
        assert gate.check_stop("ETHUSDT", 2000.0) is False

    def test_returns_true_on_breach_and_resets(self):
        gate = _gate(atr_fallback=0.02)
        sym = "ETHUSDT"
        entry = 2000.0
        gate.on_new_position(sym, +1, entry)

        breached_price = entry * 0.93  # well below stop
        triggered = gate.check_stop(sym, breached_price)
        assert triggered is True
        # State should be reset
        assert gate._get_state(sym).side == 0

    def test_returns_false_when_not_breached(self):
        gate = _gate(atr_fallback=0.02)
        sym = "ETHUSDT"
        entry = 2000.0
        gate.on_new_position(sym, +1, entry)

        safe_price = entry * 0.99
        triggered = gate.check_stop(sym, safe_price)
        assert triggered is False


# ── GateResult pass-through when no position ─────────────────────────────────

class TestNoPositionPassThrough:
    def test_no_position_allows_order(self):
        gate = _gate()
        sym = "SUIUSDT"
        result = gate.check(_Order(sym, 1.5, 100.0), _ctx(1.5))
        assert result.allowed

    def test_state_reset_after_stop(self):
        gate = _gate(atr_fallback=0.02)
        sym = "ETHUSDT"
        gate.on_new_position(sym, +1, 2000.0)

        # Trigger stop
        gate.check(_Order(sym, 1850.0, 1.0), _ctx(1850.0))

        # Subsequent order should pass (flat now)
        result = gate.check(_Order(sym, 1900.0, 1.0), _ctx(1900.0))
        assert result.allowed


# ── compute_stop_price() read-only ───────────────────────────────────────────

class TestComputeStopReadOnly:
    def test_does_not_change_phase(self):
        gate = _gate()
        sym = "ETHUSDT"
        entry = 2000.0
        atr = 0.02
        _seed_atr(gate, sym, atr)
        gate.on_new_position(sym, +1, entry)

        assert gate.get_phase(sym) == StopPhase.INITIAL

        # Query at a profit-level price that would trigger TRAILING
        profit_price = entry * (1 + atr * 1.0)
        gate.compute_stop_price(sym, profit_price)

        # Phase must remain INITIAL (read-only)
        assert gate.get_phase(sym) == StopPhase.INITIAL

    def test_returns_zero_when_flat(self):
        gate = _gate()
        assert gate.compute_stop_price("ETHUSDT", 2000.0) == 0.0


# ── ATR buffer seeding ────────────────────────────────────────────────────────

class TestAtrBuffer:
    def test_fallback_with_insufficient_data(self):
        gate = AdaptiveStopGate(atr_fallback=0.015)
        state = gate._get_state("ETHUSDT")
        assert state.current_atr() == pytest.approx(0.015)

    def test_atr_computed_from_buffer(self):
        gate = AdaptiveStopGate(atr_fallback=0.015)
        sym = "ETHUSDT"
        _seed_atr(gate, sym, 0.02, n=20)
        state = gate._get_state(sym)
        # Should be very close to the seeded 2% (small rounding)
        assert abs(state.current_atr() - 0.02) < 0.001

    def test_bar_data_from_context(self):
        """Gate should consume bar_high/bar_low/prev_close from context."""
        gate = _gate()
        sym = "ETHUSDT"
        gate.on_new_position(sym, +1, 2000.0)

        ev = _Order(sym, 2100.0, 1.0)
        ctx = _ctx(2100.0, bar_high=2120.0, bar_low=2080.0, prev_close=2000.0)
        gate.check(ev, ctx)

        state = gate._get_state(sym)
        assert len(state.atr_buffer) == 1


# ── reset_symbol() ────────────────────────────────────────────────────────────

class TestResetSymbol:
    def test_reset_clears_position(self):
        gate = _gate()
        gate.on_new_position("ETHUSDT", +1, 2000.0)
        gate.reset_symbol("ETHUSDT")
        assert gate._get_state("ETHUSDT").side == 0

    def test_reset_unknown_symbol_noop(self):
        gate = _gate()
        gate.reset_symbol("UNKNOWNUSDT")  # must not raise
