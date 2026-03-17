"""Parity test: RustAdaptiveStopGate vs Python AdaptiveStopGate.

Verifies that compute_stop_price() and check_stop() results are identical
between the Rust and Python implementations for all phases and edge cases.
"""
import pytest
from runner.gates.adaptive_stop_gate import AdaptiveStopGate

try:
    from _quant_hotpath import RustAdaptiveStopGate
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust not built")


def make_python_gate(**kwargs):
    """Force Python path."""
    gate = AdaptiveStopGate(**kwargs)
    gate._use_rust = False
    gate._rust = None
    return gate


def feed_atr(gate_or_rust, symbol, n=20, is_rust=False):
    """Feed n ATR samples (~2% TR per bar)."""
    high, low, prev = 101.0, 99.0, 100.0
    for _ in range(n):
        if is_rust:
            gate_or_rust.push_true_range(symbol, high, low, prev)
        else:
            gate_or_rust._get_state(symbol).push_true_range(high, low, prev)


class TestAdaptiveStopParity:
    def test_initial_phase_stop_long(self):
        rust = RustAdaptiveStopGate()
        py = make_python_gate()
        sym = "ETHUSDT"
        entry = 100.0

        feed_atr(rust, sym, 20, is_rust=True)
        feed_atr(py, sym, 20)

        rust.on_new_position(sym, 1, entry)
        py.on_new_position(sym, 1, entry)

        price = 95.0
        rust_stop = rust.compute_stop_price(sym, price)
        py_stop = py.compute_stop_price(sym, price)

        assert rust_stop == pytest.approx(py_stop, rel=1e-9)

    def test_initial_phase_stop_short(self):
        rust = RustAdaptiveStopGate()
        py = make_python_gate()
        sym = "ETHUSDT"
        entry = 100.0

        feed_atr(rust, sym, 20, is_rust=True)
        feed_atr(py, sym, 20)

        rust.on_new_position(sym, -1, entry)
        py.on_new_position(sym, -1, entry)

        price = 105.0
        assert rust.compute_stop_price(sym, price) == pytest.approx(
            py.compute_stop_price(sym, price), rel=1e-9
        )

    def test_phase_transition_to_trailing(self):
        rust = RustAdaptiveStopGate()
        sym = "ETHUSDT"
        feed_atr(rust, sym, 20, is_rust=True)
        rust.on_new_position(sym, 1, 100.0)

        # ATR ≈ 0.02, trail_trigger=0.8 → need profit_pct >= 0.016
        # Push price to 102.0 (profit=2%, ATR=2%, 0.8*ATR=1.6%)
        rust.check_stop(sym, 102.0)
        assert rust.get_phase(sym) == "TRAILING"

    def test_max_loss_limit(self):
        rust = RustAdaptiveStopGate()
        py = make_python_gate()
        sym = "ETHUSDT"

        # High ATR → would put stop far below entry
        for _ in range(20):
            rust.push_true_range(sym, 115.0, 85.0, 100.0)  # 30% TR
            py._get_state(sym).push_true_range(115.0, 85.0, 100.0)

        rust.on_new_position(sym, 1, 100.0)
        py.on_new_position(sym, 1, 100.0)

        rust_stop = rust.compute_stop_price(sym, 100.0)
        py_stop = py.compute_stop_price(sym, 100.0)

        assert rust_stop == pytest.approx(py_stop, rel=1e-9)
        assert rust_stop >= 95.0  # max 5% loss

    def test_max_loss_limit_short(self):
        rust = RustAdaptiveStopGate()
        py = make_python_gate()
        sym = "ETHUSDT"

        for _ in range(20):
            rust.push_true_range(sym, 115.0, 85.0, 100.0)
            py._get_state(sym).push_true_range(115.0, 85.0, 100.0)

        rust.on_new_position(sym, -1, 100.0)
        py.on_new_position(sym, -1, 100.0)

        rust_stop = rust.compute_stop_price(sym, 100.0)
        py_stop = py.compute_stop_price(sym, 100.0)

        assert rust_stop == pytest.approx(py_stop, rel=1e-9)
        assert rust_stop <= 105.0  # max 5% loss from entry

    def test_min_distance_enforcement(self):
        rust = RustAdaptiveStopGate()
        py = make_python_gate()
        sym = "ETHUSDT"

        # Very small ATR → stop close to price
        for _ in range(20):
            rust.push_true_range(sym, 100.001, 99.999, 100.0)  # tiny TR
            py._get_state(sym).push_true_range(100.001, 99.999, 100.0)

        rust.on_new_position(sym, 1, 100.0)
        py.on_new_position(sym, 1, 100.0)

        price = 100.0
        rust_stop = rust.compute_stop_price(sym, price)
        py_stop = py.compute_stop_price(sym, price)

        assert rust_stop == pytest.approx(py_stop, rel=1e-9)

    def test_stop_breach_triggers(self):
        rust = RustAdaptiveStopGate()
        sym = "ETHUSDT"
        feed_atr(rust, sym, 20, is_rust=True)
        rust.on_new_position(sym, 1, 100.0)

        stop = rust.compute_stop_price(sym, 95.0)
        # Price below stop → should trigger
        assert rust.check_stop(sym, stop - 0.01) is True

    def test_no_position_allows(self):
        rust = RustAdaptiveStopGate()
        sym = "ETHUSDT"
        # No position set → should not stop
        assert rust.check_stop(sym, 100.0) is False

    def test_atr_fallback_when_few_samples(self):
        rust = RustAdaptiveStopGate(atr_fallback=0.015)
        py = make_python_gate(atr_fallback=0.015)
        sym = "ETHUSDT"

        # Only 3 ATR samples (< 5)
        for _ in range(3):
            rust.push_true_range(sym, 101.0, 99.0, 100.0)
            py._get_state(sym).push_true_range(101.0, 99.0, 100.0)

        rust.on_new_position(sym, 1, 100.0)
        py.on_new_position(sym, 1, 100.0)

        rust_stop = rust.compute_stop_price(sym, 100.0)
        py_stop = py.compute_stop_price(sym, 100.0)

        assert rust_stop == pytest.approx(py_stop, rel=1e-6)

    def test_trailing_phase_stop_parity(self):
        """Check stop price parity in TRAILING phase."""
        rust = RustAdaptiveStopGate()
        py = make_python_gate()
        sym = "ETHUSDT"

        feed_atr(rust, sym, 20, is_rust=True)
        feed_atr(py, sym, 20)

        rust.on_new_position(sym, 1, 100.0)
        py.on_new_position(sym, 1, 100.0)

        # Advance both to trailing phase by updating peak to 102
        rust.check_stop(sym, 102.0)  # updates peak and phase
        py.check_stop(sym, 102.0)

        # Both should now be in TRAILING
        assert rust.get_phase(sym) == "TRAILING"
        from runner.gates.adaptive_stop_gate import StopPhase
        assert py.get_phase(sym) == StopPhase.TRAILING

        # Stop prices at 103 should match
        price = 103.0
        assert rust.compute_stop_price(sym, price) == pytest.approx(
            py.compute_stop_price(sym, price), rel=1e-9
        )

    def test_short_trailing_phase_stop_parity(self):
        """Short trailing stop parity."""
        rust = RustAdaptiveStopGate()
        py = make_python_gate()
        sym = "BTCUSDT"

        feed_atr(rust, sym, 20, is_rust=True)
        feed_atr(py, sym, 20)

        rust.on_new_position(sym, -1, 100.0)
        py.on_new_position(sym, -1, 100.0)

        # Push price down to trigger trailing for short (profit = (100-98)/100=2% >= 1.6%)
        rust.check_stop(sym, 98.0)
        py.check_stop(sym, 98.0)

        price = 97.0
        assert rust.compute_stop_price(sym, price) == pytest.approx(
            py.compute_stop_price(sym, price), rel=1e-9
        )

    def test_gate_result_integration(self):
        """AdaptiveStopGate.check() returns correct GateResult via Rust."""
        gate = AdaptiveStopGate()
        if not gate._use_rust:
            pytest.skip("Rust not active")

        sym = "ETHUSDT"
        context_with_atr = {
            "symbol": sym, "bar_high": 101.0, "bar_low": 99.0,
            "prev_close": 100.0, "price": 100.0
        }

        class FakeEv:
            symbol = sym

        # Feed ATR
        for _ in range(20):
            gate.check(FakeEv(), context_with_atr)

        gate.on_new_position(sym, 1, 100.0)

        # Price well above entry → no stop
        result = gate.check(FakeEv(), {**context_with_atr, "price": 101.0})
        assert result.allowed is True

    def test_gate_result_stop_triggered(self):
        """AdaptiveStopGate.check() blocks when stop is breached."""
        gate = AdaptiveStopGate()
        if not gate._use_rust:
            pytest.skip("Rust not active")

        sym = "ETHUSDT"

        class FakeEv:
            symbol = sym

        # Seed ATR directly (no context bar data to avoid ATR push during check)
        for _ in range(20):
            gate._rust.push_true_range(sym, 101.0, 99.0, 100.0)
            gate._states.setdefault(sym, __import__('runner.gates.adaptive_stop_gate',
                fromlist=['_SymbolState'])._SymbolState())
            gate._states[sym].push_true_range(101.0, 99.0, 100.0)

        gate.on_new_position(sym, 1, 100.0)

        # ATR ≈ 0.02, initial stop = 100*(1-2*0.02) = 96.0
        # Price at 95.5 is below stop → triggered
        # Use context without bar data to avoid ATR mutation during the check
        result = gate.check(FakeEv(), {"price": 95.5})
        assert result.allowed is False
        assert result.reason == "stop_triggered"
