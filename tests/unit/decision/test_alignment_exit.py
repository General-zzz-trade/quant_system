"""Tests for direction alignment exit (ETH follows BTC) during holding."""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from decision.modules.alpha import AlphaDecisionModule


# ── helpers ──────────────────────────────────────────────────────


def _make_snapshot(
    symbol: str = "ETHUSDT",
    close: float = 3000.0,
    features: dict | None = None,
) -> MagicMock:
    snap = MagicMock()
    snap.symbol = symbol

    mkt = MagicMock()
    mkt.close = Decimal(str(close))
    mkt.close_f = close
    mkt.high = Decimal(str(close * 1.005))
    mkt.high_f = close * 1.005
    mkt.low = Decimal(str(close * 0.995))
    mkt.low_f = close * 0.995
    snap.markets = {symbol: mkt}

    pos = MagicMock()
    pos.qty = Decimal("0")
    snap.positions = {symbol: pos}

    acc = MagicMock()
    acc.balance = Decimal("10000")
    snap.account = acc

    snap.features = features or {}
    snap.ts = None
    snap.event_id = "test-align"
    snap.portfolio = None
    snap.risk = None
    return snap


def _make_module(
    symbol: str = "ETHUSDT",
    runner_key: str = "ETHUSDT",
) -> AlphaDecisionModule:
    predictor = MagicMock()
    predictor.predict.return_value = 0.5

    discretizer = MagicMock()
    # Default: hold current signal (return same signal as current)
    discretizer.discretize.return_value = (1, 1.5)
    discretizer.deadzone = 0.9
    discretizer.min_hold = 18
    discretizer.max_hold = 120

    sizer = MagicMock()
    sizer.target_qty.return_value = Decimal("0.1")
    sizer.min_size = Decimal("0.001")

    mod = AlphaDecisionModule(
        symbol=symbol,
        runner_key=runner_key,
        predictor=predictor,
        discretizer=discretizer,
        sizer=sizer,
    )
    return mod


def _set_holding(mod: AlphaDecisionModule, signal: int, entry_price: float = 3000.0):
    """Simulate an existing position by setting internal state."""
    mod._signal = signal
    mod._entry_price = entry_price
    mod._trade_peak = entry_price
    mod._current_qty = Decimal("0.1")
    # Seed enough close/ATR history to avoid warmup issues
    mod._closes = [entry_price] * 30
    mod._atr_buffer = [0.01] * 20


# ── tests ────────────────────────────────────────────────────────


class TestAlignmentExit:
    """Direction alignment exit: ETH exits when BTC opposes."""

    def test_eth_long_btc_long_no_exit(self):
        """ETH long + BTC long → no alignment exit."""
        mod = _make_module()
        _set_holding(mod, signal=1)
        mod.set_consensus({"BTCUSDT": 1})

        ok, reason = mod._check_force_exits(3000.0, 1.5)
        assert not ok or "alignment_exit" not in reason

    def test_eth_long_btc_short_exit(self):
        """ETH long + BTC short → alignment exit."""
        mod = _make_module()
        _set_holding(mod, signal=1)
        mod.set_consensus({"BTCUSDT": -1})

        ok, reason = mod._check_force_exits(3000.0, 1.5)
        assert ok
        assert "alignment_exit" in reason
        assert "eth=1" in reason
        assert "btc=-1" in reason

    def test_eth_short_btc_long_exit(self):
        """ETH short + BTC long → alignment exit."""
        mod = _make_module()
        _set_holding(mod, signal=-1)
        mod.set_consensus({"BTCUSDT": 1})

        ok, reason = mod._check_force_exits(3000.0, -1.5)
        assert ok
        assert "alignment_exit" in reason
        assert "eth=-1" in reason
        assert "btc=1" in reason

    def test_eth_short_btc_short_no_exit(self):
        """ETH short + BTC short → no alignment exit."""
        mod = _make_module()
        _set_holding(mod, signal=-1)
        mod.set_consensus({"BTCUSDT": -1})

        ok, reason = mod._check_force_exits(3000.0, -1.5)
        assert not ok or "alignment_exit" not in reason

    def test_eth_long_btc_flat_no_exit(self):
        """ETH long + BTC flat (0) → no alignment exit."""
        mod = _make_module()
        _set_holding(mod, signal=1)
        mod.set_consensus({"BTCUSDT": 0})

        ok, reason = mod._check_force_exits(3000.0, 1.5)
        assert not ok or "alignment_exit" not in reason

    def test_eth_short_btc_flat_no_exit(self):
        """ETH short + BTC flat (0) → no alignment exit."""
        mod = _make_module()
        _set_holding(mod, signal=-1)
        mod.set_consensus({"BTCUSDT": 0})

        ok, reason = mod._check_force_exits(3000.0, -1.5)
        assert not ok or "alignment_exit" not in reason

    def test_btc_symbol_no_alignment_exit(self):
        """BTC symbol → alignment exit never triggers."""
        mod = _make_module(symbol="BTCUSDT", runner_key="BTCUSDT")
        _set_holding(mod, signal=1)
        mod.set_consensus({"ETHUSDT": -1})

        ok, reason = mod._check_force_exits(90000.0, 1.5)
        assert not ok or "alignment_exit" not in reason

    def test_4h_symbol_no_alignment_exit(self):
        """4h runner → alignment exit never triggers."""
        mod = _make_module(symbol="ETHUSDT", runner_key="ETHUSDT_4h")
        _set_holding(mod, signal=1)
        mod.set_consensus({"BTCUSDT": -1})

        ok, reason = mod._check_force_exits(3000.0, 1.5)
        assert not ok or "alignment_exit" not in reason

    def test_consensus_empty_no_exit(self):
        """Empty consensus → no alignment exit."""
        mod = _make_module()
        _set_holding(mod, signal=1)
        # No consensus set at all

        ok, reason = mod._check_force_exits(3000.0, 1.5)
        assert not ok or "alignment_exit" not in reason

    def test_consensus_missing_btc_key_no_exit(self):
        """Consensus has no BTC key for this ETH symbol → no exit."""
        mod = _make_module()
        _set_holding(mod, signal=1)
        mod.set_consensus({"SOLUSDT": -1})  # unrelated symbol

        ok, reason = mod._check_force_exits(3000.0, 1.5)
        assert not ok or "alignment_exit" not in reason

    def test_no_position_no_exit(self):
        """No position (signal=0) → _check_force_exits returns early."""
        mod = _make_module()
        mod._signal = 0
        mod._entry_price = 0.0
        mod.set_consensus({"BTCUSDT": -1})

        ok, reason = mod._check_force_exits(3000.0, 1.5)
        assert not ok
        assert reason == ""

    def test_alignment_exit_triggers_close_via_decide(self):
        """Full integration: alignment exit causes position close in decide()."""
        mod = _make_module()
        _set_holding(mod, signal=1, entry_price=3000.0)
        mod.set_consensus({"BTCUSDT": -1})

        # Discretizer returns hold signal (1) — same as current
        mod._discretizer.discretize.return_value = (1, 1.5)

        snap = _make_snapshot(close=3000.0)
        events = list(mod.decide(snap))

        # Should have generated close order(s) due to force exit
        assert mod._signal == 0, "Signal should be reset to 0 after alignment exit"

    def test_alignment_exit_reason_in_risk_event(self):
        """alignment_exit produces a RiskEvent with correct rule_id."""
        from event.types import RiskEvent

        mod = _make_module()
        _set_holding(mod, signal=1, entry_price=3000.0)
        mod.set_consensus({"BTCUSDT": -1})

        mod._discretizer.discretize.return_value = (1, 1.5)

        snap = _make_snapshot(close=3000.0)
        events = list(mod.decide(snap))

        risk_events = [e for e in events if isinstance(e, RiskEvent)]
        assert any("alignment_exit" in e.rule_id for e in risk_events), \
            f"Expected alignment_exit RiskEvent, got: {[e.rule_id for e in risk_events]}"
