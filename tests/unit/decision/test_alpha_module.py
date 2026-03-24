"""Tests for AlphaDecisionModule."""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from decision.modules.alpha import AlphaDecisionModule
from event.types import OrderEvent, RiskEvent, SignalEvent


# ── snapshot helper ─────────────────────────────────────────────


def _make_snapshot(
    symbol: str = "BTCUSDT",
    close: float = 90000.0,
    equity: float = 1000.0,
    features: dict | None = None,
    position_qty: float = 0,
    bar_index: int = 900,
) -> MagicMock:
    snap = MagicMock()
    snap.symbol = symbol
    snap.bar_index = bar_index

    mkt = MagicMock()
    mkt.close = Decimal(str(close))
    mkt.high = Decimal(str(close * 1.01))
    mkt.low = Decimal(str(close * 0.99))
    snap.markets = {symbol: mkt}

    pos = MagicMock()
    pos.qty = Decimal(str(position_qty))
    snap.positions = {symbol: pos}

    acc = MagicMock()
    acc.balance = Decimal(str(equity))
    snap.account = acc

    snap.features = features or {}
    snap.ts = None
    snap.event_id = "test-001"
    return snap


# ── component fixtures ──────────────────────────────────────────


def _make_predictor(return_val: float | None = 0.5) -> MagicMock:
    pred = MagicMock()
    pred.predict.return_value = return_val
    return pred


def _make_discretizer(
    signal: int = 1,
    z: float = 1.5,
    deadzone: float = 0.9,
) -> MagicMock:
    disc = MagicMock()
    disc.discretize.return_value = (signal, z)
    disc.deadzone = deadzone
    return disc


def _make_sizer(qty: float = 0.01) -> MagicMock:
    sizer = MagicMock()
    sizer.target_qty.return_value = Decimal(str(qty))
    return sizer


def _make_module(
    symbol: str = "BTCUSDT",
    runner_key: str = "BTCUSDT",
    pred_val: float | None = 0.5,
    signal: int = 1,
    z: float = 1.5,
    deadzone: float = 0.9,
    qty: float = 0.01,
) -> tuple[AlphaDecisionModule, MagicMock, MagicMock, MagicMock]:
    predictor = _make_predictor(pred_val)
    discretizer = _make_discretizer(signal, z, deadzone)
    sizer = _make_sizer(qty)
    mod = AlphaDecisionModule(
        symbol=symbol,
        runner_key=runner_key,
        predictor=predictor,
        discretizer=discretizer,
        sizer=sizer,
    )
    return mod, predictor, discretizer, sizer


# ── tests ───────────────────────────────────────────────────────


class TestAlphaDecisionModule:
    """Core AlphaDecisionModule tests."""

    def test_decide_returns_order_events(self):
        """Predict -> signal=1 -> OrderEvent + SignalEvent emitted."""
        mod, _, _, _ = _make_module(signal=1, z=1.5)
        snap = _make_snapshot()
        events = list(mod.decide(snap))
        orders = [e for e in events if isinstance(e, OrderEvent)]
        signals = [e for e in events if isinstance(e, SignalEvent)]
        assert len(orders) == 1
        assert orders[0].side == "buy"
        assert orders[0].symbol == "BTCUSDT"
        assert orders[0].qty > 0
        assert len(signals) == 1
        assert signals[0].side == "long"

    def test_decide_flat_no_events(self):
        """Predict -> signal=0 -> empty."""
        mod, _, _, _ = _make_module(signal=0, z=0.1)
        snap = _make_snapshot()
        events = list(mod.decide(snap))
        assert len(events) == 0

    def test_direction_alignment_blocks_eth(self):
        """ETH short blocked when BTC consensus is long."""
        mod, _, _, _ = _make_module(
            symbol="ETHUSDT", runner_key="ETHUSDT", signal=-1, z=-1.5,
        )
        mod.set_consensus({"BTCUSDT": 1, "BTCUSDT_4h": 1})
        snap = _make_snapshot(symbol="ETHUSDT")
        events = list(mod.decide(snap))
        orders = [e for e in events if isinstance(e, OrderEvent)]
        assert len(orders) == 0
        # RiskEvent emitted for direction alignment block
        risk_events = [e for e in events if isinstance(e, RiskEvent)]
        assert len(risk_events) == 1
        assert risk_events[0].rule_id == "direction_alignment"

    def test_direction_alignment_allows_same_dir(self):
        """ETH long allowed when BTC consensus is long."""
        mod, _, _, _ = _make_module(
            symbol="ETHUSDT", runner_key="ETHUSDT", signal=1, z=1.5,
        )
        mod.set_consensus({"BTCUSDT": 1})
        snap = _make_snapshot(symbol="ETHUSDT")
        events = list(mod.decide(snap))
        orders = [e for e in events if isinstance(e, OrderEvent)]
        assert len(orders) == 1
        assert orders[0].side == "buy"

    def test_force_exit_quick_loss(self):
        """-2% drop from entry -> close order."""
        mod, _, disc, _ = _make_module(signal=1, z=1.5)
        # First bar: open long
        snap1 = _make_snapshot(close=90000.0)
        events1 = list(mod.decide(snap1))
        orders1 = [e for e in events1 if isinstance(e, OrderEvent)]
        assert len(orders1) == 1
        assert orders1[0].side == "buy"

        # Second bar: -2% drop, signal still +1 from discretizer
        # but force exit should trigger
        disc.discretize.return_value = (1, 1.0)
        snap2 = _make_snapshot(close=88000.0)
        events2 = list(mod.decide(snap2))
        # Should emit close OrderEvent + RiskEvent
        orders2 = [e for e in events2 if isinstance(e, OrderEvent)]
        assert any(e.qty == Decimal("0") for e in orders2)
        risk_events = [e for e in events2 if isinstance(e, RiskEvent)]
        assert len(risk_events) >= 1

    def test_force_exit_z_reversal(self):
        """Long + z=-0.5 -> close order."""
        mod, _, disc, _ = _make_module(signal=1, z=1.5)
        # Open long
        snap1 = _make_snapshot(close=90000.0)
        list(mod.decide(snap1))

        # Z reverses, but discretizer still returns signal=1
        # Force exit should trigger due to z < -0.3
        disc.discretize.return_value = (1, -0.5)
        snap2 = _make_snapshot(close=90000.0)
        events = list(mod.decide(snap2))
        orders = [e for e in events if isinstance(e, OrderEvent)]
        assert any(e.qty == Decimal("0") for e in orders)

    def test_regime_filter_warmup_active(self):
        """<20 bars -> regime always active."""
        mod, _, _, _ = _make_module(signal=1, z=1.5)
        # Process a few bars
        for i in range(5):
            snap = _make_snapshot(close=90000.0 + i * 100)
            list(mod.decide(snap))
        assert mod._regime_active is True

    def test_z_scale_mapping(self):
        """Verify all 4 z_scale brackets."""
        assert AlphaDecisionModule._compute_z_scale(2.5) == 1.5
        assert AlphaDecisionModule._compute_z_scale(1.5) == 1.0
        assert AlphaDecisionModule._compute_z_scale(0.7) == 0.7
        assert AlphaDecisionModule._compute_z_scale(0.3) == 0.5

    def test_signal_change_emits_close_then_open(self):
        """prev=1, new=-1 -> SignalEvent + close + open = 3 events.

        We must avoid force-exit triggers: use a close-to-entry price
        and set the discretizer to return signal=-1 directly so the
        force exit check sees signal=1, z=-1.5 which would trigger
        z_reversal.  Instead, we bypass force exits by patching.
        """
        mod, _, disc, _ = _make_module(signal=1, z=1.5)
        # Open long
        snap1 = _make_snapshot(close=90000.0)
        list(mod.decide(snap1))
        assert mod._signal == 1

        # Flip to short — patch force exits to not trigger
        disc.discretize.return_value = (-1, -1.5)
        with patch.object(mod, "_check_force_exits", return_value=(False, "")):
            snap2 = _make_snapshot(close=90000.0)
            events = list(mod.decide(snap2))
        orders = [e for e in events if isinstance(e, OrderEvent)]
        assert len(orders) == 2
        # First order: close (sell, qty=0)
        assert orders[0].side == "sell"
        assert orders[0].qty == Decimal("0")
        # Second order: open short (sell, qty>0)
        assert orders[1].side == "sell"
        assert orders[1].qty > 0
        # SignalEvent also emitted
        signals = [e for e in events if isinstance(e, SignalEvent)]
        assert len(signals) == 1
        assert signals[0].side == "short"

    def test_no_features_returns_empty(self):
        """Predictor returns None -> no events."""
        mod, _, _, _ = _make_module(pred_val=None)
        snap = _make_snapshot()
        events = list(mod.decide(snap))
        assert len(events) == 0

    def test_4h_runner_detection(self):
        """Runner key with '4h' sets _is_4h and correct ma_window."""
        mod, _, _, _ = _make_module(runner_key="BTCUSDT_4h")
        assert mod._is_4h is True
        assert mod._ma_window == 120

    def test_1h_runner_detection(self):
        """Runner key without '4h' uses 1h defaults."""
        mod, _, _, _ = _make_module(runner_key="BTCUSDT")
        assert mod._is_4h is False
        assert mod._ma_window == 480

    def test_consensus_updated_after_decide(self):
        """Consensus dict updated with own signal after decide."""
        mod, _, _, _ = _make_module(signal=1, z=1.5, runner_key="BTCUSDT")
        snap = _make_snapshot()
        list(mod.decide(snap))
        assert mod._consensus["BTCUSDT"] == 1

    def test_set_consensus(self):
        """set_consensus merges signals."""
        mod, _, _, _ = _make_module()
        mod.set_consensus({"BTCUSDT_4h": 1, "ETHUSDT": -1})
        assert mod._consensus["BTCUSDT_4h"] == 1
        assert mod._consensus["ETHUSDT"] == -1

    def test_close_order_has_zero_qty(self):
        """Close orders always have qty=0 (close all)."""
        mod, _, disc, _ = _make_module(signal=1, z=1.5)
        # Open
        snap1 = _make_snapshot(close=90000.0)
        list(mod.decide(snap1))

        # Close (signal -> 0)
        disc.discretize.return_value = (0, 0.0)
        snap2 = _make_snapshot(close=90000.0)
        events = list(mod.decide(snap2))
        orders = [e for e in events if isinstance(e, OrderEvent)]
        assert len(orders) == 1
        assert orders[0].qty == Decimal("0")
        assert orders[0].side == "sell"  # opposite of long

    def test_deadzone_base_preserved(self):
        """Vol-adaptive deadzone modifies discretizer but base is stored."""
        mod, _, disc, _ = _make_module(deadzone=0.9)
        assert mod._deadzone_base == 0.9
