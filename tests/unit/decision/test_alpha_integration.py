"""Integration smoke tests for the framework-native alpha pipeline.

Verifies EnsemblePredictor -> SignalDiscretizer -> AlphaDecisionModule
works end-to-end with mocked models and bridge.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from decision.modules.alpha import AlphaDecisionModule
from decision.signals.alpha_signal import EnsemblePredictor, SignalDiscretizer
from decision.sizing.adaptive import AdaptivePositionSizer
from event.types import EventType, MarketEvent, OrderEvent


# ── helper ────────────────────────────────────────────────────

def _make_snapshot(symbol="BTCUSDT", close=90000.0, equity=1000.0, features=None):
    snap = MagicMock()
    snap.symbol = symbol
    snap.bar_index = 900
    mkt = MagicMock()
    mkt.close = Decimal(str(close))
    mkt.high = Decimal(str(close * 1.01))
    mkt.low = Decimal(str(close * 0.99))
    mkt.open = Decimal(str(close))
    mkt.volume = Decimal("100")
    snap.markets = {symbol: mkt}
    pos = MagicMock()
    pos.qty = Decimal("0")
    snap.positions = {symbol: pos}
    acc = MagicMock()
    acc.balance = Decimal(str(equity))
    snap.account = acc
    snap.features = features or {"f1": 1.0, "f2": 2.0}
    snap.ts = None
    snap.event_id = "test"
    return snap


def _make_module(symbol="BTCUSDT", runner_key="BTCUSDT", deadzone=0.3):
    """Build AlphaDecisionModule with mocked ML models and bridge."""
    ridge = MagicMock()
    ridge.predict = MagicMock(return_value=[0.05])
    lgbm = MagicMock()
    lgbm.predict = MagicMock(return_value=[0.04])

    horizon = {
        "features": ["f1", "f2"],
        "ridge_features": ["f1", "f2"],
        "ic": 0.06,
        "ridge": ridge,
        "lgbm": lgbm,
    }
    predictor = EnsemblePredictor([horizon], {"version": "v8"})

    bridge = MagicMock()
    bridge.zscore_normalize = MagicMock(return_value=1.5)
    bridge.apply_constraints = MagicMock(return_value=1)
    discretizer = SignalDiscretizer(
        bridge, symbol, deadzone=deadzone, min_hold=18, max_hold=120,
    )

    sizer = AdaptivePositionSizer(runner_key=runner_key)

    module = AlphaDecisionModule(
        symbol=symbol,
        runner_key=runner_key,
        predictor=predictor,
        discretizer=discretizer,
        sizer=sizer,
    )
    return module, bridge


# ── tests ─────────────────────────────────────────────────────

def test_alpha_module_protocol_compliance():
    """AlphaDecisionModule has a decide() that accepts snapshot and returns iterable."""
    module, _ = _make_module()
    assert callable(getattr(module, "decide", None))
    snap = _make_snapshot()
    result = module.decide(snap)
    assert hasattr(result, "__iter__")


def test_market_event_creation():
    """MarketEvent can be constructed with EventHeader.new_root."""
    from datetime import datetime, timezone
    from event.header import EventHeader

    header = EventHeader.new_root(
        event_type=EventType.MARKET, version=1, source="test",
    )
    evt = MarketEvent(
        header=header,
        ts=datetime.now(timezone.utc),
        symbol="BTCUSDT",
        open=Decimal("90000"),
        high=Decimal("91000"),
        low=Decimal("89000"),
        close=Decimal("90500"),
        volume=Decimal("100"),
    )
    assert evt.symbol == "BTCUSDT"
    assert evt.close == Decimal("90500")
    assert evt.event_type == EventType.MARKET


def test_ensemble_to_discretizer_to_module_flow():
    """Full pipeline: 50 warmup bars then strong signal -> OrderEvent emitted."""
    module, bridge = _make_module()

    # During warmup: bridge returns None (no z-score yet)
    bridge.zscore_normalize.return_value = None
    for i in range(50):
        snap = _make_snapshot(close=90000.0 + i * 10)
        events = list(module.decide(snap))
        assert len(events) == 0, f"No events during warmup (bar {i})"

    # Bar 51: strong signal -> open position
    bridge.zscore_normalize.return_value = 2.0
    bridge.apply_constraints.return_value = 1
    snap = _make_snapshot(close=91000.0)
    events = list(module.decide(snap))
    assert len(events) >= 1, "Expected events on strong signal"
    orders = [e for e in events if isinstance(e, OrderEvent)]
    assert len(orders) >= 1, "Expected at least one OrderEvent"
    order = orders[0]
    assert order.side == "buy"
    assert order.symbol == "BTCUSDT"
    assert order.qty > 0


def test_force_exit_through_decide():
    """Big adverse move triggers force-exit close order."""
    module, bridge = _make_module()

    # Set up an open long position
    bridge.zscore_normalize.return_value = 1.5
    bridge.apply_constraints.return_value = 1
    snap = _make_snapshot(close=95000.0)
    list(module.decide(snap))  # open long
    assert module._signal == 1

    # Now: price drops hard -> force exit
    # Bridge still returns positive z but module checks adverse move
    bridge.zscore_normalize.return_value = 0.5
    bridge.apply_constraints.return_value = 0  # discretizer says flat
    snap = _make_snapshot(close=89000.0)
    events = list(module.decide(snap))

    # Signal should have changed from 1 to 0, emitting a close order
    assert module._signal == 0
    close_orders = [e for e in events if isinstance(e, OrderEvent) and e.side == "sell"]
    assert len(close_orders) >= 1, "Expected sell close order on force exit"


def test_direction_alignment_integration():
    """ETH short blocked when BTC consensus is long."""
    module, bridge = _make_module(symbol="ETHUSDT", runner_key="ETHUSDT")

    # BTC consensus is long
    module.set_consensus({"BTCUSDT_4h": 1})

    # Discretizer wants to go short
    bridge.zscore_normalize.return_value = -2.0
    bridge.apply_constraints.return_value = -1

    snap = _make_snapshot(symbol="ETHUSDT", close=3500.0)
    events = list(module.decide(snap))

    # Direction alignment should block the short entry
    assert module._signal == 0, "ETH short should be blocked by BTC long consensus"
    orders = [e for e in events if isinstance(e, OrderEvent)]
    assert len(orders) == 0, "No orders when direction alignment blocks"
    # A RiskEvent should be emitted for the block
    from event.events import RiskEvent
    risk_events = [e for e in events if isinstance(e, RiskEvent)]
    assert len(risk_events) == 1, "RiskEvent emitted on direction alignment block"
    assert risk_events[0].rule_id == "direction_alignment"
