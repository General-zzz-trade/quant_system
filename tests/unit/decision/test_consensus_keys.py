"""Consensus keys are read exactly: "<key>_z" and "<key>_ic_ok" are not directions.

Regression for the 4h_reversal defect: the 1h module matched every consensus
key containing "4h", so the 4h runner's float z-score force-closed each 1h
position on the next bar; and the signal-only 4h runner never published a
direction because its (by design) zero size returned before ``_signal`` was set.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

from decision.modules.alpha import AlphaDecisionModule
from decision.sizing.adaptive import AdaptivePositionSizer
from event.types import OrderEvent, RiskEvent, SignalEvent
from shared.strategy_position import StrategyPosition


class _Predictor:
    _config: dict = {}

    def predict(self, features):
        return 0.001

    def update_online_ridge(self, realized_return):
        return False


class _Discretizer:
    """Returns a fixed (signal, z) on every bar."""

    def __init__(self, signal: int, z: float) -> None:
        self.signal, self.z = signal, z
        self.deadzone, self.min_hold, self.max_hold = 1.5, 3, 120

    def discretize(self, pred, bars, regime_ok, current_signal=0):
        return self.signal, self.z


def _snapshot(symbol: str, bar: int, close: float = 60000.0, equity: float = 5000.0):
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=bar)
    market = SimpleNamespace(close_f=close, high_f=close, low_f=close, close=close, high=close, low=close,
                             last_ts=None)
    return SimpleNamespace(ts=ts, markets={symbol: market}, features={"rsi_14": 50.0}, portfolio=None, risk=None,
                           account=SimpleNamespace(balance_f=equity, balance=equity))


def _module(runner_key: str, signal: int, z: float, *, signal_only: bool = False, symbol: str | None = None):
    symbol = symbol or runner_key.replace("_4h", "")
    return AlphaDecisionModule(
        symbol=symbol, runner_key=runner_key, predictor=_Predictor(), discretizer=_Discretizer(signal, z),
        sizer=AdaptivePositionSizer(runner_key=runner_key, step_size=0.001, min_size=0.001),
        leverage=1.0, signal_only=signal_only, venue="offline", offline=True)


def _confirmed_long(module: AlphaDecisionModule, price: str = "60000") -> None:
    module._pending_orders.clear()
    position = StrategyPosition()
    position.apply(SimpleNamespace(side="buy", qty=Decimal("0.01"), price=Decimal(price), fee=Decimal("0"),
                                   fee_asset="USDT"))
    module.sync_confirmed_position(position)


def _reasons(events) -> list[str]:
    return [e.rule_id for e in events if isinstance(e, RiskEvent)]


def test_four_hour_z_and_ic_keys_never_force_an_exit():
    consensus = {"BTCUSDT_4h": 0, "BTCUSDT_4h_z": 2.5, "BTCUSDT_4h_ic_ok": True}
    module = _module("BTCUSDT", signal=1, z=2.5)
    module.set_consensus(consensus)
    for bar in range(3):
        module.decide(_snapshot("BTCUSDT", bar))
    _confirmed_long(module)

    events = list(module.decide(_snapshot("BTCUSDT", 3)))

    assert not any(r.startswith("4h_reversal") for r in _reasons(events))
    # A same-direction scale-up may be emitted; nothing may close the long.
    assert not [e for e in events if isinstance(e, OrderEvent) and e.side == "sell"]
    assert module._signal == 1
    assert module._check_force_exits(60000.0, 2.5) == (False, "")


def test_opposing_four_hour_direction_still_forces_an_exit():
    module = _module("BTCUSDT", signal=1, z=2.5)
    module.set_consensus({"BTCUSDT_4h": -1, "BTCUSDT_4h_z": -2.0, "BTCUSDT_4h_ic_ok": True})
    for bar in range(3):
        module.decide(_snapshot("BTCUSDT", bar))
    _confirmed_long(module)

    assert module._check_force_exits(60000.0, 2.5) == (True, "4h_reversal(BTCUSDT_4h=-1)")


def test_signal_only_four_hour_runner_publishes_direction_without_size():
    consensus: dict = {}
    module = _module("BTCUSDT_4h", signal=1, z=2.5, signal_only=True)
    module.set_consensus(consensus)
    # The production sizer gives the signal-only key a zero cap: it never trades.
    assert module._sizer.target_qty(_snapshot("BTCUSDT", 0), "BTCUSDT", leverage=1.0) == 0

    events = list(module.decide(_snapshot("BTCUSDT", 0)))

    assert consensus["BTCUSDT_4h"] == 1
    assert module._signal == 1 and module._current_qty == 0
    assert any(isinstance(e, SignalEvent) for e in events)
    assert not [e for e in events if isinstance(e, OrderEvent)]


def test_eth_alignment_reads_only_the_btc_direction_key():
    # Insertion order puts non-direction BTC keys first; only "BTCUSDT" counts.
    blocked = _module("ETHUSDT", signal=1, z=2.5)
    blocked.set_consensus({"BTCUSDT_4h": 0, "BTCUSDT_z": 2.0, "BTCUSDT": -1})
    events = list(blocked.decide(_snapshot("ETHUSDT", 0, close=3000.0)))
    assert "direction_alignment" in _reasons(events)
    assert not [e for e in events if isinstance(e, OrderEvent)]

    allowed = _module("ETHUSDT", signal=1, z=2.5)
    allowed.set_consensus({"BTCUSDT_z": -2.0, "BTCUSDT_4h_z": -2.0})
    events = list(allowed.decide(_snapshot("ETHUSDT", 0, close=3000.0)))
    assert "direction_alignment" not in _reasons(events)
    assert [e for e in events if isinstance(e, OrderEvent)]
