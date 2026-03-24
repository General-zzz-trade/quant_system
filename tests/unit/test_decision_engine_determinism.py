from decimal import Decimal
from datetime import datetime, timezone

import pytest

from decision.engine import DecisionEngine
from decision.config import DecisionConfig
from decision.signals.technical.mean_reversion import MeanReversionSignal

from state import MarketState
from state import AccountState
from state import PositionState
from state import RiskState
from state.snapshot import StateSnapshot

_SCALE = 100_000_000


def _snap(*, o: str, c: str, pos_qty: str = "0", halted: bool = False) -> StateSnapshot:
    ts = datetime(2026, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
    ts_str = ts.isoformat()
    o_i = int(Decimal(o) * _SCALE)
    c_i = int(Decimal(c) * _SCALE)
    h_i = max(o_i, c_i)
    l_i = min(o_i, c_i)
    market = MarketState(symbol="BTCUSDT", last_price=c_i, open=o_i, high=h_i,
        low=l_i, close=c_i, volume=None, last_ts=ts_str)
    acct = AccountState(currency="USDT", balance=1000 * _SCALE, margin_used=0,
        margin_available=1000 * _SCALE, realized_pnl=0, unrealized_pnl=0,
            fees_paid=0, last_ts=ts_str)
    pos = PositionState(
        symbol="BTCUSDT", qty=int(Decimal(pos_qty) * _SCALE),
        avg_price=None, last_price=c_i, last_ts=ts_str)
    risk = RiskState(blocked=False, halted=halted, level="OK", message=None, flags=[],
        equity_peak="1000", drawdown_pct="0", last_ts=ts_str)
    return StateSnapshot(
        symbol="BTCUSDT",
        ts=ts,
        event_id="e0",
        event_type="market",
        bar_index=0,
        markets={"BTCUSDT": market},
        positions={"BTCUSDT": pos},
        account=acct,
        portfolio=None,
        risk=risk,
    )


def test_decision_engine_deterministic_output():
    cfg = DecisionConfig(symbols=["BTCUSDT"], max_positions=1, risk_fraction=Decimal("0.1"), min_notional=Decimal("5"))
    eng = DecisionEngine(cfg=cfg, signal_model=MeanReversionSignal())

    snap = _snap(o="100", c="90", pos_qty="0", halted=False)
    out1 = eng.run(snap).to_dict()
    out2 = eng.run(snap).to_dict()
    assert out1 == out2
    assert out1["strategy_id"] == cfg.strategy_id
    assert len(out1["orders"]) in (0, 1)


def test_decision_engine_respects_risk_halt():
    cfg = DecisionConfig(symbols=["BTCUSDT"], max_positions=1, risk_fraction=Decimal("0.1"))
    eng = DecisionEngine(cfg=cfg, signal_model=MeanReversionSignal())

    snap = _snap(o="100", c="90", halted=True)
    out = eng.run(snap)
    assert len(out.orders) == 0
    assert out.explain.gates["risk_overlay_allowed"] is False


def test_decision_engine_supports_rust_market_state():
    rust = pytest.importorskip("_quant_hotpath")
    ts = datetime(2026, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
    ts_str = ts.isoformat()
    snap = StateSnapshot(
        symbol="BTCUSDT",
        ts=ts,
        event_id="e1",
        event_type="market",
        bar_index=1,
        markets={
            "BTCUSDT": rust.RustMarketState(
                symbol="BTCUSDT",
                open=10_000_000_000,
                close=9_000_000_000,
                high=10_000_000_000,
                low=9_000_000_000,
                last_price=9_000_000_000,
                volume=1_000_000_000,
                last_ts=ts_str,
            )
        },
        positions={"BTCUSDT": PositionState(symbol="BTCUSDT", qty=0, avg_price=None,
            last_price=90 * _SCALE, last_ts=ts_str)},
        account=AccountState(currency="USDT", balance=1000 * _SCALE, margin_used=0,
            margin_available=1000 * _SCALE, realized_pnl=0, unrealized_pnl=0,
                fees_paid=0, last_ts=ts_str),
        portfolio=None,
        risk=RiskState(blocked=False, halted=False, level="OK", message=None, flags=[],
            equity_peak="1000", drawdown_pct="0", last_ts=ts_str),
    )
    cfg = DecisionConfig(symbols=["BTCUSDT"], max_positions=1, risk_fraction=Decimal("0.1"), min_notional=Decimal("5"))
    eng = DecisionEngine(cfg=cfg, signal_model=MeanReversionSignal())

    out = eng.run(snap)

    assert out.explain.orders is not None
