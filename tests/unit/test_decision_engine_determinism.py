from decimal import Decimal
from datetime import datetime, timezone

import pytest

from decision.engine import DecisionEngine
from decision.config import DecisionConfig
from decision.signals.technical.mean_reversion import MeanReversionSignal

from state.market import MarketState
from state.account import AccountState
from state.position import PositionState
from state.risk import RiskState
from state.snapshot import StateSnapshot


def _snap(*, o: str, c: str, pos_qty: str = "0", halted: bool = False) -> StateSnapshot:
    ts = datetime(2026, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
    market = MarketState(symbol="BTCUSDT", last_price=Decimal(c), open=Decimal(o), high=Decimal(max(o,c)), low=Decimal(min(o,c)), close=Decimal(c), volume=None, last_ts=ts)
    acct = AccountState(currency="USDT", balance=Decimal("1000"), margin_used=Decimal("0"), margin_available=Decimal("1000"), realized_pnl=Decimal("0"), unrealized_pnl=Decimal("0"), fees_paid=Decimal("0"), last_ts=ts)
    pos = PositionState(symbol="BTCUSDT", qty=Decimal(pos_qty), avg_price=None, last_price=Decimal(c), last_ts=ts)
    risk = RiskState(blocked=False, halted=halted, level="OK", message=None, flags=(), equity_peak=Decimal("1000"), drawdown_pct=Decimal("0"), last_ts=ts)
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
                last_ts=ts.isoformat(),
            )
        },
        positions={"BTCUSDT": PositionState(symbol="BTCUSDT", qty=Decimal("0"), avg_price=None, last_price=Decimal("90"), last_ts=ts)},
        account=AccountState(currency="USDT", balance=Decimal("1000"), margin_used=Decimal("0"), margin_available=Decimal("1000"), realized_pnl=Decimal("0"), unrealized_pnl=Decimal("0"), fees_paid=Decimal("0"), last_ts=ts),
        portfolio=None,
        risk=RiskState(blocked=False, halted=False, level="OK", message=None, flags=(), equity_peak=Decimal("1000"), drawdown_pct=Decimal("0"), last_ts=ts),
    )
    cfg = DecisionConfig(symbols=["BTCUSDT"], max_positions=1, risk_fraction=Decimal("0.1"), min_notional=Decimal("5"))
    eng = DecisionEngine(cfg=cfg, signal_model=MeanReversionSignal())

    out = eng.run(snap)

    assert out.explain.orders is not None
