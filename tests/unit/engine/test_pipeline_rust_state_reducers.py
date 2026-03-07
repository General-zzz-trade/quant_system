"""Tests: StatePipeline uses RustStateStore for all state management."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from engine.pipeline import PipelineInput, StatePipeline
from state.account import AccountState
from state.market import MarketState
from _quant_hotpath import RustStateStore

pytest.importorskip("_quant_hotpath")

_SCALE = 100_000_000


def _make_store(symbols=("BTCUSDT",), balance=10000):
    return RustStateStore(list(symbols), "USDT", int(balance * _SCALE))


def _make_input(event, *, markets=None, account=None, positions=None, idx=0) -> PipelineInput:
    return PipelineInput(
        event=event,
        event_index=idx,
        symbol_default="BTCUSDT",
        markets=markets or {"BTCUSDT": MarketState.empty(symbol="BTCUSDT")},
        account=account or AccountState.initial(currency="USDT", balance=Decimal("10000")),
        positions=positions or {},
    )


def test_pipeline_requires_store() -> None:
    with pytest.raises(ValueError, match="RustStateStore"):
        StatePipeline()


def test_pipeline_uses_rust_state_store() -> None:
    store = _make_store()
    pipeline = StatePipeline(store=store)
    assert pipeline._store is store


def test_rust_pipeline_state_progression() -> None:
    store = _make_store()
    pipeline = StatePipeline(store=store)

    events = [
        SimpleNamespace(
            event_type="market",
            header=SimpleNamespace(event_type="market", ts=None, event_id="m-1"),
            symbol="BTCUSDT",
            open="99", high="101", low="98", close="100", volume="2",
        ),
        SimpleNamespace(
            event_type="fill",
            header=SimpleNamespace(event_type="fill", ts=None, event_id="f-1"),
            symbol="BTCUSDT",
            side="buy", qty="2", price="100", fee="1",
            realized_pnl="5", margin_change="10", cash_delta="0",
        ),
        SimpleNamespace(
            event_type="funding",
            header=SimpleNamespace(event_type="funding", ts=None, event_id="u-1"),
            symbol="BTCUSDT",
            funding_rate="0.0001", mark_price="40000", position_qty="2",
        ),
        SimpleNamespace(
            event_type="fill",
            header=SimpleNamespace(event_type="fill", ts=None, event_id="f-2"),
            symbol="BTCUSDT",
            side="sell", qty="1", price="110", fee="0.5",
            realized_pnl="10", margin_change="-5", cash_delta="0",
        ),
    ]

    out = None
    for event in events:
        out = pipeline.apply(
            _make_input(
                event,
                markets=out.markets if out else None,
                account=out.account if out else None,
                positions=out.positions if out else None,
                idx=out.event_index if out else 0,
            )
        )

    assert out is not None
    assert out.event_index == 4
    assert out.portfolio is not None
    assert out.risk is not None
    assert out.risk.blocked is False

    def _val_f(obj, attr):
        f_attr = attr + "_f"
        v = getattr(obj, f_attr, None)
        if v is not None:
            return v
        v = getattr(obj, attr, None)
        return float(v) if v is not None else None

    assert _val_f(out.markets["BTCUSDT"], "close") == pytest.approx(100.0)
    assert "BTCUSDT" in out.positions
    assert _val_f(out.positions["BTCUSDT"], "qty") == pytest.approx(1.0)


def test_rust_pipeline_snapshot_contains_derived_state() -> None:
    pipeline = StatePipeline(store=_make_store())
    event = SimpleNamespace(
        event_type="market",
        header=SimpleNamespace(event_type="market", ts=None, event_id="m-1"),
        symbol="BTCUSDT",
        open="99", high="101", low="98", close="100", volume="2",
    )

    out = pipeline.apply(_make_input(event))

    assert out.snapshot is not None
    assert out.snapshot.portfolio is not None
    assert out.snapshot.risk is not None
    assert out.snapshot.portfolio.total_equity == Decimal("10000")
    assert out.snapshot.risk.blocked is False
