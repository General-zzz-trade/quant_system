from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from engine import pipeline as pipeline_mod
from engine.pipeline import PipelineInput, StatePipeline
from state.account import AccountState
from state.market import MarketState
from state.position import PositionState
from state.reducers.account import AccountReducer
from state.reducers.market import MarketReducer
from state.reducers.position import PositionReducer


pytest.importorskip("_quant_hotpath")


def _make_input(event, *, markets=None, account=None, positions=None, idx=0) -> PipelineInput:
    return PipelineInput(
        event=event,
        event_index=idx,
        symbol_default="BTCUSDT",
        markets=markets or {"BTCUSDT": MarketState.empty(symbol="BTCUSDT")},
        account=account or AccountState.initial(currency="USDT", balance=Decimal("10000")),
        positions=positions or {},
    )


def test_pipeline_defaults_to_rust_state_reducers() -> None:

    pipeline = StatePipeline()

    assert type(pipeline._mr).__name__ == "RustMarketReducerAdapter"
    assert type(pipeline._ar).__name__ == "RustAccountReducerAdapter"
    assert type(pipeline._pr).__name__ == "RustPositionReducerAdapter"


def test_rust_pipeline_matches_python_pipeline_on_state_progression() -> None:

    rust_pipeline = StatePipeline()
    py_pipeline = StatePipeline(
        market_reducer=MarketReducer(),
        account_reducer=AccountReducer(),
        position_reducer=PositionReducer(),
    )

    events = [
        SimpleNamespace(
            event_type="market",
            header=SimpleNamespace(event_type="market", ts=None, event_id="m-1"),
            symbol="BTCUSDT",
            open="99",
            high="101",
            low="98",
            close="100",
            volume="2",
        ),
        SimpleNamespace(
            event_type="fill",
            header=SimpleNamespace(event_type="fill", ts=None, event_id="f-1"),
            symbol="BTCUSDT",
            side="buy",
            qty="2",
            price="100",
            fee="1",
            realized_pnl="5",
            margin_change="10",
            cash_delta="0",
        ),
        SimpleNamespace(
            event_type="funding",
            header=SimpleNamespace(event_type="funding", ts=None, event_id="u-1"),
            symbol="BTCUSDT",
            funding_rate="0.0001",
            mark_price="40000",
            position_qty="2",
        ),
        SimpleNamespace(
            event_type="fill",
            header=SimpleNamespace(event_type="fill", ts=None, event_id="f-2"),
            symbol="BTCUSDT",
            side="sell",
            qty="1",
            price="110",
            fee="0.5",
            realized_pnl="10",
            margin_change="-5",
            cash_delta="0",
        ),
    ]

    rust_out = None
    py_out = None
    for idx, event in enumerate(events):
        rust_out = rust_pipeline.apply(
            _make_input(
                event,
                markets=rust_out.markets if rust_out else None,
                account=rust_out.account if rust_out else None,
                positions=rust_out.positions if rust_out else None,
                idx=rust_out.event_index if rust_out else idx,
            )
        )
        py_out = py_pipeline.apply(
            _make_input(
                event,
                markets=py_out.markets if py_out else None,
                account=py_out.account if py_out else None,
                positions=py_out.positions if py_out else None,
                idx=py_out.event_index if py_out else idx,
            )
        )

    assert rust_out is not None
    assert py_out is not None
    # Rust pipeline returns Rust types, Python pipeline returns Python types
    # Compare by value rather than by object equality
    def _val_f(obj, attr):
        f_attr = attr + "_f"
        v = getattr(obj, f_attr, None)
        if v is not None:
            return v
        v = getattr(obj, attr, None)
        return float(v) if v is not None else None

    for sym in rust_out.markets:
        assert _val_f(rust_out.markets[sym], "close") == pytest.approx(
            _val_f(py_out.markets[sym], "close"), abs=1e-6)
    assert _val_f(rust_out.account, "balance") == pytest.approx(
        _val_f(py_out.account, "balance"), abs=1e-6)
    for sym in rust_out.positions:
        assert _val_f(rust_out.positions[sym], "qty") == pytest.approx(
            _val_f(py_out.positions[sym], "qty"), abs=1e-6)
    assert rust_out.event_index == py_out.event_index
