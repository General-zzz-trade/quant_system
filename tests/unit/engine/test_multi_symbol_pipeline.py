# tests/unit/engine/test_multi_symbol_pipeline.py
"""Multi-symbol pipeline tests: state isolation, market routing, snapshot coverage."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from engine.pipeline import PipelineInput, StatePipeline
from state.account import AccountState
from state.market import MarketState
from state.position import PositionState


def _market_event(symbol: str, close: str, *, event_id: str = "m-1") -> SimpleNamespace:
    return SimpleNamespace(
        event_type="market",
        header=SimpleNamespace(event_type="market", ts=None, event_id=event_id),
        symbol=symbol,
        open=close, high=close, low=close, close=close,
        volume="100",
    )


def _fill_event(symbol: str, side: str, qty: float, price: float, *, event_id: str = "f-1") -> SimpleNamespace:
    return SimpleNamespace(
        event_type="fill",
        header=SimpleNamespace(event_type="fill", ts=None, event_id=event_id),
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        fee=0.0,
        realized_pnl=0.0,
        margin_change=0.0,
    )


def _make_input(event, *, markets=None, account=None, positions=None, idx=0):
    return PipelineInput(
        event=event,
        event_index=idx,
        symbol_default="BTCUSDT",
        markets=markets or {
            "BTCUSDT": MarketState.empty(symbol="BTCUSDT"),
            "ETHUSDT": MarketState.empty(symbol="ETHUSDT"),
        },
        account=account or AccountState.initial(currency="USDT", balance=Decimal("100000")),
        positions=positions or {},
    )


class TestMultiSymbolMarketIsolation:
    def test_btc_market_event_only_updates_btc(self):
        pipeline = StatePipeline()
        inp = _make_input(_market_event("BTCUSDT", "42000"))
        out = pipeline.apply(inp)

        assert out.advanced is True
        assert out.markets["BTCUSDT"].close == Decimal("42000")
        assert out.markets["ETHUSDT"].close is None  # unchanged

    def test_eth_market_event_only_updates_eth(self):
        pipeline = StatePipeline()
        inp = _make_input(_market_event("ETHUSDT", "3000"))
        out = pipeline.apply(inp)

        assert out.markets["ETHUSDT"].close == Decimal("3000")
        assert out.markets["BTCUSDT"].close is None  # unchanged

    def test_sequential_market_events_accumulate(self):
        pipeline = StatePipeline()
        inp1 = _make_input(_market_event("BTCUSDT", "42000", event_id="m-1"))
        out1 = pipeline.apply(inp1)

        inp2 = _make_input(
            _market_event("ETHUSDT", "3000", event_id="m-2"),
            markets=out1.markets, account=out1.account, positions=out1.positions, idx=out1.event_index,
        )
        out2 = pipeline.apply(inp2)

        assert out2.markets["BTCUSDT"].close == Decimal("42000")
        assert out2.markets["ETHUSDT"].close == Decimal("3000")
        assert out2.event_index == 2


class TestMultiSymbolFillIsolation:
    def test_btc_fill_does_not_affect_eth_position(self):
        pipeline = StatePipeline()
        inp = _make_input(_fill_event("BTCUSDT", "buy", 1.0, 42000.0))
        out = pipeline.apply(inp)

        assert "BTCUSDT" in out.positions
        btc_pos = out.positions["BTCUSDT"]
        assert btc_pos.qty != Decimal("0")

        # ETH position should not exist or be empty
        eth_pos = out.positions.get("ETHUSDT")
        assert eth_pos is None or eth_pos.qty == Decimal("0")

    def test_fills_for_different_symbols_track_separately(self):
        pipeline = StatePipeline()

        inp1 = _make_input(_fill_event("BTCUSDT", "buy", 1.0, 42000.0, event_id="f-1"))
        out1 = pipeline.apply(inp1)

        inp2 = _make_input(
            _fill_event("ETHUSDT", "buy", 10.0, 3000.0, event_id="f-2"),
            markets=out1.markets, account=out1.account, positions=out1.positions, idx=out1.event_index,
        )
        out2 = pipeline.apply(inp2)

        assert "BTCUSDT" in out2.positions
        assert "ETHUSDT" in out2.positions
        assert out2.positions["BTCUSDT"].qty != out2.positions["ETHUSDT"].qty


class TestMultiSymbolSnapshot:
    def test_snapshot_contains_all_markets(self):
        pipeline = StatePipeline()
        inp = _make_input(_market_event("BTCUSDT", "42000"))
        out = pipeline.apply(inp)

        assert out.snapshot is not None
        snap_markets = out.snapshot.get("markets") if isinstance(out.snapshot, dict) else getattr(out.snapshot, "markets", None)
        assert snap_markets is not None
        assert "BTCUSDT" in snap_markets
        assert "ETHUSDT" in snap_markets


class TestUnknownSymbolAutoCreation:
    def test_unknown_symbol_market_event_creates_state(self):
        pipeline = StatePipeline()
        inp = _make_input(
            _market_event("SOLUSDT", "150"),
            markets={"BTCUSDT": MarketState.empty(symbol="BTCUSDT")},
        )
        out = pipeline.apply(inp)

        assert "SOLUSDT" in out.markets
        assert out.markets["SOLUSDT"].close == Decimal("150")

    def test_unknown_symbol_fill_creates_position(self):
        pipeline = StatePipeline()
        inp = _make_input(
            _fill_event("SOLUSDT", "buy", 100.0, 150.0),
            markets={"BTCUSDT": MarketState.empty(symbol="BTCUSDT")},
        )
        out = pipeline.apply(inp)

        assert "SOLUSDT" in out.positions
        assert out.positions["SOLUSDT"].qty != Decimal("0")
