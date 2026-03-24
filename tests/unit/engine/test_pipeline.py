# tests/unit/engine/test_pipeline.py
"""StatePipeline unit tests — fact normalization, reducer chain, snapshot policy."""
from __future__ import annotations

import pytest
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Optional

from engine.pipeline import (
    FactNormalizationError,
    PipelineConfig,
    PipelineInput,
    StatePipeline,
    _detect_kind,
    normalize_to_facts,
)
from state import AccountState
from state import MarketState
from _quant_hotpath import RustStateStore

_SCALE = 100_000_000


def _make_store(symbols=("BTCUSDT",), balance="10000"):
    return RustStateStore(list(symbols), "USDT", int(Decimal(balance) * _SCALE))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Header:
    event_type: str = "fill"
    ts: Optional[str] = None
    event_id: Optional[str] = None


def _fill_event(*, side: str = "buy", qty: float = 1.0, price: float = 100.0,
                symbol: str = "BTCUSDT", fee: float = 0.0,
                event_id: str = "f-1") -> SimpleNamespace:
    return SimpleNamespace(
        event_type="fill",
        header=SimpleNamespace(event_type="fill", ts=None, event_id=event_id),
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        fee=fee,
        realized_pnl=0.0,
        margin_change=0.0,
    )


def _order_event(*, symbol: str = "BTCUSDT", event_id: str = "o-1") -> SimpleNamespace:
    return SimpleNamespace(
        event_type="order",
        header=SimpleNamespace(event_type="order", ts=None, event_id=event_id),
        symbol=symbol,
        side="buy",
        qty=1.0,
        price=100.0,
        order_id="ord-1",
        client_order_id="c-1",
        status="NEW",
        venue="binance",
        order_type="limit",
        tif="GTC",
        filled_qty=0.0,
        avg_price=0.0,
        order_key=None,
        payload_digest=None,
    )


def _market_event(*, symbol: str = "BTCUSDT") -> SimpleNamespace:
    return SimpleNamespace(
        event_type="market",
        header=SimpleNamespace(event_type="market", ts=None, event_id=None),
        symbol=symbol,
    )


def _signal_event() -> SimpleNamespace:
    return SimpleNamespace(
        event_type="signal",
        header=SimpleNamespace(event_type="signal", ts=None, event_id=None),
        symbol="BTCUSDT",
    )


def _make_pipeline_input(event: Any, event_index: int = 0) -> PipelineInput:
    return PipelineInput(
        event=event,
        event_index=event_index,
        symbol_default="BTCUSDT",
        markets={"BTCUSDT": MarketState.empty(symbol="BTCUSDT")},
        account=AccountState.initial(currency="USDT", balance=10000 * _SCALE),
        positions={},
    )


# ---------------------------------------------------------------------------
# Tests: _detect_kind
# ---------------------------------------------------------------------------

class TestDetectKind:
    def test_market(self) -> None:
        e = SimpleNamespace(event_type="market")
        assert _detect_kind(e) == "MARKET"

    def test_fill(self) -> None:
        e = SimpleNamespace(event_type="fill")
        assert _detect_kind(e) == "FILL"

    def test_order(self) -> None:
        e = SimpleNamespace(event_type="order")
        assert _detect_kind(e) == "ORDER"

    def test_signal(self) -> None:
        e = SimpleNamespace(event_type="signal")
        assert _detect_kind(e) == "SIGNAL"

    def test_intent(self) -> None:
        e = SimpleNamespace(event_type="intent")
        assert _detect_kind(e) == "INTENT"

    def test_risk(self) -> None:
        e = SimpleNamespace(event_type="risk")
        assert _detect_kind(e) == "RISK"

    def test_control(self) -> None:
        e = SimpleNamespace(event_type="control")
        assert _detect_kind(e) == "CONTROL"

    def test_unknown(self) -> None:
        e = SimpleNamespace(event_type="foobar")
        assert _detect_kind(e) == "UNKNOWN"

    def test_no_event_type(self) -> None:
        assert _detect_kind(object()) == "UNKNOWN"

    def test_event_type_string_fallback(self) -> None:
        """EVENT_TYPE string style (uppercase attr) also works."""
        e = SimpleNamespace(EVENT_TYPE="market_bar")
        assert _detect_kind(e) == "MARKET"

    def test_event_type_fill_string(self) -> None:
        e = SimpleNamespace(EVENT_TYPE="trade_fill")
        assert _detect_kind(e) == "FILL"




# ---------------------------------------------------------------------------
# Tests: normalize_to_facts
# ---------------------------------------------------------------------------

class TestNormalizeToFacts:
    def test_fill_event_normalized(self) -> None:
        e = _fill_event(side="buy", qty=2.0, price=50000.0)
        e.cash_delta = 12.5
        facts = normalize_to_facts(e)
        assert len(facts) == 1
        f = facts[0]
        assert f.event_type == "FILL"
        assert f.side == "buy"
        assert f.qty == 2.0
        assert f.price == 50000.0
        assert f.cash_delta == 12.5

    def test_fill_sell_side(self) -> None:
        e = _fill_event(side="sell", qty=1.0)
        facts = normalize_to_facts(e)
        assert facts[0].side == "sell"

    def test_fill_long_normalized_to_buy(self) -> None:
        e = _fill_event(side="long", qty=1.0)
        facts = normalize_to_facts(e)
        assert facts[0].side == "buy"

    def test_fill_short_normalized_to_sell(self) -> None:
        e = _fill_event(side="short", qty=1.0)
        facts = normalize_to_facts(e)
        assert facts[0].side == "sell"

    def test_fill_missing_side_raises(self) -> None:
        e = SimpleNamespace(
            event_type="fill",
            header=SimpleNamespace(event_type="fill", ts=None, event_id=None),
            symbol="BTCUSDT",
            qty=1.0,
            side=None,
        )
        with pytest.raises(FactNormalizationError, match="side"):
            normalize_to_facts(e)

    def test_fill_unsupported_side_raises(self) -> None:
        e = _fill_event(side="unknown_side")
        with pytest.raises(FactNormalizationError, match="不支持"):
            normalize_to_facts(e)

    def test_fill_qty_absolute(self) -> None:
        """Negative qty is normalized to absolute."""
        e = _fill_event(side="buy", qty=-5.0)
        facts = normalize_to_facts(e)
        assert facts[0].qty == 5.0

    def test_order_event_normalized(self) -> None:
        e = _order_event()
        facts = normalize_to_facts(e)
        assert len(facts) == 1
        assert facts[0].event_type == "ORDER_UPDATE"

    def test_signal_event_returns_empty(self) -> None:
        e = _signal_event()
        facts = normalize_to_facts(e)
        assert facts == []

    def test_market_event_produces_fact(self) -> None:
        e = SimpleNamespace(
            event_type="market",
            header=SimpleNamespace(event_type="market", ts=None, event_id=None),
            symbol="BTCUSDT",
            open="42000", high="43000", low="41000", close="42500", volume="100",
        )
        facts = normalize_to_facts(e)
        assert len(facts) == 1
        assert facts[0].event_type == "market"
        assert facts[0].symbol == "BTCUSDT"
        assert facts[0].close == "42500"

    def test_funding_event_produces_fact(self) -> None:
        e = SimpleNamespace(
            event_type="funding",
            header=SimpleNamespace(event_type="funding", ts=None, event_id=None),
            symbol="BTCUSDT",
            funding_rate="0.0001",
            mark_price="42000",
            position_qty="0.5",
        )
        facts = normalize_to_facts(e)
        assert len(facts) == 1
        assert facts[0].event_type == "funding"
        assert facts[0].funding_rate == "0.0001"
        assert facts[0].position_qty == "0.5"


# ---------------------------------------------------------------------------
# Tests: StatePipeline.apply
# ---------------------------------------------------------------------------

class TestPipelineApply:
    def test_non_fact_event_not_advanced(self) -> None:
        pipeline = StatePipeline(store=_make_store())
        inp = _make_pipeline_input(_signal_event(), event_index=5)
        out = pipeline.apply(inp)
        assert out.advanced is False
        assert out.event_index == 5
        assert out.snapshot is None

    def test_fill_event_advances_index(self) -> None:
        pipeline = StatePipeline(store=_make_store())
        inp = _make_pipeline_input(_fill_event(), event_index=0)
        out = pipeline.apply(inp)
        assert out.advanced is True
        assert out.event_index == 1

    def test_fill_updates_position(self) -> None:
        pipeline = StatePipeline(store=_make_store())
        inp = _make_pipeline_input(_fill_event(side="buy", qty=1.0, price=50000.0))
        out = pipeline.apply(inp)
        assert "BTCUSDT" in out.positions
        pos = out.positions["BTCUSDT"]
        assert pos.qty != Decimal("0") or pos.qty != 0  # position should have changed

    def test_order_event_advances_index(self) -> None:
        pipeline = StatePipeline(store=_make_store())
        inp = _make_pipeline_input(_order_event(), event_index=0)
        out = pipeline.apply(inp)
        assert out.advanced is True
        assert out.event_index == 1

    def test_snapshot_generated_on_change(self) -> None:
        pipeline = StatePipeline(store=_make_store(), config=PipelineConfig(build_snapshot_on_change_only=True))
        inp = _make_pipeline_input(_fill_event(side="buy", qty=1.0, price=50000.0))
        out = pipeline.apply(inp)
        # Fill should change state → snapshot generated
        assert out.snapshot is not None

    def test_snapshot_always_generated_when_config_off(self) -> None:
        pipeline = StatePipeline(store=_make_store(), config=PipelineConfig(build_snapshot_on_change_only=False))
        inp = _make_pipeline_input(_order_event())
        out = pipeline.apply(inp)
        # Even if no state change, snapshot should be generated
        assert out.snapshot is not None

    def test_new_symbol_creates_empty_position(self) -> None:
        pipeline = StatePipeline(store=_make_store())
        inp = _make_pipeline_input(_fill_event(symbol="ETHUSDT", side="buy", qty=2.0, price=3000.0))
        out = pipeline.apply(inp)
        assert "ETHUSDT" in out.positions

    def test_last_event_id_tracked(self) -> None:
        pipeline = StatePipeline(store=_make_store())
        inp = _make_pipeline_input(_fill_event(event_id="test-evt-99"))
        out = pipeline.apply(inp)
        assert out.last_event_id == "test-evt-99"

    def test_multiple_facts_in_sequence(self) -> None:
        """Multiple applies accumulate state correctly."""
        pipeline = StatePipeline(store=_make_store())
        inp1 = _make_pipeline_input(_fill_event(side="buy", qty=1.0, price=100.0), event_index=0)
        out1 = pipeline.apply(inp1)
        assert out1.event_index == 1

        inp2 = PipelineInput(
            event=_fill_event(side="buy", qty=2.0, price=200.0, event_id="f-2"),
            event_index=out1.event_index,
            symbol_default="BTCUSDT",
            markets=out1.markets,
            account=out1.account,
            positions=out1.positions,
        )
        out2 = pipeline.apply(inp2)
        assert out2.event_index == 2

    def test_market_event_advances_pipeline(self) -> None:
        """MARKET events produce facts and advance pipeline, updating MarketState."""
        pipeline = StatePipeline(store=_make_store())
        ev = SimpleNamespace(
            event_type="market",
            header=SimpleNamespace(event_type="market", ts=None, event_id="m-1"),
            symbol="BTCUSDT",
            open=Decimal("42000"), high=Decimal("43000"),
            low=Decimal("41000"), close=Decimal("42500"),
            volume=Decimal("100"),
        )
        inp = _make_pipeline_input(ev, event_index=0)
        out = pipeline.apply(inp)
        assert out.advanced is True
        assert out.event_index == 1
        # Pipeline now returns Rust types — use float accessor
        close_f = getattr(out.market, "close_f", None) or float(out.market.close)
        assert close_f == pytest.approx(42500.0)
