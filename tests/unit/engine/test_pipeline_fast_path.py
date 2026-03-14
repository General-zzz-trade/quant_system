"""Tests for the RustStateStore path in StatePipeline.apply()."""
from __future__ import annotations

import time
import pytest
from decimal import Decimal
from types import SimpleNamespace

from engine.pipeline import (
    PipelineConfig,
    PipelineInput,
    StatePipeline,
)
from state.account import AccountState
from state.market import MarketState
from _quant_hotpath import RustStateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCALE = 100_000_000


def _val_f(obj, attr):
    """Get float value from Rust or Python state type."""
    f_attr = attr + "_f"
    v = getattr(obj, f_attr, None)
    if v is not None:
        return v
    v = getattr(obj, attr, None)
    return float(v) if v is not None else None


def _market_event(*, symbol="BTCUSDT", close=42500.0, open=42000.0,
                  high=43000.0, low=41000.0, volume=100.0, ts=None):
    return SimpleNamespace(
        event_type="market",
        header=SimpleNamespace(event_type="market", ts=ts, event_id="m-1"),
        symbol=symbol,
        open=Decimal(str(open)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=Decimal(str(volume)),
        ts=ts,
    )


def _fill_event(*, side="buy", qty=1.0, price=50000.0, symbol="BTCUSDT",
                fee=0.5, event_id="f-1"):
    return SimpleNamespace(
        event_type="fill",
        header=SimpleNamespace(event_type="fill", ts=None, event_id=event_id),
        symbol=symbol,
        side=side,
        qty=qty,
        quantity=qty,
        price=price,
        fee=fee,
        realized_pnl=0.0,
        margin_change=0.0,
        cash_delta=0.0,
    )


def _funding_event(*, symbol="BTCUSDT", funding_rate=0.0001, mark_price=42000.0,
                   position_qty=0.5):
    return SimpleNamespace(
        event_type="funding",
        header=SimpleNamespace(event_type="funding", ts=None, event_id="fund-1"),
        symbol=symbol,
        funding_rate=funding_rate,
        mark_price=mark_price,
        position_qty=position_qty,
        ts=None,
    )


def _signal_event():
    return SimpleNamespace(
        event_type="signal",
        header=SimpleNamespace(event_type="signal", ts=None, event_id=None),
        symbol="BTCUSDT",
    )


def _make_input(event, event_index=0, account=None, markets=None, positions=None):
    return PipelineInput(
        event=event,
        event_index=event_index,
        symbol_default="BTCUSDT",
        markets=markets or {"BTCUSDT": MarketState.empty(symbol="BTCUSDT")},
        account=account or AccountState.initial(currency="USDT", balance=Decimal("10000")),
        positions=positions or {},
    )


def _make_store():
    return RustStateStore(["BTCUSDT"], "USDT", int(10000 * _SCALE))


# ---------------------------------------------------------------------------
# Tests: Store path behavior
# ---------------------------------------------------------------------------

class TestStorePath:
    def test_signal_event_not_advanced(self):
        pipe = StatePipeline(store=_make_store())
        inp = _make_input(_signal_event(), event_index=5)
        out = pipe.apply(inp)
        assert out.advanced is False
        assert out.event_index == 5
        assert out.snapshot is None

    def test_market_event_advances(self):
        pipe = StatePipeline(store=_make_store())
        ev = _market_event(close=42500.0)
        inp = _make_input(ev)
        out = pipe.apply(inp)

        assert out.advanced is True
        assert out.event_index == 1
        assert _val_f(out.market, "close") == pytest.approx(42500.0)

    def test_fill_event_updates_state(self):
        pipe = StatePipeline(store=_make_store())
        ev = _fill_event(side="buy", qty=0.5, price=50000.0, fee=1.25)
        inp = _make_input(ev)
        out = pipe.apply(inp)

        assert out.advanced is True
        assert out.event_index == 1

        pos = out.positions["BTCUSDT"]
        assert _val_f(pos, "qty") == pytest.approx(0.5)
        assert _val_f(pos, "avg_price") == pytest.approx(50000.0)
        assert _val_f(out.account, "fees_paid") == pytest.approx(1.25)

    def test_funding_event_advances(self):
        pipe = StatePipeline(store=_make_store())
        ev = _funding_event(funding_rate=0.0001, mark_price=42000.0, position_qty=0.5)
        inp = _make_input(ev)
        out = pipe.apply(inp)

        assert out.advanced is True

    def test_new_symbol_creates_position(self):
        store = RustStateStore(["BTCUSDT", "ETHUSDT"], "USDT", int(10000 * _SCALE))
        pipe = StatePipeline(store=store)
        ev = _fill_event(symbol="ETHUSDT", side="buy", qty=2.0, price=3000.0)
        inp = _make_input(ev)
        out = pipe.apply(inp)
        assert "ETHUSDT" in out.positions

    def test_snapshot_config_respected(self):
        store = _make_store()
        pipe = StatePipeline(
            store=store,
            config=PipelineConfig(build_snapshot_on_change_only=False),
        )
        ev = SimpleNamespace(
            event_type="order",
            header=SimpleNamespace(event_type="order", ts=None, event_id="o-1"),
            symbol="BTCUSDT",
            side="buy", qty=1.0, price=100.0, order_id="ord-1",
            client_order_id="c-1", status="NEW", venue="binance",
            order_type="limit", tif="GTC", filled_qty=0.0,
            avg_price=0.0, order_key=None, payload_digest=None,
        )
        inp = _make_input(ev)
        out = pipe.apply(inp)
        assert out.advanced is True
        assert out.snapshot is not None


# ---------------------------------------------------------------------------
# Benchmark: Store path throughput
# ---------------------------------------------------------------------------

class TestStorePathBenchmark:
    @pytest.mark.benchmark
    def test_store_throughput(self):
        store = _make_store()
        pipe = StatePipeline(store=store)

        N = 1000
        events = []
        for i in range(N):
            if i % 3 == 0:
                events.append(_market_event(close=50000.0 + i))
            elif i % 3 == 1:
                events.append(_fill_event(side="buy" if i % 2 == 0 else "sell",
                                          qty=0.01, price=50000.0 + i,
                                          fee=0.01, event_id=f"f-{i}"))
            else:
                events.append(_signal_event())

        # Warm up
        for ev in events[:10]:
            inp = _make_input(ev)
            pipe.apply(inp)

        # Benchmark
        t0 = time.perf_counter()
        for ev in events:
            inp = _make_input(ev)
            pipe.apply(inp)
        store_ms = (time.perf_counter() - t0) * 1000

        print(f"\nStore path: {store_ms:.2f}ms ({N} events, "
              f"{store_ms/N*1000:.1f}us/event)")

        assert store_ms < 5000, f"Store path too slow: {store_ms:.2f}ms for {N} events"
