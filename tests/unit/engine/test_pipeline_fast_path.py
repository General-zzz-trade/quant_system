# tests/unit/engine/test_pipeline_fast_path.py
"""Tests for the RustStateStore path in StatePipeline.apply()."""
from __future__ import annotations

import time
import pytest
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

from engine.pipeline import (
    PipelineConfig,
    PipelineInput,
    PipelineOutput,
    StatePipeline,
)
from state.account import AccountState
from state.market import MarketState
from state.position import PositionState
from state.reducers.market import MarketReducer
from state.reducers.account import AccountReducer
from state.reducers.position import PositionReducer
from state.rust_adapters import (
    RustMarketReducerAdapter,
    RustAccountReducerAdapter,
    RustPositionReducerAdapter,
)
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
# Tests: Store path behavior parity
# ---------------------------------------------------------------------------

class TestStorePathParity:
    """Compare store path (Rust heap) vs adapter path for behavioral parity."""

    def _make_both_pipelines(self):
        store = _make_store()
        store_pipe = StatePipeline(store=store)
        adapter_pipe = StatePipeline(
            market_reducer=RustMarketReducerAdapter(),
            account_reducer=RustAccountReducerAdapter(),
            position_reducer=RustPositionReducerAdapter(),
        )
        return store_pipe, adapter_pipe

    def test_signal_event_parity(self):
        store_pipe, adapter_pipe = self._make_both_pipelines()
        inp = _make_input(_signal_event(), event_index=5)
        out_store = store_pipe.apply(inp)
        out_adapter = adapter_pipe.apply(inp)
        assert out_store.advanced == out_adapter.advanced == False
        assert out_store.event_index == out_adapter.event_index == 5
        assert out_store.snapshot is None
        assert out_adapter.snapshot is None

    def test_market_event_parity(self):
        store_pipe, adapter_pipe = self._make_both_pipelines()
        ev = _market_event(close=42500.0)
        inp = _make_input(ev)
        out_store = store_pipe.apply(inp)
        out_adapter = adapter_pipe.apply(inp)

        assert out_store.advanced == out_adapter.advanced == True
        assert out_store.event_index == out_adapter.event_index == 1
        assert _val_f(out_store.market, "close") == pytest.approx(_val_f(out_adapter.market, "close"))
        assert _val_f(out_store.market, "last_price") == pytest.approx(_val_f(out_adapter.market, "last_price"))

    def test_fill_event_parity(self):
        store_pipe, adapter_pipe = self._make_both_pipelines()
        ev = _fill_event(side="buy", qty=0.5, price=50000.0, fee=1.25)
        inp = _make_input(ev)
        out_store = store_pipe.apply(inp)
        out_adapter = adapter_pipe.apply(inp)

        assert out_store.advanced == out_adapter.advanced == True
        assert out_store.event_index == out_adapter.event_index == 1

        pos_s = out_store.positions["BTCUSDT"]
        pos_a = out_adapter.positions["BTCUSDT"]
        assert _val_f(pos_s, "qty") == pytest.approx(_val_f(pos_a, "qty"))
        assert _val_f(pos_s, "avg_price") == pytest.approx(_val_f(pos_a, "avg_price"))

        assert _val_f(out_store.account, "balance") == pytest.approx(_val_f(out_adapter.account, "balance"))
        assert _val_f(out_store.account, "fees_paid") == pytest.approx(_val_f(out_adapter.account, "fees_paid"))

    def test_funding_event_parity(self):
        store_pipe, adapter_pipe = self._make_both_pipelines()
        ev = _funding_event(funding_rate=0.0001, mark_price=42000.0, position_qty=0.5)
        inp = _make_input(ev)
        out_store = store_pipe.apply(inp)
        out_adapter = adapter_pipe.apply(inp)

        assert out_store.advanced == out_adapter.advanced == True
        assert _val_f(out_store.account, "balance") == pytest.approx(_val_f(out_adapter.account, "balance"))

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
# Benchmark: Store path vs adapter path
# ---------------------------------------------------------------------------

class TestStorePathBenchmark:
    @pytest.mark.benchmark
    def test_store_vs_adapter_benchmark(self):
        """Benchmark: Store path should be faster than adapter path."""
        store = _make_store()
        store_pipe = StatePipeline(store=store)
        adapter_pipe = StatePipeline(
            market_reducer=RustMarketReducerAdapter(),
            account_reducer=RustAccountReducerAdapter(),
            position_reducer=RustPositionReducerAdapter(),
        )

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
            store_pipe.apply(inp)
            adapter_pipe.apply(inp)

        # Benchmark store path
        t0 = time.perf_counter()
        for ev in events:
            inp = _make_input(ev)
            store_pipe.apply(inp)
        store_ms = (time.perf_counter() - t0) * 1000

        # Benchmark adapter path
        t0 = time.perf_counter()
        for ev in events:
            inp = _make_input(ev)
            adapter_pipe.apply(inp)
        adapter_ms = (time.perf_counter() - t0) * 1000

        speedup = adapter_ms / store_ms if store_ms > 0 else float("inf")
        print(f"\nStore path: {store_ms:.2f}ms, Adapter path: {adapter_ms:.2f}ms, "
              f"Speedup: {speedup:.2f}x ({N} events)")

        assert store_ms < adapter_ms * 1.5, (
            f"Store path ({store_ms:.2f}ms) should not be much slower than "
            f"adapter path ({adapter_ms:.2f}ms)"
        )
