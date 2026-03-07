# tests/unit/engine/test_pipeline_fast_path.py
"""Tests for the Rust fast path in StatePipeline.apply()."""
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


# ---------------------------------------------------------------------------
# Tests: Fast path behavior parity
# ---------------------------------------------------------------------------

class TestFastPathParity:
    """Compare fast path (Rust) vs slow path (Python) for behavioral parity."""

    def _make_both_pipelines(self):
        fast = StatePipeline()
        assert fast._use_rust_fast_path
        # Slow path: explicit Rust adapters → same f64 math, but goes through
        # normalize_to_facts + per-reducer adapter overhead
        slow = StatePipeline(
            market_reducer=RustMarketReducerAdapter(),
            account_reducer=RustAccountReducerAdapter(),
            position_reducer=RustPositionReducerAdapter(),
        )
        assert not slow._use_rust_fast_path
        return fast, slow

    def test_signal_event_parity(self):
        fast, slow = self._make_both_pipelines()
        inp = _make_input(_signal_event(), event_index=5)
        out_fast = fast.apply(inp)
        out_slow = slow.apply(inp)
        assert out_fast.advanced == out_slow.advanced == False
        assert out_fast.event_index == out_slow.event_index == 5
        assert out_fast.snapshot is None
        assert out_slow.snapshot is None

    def test_market_event_parity(self):
        fast, slow = self._make_both_pipelines()
        ev = _market_event(close=42500.0)
        inp = _make_input(ev)
        out_fast = fast.apply(inp)
        out_slow = slow.apply(inp)

        assert out_fast.advanced == out_slow.advanced == True
        assert out_fast.event_index == out_slow.event_index == 1
        # Fast returns Rust types, slow returns Python — compare via float
        assert _val_f(out_fast.market, "close") == pytest.approx(_val_f(out_slow.market, "close"))
        assert _val_f(out_fast.market, "last_price") == pytest.approx(_val_f(out_slow.market, "last_price"))

    def test_fill_event_parity(self):
        fast, slow = self._make_both_pipelines()
        ev = _fill_event(side="buy", qty=0.5, price=50000.0, fee=1.25)
        inp = _make_input(ev)
        out_fast = fast.apply(inp)
        out_slow = slow.apply(inp)

        assert out_fast.advanced == out_slow.advanced == True
        assert out_fast.event_index == out_slow.event_index == 1

        # Position parity (compare via float)
        pos_f = out_fast.positions["BTCUSDT"]
        pos_s = out_slow.positions["BTCUSDT"]
        assert _val_f(pos_f, "qty") == pytest.approx(_val_f(pos_s, "qty"))
        assert _val_f(pos_f, "avg_price") == pytest.approx(_val_f(pos_s, "avg_price"))

        # Account parity
        assert _val_f(out_fast.account, "balance") == pytest.approx(_val_f(out_slow.account, "balance"))
        assert _val_f(out_fast.account, "fees_paid") == pytest.approx(_val_f(out_slow.account, "fees_paid"))

    def test_funding_event_parity(self):
        fast, slow = self._make_both_pipelines()
        ev = _funding_event(funding_rate=0.0001, mark_price=42000.0, position_qty=0.5)
        inp = _make_input(ev)
        out_fast = fast.apply(inp)
        out_slow = slow.apply(inp)

        assert out_fast.advanced == out_slow.advanced == True
        assert _val_f(out_fast.account, "balance") == pytest.approx(_val_f(out_slow.account, "balance"))

    def test_multi_event_sequence_parity(self):
        fast, slow = self._make_both_pipelines()

        # Event 1: market
        ev1 = _market_event(close=50000.0)
        inp1_f = _make_input(ev1, event_index=0)
        inp1_s = _make_input(ev1, event_index=0)
        out1_f = fast.apply(inp1_f)
        out1_s = slow.apply(inp1_s)

        # Event 2: fill
        ev2 = _fill_event(side="buy", qty=0.1, price=50000.0, fee=0.5, event_id="f-2")
        inp2_f = _make_input(ev2, event_index=out1_f.event_index,
                             markets=out1_f.markets, account=out1_f.account,
                             positions=out1_f.positions)
        inp2_s = _make_input(ev2, event_index=out1_s.event_index,
                             markets=out1_s.markets, account=out1_s.account,
                             positions=out1_s.positions)
        out2_f = fast.apply(inp2_f)
        out2_s = slow.apply(inp2_s)

        assert out2_f.event_index == out2_s.event_index == 2
        assert _val_f(out2_f.positions["BTCUSDT"], "qty") == pytest.approx(_val_f(out2_s.positions["BTCUSDT"], "qty"))
        assert _val_f(out2_f.account, "balance") == pytest.approx(_val_f(out2_s.account, "balance"))

    def test_new_symbol_creates_position(self):
        fast, _ = self._make_both_pipelines()
        ev = _fill_event(symbol="ETHUSDT", side="buy", qty=2.0, price=3000.0)
        inp = _make_input(ev)
        out = fast.apply(inp)
        assert "ETHUSDT" in out.positions
        assert out.positions["ETHUSDT"].qty != Decimal("0")

    def test_snapshot_config_respected(self):
        fast_always = StatePipeline(
            config=PipelineConfig(build_snapshot_on_change_only=False)
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
        out = fast_always.apply(inp)
        assert out.advanced is True
        assert out.snapshot is not None


# ---------------------------------------------------------------------------
# Benchmark: Fast path vs slow path
# ---------------------------------------------------------------------------

class TestFastPathBenchmark:
    def test_fast_vs_slow_benchmark(self):
        """Benchmark: Rust fast path should be faster than Rust adapter path."""
        fast = StatePipeline()
        slow = StatePipeline(
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
            fast.apply(inp)
            slow.apply(inp)

        # Benchmark fast path
        t0 = time.perf_counter()
        for ev in events:
            inp = _make_input(ev)
            fast.apply(inp)
        fast_ms = (time.perf_counter() - t0) * 1000

        # Benchmark slow path
        t0 = time.perf_counter()
        for ev in events:
            inp = _make_input(ev)
            slow.apply(inp)
        slow_ms = (time.perf_counter() - t0) * 1000

        speedup = slow_ms / fast_ms if fast_ms > 0 else float("inf")
        print(f"\nFast path: {fast_ms:.2f}ms, Slow path: {slow_ms:.2f}ms, "
              f"Speedup: {speedup:.2f}x ({N} events)")

        # The fast path should not be significantly slower
        # (we expect it to be faster, but CI environments vary)
        assert fast_ms < slow_ms * 1.5, (
            f"Fast path ({fast_ms:.2f}ms) should not be much slower than "
            f"slow path ({slow_ms:.2f}ms)"
        )
