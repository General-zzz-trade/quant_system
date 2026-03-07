# tests/unit/engine/test_state_store.py
"""Tests for RustStateStore — unified pipeline with Rust-heap state."""
from __future__ import annotations

import time
import pytest
from decimal import Decimal
from types import SimpleNamespace

from state.rust_adapters import _SCALE

try:
    from _quant_hotpath import RustStateStore, RustProcessResult
    _HAS_STATE_STORE = True
except ImportError:
    _HAS_STATE_STORE = False

pytestmark = pytest.mark.skipif(not _HAS_STATE_STORE, reason="RustStateStore not available")


def _balance_i64(d: str) -> int:
    return int(Decimal(d) * _SCALE)


def _market_event(*, symbol="BTCUSDT", close=42500.0, open=42000.0,
                  high=43000.0, low=41000.0, volume=100.0):
    return SimpleNamespace(
        event_type="market",
        header=SimpleNamespace(event_type="market", ts=None, event_id="m-1"),
        symbol=symbol,
        open=Decimal(str(open)), high=Decimal(str(high)),
        low=Decimal(str(low)), close=Decimal(str(close)),
        volume=Decimal(str(volume)), ts=None,
    )


def _fill_event(*, side="buy", qty=0.1, price=50000.0, symbol="BTCUSDT",
                fee=0.5, event_id="f-1"):
    return SimpleNamespace(
        event_type="fill",
        header=SimpleNamespace(event_type="fill", ts=None, event_id=event_id),
        symbol=symbol, side=side, qty=qty, quantity=qty,
        price=price, fee=fee, realized_pnl=0.0,
        margin_change=0.0, cash_delta=0.0,
    )


def _signal_event():
    return SimpleNamespace(
        event_type="signal",
        header=SimpleNamespace(event_type="signal", ts=None, event_id=None),
        symbol="BTCUSDT",
    )


class TestRustStateStore:
    def test_creation(self):
        store = RustStateStore(["BTCUSDT"], "USDT", _balance_i64("10000"))
        assert store.event_index == 0
        assert "RustStateStore" in repr(store)

    def test_signal_not_advanced(self):
        store = RustStateStore(["BTCUSDT"], "USDT", _balance_i64("10000"))
        result = store.process_event(_signal_event(), "BTCUSDT")
        assert result.advanced is False
        assert result.changed is False
        assert store.event_index == 0

    def test_market_event_advances(self):
        store = RustStateStore(["BTCUSDT"], "USDT", _balance_i64("10000"))
        result = store.process_event(_market_event(close=50000.0), "BTCUSDT")
        assert result.advanced is True
        assert result.changed is True
        assert result.kind == "MARKET"
        assert store.event_index == 1

    def test_fill_event_updates_position(self):
        store = RustStateStore(["BTCUSDT"], "USDT", _balance_i64("10000"))
        result = store.process_event(
            _fill_event(side="buy", qty=0.5, price=50000.0, fee=1.0),
            "BTCUSDT"
        )
        assert result.advanced is True
        assert result.changed is True

    def test_get_market(self):
        store = RustStateStore(["BTCUSDT"], "USDT", _balance_i64("10000"))
        store.process_event(_market_event(close=42500.0), "BTCUSDT")
        market = store.get_market("BTCUSDT")
        assert market is not None
        assert market.close is not None

    def test_get_account(self):
        store = RustStateStore(["BTCUSDT"], "USDT", _balance_i64("10000"))
        store.process_event(
            _fill_event(side="buy", qty=0.1, price=50000.0, fee=0.5),
            "BTCUSDT"
        )
        account = store.get_account()
        assert account.balance != _balance_i64("10000")  # fee deducted

    def test_get_positions(self):
        store = RustStateStore(["BTCUSDT"], "USDT", _balance_i64("10000"))
        store.process_event(
            _fill_event(side="buy", qty=0.1, price=50000.0),
            "BTCUSDT"
        )
        positions = store.get_positions()
        assert "BTCUSDT" in positions
        assert positions["BTCUSDT"].qty != 0

    def test_new_symbol_auto_creates(self):
        store = RustStateStore(["BTCUSDT"], "USDT", _balance_i64("10000"))
        result = store.process_event(
            _fill_event(symbol="ETHUSDT", side="buy", qty=1.0, price=3000.0),
            "BTCUSDT"
        )
        assert result.advanced is True
        pos = store.get_position("ETHUSDT")
        assert pos is not None

    def test_multi_event_sequence(self):
        store = RustStateStore(["BTCUSDT"], "USDT", _balance_i64("10000"))
        # Market
        store.process_event(_market_event(close=50000.0), "BTCUSDT")
        assert store.event_index == 1
        # Fill
        store.process_event(
            _fill_event(side="buy", qty=0.1, price=50000.0, event_id="f-1"),
            "BTCUSDT"
        )
        assert store.event_index == 2
        # Signal (not advanced)
        store.process_event(_signal_event(), "BTCUSDT")
        assert store.event_index == 2  # unchanged
        # Another fill
        store.process_event(
            _fill_event(side="sell", qty=0.05, price=51000.0, event_id="f-2"),
            "BTCUSDT"
        )
        assert store.event_index == 3

    def test_last_event_id_tracked(self):
        store = RustStateStore(["BTCUSDT"], "USDT", _balance_i64("10000"))
        store.process_event(
            _fill_event(event_id="test-99"),
            "BTCUSDT"
        )
        assert store.last_event_id == "test-99"


class TestStorePathPipeline:
    """Tests for StatePipeline with store= parameter (store path)."""

    def test_signal_not_advanced(self):
        from engine.pipeline import StatePipeline, PipelineInput
        store = RustStateStore(["BTCUSDT"], "USDT", _balance_i64("10000"))
        pipeline = StatePipeline(store=store)
        assert not pipeline._use_rust_fast_path
        inp = PipelineInput(
            event=_signal_event(), event_index=0, symbol_default="BTCUSDT",
            markets={}, account=None, positions={},
        )
        out = pipeline.apply(inp)
        assert out.advanced is False
        assert out.snapshot is None

    def test_market_event_advances(self):
        from engine.pipeline import StatePipeline, PipelineInput
        store = RustStateStore(["BTCUSDT"], "USDT", _balance_i64("10000"))
        pipeline = StatePipeline(store=store)
        inp = PipelineInput(
            event=_market_event(close=50000.0), event_index=0,
            symbol_default="BTCUSDT", markets={}, account=None, positions={},
        )
        out = pipeline.apply(inp)
        assert out.advanced is True
        assert out.event_index == 1
        assert out.snapshot is not None
        # Market state should be exported
        assert "BTCUSDT" in out.markets

    def test_fill_updates_account_and_position(self):
        from engine.pipeline import StatePipeline, PipelineInput
        store = RustStateStore(["BTCUSDT"], "USDT", _balance_i64("10000"))
        pipeline = StatePipeline(store=store)
        inp = PipelineInput(
            event=_fill_event(side="buy", qty=0.5, price=50000.0, fee=1.0),
            event_index=0, symbol_default="BTCUSDT",
            markets={}, account=None, positions={},
        )
        out = pipeline.apply(inp)
        assert out.advanced is True
        assert out.snapshot is not None
        assert "BTCUSDT" in out.positions
        assert out.positions["BTCUSDT"].qty != Decimal("0")
        assert out.account.balance != Decimal("10000")

    def test_snapshot_only_on_change(self):
        from engine.pipeline import StatePipeline, PipelineInput, PipelineConfig
        store = RustStateStore(["BTCUSDT"], "USDT", _balance_i64("10000"))
        pipeline = StatePipeline(
            store=store,
            config=PipelineConfig(build_snapshot_on_change_only=True),
        )
        # Signal: advanced=False → no snapshot
        inp = PipelineInput(
            event=_signal_event(), event_index=0, symbol_default="BTCUSDT",
            markets={}, account=None, positions={},
        )
        out = pipeline.apply(inp)
        assert out.snapshot is None

    def test_multi_event_sequence(self):
        from engine.pipeline import StatePipeline, PipelineInput
        store = RustStateStore(["BTCUSDT"], "USDT", _balance_i64("10000"))
        pipeline = StatePipeline(store=store)
        # Market
        inp1 = PipelineInput(
            event=_market_event(close=50000.0), event_index=0,
            symbol_default="BTCUSDT", markets={}, account=None, positions={},
        )
        out1 = pipeline.apply(inp1)
        assert out1.event_index == 1
        # Fill
        inp2 = PipelineInput(
            event=_fill_event(side="buy", qty=0.1, price=50000.0, event_id="f-1"),
            event_index=out1.event_index, symbol_default="BTCUSDT",
            markets={}, account=None, positions={},
        )
        out2 = pipeline.apply(inp2)
        assert out2.event_index == 2
        # Signal (not advanced)
        inp3 = PipelineInput(
            event=_signal_event(), event_index=out2.event_index,
            symbol_default="BTCUSDT", markets={}, account=None, positions={},
        )
        out3 = pipeline.apply(inp3)
        assert out3.event_index == 2  # unchanged

    def test_last_event_id_and_ts(self):
        from engine.pipeline import StatePipeline, PipelineInput
        store = RustStateStore(["BTCUSDT"], "USDT", _balance_i64("10000"))
        pipeline = StatePipeline(store=store)
        inp = PipelineInput(
            event=_fill_event(event_id="test-42"), event_index=0,
            symbol_default="BTCUSDT", markets={}, account=None, positions={},
        )
        out = pipeline.apply(inp)
        assert out.last_event_id == "test-42"


class TestStateStoreBenchmark:
    def test_benchmark_vs_fast_path(self):
        """RustStateStore should be faster than Phase 0 fast path."""
        from engine.pipeline import StatePipeline, PipelineInput
        from state.market import MarketState
        from state.account import AccountState

        N = 1000

        # Build events
        events = []
        for i in range(N):
            if i % 3 == 0:
                events.append(_market_event(close=50000.0 + i))
            elif i % 3 == 1:
                events.append(_fill_event(
                    side="buy" if i % 2 == 0 else "sell",
                    qty=0.01, price=50000.0 + i,
                    fee=0.01, event_id=f"f-{i}"
                ))
            else:
                events.append(_signal_event())

        # Benchmark state store (raw)
        store = RustStateStore(["BTCUSDT"], "USDT", _balance_i64("10000"))
        t0 = time.perf_counter()
        for ev in events:
            store.process_event(ev, "BTCUSDT")
        store_ms = (time.perf_counter() - t0) * 1000

        # Benchmark fast path pipeline
        pipeline = StatePipeline()
        t0 = time.perf_counter()
        for ev in events:
            inp = PipelineInput(
                event=ev, event_index=0, symbol_default="BTCUSDT",
                markets={"BTCUSDT": MarketState.empty(symbol="BTCUSDT")},
                account=AccountState.initial(currency="USDT", balance=Decimal("10000")),
                positions={},
            )
            pipeline.apply(inp)
        fast_ms = (time.perf_counter() - t0) * 1000

        # Benchmark store path pipeline
        store2 = RustStateStore(["BTCUSDT"], "USDT", _balance_i64("10000"))
        store_pipeline = StatePipeline(store=store2)
        t0 = time.perf_counter()
        for ev in events:
            inp = PipelineInput(
                event=ev, event_index=0, symbol_default="BTCUSDT",
                markets={}, account=None, positions={},
            )
            store_pipeline.apply(inp)
        store_path_ms = (time.perf_counter() - t0) * 1000

        speedup_raw = fast_ms / store_ms if store_ms > 0 else float("inf")
        speedup_path = fast_ms / store_path_ms if store_path_ms > 0 else float("inf")
        print(f"\nRaw store: {store_ms:.2f}ms, Fast path: {fast_ms:.2f}ms, "
              f"Store path: {store_path_ms:.2f}ms")
        print(f"Raw speedup: {speedup_raw:.2f}x, Store path speedup: {speedup_path:.2f}x "
              f"({N} events)")

        assert store_ms < fast_ms * 2.0, (
            f"StateStore ({store_ms:.2f}ms) should not be much slower than "
            f"fast path ({fast_ms:.2f}ms)"
        )
