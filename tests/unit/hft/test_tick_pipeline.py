"""Tests for Phase 1: Tick data pipeline."""
from __future__ import annotations

import json
import time
import threading
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from event.tick_types import DepthUpdateEvent, TradeTickEvent
from execution.adapters.binance.depth_processor import (
    OrderBookLevel,
    OrderBookSnapshot,
)
from execution.adapters.binance.ws_trade_stream import BinanceTradeStreamClient
from features.microstructure.streaming import (
    MicrostructureState,
    StreamingMicrostructureComputer,
)
from engine.tick_engine import HFTOrder, TickEngine
from state.shared_position import SharedPositionStore


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_tick(
    price: str = "50000",
    qty: str = "0.1",
    side: str = "buy",
    trade_id: int = 1,
    ts_ms: int = 1000,
) -> TradeTickEvent:
    return TradeTickEvent(
        symbol="BTCUSDT",
        price=Decimal(price),
        qty=Decimal(qty),
        side=side,
        trade_id=trade_id,
        ts_ms=ts_ms,
    )


def _make_book(
    bids: list[tuple[str, str]] | None = None,
    asks: list[tuple[str, str]] | None = None,
) -> OrderBookSnapshot:
    bids = bids or [("50000", "1.0"), ("49999", "2.0")]
    asks = asks or [("50001", "1.0"), ("50002", "2.0")]
    return OrderBookSnapshot(
        symbol="BTCUSDT",
        bids=tuple(OrderBookLevel(Decimal(p), Decimal(q)) for p, q in bids),
        asks=tuple(OrderBookLevel(Decimal(p), Decimal(q)) for p, q in asks),
        ts_ms=1000,
        last_update_id=100,
    )


def _make_depth_event(
    bids: list[tuple[str, str]] | None = None,
    asks: list[tuple[str, str]] | None = None,
) -> DepthUpdateEvent:
    return DepthUpdateEvent(snapshot=_make_book(bids, asks))


# ── TradeTickEvent / DepthUpdateEvent ────────────────────────────────────


class TestTickEvents:
    def test_trade_tick_creation(self):
        tick = _make_tick()
        assert tick.symbol == "BTCUSDT"
        assert tick.price == Decimal("50000")
        assert tick.qty == Decimal("0.1")
        assert tick.side == "buy"
        assert tick.trade_id == 1

    def test_trade_tick_frozen(self):
        tick = _make_tick()
        with pytest.raises(AttributeError):
            tick.price = Decimal("60000")  # type: ignore

    def test_depth_update_creation(self):
        event = _make_depth_event()
        assert event.snapshot.symbol == "BTCUSDT"
        assert len(event.snapshot.bids) == 2
        assert isinstance(event.received_at, float)


# ── BinanceTradeStreamClient ────────────────────────────────────────────


class TestTradeStreamClient:
    def _make_agg_trade_msg(
        self, *, price: str = "50000.00", qty: str = "0.5", buyer_is_maker: bool = False
    ) -> str:
        return json.dumps(
            {
                "data": {
                    "e": "aggTrade",
                    "s": "BTCUSDT",
                    "p": price,
                    "q": qty,
                    "m": buyer_is_maker,
                    "a": 12345,
                    "T": 1609459200000,
                }
            }
        )

    def test_parse_buy_trade(self):
        transport = MagicMock()
        transport.recv.return_value = self._make_agg_trade_msg(buyer_is_maker=False)

        client = BinanceTradeStreamClient(
            transport=transport, streams=("btcusdt@aggTrade",)
        )
        tick = client.step()
        assert tick is not None
        assert tick.side == "buy"
        assert tick.price == Decimal("50000.00")
        assert tick.qty == Decimal("0.5")
        assert tick.symbol == "BTCUSDT"

    def test_parse_sell_trade(self):
        transport = MagicMock()
        transport.recv.return_value = self._make_agg_trade_msg(buyer_is_maker=True)

        client = BinanceTradeStreamClient(
            transport=transport, streams=("btcusdt@aggTrade",)
        )
        tick = client.step()
        assert tick is not None
        assert tick.side == "sell"

    def test_parse_invalid_json(self):
        transport = MagicMock()
        transport.recv.return_value = "not json"

        client = BinanceTradeStreamClient(
            transport=transport, streams=("btcusdt@aggTrade",)
        )
        assert client.step() is None

    def test_parse_non_aggtrade(self):
        transport = MagicMock()
        transport.recv.return_value = json.dumps({"data": {"e": "depthUpdate"}})

        client = BinanceTradeStreamClient(
            transport=transport, streams=("btcusdt@aggTrade",)
        )
        assert client.step() is None

    def test_connect_builds_url(self):
        transport = MagicMock()
        client = BinanceTradeStreamClient(
            transport=transport, streams=("btcusdt@aggTrade", "ethusdt@aggTrade")
        )
        url = client.connect()
        assert "btcusdt@aggTrade" in url
        assert "ethusdt@aggTrade" in url
        transport.connect.assert_called_once()

    def test_close(self):
        transport = MagicMock()
        client = BinanceTradeStreamClient(
            transport=transport, streams=("btcusdt@aggTrade",)
        )
        client.connect()
        client.close()
        transport.close.assert_called_once()
        assert client._connected_url is None

    def test_step_auto_connects(self):
        transport = MagicMock()
        transport.recv.return_value = None
        client = BinanceTradeStreamClient(
            transport=transport, streams=("btcusdt@aggTrade",)
        )
        client.step()
        transport.connect.assert_called_once()


# ── StreamingMicrostructureComputer ──────────────────────────────────────


class TestStreamingMicrostructure:
    def test_on_trade_returns_state(self):
        comp = StreamingMicrostructureComputer(trade_buffer_size=50)
        tick = _make_tick()
        state = comp.on_trade(tick)
        assert isinstance(state, MicrostructureState)
        assert state.trade_count == 1
        assert state.last_price == Decimal("50000")

    def test_on_trade_accumulates(self):
        comp = StreamingMicrostructureComputer(trade_buffer_size=50)
        for i in range(10):
            state = comp.on_trade(
                _make_tick(price=str(50000 + i), trade_id=i, ts_ms=1000 + i)
            )
        assert state.trade_count == 10

    def test_on_depth_returns_state(self):
        comp = StreamingMicrostructureComputer()
        event = _make_depth_event()
        state = comp.on_depth(event)
        assert isinstance(state, MicrostructureState)
        assert state.ob_signal in ("buy_pressure", "sell_pressure", "neutral")
        assert state.spread_bps > 0

    def test_on_depth_updates_ob_fields(self):
        comp = StreamingMicrostructureComputer()
        # Strong buy pressure: big bids, small asks
        event = _make_depth_event(
            bids=[("50000", "10.0"), ("49999", "10.0")],
            asks=[("50001", "1.0"), ("50002", "1.0")],
        )
        state = comp.on_depth(event)
        assert state.ob_imbalance > 0.3
        assert state.ob_signal == "buy_pressure"

    def test_trade_then_depth_preserves_trade_info(self):
        comp = StreamingMicrostructureComputer(trade_buffer_size=50)
        tick = _make_tick(price="50000")
        comp.on_trade(tick)
        state = comp.on_depth(_make_depth_event())
        assert state.trade_count == 1
        assert state.last_price == Decimal("50000")

    def test_vpin_computes_with_enough_trades(self):
        comp = StreamingMicrostructureComputer(
            trade_buffer_size=200,
            vpin_bucket_volume=Decimal("1"),
            vpin_n_buckets=5,
        )
        for i in range(20):
            side = "buy" if i % 2 == 0 else "sell"
            state = comp.on_trade(
                _make_tick(
                    price=str(50000 + i),
                    qty="1.0",
                    side=side,
                    trade_id=i,
                    ts_ms=1000 + i,
                )
            )
        # With alternating buy/sell, VPIN should be low (balanced flow)
        assert state.vpin >= 0.0
        assert state.vpin <= 1.0


# ── TickEngine ───────────────────────────────────────────────────────────


class TestTickEngine:
    def test_register_strategy(self):
        engine = TickEngine()
        strategy = MagicMock()
        strategy.strategy_id = "test"
        engine.register_strategy(strategy)
        assert len(engine._strategies) == 1

    def test_on_trade_tick_queues(self):
        engine = TickEngine()
        tick = _make_tick()
        engine.on_trade_tick(tick)
        assert engine.tick_queue.qsize() == 1

    def test_on_depth_update_queues(self):
        engine = TickEngine()
        event = _make_depth_event()
        engine.on_depth_update(event)
        assert engine.tick_queue.qsize() == 1

    def test_process_trade_dispatches_to_strategy(self):
        engine = TickEngine()
        strategy = MagicMock()
        strategy.strategy_id = "test"
        strategy.on_trade.return_value = []
        engine.register_strategy(strategy)

        tick = _make_tick()
        engine._process_event(tick)

        strategy.on_trade.assert_called_once()
        args = strategy.on_trade.call_args
        assert args[0][0] is tick
        assert isinstance(args[0][1], MicrostructureState)

    def test_process_depth_dispatches_to_strategy(self):
        engine = TickEngine()
        strategy = MagicMock()
        strategy.strategy_id = "test"
        strategy.on_depth.return_value = []
        engine.register_strategy(strategy)

        event = _make_depth_event()
        engine._process_event(event)

        strategy.on_depth.assert_called_once()

    def test_strategy_orders_reach_order_queue(self):
        engine = TickEngine()
        order = HFTOrder(symbol="BTCUSDT", side="buy", qty=0.01, strategy_id="test")
        strategy = MagicMock()
        strategy.strategy_id = "test"
        strategy.on_trade.return_value = [order]
        engine.register_strategy(strategy)

        engine._process_event(_make_tick())
        assert engine.order_queue.qsize() == 1
        assert engine.order_queue.get_nowait() == order

    def test_risk_checker_blocks_order(self):
        risk = MagicMock()
        risk.check.return_value = (False, "position_limit")
        engine = TickEngine(risk_checker=risk)

        order = HFTOrder(symbol="BTCUSDT", side="buy", qty=0.01, strategy_id="test")
        strategy = MagicMock()
        strategy.strategy_id = "test"
        strategy.on_trade.return_value = [order]
        engine.register_strategy(strategy)

        engine._process_event(_make_tick())
        assert engine.order_queue.qsize() == 0

    def test_start_stop(self):
        engine = TickEngine()
        engine.start()
        assert engine._running
        assert engine._thread is not None
        engine.stop()
        assert not engine._running

    def test_full_cycle(self):
        """End-to-end: push tick → engine processes → strategy gets callback."""
        engine = TickEngine()
        received = []

        class Recorder:
            strategy_id = "recorder"

            def on_trade(self, tick, micro):
                received.append(("trade", tick, micro))
                return []

            def on_depth(self, event, micro):
                received.append(("depth", event, micro))
                return []

        engine.register_strategy(Recorder())
        engine.start()

        engine.on_trade_tick(_make_tick())
        engine.on_depth_update(_make_depth_event())
        time.sleep(0.1)
        engine.stop()

        assert len(received) == 2
        assert received[0][0] == "trade"
        assert received[1][0] == "depth"


# ── SharedPositionStore ──────────────────────────────────────────────────


class TestSharedPositionStore:
    def test_get_default_zero(self):
        store = SharedPositionStore()
        assert store.get_position("BTCUSDT") == Decimal("0")

    def test_update_and_get(self):
        store = SharedPositionStore()
        store.update_position("BTCUSDT", Decimal("1.5"))
        assert store.get_position("BTCUSDT") == Decimal("1.5")

    def test_add_fill_buy(self):
        store = SharedPositionStore()
        new_pos = store.add_fill("BTCUSDT", Decimal("0.1"), "buy")
        assert new_pos == Decimal("0.1")

    def test_add_fill_sell(self):
        store = SharedPositionStore()
        store.update_position("BTCUSDT", Decimal("1.0"))
        new_pos = store.add_fill("BTCUSDT", Decimal("0.3"), "sell")
        assert new_pos == Decimal("0.7")

    def test_all_positions(self):
        store = SharedPositionStore()
        store.update_position("BTCUSDT", Decimal("1.0"))
        store.update_position("ETHUSDT", Decimal("10.0"))
        positions = store.all_positions()
        assert positions == {
            "BTCUSDT": Decimal("1.0"),
            "ETHUSDT": Decimal("10.0"),
        }

    def test_thread_safety(self):
        store = SharedPositionStore()

        def writer(start_val: int):
            for i in range(100):
                store.add_fill("BTCUSDT", Decimal("0.001"), "buy")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 10 threads * 100 fills * 0.001 = 1.0
        assert store.get_position("BTCUSDT") == Decimal("1.000")
