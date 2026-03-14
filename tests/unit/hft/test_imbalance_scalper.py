"""Tests for ImbalanceScalper HFT strategy."""
from __future__ import annotations

import time
from decimal import Decimal


from event.tick_types import DepthUpdateEvent, TradeTickEvent
from execution.adapters.binance.depth_processor import (
    OrderBookLevel,
    OrderBookSnapshot,
)
from features.microstructure.streaming import MicrostructureState
from state.shared_position import SharedPositionStore
from strategies.hft.imbalance_scalper import ImbalanceScalper, ImbalanceScalperConfig


def _micro(
    *,
    ob_imbalance: float = 0.0,
    vpin: float = 0.1,
    spread_bps: float = 5.0,
    weighted_mid: Decimal = Decimal("50000"),
) -> MicrostructureState:
    return MicrostructureState(
        vpin=vpin,
        kyle_lambda=0.0,
        kyle_r_squared=0.0,
        ob_imbalance=ob_imbalance,
        spread_bps=spread_bps,
        weighted_mid=weighted_mid,
        ob_signal="neutral",
        depth_ratio=1.0,
        trade_count=100,
        last_price=Decimal("50000"),
        last_trade_ts_ms=1000,
    )


def _depth_event() -> DepthUpdateEvent:
    snap = OrderBookSnapshot(
        symbol="BTCUSDT",
        bids=(OrderBookLevel(Decimal("50000"), Decimal("1")),),
        asks=(OrderBookLevel(Decimal("50001"), Decimal("1")),),
        ts_ms=1000,
        last_update_id=1,
    )
    return DepthUpdateEvent(snapshot=snap)


class TestImbalanceScalper:
    def test_buy_on_strong_imbalance(self):
        store = SharedPositionStore()
        scalper = ImbalanceScalper(position_store=store, symbol="BTCUSDT")

        orders = scalper.on_depth(
            _depth_event(), _micro(ob_imbalance=0.6, vpin=0.3)
        )
        assert len(orders) == 1
        assert orders[0].side == "buy"
        assert orders[0].symbol == "BTCUSDT"

    def test_sell_on_negative_imbalance(self):
        store = SharedPositionStore()
        scalper = ImbalanceScalper(position_store=store, symbol="BTCUSDT")

        orders = scalper.on_depth(
            _depth_event(), _micro(ob_imbalance=-0.6, vpin=0.3)
        )
        assert len(orders) == 1
        assert orders[0].side == "sell"

    def test_no_trade_on_neutral(self):
        scalper = ImbalanceScalper()
        orders = scalper.on_depth(_depth_event(), _micro(ob_imbalance=0.1))
        assert len(orders) == 0

    def test_vpin_toxicity_blocks_entry(self):
        scalper = ImbalanceScalper()
        orders = scalper.on_depth(
            _depth_event(), _micro(ob_imbalance=0.6, vpin=0.9)
        )
        assert len(orders) == 0

    def test_vpin_toxicity_exits_position(self):
        store = SharedPositionStore()
        store.update_position("BTCUSDT", Decimal("0.05"))
        scalper = ImbalanceScalper(position_store=store, symbol="BTCUSDT")

        orders = scalper.on_depth(
            _depth_event(), _micro(ob_imbalance=0.1, vpin=0.9)
        )
        assert len(orders) == 1
        assert orders[0].reduce_only is True
        assert orders[0].side == "sell"  # close long

    def test_timeout_exits_position(self):
        store = SharedPositionStore()
        store.update_position("BTCUSDT", Decimal("0.05"))
        cfg = ImbalanceScalperConfig(max_hold_seconds=0.01)
        scalper = ImbalanceScalper(
            cfg=cfg, position_store=store, symbol="BTCUSDT"
        )

        # First: enter position
        scalper.on_depth(
            _depth_event(), _micro(ob_imbalance=0.6, vpin=0.3)
        )
        time.sleep(0.02)

        # Second: should trigger timeout exit
        orders = scalper.on_depth(
            _depth_event(), _micro(ob_imbalance=0.6, vpin=0.3)
        )
        assert len(orders) == 1
        assert orders[0].reduce_only is True

    def test_position_limit_blocks_entry(self):
        store = SharedPositionStore()
        store.update_position("BTCUSDT", Decimal("0.1"))
        cfg = ImbalanceScalperConfig(max_position=0.1)
        scalper = ImbalanceScalper(
            cfg=cfg, position_store=store, symbol="BTCUSDT"
        )

        # Already at max position, should not add more
        orders = scalper.on_depth(
            _depth_event(), _micro(ob_imbalance=0.6, vpin=0.3)
        )
        assert len(orders) == 0

    def test_spread_too_tight_skips(self):
        scalper = ImbalanceScalper()
        orders = scalper.on_depth(
            _depth_event(), _micro(ob_imbalance=0.6, vpin=0.3, spread_bps=0.5)
        )
        assert len(orders) == 0

    def test_on_trade_returns_empty(self):
        scalper = ImbalanceScalper()
        tick = TradeTickEvent(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            qty=Decimal("0.1"),
            side="buy",
            trade_id=1,
            ts_ms=1000,
        )
        orders = scalper.on_trade(tick, _micro())
        assert orders == []
