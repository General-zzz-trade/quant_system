"""Tests for OrderManager."""

import pytest
from unittest.mock import MagicMock
from execution.market_maker.config import MarketMakerConfig
from execution.market_maker.order_manager import OrderManager


@pytest.fixture
def cfg():
    return MarketMakerConfig(dry_run=True, stale_order_s=2.0)


@pytest.fixture
def mgr(cfg):
    return OrderManager(cfg)


class TestOrderManager:
    def test_submit_bid_and_ask(self, mgr):
        mgr.update_quotes(
            target_bid=1999.0, target_ask=2001.0,
            bid_size=0.01, ask_size=0.01,
            best_bid=1999.5, best_ask=2000.5,
        )
        assert mgr.live_bid is not None
        assert mgr.live_ask is not None
        assert mgr.live_bid.price == 1999.0
        assert mgr.live_ask.price == 2001.0

    def test_no_duplicate_submit(self, mgr):
        mgr.update_quotes(
            target_bid=1999.0, target_ask=2001.0,
            bid_size=0.01, ask_size=0.01,
            best_bid=1999.0, best_ask=2001.0,
        )
        bid_coid = mgr.live_bid.client_order_id
        # Same prices → no new orders
        mgr.update_quotes(
            target_bid=1999.0, target_ask=2001.0,
            bid_size=0.01, ask_size=0.01,
            best_bid=1999.0, best_ask=2001.0,
        )
        assert mgr.live_bid.client_order_id == bid_coid

    def test_cancel_on_price_change(self, mgr):
        mgr.update_quotes(
            target_bid=1999.0, target_ask=2001.0,
            bid_size=0.01, ask_size=0.01,
            best_bid=1999.5, best_ask=2000.5,
        )
        old_bid = mgr.live_bid.client_order_id
        # Price changes by more than tick
        mgr.update_quotes(
            target_bid=1998.0, target_ask=2002.0,
            bid_size=0.01, ask_size=0.01,
            best_bid=1998.5, best_ask=2001.5,
        )
        new_bid = mgr.live_bid
        assert new_bid is not None
        assert new_bid.client_order_id != old_bid

    def test_cancel_all(self, mgr):
        mgr.update_quotes(
            target_bid=1999.0, target_ask=2001.0,
            bid_size=0.01, ask_size=0.01,
            best_bid=1999.5, best_ask=2000.5,
        )
        count = mgr.cancel_all()
        assert count == 2
        assert mgr.live_bid is None
        assert mgr.live_ask is None

    def test_one_side_only(self, mgr):
        mgr.update_quotes(
            target_bid=1999.0, target_ask=None,
            bid_size=0.01, ask_size=0.01,
            best_bid=1999.5, best_ask=2000.5,
        )
        assert mgr.live_bid is not None
        assert mgr.live_ask is None

    def test_fill_clears_slot(self, mgr):
        mgr.update_quotes(
            target_bid=1999.0, target_ask=2001.0,
            bid_size=0.01, ask_size=0.01,
            best_bid=1999.5, best_ask=2000.5,
        )
        coid = mgr.live_bid.client_order_id
        mgr.on_fill(coid, 0.01)
        # After full fill, slot should be cleared
        assert mgr.live_bid is None

    def test_on_order_response(self, mgr):
        mgr.update_quotes(
            target_bid=1999.0, target_ask=2001.0,
            bid_size=0.01, ask_size=0.01,
            best_bid=1999.5, best_ask=2000.5,
        )
        coid = mgr.live_bid.client_order_id
        mgr.on_order_response({"clientOrderId": coid, "status": "NEW", "orderId": "123"})
        assert mgr.live_bid.status == "new"
        assert mgr.live_bid.exchange_order_id == "123"

    def test_cleanup_done_orders(self, mgr):
        mgr.update_quotes(
            target_bid=1999.0, target_ask=2001.0,
            bid_size=0.01, ask_size=0.01,
            best_bid=1999.5, best_ask=2000.5,
        )
        coid = mgr.live_bid.client_order_id
        mgr.on_fill(coid, 0.01)
        assert coid in mgr._orders
        mgr.cleanup_done_orders()
        assert coid not in mgr._orders

    def test_gateway_called_in_live_mode(self):
        cfg = MarketMakerConfig(dry_run=False)
        gw = MagicMock()
        mgr = OrderManager(cfg, gateway=gw)
        mgr.update_quotes(
            target_bid=1999.0, target_ask=2001.0,
            bid_size=0.01, ask_size=0.01,
            best_bid=1999.5, best_ask=2000.5,
        )
        assert gw.submit_order.call_count == 2
