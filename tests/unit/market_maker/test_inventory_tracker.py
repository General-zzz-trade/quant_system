"""Tests for InventoryTracker."""

import pytest
from execution.market_maker.inventory_tracker import InventoryTracker


@pytest.fixture
def tracker():
    return InventoryTracker(max_notional=50.0, daily_loss_limit=10.0)


class TestInventoryTracker:
    def test_initial_state(self, tracker):
        assert tracker.net_qty == 0.0
        assert tracker.realised_pnl == 0.0
        assert tracker.total_fills == 0

    def test_buy_fill(self, tracker):
        rpnl = tracker.on_fill("buy", 0.01, 2000.0)
        assert rpnl == 0.0  # opening trade, no realised PnL
        assert tracker.net_qty == 0.01
        assert tracker.avg_entry == 2000.0
        assert tracker.buy_fills == 1

    def test_sell_fill(self, tracker):
        rpnl = tracker.on_fill("sell", 0.01, 2000.0)
        assert rpnl == 0.0
        assert tracker.net_qty == -0.01
        assert tracker.avg_entry == 2000.0

    def test_round_trip_profit(self, tracker):
        tracker.on_fill("buy", 0.01, 2000.0)
        rpnl = tracker.on_fill("sell", 0.01, 2001.0)
        assert abs(rpnl - 0.01) < 1e-8  # 0.01 ETH * $1 = $0.01
        assert tracker.net_qty == 0.0
        assert tracker.realised_pnl == rpnl

    def test_round_trip_loss(self, tracker):
        tracker.on_fill("buy", 0.01, 2000.0)
        rpnl = tracker.on_fill("sell", 0.01, 1999.0)
        assert rpnl < 0
        assert tracker.consecutive_losses == 1

    def test_short_round_trip(self, tracker):
        tracker.on_fill("sell", 0.01, 2000.0)
        rpnl = tracker.on_fill("buy", 0.01, 1999.0)
        assert abs(rpnl - 0.01) < 1e-8  # profit on short

    def test_add_to_long(self, tracker):
        tracker.on_fill("buy", 0.01, 2000.0)
        tracker.on_fill("buy", 0.01, 2010.0)
        assert tracker.net_qty == 0.02
        assert abs(tracker.avg_entry - 2005.0) < 1e-8

    def test_position_flip(self, tracker):
        tracker.on_fill("buy", 0.01, 2000.0)
        rpnl = tracker.on_fill("sell", 0.02, 2001.0)
        assert rpnl > 0  # profit on closing the long part
        assert tracker.net_qty == -0.01
        assert tracker.avg_entry == 2001.0  # new entry at flip price

    def test_can_buy_sell_limits(self, tracker):
        assert tracker.can_buy(2000.0)
        assert tracker.can_sell(2000.0)

        # Build long to limit
        tracker.on_fill("buy", 0.025, 2000.0)  # $50 notional
        assert not tracker.can_buy(2000.0)  # at limit
        assert tracker.can_sell(2000.0)     # selling reduces

    def test_can_sell_short_limit(self, tracker):
        tracker.on_fill("sell", 0.025, 2000.0)
        assert tracker.can_buy(2000.0)      # buying reduces short
        assert not tracker.can_sell(2000.0)  # at limit

    def test_daily_loss_limit(self, tracker):
        assert not tracker.hit_daily_limit
        # Simulate losses
        tracker.on_fill("buy", 0.01, 2000.0)
        tracker.on_fill("sell", 0.01, 1000.0)  # $10 loss
        assert tracker.hit_daily_limit

    def test_unrealised_pnl(self, tracker):
        tracker.on_fill("buy", 0.01, 2000.0)
        tracker.update_unrealised(2010.0)
        assert abs(tracker.unrealised_pnl - 0.10) < 1e-8

    def test_unrealised_short(self, tracker):
        tracker.on_fill("sell", 0.01, 2000.0)
        tracker.update_unrealised(1990.0)
        assert abs(tracker.unrealised_pnl - 0.10) < 1e-8

    def test_reset_daily(self, tracker):
        tracker.on_fill("buy", 0.01, 2000.0)
        tracker.on_fill("sell", 0.01, 1999.0)
        assert tracker.daily_pnl < 0
        tracker.reset_daily()
        assert tracker.daily_pnl == 0.0
        assert tracker.consecutive_losses == 0

    def test_consecutive_losses_reset_on_win(self, tracker):
        tracker.on_fill("buy", 0.01, 2000.0)
        tracker.on_fill("sell", 0.01, 1999.0)
        assert tracker.consecutive_losses == 1
        tracker.on_fill("buy", 0.01, 2000.0)
        tracker.on_fill("sell", 0.01, 2002.0)
        assert tracker.consecutive_losses == 0
