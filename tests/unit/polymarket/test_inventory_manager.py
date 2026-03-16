"""Tests for polymarket.strategies.inventory_manager.InventoryManager."""
from __future__ import annotations

import pytest

from polymarket.strategies.inventory_manager import InventoryManager


class TestInventoryManager:

    def test_initial_state(self):
        mgr = InventoryManager(max_inventory=100)
        assert mgr.net_inventory == 0.0
        assert mgr.utilization == 0.0
        assert mgr.yes_qty == 0.0
        assert mgr.no_qty == 0.0

    def test_update_sets_quantities(self):
        mgr = InventoryManager(max_inventory=100)
        mgr.update(yes_qty=60, no_qty=20)
        assert mgr.yes_qty == 60
        assert mgr.no_qty == 20
        assert mgr.net_inventory == 40
        assert abs(mgr.utilization - 0.4) < 1e-9

    def test_add_fill_yes(self):
        mgr = InventoryManager(max_inventory=100)
        mgr.add_fill("yes", 10)
        mgr.add_fill("yes", 5)
        assert mgr.yes_qty == 15
        assert mgr.net_inventory == 15

    def test_add_fill_no(self):
        mgr = InventoryManager(max_inventory=100)
        mgr.add_fill("no", 20)
        assert mgr.no_qty == 20
        assert mgr.net_inventory == -20

    def test_add_fill_invalid_side(self):
        mgr = InventoryManager(max_inventory=100)
        with pytest.raises(ValueError, match="side must be"):
            mgr.add_fill("up", 10)

    def test_should_quote_both_below_warn(self):
        mgr = InventoryManager(max_inventory=100, warn_pct=0.80)
        mgr.update(yes_qty=30, no_qty=0)  # net=30, util=30%
        assert mgr.should_quote_side("yes") is True
        assert mgr.should_quote_side("no") is True

    def test_should_quote_restrict_at_high_util(self):
        mgr = InventoryManager(max_inventory=100, warn_pct=0.80)
        mgr.update(yes_qty=85, no_qty=0)  # net=85, util=85%
        # Long YES -> should NOT buy more YES, should buy NO
        assert mgr.should_quote_side("yes") is False
        assert mgr.should_quote_side("no") is True

    def test_should_quote_restrict_short(self):
        mgr = InventoryManager(max_inventory=100, warn_pct=0.80)
        mgr.update(yes_qty=0, no_qty=90)  # net=-90, util=90%
        # Short (long NO) -> should buy YES, should NOT buy more NO
        assert mgr.should_quote_side("yes") is True
        assert mgr.should_quote_side("no") is False

    def test_time_to_expiry_normal(self):
        mgr = InventoryManager()
        assert mgr.time_to_expiry_action(300) == "normal"

    def test_time_to_expiry_reduce_only(self):
        mgr = InventoryManager()
        assert mgr.time_to_expiry_action(90) == "reduce_only"

    def test_time_to_expiry_taker_reduce(self):
        mgr = InventoryManager()
        mgr.update(yes_qty=50, no_qty=0)  # some inventory
        assert mgr.time_to_expiry_action(45) == "taker_reduce"

    def test_time_to_expiry_cancel_all(self):
        mgr = InventoryManager()
        assert mgr.time_to_expiry_action(20) == "cancel_all"
        assert mgr.time_to_expiry_action(0) == "cancel_all"
        assert mgr.time_to_expiry_action(-5) == "cancel_all"

    def test_reset_clears_state(self):
        mgr = InventoryManager()
        mgr.update(yes_qty=50, no_qty=30)
        mgr.reset()
        assert mgr.net_inventory == 0.0
        assert mgr.yes_qty == 0.0

    def test_snapshot(self):
        mgr = InventoryManager(max_inventory=100)
        mgr.update(yes_qty=60, no_qty=20)
        snap = mgr.snapshot()
        assert snap.yes_qty == 60
        assert snap.no_qty == 20
        assert snap.net == 40
        assert abs(snap.utilization - 0.4) < 1e-9

    def test_invalid_max_inventory(self):
        with pytest.raises(ValueError, match="max_inventory"):
            InventoryManager(max_inventory=0)

    def test_invalid_warn_pct(self):
        with pytest.raises(ValueError, match="warn_pct"):
            InventoryManager(warn_pct=0.0)
