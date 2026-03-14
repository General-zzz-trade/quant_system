# tests/unit/execution/test_venue_manager.py
"""Tests for VenueManager and SmartRouter."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from execution.venue_manager import VenueManager, SmartRouter
from execution.models.venue import VenueInfo, VenueType


class _MockGateway:
    def __init__(self, venue: str = "mock"):
        self.venue = venue
        self.submitted: list = []
        self.cancelled: list = []

    def submit_order(self, cmd: Any) -> Dict[str, Any]:
        self.submitted.append(cmd)
        return {"orderId": len(self.submitted)}

    def cancel_order(self, cmd: Any) -> Dict[str, Any]:
        self.cancelled.append(cmd)
        return {"status": "CANCELED"}


class TestVenueManager:
    def test_register_and_lookup(self):
        mgr = VenueManager()
        info = VenueInfo(name="binance")
        gw = _MockGateway("binance")
        mgr.register(info, gw, fee_bps=4.0)

        assert "binance" in mgr.venues
        assert mgr.get_gateway("binance") is gw
        assert mgr.is_healthy("binance")

    def test_unregister(self):
        mgr = VenueManager()
        mgr.register(VenueInfo(name="binance"), _MockGateway("binance"))
        mgr.unregister("binance")
        assert "binance" not in mgr.venues

    def test_get_gateway_missing(self):
        mgr = VenueManager()
        with pytest.raises(KeyError, match="not registered"):
            mgr.get_gateway("nonexistent")

    def test_health_management(self):
        mgr = VenueManager()
        mgr.register(VenueInfo(name="binance"), _MockGateway("binance"))
        assert mgr.is_healthy("binance")

        mgr.set_health("binance", False)
        assert not mgr.is_healthy("binance")
        assert "binance" not in mgr.healthy_venues

    def test_submit_order(self):
        mgr = VenueManager()
        gw = _MockGateway("binance")
        mgr.register(VenueInfo(name="binance"), gw)

        cmd = SimpleNamespace(symbol="BTCUSDT", side="buy")
        result = mgr.submit_order("binance", cmd)
        assert result["orderId"] == 1
        assert len(gw.submitted) == 1

    def test_submit_unhealthy_raises(self):
        mgr = VenueManager()
        mgr.register(VenueInfo(name="binance"), _MockGateway("binance"))
        mgr.set_health("binance", False)

        with pytest.raises(RuntimeError, match="unhealthy"):
            mgr.submit_order("binance", SimpleNamespace())

    def test_cancel_order(self):
        mgr = VenueManager()
        gw = _MockGateway("binance")
        mgr.register(VenueInfo(name="binance"), gw)

        result = mgr.cancel_order("binance", SimpleNamespace(symbol="BTCUSDT"))
        assert result["status"] == "CANCELED"

    def test_multiple_venues(self):
        mgr = VenueManager()
        mgr.register(VenueInfo(name="binance"), _MockGateway("binance"))
        mgr.register(VenueInfo(name="sim"), _MockGateway("sim"))
        assert set(mgr.venues) == {"binance", "sim"}

    def test_get_info(self):
        mgr = VenueManager()
        info = VenueInfo(name="binance", venue_type=VenueType.PERPETUAL)
        mgr.register(info, _MockGateway("binance"))
        assert mgr.get_info("binance") is info
        assert mgr.get_info("nonexistent") is None


class TestSmartRouter:
    def _setup_mgr(self) -> VenueManager:
        mgr = VenueManager()
        mgr.register(VenueInfo(name="binance"), _MockGateway("binance"), fee_bps=4.0)
        mgr.register(VenueInfo(name="sim"), _MockGateway("sim"), fee_bps=5.0)
        return mgr

    def test_route_single_venue(self):
        mgr = VenueManager()
        mgr.register(VenueInfo(name="binance"), _MockGateway("binance"))
        router = SmartRouter(manager=mgr)

        decision = router.route("BTCUSDT", "buy", Decimal("0.1"))
        assert decision.venue == "binance"
        assert decision.reason == "only_venue"

    def test_route_lowest_fee(self):
        mgr = self._setup_mgr()
        router = SmartRouter(manager=mgr)

        decision = router.route("BTCUSDT", "buy", Decimal("1.0"))
        assert decision.venue == "binance"  # lower fee

    def test_route_respects_health(self):
        mgr = self._setup_mgr()
        mgr.set_health("binance", False)
        router = SmartRouter(manager=mgr)

        decision = router.route("BTCUSDT", "buy", Decimal("1.0"))
        assert decision.venue == "sim"

    def test_route_no_healthy_raises(self):
        mgr = self._setup_mgr()
        mgr.set_health("binance", False)
        mgr.set_health("sim", False)
        router = SmartRouter(manager=mgr)

        with pytest.raises(RuntimeError, match="No healthy venue"):
            router.route("BTCUSDT", "buy", Decimal("1.0"))

    def test_symbol_specific_routing(self):
        mgr = self._setup_mgr()
        router = SmartRouter(manager=mgr)
        router.register_symbol("ETHUSDT", ["sim"])

        decision = router.route("ETHUSDT", "buy", Decimal("1.0"))
        assert decision.venue == "sim"

    def test_latency_affects_routing(self):
        mgr = self._setup_mgr()
        router = SmartRouter(manager=mgr)
        # Give binance much higher latency
        router.set_latency("binance", 5000.0)
        router.set_latency("sim", 50.0)

        decision = router.route("BTCUSDT", "buy", Decimal("1.0"))
        assert decision.venue == "sim"

    def test_route_and_submit(self):
        mgr = self._setup_mgr()
        router = SmartRouter(manager=mgr)

        cmd = SimpleNamespace(symbol="BTCUSDT", side="buy", qty=Decimal("0.1"))
        decision, result = router.route_and_submit(cmd)
        assert decision.venue in ("binance", "sim")
        assert "orderId" in result
