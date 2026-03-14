"""Tests for build_live_meta_builder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


from risk.meta_builder_live import build_live_meta_builder


# ── Stub coordinator ─────────────────────────────────────────

@dataclass
class FakePosition:
    qty: float = 0.0
    mark_price: float = 0.0
    entry_price: float = 0.0


class FakeCoordinator:
    def __init__(self, positions: dict[str, FakePosition] | None = None):
        self._positions = positions or {}

    def get_state_view(self) -> Dict[str, Any]:
        return {"positions": self._positions}


@dataclass
class FakeIntent:
    symbol: str = "BTCUSDT"


@dataclass
class FakeOrder:
    symbol: str = "BTCUSDT"
    price: Optional[float] = 50000.0


# ── Tests ────────────────────────────────────────────────────

class TestBuildLiveMetaBuilder:
    def test_empty_positions(self):
        coordinator = FakeCoordinator()
        builder = build_live_meta_builder(coordinator, equity_source=lambda: 10000.0)

        meta = builder.build_for_intent(FakeIntent())
        assert meta["equity"] == 10000.0
        assert meta["gross_notional"] == 0.0
        assert meta["net_notional"] == 0.0
        assert meta["symbol_weight"] == 0.0
        assert meta["positions_notional"] == {}

    def test_single_long_position(self):
        positions = {"BTCUSDT": FakePosition(qty=0.1, mark_price=50000.0)}
        coordinator = FakeCoordinator(positions)
        builder = build_live_meta_builder(coordinator, equity_source=lambda: 10000.0)

        meta = builder.build_for_intent(FakeIntent(symbol="BTCUSDT"))
        assert meta["equity"] == 10000.0
        assert meta["gross_notional"] == 5000.0
        assert meta["net_notional"] == 5000.0
        assert meta["positions_notional"]["BTCUSDT"] == 5000.0
        assert meta["symbol_weight"] == 1.0  # only one position

    def test_long_and_short_positions(self):
        positions = {
            "BTCUSDT": FakePosition(qty=0.1, mark_price=50000.0),   # +5000 notional
            "ETHUSDT": FakePosition(qty=-2.0, mark_price=3000.0),   # -6000 notional
        }
        coordinator = FakeCoordinator(positions)
        builder = build_live_meta_builder(coordinator, equity_source=lambda: 10000.0)

        meta = builder.build_for_intent(FakeIntent(symbol="BTCUSDT"))
        assert meta["gross_notional"] == 11000.0
        assert meta["net_notional"] == -1000.0
        assert abs(meta["symbol_weight"] - 5000.0 / 11000.0) < 1e-10

    def test_build_for_order_includes_market_price(self):
        coordinator = FakeCoordinator()
        builder = build_live_meta_builder(coordinator, equity_source=lambda: 10000.0)

        meta = builder.build_for_order(FakeOrder(symbol="BTCUSDT", price=55000.0))
        assert meta["market_price"] == 55000.0

    def test_build_for_order_no_price(self):
        coordinator = FakeCoordinator()
        builder = build_live_meta_builder(coordinator, equity_source=lambda: 10000.0)

        meta = builder.build_for_order(FakeOrder(symbol="BTCUSDT", price=None))
        assert "market_price" not in meta

    def test_equity_source_called(self):
        call_count = [0]
        def equity_fn():
            call_count[0] += 1
            return 20000.0

        coordinator = FakeCoordinator()
        builder = build_live_meta_builder(coordinator, equity_source=equity_fn)

        meta = builder.build_for_intent(FakeIntent())
        assert meta["equity"] == 20000.0
        assert call_count[0] == 1

    def test_position_with_zero_qty(self):
        positions = {"BTCUSDT": FakePosition(qty=0.0, mark_price=50000.0)}
        coordinator = FakeCoordinator(positions)
        builder = build_live_meta_builder(coordinator, equity_source=lambda: 10000.0)

        meta = builder.build_for_intent(FakeIntent(symbol="BTCUSDT"))
        assert meta["gross_notional"] == 0.0
        assert meta["symbol_weight"] == 0.0

    def test_uses_entry_price_fallback(self):
        positions = {"BTCUSDT": FakePosition(qty=1.0, mark_price=0, entry_price=40000.0)}
        coordinator = FakeCoordinator(positions)
        builder = build_live_meta_builder(coordinator, equity_source=lambda: 10000.0)

        meta = builder.build_for_intent(FakeIntent(symbol="BTCUSDT"))
        assert meta["gross_notional"] == 40000.0

    def test_multiple_symbols_concentration(self):
        positions = {
            "BTCUSDT": FakePosition(qty=0.1, mark_price=50000.0),   # 5000
            "ETHUSDT": FakePosition(qty=1.0, mark_price=3000.0),    # 3000
            "SOLUSDT": FakePosition(qty=10.0, mark_price=200.0),    # 2000
        }
        coordinator = FakeCoordinator(positions)
        builder = build_live_meta_builder(coordinator, equity_source=lambda: 10000.0)

        meta = builder.build_for_intent(FakeIntent(symbol="BTCUSDT"))
        assert meta["gross_notional"] == 10000.0
        assert meta["net_notional"] == 10000.0
        assert abs(meta["symbol_weight"] - 0.5) < 1e-10  # 5000/10000
