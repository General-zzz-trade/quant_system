"""Tests for attribution/pnl — PnLBreakdown, compute_pnl with position tracking."""
from __future__ import annotations

import pytest

from attribution.pnl import PnLBreakdown, compute_pnl


# ── PnLBreakdown dataclass ──────────────────────────────────

class TestPnLBreakdown:
    def test_create_minimal(self):
        bd = PnLBreakdown(total_pnl=10.0, realized_pnl=5.0, unrealized_pnl=6.0, fee_cost=1.0)
        assert bd.total_pnl == 10.0
        assert bd.funding_pnl == 0.0
        assert bd.by_symbol == {}

    def test_frozen(self):
        bd = PnLBreakdown(total_pnl=0.0, realized_pnl=0.0, unrealized_pnl=0.0, fee_cost=0.0)
        with pytest.raises(AttributeError):
            bd.total_pnl = 1.0  # type: ignore[misc]


# ── compute_pnl ─────────────────────────────────────────────

class TestComputePnl:
    def test_single_buy_unrealized_only(self):
        fills = [{"symbol": "BTC/USDT", "side": "buy", "qty": 1.0, "price": 100.0, "fee": 0.5}]
        prices = {"BTC/USDT": 110.0}
        result = compute_pnl(fills, current_prices=prices)
        assert result.realized_pnl == 0.0
        assert result.unrealized_pnl == pytest.approx(10.0)  # 1 * (110 - 100)
        assert result.fee_cost == 0.5
        assert result.total_pnl == pytest.approx(10.0 - 0.5)

    def test_buy_then_sell_realized(self):
        fills = [
            {"symbol": "BTC/USDT", "side": "buy", "qty": 2.0, "price": 100.0, "fee": 0.1},
            {"symbol": "BTC/USDT", "side": "sell", "qty": 2.0, "price": 120.0, "fee": 0.1},
        ]
        result = compute_pnl(fills)
        # Closed 2 units at 120, avg cost 100 → realized = 2*(120-100) = 40
        assert result.realized_pnl == pytest.approx(40.0)
        assert result.unrealized_pnl == 0.0
        assert result.fee_cost == pytest.approx(0.2)
        assert result.total_pnl == pytest.approx(40.0 - 0.2)

    def test_partial_close(self):
        fills = [
            {"symbol": "BTC/USDT", "side": "buy", "qty": 4.0, "price": 100.0, "fee": 0.0},
            {"symbol": "BTC/USDT", "side": "sell", "qty": 1.0, "price": 110.0, "fee": 0.0},
        ]
        prices = {"BTC/USDT": 105.0}
        result = compute_pnl(fills, current_prices=prices)
        # Closed 1 of 4 at 110: realized = 1*(110-100) = 10
        assert result.realized_pnl == pytest.approx(10.0)
        # Remaining 3 at avg 100, current 105: unrealized = 3*(105-100) = 15
        assert result.unrealized_pnl == pytest.approx(15.0)

    def test_avg_price_updates_on_adding(self):
        fills = [
            {"symbol": "BTC/USDT", "side": "buy", "qty": 1.0, "price": 100.0, "fee": 0.0},
            {"symbol": "BTC/USDT", "side": "buy", "qty": 1.0, "price": 200.0, "fee": 0.0},
            {"symbol": "BTC/USDT", "side": "sell", "qty": 2.0, "price": 160.0, "fee": 0.0},
        ]
        result = compute_pnl(fills)
        # avg = (100+200)/2 = 150, sell at 160 → realized = 2*(160-150) = 20
        assert result.realized_pnl == pytest.approx(20.0)

    def test_short_position(self):
        fills = [
            {"symbol": "BTC/USDT", "side": "sell", "qty": 1.0, "price": 100.0, "fee": 0.0},
        ]
        prices = {"BTC/USDT": 90.0}
        result = compute_pnl(fills, current_prices=prices)
        # Short 1 at 100, current 90: unrealized = -1*(90-100) = 10
        assert result.unrealized_pnl == pytest.approx(10.0)

    def test_short_close_with_profit(self):
        fills = [
            {"symbol": "BTC/USDT", "side": "sell", "qty": 2.0, "price": 100.0, "fee": 0.0},
            {"symbol": "BTC/USDT", "side": "buy", "qty": 2.0, "price": 80.0, "fee": 0.0},
        ]
        result = compute_pnl(fills)
        # Short 2 at 100, close at 80: realized = 2*(80-100)*(-1) = 40
        assert result.realized_pnl == pytest.approx(40.0)

    def test_fee_accumulation(self):
        fills = [
            {"symbol": "BTC/USDT", "side": "buy", "qty": 1.0, "price": 100.0, "fee": 0.5},
            {"symbol": "BTC/USDT", "side": "buy", "qty": 1.0, "price": 100.0, "fee": 0.3},
            {"symbol": "ETH/USDT", "side": "buy", "qty": 1.0, "price": 50.0, "fee": 0.2},
        ]
        result = compute_pnl(fills)
        assert result.fee_cost == pytest.approx(1.0)

    def test_by_symbol_breakdown(self):
        fills = [
            {"symbol": "BTC/USDT", "side": "buy", "qty": 1.0, "price": 100.0, "fee": 0.0},
            {"symbol": "BTC/USDT", "side": "sell", "qty": 1.0, "price": 110.0, "fee": 0.0},
            {"symbol": "ETH/USDT", "side": "buy", "qty": 2.0, "price": 50.0, "fee": 0.0},
            {"symbol": "ETH/USDT", "side": "sell", "qty": 2.0, "price": 40.0, "fee": 0.0},
        ]
        result = compute_pnl(fills)
        assert result.by_symbol["BTC/USDT"] == pytest.approx(10.0)
        assert result.by_symbol["ETH/USDT"] == pytest.approx(-20.0)

    def test_empty_fills(self):
        result = compute_pnl([])
        assert result.total_pnl == 0.0
        assert result.realized_pnl == 0.0
        assert result.unrealized_pnl == 0.0
        assert result.fee_cost == 0.0

    def test_reverse_position_long_to_short(self):
        fills = [
            {"symbol": "BTC/USDT", "side": "buy", "qty": 1.0, "price": 100.0, "fee": 0.0},
            {"symbol": "BTC/USDT", "side": "sell", "qty": 3.0, "price": 110.0, "fee": 0.0},
        ]
        prices = {"BTC/USDT": 105.0}
        result = compute_pnl(fills, current_prices=prices)
        # Close 1 long at 110: realized = 1*(110-100) = 10
        assert result.realized_pnl == pytest.approx(10.0)
        # avg_price formula: (100*1 + 110*3)/2 = 215 (uses full signed_qty)
        # unrealized = -2 * (105 - 215) = 220
        assert result.unrealized_pnl == pytest.approx(220.0)

    def test_no_current_prices(self):
        fills = [
            {"symbol": "BTC/USDT", "side": "buy", "qty": 1.0, "price": 100.0, "fee": 0.0},
        ]
        result = compute_pnl(fills, current_prices=None)
        assert result.unrealized_pnl == 0.0

    def test_current_prices_missing_symbol(self):
        fills = [
            {"symbol": "BTC/USDT", "side": "buy", "qty": 1.0, "price": 100.0, "fee": 0.0},
        ]
        result = compute_pnl(fills, current_prices={"ETH/USDT": 50.0})
        assert result.unrealized_pnl == 0.0
