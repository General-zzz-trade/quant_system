"""Tests for attribution/cost — CostBreakdown, slippage, cost attribution."""
from __future__ import annotations

import pytest

from attribution.cost import CostBreakdown, compute_cost_attribution, estimate_slippage


# ── CostBreakdown dataclass ─────────────────────────────────

class TestCostBreakdown:
    def test_create(self):
        cb = CostBreakdown(total_cost_bps=15.0, fee_bps=10.0, slippage_bps=5.0, market_impact_bps=0.0)
        assert cb.total_cost_bps == 15.0
        assert cb.fee_bps == 10.0
        assert cb.slippage_bps == 5.0
        assert cb.market_impact_bps == 0.0

    def test_frozen(self):
        cb = CostBreakdown(total_cost_bps=1.0, fee_bps=1.0, slippage_bps=0.0, market_impact_bps=0.0)
        with pytest.raises(AttributeError):
            cb.total_cost_bps = 2.0  # type: ignore[misc]


# ── estimate_slippage ────────────────────────────────────────

class TestEstimateSlippage:
    def test_known_fills(self):
        fills = [
            {"symbol": "BTC/USDT", "price": 101.0, "qty": 1.0},
            {"symbol": "ETH/USDT", "price": 52.0, "qty": 2.0},
        ]
        ref = {"BTC/USDT": 100.0, "ETH/USDT": 50.0}
        slip = estimate_slippage(fills, ref)
        # BTC: |101-100|/100 = 0.01, notional=101
        # ETH: |52-50|/50 = 0.04, notional=104
        # weighted = (0.01*101 + 0.04*104) / (101+104) * 10000
        expected = (0.01 * 101.0 + 0.04 * 104.0) / (101.0 + 104.0) * 10_000
        assert abs(slip - expected) < 1e-6

    def test_empty_fills(self):
        assert estimate_slippage([], {"BTC/USDT": 100.0}) == 0.0

    def test_zero_reference_price(self):
        fills = [{"symbol": "BTC/USDT", "price": 100.0, "qty": 1.0}]
        ref = {"BTC/USDT": 0.0}
        # ref <= 0 → skipped entirely, total_notional stays 0
        assert estimate_slippage(fills, ref) == 0.0

    def test_missing_reference_uses_fill_price(self):
        fills = [{"symbol": "BTC/USDT", "price": 100.0, "qty": 1.0}]
        ref = {}  # no ref for BTC/USDT → ref defaults to fill price
        assert estimate_slippage(fills, ref) == 0.0

    def test_negative_slippage_treated_as_abs(self):
        fills = [{"symbol": "BTC/USDT", "price": 99.0, "qty": 1.0}]
        ref = {"BTC/USDT": 100.0}
        slip = estimate_slippage(fills, ref)
        assert slip > 0  # abs(99-100)/100 > 0


# ── compute_cost_attribution ─────────────────────────────────

class TestComputeCostAttribution:
    def test_with_fees_and_slippage(self):
        fills = [
            {"symbol": "BTC/USDT", "price": 101.0, "qty": 1.0, "fee": 0.5},
        ]
        ref = {"BTC/USDT": 100.0}
        result = compute_cost_attribution(fills, reference_prices=ref)
        assert result.fee_bps == pytest.approx(0.5 / 101.0 * 10_000)
        assert result.slippage_bps > 0
        assert result.total_cost_bps == pytest.approx(result.fee_bps + result.slippage_bps)
        assert result.market_impact_bps == 0.0

    def test_without_reference_prices(self):
        fills = [
            {"symbol": "BTC/USDT", "price": 100.0, "qty": 2.0, "fee": 1.0},
        ]
        result = compute_cost_attribution(fills, reference_prices=None)
        assert result.slippage_bps == 0.0
        assert result.fee_bps == pytest.approx(1.0 / 200.0 * 10_000)

    def test_zero_notional(self):
        fills = [{"symbol": "BTC/USDT", "price": 0.0, "qty": 0.0, "fee": 0.0}]
        result = compute_cost_attribution(fills)
        assert result.fee_bps == 0.0
        assert result.total_cost_bps == 0.0

    def test_missing_fee_field(self):
        fills = [{"symbol": "BTC/USDT", "price": 100.0, "qty": 1.0}]
        result = compute_cost_attribution(fills)
        assert result.fee_bps == 0.0

    def test_empty_fills(self):
        result = compute_cost_attribution([])
        assert result.total_cost_bps == 0.0
        assert result.fee_bps == 0.0
        assert result.slippage_bps == 0.0
