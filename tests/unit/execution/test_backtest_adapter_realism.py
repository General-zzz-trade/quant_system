"""Realism tests for BacktestExecutionAdapter.

Covers: market fills, limit unfilled, partial fills, trading rule rejections,
funding settlement, summary output, backward compatibility.
"""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from runner.backtest.adapter import BacktestExecutionAdapter, TradingRules
from event.header import EventHeader
from event.types import EventType


def _order(symbol="ETHUSDT", side="buy", qty=1.0, price=None, order_type="market"):
    return SimpleNamespace(
        header=EventHeader.new_root(event_type=EventType.ORDER, version=1, source="test"),
        symbol=symbol, side=side, qty=Decimal(str(qty)),
        price=Decimal(str(price)) if price else None,
        order_type=order_type,
    )


def _adapter(**kw):
    defaults = dict(
        price_source=lambda s: Decimal("2000"),
        ts_source=lambda: None,
        fee_bps=Decimal("4"),
        slippage_bps=Decimal("1"),
        source="test",
    )
    defaults.update(kw)
    return BacktestExecutionAdapter(**defaults)


# ── 1. Market order fills ────────────────────────────────────────

class TestMarketOrderFill:
    def test_market_buy_fills_immediately(self):
        adapter = _adapter()
        fills = adapter.send_order(_order(side="buy", qty=0.5))
        assert len(fills) == 1
        assert fills[0].status == "filled"
        assert fills[0].qty == Decimal("0.5")
        assert fills[0].side == "buy"

    def test_market_sell_fills_immediately(self):
        adapter = _adapter()
        fills = adapter.send_order(_order(side="sell", qty=0.3))
        assert len(fills) == 1
        assert fills[0].status == "filled"

    def test_fill_has_cost_breakdown(self):
        adapter = _adapter()
        fills = adapter.send_order(_order(side="buy", qty=1.0))
        fill = fills[0]
        assert hasattr(fill, "fee")
        assert hasattr(fill, "slippage")
        assert hasattr(fill, "gross_pnl")
        assert hasattr(fill, "net_pnl")
        assert fill.fee > 0
        assert fill.slippage > 0


# ── 2. Limit order unfilled ──────────────────────────────────────

class TestLimitOrderUnfilled:
    def test_limit_buy_not_touched(self):
        adapter = _adapter(enable_limit_check=True)
        adapter.set_bar_hlc(high=2050, low=2010, close=2030)  # low=2010 > limit=1990
        fills = adapter.send_order(_order(side="buy", qty=0.5, price=1990, order_type="limit"))
        assert len(fills) == 0  # Not filled
        assert adapter.summary.expired_orders == 1

    def test_limit_buy_touched(self):
        adapter = _adapter(enable_limit_check=True)
        adapter.set_bar_hlc(high=2050, low=1980, close=2030)  # low=1980 <= limit=1990
        fills = adapter.send_order(_order(side="buy", qty=0.5, price=1990, order_type="limit"))
        assert len(fills) == 1
        assert fills[0].status == "filled"

    def test_limit_sell_not_touched(self):
        adapter = _adapter(enable_limit_check=True)
        adapter.set_bar_hlc(high=2050, low=2010, close=2030)  # high=2050 < limit=2100
        fills = adapter.send_order(_order(side="sell", qty=0.5, price=2100, order_type="limit"))
        assert len(fills) == 0

    def test_limit_sell_touched(self):
        adapter = _adapter(enable_limit_check=True)
        adapter.set_bar_hlc(high=2110, low=2010, close=2030)  # high=2110 >= limit=2100
        fills = adapter.send_order(_order(side="sell", qty=0.5, price=2100, order_type="limit"))
        assert len(fills) == 1


# ── 3. Partial fills ─────────────────────────────────────────────

class TestPartialFills:
    def test_partial_fill_on_limit(self):
        rules = TradingRules(min_qty=Decimal("0.01"), step_size=Decimal("0.01"))
        adapter = _adapter(partial_fill_rate=0.5, trading_rules=rules)
        fills = adapter.send_order(_order(side="buy", qty=1.0, price=2000, order_type="limit"))
        assert len(fills) == 1
        assert fills[0].qty == Decimal("0.50")
        assert fills[0].is_partial is True
        assert fills[0].status == "partially_filled"
        assert adapter.summary.partial_fill_count == 1

    def test_no_partial_on_market(self):
        adapter = _adapter(partial_fill_rate=0.5)
        fills = adapter.send_order(_order(side="buy", qty=1.0, order_type="market"))
        assert len(fills) == 1
        assert fills[0].qty == Decimal("1")  # Market orders always full fill
        assert fills[0].is_partial is False


# ── 4. Trading rule rejections ───────────────────────────────────

class TestTradingRuleRejection:
    def test_reject_below_min_qty(self):
        rules = TradingRules(min_qty=Decimal("0.1"), step_size=Decimal("0.01"))
        adapter = _adapter(trading_rules=rules)
        fills = adapter.send_order(_order(side="buy", qty=0.05))
        assert len(fills) == 0
        assert adapter.summary.rejected_orders == 1

    def test_reject_below_min_notional(self):
        rules = TradingRules(min_qty=Decimal("0.001"), min_notional=Decimal("100"))
        adapter = _adapter(
            price_source=lambda s: Decimal("50"),  # 0.5 * 50 = $25 < $100
            trading_rules=rules,
        )
        fills = adapter.send_order(_order(side="buy", qty=0.5))
        assert len(fills) == 0
        assert adapter.summary.rejected_orders == 1

    def test_qty_rounded_to_step_size(self):
        rules = TradingRules(min_qty=Decimal("0.01"), step_size=Decimal("0.01"),
                             min_notional=Decimal("1"))
        adapter = _adapter(trading_rules=rules)
        fills = adapter.send_order(_order(side="buy", qty=0.567))
        assert len(fills) == 1
        assert fills[0].qty == Decimal("0.56")  # Rounded down

    def test_on_reject_callback(self):
        rejections = []
        rules = TradingRules(min_qty=Decimal("1.0"))
        adapter = _adapter(trading_rules=rules, on_reject=lambda r: rejections.append(r))
        adapter.send_order(_order(side="buy", qty=0.1))
        assert len(rejections) == 1
        assert rejections[0].reason.startswith("qty")


# ── 5. Funding settlement ────────────────────────────────────────

class TestFundingSettlement:
    def test_funding_accrues_on_long(self):
        adapter = _adapter()
        adapter.send_order(_order(side="buy", qty=1.0))
        adapter.accrue_funding("ETHUSDT", 0.0001)  # 0.01% funding rate
        assert adapter.summary.total_funding > 0

    def test_funding_settles_on_close(self):
        adapter = _adapter()
        adapter.send_order(_order(side="buy", qty=1.0))
        adapter.accrue_funding("ETHUSDT", 0.001)  # 0.1%
        # Close position
        fills = adapter.send_order(_order(side="sell", qty=1.0))
        # Funding should be settled (deducted from net_pnl)
        assert fills[0].funding_settled > 0

    def test_no_funding_without_position(self):
        adapter = _adapter()
        adapter.accrue_funding("ETHUSDT", 0.001)
        assert adapter.summary.total_funding == 0


# ── 6. Summary output ───────────────────────────────────────────

class TestSummaryOutput:
    def test_summary_tracks_all_fields(self):
        rules = TradingRules(min_qty=Decimal("0.01"), step_size=Decimal("0.01"),
                             min_notional=Decimal("1"))
        adapter = _adapter(trading_rules=rules)

        # Successful trade
        adapter.send_order(_order(side="buy", qty=1.0))
        adapter.send_order(_order(side="sell", qty=1.0))

        # Rejected trade
        adapter.send_order(_order(side="buy", qty=0.001))

        s = adapter.summary
        assert s.total_orders == 3
        assert s.filled_orders == 2
        assert s.rejected_orders == 1
        assert s.total_fills == 2
        assert s.total_fees > 0
        assert s.total_slippage > 0

    def test_summary_to_dict(self):
        adapter = _adapter()
        adapter.send_order(_order(side="buy", qty=1.0))
        d = adapter.summary.to_dict()
        assert "gross_pnl" in d
        assert "net_pnl" in d
        assert "total_fees" in d
        assert "total_slippage" in d
        assert "total_funding" in d
        assert "rejected_orders" in d
        assert "partial_fill_count" in d


# ── 7. Backward compatibility ────────────────────────────────────

# ── 8. Volume-based slippage ──────────────────────────────────

class TestVolumeSlippage:
    def test_large_order_higher_slippage(self):
        adapter = _adapter(volume_impact_factor=0.1, slippage_bps=Decimal("0"))
        adapter.set_bar_hlc(2010, 1990, 2000, 2000, volume=100)  # small volume
        fills = adapter.send_order(_order(side="buy", qty=50.0))  # huge relative to volume
        # Price should be significantly impacted
        assert fills[0].slippage > 0

    def test_small_order_minimal_slippage(self):
        adapter = _adapter(volume_impact_factor=0.1, slippage_bps=Decimal("0"))
        adapter.set_bar_hlc(2010, 1990, 2000, 2000, volume=100000)  # large volume
        fills = adapter.send_order(_order(side="buy", qty=0.01))  # tiny
        assert fills[0].slippage < Decimal("1")  # < $1 slippage

    def test_no_volume_impact_when_disabled(self):
        adapter = _adapter(volume_impact_factor=0.0)
        adapter.set_bar_hlc(2010, 1990, 2000, 2000, volume=10)
        f1 = adapter.send_order(_order(side="buy", qty=100.0))
        # Only base slippage, no volume impact
        base_slip = Decimal("2000") * Decimal("100") * Decimal("1") / Decimal("10000")
        assert f1[0].slippage <= base_slip * Decimal("1.1")


# ── 9. Margin/liquidation ────────────────────────────────────

class TestMarginLiquidation:
    def test_no_liquidation_above_margin(self):
        adapter = _adapter(maintenance_margin=0.005, initial_equity=10000, leverage=3.0)
        adapter.send_order(_order(side="buy", qty=1.0))
        result = adapter.check_margin("ETHUSDT", 1990)  # small loss
        assert result is None  # Not liquidated

    def test_liquidation_on_deep_loss(self):
        adapter = _adapter(maintenance_margin=0.01, initial_equity=100, leverage=10.0)
        adapter.send_order(_order(side="buy", qty=0.5))  # $1000 notional on $100 equity
        result = adapter.check_margin("ETHUSDT", 1800)  # -10% = -$100 → equity ~$0
        assert result is not None  # Liquidated

    def test_no_liquidation_when_disabled(self):
        adapter = _adapter(maintenance_margin=0.0)
        adapter.send_order(_order(side="buy", qty=1.0))
        result = adapter.check_margin("ETHUSDT", 1000)  # -50%
        assert result is None  # Disabled


# ── 10. Position cap (gate chain proxy) ──────────────────────

class TestPositionCap:
    def test_cap_rejects_oversized_order(self):
        adapter = _adapter(
            max_position_pct=0.3,
            initial_equity=10000,
            leverage=1.0,
            maintenance_margin=0.001,  # enable equity tracking
        )
        # First order: 0.5 ETH × $2000 = $1000 (10% of $10K) — OK
        fills = adapter.send_order(_order(side="buy", qty=0.5))
        assert len(fills) == 1

        # Second order: 2.0 ETH × $2000 = $4000 → total $5000 (50% > 30%) — should be capped
        fills2 = adapter.send_order(_order(side="buy", qty=2.0))
        # Either partially filled or rejected
        if len(fills2) == 1:
            # Partially filled to cap
            assert fills2[0].qty < Decimal("2.0")
        else:
            # Rejected entirely
            assert adapter.summary.rejected_orders >= 1

    def test_no_cap_when_disabled(self):
        adapter = _adapter(max_position_pct=1.0)
        fills = adapter.send_order(_order(side="buy", qty=100.0))
        assert len(fills) == 1
        assert fills[0].qty == Decimal("100")


# ── 11. Pending limit orders (cross-bar) ─────────────────────

class TestPendingOrders:
    def test_pending_order_fills_on_later_bar(self):
        adapter = _adapter(enable_limit_check=True)
        # Bar 1: limit buy at 1950, bar doesn't touch
        adapter.set_bar_hlc(2010, 1960, 2000)
        fills = adapter.send_order(_order(side="buy", qty=0.5, price=1950, order_type="limit"))
        assert len(fills) == 0  # Not filled

        # Manually add to pending
        adapter._pending_orders.append(
            _order(side="buy", qty=0.5, price=1950, order_type="limit")
        )

        # Bar 2: price dips to 1940 → should fill
        adapter.set_bar_hlc(2000, 1940, 1960)
        filled = adapter.process_pending_orders()
        assert len(filled) == 1


# ── 12. Summary with all new fields ─────────────────────────

# ── 13. Continuous partial fills (FIFO queue) ────────────────

class TestContinuousPartialFills:
    def test_partial_remainder_requeued(self):
        rules = TradingRules(min_qty=Decimal("0.01"), step_size=Decimal("0.01"),
                             min_notional=Decimal("1"))
        adapter = _adapter(partial_fill_rate=0.5, trading_rules=rules,
                           enable_limit_check=True)

        # Submit limit order
        adapter.submit_limit_order(
            _order(side="buy", qty=1.0, price=1990, order_type="limit"),
            ttl_bars=5,
        )

        # Bar 1: touches limit → fills 50%
        adapter.set_bar_hlc(2000, 1980, 1990)
        fills = adapter.process_pending_orders()
        assert len(fills) == 1
        assert fills[0].qty == Decimal("0.50")

        # Remaining 0.50 should still be in queue
        assert len(adapter._pending_orders) == 1
        assert adapter._pending_orders[0].qty == Decimal("0.50")


# ── 14. Pending order TTL expiry ─────────────────────────────

class TestPendingOrderTTL:
    def test_order_expires_after_ttl(self):
        rejections = []
        adapter = _adapter(enable_limit_check=True, on_reject=lambda r: rejections.append(r))

        adapter.submit_limit_order(
            _order(side="buy", qty=0.5, price=1900, order_type="limit"),
            ttl_bars=3,
        )

        # 3 bars where limit isn't touched → TTL expires
        for _ in range(4):
            adapter.set_bar_hlc(2010, 1950, 2000)  # low=1950 > limit=1900
            adapter.process_pending_orders()

        assert len(adapter._pending_orders) == 0
        assert adapter.summary.expired_orders >= 1
        assert len(rejections) >= 1
        assert rejections[-1].reason == "ttl_expired"

    def test_order_fills_before_ttl(self):
        adapter = _adapter(enable_limit_check=True)
        adapter.submit_limit_order(
            _order(side="buy", qty=0.5, price=1990, order_type="limit"),
            ttl_bars=10,
        )

        # Bar touches limit on first try
        adapter.set_bar_hlc(2000, 1980, 1990)
        fills = adapter.process_pending_orders()
        assert len(fills) == 1
        assert len(adapter._pending_orders) == 0  # Filled, no pending


# ── 15. Gate chain proxy ─────────────────────────────────────

class TestGateChainProxy:
    def test_alpha_health_zero_rejects(self):
        rules = TradingRules(min_qty=Decimal("0.01"))
        adapter = _adapter(trading_rules=rules)
        result = adapter.apply_gate_chain(
            _order(side="buy", qty=1.0),
            alpha_health_scale=0.0,
        )
        assert result is None
        assert adapter.summary.rejected_orders == 1

    def test_regime_scale_reduces_qty(self):
        rules = TradingRules(min_qty=Decimal("0.01"), step_size=Decimal("0.01"))
        adapter = _adapter(trading_rules=rules)
        result = adapter.apply_gate_chain(
            _order(side="buy", qty=1.0),
            alpha_health_scale=1.0,
            regime_scale=0.5,
        )
        assert result is not None
        assert result.qty == Decimal("0.50")

    def test_combined_scaling(self):
        rules = TradingRules(min_qty=Decimal("0.01"), step_size=Decimal("0.01"))
        adapter = _adapter(trading_rules=rules)
        result = adapter.apply_gate_chain(
            _order(side="buy", qty=1.0),
            alpha_health_scale=0.5,
            regime_scale=0.5,
        )
        assert result is not None
        assert result.qty == Decimal("0.25")

    def test_scaling_below_min_qty_rejects(self):
        rules = TradingRules(min_qty=Decimal("0.1"))
        adapter = _adapter(trading_rules=rules)
        result = adapter.apply_gate_chain(
            _order(side="buy", qty=0.1),
            alpha_health_scale=0.5,
        )
        assert result is None  # 0.1 × 0.5 = 0.05 < min_qty 0.1


class TestEnhancedSummary:
    def test_summary_has_liquidation_count(self):
        adapter = _adapter(maintenance_margin=0.01, initial_equity=100, leverage=10.0)
        adapter.send_order(_order(side="buy", qty=0.5))
        adapter.check_margin("ETHUSDT", 1800)  # Liquidate
        d = adapter.summary.to_dict()
        assert "liquidation_count" in d


class TestBackwardCompatibility:
    def test_default_constructor_works(self):
        """Original constructor with no new params should work unchanged."""
        adapter = BacktestExecutionAdapter(
            price_source=lambda s: Decimal("100"),
            ts_source=lambda: None,
            fee_bps=Decimal("4"),
            slippage_bps=Decimal("1"),
        )
        fills = adapter.send_order(_order(side="buy", qty=1.0))
        assert len(fills) == 1
        assert fills[0].qty == Decimal("1")

    def test_old_api_position_tracking(self):
        adapter = _adapter()
        adapter.send_order(_order(side="buy", qty=0.5))
        assert adapter.get_position("ETHUSDT") == Decimal("0.5")
        adapter.send_order(_order(side="sell", qty=0.5))
        assert adapter.get_position("ETHUSDT") == Decimal("0")

    def test_pnl_tracking(self):
        adapter = BacktestExecutionAdapter(
            price_source=lambda s: Decimal("2100"),  # Price went up
            ts_source=lambda: None,
            fee_bps=Decimal("0"),
            slippage_bps=Decimal("0"),
        )
        adapter.send_order(_order(side="buy", qty=1.0, price=2000))
        pnl = adapter.get_pnl("ETHUSDT")
        assert pnl == Decimal("100")  # 2100 - 2000
