# tests/unit/execution/test_partial_fills_and_impact.py
"""Tests for partial fill support and sqrt impact slippage model."""
from __future__ import annotations

from decimal import Decimal


from execution.sim.slippage import (
    SqrtImpactSlippage,
    NoSlippage,
)
from execution.sim.paper_broker import PaperBroker, PaperBrokerConfig


# ---------------------------------------------------------------------------
# SqrtImpactSlippage tests
# ---------------------------------------------------------------------------

class TestSqrtImpactSlippage:
    def test_buy_increases_price(self) -> None:
        model = SqrtImpactSlippage(eta=Decimal("0.5"), sigma_daily=Decimal("0.02"), adv=Decimal("1000"))
        result = model.apply(price=Decimal("40000"), side="buy", qty=Decimal("10"))
        assert result > Decimal("40000")

    def test_sell_decreases_price(self) -> None:
        model = SqrtImpactSlippage(eta=Decimal("0.5"), sigma_daily=Decimal("0.02"), adv=Decimal("1000"))
        result = model.apply(price=Decimal("40000"), side="sell", qty=Decimal("10"))
        assert result < Decimal("40000")

    def test_sublinear_impact(self) -> None:
        """Doubling qty should increase impact by ~41% (sqrt(2) ≈ 1.414), not 100%."""
        model = SqrtImpactSlippage(eta=Decimal("0.5"), sigma_daily=Decimal("0.02"), adv=Decimal("1000"))
        price = Decimal("40000")

        impact_small = model.apply(price=price, side="buy", qty=Decimal("10")) - price
        impact_double = model.apply(price=price, side="buy", qty=Decimal("20")) - price

        ratio = float(impact_double / impact_small)
        # sqrt(2) ≈ 1.414, allow small tolerance
        assert 1.3 < ratio < 1.5

    def test_zero_qty_no_impact(self) -> None:
        model = SqrtImpactSlippage()
        result = model.apply(price=Decimal("40000"), side="buy", qty=Decimal("0"))
        assert result == Decimal("40000")

    def test_large_order_significant_impact(self) -> None:
        """Order = 100% of ADV should have ~1% impact with default params."""
        model = SqrtImpactSlippage(eta=Decimal("1.0"), sigma_daily=Decimal("0.02"), adv=Decimal("1000"))
        result = model.apply(price=Decimal("40000"), side="buy", qty=Decimal("1000"))
        impact_pct = float((result - Decimal("40000")) / Decimal("40000"))
        # eta * sigma * sqrt(1.0) = 1.0 * 0.02 * 1.0 = 2%
        assert 0.015 < impact_pct < 0.025


# ---------------------------------------------------------------------------
# PaperBroker partial fill tests
# ---------------------------------------------------------------------------

class TestPaperBrokerPartialFills:
    def test_full_fill_when_volume_sufficient(self) -> None:
        """Small order relative to bar volume fills completely."""
        broker = PaperBroker(config=PaperBrokerConfig(
            initial_balance=Decimal("100000"),
            volume_participation=Decimal("0.1"),
        ))
        order = broker.submit_order(
            symbol="BTCUSDT", side="buy", qty=Decimal("1"),
            price=Decimal("40000"), order_type="market",
        )
        fills = broker.try_fill(order.order_id, market_price=Decimal("40000"),
                                bar_volume=Decimal("1000"))
        assert len(fills) == 1
        assert fills[0].qty == Decimal("1")

    def test_partial_fill_when_volume_limited(self) -> None:
        """Large order relative to bar volume gets partially filled."""
        broker = PaperBroker(config=PaperBrokerConfig(
            initial_balance=Decimal("1000000"),
            volume_participation=Decimal("0.1"),  # max 10% of bar volume
        ))
        order = broker.submit_order(
            symbol="BTCUSDT", side="buy", qty=Decimal("200"),
            price=Decimal("40000"), order_type="market",
        )
        # Bar volume = 100, so max fill = 100 * 0.1 = 10
        fills = broker.try_fill(order.order_id, market_price=Decimal("40000"),
                                bar_volume=Decimal("100"))
        assert len(fills) == 1
        assert fills[0].qty == Decimal("10")

        # Order should be partially filled
        remaining_order = broker._orders[order.order_id]
        assert remaining_order.status == "partially_filled"
        assert remaining_order.filled_qty == Decimal("10")

    def test_subsequent_fills_complete_order(self) -> None:
        """Multiple bars can fully fill a large order."""
        broker = PaperBroker(config=PaperBrokerConfig(
            initial_balance=Decimal("1000000"),
            volume_participation=Decimal("0.5"),
        ))
        order = broker.submit_order(
            symbol="BTCUSDT", side="buy", qty=Decimal("30"),
            price=Decimal("40000"), order_type="market",
        )
        # First bar: volume=40, fill=20
        fills1 = broker.try_fill(order.order_id, market_price=Decimal("40000"),
                                 bar_volume=Decimal("40"))
        assert fills1[0].qty == Decimal("20")

        # Second bar: remaining=10, volume=40, fill=10
        fills2 = broker.try_fill(order.order_id, market_price=Decimal("40100"),
                                 bar_volume=Decimal("40"))
        assert fills2[0].qty == Decimal("10")

        final = broker._orders[order.order_id]
        assert final.status == "filled"
        assert final.filled_qty == Decimal("30")

    def test_no_bar_volume_uses_fill_ratio_only(self) -> None:
        """Without bar_volume, falls back to fill_ratio behavior."""
        broker = PaperBroker(config=PaperBrokerConfig(
            initial_balance=Decimal("100000"),
            fill_ratio=Decimal("1"),
        ))
        order = broker.submit_order(
            symbol="BTCUSDT", side="buy", qty=Decimal("5"),
            price=Decimal("40000"), order_type="market",
        )
        fills = broker.try_fill(order.order_id, market_price=Decimal("40000"))
        assert len(fills) == 1
        assert fills[0].qty == Decimal("5")

    def test_avg_price_weighted_across_partials(self) -> None:
        """Average price should be volume-weighted across partial fills."""
        broker = PaperBroker(
            config=PaperBrokerConfig(
                initial_balance=Decimal("1000000"),
                volume_participation=Decimal("0.5"),
                taker_fee_bps=Decimal("0"),
                maker_fee_bps=Decimal("0"),
            ),
            slippage=NoSlippage(),
        )
        order = broker.submit_order(
            symbol="BTCUSDT", side="buy", qty=Decimal("20"),
            price=Decimal("50000"), order_type="market",
        )

        broker.try_fill(order.order_id, market_price=Decimal("40000"), bar_volume=Decimal("20"))
        broker.try_fill(order.order_id, market_price=Decimal("42000"), bar_volume=Decimal("20"))

        final = broker._orders[order.order_id]
        # 10 @ 40000 + 10 @ 42000 = avg 41000
        assert final.avg_price == Decimal("41000")

    def test_filled_order_returns_empty(self) -> None:
        """Trying to fill an already filled order returns empty list."""
        broker = PaperBroker(config=PaperBrokerConfig(initial_balance=Decimal("100000")))
        order = broker.submit_order(
            symbol="BTCUSDT", side="buy", qty=Decimal("1"),
            price=Decimal("40000"), order_type="market",
        )
        broker.try_fill(order.order_id, market_price=Decimal("40000"))
        fills = broker.try_fill(order.order_id, market_price=Decimal("40000"))
        assert fills == []
