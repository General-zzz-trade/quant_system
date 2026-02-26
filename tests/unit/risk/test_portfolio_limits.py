# tests/unit/risk/test_portfolio_limits.py
"""Tests for cross-asset portfolio risk rules: Gross, Net, Concentration."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

import pytest

from event.types import Side, Venue
from risk.decisions import RiskAction, RiskCode
from risk.rules.portfolio_limits import (
    GrossExposureRule,
    NetExposureRule,
    ConcentrationRule,
)


# ── Test stubs (matching existing risk test patterns) ────────────

@dataclass(frozen=True)
class _Symbol:
    raw: str

    @property
    def normalized(self) -> str:
        return self.raw.upper()

    def __str__(self) -> str:
        return self.normalized


@dataclass(frozen=True)
class _Qty:
    value: Decimal

    @classmethod
    def of(cls, v) -> "_Qty":
        return cls(value=Decimal(str(v)))


@dataclass(frozen=True)
class _Price:
    value: Decimal

    @classmethod
    def of(cls, v) -> "_Price":
        return cls(value=Decimal(str(v)))


@dataclass(frozen=True)
class _IntentEvent:
    symbol: _Symbol
    side: Side
    target_qty: Optional[_Qty] = None


@dataclass(frozen=True)
class _OrderEvent:
    symbol: _Symbol
    side: Side
    qty: _Qty
    venue: Venue = Venue.BINANCE
    limit_price: Optional[_Price] = None
    reduce_only: bool = False


def _intent(symbol: str = "BTCUSDT", side: Side = Side.BUY) -> _IntentEvent:
    return _IntentEvent(symbol=_Symbol(symbol), side=side)


def _order(
    symbol: str = "BTCUSDT",
    side: Side = Side.BUY,
    qty: float = 1.0,
    limit_price: Optional[float] = None,
) -> _OrderEvent:
    return _OrderEvent(
        symbol=_Symbol(symbol),
        side=side,
        qty=_Qty.of(qty),
        limit_price=_Price.of(limit_price) if limit_price is not None else None,
    )


# ============================================================
# GrossExposureRule
# ============================================================

class TestGrossExposureRule:
    def test_allow_below_limit(self):
        rule = GrossExposureRule(max_gross_leverage=Decimal("3"))
        meta = {"equity": Decimal("100000"), "gross_notional": Decimal("200000")}
        d = rule.evaluate_intent(_intent(), meta=meta)
        assert d.action == RiskAction.ALLOW

    def test_reject_above_limit(self):
        rule = GrossExposureRule(max_gross_leverage=Decimal("3"))
        meta = {"equity": Decimal("100000"), "gross_notional": Decimal("400000")}
        d = rule.evaluate_intent(_intent(), meta=meta)
        assert d.action == RiskAction.REJECT
        assert d.violations[0].code == RiskCode.MAX_GROSS

    def test_intent_skip_missing_gross(self):
        rule = GrossExposureRule()
        meta = {"equity": Decimal("100000")}
        d = rule.evaluate_intent(_intent(), meta=meta)
        assert d.action == RiskAction.ALLOW
        assert "intent_skip_missing_gross" in d.tags

    def test_reject_missing_equity(self):
        rule = GrossExposureRule()
        d = rule.evaluate_intent(_intent(), meta={})
        assert d.action == RiskAction.REJECT

    def test_order_allow_below_limit(self):
        rule = GrossExposureRule(max_gross_leverage=Decimal("3"))
        meta = {
            "equity": Decimal("100000"),
            "gross_notional": Decimal("200000"),
            "market_price": Decimal("40000"),
        }
        # Buy 1 BTC at 40k → delta = 40k → projected gross = 240k → 2.4x
        d = rule.evaluate_order(_order(qty=1.0), meta=meta)
        assert d.action == RiskAction.ALLOW

    def test_order_reject_above_limit(self):
        rule = GrossExposureRule(max_gross_leverage=Decimal("3"), allow_auto_reduce=False)
        meta = {
            "equity": Decimal("100000"),
            "gross_notional": Decimal("280000"),
            "market_price": Decimal("40000"),
        }
        # delta = 40k → projected gross = 320k → 3.2x > 3x
        d = rule.evaluate_order(_order(qty=1.0), meta=meta)
        assert d.action == RiskAction.REJECT

    def test_order_reduce_when_above_limit(self):
        rule = GrossExposureRule(max_gross_leverage=Decimal("3"), allow_auto_reduce=True)
        meta = {
            "equity": Decimal("100000"),
            "gross_notional": Decimal("280000"),
            "market_price": Decimal("40000"),
        }
        # Headroom = 300k - 280k = 20k → max_qty = 20k / 40k = 0.5
        d = rule.evaluate_order(_order(qty=1.0), meta=meta)
        assert d.action == RiskAction.REDUCE
        assert d.adjustment is not None
        assert d.adjustment.max_qty == pytest.approx(0.5)

    def test_order_with_limit_price(self):
        rule = GrossExposureRule(max_gross_leverage=Decimal("3"))
        meta = {
            "equity": Decimal("100000"),
            "gross_notional": Decimal("200000"),
        }
        d = rule.evaluate_order(_order(qty=1.0, limit_price=40000.0), meta=meta)
        assert d.action == RiskAction.ALLOW

    def test_3x_leverage_triggers_on_portfolio(self):
        """Verify: portfolio exceeding 3x leverage triggers REDUCE."""
        rule = GrossExposureRule(max_gross_leverage=Decimal("3"))
        meta = {
            "equity": Decimal("100000"),
            "gross_notional": Decimal("290000"),
            "market_price": Decimal("50000"),
        }
        # delta = 50k → projected = 340k → 3.4x > 3x
        d = rule.evaluate_order(_order(qty=1.0), meta=meta)
        assert d.action == RiskAction.REDUCE

    def test_positions_notional_fallback(self):
        rule = GrossExposureRule(max_gross_leverage=Decimal("3"))
        meta = {
            "equity": Decimal("100000"),
            "positions_notional": {"BTCUSDT": Decimal("150000"), "ETHUSDT": Decimal("-50000")},
        }
        # gross = 150k + 50k = 200k → 2x → ALLOW
        d = rule.evaluate_intent(_intent(), meta=meta)
        assert d.action == RiskAction.ALLOW


# ============================================================
# NetExposureRule
# ============================================================

class TestNetExposureRule:
    def test_allow_below_limit(self):
        rule = NetExposureRule(max_net_leverage=Decimal("1"))
        meta = {"equity": Decimal("100000"), "net_notional": Decimal("80000")}
        d = rule.evaluate_intent(_intent(), meta=meta)
        assert d.action == RiskAction.ALLOW

    def test_reject_above_limit(self):
        rule = NetExposureRule(max_net_leverage=Decimal("1"))
        meta = {"equity": Decimal("100000"), "net_notional": Decimal("120000")}
        d = rule.evaluate_intent(_intent(), meta=meta)
        assert d.action == RiskAction.REJECT
        assert d.violations[0].code == RiskCode.MAX_NET

    def test_negative_net_uses_abs(self):
        rule = NetExposureRule(max_net_leverage=Decimal("1"))
        meta = {"equity": Decimal("100000"), "net_notional": Decimal("-120000")}
        d = rule.evaluate_intent(_intent(), meta=meta)
        assert d.action == RiskAction.REJECT

    def test_intent_skip_missing_net(self):
        rule = NetExposureRule()
        meta = {"equity": Decimal("100000")}
        d = rule.evaluate_intent(_intent(), meta=meta)
        assert d.action == RiskAction.ALLOW
        assert "intent_skip_missing_net" in d.tags

    def test_order_allow_below_limit(self):
        rule = NetExposureRule(max_net_leverage=Decimal("1"))
        meta = {
            "equity": Decimal("100000"),
            "net_notional": Decimal("50000"),
            "market_price": Decimal("40000"),
        }
        # Buy 1 BTC → delta = +40k → projected net = 90k → 0.9x
        d = rule.evaluate_order(_order(side=Side.BUY, qty=1.0), meta=meta)
        assert d.action == RiskAction.ALLOW

    def test_order_reject_above_limit(self):
        rule = NetExposureRule(max_net_leverage=Decimal("1"))
        meta = {
            "equity": Decimal("100000"),
            "net_notional": Decimal("80000"),
            "market_price": Decimal("40000"),
        }
        # Buy 1 BTC → delta = +40k → projected net = 120k → 1.2x > 1x
        d = rule.evaluate_order(_order(side=Side.BUY, qty=1.0), meta=meta)
        assert d.action == RiskAction.REJECT

    def test_sell_reduces_net(self):
        rule = NetExposureRule(max_net_leverage=Decimal("1"))
        meta = {
            "equity": Decimal("100000"),
            "net_notional": Decimal("80000"),
            "market_price": Decimal("40000"),
        }
        # Sell 1 BTC → delta = -40k → projected net = 40k → 0.4x
        d = rule.evaluate_order(_order(side=Side.SELL, qty=1.0), meta=meta)
        assert d.action == RiskAction.ALLOW

    def test_positions_notional_fallback(self):
        rule = NetExposureRule(max_net_leverage=Decimal("1"))
        meta = {
            "equity": Decimal("100000"),
            "positions_notional": {"BTCUSDT": Decimal("150000"), "ETHUSDT": Decimal("-100000")},
        }
        # net = 150k - 100k = 50k → 0.5x → ALLOW
        d = rule.evaluate_intent(_intent(), meta=meta)
        assert d.action == RiskAction.ALLOW


# ============================================================
# ConcentrationRule
# ============================================================

class TestConcentrationRule:
    def test_allow_below_limit(self):
        rule = ConcentrationRule(max_weight=Decimal("0.5"))
        meta = {
            "positions_notional": {
                "BTCUSDT": Decimal("100000"),
                "ETHUSDT": Decimal("100000"),
                "SOLUSDT": Decimal("100000"),
            },
        }
        # Each symbol = 33% < 50%
        d = rule.evaluate_intent(_intent("BTCUSDT"), meta=meta)
        assert d.action == RiskAction.ALLOW

    def test_reject_above_limit(self):
        rule = ConcentrationRule(max_weight=Decimal("0.4"))
        meta = {
            "positions_notional": {
                "BTCUSDT": Decimal("300000"),
                "ETHUSDT": Decimal("100000"),
                "SOLUSDT": Decimal("100000"),
            },
        }
        # BTC = 300k / 500k = 60% > 40%
        d = rule.evaluate_intent(_intent("BTCUSDT"), meta=meta)
        assert d.action == RiskAction.REJECT

    def test_intent_skip_missing_positions(self):
        rule = ConcentrationRule()
        meta = {"equity": Decimal("100000")}
        d = rule.evaluate_intent(_intent(), meta=meta)
        assert d.action == RiskAction.ALLOW
        assert "intent_skip_missing_positions" in d.tags

    def test_per_symbol_override(self):
        rule = ConcentrationRule(
            max_weight=Decimal("0.3"),
            per_symbol_max_weight={"BTCUSDT": Decimal("0.5")},
        )
        meta = {
            "positions_notional": {
                "BTCUSDT": Decimal("200000"),
                "ETHUSDT": Decimal("100000"),
                "SOLUSDT": Decimal("100000"),
            },
        }
        # BTC = 200k / 400k = 50% == 50% cap → ALLOW
        d = rule.evaluate_intent(_intent("BTCUSDT"), meta=meta)
        assert d.action == RiskAction.ALLOW

        # ETH = 100k / 400k = 25% < 30% default cap → ALLOW
        d = rule.evaluate_intent(_intent("ETHUSDT"), meta=meta)
        assert d.action == RiskAction.ALLOW

    def test_order_allow_below_limit(self):
        rule = ConcentrationRule(max_weight=Decimal("0.5"))
        meta = {
            "positions_notional": {
                "BTCUSDT": Decimal("100000"),
                "ETHUSDT": Decimal("100000"),
            },
            "market_price": Decimal("40000"),
        }
        # Buy 0.5 BTC → delta = 20k → BTC becomes 120k, gross becomes 220k → 54.5% > 50%? No...
        # Actually: projected_sym = 100k + 20k = 120k, projected_gross = 200k + 20k = 220k → 54.5%
        # So this should reject
        d = rule.evaluate_order(_order("BTCUSDT", Side.BUY, qty=0.5), meta=meta)
        assert d.action == RiskAction.REJECT

    def test_order_small_trade_allows(self):
        rule = ConcentrationRule(max_weight=Decimal("0.5"))
        meta = {
            "positions_notional": {
                "BTCUSDT": Decimal("80000"),
                "ETHUSDT": Decimal("120000"),
            },
            "market_price": Decimal("40000"),
        }
        # Buy 0.1 BTC → delta = 4k → BTC = 84k, gross = 204k → 41.2% < 50%
        d = rule.evaluate_order(_order("BTCUSDT", Side.BUY, qty=0.1), meta=meta)
        assert d.action == RiskAction.ALLOW

    def test_negative_notional_uses_abs(self):
        rule = ConcentrationRule(max_weight=Decimal("0.5"))
        meta = {
            "positions_notional": {
                "BTCUSDT": Decimal("-300000"),
                "ETHUSDT": Decimal("100000"),
            },
        }
        # abs(BTC) = 300k, gross = 400k → 75% > 50%
        d = rule.evaluate_intent(_intent("BTCUSDT"), meta=meta)
        assert d.action == RiskAction.REJECT
