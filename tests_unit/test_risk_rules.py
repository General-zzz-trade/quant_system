"""Comprehensive tests for risk module rules and aggregator.

Tests cover:
- LeverageCapRule: evaluate_intent and evaluate_order
- MaxDrawdownRule: evaluate_intent and evaluate_order
- MaxPositionRule: evaluate_intent and evaluate_order
- RiskAggregator: multi-rule evaluation, short-circuit, fail-safe, stats, on_error
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Mapping, Optional, Sequence
from unittest.mock import MagicMock

import pytest

from event.types import Side, Symbol, Venue, Qty, Price, Money
from risk.decisions import (
    RiskAction,
    RiskCode,
    RiskDecision,
    RiskScope,
    RiskViolation,
)
from risk.rules.leverage_cap import LeverageCapRule
from risk.rules.max_drawdown import MaxDrawdownRule, RiskAction as MaxDDRiskAction
from risk.rules.max_position import MaxPositionRule
from risk.aggregator import RiskAggregator, RiskEvalMetaBuilder


# ============================================================
# Test Event Factories
# ============================================================


@dataclass(frozen=True)
class _Qty:
    """Lightweight Qty stub with .value attribute."""
    value: Decimal

    @classmethod
    def of(cls, v) -> "_Qty":
        return cls(value=Decimal(str(v)))


@dataclass(frozen=True)
class _Price:
    """Lightweight Price stub with .value attribute."""
    value: Decimal

    @classmethod
    def of(cls, v) -> "_Price":
        return cls(value=Decimal(str(v)))


@dataclass(frozen=True)
class _Money:
    """Lightweight Money stub with .amount attribute."""
    amount: Decimal

    @classmethod
    def of(cls, v) -> "_Money":
        return cls(amount=Decimal(str(v)))


@dataclass(frozen=True)
class _Symbol:
    """Symbol stub with .normalized property."""
    raw: str

    @property
    def normalized(self) -> str:
        return self.raw.upper()

    def __str__(self) -> str:
        return self.normalized


@dataclass(frozen=True)
class _IntentEvent:
    """Test IntentEvent with fields expected by risk rules."""
    symbol: _Symbol
    side: Side
    target_qty: Optional[_Qty] = None
    target_position_notional: Optional[_Money] = None


@dataclass(frozen=True)
class _OrderEvent:
    """Test OrderEvent with fields expected by risk rules."""
    symbol: _Symbol
    side: Side
    qty: _Qty
    venue: Venue = Venue.BINANCE
    limit_price: Optional[_Price] = None
    reduce_only: bool = False


def make_symbol(raw: str = "BTCUSDT") -> _Symbol:
    """Create a test symbol."""
    return _Symbol(raw=raw)


def make_intent(
    symbol: str = "BTCUSDT",
    side: Side = Side.BUY,
    target_qty: Optional[float] = None,
    target_notional: Optional[float] = None,
) -> _IntentEvent:
    """Create a test IntentEvent."""
    return _IntentEvent(
        symbol=make_symbol(symbol),
        side=side,
        target_qty=_Qty.of(target_qty) if target_qty is not None else None,
        target_position_notional=_Money.of(target_notional) if target_notional is not None else None,
    )


def make_order(
    symbol: str = "BTCUSDT",
    side: Side = Side.BUY,
    qty: float = 1.0,
    limit_price: Optional[float] = None,
    venue: Venue = Venue.BINANCE,
    reduce_only: bool = False,
) -> _OrderEvent:
    """Create a test OrderEvent."""
    return _OrderEvent(
        symbol=make_symbol(symbol),
        side=side,
        qty=_Qty.of(qty),
        venue=venue,
        limit_price=_Price.of(limit_price) if limit_price is not None else None,
        reduce_only=reduce_only,
    )


# ============================================================
# LeverageCapRule Tests
# ============================================================


class TestLeverageCapRule:
    """Tests for LeverageCapRule."""

    def test_allow_when_leverage_below_cap(self) -> None:
        """equity=10000, gross_notional=20000 (2x leverage), cap=3x → ALLOW."""
        # Arrange
        rule = LeverageCapRule(max_leverage=Decimal("3"))
        order = make_order(side=Side.BUY, qty=0.1, limit_price=10000.0)
        meta = {
            "equity": Decimal("10000"),
            "gross_notional": Decimal("20000"),
            "position_qty": Decimal("0"),
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.ALLOW

    def test_reject_when_no_equity(self) -> None:
        """meta has no equity → REJECT."""
        # Arrange
        rule = LeverageCapRule(max_leverage=Decimal("3"))
        order = make_order(side=Side.BUY, qty=1.0, limit_price=10000.0)
        meta = {
            "gross_notional": Decimal("20000"),
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.REJECT
        assert any(v.code == RiskCode.MAX_LEVERAGE for v in decision.violations)

    def test_reject_when_equity_is_zero(self) -> None:
        """equity=0 → REJECT (invalid equity)."""
        # Arrange
        rule = LeverageCapRule(max_leverage=Decimal("3"))
        order = make_order(side=Side.BUY, qty=1.0, limit_price=10000.0)
        meta = {
            "equity": Decimal("0"),
            "gross_notional": Decimal("20000"),
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.REJECT

    def test_reject_when_leverage_exceeds_cap(self) -> None:
        """equity=10000, gross_notional=40000 (4x), cap=3x, new order adds more → REJECT (no headroom)."""
        # Arrange
        rule = LeverageCapRule(max_leverage=Decimal("3"), allow_auto_reduce=False)
        # Order would add more notional: qty=1, price=10000 → +10000 notional
        order = make_order(side=Side.BUY, qty=1.0, limit_price=10000.0)
        meta = {
            "equity": Decimal("10000"),
            "gross_notional": Decimal("40000"),  # already 4x, exceeds 3x cap
            "position_qty": Decimal("0"),
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.REJECT
        assert any(v.code == RiskCode.MAX_LEVERAGE for v in decision.violations)

    def test_reduce_auto_suggest_max_qty(self) -> None:
        """equity=10000, gross=25000, cap=3x, new order would push to 4x → REDUCE with max_qty."""
        # Arrange
        rule = LeverageCapRule(max_leverage=Decimal("3"), allow_auto_reduce=True)
        # cap=3x, equity=10000 → max_gross=30000
        # current gross=25000, headroom=5000
        # order: qty=2, price=10000 → delta_notional=20000 → would exceed cap
        order = make_order(side=Side.BUY, qty=2.0, limit_price=10000.0)
        meta = {
            "equity": Decimal("10000"),
            "gross_notional": Decimal("25000"),
            "position_qty": Decimal("0"),
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.REDUCE
        assert decision.adjustment is not None
        # max_qty = headroom / (price * mult) = 5000 / 10000 = 0.5
        assert decision.adjustment.max_qty == pytest.approx(0.5, rel=1e-6)

    def test_allow_reducing_exposure_order(self) -> None:
        """Current position is long, order is SELL (reducing) → ALLOW even if over cap."""
        # Arrange
        rule = LeverageCapRule(max_leverage=Decimal("3"))
        order = make_order(side=Side.SELL, qty=1.0, limit_price=10000.0)
        meta = {
            "equity": Decimal("10000"),
            "gross_notional": Decimal("50000"),  # already 5x, way over cap
            "position_qty": Decimal("2"),  # long 2 units, selling reduces exposure
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.ALLOW
        assert "reduce_only_or_reducing" in decision.tags

    def test_intent_skip_when_gross_missing(self) -> None:
        """No gross_notional in meta → ALLOW (intent is conservative, order will enforce)."""
        # Arrange
        rule = LeverageCapRule(max_leverage=Decimal("3"))
        intent = make_intent(side=Side.BUY)
        meta = {
            "equity": Decimal("10000"),
            # no gross_notional
        }

        # Act
        decision = rule.evaluate_intent(intent, meta=meta)

        # Assert
        assert decision.action == RiskAction.ALLOW
        assert "intent_skip_missing_gross" in decision.tags

    def test_intent_reject_when_leverage_exceeded(self) -> None:
        """Intent evaluation: leverage already exceeded → REJECT."""
        # Arrange
        rule = LeverageCapRule(max_leverage=Decimal("3"))
        intent = make_intent(side=Side.BUY)
        meta = {
            "equity": Decimal("10000"),
            "gross_notional": Decimal("40000"),  # 4x > 3x cap
        }

        # Act
        decision = rule.evaluate_intent(intent, meta=meta)

        # Assert
        assert decision.action == RiskAction.REJECT

    def test_intent_reject_when_no_equity(self) -> None:
        """Intent evaluation: no equity in meta → REJECT."""
        # Arrange
        rule = LeverageCapRule(max_leverage=Decimal("3"))
        intent = make_intent(side=Side.BUY)
        meta = {}

        # Act
        decision = rule.evaluate_intent(intent, meta=meta)

        # Assert
        assert decision.action == RiskAction.REJECT

    def test_allow_when_order_projected_within_cap(self) -> None:
        """Order that keeps projected leverage within cap → ALLOW."""
        # Arrange
        rule = LeverageCapRule(max_leverage=Decimal("3"))
        # equity=10000, gross=20000 (2x), order adds 1 unit at 5000 → delta=5000 → projected=25000 (2.5x < 3x)
        order = make_order(side=Side.BUY, qty=1.0, limit_price=5000.0)
        meta = {
            "equity": Decimal("10000"),
            "gross_notional": Decimal("20000"),
            "position_qty": Decimal("0"),
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.ALLOW

    def test_reject_when_no_headroom(self) -> None:
        """When gross already at or above cap and order would increase it → REJECT with no_headroom."""
        # Arrange
        rule = LeverageCapRule(max_leverage=Decimal("3"), allow_auto_reduce=True)
        order = make_order(side=Side.BUY, qty=1.0, limit_price=10000.0)
        meta = {
            "equity": Decimal("10000"),
            "gross_notional": Decimal("30000"),  # exactly at 3x cap, no headroom
            "position_qty": Decimal("0"),
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.REJECT
        assert "no_headroom" in decision.tags

    def test_per_symbol_max_leverage_override(self) -> None:
        """Per-symbol cap overrides global cap."""
        # Arrange
        rule = LeverageCapRule(
            max_leverage=Decimal("3"),
            per_symbol_max_leverage={"BTCUSDT": Decimal("5")},
            allow_auto_reduce=False,
        )
        # With 5x cap, equity=10000, gross=40000 (4x) + new order delta → still within 5x if delta is small
        order = make_order(symbol="BTCUSDT", side=Side.BUY, qty=0.5, limit_price=10000.0)
        meta = {
            "equity": Decimal("10000"),
            "gross_notional": Decimal("40000"),
            "position_qty": Decimal("0"),
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert: 40000 + 5000 = 45000 < 50000 (5x * 10000), so ALLOW
        assert decision.action == RiskAction.ALLOW

    def test_reject_when_missing_price_and_reject_if_missing(self) -> None:
        """No limit_price and no market_price in meta → REJECT (when reject_if_missing_price=True)."""
        # Arrange
        rule = LeverageCapRule(max_leverage=Decimal("3"), reject_if_missing_price=True)
        order = make_order(side=Side.BUY, qty=1.0, limit_price=None)
        meta = {
            "equity": Decimal("10000"),
            "gross_notional": Decimal("25000"),  # 2.5x, adding order would exceed 3x
            "position_qty": Decimal("0"),
            # no market_price
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.REJECT


# ============================================================
# MaxDrawdownRule Tests
# ============================================================


class TestMaxDrawdownRule:
    """Tests for MaxDrawdownRule."""

    def test_allow_within_drawdown(self) -> None:
        """peak_equity=10000, current=9000 (10% drawdown), threshold=20% → ALLOW."""
        # Arrange
        rule = MaxDrawdownRule(max_drawdown_pct=Decimal("0.20"))
        order = make_order(side=Side.BUY, qty=1.0, limit_price=10000.0)
        meta = {
            "equity": Decimal("9000"),
            "peak_equity": Decimal("10000"),
            # drawdown = (10000 - 9000) / 10000 = 0.10, within 0.20 threshold
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.ALLOW

    def test_kill_when_drawdown_exceeds_threshold(self) -> None:
        """peak_equity=10000, current=7000 (30% drawdown), threshold=20% → KILL."""
        # Arrange
        rule = MaxDrawdownRule(
            max_drawdown_pct=Decimal("0.20"),
            action_on_breach=RiskAction.KILL,
        )
        order = make_order(side=Side.BUY, qty=1.0, limit_price=10000.0)
        meta = {
            "equity": Decimal("7000"),
            "peak_equity": Decimal("10000"),
            # drawdown = (10000 - 7000) / 10000 = 0.30 > 0.20
            "position_qty": Decimal("0"),
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.KILL
        assert any(v.code == RiskCode.MAX_DRAWDOWN for v in decision.violations)

    def test_reject_when_drawdown_exceeds_and_action_is_reject(self) -> None:
        """When action_on_breach=REJECT and drawdown exceeds threshold → REJECT."""
        # Arrange
        rule = MaxDrawdownRule(
            max_drawdown_pct=Decimal("0.20"),
            action_on_breach=RiskAction.REJECT,
        )
        order = make_order(side=Side.BUY, qty=1.0, limit_price=10000.0)
        meta = {
            "equity": Decimal("7000"),
            "peak_equity": Decimal("10000"),
            "position_qty": Decimal("0"),
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.REJECT

    def test_allow_reduce_only_during_drawdown(self) -> None:
        """During drawdown breach, reduce-only order is allowed when allow_reduce_after_breach=True."""
        # Arrange
        rule = MaxDrawdownRule(
            max_drawdown_pct=Decimal("0.20"),
            action_on_breach=RiskAction.KILL,
            allow_reduce_after_breach=True,
        )
        # Long position, SELL order reduces exposure
        order = make_order(side=Side.SELL, qty=1.0, limit_price=10000.0)
        meta = {
            "equity": Decimal("7000"),
            "peak_equity": Decimal("10000"),  # 30% drawdown, over threshold
            "position_qty": Decimal("2"),  # long 2 units; selling reduces
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.ALLOW
        assert "reduce_only_after_dd" in decision.tags

    def test_allow_reduce_only_flag_during_drawdown(self) -> None:
        """Order with reduce_only=True is allowed during drawdown."""
        # Arrange
        rule = MaxDrawdownRule(
            max_drawdown_pct=Decimal("0.20"),
            action_on_breach=RiskAction.KILL,
            allow_reduce_after_breach=True,
        )
        order = make_order(side=Side.SELL, qty=1.0, reduce_only=True)
        meta = {
            "equity": Decimal("6000"),
            "peak_equity": Decimal("10000"),  # 40% drawdown
            "position_qty": Decimal("1"),
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.ALLOW

    def test_reject_when_cannot_evaluate_drawdown(self) -> None:
        """No drawdown data and reject_if_cannot_evaluate=True → REJECT."""
        # Arrange
        rule = MaxDrawdownRule(
            max_drawdown_pct=Decimal("0.20"),
            reject_if_cannot_evaluate=True,
        )
        order = make_order(side=Side.BUY, qty=1.0)
        meta = {}  # no equity, no peak_equity, no drawdown_pct

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.REJECT
        assert any(v.code == RiskCode.MAX_DRAWDOWN for v in decision.violations)

    def test_allow_when_cannot_evaluate_and_reject_false(self) -> None:
        """No drawdown data and reject_if_cannot_evaluate=False → ALLOW."""
        # Arrange
        rule = MaxDrawdownRule(
            max_drawdown_pct=Decimal("0.20"),
            reject_if_cannot_evaluate=False,
        )
        order = make_order(side=Side.BUY, qty=1.0)
        meta = {}

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.ALLOW

    def test_intent_skip_when_drawdown_missing(self) -> None:
        """Intent evaluation with no drawdown data → ALLOW (conservative skip)."""
        # Arrange
        rule = MaxDrawdownRule(max_drawdown_pct=Decimal("0.20"))
        intent = make_intent(side=Side.BUY)
        meta = {}  # no drawdown data

        # Act
        decision = rule.evaluate_intent(intent, meta=meta)

        # Assert
        assert decision.action == RiskAction.ALLOW
        assert "intent_skip_missing_dd" in decision.tags

    def test_intent_reject_when_drawdown_exceeded(self) -> None:
        """Intent evaluation with exceeded drawdown → REJECT."""
        # Arrange
        rule = MaxDrawdownRule(max_drawdown_pct=Decimal("0.20"))
        intent = make_intent(side=Side.BUY)
        meta = {
            "drawdown_pct": Decimal("0.30"),  # 30% > 20% cap
        }

        # Act
        decision = rule.evaluate_intent(intent, meta=meta)

        # Assert
        assert decision.action == RiskAction.REJECT

    def test_allow_when_at_drawdown_boundary(self) -> None:
        """Drawdown exactly at threshold → ALLOW (not exceeded)."""
        # Arrange
        rule = MaxDrawdownRule(max_drawdown_pct=Decimal("0.20"))
        order = make_order(side=Side.BUY, qty=1.0)
        meta = {
            "drawdown_pct": Decimal("0.20"),  # exactly at threshold, not exceeded
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.ALLOW

    def test_drawdown_from_precomputed_pct(self) -> None:
        """Use pre-computed drawdown_pct from meta if available."""
        # Arrange
        rule = MaxDrawdownRule(max_drawdown_pct=Decimal("0.15"))
        order = make_order(side=Side.BUY, qty=1.0)
        meta = {
            "drawdown_pct": Decimal("0.25"),  # 25% > 15% threshold
            "position_qty": Decimal("0"),
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.KILL

    def test_per_strategy_drawdown_override(self) -> None:
        """Per-strategy drawdown cap overrides global threshold."""
        # Arrange
        rule = MaxDrawdownRule(
            max_drawdown_pct=Decimal("0.20"),
            per_strategy_max_dd={"conservative": Decimal("0.10")},
        )
        order = make_order(side=Side.BUY, qty=1.0)
        meta = {
            "drawdown_pct": Decimal("0.15"),  # 15% < global 20% but > per-strategy 10%
            "strategy_id": "conservative",
            "position_qty": Decimal("0"),
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.KILL

    def test_block_new_long_after_drawdown_when_reduce_not_allowed(self) -> None:
        """When allow_reduce_after_breach=False, even reducing orders are blocked."""
        # Arrange
        rule = MaxDrawdownRule(
            max_drawdown_pct=Decimal("0.20"),
            action_on_breach=RiskAction.KILL,
            allow_reduce_after_breach=False,
        )
        order = make_order(side=Side.SELL, qty=1.0)  # reducing, but rule blocks all
        meta = {
            "drawdown_pct": Decimal("0.30"),  # breached
            "position_qty": Decimal("2"),
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.KILL


# ============================================================
# MaxPositionRule Tests
# ============================================================


class TestMaxPositionRule:
    """Tests for MaxPositionRule."""

    def test_allow_within_position_limit(self) -> None:
        """Order keeps position within max_abs_qty limit → ALLOW."""
        # Arrange
        rule = MaxPositionRule(max_abs_qty=Decimal("10"))
        order = make_order(side=Side.BUY, qty=3.0)
        meta = {
            "position_qty": Decimal("5"),  # current: 5, after: 8 < 10 limit
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.ALLOW

    def test_reject_exceeds_position_limit(self) -> None:
        """Order would push position over max_abs_qty → REJECT (auto_reduce disabled)."""
        # Arrange
        rule = MaxPositionRule(max_abs_qty=Decimal("10"), allow_auto_reduce=False)
        order = make_order(side=Side.BUY, qty=6.0)
        meta = {
            "position_qty": Decimal("7"),  # current: 7, after: 13 > 10 limit
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.REJECT
        assert any(v.code == RiskCode.MAX_POSITION for v in decision.violations)

    def test_reduce_suggests_max_qty_at_limit(self) -> None:
        """Order exceeds limit but auto_reduce=True → REDUCE with max_qty suggestion."""
        # Arrange
        rule = MaxPositionRule(max_abs_qty=Decimal("10"), allow_auto_reduce=True)
        order = make_order(side=Side.BUY, qty=6.0)
        meta = {
            "position_qty": Decimal("7"),  # current: 7, headroom: 3
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.REDUCE
        assert decision.adjustment is not None
        # headroom = 10 - 7 = 3
        assert decision.adjustment.max_qty == pytest.approx(3.0, rel=1e-6)

    def test_allow_reducing_position_even_over_limit(self) -> None:
        """SELL order reduces existing long position → ALLOW even if currently over limit."""
        # Arrange
        rule = MaxPositionRule(max_abs_qty=Decimal("10"), allow_auto_reduce=False)
        order = make_order(side=Side.SELL, qty=2.0)
        meta = {
            "position_qty": Decimal("12"),  # over limit, but selling reduces it
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.ALLOW
        assert "reduce_only_or_reducing" in decision.tags

    def test_allow_reduce_only_order(self) -> None:
        """Order with reduce_only=True → ALLOW regardless of position."""
        # Arrange
        rule = MaxPositionRule(max_abs_qty=Decimal("10"), allow_auto_reduce=False)
        order = make_order(side=Side.BUY, qty=5.0, reduce_only=True)
        meta = {
            "position_qty": Decimal("8"),
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.ALLOW

    def test_reject_when_no_headroom(self) -> None:
        """Current position at limit already → REJECT with no_headroom tag."""
        # Arrange
        rule = MaxPositionRule(max_abs_qty=Decimal("10"), allow_auto_reduce=True)
        order = make_order(side=Side.BUY, qty=2.0)
        meta = {
            "position_qty": Decimal("10"),  # exactly at limit, no headroom
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.REJECT
        assert "no_headroom" in decision.tags

    def test_allow_with_no_limits_configured(self) -> None:
        """No limits configured → always ALLOW."""
        # Arrange
        rule = MaxPositionRule()  # max_abs_qty=None, max_abs_notional=None
        order = make_order(side=Side.BUY, qty=1000.0)
        meta = {}

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.ALLOW

    def test_per_symbol_limit_override(self) -> None:
        """Per-symbol limit overrides global limit."""
        # Arrange
        rule = MaxPositionRule(
            max_abs_qty=Decimal("10"),
            per_symbol_limits={"BTCUSDT": (Decimal("5"), None)},
            allow_auto_reduce=False,
        )
        order = make_order(symbol="BTCUSDT", side=Side.BUY, qty=3.0)
        meta = {
            "position_qty": Decimal("4"),  # 4 + 3 = 7 > per-symbol 5
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.REJECT

    def test_intent_allow_within_limit(self) -> None:
        """Intent with target_qty within limit → ALLOW."""
        # Arrange
        rule = MaxPositionRule(max_abs_qty=Decimal("10"))
        intent = make_intent(side=Side.BUY, target_qty=3.0)
        meta = {
            "position_qty": Decimal("5"),  # 5 + 3 = 8 < 10
        }

        # Act
        decision = rule.evaluate_intent(intent, meta=meta)

        # Assert
        assert decision.action == RiskAction.ALLOW

    def test_intent_reject_exceeds_limit(self) -> None:
        """Intent with target_qty exceeding limit → REJECT."""
        # Arrange
        rule = MaxPositionRule(max_abs_qty=Decimal("10"))
        intent = make_intent(side=Side.BUY, target_qty=6.0)
        meta = {
            "position_qty": Decimal("7"),  # 7 + 6 = 13 > 10
        }

        # Act
        decision = rule.evaluate_intent(intent, meta=meta)

        # Assert
        assert decision.action == RiskAction.REJECT
        assert any(v.code == RiskCode.MAX_POSITION for v in decision.violations)

    def test_intent_allow_reducing_even_over_limit(self) -> None:
        """Intent that reduces existing long position → ALLOW even over limit."""
        # Arrange
        rule = MaxPositionRule(max_abs_qty=Decimal("10"))
        intent = make_intent(side=Side.SELL, target_qty=2.0)
        meta = {
            "position_qty": Decimal("12"),  # over limit, selling reduces it
        }

        # Act
        decision = rule.evaluate_intent(intent, meta=meta)

        # Assert
        assert decision.action == RiskAction.ALLOW

    def test_short_position_tracked_correctly(self) -> None:
        """Short position (negative qty) is checked against abs limit."""
        # Arrange
        rule = MaxPositionRule(max_abs_qty=Decimal("10"), allow_auto_reduce=False)
        order = make_order(side=Side.SELL, qty=5.0)
        meta = {
            "position_qty": Decimal("-7"),  # currently short 7, selling 5 more → -12, |12| > 10
        }

        # Act
        decision = rule.evaluate_order(order, meta=meta)

        # Assert
        assert decision.action == RiskAction.REJECT


# ============================================================
# RiskAggregator Tests
# ============================================================


class _AlwaysAllowRule:
    """Stub rule that always ALLOWs."""
    name: str = "always_allow"

    def evaluate_intent(self, intent, *, meta) -> RiskDecision:
        return RiskDecision.allow(tags=(self.name,))

    def evaluate_order(self, order, *, meta) -> RiskDecision:
        return RiskDecision.allow(tags=(self.name,))


class _AlwaysRejectRule:
    """Stub rule that always REJECTs."""
    name: str = "always_reject"

    def evaluate_intent(self, intent, *, meta) -> RiskDecision:
        v = RiskViolation(code=RiskCode.UNKNOWN, message="reject rule", scope=RiskScope.GLOBAL)
        return RiskDecision.reject((v,), scope=RiskScope.GLOBAL, tags=(self.name,))

    def evaluate_order(self, order, *, meta) -> RiskDecision:
        v = RiskViolation(code=RiskCode.UNKNOWN, message="reject rule", scope=RiskScope.GLOBAL)
        return RiskDecision.reject((v,), scope=RiskScope.GLOBAL, tags=(self.name,))


class _AlwaysKillRule:
    """Stub rule that always KILLs."""
    name: str = "always_kill"

    def evaluate_intent(self, intent, *, meta) -> RiskDecision:
        v = RiskViolation(code=RiskCode.UNKNOWN, message="kill rule", scope=RiskScope.GLOBAL)
        return RiskDecision.kill((v,), scope=RiskScope.GLOBAL, tags=(self.name,))

    def evaluate_order(self, order, *, meta) -> RiskDecision:
        v = RiskViolation(code=RiskCode.UNKNOWN, message="kill rule", scope=RiskScope.GLOBAL)
        return RiskDecision.kill((v,), scope=RiskScope.GLOBAL, tags=(self.name,))


class _ExplodingRule:
    """Stub rule that always raises an exception."""
    name: str = "exploding_rule"

    def evaluate_intent(self, intent, *, meta) -> RiskDecision:
        raise RuntimeError("Simulated rule failure")

    def evaluate_order(self, order, *, meta) -> RiskDecision:
        raise RuntimeError("Simulated rule failure")


def _make_meta_builder(meta: dict = None) -> RiskEvalMetaBuilder:
    """Create a meta builder that returns a fixed meta dict."""
    fixed_meta = meta or {}
    return RiskEvalMetaBuilder(
        build_for_intent=lambda intent: fixed_meta,
        build_for_order=lambda order: fixed_meta,
    )


class TestRiskAggregator:
    """Tests for RiskAggregator."""

    def test_evaluates_all_rules_all_allow(self) -> None:
        """Multiple rules, all ALLOW → merged ALLOW."""
        # Arrange
        rule1 = _AlwaysAllowRule()
        rule1.name = "rule_1"
        rule2 = _AlwaysAllowRule()
        rule2.name = "rule_2"

        agg = RiskAggregator(
            rules=[rule1, rule2],
            meta_builder=_make_meta_builder(),
        )
        order = make_order(side=Side.BUY, qty=1.0)

        # Act
        decision = agg.evaluate_order(order)

        # Assert
        assert decision.action == RiskAction.ALLOW

    def test_stops_on_kill_when_configured(self) -> None:
        """First rule returns KILL, stop_on_kill=True → stops evaluating remaining rules."""
        # Arrange
        kill_rule = _AlwaysKillRule()
        second_rule = _AlwaysAllowRule()
        call_count = [0]

        original_evaluate = second_rule.evaluate_order
        def counting_evaluate(order, *, meta):
            call_count[0] += 1
            return original_evaluate(order, meta=meta)
        second_rule.evaluate_order = counting_evaluate

        agg = RiskAggregator(
            rules=[kill_rule, second_rule],
            meta_builder=_make_meta_builder(),
            stop_on_kill=True,
        )
        order = make_order(side=Side.BUY, qty=1.0)

        # Act
        decision = agg.evaluate_order(order)

        # Assert
        assert decision.action == RiskAction.KILL
        # Second rule should NOT have been called
        assert call_count[0] == 0

    def test_continues_on_kill_when_stop_on_kill_false(self) -> None:
        """stop_on_kill=False → continues evaluating after KILL."""
        # Arrange
        call_count = [0]

        class _CountingAllowRule:
            name = "counting_allow"
            def evaluate_order(self, order, *, meta):
                call_count[0] += 1
                return RiskDecision.allow(tags=(self.name,))
            def evaluate_intent(self, intent, *, meta):
                return RiskDecision.allow()

        agg = RiskAggregator(
            rules=[_AlwaysKillRule(), _CountingAllowRule()],
            meta_builder=_make_meta_builder(),
            stop_on_kill=False,
        )
        order = make_order(side=Side.BUY, qty=1.0)

        # Act
        decision = agg.evaluate_order(order)

        # Assert: still KILL (due to merge), but second rule was called
        assert decision.action == RiskAction.KILL
        assert call_count[0] == 1

    def test_fail_safe_on_rule_exception(self) -> None:
        """One rule raises an exception → fail_safe REJECT."""
        # Arrange
        agg = RiskAggregator(
            rules=[_ExplodingRule()],
            meta_builder=_make_meta_builder(),
            fail_safe_action=RiskAction.REJECT,
        )
        order = make_order(side=Side.BUY, qty=1.0)

        # Act
        decision = agg.evaluate_order(order)

        # Assert
        assert decision.action == RiskAction.REJECT
        assert "fail_safe" in decision.tags

    def test_fail_safe_kill_on_rule_exception(self) -> None:
        """fail_safe_action=KILL → exception causes KILL decision."""
        # Arrange
        agg = RiskAggregator(
            rules=[_ExplodingRule()],
            meta_builder=_make_meta_builder(),
            fail_safe_action=RiskAction.KILL,
        )
        order = make_order(side=Side.BUY, qty=1.0)

        # Act
        decision = agg.evaluate_order(order)

        # Assert
        assert decision.action == RiskAction.KILL
        assert "fail_safe" in decision.tags

    def test_stats_update_calls_count(self) -> None:
        """After evaluating, stats.calls is incremented for enabled rules."""
        # Arrange
        rule = _AlwaysAllowRule()
        agg = RiskAggregator(
            rules=[rule],
            meta_builder=_make_meta_builder(),
        )
        order = make_order(side=Side.BUY, qty=1.0)

        # Act: evaluate multiple times
        agg.evaluate_order(order)
        agg.evaluate_order(order)
        agg.evaluate_order(order)

        # Assert
        snap = agg.snapshot()
        stats = {s.name: s for s in snap.stats}
        assert stats["always_allow"].calls == 3
        assert stats["always_allow"].allow == 3

    def test_stats_track_reject_count(self) -> None:
        """Stats correctly track reject count."""
        # Arrange
        rule = _AlwaysRejectRule()
        agg = RiskAggregator(
            rules=[rule],
            meta_builder=_make_meta_builder(),
        )
        order = make_order(side=Side.BUY, qty=1.0)

        # Act
        agg.evaluate_order(order)
        agg.evaluate_order(order)

        # Assert
        snap = agg.snapshot()
        stats = {s.name: s for s in snap.stats}
        assert stats["always_reject"].reject == 2

    def test_stats_track_error_count(self) -> None:
        """Stats correctly track error count when rule throws."""
        # Arrange
        agg = RiskAggregator(
            rules=[_ExplodingRule()],
            meta_builder=_make_meta_builder(),
        )
        order = make_order(side=Side.BUY, qty=1.0)

        # Act
        agg.evaluate_order(order)

        # Assert
        snap = agg.snapshot()
        stats = {s.name: s for s in snap.stats}
        assert stats["exploding_rule"].errors == 1

    def test_disable_rule_skips_it(self) -> None:
        """Disable a rule, it should not be evaluated."""
        # Arrange
        call_count = [0]

        class _TrackingRule:
            name = "tracking_rule"
            def evaluate_order(self, order, *, meta):
                call_count[0] += 1
                return RiskDecision.allow()
            def evaluate_intent(self, intent, *, meta):
                call_count[0] += 1
                return RiskDecision.allow()

        agg = RiskAggregator(
            rules=[_TrackingRule()],
            meta_builder=_make_meta_builder(),
        )
        agg.disable("tracking_rule")
        order = make_order(side=Side.BUY, qty=1.0)

        # Act
        decision = agg.evaluate_order(order)

        # Assert
        assert call_count[0] == 0
        # With no rules evaluated, merge_decisions returns ALLOW
        assert decision.action == RiskAction.ALLOW

    def test_disabled_rule_appears_in_snapshot(self) -> None:
        """Disabled rule appears in snapshot.disabled."""
        # Arrange
        rule = _AlwaysAllowRule()
        agg = RiskAggregator(
            rules=[rule],
            meta_builder=_make_meta_builder(),
        )
        agg.disable("always_allow")

        # Act
        snap = agg.snapshot()

        # Assert
        assert "always_allow" in snap.disabled
        assert "always_allow" not in snap.enabled

    def test_enable_disabled_rule_resumes_evaluation(self) -> None:
        """Re-enabling a disabled rule causes it to be evaluated again."""
        # Arrange
        rule = _AlwaysRejectRule()
        agg = RiskAggregator(
            rules=[rule],
            meta_builder=_make_meta_builder(),
        )
        agg.disable("always_reject")
        order = make_order(side=Side.BUY, qty=1.0)

        # Disabled: should ALLOW (no rules evaluated)
        d1 = agg.evaluate_order(order)
        assert d1.action == RiskAction.ALLOW

        # Re-enable
        agg.enable("always_reject")

        # Act
        d2 = agg.evaluate_order(order)

        # Assert
        assert d2.action == RiskAction.REJECT

    def test_on_error_callback_called(self) -> None:
        """When rule throws, on_error callback is invoked."""
        # Arrange
        error_calls = []

        def on_error(rule_name, mode, exc, meta):
            error_calls.append((rule_name, mode, type(exc).__name__))

        agg = RiskAggregator(
            rules=[_ExplodingRule()],
            meta_builder=_make_meta_builder(),
            on_error=on_error,
        )
        order = make_order(side=Side.BUY, qty=1.0)

        # Act
        agg.evaluate_order(order)

        # Assert
        assert len(error_calls) == 1
        rule_name, mode, exc_type = error_calls[0]
        assert rule_name == "exploding_rule"
        assert mode == "order"
        assert exc_type == "RuntimeError"

    def test_on_error_callback_called_for_intent(self) -> None:
        """on_error callback is also invoked for intent evaluation failures."""
        # Arrange
        error_calls = []

        def on_error(rule_name, mode, exc, meta):
            error_calls.append((rule_name, mode))

        agg = RiskAggregator(
            rules=[_ExplodingRule()],
            meta_builder=_make_meta_builder(),
            on_error=on_error,
        )
        intent = make_intent(side=Side.BUY)

        # Act
        agg.evaluate_intent(intent)

        # Assert
        assert len(error_calls) == 1
        assert error_calls[0] == ("exploding_rule", "intent")

    def test_duplicate_rule_names_raises(self) -> None:
        """Duplicate rule names raise RiskAggregatorError."""
        # Arrange
        rule1 = _AlwaysAllowRule()
        rule2 = _AlwaysAllowRule()  # same name "always_allow"

        # Act / Assert
        from risk.aggregator import RiskAggregatorError
        with pytest.raises(RiskAggregatorError, match="唯一"):
            RiskAggregator(
                rules=[rule1, rule2],
                meta_builder=_make_meta_builder(),
            )

    def test_empty_rules_raises(self) -> None:
        """Empty rules list raises RiskAggregatorError."""
        from risk.aggregator import RiskAggregatorError
        with pytest.raises(RiskAggregatorError, match="不能为空"):
            RiskAggregator(
                rules=[],
                meta_builder=_make_meta_builder(),
            )

    def test_initial_disabled_rule_not_evaluated(self) -> None:
        """Rules in disabled= list at construction are not evaluated."""
        # Arrange
        call_count = [0]

        class _TrackedRule:
            name = "track"
            def evaluate_order(self, order, *, meta):
                call_count[0] += 1
                return RiskDecision.allow()
            def evaluate_intent(self, intent, *, meta):
                return RiskDecision.allow()

        agg = RiskAggregator(
            rules=[_TrackedRule()],
            meta_builder=_make_meta_builder(),
            disabled=["track"],
        )
        order = make_order(side=Side.BUY, qty=1.0)

        # Act
        agg.evaluate_order(order)

        # Assert
        assert call_count[0] == 0

    def test_multiple_violations_merged(self) -> None:
        """Multiple rules with violations → all violations appear in merged decision."""
        # Arrange
        class _RejectWithCodeRule:
            def __init__(self, name, code):
                self.name = name
                self._code = code
            def evaluate_order(self, order, *, meta):
                v = RiskViolation(code=self._code, message=f"violation {self._code}")
                return RiskDecision.reject((v,), scope=RiskScope.GLOBAL)
            def evaluate_intent(self, intent, *, meta):
                return RiskDecision.allow()

        rule1 = _RejectWithCodeRule("rule_a", RiskCode.MAX_LEVERAGE)
        rule2 = _RejectWithCodeRule("rule_b", RiskCode.MAX_DRAWDOWN)

        agg = RiskAggregator(
            rules=[rule1, rule2],
            meta_builder=_make_meta_builder(),
            stop_on_reject=False,
        )
        order = make_order(side=Side.BUY, qty=1.0)

        # Act
        decision = agg.evaluate_order(order)

        # Assert
        assert decision.action == RiskAction.REJECT
        codes = {v.code for v in decision.violations}
        assert RiskCode.MAX_LEVERAGE in codes
        assert RiskCode.MAX_DRAWDOWN in codes

    def test_intent_evaluation_via_aggregator(self) -> None:
        """RiskAggregator.evaluate_intent routes correctly to rules."""
        # Arrange
        rule = _AlwaysRejectRule()
        agg = RiskAggregator(
            rules=[rule],
            meta_builder=_make_meta_builder(),
        )
        intent = make_intent(side=Side.BUY)

        # Act
        decision = agg.evaluate_intent(intent)

        # Assert
        assert decision.action == RiskAction.REJECT
