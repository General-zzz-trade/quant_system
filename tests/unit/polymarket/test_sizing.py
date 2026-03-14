"""Tests for polymarket.sizing — Kelly criterion position sizing."""
from __future__ import annotations
import pytest
from polymarket.sizing import kelly_size


def test_kelly_positive_edge():
    """When estimated_prob > market_price, should get positive size."""
    size = kelly_size(estimated_prob=0.6, market_price=0.4, bankroll=10000)
    assert size > 0
    assert size <= 10000 * 0.10  # max_position_pct default


def test_kelly_no_edge():
    """When estimated_prob == market_price, no edge -> 0."""
    size = kelly_size(estimated_prob=0.5, market_price=0.5, bankroll=10000)
    assert size == 0.0


def test_kelly_negative_edge():
    """When estimated_prob < market_price, negative edge -> 0."""
    size = kelly_size(estimated_prob=0.3, market_price=0.5, bankroll=10000)
    assert size == 0.0


def test_kelly_respects_max_position():
    """Even with huge edge, should not exceed max_position_pct."""
    size = kelly_size(
        estimated_prob=0.99,
        market_price=0.01,
        bankroll=100000,
        max_position_pct=0.05,
    )
    assert size <= 100000 * 0.05 + 0.01  # rounding tolerance


def test_kelly_boundary_prices():
    """Edge cases: price at 0 or 1."""
    assert kelly_size(0.5, 0.0, 10000) == 0.0
    assert kelly_size(0.5, 1.0, 10000) == 0.0


def test_kelly_fraction_reduces_size():
    """Half-Kelly should be smaller than full Kelly."""
    # Use a moderate edge so max_position_pct doesn't cap both
    full = kelly_size(0.55, 0.45, 10000, kelly_fraction=1.0, max_position_pct=0.50)
    half = kelly_size(0.55, 0.45, 10000, kelly_fraction=0.5, max_position_pct=0.50)
    assert full > 0
    assert half < full
    assert half == pytest.approx(full * 0.5, rel=0.01)
