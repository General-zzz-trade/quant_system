"""Tests for RustPortfolioState — creation, to_dict/from_dict round-trip, field access."""
from __future__ import annotations

import pytest

from _quant_hotpath import RustPortfolioState  # type: ignore[import-untyped]

_SCALE = 100_000_000  # Fd8 scale factor


def _make_portfolio(**overrides):
    """Create a RustPortfolioState with sensible defaults (all Fd8 strings)."""
    defaults = dict(
        total_equity=str(100_000 * _SCALE),
        cash_balance=str(100_000 * _SCALE),
        realized_pnl="0",
        unrealized_pnl="0",
        fees_paid="0",
        gross_exposure="0",
        net_exposure="0",
        leverage="0",
        margin_used="0",
        margin_available=str(100_000 * _SCALE),
        margin_ratio="0",
    )
    defaults.update(overrides)
    return RustPortfolioState(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_create_with_defaults(self):
        ps = _make_portfolio()
        assert str(ps.total_equity) == str(100_000 * _SCALE)

    def test_create_with_custom_equity(self):
        ps = _make_portfolio(total_equity=str(200_000 * _SCALE))
        assert str(ps.total_equity) == str(200_000 * _SCALE)

    def test_leverage_field(self):
        ps = _make_portfolio(leverage=str(3 * _SCALE))
        assert str(ps.leverage) == str(3 * _SCALE)

    def test_exposure_fields(self):
        ps = _make_portfolio(
            gross_exposure=str(50_000 * _SCALE),
            net_exposure=str(30_000 * _SCALE),
        )
        assert str(ps.gross_exposure) == str(50_000 * _SCALE)
        assert str(ps.net_exposure) == str(30_000 * _SCALE)


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_to_dict_returns_dict(self):
        ps = _make_portfolio()
        d = ps.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_required_keys(self):
        ps = _make_portfolio()
        d = ps.to_dict()
        required = {
            "total_equity", "cash_balance", "realized_pnl",
            "unrealized_pnl", "fees_paid", "gross_exposure",
            "net_exposure", "leverage", "margin_used",
            "margin_available", "margin_ratio",
        }
        assert required.issubset(set(d.keys()))

    def test_round_trip_preserves_equity(self):
        ps = _make_portfolio(total_equity=str(55_555 * _SCALE))
        d = ps.to_dict()
        ps2 = RustPortfolioState.from_dict(d)
        assert str(ps2.total_equity) == str(55_555 * _SCALE)

    def test_round_trip_preserves_all_fields(self):
        ps = _make_portfolio(
            total_equity=str(100_000 * _SCALE),
            realized_pnl=str(500 * _SCALE),
            fees_paid=str(25 * _SCALE),
            leverage=str(2 * _SCALE),
        )
        d = ps.to_dict()
        ps2 = RustPortfolioState.from_dict(d)
        assert str(ps2.realized_pnl) == str(500 * _SCALE)
        assert str(ps2.fees_paid) == str(25 * _SCALE)
        assert str(ps2.leverage) == str(2 * _SCALE)

    def test_from_dict_produces_independent_object(self):
        ps = _make_portfolio()
        d = ps.to_dict()
        ps2 = RustPortfolioState.from_dict(d)
        # They should have the same values but be different objects
        assert ps2.total_equity == ps.total_equity
        assert ps2 is not ps


# ---------------------------------------------------------------------------
# Field accessibility
# ---------------------------------------------------------------------------

class TestFieldAccess:
    def test_cash_balance(self):
        ps = _make_portfolio(cash_balance=str(80_000 * _SCALE))
        assert str(ps.cash_balance) == str(80_000 * _SCALE)

    def test_margin_fields(self):
        ps = _make_portfolio(
            margin_used=str(20_000 * _SCALE),
            margin_available=str(80_000 * _SCALE),
            margin_ratio=str(int(0.25 * _SCALE)),
        )
        assert str(ps.margin_used) == str(20_000 * _SCALE)
        assert str(ps.margin_available) == str(80_000 * _SCALE)
