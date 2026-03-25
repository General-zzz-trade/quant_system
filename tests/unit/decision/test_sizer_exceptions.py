"""Exception and boundary-condition tests for AdaptivePositionSizer.

Covers: zero/negative equity, zero price, zero max_qty, zero leverage,
step_size edge cases, z_scale boundaries.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch


from decision.sizing.adaptive import AdaptivePositionSizer


# ── helpers ────────────────────────────────────────────────────────


def _snap(
    equity: float, price: float, symbol: str = "BTCUSDT",
) -> MagicMock:
    """Build a minimal MagicMock StateSnapshot (Python fallback path)."""
    snap = MagicMock()
    # Use balance (not balance_f) to exercise the fallback code path
    acc = MagicMock()
    acc.balance_f = equity
    acc.balance = Decimal(str(equity))
    snap.account = acc
    market = MagicMock()
    market.close_f = price
    market.close = Decimal(str(price))
    snap.markets = {symbol: market}
    return snap


# ── tests ──────────────────────────────────────────────────────────


@patch("decision.sizing.adaptive._RUST_SIZER", False)
class TestSizerExceptions:
    """Edge cases for AdaptivePositionSizer.target_qty() (Python path)."""

    def test_equity_zero_returns_min_size(self):
        sizer = AdaptivePositionSizer(runner_key="BTCUSDT", min_size=0.001)
        snap = _snap(equity=0, price=60000.0)
        qty = sizer.target_qty(snap, "BTCUSDT")
        assert qty == Decimal("0.001")

    def test_equity_negative_returns_min_size(self):
        sizer = AdaptivePositionSizer(runner_key="BTCUSDT", min_size=0.001)
        snap = _snap(equity=-1000, price=60000.0)
        qty = sizer.target_qty(snap, "BTCUSDT")
        assert qty == Decimal("0.001")

    def test_price_zero_returns_min_size(self):
        sizer = AdaptivePositionSizer(runner_key="BTCUSDT", min_size=0.001)
        snap = _snap(equity=5000, price=0.0)
        qty = sizer.target_qty(snap, "BTCUSDT")
        assert qty == Decimal("0.001")

    def test_price_negative_returns_min_size(self):
        sizer = AdaptivePositionSizer(runner_key="BTCUSDT", min_size=0.001)
        snap = _snap(equity=5000, price=-100.0)
        # price < 0 is treated as price <= 0
        qty = sizer.target_qty(snap, "BTCUSDT")
        assert qty == Decimal("0.001")

    def test_max_qty_zero_means_unlimited(self):
        sizer = AdaptivePositionSizer(
            runner_key="BTCUSDT_4h", max_qty=0, min_size=0.001,
        )
        snap = _snap(equity=50000, price=60000.0)
        qty = sizer.target_qty(snap, "BTCUSDT")
        # max_qty=0 → no upper clamp, should be > min_size
        assert qty > Decimal("0.001")

    def test_max_qty_positive_clamps(self):
        sizer = AdaptivePositionSizer(
            runner_key="BTCUSDT_4h", max_qty=0.005, min_size=0.001,
        )
        snap = _snap(equity=50000, price=60000.0)
        qty = sizer.target_qty(snap, "BTCUSDT")
        assert qty <= Decimal("0.005")

    def test_leverage_zero_produces_zero_notional(self):
        sizer = AdaptivePositionSizer(runner_key="BTCUSDT", min_size=0.001)
        snap = _snap(equity=5000, price=60000.0)
        qty = sizer.target_qty(snap, "BTCUSDT", leverage=0.0)
        # notional = equity * cap * 0 = 0, then max(0/price, min_size) = min_size
        assert qty == Decimal("0.001")

    def test_leverage_negative_returns_min_size(self):
        sizer = AdaptivePositionSizer(runner_key="BTCUSDT", min_size=0.001)
        snap = _snap(equity=5000, price=60000.0)
        qty = sizer.target_qty(snap, "BTCUSDT", leverage=-5.0)
        # Negative notional → size < 0 → clamped to min_size
        assert qty >= Decimal("0.001")

    def test_z_scale_zero_returns_min_size(self):
        sizer = AdaptivePositionSizer(runner_key="BTCUSDT", min_size=0.001)
        snap = _snap(equity=5000, price=60000.0)
        qty = sizer.target_qty(snap, "BTCUSDT", z_scale=0.0)
        # size = notional/price * 0 = 0 → min_size
        assert qty == Decimal("0.001")

    def test_z_scale_large_increases_qty(self):
        sizer = AdaptivePositionSizer(runner_key="BTCUSDT_4h", min_size=0.001)
        snap = _snap(equity=5000, price=60000.0)
        qty_normal = sizer.target_qty(snap, "BTCUSDT", z_scale=1.0)
        qty_large = sizer.target_qty(snap, "BTCUSDT", z_scale=2.0)
        assert qty_large > qty_normal

    def test_step_size_zero_no_crash(self):
        """step_size=0 → _round_to_step returns raw value."""
        sizer = AdaptivePositionSizer(
            runner_key="BTCUSDT", step_size=0.0, min_size=0.001,
        )
        rounded = sizer._round_to_step(0.12345)
        assert rounded == Decimal("0.12345")

    def test_step_size_negative_no_crash(self):
        """step_size < 0 → treated same as step_size=0."""
        sizer = AdaptivePositionSizer(
            runner_key="BTCUSDT", step_size=-0.01, min_size=0.001,
        )
        rounded = sizer._round_to_step(0.12345)
        assert rounded == Decimal("0.12345")

    def test_round_to_step_very_small_step(self):
        sizer = AdaptivePositionSizer(
            runner_key="BTCUSDT", step_size=0.00001,
        )
        result = sizer._round_to_step(0.123456)
        assert result == Decimal("0.12345")

    def test_equity_tier_boundaries(self):
        assert AdaptivePositionSizer._equity_tier(0) == "small"
        assert AdaptivePositionSizer._equity_tier(499) == "small"
        assert AdaptivePositionSizer._equity_tier(500) == "medium"
        assert AdaptivePositionSizer._equity_tier(9999) == "medium"
        assert AdaptivePositionSizer._equity_tier(10000) == "large"
        assert AdaptivePositionSizer._equity_tier(1_000_000) == "large"

    def test_unknown_runner_key_uses_default_cap(self):
        sizer = AdaptivePositionSizer(
            runner_key="UNKNOWN_SYMBOL", min_size=0.001,
        )
        snap = _snap(equity=5000, price=60000.0)
        qty = sizer.target_qty(snap, "BTCUSDT")
        # Should use _DEFAULT_CAP=0.15 and not crash
        assert qty > Decimal("0")

    def test_regime_inactive_reduces_cap(self):
        sizer = AdaptivePositionSizer(runner_key="BTCUSDT", min_size=0.001)
        snap = _snap(equity=5000, price=60000.0)
        qty_active = sizer.target_qty(snap, "BTCUSDT", regime_active=True)
        qty_inactive = sizer.target_qty(snap, "BTCUSDT", regime_active=False)
        # Inactive caps at 60% of active
        assert qty_inactive < qty_active

    def test_ic_scale_zero_produces_min(self):
        sizer = AdaptivePositionSizer(runner_key="BTCUSDT", min_size=0.001)
        snap = _snap(equity=5000, price=60000.0)
        qty = sizer.target_qty(snap, "BTCUSDT", ic_scale=0.0)
        # notional * 0 = 0 → min_size
        assert qty == Decimal("0.001")

    def test_market_missing_returns_min(self):
        """Symbol not in snapshot.markets → price=0 → min_size."""
        sizer = AdaptivePositionSizer(runner_key="BTCUSDT", min_size=0.001)
        snap = _snap(equity=5000, price=60000.0, symbol="ETHUSDT")
        # BTCUSDT not in snap.markets (only ETHUSDT)
        qty = sizer.target_qty(snap, "BTCUSDT")
        # markets.get("BTCUSDT") returns MagicMock (not None) due to MagicMock auto-attr
        # So this tests the mock path. For real missing, we need explicit dict:
        snap.markets = {"ETHUSDT": MagicMock()}
        snap.markets["ETHUSDT"].close_f = 3000.0
        qty = sizer.target_qty(snap, "BTCUSDT")
        assert qty == Decimal("0.001")
