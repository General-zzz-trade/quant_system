"""Tests for AdaptivePositionSizer."""
from decimal import Decimal
from unittest.mock import MagicMock


from decision.sizing.adaptive import AdaptivePositionSizer


def _snap(equity: float, price: float, symbol: str = "BTCUSDT") -> MagicMock:
    """Build a minimal MagicMock StateSnapshot."""
    snap = MagicMock()
    snap.account.balance = equity
    market = MagicMock()
    market.close = price
    snap.markets = {symbol: market}
    return snap


class TestAdaptivePositionSizer:
    def test_basic_sizing_small_account(self):
        sizer = AdaptivePositionSizer(runner_key="BTCUSDT_4h")
        snap = _snap(equity=400, price=60000.0, symbol="BTCUSDT")
        qty = sizer.target_qty(snap, "BTCUSDT")
        # small tier, BTCUSDT_4h cap=0.35, lev=10 → notional=400*0.35*10=1400
        # size = 1400/60000 ≈ 0.023
        assert qty > Decimal("0.01")
        assert qty < Decimal("1.0")

    def test_basic_sizing_medium_account(self):
        sizer = AdaptivePositionSizer(runner_key="BTCUSDT_4h")
        snap_s = _snap(equity=400, price=60000.0, symbol="BTCUSDT")
        snap_m = _snap(equity=5000, price=60000.0, symbol="BTCUSDT")
        qty_s = sizer.target_qty(snap_s, "BTCUSDT")
        qty_m = sizer.target_qty(snap_m, "BTCUSDT")
        # medium cap (0.25) < small cap (0.35) per dollar, but higher equity
        # per-dollar allocation is smaller for medium
        cap_per_dollar_s = float(qty_s) * 60000.0 / 400.0
        cap_per_dollar_m = float(qty_m) * 60000.0 / 5000.0
        assert cap_per_dollar_m < cap_per_dollar_s

    def test_ic_health_scaling(self):
        sizer = AdaptivePositionSizer(runner_key="ETHUSDT_4h")
        snap = _snap(equity=1000, price=3000.0, symbol="ETHUSDT")
        qty_green = sizer.target_qty(snap, "ETHUSDT", ic_scale=1.2)
        qty_red = sizer.target_qty(snap, "ETHUSDT", ic_scale=0.4)
        assert qty_green > qty_red

    def test_regime_inactive_reduces(self):
        sizer = AdaptivePositionSizer(runner_key="BTCUSDT")
        snap = _snap(equity=2000, price=60000.0, symbol="BTCUSDT")
        qty_active = sizer.target_qty(snap, "BTCUSDT", regime_active=True)
        qty_inactive = sizer.target_qty(snap, "BTCUSDT", regime_active=False)
        assert qty_active > qty_inactive

    def test_zero_equity_returns_min(self):
        sizer = AdaptivePositionSizer(runner_key="BTCUSDT", min_size=0.001)
        snap = _snap(equity=0, price=60000.0, symbol="BTCUSDT")
        qty = sizer.target_qty(snap, "BTCUSDT")
        assert qty == Decimal("0.001")

    def test_zero_price_returns_min(self):
        sizer = AdaptivePositionSizer(runner_key="BTCUSDT", min_size=0.001)
        snap = _snap(equity=5000, price=0, symbol="BTCUSDT")
        qty = sizer.target_qty(snap, "BTCUSDT")
        assert qty == Decimal("0.001")

    def test_max_qty_clamp(self):
        sizer = AdaptivePositionSizer(runner_key="BTCUSDT_4h", max_qty=0.01)
        snap = _snap(equity=50000, price=60000.0, symbol="BTCUSDT")
        qty = sizer.target_qty(snap, "BTCUSDT")
        assert qty <= Decimal("0.01")

    def test_step_rounding(self):
        sizer = AdaptivePositionSizer(runner_key="BTCUSDT_4h", step_size=0.001)
        # Manually verify rounding: 0.0155 should floor to 0.015
        rounded = sizer._round_to_step(0.0155)
        assert rounded == Decimal("0.015")
        # 0.0199 → 0.019
        assert sizer._round_to_step(0.0199) == Decimal("0.019")
