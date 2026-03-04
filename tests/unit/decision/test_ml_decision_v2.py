"""Tests for MLDecisionModule v2 features: stops, asymmetric thresholds, vol sizing."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from decision.ml_decision import MLDecisionModule


def _snap(
    close: float,
    ml_score: float,
    qty: float = 0,
    balance: float = 10000,
    atr_norm: float | None = 0.02,
    symbol: str = "BTCUSDT",
) -> dict:
    """Build a minimal snapshot dict for testing."""
    market = SimpleNamespace(close=Decimal(str(close)))
    positions = {}
    if qty != 0:
        positions[symbol] = SimpleNamespace(qty=Decimal(str(qty)))
    account = SimpleNamespace(balance=Decimal(str(balance)))
    features: dict = {"ml_score": ml_score}
    if atr_norm is not None:
        features["atr_norm_14"] = atr_norm
    return {
        "market": market,
        "positions": positions,
        "features": features,
        "account": account,
    }


class TestBackwardCompat:
    """Default params = current behavior (all new features disabled)."""

    def test_defaults_produce_same_behavior(self):
        mod = MLDecisionModule(symbol="BTCUSDT", risk_pct=0.30, threshold=0.005)
        assert mod.atr_stop == 0.0
        assert mod.trailing_atr == 0.0
        assert mod.min_hold_bars == 0
        assert mod.vol_target == 0.0
        assert mod.threshold_short == 0.005  # mirrors threshold

        # Long signal
        orders = list(mod.decide(_snap(50000, 0.8)))
        assert len(orders) == 1
        assert orders[0].side == "BUY"
        assert orders[0].reason == "open_long"

    def test_flat_signal_no_orders(self):
        mod = MLDecisionModule(symbol="BTCUSDT", threshold=0.005)
        orders = list(mod.decide(_snap(50000, 0.003)))
        assert len(orders) == 0


class TestAsymmetricThreshold:
    """threshold_short separate from threshold."""

    def test_short_suppressed_by_higher_threshold(self):
        mod = MLDecisionModule(
            symbol="BTCUSDT", threshold=0.005, threshold_short=0.01,
        )
        # ml_score=-0.007: passes threshold (0.005) but not threshold_short (0.01)
        orders = list(mod.decide(_snap(50000, -0.007)))
        assert len(orders) == 0

    def test_short_triggers_at_threshold_short(self):
        mod = MLDecisionModule(
            symbol="BTCUSDT", threshold=0.005, threshold_short=0.01,
        )
        orders = list(mod.decide(_snap(50000, -0.8)))
        assert len(orders) == 1
        assert orders[0].side == "SELL"
        assert orders[0].reason == "open_short"


class TestHardStopLoss:
    """atr_stop: hard stop-loss in ATR multiples."""

    def test_stop_loss_long(self):
        mod = MLDecisionModule(
            symbol="BTCUSDT", threshold=0.005, atr_stop=2.0,
        )
        # Open long
        orders = list(mod.decide(_snap(50000, 0.8, atr_norm=0.02)))
        assert len(orders) == 1
        assert orders[0].reason == "open_long"

        # Price drops beyond 2 * 0.02 * 50000 = 2000 from entry
        # entry=50000, stop at 48000, close=47900 triggers
        orders = list(mod.decide(_snap(47900, 0.8, qty=0.06, atr_norm=0.02)))
        assert len(orders) == 1
        assert orders[0].reason == "stop_loss"
        assert orders[0].side == "SELL"

    def test_stop_loss_short(self):
        mod = MLDecisionModule(
            symbol="BTCUSDT", threshold=0.005, atr_stop=2.0,
        )
        # Open short
        orders = list(mod.decide(_snap(50000, -0.8, atr_norm=0.02)))
        assert len(orders) == 1
        assert orders[0].reason == "open_short"

        # Price rises beyond 2 * 0.02 * 50000 = 2000 from entry
        orders = list(mod.decide(_snap(52100, -0.8, qty=-0.06, atr_norm=0.02)))
        assert len(orders) == 1
        assert orders[0].reason == "stop_loss"
        assert orders[0].side == "BUY"


class TestTrailingStop:
    """trailing_atr: trailing stop from peak."""

    def test_trailing_stop_long(self):
        mod = MLDecisionModule(
            symbol="BTCUSDT", threshold=0.005, risk_pct=0.30, trailing_atr=3.0,
        )
        # Open long at 50000 with ml_score=1.0 → qty=0.060
        orders = list(mod.decide(_snap(50000, 1.0, atr_norm=0.02)))
        assert len(orders) == 1
        assert orders[0].reason == "open_long"
        open_qty = float(orders[0].qty)

        # Price rises to 55000 — use matching target qty to avoid rebalance
        # target = 10000*0.30*1.0/55000 = 0.054
        orders = list(mod.decide(_snap(55000, 1.0, qty=0.054, atr_norm=0.02)))
        assert mod._peak_price == 55000

        # Price drops from 55000: trail_dist = 0.02 * 51700 * 3.0 = 3102
        # At 51700 the trail from peak 55000 is 3300 > 3102 → triggers
        orders = list(mod.decide(_snap(51700, 1.0, qty=0.054, atr_norm=0.02)))
        assert len(orders) == 1
        assert orders[0].reason == "trailing_stop"


class TestMinHoldBars:
    """min_hold_bars: suppress signal flip during hold period."""

    def test_min_hold_suppresses_flip(self):
        mod = MLDecisionModule(
            symbol="BTCUSDT", threshold=0.005, min_hold_bars=3,
        )
        # Open long
        orders = list(mod.decide(_snap(50000, 0.8)))
        assert len(orders) == 1

        # Bar 1: short signal but held only 1 bar — suppressed
        orders = list(mod.decide(_snap(49000, -0.8, qty=0.06)))
        assert len(orders) == 0

        # Bar 2: still suppressed
        orders = list(mod.decide(_snap(48000, -0.8, qty=0.06)))
        assert len(orders) == 0

        # Bar 3: now allowed (bars_held=3 >= min_hold_bars=3)
        orders = list(mod.decide(_snap(47000, -0.8, qty=0.06)))
        assert len(orders) > 0


class TestVolTargetSizing:
    """vol_target: qty inversely proportional to ATR."""

    def test_vol_target_qty_inverse_atr(self):
        # With vol_target enabled + atr_stop, qty = equity * risk_pct / (atr_norm * atr_stop) / price
        mod_low_vol = MLDecisionModule(
            symbol="BTCUSDT", threshold=0.005, risk_pct=0.30,
            atr_stop=2.0, vol_target=0.15,
        )
        mod_high_vol = MLDecisionModule(
            symbol="BTCUSDT", threshold=0.005, risk_pct=0.30,
            atr_stop=2.0, vol_target=0.15,
        )

        # Low vol: atr_norm=0.01 → base = 10000*0.30 / (0.01*2.0) / 50000 = 3.0, * weight=1.0 = 3.0
        orders_low = list(mod_low_vol.decide(_snap(50000, 1.0, atr_norm=0.01)))
        # High vol: atr_norm=0.04 → base = 10000*0.30 / (0.04*2.0) / 50000 = 0.75, * weight=1.0 = 0.75
        orders_high = list(mod_high_vol.decide(_snap(50000, 1.0, atr_norm=0.04)))

        assert len(orders_low) == 1
        assert len(orders_high) == 1
        qty_low = float(orders_low[0].qty)
        qty_high = float(orders_high[0].qty)
        # Low vol → bigger position
        assert qty_low > qty_high
        # Approximate check
        assert abs(qty_low - 3.0) < 0.01
        assert abs(qty_high - 0.75) < 0.01


class TestStopPriority:
    """Stop-loss fires before signal evaluation."""

    def test_stop_fires_even_with_same_direction_signal(self):
        mod = MLDecisionModule(
            symbol="BTCUSDT", threshold=0.005, atr_stop=2.0,
        )
        # Open long
        list(mod.decide(_snap(50000, 0.8, atr_norm=0.02)))

        # Price crashes but ml_score still says long
        # stop_dist = 0.02 * 50000 * 2.0 = 2000, stop at 48000
        orders = list(mod.decide(_snap(47500, 0.8, qty=0.06, atr_norm=0.02)))
        assert len(orders) == 1
        assert orders[0].reason == "stop_loss"


class TestATRMissingSafety:
    """When atr_norm is missing, stops don't fire (warmup safety)."""

    def test_no_stop_without_atr(self):
        mod = MLDecisionModule(
            symbol="BTCUSDT", threshold=0.005, atr_stop=2.0,
        )
        # Open long (without ATR in features)
        orders = list(mod.decide(_snap(50000, 0.8, atr_norm=None)))
        assert len(orders) == 1

        # Price crashes but no ATR → stop should NOT fire
        # May generate rebalance due to price change, but must NOT be stop_loss
        orders = list(mod.decide(_snap(40000, 0.8, qty=0.06, atr_norm=None)))
        for o in orders:
            assert o.reason != "stop_loss"
