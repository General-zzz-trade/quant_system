"""Tests for MLDecisionModule: fractional sizing, DD breaker, rebalance."""
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


class TestFractionalSizing:
    """ml_score magnitude scales position size."""

    def test_half_score_half_qty(self):
        mod_full = MLDecisionModule(symbol="BTCUSDT", risk_pct=0.30, threshold=0.005)
        mod_half = MLDecisionModule(symbol="BTCUSDT", risk_pct=0.30, threshold=0.005)

        orders_full = list(mod_full.decide(_snap(50000, 1.0)))
        orders_half = list(mod_half.decide(_snap(50000, 0.5)))

        assert len(orders_full) == 1
        assert len(orders_half) == 1

        qty_full = float(orders_full[0].qty)
        qty_half = float(orders_half[0].qty)
        # ml_score=0.5 → qty ~50% of ml_score=1.0
        assert qty_full > 0
        assert abs(qty_half / qty_full - 0.5) < 0.02

    def test_score_06_gives_60pct(self):
        mod = MLDecisionModule(symbol="BTCUSDT", risk_pct=0.30, threshold=0.005)
        # ml_score=1.0: qty = 10000 * 0.30 * 1.0 / 50000 = 0.060
        orders_full = list(mod.decide(_snap(50000, 1.0)))
        qty_full = float(orders_full[0].qty)

        mod2 = MLDecisionModule(symbol="BTCUSDT", risk_pct=0.30, threshold=0.005)
        orders_06 = list(mod2.decide(_snap(50000, 0.6)))
        qty_06 = float(orders_06[0].qty)
        # 0.6 / 1.0 ≈ 0.6
        assert abs(qty_06 / qty_full - 0.6) < 0.02

    def test_score_capped_at_1(self):
        """ml_score > 1.0 should be capped at weight=1.0."""
        mod1 = MLDecisionModule(symbol="BTCUSDT", risk_pct=0.30, threshold=0.005)
        mod2 = MLDecisionModule(symbol="BTCUSDT", risk_pct=0.30, threshold=0.005)

        orders_1 = list(mod1.decide(_snap(50000, 1.0)))
        orders_2 = list(mod2.decide(_snap(50000, 2.0)))

        assert float(orders_1[0].qty) == float(orders_2[0].qty)

    def test_vol_target_also_scaled(self):
        """Vol-target branch should also apply ml_score weight."""
        mod_full = MLDecisionModule(
            symbol="BTCUSDT", risk_pct=0.30, threshold=0.005,
            atr_stop=2.0, vol_target=0.15,
        )
        mod_half = MLDecisionModule(
            symbol="BTCUSDT", risk_pct=0.30, threshold=0.005,
            atr_stop=2.0, vol_target=0.15,
        )

        orders_full = list(mod_full.decide(_snap(50000, 1.0, atr_norm=0.02)))
        orders_half = list(mod_half.decide(_snap(50000, 0.5, atr_norm=0.02)))

        qty_full = float(orders_full[0].qty)
        qty_half = float(orders_half[0].qty)
        assert qty_full > 0
        assert abs(qty_half / qty_full - 0.5) < 0.02


class TestDDBreaker:
    """Drawdown circuit breaker: flatten + cooldown."""

    def test_dd_triggers_flatten(self):
        mod = MLDecisionModule(
            symbol="BTCUSDT", threshold=0.005, risk_pct=0.30,
            dd_limit=0.10, dd_cooldown=3,
        )
        # Open long at balance=10000 → sets hwm=10000
        orders = list(mod.decide(_snap(50000, 0.8, balance=10000)))
        assert len(orders) == 1
        assert orders[0].reason == "open_long"
        qty = orders[0].qty

        # Balance drops to 8900 (11% DD > 10% limit), signal wants flat
        orders = list(mod.decide(_snap(50000, -0.001, qty=float(qty), balance=8900)))
        assert len(orders) == 1
        assert orders[0].reason == "dd_breaker"

    def test_dd_cooldown_blocks_new_entry(self):
        mod = MLDecisionModule(
            symbol="BTCUSDT", threshold=0.005, risk_pct=0.30,
            dd_limit=0.10, dd_cooldown=2,
        )
        # Build HWM
        list(mod.decide(_snap(50000, 0.003, balance=10000)))  # flat, just updates hwm

        # Trigger DD breaker while flat (balance dropped, no position)
        # dd=11% >= 10% → cooldown=2, decrement→1, blocked
        orders = list(mod.decide(_snap(50000, 0.8, balance=8900)))
        assert len(orders) == 0

        # Balance recovers (dd=5% < 10%), but cooldown=1 still active
        # no reset, decrement→0, blocked
        orders = list(mod.decide(_snap(50000, 0.8, balance=9500)))
        assert len(orders) == 0

        # Cooldown fully expired (remaining=0), can trade
        orders = list(mod.decide(_snap(50000, 0.8, balance=9500)))
        assert len(orders) == 1
        assert orders[0].reason == "open_long"

    def test_dd_disabled_by_default(self):
        mod = MLDecisionModule(symbol="BTCUSDT", threshold=0.005, risk_pct=0.30)
        # Even with massive DD, no breaker since dd_limit=0
        orders = list(mod.decide(_snap(50000, 0.8, balance=5000)))
        assert len(orders) == 1
        assert orders[0].reason == "open_long"


class TestRebalance:
    """Same direction, qty changes → rebalance orders."""

    def test_rebalance_up(self):
        mod = MLDecisionModule(symbol="BTCUSDT", risk_pct=0.30, threshold=0.005)
        # Open long with ml_score=0.5 → qty = 10000*0.30*0.5/50000 = 0.030
        orders = list(mod.decide(_snap(50000, 0.5)))
        assert len(orders) == 1
        initial_qty = float(orders[0].qty)

        # Now ml_score rises to 1.0 → target = 0.060, delta = +0.030 (>1%)
        orders = list(mod.decide(_snap(50000, 1.0, qty=initial_qty)))
        assert len(orders) == 1
        assert orders[0].reason == "rebalance_up"
        assert orders[0].side == "BUY"

    def test_rebalance_down(self):
        mod = MLDecisionModule(symbol="BTCUSDT", risk_pct=0.30, threshold=0.005)
        # Open long with ml_score=1.0 → qty = 0.060
        orders = list(mod.decide(_snap(50000, 1.0)))
        assert len(orders) == 1
        initial_qty = float(orders[0].qty)

        # ml_score drops to 0.3 → target = 0.018, delta = -0.042 (>1%)
        orders = list(mod.decide(_snap(50000, 0.3, qty=initial_qty)))
        assert len(orders) == 1
        assert orders[0].reason == "rebalance_down"
        assert orders[0].side == "SELL"

    def test_no_rebalance_when_same_score(self):
        mod = MLDecisionModule(symbol="BTCUSDT", risk_pct=0.30, threshold=0.005)
        # Open long with ml_score=0.8 → qty = 0.048
        orders = list(mod.decide(_snap(50000, 0.8)))
        initial_qty = float(orders[0].qty)

        # Same score → same target qty → no rebalance
        orders = list(mod.decide(_snap(50000, 0.8, qty=initial_qty)))
        assert len(orders) == 0

    def test_rebalance_short_direction(self):
        mod = MLDecisionModule(symbol="BTCUSDT", risk_pct=0.30, threshold=0.005)
        # Open short with ml_score=-1.0
        orders = list(mod.decide(_snap(50000, -1.0)))
        assert orders[0].reason == "open_short"
        initial_qty = float(orders[0].qty)

        # ml_score weakens to -0.5 → should rebalance down (reduce short)
        orders = list(mod.decide(_snap(50000, -0.5, qty=-initial_qty)))
        assert len(orders) == 1
        assert orders[0].reason == "rebalance_down"
        assert orders[0].side == "BUY"  # buying back to reduce short


class TestBackwardCompat:
    """Without new params, behavior matches old code."""

    def test_defaults_unchanged(self):
        mod = MLDecisionModule(symbol="BTCUSDT", risk_pct=0.30, threshold=0.005)
        assert mod.dd_limit == 0.0
        assert mod.dd_cooldown == 48
        assert mod._hwm == 0.0
        assert mod._dd_cooldown_remaining == 0

    def test_old_behavior_preserved(self):
        """Without position, ml_score>threshold → open_long with same qty regardless of score magnitude.
        Actually, now ml_score DOES affect qty, but ml_score=1.0 gives same result as before."""
        mod = MLDecisionModule(symbol="BTCUSDT", risk_pct=0.30, threshold=0.005)
        orders = list(mod.decide(_snap(50000, 1.0)))
        assert len(orders) == 1
        assert orders[0].side == "BUY"
        assert orders[0].reason == "open_long"
        # qty = 10000 * 0.30 * 1.0 / 50000 = 0.060
        assert float(orders[0].qty) == 0.06

    def test_flat_signal_still_no_orders(self):
        mod = MLDecisionModule(symbol="BTCUSDT", threshold=0.005)
        orders = list(mod.decide(_snap(50000, 0.003)))
        assert len(orders) == 0
