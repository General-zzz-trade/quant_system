"""Tests for polymarket.features — probability market feature computation."""
from __future__ import annotations
import math
import numpy as np
import pytest
from polymarket.features import compute_features


EXPECTED_KEYS = {
    "prob_ret_1h", "prob_ret_4h", "prob_ret_12h", "prob_ret_24h",
    "prob_zscore_24h", "prob_zscore_72h",
    "prob_vol_12h", "prob_vol_24h",
    "bid_ask_spread", "mid_price", "depth_imbalance", "total_depth",
    "trade_intensity", "buy_sell_ratio", "large_trade_flag", "net_flow",
    "hours_to_expiry", "time_decay_rate", "log_hours_to_expiry",
    "btc_price_vs_strike", "btc_momentum_12h",
    "prob_level", "prob_from_center", "prob_extreme",
    "prob_vs_ma24", "prob_acceleration",
}


def _make_history(n: int = 72, base: float = 0.5, noise: float = 0.02) -> np.ndarray:
    rng = np.random.RandomState(42)
    return base + rng.randn(n) * noise


def test_compute_features_returns_expected_keys():
    hist = _make_history(72)
    ob = {"best_bid": 0.48, "best_ask": 0.52, "bid_depth": 100, "ask_depth": 80}
    trades = {"count_1h": 5, "buy_volume": 200, "sell_volume": 150, "max_trade_size": 100}
    feats = compute_features(hist, ob, trades, expiry_hours=72.0, btc_price=60000, btc_strike=55000)
    assert set(feats.keys()) == EXPECTED_KEYS
    # All values should be finite floats (no NaN with 48-point history)
    for k, v in feats.items():
        assert isinstance(v, float), f"{k} is not float: {type(v)}"
        assert not math.isnan(v), f"{k} is NaN with 48-point history"


def test_features_handles_short_history():
    """Short history should not crash; missing features become NaN."""
    hist = np.array([0.5])
    ob = {"best_bid": 0, "best_ask": 0}
    trades = {}
    feats = compute_features(hist, ob, trades, expiry_hours=100.0)
    assert set(feats.keys()) == EXPECTED_KEYS
    # Momentum features should be NaN with single point
    assert math.isnan(feats["prob_ret_1h"])
    assert math.isnan(feats["prob_zscore_24h"])
    assert math.isnan(feats["prob_vol_12h"])
    # Expiry features should still be valid
    assert feats["hours_to_expiry"] == 100.0
    assert not math.isnan(feats["prob_level"])


def test_features_orderbook_imbalance():
    hist = _make_history(2)
    ob = {"best_bid": 0.45, "best_ask": 0.55, "bid_depth": 300, "ask_depth": 100}
    feats = compute_features(hist, ob, {}, expiry_hours=48.0)
    assert feats["depth_imbalance"] == pytest.approx(0.5)  # (300-100)/400
    assert feats["bid_ask_spread"] == pytest.approx(0.10)


def test_features_btc_cross_market():
    hist = _make_history(2)
    feats = compute_features(hist, {}, {}, expiry_hours=48.0, btc_price=60000, btc_strike=50000)
    assert feats["btc_price_vs_strike"] == pytest.approx(0.2)  # (60k-50k)/50k
