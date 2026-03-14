"""Tests for polymarket.signals — rule-based signal generation."""
from __future__ import annotations
import pytest
from polymarket.signals import generate_signal


def _base_features(**overrides) -> dict:
    feats = {
        "hours_to_expiry": 48.0,
        "btc_price_vs_strike": 0.0,
        "prob_zscore_24h": 0.0,
        "depth_imbalance": 0.0,
        "prob_level": 0.5,
    }
    feats.update(overrides)
    return feats


def test_bullish_signal_when_btc_above_and_prob_underpriced():
    """BTC well above strike + prob below average -> positive signal."""
    feats = _base_features(btc_price_vs_strike=0.10, prob_zscore_24h=-1.0)
    signal = generate_signal(feats)
    assert signal > 0, f"Expected positive signal, got {signal}"


def test_no_signal_when_conflicting():
    """BTC above strike but prob also above average -> no signal."""
    feats = _base_features(btc_price_vs_strike=0.10, prob_zscore_24h=1.0)
    signal = generate_signal(feats)
    assert signal == 0.0


def test_no_signal_near_expiry():
    """Should return 0 when too close to expiry."""
    feats = _base_features(
        btc_price_vs_strike=0.10,
        prob_zscore_24h=-1.0,
        hours_to_expiry=2.0,
    )
    signal = generate_signal(feats)
    assert signal == 0.0


def test_signal_dampened_at_extreme_probability():
    """Signal should be dampened when probability is near 0 or 1."""
    feats_normal = _base_features(btc_price_vs_strike=0.10, prob_zscore_24h=-1.0, prob_level=0.5)
    feats_extreme = _base_features(btc_price_vs_strike=0.10, prob_zscore_24h=-1.0, prob_level=0.10)
    sig_normal = generate_signal(feats_normal)
    sig_extreme = generate_signal(feats_extreme)
    # Extreme prob signal should be weaker (or zero)
    assert sig_extreme < sig_normal or sig_extreme == 0.0


def test_bearish_signal():
    """BTC below strike + prob above average -> negative signal."""
    feats = _base_features(btc_price_vs_strike=-0.10, prob_zscore_24h=1.0)
    signal = generate_signal(feats)
    assert signal < 0, f"Expected negative signal, got {signal}"
