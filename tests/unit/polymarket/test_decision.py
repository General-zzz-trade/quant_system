"""Tests for polymarket.decision — PolymarketDecisionModule."""
from __future__ import annotations
import pytest
from polymarket.config import PolymarketConfig
from polymarket.decision import PolymarketDecisionModule


def _make_module(
    signal_threshold: float = 0.15,
    min_probability: float = 0.15,
    max_probability: float = 0.85,
) -> PolymarketDecisionModule:
    config = PolymarketConfig(
        signal_threshold=signal_threshold,
        min_probability=min_probability,
        max_probability=max_probability,
    )
    return PolymarketDecisionModule(config)


def _market_data(
    btc_vs_strike: float = 0.0,
    prob_zscore: float = 0.0,
    market_price: float = 0.5,
    hours: float = 48.0,
    bankroll: float = 10000.0,
) -> dict:
    return {
        "symbol": "POLY:btc-above-100k:YES",
        "features": {
            "hours_to_expiry": hours,
            "btc_price_vs_strike": btc_vs_strike,
            "prob_zscore_24h": prob_zscore,
            "depth_imbalance": 0.0,
            "prob_level": market_price,
        },
        "market_price": market_price,
        "bankroll": bankroll,
    }


def test_generates_intent_for_strong_signal():
    mod = _make_module()
    data = _market_data(btc_vs_strike=0.10, prob_zscore=-1.0, bankroll=10000)
    intents = mod.evaluate(data)
    assert len(intents) == 1
    intent = intents[0]
    assert intent["side"] == "buy"
    assert intent["outcome"] == "YES"
    assert intent["size"] > 0
    assert intent["signal_strength"] > 0


def test_skips_extreme_probability():
    mod = _make_module()
    data = _market_data(btc_vs_strike=0.10, prob_zscore=-1.0, market_price=0.05)
    intents = mod.evaluate(data)
    assert intents == []


def test_generates_exit_on_stop_loss():
    mod = _make_module()
    positions = [
        {"symbol": "POLY:test:YES", "entry_price": 0.50, "current_price": 0.40, "qty": 100},
    ]
    exits = mod.check_exits(positions)
    assert len(exits) == 1
    assert exits[0]["action"] == "close"
    assert "stop_loss" in exits[0]["reason"]


def test_no_signal_returns_empty():
    mod = _make_module()
    # Conflicting signals: BTC above strike but prob also above average
    data = _market_data(btc_vs_strike=0.10, prob_zscore=1.0)
    intents = mod.evaluate(data)
    assert intents == []


def test_no_exit_when_position_healthy():
    mod = _make_module()
    positions = [
        {"symbol": "POLY:test:YES", "entry_price": 0.50, "current_price": 0.55, "qty": 100},
    ]
    exits = mod.check_exits(positions)
    assert exits == []
