"""Tests for MultiTFConfluenceGate."""

from unittest.mock import MagicMock
from strategy.gates.multi_tf_confluence_gate import (
    MultiTFConfluenceGate,
    MultiTFConfluenceConfig,
)


def _make_ev(signal=0):
    ev = MagicMock()
    ev.metadata = {"signal": signal}
    return ev


class TestMultiTFConfluenceGate:
    def test_no_signal_passthrough(self):
        gate = MultiTFConfluenceGate()
        r = gate.check(_make_ev(0), {"tf4h_close_vs_ma20": 0.01})
        assert r.allowed
        assert r.scale == 1.0

    def test_no_4h_data_passthrough(self):
        gate = MultiTFConfluenceGate()
        r = gate.check(_make_ev(1), {})
        assert r.allowed
        assert r.scale == 1.0

    def test_aligned_bullish_boost(self):
        """Long signal + 4h bullish → boost."""
        gate = MultiTFConfluenceGate()
        r = gate.check(_make_ev(1), {
            "tf4h_close_vs_ma20": 0.02,   # above MA
            "tf4h_rsi_14": 70.0,            # overbought
            "tf4h_macd_hist": 0.5,          # positive
        })
        assert r.allowed
        assert r.scale == 1.2

    def test_opposed_bearish_reduce(self):
        """Long signal + 4h bearish → reduce."""
        gate = MultiTFConfluenceGate()
        r = gate.check(_make_ev(1), {
            "tf4h_close_vs_ma20": -0.02,
            "tf4h_rsi_14": 30.0,
            "tf4h_macd_hist": -0.5,
        })
        assert r.allowed
        assert r.scale == 0.5

    def test_short_aligned_with_bearish(self):
        """Short signal + 4h bearish → boost."""
        gate = MultiTFConfluenceGate()
        r = gate.check(_make_ev(-1), {
            "tf4h_close_vs_ma20": -0.02,
            "tf4h_rsi_14": 30.0,
            "tf4h_macd_hist": -0.5,
        })
        assert r.allowed
        assert r.scale == 1.2

    def test_short_opposed_bullish(self):
        """Short signal + 4h bullish → reduce."""
        gate = MultiTFConfluenceGate()
        r = gate.check(_make_ev(-1), {
            "tf4h_close_vs_ma20": 0.02,
            "tf4h_rsi_14": 70.0,
            "tf4h_macd_hist": 0.5,
        })
        assert r.allowed
        assert r.scale == 0.5

    def test_neutral_4h_no_change(self):
        """4h indicators mixed → neutral scale."""
        gate = MultiTFConfluenceGate()
        r = gate.check(_make_ev(1), {
            "tf4h_close_vs_ma20": 0.001,   # neutral
            "tf4h_rsi_14": 50.0,            # neutral
            "tf4h_macd_hist": 0.0,          # neutral (but > threshold)
        })
        assert r.allowed
        assert r.scale == 1.0

    def test_min_confirming_2(self):
        """Need 2 of 3 indicators for trend classification."""
        gate = MultiTFConfluenceGate()
        # Only RSI is bullish, MA and MACD neutral/bearish
        r = gate.check(_make_ev(1), {
            "tf4h_close_vs_ma20": 0.001,   # neutral
            "tf4h_rsi_14": 70.0,            # bullish
            "tf4h_macd_hist": -0.1,         # bearish
        })
        assert r.allowed
        assert r.scale == 1.0  # no consensus → neutral

    def test_partial_data(self):
        """Only some 4h indicators available."""
        gate = MultiTFConfluenceGate()
        r = gate.check(_make_ev(1), {
            "tf4h_close_vs_ma20": 0.02,
            "tf4h_rsi_14": 70.0,
            # no macd
        })
        assert r.allowed
        assert r.scale == 1.2  # 2 of 2 available → bullish

    def test_disabled(self):
        gate = MultiTFConfluenceGate(MultiTFConfluenceConfig(enabled=False))
        r = gate.check(_make_ev(1), {
            "tf4h_close_vs_ma20": -0.05,
            "tf4h_rsi_14": 20.0,
            "tf4h_macd_hist": -1.0,
        })
        assert r.allowed
        assert r.scale == 1.0

    def test_nan_values_treated_as_missing(self):
        gate = MultiTFConfluenceGate()
        r = gate.check(_make_ev(1), {
            "tf4h_close_vs_ma20": float("nan"),
            "tf4h_rsi_14": float("nan"),
        })
        assert r.allowed
        assert r.scale == 1.0

    def test_stats(self):
        gate = MultiTFConfluenceGate()
        gate.check(_make_ev(1), {
            "tf4h_close_vs_ma20": 0.02,
            "tf4h_rsi_14": 70.0,
            "tf4h_macd_hist": 0.5,
        })
        gate.check(_make_ev(1), {
            "tf4h_close_vs_ma20": -0.02,
            "tf4h_rsi_14": 30.0,
            "tf4h_macd_hist": -0.5,
        })
        stats = gate.stats
        assert stats["total_checks"] == 2
        assert stats["aligned"] == 1
        assert stats["opposed"] == 1

    def test_custom_config(self):
        cfg = MultiTFConfluenceConfig(
            aligned_scale=1.5,
            opposed_scale=0.3,
            min_confirming=1,
        )
        gate = MultiTFConfluenceGate(cfg)
        r = gate.check(_make_ev(1), {
            "tf4h_close_vs_ma20": 0.02,
        })
        assert r.scale == 1.5

    def test_signal_from_context(self):
        """Signal from context when not in metadata."""
        gate = MultiTFConfluenceGate()
        ev = MagicMock()
        ev.metadata = {}
        r = gate.check(ev, {
            "signal": 1,
            "tf4h_close_vs_ma20": 0.02,
            "tf4h_rsi_14": 70.0,
            "tf4h_macd_hist": 0.5,
        })
        assert r.scale == 1.2
