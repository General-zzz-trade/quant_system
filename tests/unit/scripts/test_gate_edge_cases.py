"""Edge case tests for alpha expansion gates — boundary values, NaN, overflow."""

from unittest.mock import MagicMock

from runner.gates.multi_tf_confluence_gate import MultiTFConfluenceGate
from runner.gates.liquidation_cascade_gate import LiquidationCascadeGate
from runner.gates.carry_cost_gate import CarryCostGate


def _ev(signal=1):
    ev = MagicMock()
    ev.metadata = {"signal": signal}
    return ev


class TestMultiTFEdgeCases:
    def test_all_nan_features(self):
        gate = MultiTFConfluenceGate()
        r = gate.check(_ev(1), {
            "tf4h_close_vs_ma20": float("nan"),
            "tf4h_rsi_14": float("nan"),
            "tf4h_macd_hist": float("nan"),
        })
        assert r.allowed
        assert r.scale == 1.0  # no data → passthrough

    def test_inf_values(self):
        gate = MultiTFConfluenceGate()
        r = gate.check(_ev(1), {
            "tf4h_close_vs_ma20": float("inf"),
            "tf4h_rsi_14": float("-inf"),
        })
        assert r.allowed  # should not crash

    def test_extreme_rsi(self):
        gate = MultiTFConfluenceGate()
        r = gate.check(_ev(1), {
            "tf4h_rsi_14": 99.99,
            "tf4h_close_vs_ma20": 0.5,
        })
        assert r.allowed
        assert r.scale == 1.2  # bullish

    def test_zero_signal_always_passthrough(self):
        gate = MultiTFConfluenceGate()
        r = gate.check(_ev(0), {
            "tf4h_close_vs_ma20": -0.1,
            "tf4h_rsi_14": 10.0,
            "tf4h_macd_hist": -5.0,
        })
        assert r.scale == 1.0

    def test_string_values_ignored(self):
        gate = MultiTFConfluenceGate()
        r = gate.check(_ev(1), {
            "tf4h_close_vs_ma20": "bad_data",
            "tf4h_rsi_14": None,
        })
        assert r.allowed
        assert r.scale == 1.0


class TestLiquidationCascadeEdgeCases:
    def test_negative_zscore(self):
        """Negative zscore (low liquidation) should not trigger."""
        gate = LiquidationCascadeGate()
        r = gate.check(_ev(1), {
            "liquidation_volume_zscore_24": -2.0,
            "oi_acceleration": 0.5,
        })
        assert r.allowed
        assert r.scale == 1.0

    def test_exactly_at_threshold(self):
        """Boundary: zscore exactly at 3.0 → should block."""
        gate = LiquidationCascadeGate()
        r = gate.check(_ev(1), {
            "liquidation_volume_zscore_24": 3.0,
        })
        assert not r.allowed

    def test_huge_zscore(self):
        gate = LiquidationCascadeGate()
        r = gate.check(_ev(1), {
            "liquidation_volume_zscore_24": 100.0,
        })
        assert not r.allowed

    def test_nan_defaults_safe(self):
        gate = LiquidationCascadeGate()
        r = gate.check(_ev(1), {
            "liquidation_volume_zscore_24": float("nan"),
            "oi_acceleration": float("nan"),
        })
        assert r.allowed
        assert r.scale == 1.0

    def test_no_context_keys(self):
        gate = LiquidationCascadeGate()
        r = gate.check(_ev(1), {})
        assert r.allowed
        assert r.scale == 1.0


class TestCarryCostEdgeCases:
    def test_nan_funding(self):
        gate = CarryCostGate()
        r = gate.check(_ev(1), {
            "funding_rate": float("nan"),
            "basis": 0.0,
        })
        assert r.allowed
        assert r.scale == 1.0  # NaN → default 0.0 → no carry

    def test_zero_everything(self):
        gate = CarryCostGate()
        r = gate.check(_ev(1), {
            "funding_rate": 0.0,
            "basis": 0.0,
        })
        assert r.allowed
        assert r.scale == 1.0

    def test_extreme_funding(self):
        """Extremely high funding should reduce heavily."""
        gate = CarryCostGate()
        r = gate.check(_ev(1), {
            "funding_rate": 0.01,  # 1% per 8h = 1095% annualized
            "basis": 0.0,
        })
        assert r.allowed
        assert r.scale <= 0.5

    def test_negative_signal_reverses_carry(self):
        """Short position receives positive funding."""
        gate = CarryCostGate()
        # Same high funding, but short → receives carry
        r = gate.check(_ev(-1), {
            "funding_rate": 0.001,
            "basis": 0.0,
        })
        assert r.scale >= 1.0  # favorable

    def test_inf_funding(self):
        gate = CarryCostGate()
        r = gate.check(_ev(1), {
            "funding_rate": float("inf"),
            "basis": 0.0,
        })
        # Should not crash
        assert r.allowed


class TestNonNumericSignal:
    """Gates should handle non-numeric signal values gracefully."""

    def test_string_signal_in_metadata(self):
        for GateClass in [MultiTFConfluenceGate, LiquidationCascadeGate, CarryCostGate]:
            gate = GateClass()
            ev = MagicMock()
            ev.metadata = {"signal": "buy"}  # non-numeric
            r = gate.check(ev, {"tf4h_rsi_14": 70.0})
            assert r.allowed
            assert r.scale == 1.0  # treated as signal=0

    def test_none_signal_in_context(self):
        for GateClass in [MultiTFConfluenceGate, LiquidationCascadeGate, CarryCostGate]:
            gate = GateClass()
            ev = MagicMock()
            ev.metadata = {}
            r = gate.check(ev, {"signal": None})
            assert r.allowed


class TestGateChainCumulative:
    """Test gates applied in sequence don't produce invalid scales."""

    def test_all_gates_combined_no_crash(self):
        """Run all 3 gates with various context values."""
        liq = LiquidationCascadeGate()
        mtf = MultiTFConfluenceGate()
        carry = CarryCostGate()

        contexts = [
            {},
            {"liquidation_volume_zscore_24": 1.5, "tf4h_rsi_14": 70, "funding_rate": 0.0001},
            {"liquidation_volume_zscore_24": 0, "tf4h_close_vs_ma20": float("nan")},
            {"funding_rate": float("inf"), "basis": float("-inf")},
        ]

        for ctx in contexts:
            ctx["signal"] = 1
            ev = _ev(1)
            scale = 1.0
            r = liq.check(ev, ctx)
            if r.allowed:
                scale *= r.scale
            r = mtf.check(ev, ctx)
            scale *= r.scale
            r = carry.check(ev, ctx)
            scale *= r.scale
            # Scale should always be finite
            assert scale == scale  # not NaN
            assert abs(scale) < 100  # reasonable range

    def test_scale_never_negative(self):
        """Gate scales should never be negative."""
        for GateClass in [LiquidationCascadeGate, MultiTFConfluenceGate, CarryCostGate]:
            gate = GateClass()
            for signal in [-1, 0, 1]:
                r = gate.check(_ev(signal), {
                    "signal": signal,
                    "liquidation_volume_zscore_24": 2.5,
                    "tf4h_close_vs_ma20": -0.05,
                    "funding_rate": 0.001,
                })
                assert r.scale >= 0.0, f"{GateClass.__name__} returned negative scale"
