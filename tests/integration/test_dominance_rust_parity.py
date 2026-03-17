"""Verify Rust dominance features match Python DominanceComputer.

Run with:
    pytest tests/integration/test_dominance_rust_parity.py -xvs
"""

import math
import pytest


class TestDominanceRustParity:
    """500-bar parity test: Rust push_dominance vs Python DominanceComputer."""

    def test_500_bar_parity(self):
        pytest.importorskip("_quant_hotpath")
        from _quant_hotpath import RustFeatureEngine
        from features.dominance_computer import DominanceComputer

        rust_engine = RustFeatureEngine()
        py_comp = DominanceComputer()

        FEATURES = [
            "btc_dom_ratio_dev_20",
            "btc_dom_ratio_mom_10",
            "btc_dom_return_diff_6h",
            "btc_dom_return_diff_24h",
        ]

        for i in range(500):
            btc = 60000.0 + i * 10 + (i % 7) * 50
            eth = 3000.0 + i * 0.5 + (i % 5) * 25

            rust_feats = rust_engine.push_dominance(btc, eth)
            py_feats = py_comp.update(btc, eth)

            for key in FEATURES:
                rv = rust_feats.get(key)
                pv = py_feats.get(key)

                py_is_nan = pv is None or (isinstance(pv, float) and math.isnan(pv))
                rust_is_nan = rv is None or (isinstance(rv, float) and math.isnan(rv))

                if py_is_nan:
                    assert rust_is_nan, (
                        f"bar {i}, {key}: Rust={rv!r} should be NaN/None but Python={pv!r}"
                    )
                else:
                    assert not rust_is_nan, (
                        f"bar {i}, {key}: Rust=NaN/None but Python={pv!r}"
                    )
                    assert abs(rv - pv) < 1e-10, (
                        f"bar {i}, {key}: Rust={rv!r}, Python={pv!r}, diff={abs(rv - pv):.2e}"
                    )

    def test_invalid_prices_return_none(self):
        """Zero/negative prices should produce all-None output."""
        pytest.importorskip("_quant_hotpath")
        from _quant_hotpath import RustFeatureEngine

        engine = RustFeatureEngine()
        result = engine.push_dominance(0.0, 3000.0)
        for key in ["btc_dom_ratio_dev_20", "btc_dom_ratio_mom_10",
                    "btc_dom_return_diff_6h", "btc_dom_return_diff_24h"]:
            assert result[key] is None, f"{key} should be None for zero BTC price"

        result2 = engine.push_dominance(60000.0, 0.0)
        for key in ["btc_dom_ratio_dev_20", "btc_dom_ratio_mom_10",
                    "btc_dom_return_diff_6h", "btc_dom_return_diff_24h"]:
            assert result2[key] is None, f"{key} should be None for zero ETH price"

    def test_warmup_progression(self):
        """Features become available at the correct bar thresholds."""
        pytest.importorskip("_quant_hotpath")
        from _quant_hotpath import RustFeatureEngine

        engine = RustFeatureEngine()
        results = []
        for i in range(30):
            btc = 60000.0 + i * 100
            eth = 3000.0 + i * 5
            results.append(engine.push_dominance(btc, eth))

        # btc_dom_ratio_mom_10 needs 11 ratio bars (bars 0..10 → available at bar index 10)
        assert results[9]["btc_dom_ratio_mom_10"] is None
        assert results[10]["btc_dom_ratio_mom_10"] is not None

        # btc_dom_ratio_dev_20 needs 20 ratio bars (available at bar index 19)
        assert results[18]["btc_dom_ratio_dev_20"] is None
        assert results[19]["btc_dom_ratio_dev_20"] is not None

        # btc_dom_return_diff_6h needs 6 return bars (returns start at bar 1,
        # so 6 returns exist after bar index 6)
        assert results[5]["btc_dom_return_diff_6h"] is None
        assert results[6]["btc_dom_return_diff_6h"] is not None

    def test_push_dominance_independent_from_push_bar(self):
        """push_dominance state must not bleed into push_bar features."""
        pytest.importorskip("_quant_hotpath")
        from _quant_hotpath import RustFeatureEngine

        engine_with_dom = RustFeatureEngine()
        engine_without_dom = RustFeatureEngine()

        for i in range(50):
            price = 100.0 + i * 0.5
            kwargs = dict(
                close=price, volume=1000.0, high=price + 0.5,
                low=price - 0.5, open=price - 0.1,
            )
            engine_with_dom.push_bar(**kwargs)
            engine_without_dom.push_bar(**kwargs)
            engine_with_dom.push_dominance(60000.0 + i * 10, 3000.0 + i * 0.5)

        feats_with = engine_with_dom.get_features()
        feats_without = engine_without_dom.get_features()

        # All standard features should be identical
        for name, val_without in feats_without.items():
            val_with = feats_with.get(name)
            if val_without is None and val_with is None:
                continue
            if val_without is None or val_with is None:
                pytest.fail(f"{name}: with_dom={val_with!r}, without_dom={val_without!r}")
            assert abs(val_with - val_without) < 1e-12, (
                f"{name}: push_dominance polluted push_bar state "
                f"(with={val_with!r}, without={val_without!r})"
            )
