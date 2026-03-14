# tests/integration/test_fast_slow_path_parity.py
"""Fast/slow path feature parity tests.

The coordinator has two paths for processing market events:
- **Fast path** (tick_processor): single Rust call `process_tick_full()`
- **Slow path** (no tick_processor): Python `feature_hook.on_event()` + `pipeline.apply()`

This module validates that RustFeatureEngine (used by both paths for feature
computation) produces deterministic, identical features for identical inputs.

Full end-to-end coordinator parity requires production model files for
RustTickProcessor; the simpler feature-engine-level test below validates the
shared feature computation component without model dependencies.
"""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "/quant_system")

_hp = pytest.importorskip("_quant_hotpath")
RustFeatureEngine = _hp.RustFeatureEngine


# ============================================================
# Helper: push a bar sequence and collect features
# ============================================================

def _push_bars(engine: RustFeatureEngine, bars: list[dict]) -> list[dict]:
    """Push a sequence of bars into a RustFeatureEngine and return features."""
    results = []
    for bar in bars:
        engine.push_bar(
            bar["close"], bar["volume"], bar["high"], bar["low"], bar["open"],
        )
        feats = engine.get_features()
        results.append(dict(feats))
    return results


def _make_bars(n: int = 200, seed: int = 42) -> list[dict]:
    """Generate deterministic synthetic bar data."""
    import random
    rng = random.Random(seed)
    price = 40000.0
    bars = []
    for _ in range(n):
        change = rng.gauss(0, 50)
        price = max(price + change, 100.0)
        high = price + abs(rng.gauss(0, 20))
        low = price - abs(rng.gauss(0, 20))
        volume = max(rng.gauss(100, 30), 1.0)
        bars.append({
            "close": price,
            "high": high,
            "low": low,
            "open": price - change * 0.5,
            "volume": volume,
        })
    return bars


# ============================================================
# a. Two independent engines produce identical features
# ============================================================

class TestFeatureEngineDeterminism:
    """Two independent RustFeatureEngine instances fed identical bars
    must produce bit-identical features — this is the foundation of
    fast/slow path parity."""

    def test_two_engines_identical_output(self):
        bars = _make_bars(n=200)
        engine_a = RustFeatureEngine()
        engine_b = RustFeatureEngine()

        feats_a = _push_bars(engine_a, bars)
        feats_b = _push_bars(engine_b, bars)

        assert len(feats_a) == len(feats_b)
        for i, (fa, fb) in enumerate(zip(feats_a, feats_b)):
            assert fa.keys() == fb.keys(), f"Key mismatch at bar {i}"
            for key in fa:
                va, vb = fa[key], fb[key]
                if isinstance(va, float) and isinstance(vb, float):
                    # NaN == NaN for this comparison
                    import math
                    if math.isnan(va) and math.isnan(vb):
                        continue
                    assert va == vb, f"Mismatch at bar {i}, feature {key}: {va} != {vb}"
                else:
                    assert va == vb, f"Mismatch at bar {i}, feature {key}: {va} != {vb}"

    def test_features_non_empty_after_warmup(self):
        """After sufficient bars, features dict should contain values."""
        bars = _make_bars(n=100)
        engine = RustFeatureEngine()
        feats = _push_bars(engine, bars)
        last = feats[-1]
        assert len(last) > 0, "Features should be non-empty after 100 bars"

    def test_feature_count_stable(self):
        """Feature count should stabilize after warmup."""
        bars = _make_bars(n=120)
        engine = RustFeatureEngine()
        feats = _push_bars(engine, bars)
        # After warmup, count should be stable
        counts = [len(f) for f in feats[80:]]
        assert len(set(counts)) == 1, f"Feature count should stabilize, got {set(counts)}"


# ============================================================
# b. Full coordinator fast/slow path parity (requires models)
# ============================================================

@pytest.mark.skip(reason="Requires production model files for RustTickProcessor")
class TestCoordinatorFastSlowParity:
    """Full fast/slow path parity test at coordinator level.

    This test would:
    1. Create a coordinator with tick_processor (fast path)
    2. Create a coordinator without tick_processor (slow path)
    3. Feed identical market events to both
    4. Compare output snapshots

    Currently covered partially by:
    - tests/integration/test_cross_path_parity.py (signal constraints)
    - tests/unit/features/test_rust_feature_engine.py (feature computation)
    - TestFeatureEngineDeterminism above (engine-level determinism)
    """

    def test_fast_slow_path_parity_placeholder(self):
        pass
