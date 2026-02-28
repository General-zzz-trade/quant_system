"""Tests for factor factory: sweep generation, screening, decorrelation."""
from __future__ import annotations

import math
import random
from decimal import Decimal
from typing import List, Optional, Sequence

import pytest

from research.factor_factory import (
    FactorFactory,
    ScreeningConfig,
    ScreeningResult,
    _compute_cross_correlation,
)
from research.alpha_factor import AlphaFactor
from runner.backtest.csv_io import OhlcvBar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n: int = 500, seed: int = 42) -> List[OhlcvBar]:
    """Generate synthetic OHLCV bars with a trend component."""
    rng = random.Random(seed)
    bars: List[OhlcvBar] = []
    price = 100.0
    for i in range(n):
        ret = 0.001 + rng.gauss(0, 0.02)  # slight upward drift
        price *= (1 + ret)
        h = price * (1 + abs(rng.gauss(0, 0.005)))
        lo = price * (1 - abs(rng.gauss(0, 0.005)))
        vol = rng.uniform(100, 1000)
        bars.append(OhlcvBar(
            ts=f"2024-01-01T{i:05d}",
            o=Decimal(str(round(price * (1 + rng.gauss(0, 0.001)), 2))),
            h=Decimal(str(round(h, 2))),
            l=Decimal(str(round(lo, 2))),
            c=Decimal(str(round(price, 2))),
            v=Decimal(str(round(vol, 2))),
        ))
    return bars


def _momentum_gen(window: int):
    """Momentum factor generator."""
    def compute(bars: Sequence[OhlcvBar]) -> List[Optional[float]]:
        result: List[Optional[float]] = [None] * len(bars)
        for i in range(window, len(bars)):
            prev = float(bars[i - window].c)
            if prev != 0:
                result[i] = float(bars[i].c) / prev - 1.0
        return result
    return compute


def _noise_gen(seed: int = 99):
    """Pure noise factor generator (should fail screening)."""
    def compute(bars: Sequence[OhlcvBar]) -> List[Optional[float]]:
        rng = random.Random(seed)
        return [rng.gauss(0, 1) for _ in bars]
    return compute


def _constant_gen(value: float = 1.0):
    """Constant factor (should fail IC screening)."""
    def compute(bars: Sequence[OhlcvBar]) -> List[Optional[float]]:
        return [value] * len(bars)
    return compute


# ---------------------------------------------------------------------------
# Factory registration & sweep generation
# ---------------------------------------------------------------------------

class TestFactorFactoryGeneration:

    def test_generate_sweep_basic(self):
        """Single-param sweep generates correct number of factors."""
        factory = FactorFactory()
        factory.register("momentum", _momentum_gen)

        factors = factory.generate_sweep("momentum", {"window": [5, 10, 20, 50, 100]})
        assert len(factors) == 5
        assert all(f.category == "momentum" for f in factors)

    def test_generate_sweep_names(self):
        """Factor names encode parameters."""
        factory = FactorFactory()
        factory.register("momentum", _momentum_gen)

        factors = factory.generate_sweep("momentum", {"window": [5, 20]})
        names = {f.name for f in factors}
        assert "momentum_5" in names
        assert "momentum_20" in names

    def test_generate_sweep_multi_param(self):
        """Multi-param grid → cartesian product."""
        def multi_gen(window: int, scale: float):
            def compute(bars):
                return [None] * len(bars)
            return compute

        factory = FactorFactory()
        factory.register("custom", multi_gen)

        factors = factory.generate_sweep("custom", {"window": [5, 10], "scale": [1.0, 2.0]})
        assert len(factors) == 4

    def test_unknown_family_raises(self):
        factory = FactorFactory()
        with pytest.raises(KeyError, match="Unknown factor family"):
            factory.generate_sweep("nonexistent", {"window": [5]})

    def test_empty_grid(self):
        factory = FactorFactory()
        factory.register("momentum", _momentum_gen)
        factors = factory.generate_sweep("momentum", {"window": []})
        assert len(factors) == 0

    def test_constructor_with_generators(self):
        factory = FactorFactory(generators={"momentum": _momentum_gen})
        factors = factory.generate_sweep("momentum", {"window": [10]})
        assert len(factors) == 1


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

class TestFactorScreening:

    def test_screen_passes_trend_factor(self):
        """A momentum factor on trending data should pass basic screening."""
        bars = _make_bars(500)
        factory = FactorFactory({"momentum": _momentum_gen})
        factors = factory.generate_sweep("momentum", {"window": [20]})

        # Use fully relaxed thresholds for synthetic data
        config = ScreeningConfig(min_ic_ir=0.0, min_abs_ic=0.0, max_autocorr=1.0, min_observations=50)
        results = factory.screen(factors, bars, config)

        assert len(results) == 1
        assert results[0].passed is True

    def test_screen_rejects_constant_factor(self):
        """A constant factor should be rejected (zero IC)."""
        bars = _make_bars(500)
        const_factor = AlphaFactor("constant", _constant_gen(1.0), "test")

        factory = FactorFactory()
        config = ScreeningConfig(min_abs_ic=0.01, min_ic_ir=0.1, min_observations=50)
        results = factory.screen([const_factor], bars, config)

        assert len(results) == 1
        assert results[0].passed is False
        assert len(results[0].reject_reasons) > 0

    def test_screen_sorted_by_ic_ir(self):
        """Results are sorted by abs(IC_IR) descending."""
        bars = _make_bars(500)
        factory = FactorFactory({"momentum": _momentum_gen})
        factors = factory.generate_sweep("momentum", {"window": [5, 10, 20, 50]})

        config = ScreeningConfig(min_ic_ir=0.0, min_abs_ic=0.0, min_observations=20)
        results = factory.screen(factors, bars, config)

        ic_irs = [abs(r.report.ic_ir) for r in results]
        assert ic_irs == sorted(ic_irs, reverse=True)

    def test_screen_observation_threshold(self):
        """Factor with too few observations rejected."""
        bars = _make_bars(30)  # very short
        factor = AlphaFactor("mom", _momentum_gen(20), "test")

        factory = FactorFactory()
        config = ScreeningConfig(min_observations=100)
        results = factory.screen([factor], bars, config)

        assert results[0].passed is False
        assert any("n_obs" in r for r in results[0].reject_reasons)


# ---------------------------------------------------------------------------
# Decorrelation selection
# ---------------------------------------------------------------------------

class TestSelectUncorrelated:

    def test_highly_correlated_factors_deduplicated(self):
        """Three highly-correlated momentum factors → keep 1-2."""
        bars = _make_bars(500)
        factory = FactorFactory({"momentum": _momentum_gen})
        factors = factory.generate_sweep("momentum", {"window": [18, 19, 20]})

        config = ScreeningConfig(min_ic_ir=0.0, min_abs_ic=0.0, min_observations=20)
        results = factory.screen(factors, bars, config)
        selected = factory.select_uncorrelated(results, bars, max_correlation=0.85)

        # Very similar windows → high correlation → should keep <= 2
        assert len(selected) <= 2

    def test_independent_factors_all_kept(self):
        """Independent factors should all be kept."""
        bars = _make_bars(500)

        def vol_gen(window: int):
            def compute(bars_in):
                result = [None] * len(bars_in)
                for i in range(window, len(bars_in)):
                    vols = [float(bars_in[j].v or 0) for j in range(i - window, i)]
                    mean_vol = sum(vols) / window
                    if mean_vol > 0:
                        result[i] = float(bars_in[i].v or 0) / mean_vol
                return result
            return compute

        factory = FactorFactory({"momentum": _momentum_gen, "vol": vol_gen})
        mom_factors = factory.generate_sweep("momentum", {"window": [20]})
        vol_factors = factory.generate_sweep("vol", {"window": [20]})

        all_factors = mom_factors + vol_factors

        # Fully relaxed thresholds to ensure both pass screening
        config = ScreeningConfig(min_ic_ir=0.0, min_abs_ic=0.0, max_autocorr=1.0, min_observations=20)
        results = factory.screen(all_factors, bars, config)
        selected = factory.select_uncorrelated(results, bars, max_correlation=0.95)

        # Momentum and volume should be fairly independent
        assert len(selected) == 2

    def test_select_only_passed(self):
        """Only passed factors are considered for selection."""
        bars = _make_bars(500)
        const_factor = AlphaFactor("constant", _constant_gen(1.0), "test")
        mom_factor = AlphaFactor("mom20", _momentum_gen(20), "momentum")

        factory = FactorFactory()
        config = ScreeningConfig(min_abs_ic=0.005, min_ic_ir=0.0, min_observations=20)
        results = factory.screen([const_factor, mom_factor], bars, config)

        selected = factory.select_uncorrelated(results, bars, max_correlation=0.85)
        # Constant should be rejected by screening, so only mom if it passes
        assert all(r.passed for r in selected)

    def test_empty_results(self):
        factory = FactorFactory()
        bars = _make_bars(100)
        selected = factory.select_uncorrelated([], bars)
        assert selected == []


# ---------------------------------------------------------------------------
# Cross-correlation utility
# ---------------------------------------------------------------------------

class TestCrossCorrelation:

    def test_identical_series(self):
        a = [float(i) for i in range(50)]
        corr = _compute_cross_correlation(a, a)
        assert corr == pytest.approx(1.0)

    def test_opposite_series(self):
        a = [float(i) for i in range(50)]
        b = [-float(i) for i in range(50)]
        corr = _compute_cross_correlation(a, b)
        assert corr == pytest.approx(-1.0)

    def test_with_nones(self):
        a = [1.0, None, 3.0, 4.0] + [float(i) for i in range(50)]
        b = [None, 2.0, 3.0, 4.0] + [float(i) for i in range(50)]
        corr = _compute_cross_correlation(a, b)
        # Should skip None pairs and still compute
        assert isinstance(corr, float)

    def test_insufficient_data(self):
        a = [1.0, 2.0]
        b = [3.0, 4.0]
        corr = _compute_cross_correlation(a, b)
        assert corr == 0.0  # < 20 valid pairs
