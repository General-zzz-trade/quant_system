"""Tests for alpha factor research pipeline."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Sequence

import pytest

from research.alpha_factor import (
    AlphaFactor,
    ComparisonReport,
    FactorReport,
    _pearson_corr,
    _spearman_rank_corr,
    compare_factors,
    compute_forward_returns,
    evaluate_factor,
)
from runner.backtest.csv_io import OhlcvBar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(closes: List[float], base_ts: datetime | None = None) -> List[OhlcvBar]:
    """Create OhlcvBar list from close prices (open=high=low=close for simplicity)."""
    ts = base_ts or datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = []
    for i, c in enumerate(closes):
        d = Decimal(str(c))
        bars.append(OhlcvBar(
            ts=ts + timedelta(hours=i),
            o=d, h=d, l=d, c=d,
            v=Decimal("100"),
        ))
    return bars


def _trending_closes(n: int, start: float = 100.0, drift: float = 0.001, noise: float = 0.005) -> List[float]:
    """Generate trending price series with drift + noise."""
    random.seed(42)
    prices = [start]
    for _ in range(n - 1):
        ret = drift + random.gauss(0, noise)
        prices.append(prices[-1] * (1 + ret))
    return prices


def _noisy_closes(n: int, start: float = 100.0, noise: float = 0.01) -> List[float]:
    """Generate random walk (no drift)."""
    random.seed(123)
    prices = [start]
    for _ in range(n - 1):
        ret = random.gauss(0, noise)
        prices.append(prices[-1] * (1 + ret))
    return prices


# ---------------------------------------------------------------------------
# Test: forward returns
# ---------------------------------------------------------------------------

class TestForwardReturns:
    def test_basic(self):
        closes = [100.0, 105.0, 110.0, 100.0, 120.0]
        bars = _make_bars(closes)
        rets = compute_forward_returns(bars, horizon=1)
        assert rets[0] == pytest.approx(0.05, abs=1e-9)
        assert rets[1] == pytest.approx(110.0 / 105.0 - 1, abs=1e-9)
        assert rets[-1] is None  # last bar has no forward

    def test_horizon_2(self):
        closes = [100.0, 102.0, 108.0, 105.0]
        bars = _make_bars(closes)
        rets = compute_forward_returns(bars, horizon=2)
        assert rets[0] == pytest.approx(108.0 / 100.0 - 1, abs=1e-9)
        assert rets[1] == pytest.approx(105.0 / 102.0 - 1, abs=1e-9)
        assert rets[2] is None
        assert rets[3] is None


# ---------------------------------------------------------------------------
# Test: Pearson correlation
# ---------------------------------------------------------------------------

class TestPearsonCorr:
    def test_perfect_positive(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert _pearson_corr(x, y) == pytest.approx(1.0, abs=1e-9)

    def test_perfect_negative(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        assert _pearson_corr(x, y) == pytest.approx(-1.0, abs=1e-9)

    def test_uncorrelated(self):
        random.seed(99)
        n = 1000
        x = [random.gauss(0, 1) for _ in range(n)]
        y = [random.gauss(0, 1) for _ in range(n)]
        assert abs(_pearson_corr(x, y)) < 0.1

    def test_degenerate(self):
        assert _pearson_corr([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0


# ---------------------------------------------------------------------------
# Test: Spearman rank correlation
# ---------------------------------------------------------------------------

class TestSpearmanRankCorr:
    def test_monotonic(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 20.0, 50.0, 80.0, 100.0]
        assert _spearman_rank_corr(x, y) == pytest.approx(1.0, abs=1e-9)

    def test_anti_monotonic(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [100.0, 80.0, 50.0, 20.0, 10.0]
        assert _spearman_rank_corr(x, y) == pytest.approx(-1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test: evaluate trending factor
# ---------------------------------------------------------------------------

class TestEvaluateTrending:
    def test_predictive_factor_positive_ic(self):
        """A factor that directly predicts forward returns should have positive IC."""
        # Build prices where next return = factor_value + noise
        random.seed(42)
        n = 500
        signals = [random.gauss(0, 0.01) for _ in range(n)]
        prices = [100.0]
        for i in range(n - 1):
            # Next return is correlated with current signal
            ret = signals[i] * 0.5 + random.gauss(0, 0.005)
            prices.append(prices[-1] * (1 + ret))
        bars = _make_bars(prices)

        # Factor that "leaks" the signal (simulating a good predictor)
        captured_signals = list(signals)

        def predictive_factor(b: Sequence[OhlcvBar]) -> List[Optional[float]]:
            return [captured_signals[i] if i < len(captured_signals) else None for i in range(len(b))]

        factor = AlphaFactor("predict_test", predictive_factor, "test")
        report = evaluate_factor(factor, bars, horizons=(1, 5), ic_window=50)

        assert report.name == "predict_test"
        assert report.n_observations > 100
        assert report.ic_mean > 0  # predictive factor → positive IC


# ---------------------------------------------------------------------------
# Test: evaluate noise factor
# ---------------------------------------------------------------------------

class TestEvaluateNoise:
    def test_noise_factor_near_zero_ic(self):
        """A random factor should have IC near 0."""
        closes = _noisy_closes(500)
        bars = _make_bars(closes)

        rng = random.Random(777)

        def noise_factor(b: Sequence[OhlcvBar]) -> List[Optional[float]]:
            return [rng.gauss(0, 1) for _ in b]

        factor = AlphaFactor("noise_test", noise_factor, "random")
        report = evaluate_factor(factor, bars, horizons=(1,), ic_window=50)

        assert report.n_observations > 100
        assert abs(report.ic_mean) < 0.15  # should be near zero


# ---------------------------------------------------------------------------
# Test: compare factors
# ---------------------------------------------------------------------------

class TestCompareFactors:
    def test_comparison_structure(self):
        closes = _trending_closes(300)
        bars = _make_bars(closes)

        def mom5(b: Sequence[OhlcvBar]) -> List[Optional[float]]:
            result: List[Optional[float]] = [None] * len(b)
            for i in range(5, len(b)):
                prev = float(b[i - 5].c)
                if prev != 0:
                    result[i] = float(b[i].c) / prev - 1.0
            return result

        def mom20(b: Sequence[OhlcvBar]) -> List[Optional[float]]:
            result: List[Optional[float]] = [None] * len(b)
            for i in range(20, len(b)):
                prev = float(b[i - 20].c)
                if prev != 0:
                    result[i] = float(b[i].c) / prev - 1.0
            return result

        factors = [
            AlphaFactor("mom5", mom5, "momentum"),
            AlphaFactor("mom20", mom20, "momentum"),
        ]
        report = compare_factors(factors, bars, horizons=(1,), ic_window=50)

        assert len(report.factor_reports) == 2
        assert "mom5" in report.correlation_matrix
        assert "mom20" in report.correlation_matrix["mom5"]
        assert report.correlation_matrix["mom5"]["mom5"] == pytest.approx(1.0, abs=0.01)
        assert "mom5" in report.marginal_ic
        assert "mom20" in report.marginal_ic


# ---------------------------------------------------------------------------
# Test: FactorDecisionModule
# ---------------------------------------------------------------------------

class TestFactorDecisionModule:
    def test_module_runs_without_error(self):
        """FactorDecisionModule should integrate with run_backtest."""
        import csv
        import tempfile
        from research.factor_backtest import FactorDecisionModule, FactorStrategyConfig

        closes = _trending_closes(200, drift=0.003, noise=0.01)
        bars = _make_bars(closes)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["ts", "open", "high", "low", "close", "volume"])
            for bar in bars:
                writer.writerow([
                    bar.ts.isoformat(),
                    str(bar.o), str(bar.h), str(bar.l), str(bar.c),
                    str(bar.v),
                ])
            tmp_path = f.name

        try:
            from runner.backtest_runner import run_backtest

            def simple_mom(b: Sequence[OhlcvBar]) -> List[Optional[float]]:
                result: List[Optional[float]] = [None] * len(b)
                for i in range(10, len(b)):
                    prev = float(b[i - 10].c)
                    if prev != 0:
                        result[i] = float(b[i].c) / prev - 1.0
                return result

            factor = AlphaFactor("test_mom", simple_mom, "momentum")
            cfg = FactorStrategyConfig(symbol="BTCUSDT", zscore_window=30)
            module = FactorDecisionModule(factor, cfg)

            equity, fills = run_backtest(
                csv_path=Path(tmp_path),
                symbol="BTCUSDT",
                starting_balance=Decimal("10000"),
                fee_bps=Decimal("4"),
                decision_modules=[module],
            )
            assert len(equity) > 0
        finally:
            import os
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test: backtest_factor e2e
# ---------------------------------------------------------------------------

class TestBacktestFactorE2E:
    def test_returns_summary(self):
        import csv
        import tempfile
        from research.factor_backtest import backtest_factor

        closes = _trending_closes(200, drift=0.003, noise=0.01)
        bars = _make_bars(closes)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["ts", "open", "high", "low", "close", "volume"])
            for bar in bars:
                writer.writerow([
                    bar.ts.isoformat(),
                    str(bar.o), str(bar.h), str(bar.l), str(bar.c),
                    str(bar.v),
                ])
            tmp_path = f.name

        try:
            def simple_mom(b: Sequence[OhlcvBar]) -> List[Optional[float]]:
                result: List[Optional[float]] = [None] * len(b)
                for i in range(10, len(b)):
                    prev = float(b[i - 10].c)
                    if prev != 0:
                        result[i] = float(b[i].c) / prev - 1.0
                return result

            factor = AlphaFactor("test_mom", simple_mom, "momentum")
            summary = backtest_factor(factor, Path(tmp_path))
            assert "return" in summary
            assert "max_drawdown" in summary
            assert "sharpe_ratio" in summary
            assert summary["factor_name"] == "test_mom"
            assert summary["bars"] > 0
        finally:
            import os
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test: all built-in factors compute without errors
# ---------------------------------------------------------------------------

class TestBuiltinFactors:
    def test_all_builtins(self):
        from scripts.run_alpha_research import BUILTIN_FACTORS

        closes = _trending_closes(300)
        bars = _make_bars(closes)

        for name, factor in BUILTIN_FACTORS.items():
            vals = factor.compute_fn(bars)
            assert len(vals) == len(bars), f"{name} returned wrong length"
            non_none = [v for v in vals if v is not None]
            assert len(non_none) > 0, f"{name} produced all None values"
            for v in non_none:
                assert math.isfinite(v), f"{name} produced non-finite value: {v}"
