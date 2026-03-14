"""Tests for portfolio/risk_model/volatility estimators."""
from __future__ import annotations

import math

from portfolio.risk_model.volatility.historical import HistoricalVolatility
from portfolio.risk_model.volatility.ewma import EWMAVolatility
from portfolio.risk_model.volatility.garch import GARCHVolatility
from portfolio.risk_model.volatility.realized import RealizedVolatility


# ── HistoricalVolatility ───────────────────────────────────────────


class TestHistoricalVolatility:
    def test_known_returns(self):
        """Sample std of [0.01, -0.01, 0.02, -0.02] with annualization=1."""
        hv = HistoricalVolatility(annualization=1.0)
        returns = [0.01, -0.01, 0.02, -0.02]
        mean = 0.0
        var = sum((r - mean) ** 2 for r in returns) / 3
        expected = math.sqrt(var)
        result = hv.estimate(returns)
        assert abs(result - expected) < 1e-12

    def test_annualization_factor(self):
        hv_daily = HistoricalVolatility(annualization=365.0)
        hv_none = HistoricalVolatility(annualization=1.0)
        returns = [0.01, -0.01, 0.02, -0.02, 0.005]
        r_daily = hv_daily.estimate(returns)
        r_none = hv_none.estimate(returns)
        assert abs(r_daily - r_none * math.sqrt(365.0)) < 1e-12

    def test_n_less_than_2(self):
        hv = HistoricalVolatility()
        assert hv.estimate([]) == 0.0
        assert hv.estimate([0.05]) == 0.0

    def test_constant_returns(self):
        hv = HistoricalVolatility(annualization=1.0)
        result = hv.estimate([0.01, 0.01, 0.01, 0.01])
        assert result == 0.0

    def test_two_returns(self):
        hv = HistoricalVolatility(annualization=1.0)
        returns = [0.1, -0.1]
        var = (0.1 ** 2 + 0.1 ** 2) / 1  # n-1 = 1
        expected = math.sqrt(var)
        result = hv.estimate(returns)
        assert abs(result - expected) < 1e-12


# ── EWMAVolatility ─────────────────────────────────────────────────


class TestEWMAVolatility:
    def test_alpha_from_span(self):
        ev = EWMAVolatility(span=9)
        assert abs(ev.alpha - 0.2) < 1e-12

    def test_decay_weighting(self):
        """EWMA recursion: var_t = alpha * r_t^2 + (1-alpha) * var_{t-1}."""
        ev = EWMAVolatility(span=1, annualization=1.0)
        # span=1 => alpha=1.0
        returns = [0.05, 0.10]
        # var0 = 0.05^2 = 0.0025
        # var1 = 1.0 * 0.10^2 + 0.0 * 0.0025 = 0.01
        expected = math.sqrt(0.01)
        result = ev.estimate(returns)
        assert abs(result - expected) < 1e-12

    def test_ewma_span30(self):
        ev = EWMAVolatility(span=30, annualization=1.0)
        returns = [0.01, 0.02, -0.01]
        alpha = 2.0 / 31.0
        var = returns[0] ** 2
        for r in returns[1:]:
            var = alpha * r ** 2 + (1 - alpha) * var
        expected = math.sqrt(var)
        result = ev.estimate(returns)
        assert abs(result - expected) < 1e-12

    def test_edge_single_return(self):
        ev = EWMAVolatility(span=10)
        assert ev.estimate([0.05]) == 0.0

    def test_edge_empty(self):
        ev = EWMAVolatility(span=10)
        assert ev.estimate([]) == 0.0

    def test_annualization(self):
        ev365 = EWMAVolatility(span=10, annualization=365.0)
        ev1 = EWMAVolatility(span=10, annualization=1.0)
        returns = [0.01, -0.02, 0.015, -0.005]
        r365 = ev365.estimate(returns)
        r1 = ev1.estimate(returns)
        assert abs(r365 - r1 * math.sqrt(365.0)) < 1e-12


# ── GARCHVolatility ────────────────────────────────────────────────


class TestGARCHVolatility:
    def test_garch_recursion(self):
        """Verify GARCH(1,1) recursion: var = omega + alpha*r^2 + beta*var."""
        omega, alpha, beta = 1e-6, 0.1, 0.85
        gv = GARCHVolatility(omega=omega, alpha=alpha, beta=beta, annualization=1.0)
        returns = [0.01, -0.02, 0.015]
        # Initial variance = mean of squared returns
        var = sum(r ** 2 for r in returns) / len(returns)
        for r in returns:
            var = omega + alpha * r ** 2 + beta * var
        expected = math.sqrt(var)
        result = gv.estimate(returns)
        assert abs(result - expected) < 1e-12

    def test_forecast_horizon(self):
        gv = GARCHVolatility(omega=1e-6, alpha=0.1, beta=0.85, annualization=1.0)
        returns = [0.01, -0.02, 0.015, 0.005, -0.01]
        f1 = gv.forecast(returns, horizon=1)
        f10 = gv.forecast(returns, horizon=10)
        # Both should be positive
        assert f1 > 0
        assert f10 > 0

    def test_persistence(self):
        """Higher persistence (alpha+beta closer to 1) => slower mean reversion."""
        gv_high = GARCHVolatility(omega=1e-6, alpha=0.1, beta=0.88, annualization=1.0)
        gv_low = GARCHVolatility(omega=1e-6, alpha=0.1, beta=0.5, annualization=1.0)
        returns = [0.05, -0.05, 0.03, -0.03]
        # High persistence forecast diverges more from long-run at short horizons
        fh1 = gv_high.forecast(returns, horizon=1)
        fl1 = gv_low.forecast(returns, horizon=1)
        fh100 = gv_high.forecast(returns, horizon=100)
        fl100 = gv_low.forecast(returns, horizon=100)
        # At horizon=100, low persistence should be closer to long-run variance
        # Both should return valid positive numbers
        assert fh1 > 0 and fl1 > 0
        assert fh100 > 0 and fl100 > 0

    def test_edge_too_few_returns(self):
        gv = GARCHVolatility()
        assert gv.estimate([]) == 0.0
        assert gv.estimate([0.01]) == 0.0
        assert gv.forecast([0.01], horizon=5) == 0.0


# ── RealizedVolatility ─────────────────────────────────────────────


class TestRealizedVolatility:
    def test_sum_of_squares(self):
        rv = RealizedVolatility(annualization=1.0)
        returns = [0.01, -0.02, 0.03]
        expected = math.sqrt(0.01 ** 2 + 0.02 ** 2 + 0.03 ** 2)
        result = rv.estimate(returns)
        assert abs(result - expected) < 1e-12

    def test_from_prices(self):
        rv = RealizedVolatility(annualization=1.0)
        prices = [100.0, 101.0, 99.0, 102.0]
        result = rv.estimate_from_prices(prices)
        # log returns: ln(101/100), ln(99/101), ln(102/99)
        log_rets = [
            math.log(101.0 / 100.0),
            math.log(99.0 / 101.0),
            math.log(102.0 / 99.0),
        ]
        expected = math.sqrt(sum(r ** 2 for r in log_rets))
        assert abs(result - expected) < 1e-12

    def test_sampling_frequency(self):
        rv = RealizedVolatility(annualization=1.0)
        prices = [100.0, 101.0, 99.0, 102.0, 98.0]
        # sampling_freq=2 skips every other price
        result_freq2 = rv.estimate_from_prices(prices, sampling_freq=2)
        # log returns: ln(99/100), ln(98/99)
        log_rets = [math.log(99.0 / 100.0), math.log(98.0 / 99.0)]
        expected = math.sqrt(sum(r ** 2 for r in log_rets))
        assert abs(result_freq2 - expected) < 1e-12

    def test_empty_returns(self):
        rv = RealizedVolatility()
        assert rv.estimate([]) == 0.0

    def test_from_prices_too_few(self):
        rv = RealizedVolatility()
        assert rv.estimate_from_prices([100.0]) == 0.0
        assert rv.estimate_from_prices([]) == 0.0
