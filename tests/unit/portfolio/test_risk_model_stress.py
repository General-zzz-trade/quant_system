"""Tests for portfolio/risk_model/stress testing modules."""
from __future__ import annotations

from portfolio.risk_model.stress.scenarios import (
    CORRELATION_SPIKE,
    CRYPTO_CRASH_2022,
    FLASH_CRASH,
    ScenarioType,
    StressScenario,
)
from portfolio.risk_model.stress.historical import HistoricalStress
from portfolio.risk_model.stress.factor_shock import FactorShock, FactorShockTest


# ── StressScenario ─────────────────────────────────────────────────


class TestStressScenario:
    def test_predefined_scenarios_exist(self):
        assert CRYPTO_CRASH_2022.name == "crypto_crash_2022"
        assert FLASH_CRASH.scenario_type == ScenarioType.HYPOTHETICAL
        assert CORRELATION_SPIKE.shocks["BTC"] == -0.15

    def test_crypto_crash_shocks(self):
        assert CRYPTO_CRASH_2022.shocks["BTC"] == -0.50
        assert CRYPTO_CRASH_2022.shocks["ETH"] == -0.60
        assert CRYPTO_CRASH_2022.shocks["SOL"] == -0.80

    def test_custom_scenario(self):
        scenario = StressScenario(
            name="custom_drop",
            scenario_type=ScenarioType.HYPOTHETICAL,
            shocks={"BTC": -0.10, "ETH": -0.15},
            description="mild correction",
        )
        assert scenario.name == "custom_drop"
        assert scenario.shocks["BTC"] == -0.10
        assert scenario.description == "mild correction"

    def test_scenario_immutability(self):
        try:
            CRYPTO_CRASH_2022.name = "changed"
            assert False, "Should have raised"
        except AttributeError:
            pass


# ── HistoricalStress ───────────────────────────────────────────────


class TestHistoricalStress:
    def test_portfolio_loss_calculation(self):
        """Loss = sum(weight_i * shock_i) for each asset."""
        hs = HistoricalStress()
        weights = {"BTC": 0.5, "ETH": 0.3, "SOL": 0.2}
        result = hs.run(weights, CRYPTO_CRASH_2022)
        expected = 0.5 * (-0.50) + 0.3 * (-0.60) + 0.2 * (-0.80)
        assert abs(result.portfolio_loss - expected) < 1e-12
        assert result.scenario_name == "crypto_crash_2022"

    def test_asset_losses(self):
        hs = HistoricalStress()
        weights = {"BTC": 0.6, "ETH": 0.4}
        result = hs.run(weights, FLASH_CRASH)
        assert abs(result.asset_losses["BTC"] - 0.6 * (-0.20)) < 1e-12
        assert abs(result.asset_losses["ETH"] - 0.4 * (-0.25)) < 1e-12

    def test_max_drawdown(self):
        hs = HistoricalStress()
        weights = {"BTC": 0.5, "ETH": 0.5}
        result = hs.run(weights, FLASH_CRASH)
        expected_loss = 0.5 * (-0.20) + 0.5 * (-0.25)
        assert abs(result.max_drawdown - abs(expected_loss)) < 1e-12

    def test_missing_symbol_shock(self):
        """Assets not in scenario shocks get shock=0."""
        hs = HistoricalStress()
        weights = {"BTC": 0.5, "DOGE": 0.5}
        result = hs.run(weights, CRYPTO_CRASH_2022)
        expected = 0.5 * (-0.50) + 0.5 * 0.0
        assert abs(result.portfolio_loss - expected) < 1e-12

    def test_worst_n_days(self):
        hs = HistoricalStress()
        weights = {"A": 0.6, "B": 0.4}
        returns = {
            "A": [0.01, -0.05, 0.02, -0.08, 0.03],
            "B": [0.02, -0.03, 0.01, -0.06, 0.04],
        }
        results = hs.run_from_returns(weights, returns, worst_n=2)
        assert len(results) == 2
        # Worst day should be day 3: 0.6*(-0.08) + 0.4*(-0.06) = -0.072
        assert results[0].portfolio_loss < results[1].portfolio_loss

    def test_run_from_returns_empty(self):
        hs = HistoricalStress()
        results = hs.run_from_returns({"A": 0.5}, {"A": []}, worst_n=3)
        assert results == []


# ── FactorShockTest ────────────────────────────────────────────────


class TestFactorShockTest:
    def test_factor_transmission(self):
        """shock propagates through betas: asset_shock = sum(beta_f * shock_f)."""
        fst = FactorShockTest()
        weights = {"BTC": 0.5, "ETH": 0.5}
        exposures = {
            "BTC": {"market": 1.0, "momentum": 0.5},
            "ETH": {"market": 1.2, "momentum": 0.8},
        }
        shocks = [
            FactorShock(factor_name="market", shock_magnitude=-0.10),
            FactorShock(factor_name="momentum", shock_magnitude=-0.05),
        ]
        result = fst.run(weights, exposures, shocks)

        btc_shock = 1.0 * (-0.10) + 0.5 * (-0.05)
        eth_shock = 1.2 * (-0.10) + 0.8 * (-0.05)
        expected_loss = 0.5 * btc_shock + 0.5 * eth_shock
        assert abs(result.portfolio_loss - expected_loss) < 1e-12

    def test_portfolio_aggregation(self):
        fst = FactorShockTest()
        weights = {"A": 1.0}
        exposures = {"A": {"market": 2.0}}
        shocks = [FactorShock(factor_name="market", shock_magnitude=-0.05)]
        result = fst.run(weights, exposures, shocks)
        assert abs(result.portfolio_loss - 1.0 * 2.0 * (-0.05)) < 1e-12
        assert abs(result.max_drawdown - 0.10) < 1e-12

    def test_missing_factor_exposure(self):
        """Assets with no exposure to a factor get 0 shock from it."""
        fst = FactorShockTest()
        weights = {"A": 1.0}
        exposures = {"A": {}}
        shocks = [FactorShock(factor_name="market", shock_magnitude=-0.10)]
        result = fst.run(weights, exposures, shocks)
        assert abs(result.portfolio_loss) < 1e-12

    def test_scenario_name(self):
        fst = FactorShockTest()
        shocks = [
            FactorShock(factor_name="alpha", shock_magnitude=-0.1),
            FactorShock(factor_name="beta", shock_magnitude=-0.2),
        ]
        result = fst.run({"A": 1.0}, {"A": {"alpha": 1.0, "beta": 1.0}}, shocks)
        assert result.scenario_name == "factor_shock_alpha+beta"
