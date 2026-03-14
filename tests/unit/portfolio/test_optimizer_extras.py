"""Tests for portfolio optimizer diagnostics, stress constraint, views, rebalancer, and risk budget."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from portfolio.optimizer.base import OptimizationResult
from portfolio.optimizer.constraints import LongOnlyConstraint
from portfolio.optimizer.diagnostics import (
    check_concentration,
    check_convergence,
    check_feasibility,
    check_weight_sum,
    run_diagnostics,
)
from portfolio.optimizer.stress_constraint import (
    StressConstraintConfig,
    StressLossConstraint,
)
from portfolio.optimizer.views import ViewGenerator
from portfolio.rebalance import (
    InstrumentRules,
    RebalanceConfig,
    Rebalancer,
)
from portfolio.risk_budget import RiskBudget, RiskBudgetObjective


# ── Diagnostics ────────────────────────────────────────────────────


class TestDiagnostics:
    def _make_result(self, weights, converged=True, message="ok"):
        return OptimizationResult(
            weights=weights,
            objective_value=0.0,
            converged=converged,
            iterations=10,
            message=message,
        )

    def test_convergence_pass(self):
        result = self._make_result({"A": 0.5, "B": 0.5})
        item = check_convergence(result)
        assert item.passed is True

    def test_convergence_fail(self):
        result = self._make_result({"A": 0.5, "B": 0.5}, converged=False)
        item = check_convergence(result)
        assert item.passed is False

    def test_feasibility_pass(self):
        result = self._make_result({"A": 0.5, "B": 0.5})
        item = check_feasibility(result, [LongOnlyConstraint()])
        assert item.passed is True

    def test_feasibility_fail(self):
        result = self._make_result({"A": 0.5, "B": -0.5})
        item = check_feasibility(result, [LongOnlyConstraint()])
        assert item.passed is False
        assert "long_only" in item.detail

    def test_weight_sum_pass(self):
        result = self._make_result({"A": 0.5, "B": 0.5})
        item = check_weight_sum(result, tolerance=0.01)
        assert item.passed is True

    def test_weight_sum_fail(self):
        result = self._make_result({"A": 0.5, "B": 0.3})
        item = check_weight_sum(result, tolerance=0.01)
        assert item.passed is False

    def test_concentration_pass(self):
        result = self._make_result({"A": 0.3, "B": 0.3, "C": 0.4})
        item = check_concentration(result, max_weight=0.5)
        assert item.passed is True

    def test_concentration_fail(self):
        result = self._make_result({"A": 0.8, "B": 0.2})
        item = check_concentration(result, max_weight=0.5)
        assert item.passed is False

    def test_run_diagnostics_all_pass(self):
        result = self._make_result({"A": 0.4, "B": 0.3, "C": 0.3})
        report = run_diagnostics(result, max_weight=0.5)
        assert report.all_passed is True
        assert len(report.items) == 4

    def test_run_diagnostics_partial_fail(self):
        result = self._make_result({"A": 0.9, "B": 0.1}, converged=False)
        report = run_diagnostics(result, max_weight=0.5)
        assert report.all_passed is False


# ── StressLossConstraint ───────────────────────────────────────────


class TestStressLossConstraint:
    def test_is_feasible_no_fn(self):
        slc = StressLossConstraint()
        assert slc.is_feasible({"A": 0.5, "B": 0.5}) is True

    def test_is_feasible_within_threshold(self):
        config = StressConstraintConfig(max_stress_loss_pct=20.0)
        slc = StressLossConstraint(config=config, stress_fn=lambda w: 10.0)
        assert slc.is_feasible({"A": 0.5}) is True

    def test_is_feasible_above_threshold(self):
        config = StressConstraintConfig(max_stress_loss_pct=20.0)
        slc = StressLossConstraint(config=config, stress_fn=lambda w: 25.0)
        assert slc.is_feasible({"A": 0.5}) is False

    def test_project_feasible(self):
        """Already feasible weights should be returned unchanged."""
        config = StressConstraintConfig(max_stress_loss_pct=20.0)
        slc = StressLossConstraint(config=config, stress_fn=lambda w: 10.0)
        weights = {"A": 0.5, "B": 0.5}
        result = slc.project(weights)
        assert result == weights

    def test_project_binary_search(self):
        """Infeasible weights should be scaled down."""
        config = StressConstraintConfig(max_stress_loss_pct=10.0)
        # stress_fn returns sum of abs weights * 100 (linear in scale)
        def stress_fn(w):
            return sum(abs(v) for v in w.values()) * 100.0

        slc = StressLossConstraint(config=config, stress_fn=stress_fn)
        weights = {"A": 0.5, "B": 0.5}
        # Original loss = 100.0, threshold = 10.0
        projected = slc.project(weights)
        # Projected should have smaller weights
        assert sum(abs(v) for v in projected.values()) < sum(abs(v) for v in weights.values())
        # And should now be feasible
        assert stress_fn(projected) <= 10.0 + 1e-6

    def test_stress_loss_no_fn(self):
        slc = StressLossConstraint()
        assert slc.stress_loss({"A": 0.5}) == 0.0


# ── ViewGenerator ──────────────────────────────────────────────────


class TestViewGenerator:
    def test_from_signals(self):
        vg = ViewGenerator(base_confidence=0.5, signal_scaling=0.1)
        signals = {"BTC": 0.8, "ETH": -0.5, "SOL": 0.0}
        views = vg.from_signals(signals, ["BTC", "ETH", "SOL"])
        assert len(views) == 2  # SOL has 0 signal
        btc_view = [v for v in views if v.assets == ("BTC",)][0]
        assert abs(btc_view.expected_return - 0.08) < 1e-12
        assert abs(btc_view.confidence - 0.4) < 1e-12

    def test_from_signals_negative(self):
        vg = ViewGenerator(base_confidence=1.0, signal_scaling=0.1)
        signals = {"X": -0.6}
        views = vg.from_signals(signals, ["X"])
        assert len(views) == 1
        assert views[0].expected_return < 0
        assert abs(views[0].expected_return - (-0.06)) < 1e-12

    def test_relative_view(self):
        vg = ViewGenerator()
        view = vg.relative_view("BTC", "ETH", expected_diff=0.02, confidence=0.8)
        assert view.assets == ("BTC", "ETH")
        assert view.weights == (1.0, -1.0)
        assert view.expected_return == 0.02
        assert view.confidence == 0.8

    def test_multi_asset_view(self):
        vg = ViewGenerator()
        view = vg.multi_asset_view(
            assets=["A", "B", "C"],
            weights=[0.5, 0.3, 0.2],
            expected_return=0.05,
            confidence=0.9,
        )
        assert view.assets == ("A", "B", "C")
        assert view.weights == (0.5, 0.3, 0.2)
        assert view.expected_return == 0.05

    def test_from_signals_skips_zero(self):
        vg = ViewGenerator()
        views = vg.from_signals({"A": 0.0, "B": 1e-15}, ["A", "B"])
        assert len(views) == 0


# ── Rebalancer ─────────────────────────────────────────────────────


@dataclass
class FakeAccount:
    equity: Decimal
    positions_qty: Mapping[str, Decimal]


class FakePrice:
    def __init__(self, prices: dict[str, Decimal]):
        self._prices = prices

    def price(self, symbol: str) -> Decimal:
        return self._prices[symbol]


class TestRebalancer:
    def _make_rules(self):
        return {
            "BTC": InstrumentRules(
                symbol="BTC",
                qty_step=Decimal("0.001"),
                min_qty=Decimal("0.001"),
                min_notional=Decimal("5"),
            ),
            "ETH": InstrumentRules(
                symbol="ETH",
                qty_step=Decimal("0.01"),
                min_qty=Decimal("0.01"),
                min_notional=Decimal("5"),
            ),
        }

    def test_basic_build_plan(self):
        rb = Rebalancer(cfg=RebalanceConfig(
            deadband_notional_pct=Decimal("0"),
            per_symbol_delta_cap_pct=None,
        ))
        account = FakeAccount(
            equity=Decimal("10000"),
            positions_qty={"BTC": Decimal("0"), "ETH": Decimal("0")},
        )
        prices = FakePrice({"BTC": Decimal("30000"), "ETH": Decimal("2000")})
        plan = rb.build_plan(
            ts=0,
            symbols=["BTC", "ETH"],
            targets_qty={"BTC": Decimal("0.1"), "ETH": Decimal("1.0")},
            account=account,
            prices=prices,
            rules=self._make_rules(),
        )
        assert len(plan.intents) == 2
        syms = {i.symbol for i in plan.intents}
        assert "BTC" in syms and "ETH" in syms

    def test_deadband_filtering(self):
        """Small deltas below deadband are filtered out."""
        cfg = RebalanceConfig(
            deadband_notional_pct=Decimal("0.01"),
            per_symbol_delta_cap_pct=None,
        )
        rb = Rebalancer(cfg=cfg)
        account = FakeAccount(
            equity=Decimal("10000"),
            positions_qty={"BTC": Decimal("0.100"), "ETH": Decimal("1.00")},
        )
        prices = FakePrice({"BTC": Decimal("30000"), "ETH": Decimal("2000")})
        # Tiny delta: 0.001 BTC * 30000 = 30 USDT > 100 deadband? No.
        # deadband = 0.01 * 10000 = 100
        # BTC delta = 0.001 * 30000 = 30 < 100 => filtered
        plan = rb.build_plan(
            ts=0,
            symbols=["BTC", "ETH"],
            targets_qty={"BTC": Decimal("0.101"), "ETH": Decimal("1.50")},
            account=account,
            prices=prices,
            rules=self._make_rules(),
        )
        # BTC delta too small, ETH delta = 0.50 * 2000 = 1000 > 100
        btc_intents = [i for i in plan.intents if i.symbol == "BTC"]
        eth_intents = [i for i in plan.intents if i.symbol == "ETH"]
        assert len(btc_intents) == 0
        assert len(eth_intents) == 1

    def test_reduce_only_flag(self):
        """Reducing position should set reduce_only=True."""
        rb = Rebalancer(cfg=RebalanceConfig(
            deadband_notional_pct=Decimal("0"),
            per_symbol_delta_cap_pct=None,
        ))
        account = FakeAccount(
            equity=Decimal("10000"),
            positions_qty={"BTC": Decimal("0.5")},
        )
        prices = FakePrice({"BTC": Decimal("30000")})
        rules = {
            "BTC": InstrumentRules(
                symbol="BTC", qty_step=Decimal("0.001"),
                min_qty=Decimal("0.001"), min_notional=Decimal("5"),
            ),
        }
        plan = rb.build_plan(
            ts=0,
            symbols=["BTC"],
            targets_qty={"BTC": Decimal("0.1")},
            account=account,
            prices=prices,
            rules=rules,
        )
        assert len(plan.intents) == 1
        assert plan.intents[0].reduce_only is True
        assert plan.intents[0].qty_delta < 0

    def test_delta_capping(self):
        """Per-symbol delta cap limits the notional change."""
        cfg = RebalanceConfig(
            deadband_notional_pct=Decimal("0"),
            per_symbol_delta_cap_pct=Decimal("0.05"),  # 5% of equity
        )
        rb = Rebalancer(cfg=cfg)
        account = FakeAccount(
            equity=Decimal("10000"),
            positions_qty={"BTC": Decimal("0")},
        )
        prices = FakePrice({"BTC": Decimal("30000")})
        rules = {
            "BTC": InstrumentRules(
                symbol="BTC", qty_step=Decimal("0.001"),
                min_qty=Decimal("0.001"), min_notional=Decimal("5"),
            ),
        }
        plan = rb.build_plan(
            ts=0,
            symbols=["BTC"],
            targets_qty={"BTC": Decimal("1.0")},  # 30000 USDT >> 500 cap
            account=account,
            prices=prices,
            rules=rules,
        )
        assert len(plan.intents) == 1
        # Notional should be capped near 500
        assert plan.intents[0].notional_delta_abs <= Decimal("600")


# ── RiskBudget ─────────────────────────────────────────────────────


class TestRiskBudget:
    def test_equal_budget(self):
        rb = RiskBudget.equal(("A", "B", "C"))
        assert abs(rb.total - 1.0) < 1e-12
        assert rb.is_valid()
        assert abs(rb.target_risk_contribution("A") - 1 / 3) < 1e-12

    def test_validation_negative(self):
        rb = RiskBudget(budgets={"A": 0.5, "B": -0.5})
        assert rb.is_valid() is False

    def test_validation_sum(self):
        rb = RiskBudget(budgets={"A": 0.5, "B": 0.3})
        assert rb.is_valid() is False

    def test_empty(self):
        rb = RiskBudget.equal(())
        assert rb.budgets == {}

    def test_objective_evaluation(self):
        budget = RiskBudget.equal(("A", "B"))

        @dataclass
        class FakeInput:
            covariance: dict

        cov = {
            "A": {"A": 0.04, "B": 0.0},
            "B": {"A": 0.0, "B": 0.04},
        }
        obj = RiskBudgetObjective(budget)
        # Equal weights + equal variance + zero corr => perfect match
        val = obj.evaluate({"A": 0.5, "B": 0.5}, FakeInput(covariance=cov))
        assert abs(val) < 1e-12

    def test_objective_nonzero_deviation(self):
        budget = RiskBudget.equal(("A", "B"))

        @dataclass
        class FakeInput:
            covariance: dict

        cov = {
            "A": {"A": 0.04, "B": 0.0},
            "B": {"A": 0.0, "B": 0.16},
        }
        obj = RiskBudgetObjective(budget)
        # Unequal variance => unequal risk contribution with equal weights
        val = obj.evaluate({"A": 0.5, "B": 0.5}, FakeInput(covariance=cov))
        assert val > 0
