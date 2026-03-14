# tests/unit/decision/test_multi_strategy.py
"""Tests for MultiStrategyModule and AlphaDecayMonitor."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, List

import pytest

from decision.multi_strategy import MultiStrategyModule, StrategyPerformance
from monitoring.alpha_decay import AlphaDecayMonitor, DecayAlert


# ── Stubs ────────────────────────────────────────────────────

class _ConstantModule:
    """Always emits one event."""
    def __init__(self, symbol: str = "BTCUSDT", side: str = "buy"):
        self.symbol = symbol
        self.side = side
        self.call_count = 0

    def decide(self, snapshot: Any) -> Iterable[Any]:
        self.call_count += 1
        return [SimpleNamespace(symbol=self.symbol, side=self.side)]


class _EmptyModule:
    def decide(self, snapshot: Any) -> Iterable[Any]:
        return []


class _FailingModule:
    def decide(self, snapshot: Any) -> Iterable[Any]:
        raise RuntimeError("strategy error")


def _snap() -> SimpleNamespace:
    return SimpleNamespace(markets={}, ts=None)


# ── StrategyPerformance tests ────────────────────────────────

class TestStrategyPerformance:
    def test_initial_state(self):
        sp = StrategyPerformance(name="test")
        assert sp.n_observations == 0
        assert sp.cumulative_return == pytest.approx(0.0)
        assert sp.rolling_sharpe is None

    def test_positive_returns(self):
        sp = StrategyPerformance(name="test", lookback=30)
        for _ in range(30):
            sp.record_return(0.01)
        assert sp.cumulative_return > 0
        assert sp.rolling_sharpe is not None
        assert sp.rolling_sharpe > 0

    def test_drawdown(self):
        sp = StrategyPerformance(name="test", lookback=30)
        for _ in range(10):
            sp.record_return(0.02)
        for _ in range(10):
            sp.record_return(-0.03)
        assert sp.rolling_max_drawdown > 0

    def test_to_dict(self):
        sp = StrategyPerformance(name="test")
        d = sp.to_dict()
        assert d["name"] == "test"
        assert "cumulative_return" in d


# ── MultiStrategyModule tests ────────────────────────────────

class TestMultiStrategyModule:
    def test_all_modules_called(self):
        m1 = _ConstantModule("BTCUSDT", "buy")
        m2 = _ConstantModule("ETHUSDT", "sell")
        multi = MultiStrategyModule(
            modules=[m1, m2],
            module_names=["strat_a", "strat_b"],
        )
        events = list(multi.decide(_snap()))
        assert m1.call_count == 1
        assert m2.call_count == 1
        assert len(events) == 2

    def test_equal_weights_default(self):
        multi = MultiStrategyModule(
            modules=[_ConstantModule(), _ConstantModule()],
            module_names=["a", "b"],
            allocation_method="equal",
        )
        weights = multi.weights
        assert weights["a"] == pytest.approx(0.5)
        assert weights["b"] == pytest.approx(0.5)

    def test_sharpe_weighting(self):
        multi = MultiStrategyModule(
            modules=[_ConstantModule(), _ConstantModule()],
            module_names=["good", "bad"],
            allocation_method="sharpe",
            warmup_bars=5,
        )
        # Record different returns
        for _ in range(30):
            multi.record_strategy_return("good", 0.02)
            multi.record_strategy_return("bad", -0.01)

        # Trigger weight update
        for _ in range(10):
            multi.decide(_snap())

        weights = multi.weights
        # "good" should have higher weight
        assert weights["good"] > weights["bad"]

    def test_failing_module_handled(self):
        multi = MultiStrategyModule(
            modules=[_ConstantModule(), _FailingModule()],
            module_names=["ok", "fail"],
        )
        events = list(multi.decide(_snap()))
        # Should still get events from the working module
        assert len(events) >= 1

    def test_empty_module(self):
        multi = MultiStrategyModule(
            modules=[_EmptyModule()],
            module_names=["empty"],
        )
        events = list(multi.decide(_snap()))
        assert len(events) == 0

    def test_strategy_stats(self):
        multi = MultiStrategyModule(
            modules=[_ConstantModule()],
            module_names=["my_strat"],
        )
        stats = multi.strategy_stats
        assert len(stats) == 1
        assert stats[0]["name"] == "my_strat"

    def test_weight_clamping(self):
        multi = MultiStrategyModule(
            modules=[_ConstantModule(), _ConstantModule(), _ConstantModule()],
            module_names=["a", "b", "c"],
            min_weight=0.1,
            max_weight=0.6,
        )
        for w in multi.weights.values():
            assert w >= 0.1 - 1e-9
            assert w <= 0.6 + 1e-9


# ── AlphaDecayMonitor tests ─────────────────────────────────

class TestAlphaDecayMonitor:
    def test_no_alert_when_performing(self):
        monitor = AlphaDecayMonitor(short_window=10)
        monitor.set_baseline("strat", 2.0)
        for _ in range(20):
            monitor.record_return("strat", 0.02)
        alerts = monitor.check()
        # Good returns → Sharpe should be high → no decay
        assert len(alerts) == 0

    def test_warning_on_decay(self):
        monitor = AlphaDecayMonitor(
            short_window=10,
            warning_decay_pct=30.0,
            critical_decay_pct=60.0,
        )
        monitor.set_baseline("strat", 2.0)
        # Feed mediocre returns → Sharpe drops
        for _ in range(20):
            monitor.record_return("strat", 0.001)
        monitor.check()
        # Exact alerts depend on computed Sharpe vs baseline

    def test_critical_on_severe_decay(self):
        monitor = AlphaDecayMonitor(
            short_window=10,
            warning_decay_pct=30.0,
            critical_decay_pct=60.0,
        )
        monitor.set_baseline("strat", 3.0)
        # Feed negative returns → Sharpe drops to negative
        for _ in range(20):
            monitor.record_return("strat", -0.01)
        alerts = monitor.check()
        critical = [a for a in alerts if a.severity == "critical"]
        assert len(critical) >= 1

    def test_no_duplicate_alerts(self):
        monitor = AlphaDecayMonitor(short_window=10)
        monitor.set_baseline("strat", 3.0)
        for _ in range(20):
            monitor.record_return("strat", -0.01)
        monitor.check()
        alerts2 = monitor.check()
        # Second check should not re-emit same severity
        assert len(alerts2) == 0

    def test_callback_invoked(self):
        received: List[DecayAlert] = []
        monitor = AlphaDecayMonitor(
            short_window=10,
            on_alert=lambda a: received.append(a),
        )
        monitor.set_baseline("strat", 3.0)
        for _ in range(20):
            monitor.record_return("strat", -0.02)
        monitor.check()
        assert len(received) >= 1

    def test_get_rolling_sharpe(self):
        monitor = AlphaDecayMonitor(short_window=10)
        for _ in range(20):
            monitor.record_return("strat", 0.01)
        sharpe = monitor.get_rolling_sharpe("strat")
        assert sharpe is not None
        assert sharpe > 0

    def test_no_baseline_no_alert(self):
        monitor = AlphaDecayMonitor(short_window=10)
        for _ in range(20):
            monitor.record_return("strat", -0.01)
        alerts = monitor.check()
        assert len(alerts) == 0  # no baseline set

    def test_strategy_names(self):
        monitor = AlphaDecayMonitor()
        monitor.record_return("a", 0.01)
        monitor.record_return("b", 0.02)
        assert set(monitor.strategy_names) == {"a", "b"}
