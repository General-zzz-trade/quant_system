"""Tests for module exports — verify all __init__.py provide clean APIs."""
from __future__ import annotations


class TestStateExports:
    def test_all_exported(self) -> None:
        from state import (
            StateProjector,
            StateSnapshot,
            PortfolioState,
            RiskState,
            RiskLimits,
            SnapshotDiff,
            compute_diff,
            SchemaVersion,
            check_compatibility,
        )
        assert StateSnapshot is not None
        assert SnapshotDiff is not None


class TestDecisionExports:
    def test_all_exported(self) -> None:
        from decision import (
            DecisionEngine,
            DecisionOutput,
            OrderSpec,
            SignalResult,
            Candidate,
            TargetPosition,
        )
        assert DecisionEngine is not None

    def test_rebalancing_exports(self) -> None:
        from decision.rebalancing import (
            AlwaysRebalance,
            BarCountSchedule,
            RebalanceSchedule,
            ThresholdRebalance,
            TimeIntervalSchedule,
        )
        assert AlwaysRebalance is not None

    def test_sizing_exports(self) -> None:
        from decision.sizing import (
            PositionSizer,
            FixedFractionSizer,
            VolatilityAdjustedSizer,
        )
        assert VolatilityAdjustedSizer is not None

    def test_risk_overlay_exports(self) -> None:
        from decision.risk_overlay import (
            RiskOverlay,
            AlwaysAllow,
            CompositeOverlay,
            BasicKillOverlay,
        )
        assert CompositeOverlay is not None


class TestMonitoringExports:
    def test_all_exported(self) -> None:
        from monitoring import (
            Counter,
            Gauge,
            Timer,
            MetricsRegistry,
            EventLogger,
            Alert,
            AlertSink,
            ConsoleAlertSink,
            Severity,
        )
        assert Alert is not None

    def test_alerts_package(self) -> None:
        from monitoring.alerts import (
            Alert,
            AlertSink,
            CompositeAlertSink,
            ConsoleAlertSink,
            DedupAlertSink,
            LogAlertSink,
            Severity,
            WebhookAlertSink,
        )
        assert WebhookAlertSink is not None


class TestRiskExports:
    def test_all_exported(self) -> None:
        from risk import (
            RiskAction,
            RiskAggregator,
            RiskDecision,
            RiskRule,
            KillSwitch,
            KillScope,
            KillMode,
            merge_decisions,
        )
        assert RiskAggregator is not None


class TestPortfolioExports:
    def test_all_exported(self) -> None:
        from portfolio import (
            TargetWeightAllocator,
            EqualWeightAllocator,
            VolTargetAllocator,
            Rebalancer,
            RebalancePlan,
        )
        assert Rebalancer is not None


class TestExecutionExports:
    def test_all_exported(self) -> None:
        from execution import (
            ExecutionBridge,
            CanonicalOrder,
            CanonicalFill,
        )
        assert ExecutionBridge is not None


class TestAlphaExports:
    def test_all_exported(self) -> None:
        from alpha import AlphaModel, Signal, AlphaRegistry
        assert Signal is not None


class TestRegimeExports:
    def test_all_exported(self) -> None:
        from regime import (
            RegimeDetector,
            RegimeLabel,
            TrendRegimeDetector,
            VolatilityRegimeDetector,
        )
        assert RegimeLabel is not None


class TestFeaturesExports:
    def test_all_exported(self) -> None:
        from features import Bar, FeatureStore, sma, ema, rsi, atr
        assert FeatureStore is not None
