"""Structural tests for LiveRunner — verifies build() phase contract.

These tests lock the build phase order and method signatures so that
extraction to runner/builders/ can be verified mechanically.
"""
from __future__ import annotations

import inspect
import re

import pytest


class TestBuildPhaseStructure:
    """Verify build() delegates to builder functions in correct order."""

    def _get_source(self):
        from runner.live_runner import LiveRunner
        return inspect.getsource(LiveRunner.build)

    def test_build_has_12_phases(self):
        """build() must call exactly 12 phases (10 _build_* + 2 named builders)."""
        src = self._get_source()
        build_phases = re.findall(r"_build_(\w+)\(", src)
        named_builders = re.findall(r"(_persistence_builder|_shutdown_builder)\(", src)
        total = len(build_phases) + len(named_builders)
        assert total == 12, f"Expected 12 phases, got {total}: {build_phases} + {named_builders}"

    def test_phase_order(self):
        """Phases must execute in the documented order."""
        src = self._get_source()
        phases = re.findall(r"_build_(\w+)\(", src)
        expected_order = [
            "core_infra",
            "monitoring",
            "portfolio_and_correlation",
            "order_infra",
            "features_and_inference",
            "coordinator_and_pipeline",
            "execution",
            "decision",
            "market_data",
            "user_stream",
        ]
        # Phases 11 and 12 use _persistence_builder and _shutdown_builder
        # (different naming pattern), so only check the first 10 _build_* phases
        assert phases == expected_order, f"Phase order mismatch:\n  got:      {phases}\n  expected: {expected_order}"

    def test_persistence_and_shutdown_phases(self):
        """build() must delegate to persistence and shutdown builders."""
        src = self._get_source()
        assert "_persistence_builder(" in src, "Phase 11 must call _persistence_builder"
        assert "_shutdown_builder(" in src, "Phase 12 must call _shutdown_builder"

    def test_builder_modules_importable(self):
        """All phase builder modules must be importable."""
        from runner.builders.core_infra_builder import build_core_infra
        from runner.builders.monitoring_builder import build_monitoring
        from runner.builders.portfolio_builder import build_portfolio_and_correlation
        from runner.builders.order_infra_builder import build_order_infra
        from runner.builders.features_builder import build_features_and_inference
        from runner.builders.engine_builder import build_coordinator_and_pipeline
        from runner.builders.execution_builder import build_execution_phase
        from runner.builders.decision_builder import build_decision
        from runner.builders.market_data_builder import build_market_data
        from runner.builders.user_stream_builder import build_user_stream
        from runner.builders.persistence import build_persistence_and_recovery
        from runner.builders.shutdown import build_shutdown

    def test_builders_package_exists(self):
        """runner/builders/ package must exist for future extraction."""
        import runner.builders
        assert runner.builders.__doc__ is not None


class TestBuildPhaseExtraction:
    """Contract tests for the builder extraction target."""

    def test_builder_functions_have_signatures(self):
        """Each builder function should have a callable signature."""
        from runner.builders.core_infra_builder import build_core_infra
        from runner.builders.monitoring_builder import build_monitoring
        from runner.builders.portfolio_builder import build_portfolio_and_correlation
        from runner.builders.order_infra_builder import build_order_infra
        from runner.builders.features_builder import build_features_and_inference
        from runner.builders.engine_builder import build_coordinator_and_pipeline
        from runner.builders.execution_builder import build_execution_phase
        from runner.builders.decision_builder import build_decision
        from runner.builders.market_data_builder import build_market_data
        from runner.builders.user_stream_builder import build_user_stream

        builders = [
            build_core_infra, build_monitoring, build_portfolio_and_correlation,
            build_order_infra, build_features_and_inference,
            build_coordinator_and_pipeline, build_execution_phase,
            build_decision, build_market_data, build_user_stream,
        ]
        for fn in builders:
            sig = inspect.signature(fn)
            assert len(sig.parameters) > 0, f"{fn.__name__} should accept parameters"
