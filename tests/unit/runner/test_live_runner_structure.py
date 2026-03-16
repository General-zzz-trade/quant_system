"""Structural tests for LiveRunner — verifies build() phase contract.

These tests lock the build phase order and method signatures so that
extraction to runner/builders/ can be verified mechanically.
"""
from __future__ import annotations

import inspect
import re

import pytest


class TestBuildPhaseStructure:
    """Verify build() delegates to _build_* methods in correct order."""

    def _get_source(self):
        from runner.live_runner import LiveRunner
        return inspect.getsource(LiveRunner.build)

    def test_build_has_12_phases(self):
        """build() must call exactly 12 _build_* phases."""
        src = self._get_source()
        phases = re.findall(r"cls\._build_(\w+)\(", src)
        assert len(phases) == 12, f"Expected 12 phases, got {len(phases)}: {phases}"

    def test_phase_order(self):
        """Phases must execute in the documented order."""
        src = self._get_source()
        phases = re.findall(r"cls\._build_(\w+)\(", src)
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
            "persistence_and_recovery",
            "shutdown",
        ]
        assert phases == expected_order, f"Phase order mismatch:\n  got:      {phases}\n  expected: {expected_order}"

    def test_all_build_methods_are_static(self):
        """All _build_* methods must be @staticmethod (no self dependency)."""
        from runner.live_runner import LiveRunner
        for name in dir(LiveRunner):
            if name.startswith("_build_"):
                method = getattr(LiveRunner, name)
                assert isinstance(inspect.getattr_static(LiveRunner, name), staticmethod), \
                    f"{name} must be @staticmethod for safe extraction"

    def test_build_method_count(self):
        """LiveRunner should have exactly 12 _build_* methods."""
        from runner.live_runner import LiveRunner
        build_methods = [n for n in dir(LiveRunner) if n.startswith("_build_")]
        assert len(build_methods) == 12, f"Expected 12, got {len(build_methods)}: {build_methods}"

    def test_builders_package_exists(self):
        """runner/builders/ package must exist for future extraction."""
        import runner.builders
        assert runner.builders.__doc__ is not None


class TestBuildPhaseExtraction:
    """Contract tests for the builder extraction target."""

    def test_build_methods_have_signatures(self):
        """Each _build_* method should have a callable signature."""
        from runner.live_runner import LiveRunner
        for name in dir(LiveRunner):
            if name.startswith("_build_"):
                method = getattr(LiveRunner, name)
                sig = inspect.signature(method)
                assert len(sig.parameters) > 0, f"{name} should accept parameters"
