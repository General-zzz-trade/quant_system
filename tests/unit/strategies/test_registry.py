"""Tests for strategies.registry.StrategyRegistry."""
from __future__ import annotations

import pytest

from strategies.base import Signal
from strategies.registry import StrategyRegistry


# ---------------------------------------------------------------------------
# Minimal strategy stub for testing
# ---------------------------------------------------------------------------

class _StubStrategy:
    name = "stub_alpha"
    version = "0.1"
    venue = "test"
    timeframe = "1h"

    def generate_signal(self, features):
        return Signal(direction=1, confidence=0.5)

    def validate_config(self):
        return True

    def describe(self):
        return "Stub strategy for testing"


class _StubStrategy2:
    name = "stub_beta"
    version = "0.2"
    venue = "test"
    timeframe = "5m"

    def generate_signal(self, features):
        return Signal(direction=0, confidence=0.0)

    def validate_config(self):
        return True

    def describe(self):
        return "Second stub"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStrategyRegistry:

    def test_register_and_get(self):
        reg = StrategyRegistry()
        s = _StubStrategy()
        reg.register(s)
        assert reg.get_strategy("stub_alpha") is s

    def test_get_missing_returns_none(self):
        reg = StrategyRegistry()
        assert reg.get_strategy("nonexistent") is None

    def test_duplicate_register_raises(self):
        reg = StrategyRegistry()
        reg.register(_StubStrategy())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_StubStrategy())

    def test_list_strategies(self):
        reg = StrategyRegistry()
        reg.register(_StubStrategy())
        reg.register(_StubStrategy2())
        listing = reg.list_strategies()
        assert len(listing) == 2
        names = {s["name"] for s in listing}
        assert names == {"stub_alpha", "stub_beta"}

    def test_list_strategies_empty(self):
        reg = StrategyRegistry()
        assert reg.list_strategies() == []

    def test_unregister(self):
        reg = StrategyRegistry()
        reg.register(_StubStrategy())
        assert reg.unregister("stub_alpha") is True
        assert reg.get_strategy("stub_alpha") is None

    def test_unregister_missing(self):
        reg = StrategyRegistry()
        assert reg.unregister("nothing") is False

    def test_len_and_contains(self):
        reg = StrategyRegistry()
        assert len(reg) == 0
        assert "stub_alpha" not in reg
        reg.register(_StubStrategy())
        assert len(reg) == 1
        assert "stub_alpha" in reg

    def test_list_includes_description(self):
        reg = StrategyRegistry()
        reg.register(_StubStrategy())
        info = reg.list_strategies()[0]
        assert info["description"] == "Stub strategy for testing"
        assert info["venue"] == "test"
        assert info["timeframe"] == "1h"

    def test_register_after_unregister(self):
        reg = StrategyRegistry()
        reg.register(_StubStrategy())
        reg.unregister("stub_alpha")
        # Re-register should work
        reg.register(_StubStrategy())
        assert reg.get_strategy("stub_alpha") is not None
