"""Tests for strategy/execution_policy/ — ExecutionPolicy protocol and implementations."""
from __future__ import annotations

from decimal import Decimal


from strategy.execution_policy.base import ExecutionPolicy


# ===========================================================================
# ExecutionPolicy Protocol
# ===========================================================================

class TestExecutionPolicyProtocol:
    def test_protocol_defines_name_and_apply(self):
        """ExecutionPolicy requires 'name' and 'apply'."""
        assert hasattr(ExecutionPolicy, "apply")

    def test_conforming_class_has_required_methods(self):
        """A class with name + apply conforms to the Protocol shape."""
        class MyPolicy:
            name = "my_policy"
            def apply(self, snapshot, order):
                return order

        p = MyPolicy()
        assert hasattr(p, "name")
        assert callable(getattr(p, "apply", None))

    def test_non_conforming_class_missing_apply(self):
        """A class missing apply lacks the required method."""
        class BadPolicy:
            name = "bad"

        p = BadPolicy()
        assert not callable(getattr(p, "apply", None))


# ===========================================================================
# MarketableLimitPolicy (smoke — requires Rust import)
# ===========================================================================

class TestMarketableLimitPolicySmoke:
    def test_importable(self):
        """MarketableLimitPolicy can be imported."""
        from strategy.execution_policy.marketable_limit import MarketableLimitPolicy
        p = MarketableLimitPolicy()
        assert p.name == "marketable_limit"
        assert p.slippage_bps == Decimal("10")


class TestPassivePolicySmoke:
    def test_importable(self):
        """PassivePolicy can be imported."""
        from strategy.execution_policy.passive import PassivePolicy
        p = PassivePolicy()
        assert p.name == "passive"
        assert p.offset_bps == Decimal("5")
