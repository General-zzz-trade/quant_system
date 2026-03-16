# tests/unit/execution/test_incident_taxonomy.py
"""Tests for incident taxonomy — verify enum completeness and alert factory usage."""

from execution.observability.incidents import (
    IncidentCategory,
    IncidentState,
    RecommendedAction,
    timeout_to_alert,
    reconcile_report_to_alert,
    synthetic_fill_to_alert,
)


class TestIncidentCategoryCompleteness:
    def test_all_categories_are_strings(self):
        for cat in IncidentCategory:
            assert isinstance(cat.value, str)

    def test_expected_categories_exist(self):
        expected = {
            "execution_timeout", "execution_reconcile", "execution_rejection",
            "execution_fill", "execution_stream", "operator_control", "model_reload",
        }
        actual = {c.value for c in IncidentCategory}
        assert expected == actual


class TestIncidentState:
    def test_three_states(self):
        assert len(IncidentState) == 3
        assert set(IncidentState) == {
            IncidentState.NORMAL, IncidentState.DEGRADED, IncidentState.CRITICAL,
        }


class TestRecommendedAction:
    def test_four_actions(self):
        assert len(RecommendedAction) == 4
        assert set(RecommendedAction) == {
            RecommendedAction.NONE, RecommendedAction.REVIEW,
            RecommendedAction.REDUCE_ONLY, RecommendedAction.HALT,
        }


class TestAlertFactoriesUseCanonicalCategories:
    def test_timeout_uses_canonical_category(self):
        alert = timeout_to_alert(
            venue="binance", symbol="BTCUSDT", order_id="1", timeout_sec=30.0,
        )
        assert alert.meta.get("category") == IncidentCategory.EXECUTION_TIMEOUT.value

    def test_reconcile_uses_canonical_category(self):
        from types import SimpleNamespace
        report = SimpleNamespace(venue="binance", all_drifts=[], should_halt=False)
        alert = reconcile_report_to_alert(report)
        assert alert.meta.get("category") == IncidentCategory.EXECUTION_RECONCILE.value

    def test_synthetic_fill_uses_canonical_category(self):
        from types import SimpleNamespace
        fill = SimpleNamespace(venue="binance", symbol="BTCUSDT",
                               fill_id="f1", order_id="o1", qty="0.1", side="buy")
        alert = synthetic_fill_to_alert(fill)
        assert alert.meta.get("category") == IncidentCategory.EXECUTION_FILL.value
