"""Tests for retrain scheduling and pipeline."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock


from research.retrain.pipeline import RetrainPipeline
from research.retrain.scheduler import RetrainConfig, RetrainTrigger


class TestRetrainTrigger:
    def test_first_call_always_triggers(self):
        trigger = RetrainTrigger(RetrainConfig(schedule="weekly"))
        assert trigger.should_retrain()

    def test_time_based_weekly_not_elapsed(self):
        trigger = RetrainTrigger(RetrainConfig(schedule="weekly"))
        trigger.record_retrain(sharpe=1.5)

        now = datetime.now(timezone.utc) + timedelta(days=3)
        assert not trigger.should_retrain(now=now)

    def test_time_based_weekly_elapsed(self):
        trigger = RetrainTrigger(RetrainConfig(schedule="weekly"))
        trigger.record_retrain(sharpe=1.5)

        now = datetime.now(timezone.utc) + timedelta(days=8)
        assert trigger.should_retrain(now=now)

    def test_time_based_daily(self):
        trigger = RetrainTrigger(RetrainConfig(schedule="daily"))
        trigger.record_retrain(sharpe=1.0)

        now = datetime.now(timezone.utc) + timedelta(hours=25)
        assert trigger.should_retrain(now=now)

    def test_degradation_trigger(self):
        config = RetrainConfig(schedule="on_degradation", degradation_threshold=0.5)
        trigger = RetrainTrigger(config)
        trigger.record_retrain(sharpe=2.0)

        assert not trigger.should_retrain(current_sharpe=1.8)
        assert trigger.should_retrain(current_sharpe=1.3)

    def test_degradation_on_time_schedule(self):
        config = RetrainConfig(schedule="weekly", degradation_threshold=0.5)
        trigger = RetrainTrigger(config)
        trigger.record_retrain(sharpe=2.0)

        now = datetime.now(timezone.utc) + timedelta(days=2)
        assert not trigger.should_retrain(current_sharpe=1.8, now=now)
        assert trigger.should_retrain(current_sharpe=1.0, now=now)

    def test_record_retrain_updates_state(self):
        trigger = RetrainTrigger(RetrainConfig())
        assert trigger.last_retrain is None
        assert trigger.last_sharpe is None

        trigger.record_retrain(sharpe=1.5)
        assert trigger.last_retrain is not None
        assert trigger.last_sharpe == 1.5

    def test_no_degradation_without_baseline(self):
        config = RetrainConfig(schedule="on_degradation", degradation_threshold=0.5)
        trigger = RetrainTrigger(config)
        assert not trigger.should_retrain(current_sharpe=0.5)


class TestRetrainPipeline:
    def test_promotes_first_model(self):
        registry = MagicMock()
        registry.get_production.return_value = None
        mv_mock = MagicMock()
        mv_mock.model_id = "model-1"
        registry.register.return_value = mv_mock

        pipeline = RetrainPipeline(
            train_fn=lambda params: {"sharpe": 1.5},
            registry=registry,
            model_name="test_model",
        )
        result = pipeline.run(params={"window": 20}, features=["sma_20"])

        assert result.promoted
        assert result.new_sharpe == 1.5
        assert result.old_sharpe is None
        registry.promote.assert_called_once_with("model-1")

    def test_promotes_better_model(self):
        registry = MagicMock()
        old_prod = MagicMock()
        old_prod.metrics = {"sharpe": 1.0}
        registry.get_production.return_value = old_prod
        mv_mock = MagicMock()
        mv_mock.model_id = "model-2"
        registry.register.return_value = mv_mock

        pipeline = RetrainPipeline(
            train_fn=lambda params: {"sharpe": 1.5},
            registry=registry,
            model_name="test_model",
        )
        result = pipeline.run(params={}, features=[])

        assert result.promoted
        assert result.new_sharpe == 1.5
        assert result.old_sharpe == 1.0
        registry.promote.assert_called_once()

    def test_does_not_promote_worse_model(self):
        registry = MagicMock()
        old_prod = MagicMock()
        old_prod.metrics = {"sharpe": 2.0}
        registry.get_production.return_value = old_prod
        mv_mock = MagicMock()
        mv_mock.model_id = "model-3"
        registry.register.return_value = mv_mock

        pipeline = RetrainPipeline(
            train_fn=lambda params: {"sharpe": 1.0},
            registry=registry,
            model_name="test_model",
        )
        result = pipeline.run(params={}, features=[])

        assert not result.promoted
        assert result.new_sharpe == 1.0
        assert result.old_sharpe == 2.0
        registry.promote.assert_not_called()

    def test_records_retrain_on_trigger(self):
        trigger = MagicMock()
        pipeline = RetrainPipeline(
            train_fn=lambda params: {"sharpe": 1.2},
            trigger=trigger,
            model_name="test",
        )
        pipeline.run(params={}, features=[])

        trigger.record_retrain.assert_called_once_with(1.2)

    def test_without_registry(self):
        pipeline = RetrainPipeline(
            train_fn=lambda params: {"sharpe": 1.0},
            model_name="no_reg",
        )
        result = pipeline.run(params={"x": 1}, features=["f1"])

        assert result.promoted
        assert result.model_id == ""
        assert result.new_sharpe == 1.0
