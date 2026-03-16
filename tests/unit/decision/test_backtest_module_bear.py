"""Tests for bear model support in MLSignalDecisionModule."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest.mock import patch

from decision.backtest_module import MLSignalDecisionModule
from state.position import PositionState


def _snapshot(*, close: float, features: dict[str, float] | None = None, qty: float = 0.0):
    market = SimpleNamespace(close=Decimal(str(close)), last_price=Decimal(str(close)))
    positions = {}
    if qty != 0:
        positions["BTCUSDT"] = PositionState(
            symbol="BTCUSDT",
            qty=Decimal(str(qty)),
            avg_price=Decimal(str(close)),
        )
    return SimpleNamespace(
        market=market,
        markets={"BTCUSDT": market},
        positions=positions,
        features=features or {},
        event_id="evt-1",
        timestamp=int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000),
    )


def _build_module(tmp_path: Path, **kwargs) -> MLSignalDecisionModule:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        '{"deadzone": 0.5, "min_hold": 1, "max_hold": 10, "long_only": false, "lgbm_xgb_weight": 0.5, "zscore_warmup": 1}'  # noqa: E501
    )
    with (model_dir / "lgbm_v8.pkl").open("wb") as f:
        pickle.dump({"model": None, "features": ["atr_norm_14", "trend"]}, f)
    mod = MLSignalDecisionModule(symbol="BTCUSDT", model_dir=model_dir, **kwargs)
    mod._rust_bridge = None  # tests control z-scores directly via _zscore_buf.push
    return mod


class _MockBearModel:
    """Mock bear model returning configurable predict results."""

    def __init__(self, side: str = "long", strength: float = 0.3):
        self._side = side
        self._strength = strength

    def predict(self, symbol, ts, features):
        return SimpleNamespace(side=self._side, strength=self._strength)


@patch.object(MLSignalDecisionModule, "_predict", return_value=1.0)
def test_bear_model_overrides_monthly_gate_exit(_mock_predict, tmp_path: Path):
    """Bear model returning side='long' should prevent flatten on monthly gate failure."""
    bear = _MockBearModel(side="long", strength=0.3)
    mod = _build_module(
        tmp_path,
        monthly_gate=True,
        monthly_gate_window=3,
        bear_model=bear,
    )
    mod._zscore_buf.push = lambda _: 1.0

    # Fill close_buf with high prices, then enter
    mod.decide(_snapshot(close=100))
    mod.decide(_snapshot(close=101))
    entry_events = list(mod.decide(_snapshot(close=102)))
    assert len(entry_events) == 2  # entry happened

    # Now price drops below MA so monthly gate fails
    # Position is open (qty=0.01), gate fails, but bear model says stay
    events = list(mod.decide(_snapshot(close=90, qty=Decimal("0.01"))))
    assert events == []  # no exit order emitted
    assert mod._position != 0  # position preserved


@patch.object(MLSignalDecisionModule, "_predict", return_value=1.0)
def test_bear_model_flat_when_short(_mock_predict, tmp_path: Path):
    """Bear model returning side='short' should result in flatten (score=0)."""
    bear = _MockBearModel(side="short", strength=0.3)
    mod = _build_module(
        tmp_path,
        monthly_gate=True,
        monthly_gate_window=3,
        bear_model=bear,
    )
    mod._zscore_buf.push = lambda _: 1.0

    # Fill close_buf and enter
    mod.decide(_snapshot(close=100))
    mod.decide(_snapshot(close=101))
    mod.decide(_snapshot(close=102))

    # Price drops, gate fails, bear says short → flatten
    events = list(mod.decide(_snapshot(close=90, qty=Decimal("0.01"))))
    assert len(events) == 2  # exit orders emitted
    assert mod._position == 0.0


@patch.object(MLSignalDecisionModule, "_predict", return_value=1.0)
def test_bear_model_threshold_mapping(_mock_predict, tmp_path: Path):
    """Bear thresholds [(0.7, 1.0), (0.6, 0.5)] with strength=0.25 → prob=0.75 → score=1.0."""
    bear = _MockBearModel(side="long", strength=0.25)
    mod = _build_module(
        tmp_path,
        monthly_gate=True,
        monthly_gate_window=3,
        bear_model=bear,
        bear_thresholds=[(0.7, 1.0), (0.6, 0.5)],
    )
    mod._zscore_buf.push = lambda _: 1.0

    # Fill close_buf and enter
    mod.decide(_snapshot(close=100))
    mod.decide(_snapshot(close=101))
    mod.decide(_snapshot(close=102))

    # Gate fails, bear returns strength=0.25 → prob=0.75 → exceeds 0.7 → score=1.0
    events = list(mod.decide(_snapshot(close=90, qty=Decimal("0.01"))))
    assert events == []  # no exit
    assert mod._position == 1.0  # position set to bear score


@patch.object(MLSignalDecisionModule, "_predict", return_value=1.0)
def test_no_bear_model_preserves_existing_behavior(_mock_predict, tmp_path: Path):
    """Without bear model, monthly gate failure should flatten as before."""
    mod = _build_module(
        tmp_path,
        monthly_gate=True,
        monthly_gate_window=3,
    )
    mod._zscore_buf.push = lambda _: 1.0

    # Fill close_buf and enter
    mod.decide(_snapshot(close=100))
    mod.decide(_snapshot(close=101))
    mod.decide(_snapshot(close=102))

    # Gate fails, no bear model → flatten
    events = list(mod.decide(_snapshot(close=90, qty=Decimal("0.01"))))
    assert len(events) == 2  # exit orders
    assert mod._position == 0.0
