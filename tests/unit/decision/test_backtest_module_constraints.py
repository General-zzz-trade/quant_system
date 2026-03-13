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
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        '{"deadzone": 0.5, "min_hold": 1, "max_hold": 10, "long_only": false, "lgbm_xgb_weight": 0.5, "zscore_warmup": 1}'
    )
    with (model_dir / "lgbm_v8.pkl").open("wb") as f:
        pickle.dump({"model": None, "features": ["atr_norm_14", "trend"]}, f)
    mod = MLSignalDecisionModule(symbol="BTCUSDT", model_dir=model_dir, **kwargs)
    mod._rust_bridge = None  # tests control z-scores directly via _zscore_buf.push
    return mod


@patch.object(MLSignalDecisionModule, "_predict", return_value=1.0)
def test_monthly_gate_blocks_entry_when_below_ma(_mock_predict, tmp_path: Path):
    mod = _build_module(tmp_path, monthly_gate=True, monthly_gate_window=3)
    zs = iter([0.0, 0.0, 1.0])
    mod._zscore_buf.push = lambda _: next(zs)

    assert list(mod.decide(_snapshot(close=100))) == []
    assert list(mod.decide(_snapshot(close=99))) == []
    events = list(mod.decide(_snapshot(close=98)))
    assert events == []


@patch.object(MLSignalDecisionModule, "_predict", return_value=1.0)
def test_monthly_gate_allows_entry_when_above_ma(_mock_predict, tmp_path: Path):
    mod = _build_module(tmp_path, monthly_gate=True, monthly_gate_window=3)
    zs = iter([0.0, 0.0, 1.0])
    mod._zscore_buf.push = lambda _: next(zs)

    assert list(mod.decide(_snapshot(close=100))) == []
    assert list(mod.decide(_snapshot(close=101))) == []
    events = list(mod.decide(_snapshot(close=102)))
    assert len(events) == 2


@patch.object(MLSignalDecisionModule, "_predict", side_effect=[1.0, 1.0, 0.1, 0.1])
def test_trend_follow_extends_position_until_trend_breaks(_mock_predict, tmp_path: Path):
    mod = _build_module(
        tmp_path,
        trend_follow=True,
        trend_indicator="trend",
        trend_threshold=0.0,
        min_hold=1,
    )
    zs = iter([1.0, 1.0, 0.1, 0.1])
    mod._zscore_buf.push = lambda _: next(zs)

    assert len(list(mod.decide(_snapshot(close=100, features={"trend": 0.5})))) == 2
    assert list(mod.decide(_snapshot(close=101, features={"trend": 0.5}, qty=10))) == []
    assert list(mod.decide(_snapshot(close=102, features={"trend": 0.4}, qty=10))) == []
    exit_events = list(mod.decide(_snapshot(close=103, features={"trend": -0.2}, qty=10)))
    assert len(exit_events) == 2


@patch.object(MLSignalDecisionModule, "_predict", return_value=1.0)
def test_trend_follow_does_not_block_entry(_mock_predict, tmp_path: Path):
    mod = _build_module(
        tmp_path,
        trend_follow=True,
        trend_indicator="trend",
        trend_threshold=0.0,
    )
    mod._zscore_buf.push = lambda _: 1.0

    events = list(mod.decide(_snapshot(close=100, features={"trend": -0.2})))
    assert len(events) == 2


@patch.object(MLSignalDecisionModule, "_predict", return_value=1.0)
def test_vol_target_reduces_qty_in_high_vol(_mock_predict, tmp_path: Path):
    low = _build_module(tmp_path / "low", vol_target=0.15, vol_feature="atr_norm_14")
    high = _build_module(tmp_path / "high", vol_target=0.15, vol_feature="atr_norm_14")
    low._zscore_buf.push = lambda _: 1.0
    high._zscore_buf.push = lambda _: 1.0

    low_events = list(low.decide(_snapshot(close=100, features={"atr_norm_14": 0.05})))
    high_events = list(high.decide(_snapshot(close=100, features={"atr_norm_14": 0.30})))

    assert len(low_events) == 2
    assert len(high_events) == 2
    low_qty = low_events[1].qty
    high_qty = high_events[1].qty
    assert low_qty > high_qty
