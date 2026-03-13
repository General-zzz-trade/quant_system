"""Tests for vol-scale timing fix (Step 2) and short model support (Step 3)."""
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
        '{"deadzone": 0.5, "min_hold": 1, "max_hold": 10, "long_only": false, "lgbm_xgb_weight": 0.5, "zscore_warmup": 1}'
    )
    with (model_dir / "lgbm_v8.pkl").open("wb") as f:
        pickle.dump({"model": None, "features": ["atr_norm_14", "trend"]}, f)
    mod = MLSignalDecisionModule(symbol="BTCUSDT", model_dir=model_dir, **kwargs)
    mod._rust_bridge = None
    return mod


# ── Step 2: Vol-scale timing fix tests ──


@patch.object(MLSignalDecisionModule, "_predict", return_value=1.0)
def test_vol_scale_zeros_marginal_signal(_mock_predict, tmp_path: Path):
    """z=0.55 with vol_target=0.01 and vol_val=1.0 → scaled z=0.0055 → below deadzone → no entry."""
    mod = _build_module(tmp_path, vol_target=0.01, vol_feature="atr_norm_14")
    # Push z=0.55 (above deadzone of 0.5 before vol-scale)
    mod._zscore_buf.push = lambda _: 0.55

    features = {"atr_norm_14": 1.0}  # vol_val = 1.0, so scale = 0.01/1.0 = 0.01
    events = list(mod.decide(_snapshot(close=50000, features=features)))
    assert events == []  # Marginal signal zeroed by vol-scale


@patch.object(MLSignalDecisionModule, "_predict", return_value=1.0)
def test_vol_scale_preserves_strong_signal(_mock_predict, tmp_path: Path):
    """z=2.0 with vol_target=0.5 and vol_val=0.3 → scaled z=2.0*min(0.5/0.3,1.0)=2.0 → entry."""
    mod = _build_module(tmp_path, vol_target=0.5, vol_feature="atr_norm_14")
    mod._zscore_buf.push = lambda _: 2.0

    features = {"atr_norm_14": 0.3}  # scale = min(0.5/0.3, 1.0) = 1.0 → z stays 2.0
    events = list(mod.decide(_snapshot(close=50000, features=features)))
    assert len(events) == 2  # entry orders


@patch.object(MLSignalDecisionModule, "_predict", return_value=1.0)
def test_vol_scale_at_signal_not_notional(_mock_predict, tmp_path: Path):
    """Verify vol-scale affects signal discretization, not just notional sizing."""
    # Without vol-scale fix, z=0.55 > deadzone=0.5 would enter regardless of vol
    mod = _build_module(tmp_path, vol_target=0.1, vol_feature="atr_norm_14")
    mod._zscore_buf.push = lambda _: 0.55

    features = {"atr_norm_14": 0.5}  # scale = min(0.1/0.5, 1.0) = 0.2 → z=0.11 < 0.5
    events = list(mod.decide(_snapshot(close=50000, features=features)))
    assert events == []  # No entry — vol-scaled z below deadzone


# ── Step 3: Short model tests ──


class _MockShortModel:
    def __init__(self, side="short", strength=0.7):
        self._side = side
        self._strength = strength

    def predict(self, symbol, ts, features):
        return SimpleNamespace(side=self._side, strength=self._strength)


@patch.object(MLSignalDecisionModule, "_predict", return_value=1.0)
def test_short_model_score_in_features(_mock_predict, tmp_path: Path):
    """Short model should compute ml_short_score."""
    short = _MockShortModel(side="short", strength=0.7)
    mod = _build_module(tmp_path, short_model=short)
    mod._zscore_buf.push = lambda _: 1.0

    snap = _snapshot(close=50000)
    mod.decide(snap)
    assert hasattr(mod, "_last_short_score")
    assert mod._last_short_score != 0.0  # strength 0.7 > deadzone 0.5


@patch.object(MLSignalDecisionModule, "_predict", return_value=1.0)
def test_short_model_none_preserves_behavior(_mock_predict, tmp_path: Path):
    """Without short model, no short score should be computed."""
    mod = _build_module(tmp_path)
    mod._zscore_buf.push = lambda _: 1.0

    snap = _snapshot(close=50000)
    mod.decide(snap)
    assert not hasattr(mod, "_last_short_score")


@patch.object(MLSignalDecisionModule, "_predict", return_value=1.0)
def test_short_model_does_not_generate_orders(_mock_predict, tmp_path: Path):
    """Short model should only produce a score, not generate orders itself."""
    short = _MockShortModel(side="short", strength=0.7)
    mod = _build_module(tmp_path, short_model=short)
    mod._zscore_buf.push = lambda _: 1.0

    snap = _snapshot(close=50000)
    events = list(mod.decide(snap))
    # Should only have the main model's entry orders, not extra orders from short model
    assert len(events) == 2  # IntentEvent + OrderEvent from main signal


@patch.object(MLSignalDecisionModule, "_predict", return_value=1.0)
def test_short_model_custom_score_key(_mock_predict, tmp_path: Path):
    """Custom short_score_key should be used in the module."""
    short = _MockShortModel(side="short", strength=0.7)
    mod = _build_module(tmp_path, short_model=short, short_score_key="custom_short")
    mod._zscore_buf.push = lambda _: 1.0

    snap = _snapshot(close=50000)
    mod.decide(snap)
    assert mod._short_score_key == "custom_short"
    assert hasattr(mod, "_last_short_score")
