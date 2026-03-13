from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest.mock import patch

from alpha.base import Signal
from alpha.inference.bridge import LiveInferenceBridge
from decision.backtest_module import MLSignalDecisionModule
from state.position import PositionState


def _snapshot(*, close: float, features: dict[str, float] | None = None, qty: float = 0.0):
    market = SimpleNamespace(close=Decimal(str(close)), last_price=Decimal(str(close)))
    positions = {}
    if qty != 0.0:
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
        timestamp=int(datetime(2026, 1, 1).timestamp() * 1000),
    )


def _build_module(tmp_path: Path, **kwargs) -> MLSignalDecisionModule:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        '{"deadzone": 0.5, "min_hold": 1, "max_hold": 10, "long_only": false, '
        '"lgbm_xgb_weight": 0.5, "zscore_warmup": 1}'
    )
    with (model_dir / "lgbm_v8.pkl").open("wb") as f:
        pickle.dump({"model": None, "features": ["atr_norm_14", "trend", "close"]}, f)
    return MLSignalDecisionModule(symbol="BTCUSDT", model_dir=model_dir, **kwargs)


class _SequenceModel:
    def __init__(self, scores: list[float]) -> None:
        self.name = "seq"
        self._scores = scores
        self._idx = 0

    def predict(self, *, symbol: str, ts: datetime, features: dict[str, float]):
        score = self._scores[self._idx]
        self._idx += 1
        side = "long" if score > 0 else ("short" if score < 0 else "flat")
        return Signal(symbol=symbol, ts=ts, side=side, strength=abs(score))


def _run_live_scores(
    *,
    scores: list[float],
    features_per_bar: list[dict[str, float]],
    **bridge_kwargs,
) -> list[float]:
    bridge_kwargs.setdefault("min_hold_bars", {"BTCUSDT": 1})
    bridge_kwargs.setdefault("deadzone", 0.5)
    bridge_kwargs.setdefault("zscore_warmup", 0)
    bridge = LiveInferenceBridge(
        models=[_SequenceModel(scores)],
        **bridge_kwargs,
    )
    ts0 = datetime(2024, 1, 1)
    out: list[float] = []
    for i, feats in enumerate(features_per_bar):
        current = dict(feats)
        bridge.enrich("BTCUSDT", ts0 + timedelta(hours=i), current)
        out.append(current.get("ml_score", 0.0))
    return out


def _run_backtest_position_signs(
    mod: MLSignalDecisionModule,
    *,
    closes: list[float],
    features_per_bar: list[dict[str, float]],
    zscores: list[float],
) -> list[int]:
    qty = 0.0
    out: list[int] = []
    z_iter = iter(zscores)
    mod._zscore_buf.push = lambda _: next(z_iter)
    mod._rust_bridge = None  # disable Rust bridge when controlling z-scores directly

    for close, feats in zip(closes, features_per_bar):
        events = list(mod.decide(_snapshot(close=close, features=feats, qty=qty)))
        for order in events[1::2]:
            signed_qty = float(order.qty)
            if order.side == "buy":
                qty += signed_qty
            else:
                qty -= signed_qty
        out.append(1 if qty > 0 else (-1 if qty < 0 else 0))
    return out


def _run_backtest_qtys(
    mod: MLSignalDecisionModule,
    *,
    closes: list[float],
    features_per_bar: list[dict[str, float]],
    zscores: list[float],
) -> list[float]:
    qty = 0.0
    out: list[float] = []
    z_iter = iter(zscores)
    mod._zscore_buf.push = lambda _: next(z_iter)
    mod._rust_bridge = None  # disable Rust bridge when controlling z-scores directly

    for close, feats in zip(closes, features_per_bar):
        events = list(mod.decide(_snapshot(close=close, features=feats, qty=qty)))
        for order in events[1::2]:
            signed_qty = float(order.qty)
            if order.side == "buy":
                qty += signed_qty
            else:
                qty -= signed_qty
        out.append(qty)
    return out


def _entry_qty(
    mod: MLSignalDecisionModule,
    *,
    close: float,
    features: dict[str, float],
    zscore: float,
) -> float:
    mod._zscore_buf.push = lambda _: zscore
    mod._rust_bridge = None  # disable Rust bridge when controlling z-scores directly
    events = list(mod.decide(_snapshot(close=close, features=features, qty=0.0)))
    assert events
    return float(events[-1].qty)


@patch.object(MLSignalDecisionModule, "_predict", side_effect=[1.0, 1.0, 0.1, 0.1, 0.1])
def test_trend_hold_position_path_matches_live(_mock_predict, tmp_path: Path):
    features = [
        {"trend": 0.5},
        {"trend": 0.5},
        {"trend": 0.4},
        {"trend": 0.3},
        {"trend": -0.1},
    ]
    live_scores = _run_live_scores(
        scores=[0.8, 0.8, 0.1, 0.1, 0.1],
        features_per_bar=features,
        trend_follow=True,
        trend_indicator="trend",
        trend_threshold=0.0,
        max_hold=120,
    )

    mod = _build_module(
        tmp_path,
        trend_follow=True,
        trend_indicator="trend",
        trend_threshold=0.0,
        min_hold=1,
    )
    positions = _run_backtest_position_signs(
        mod,
        closes=[100, 101, 102, 103, 104],
        features_per_bar=features,
        zscores=[1.0, 1.0, 0.1, 0.1, 0.1],
    )

    assert [1 if s > 0 else (-1 if s < 0 else 0) for s in live_scores] == positions


@patch.object(MLSignalDecisionModule, "_predict", side_effect=[1.0, 1.0, 1.0, 1.0])
def test_monthly_gate_exit_matches_live_flattening(_mock_predict, tmp_path: Path):
    closes = [100.0, 99.0, 98.0, 97.0]
    features = [{"close": close} for close in closes]

    live_scores = _run_live_scores(
        scores=[0.8, 0.8, 0.8, 0.8],
        features_per_bar=features,
        monthly_gate=True,
        monthly_gate_window=3,
    )

    mod = _build_module(
        tmp_path,
        monthly_gate=True,
        monthly_gate_window=3,
        min_hold=1,
    )
    positions = _run_backtest_position_signs(
        mod,
        closes=closes,
        features_per_bar=features,
        zscores=[1.0, 1.0, 1.0, 1.0],
    )

    assert [1 if s > 0 else (-1 if s < 0 else 0) for s in live_scores] == positions


@patch.object(MLSignalDecisionModule, "_predict", side_effect=[1.0, -1.0, -1.0, -1.0])
def test_min_hold_flip_timing_matches_live(_mock_predict, tmp_path: Path):
    features = [{}, {}, {}, {}]
    live_scores = _run_live_scores(
        scores=[0.8, -0.8, -0.8, -0.8],
        features_per_bar=features,
        min_hold_bars={"BTCUSDT": 3},
    )

    mod = _build_module(tmp_path, min_hold=3)
    positions = _run_backtest_position_signs(
        mod,
        closes=[100, 101, 102, 103],
        features_per_bar=features,
        zscores=[1.0, -1.0, -1.0, -1.0],
    )

    assert [1 if s > 0 else (-1 if s < 0 else 0) for s in live_scores] == positions


@patch.object(MLSignalDecisionModule, "_predict", side_effect=[0.49, -0.49, 0.0])
def test_deadzone_blocks_entry_matches_live(_mock_predict, tmp_path: Path):
    features = [{}, {}, {}]
    live_scores = _run_live_scores(
        scores=[0.49, -0.49, 0.0],
        features_per_bar=features,
        deadzone=0.5,
        min_hold_bars={"BTCUSDT": 1},
    )

    mod = _build_module(tmp_path, deadzone=0.5, min_hold=1)
    positions = _run_backtest_position_signs(
        mod,
        closes=[100, 101, 102],
        features_per_bar=features,
        zscores=[0.49, -0.49, 0.0],
    )

    assert [1 if s > 0 else (-1 if s < 0 else 0) for s in live_scores] == positions
    assert positions == [0, 0, 0]


@patch.object(MLSignalDecisionModule, "_predict", side_effect=[1.0, 0.49, 0.49])
def test_deadzone_fade_exit_matches_live_without_trend_hold(_mock_predict, tmp_path: Path):
    features = [{}, {}, {}]
    live_scores = _run_live_scores(
        scores=[1.0, 0.49, 0.49],
        features_per_bar=features,
        deadzone=0.5,
        min_hold_bars={"BTCUSDT": 1},
    )

    mod = _build_module(tmp_path, deadzone=0.5, min_hold=1)
    positions = _run_backtest_position_signs(
        mod,
        closes=[100, 101, 102],
        features_per_bar=features,
        zscores=[1.0, 0.49, 0.49],
    )

    assert [1 if s > 0 else (-1 if s < 0 else 0) for s in live_scores] == positions
    assert positions == [1, 0, 0]


@patch.object(MLSignalDecisionModule, "_predict", side_effect=[1.0, 1.0])
def test_vol_target_sizing_ratio_matches_live_scale(_mock_predict, tmp_path: Path):
    low_vol_score = _run_live_scores(
        scores=[0.8, 0.8],
        features_per_bar=[{"atr_norm_14": 0.05}],
        vol_target=0.15,
        vol_feature="atr_norm_14",
    )[0]
    high_vol_score = _run_live_scores(
        scores=[0.8, 0.8],
        features_per_bar=[{"atr_norm_14": 0.30}],
        vol_target=0.15,
        vol_feature="atr_norm_14",
    )[0]

    low_mod = _build_module(tmp_path / "low", vol_target=0.15, vol_feature="atr_norm_14", min_hold=1)
    high_mod = _build_module(tmp_path / "high", vol_target=0.15, vol_feature="atr_norm_14", min_hold=1)
    low_qty = _entry_qty(low_mod, close=100.0, features={"atr_norm_14": 0.05}, zscore=1.0)
    high_qty = _entry_qty(high_mod, close=100.0, features={"atr_norm_14": 0.30}, zscore=1.0)

    live_ratio = high_vol_score / low_vol_score
    qty_ratio = high_qty / low_qty
    assert live_ratio == 0.5
    assert qty_ratio == 0.5
