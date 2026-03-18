from __future__ import annotations

import json
import pickle
from pathlib import Path

from scripts import backtest_engine


def test_build_decision_module_extracts_live_like_constraints(tmp_path: Path):
    model_dir = tmp_path / "BTCUSDT_gate_v2"
    model_dir.mkdir(parents=True)

    with (model_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "deadzone": 0.7,
                "min_hold": 12,
                "max_hold": 96,
                "long_only": True,
                "trend_follow": True,
                "trend_indicator": "trend",
                "trend_threshold": 0.1,
                "monthly_gate": True,
                "monthly_gate_window": 120,
                "position_management": {
                    "vol_target": 0.2,
                    "vol_feature": "atr_norm_14",
                },
            },
            f,
        )
    with (model_dir / "lgbm_v8.pkl").open("wb") as f:
        pickle.dump({"model": None, "features": ["trend", "atr_norm_14"]}, f)

    old_model_dir = backtest_engine.MODEL_DIR
    try:
        backtest_engine.MODEL_DIR = tmp_path
        module = backtest_engine.build_decision_module("BTCUSDT", equity_share=10_000.0)
    finally:
        backtest_engine.MODEL_DIR = old_model_dir

    assert module is not None
    assert module._deadzone == 0.7
    assert module._min_hold == 12
    assert module._max_hold == 96
    assert module._long_only is True
    assert module._trend_follow is True
    assert module._trend_indicator == "trend"
    assert module._trend_threshold == 0.1
    assert module._monthly_gate is True
    assert module._monthly_gate_window == 120
    assert module._vol_target == 0.2
    assert module._vol_feature == "atr_norm_14"


def test_build_decision_module_accepts_config_referenced_primary_artifact(tmp_path: Path):
    model_dir = tmp_path / "BTCUSDT_gate_v2"
    model_dir.mkdir(parents=True)

    with (model_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "multi_horizon": False,
                "primary_horizon": 24,
                "horizon_models": [
                    {
                        "horizon": 24,
                        "lgbm": "lgbm_h24.pkl",
                        "features": ["trend", "atr_norm_14"],
                    }
                ],
            },
            f,
        )
    with (model_dir / "lgbm_h24.pkl").open("wb") as f:
        pickle.dump({"model": None, "features": ["trend", "atr_norm_14"]}, f)

    old_model_dir = backtest_engine.MODEL_DIR
    try:
        backtest_engine.MODEL_DIR = tmp_path
        module = backtest_engine.build_decision_module("BTCUSDT", equity_share=10_000.0)
    finally:
        backtest_engine.MODEL_DIR = old_model_dir

    assert module is not None
    assert module._features == ["trend", "atr_norm_14"]
