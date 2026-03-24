"""Helper functions for LiveRunner.

Extracted from live_runner.py to keep file sizes manageable.
Contains: from_config logic, model reload, adaptive BTC check, attribution feedback.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def build_config_from_file(
    config_path: Path,
    *,
    shadow_mode: bool = False,
):
    """Parse a YAML/JSON config file and return a LiveRunnerConfig.

    Supports two config formats:
      - Flat: keys map 1:1 to LiveRunnerConfig fields (production.yaml)
      - Nested: trading.symbol, risk.*, execution.* sections (legacy)
    """
    import dataclasses
    from infra.config.loader import load_config_secure, resolve_credentials
    from runner.config import LiveRunnerConfig

    raw = load_config_secure(config_path)

    # Detect config format: flat vs nested
    is_flat = "symbols" in raw or "venue" in raw
    is_nested = "trading" in raw

    if is_flat:
        config_fields = {f.name for f in dataclasses.fields(LiveRunnerConfig)}
        kwargs: Dict[str, Any] = {}
        for k, v in raw.items():
            if k in config_fields:
                kwargs[k] = v
        if "symbols" in kwargs:
            kwargs["symbols"] = tuple(kwargs["symbols"])
        kwargs["shadow_mode"] = shadow_mode
        runner_config = LiveRunnerConfig(**kwargs)
    elif is_nested:
        runner_config = _build_nested_config(raw, config_path, shadow_mode)
    else:
        raise ValueError(
            f"Config format not recognized ({config_path}): "
            "expected flat keys (symbols, venue) or nested sections (trading, risk)"
        )

    resolve_credentials(raw)
    return runner_config, raw


def _build_nested_config(raw: dict, config_path: Path, shadow_mode: bool):
    """Build LiveRunnerConfig from legacy nested format."""
    from infra.config.schema import validate_trading_config
    from runner.config import LiveRunnerConfig

    errors = validate_trading_config(raw)
    if errors:
        raise ValueError(
            f"Config validation failed ({config_path}):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
    trading = raw.get("trading", {})
    risk = raw.get("risk", {})
    monitoring = raw.get("monitoring", {})
    log_cfg = raw.get("logging", {})

    symbol = trading.get("symbol", "BTCUSDT")
    symbols = tuple(trading["symbols"]) if "symbols" in trading else (symbol,)

    kwargs: Dict[str, Any] = {}
    if risk.get("max_position_notional") is not None:
        kwargs["max_position_notional"] = float(risk["max_position_notional"])
    if risk.get("max_order_notional") is not None:
        kwargs["max_order_notional"] = float(risk["max_order_notional"])
    if risk.get("max_leverage") is not None:
        leverage = float(risk["max_leverage"])
        kwargs["max_gross_leverage"] = leverage
        kwargs["max_net_leverage"] = leverage
    if risk.get("max_drawdown_pct") is not None:
        dd_kill = float(risk["max_drawdown_pct"])
        kwargs["dd_warning_pct"] = dd_kill * 0.5
        kwargs["dd_reduce_pct"] = dd_kill * 0.75
        kwargs["dd_kill_pct"] = dd_kill
    if monitoring.get("health_check_interval") is not None:
        kwargs["health_stale_data_sec"] = float(monitoring["health_check_interval"])
    if monitoring.get("health_port") is not None:
        kwargs["health_port"] = int(monitoring["health_port"])
    if monitoring.get("health_host") is not None:
        kwargs["health_host"] = str(monitoring["health_host"])
    if monitoring.get("health_auth_token_env") is not None:
        kwargs["health_auth_token_env"] = str(monitoring["health_auth_token_env"])

    return LiveRunnerConfig(
        symbols=symbols,
        venue=trading.get("exchange", "binance"),
        enable_structured_logging=log_cfg.get("structured", True),
        log_level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file"),
        shadow_mode=shadow_mode,
        testnet=bool(trading.get("testnet", False)),
        **kwargs,
    )


def auto_discover_models(
    runner_config,
    *,
    feature_computer=None,
    inference_bridges=None,
    metrics_exporter=None,
):
    """Auto-discover and load models from models_v8/ if not provided."""
    from runner.model_discovery import (
        discover_active_models,
        load_symbol_models,
        build_inference_bridge,
        build_feature_computer,
    )

    if feature_computer is None:
        feature_computer = build_feature_computer()

    if inference_bridges is None:
        active = discover_active_models()
        bridges: Dict[str, Any] = {}
        for sym in runner_config.symbols:
            if sym in active:
                info = active[sym]
                models, weights = load_symbol_models(
                    sym, info["dir"], info["config"],
                )
                if models:
                    bridges[sym] = build_inference_bridge(
                        sym, models, info["config"], runner_config,
                        metrics_exporter=metrics_exporter,
                        ensemble_weights=weights,
                    )
                else:
                    logger.warning("No models loaded for %s -- no inference bridge", sym)
            else:
                logger.warning("No active model found for %s in models_v8/", sym)
        inference_bridges = bridges if bridges else None

    return feature_computer, inference_bridges


def handle_model_reload(runner) -> None:
    """Handle SIGHUP model reload via ModelRegistry or direct file reload."""
    cfg = getattr(runner, '_config', None)

    # Path 1: ModelRegistry-based reload
    if runner.model_loader is not None:
        try:
            names = tuple(cfg.model_names) if cfg and cfg.model_names else ()
            new_models = runner.model_loader.reload_if_changed(names)
            if new_models is not None and runner.inference_bridge is not None:
                runner.inference_bridge.update_models(new_models)
                runner._record_model_reload(
                    outcome="reloaded",
                    model_names=names,
                    detail={"reloaded_count": len(new_models)},
                )
            elif new_models is None:
                runner._record_model_reload(
                    outcome="noop",
                    model_names=names,
                    detail={"reloaded_count": 0},
                )
        except Exception:
            names = tuple(cfg.model_names) if cfg and cfg.model_names else ()
            runner._record_model_reload(
                outcome="failed",
                model_names=names,
                detail=None,
                error="model_hot_reload_failed",
            )
            logger.exception("Model hot-reload failed")
        return

    # Path 2: Direct file reload (for auto_retrain.py SIGHUP)
    if runner.inference_bridge is not None:
        try:
            from alpha.models.lgbm_alpha import LGBMAlphaModel
            from pathlib import Path as _Path

            symbols = tuple(cfg.symbols) if cfg else ()
            models = []
            for sym in symbols:
                model_dir = _Path(f"models_v8/{sym}_gate_v2")
                if not model_dir.exists():
                    continue
                pkl_files = sorted(model_dir.glob("*.pkl"))
                for pkl in pkl_files:
                    m = LGBMAlphaModel(name=f"{sym}_{pkl.stem}")
                    m.load(pkl)
                    models.append(m)
            if models:
                bridge = runner.inference_bridge
                if isinstance(bridge, dict):
                    for sym in symbols:
                        b = bridge.get(sym)
                        if b is not None:
                            sym_models = [m for m in models if sym in m.name]
                            if sym_models:
                                b.update_models(sym_models)
                else:
                    bridge.update_models(models)
                runner._record_model_reload(
                    outcome="reloaded",
                    model_names=symbols,
                    detail={"reloaded_count": len(models), "source": "file_reload"},
                )
                logger.info("Direct model reload: %d model(s) from disk", len(models))
            else:
                runner._record_model_reload(
                    outcome="noop",
                    model_names=symbols,
                    detail={"reloaded_count": 0, "source": "file_reload"},
                )
        except Exception:
            symbols = tuple(cfg.symbols) if cfg else ()
            runner._record_model_reload(
                outcome="failed",
                model_names=symbols,
                detail=None,
                error="direct_file_reload_failed",
            )
            logger.exception("Direct model file reload failed")


def run_adaptive_btc_check(runner, selector: Any) -> None:
    """Run adaptive config selection for BTC and update inference bridge params."""
    try:
        import numpy as np
        import pandas as pd

        data_path = "data_files/BTCUSDT_1h.csv"
        df = pd.read_csv(data_path)
        if len(df) < 720:
            logger.warning("Adaptive BTC: insufficient data (%d rows)", len(df))
            return

        closes = df["close"].values.astype(np.float64)
        returns = np.diff(np.log(closes))
        window = 720
        if len(returns) < window:
            return
        rolling_mean = pd.Series(returns).rolling(window).mean().values
        rolling_std = pd.Series(returns).rolling(window).std().values
        z_scores = np.where(rolling_std > 1e-10, (returns - rolling_mean) / rolling_std, 0.0)

        result = selector.select_robust(z_scores, closes[1:])

        if result.confidence != "high":
            logger.info(
                "Adaptive BTC: confidence=%s (need 'high'), keeping fixed params. "
                "sharpe=%.2f trades=%d",
                result.confidence, result.sharpe, result.trades,
            )
            return

        bridge = runner.inference_bridge
        if bridge is None:
            return
        if isinstance(bridge, dict):
            bridge = bridge.get("BTCUSDT")
        if bridge is None or not hasattr(bridge, "update_params"):
            return

        bridge.update_params(
            "BTCUSDT",
            deadzone=result.deadzone,
            min_hold=result.min_hold,
            max_hold=result.max_hold,
            long_only=result.long_only,
        )
        logger.info(
            "Adaptive BTC applied: deadzone=%.1f min_hold=%d max_hold=%d "
            "long_only=%s sharpe=%.2f confidence=%s",
            result.deadzone, result.min_hold, result.max_hold,
            result.long_only, result.sharpe, result.confidence,
        )
    except Exception:
        logger.warning("Adaptive BTC check failed", exc_info=True)


def apply_attribution_feedback(runner) -> None:
    """Apply attribution-based weight adjustments to ensemble combiner (Direction 18)."""
    tracker = runner.live_signal_tracker
    if tracker is None or not hasattr(tracker, 'compute_weight_recommendations'):
        return

    recommendations = tracker.compute_weight_recommendations(
        alpha_health_monitor=runner.alpha_health_monitor,
    )

    combiner = runner.ensemble_combiner
    if combiner is None:
        return

    combiners = combiner.values() if isinstance(combiner, dict) else [combiner]
    for c in combiners:
        if hasattr(c, 'update_weight'):
            for origin, weight_mult in recommendations.items():
                if weight_mult < 1.0:
                    c.update_weight(origin, weight_mult)
                    logger.info(
                        "Attribution feedback: %s weight -> %.2f",
                        origin, weight_mult,
                    )
