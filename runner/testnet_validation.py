# runner/testnet_validation.py
"""3-phase testnet validation workflow: paper → shadow → live → longrun → compare.

Usage:
    python -m runner.testnet_validation --config testnet_binance.yaml --phase paper --duration 300
    python -m runner.testnet_validation --config testnet_binance.yaml --phase shadow --duration 300
    python -m runner.testnet_validation --config testnet_binance.yaml --phase live --duration 300
    python -m runner.testnet_validation --config testnet_binance.yaml --phase longrun
    python -m runner.testnet_validation --config testnet_binance.yaml --phase compare
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _load_symbol_models(
    model_dir: Path,
    symbol: str,
) -> Tuple[List[Any], Dict[str, Any], Dict[str, Any]]:
    """Load models + extract signal_kwargs + pos_mgmt from a single model dir.

    Returns (models, signal_kwargs, pos_mgmt).
    """
    from alpha.models.lgbm_alpha import LGBMAlphaModel
    from alpha.models.xgb_alpha import XGBAlphaModel

    models: List[Any] = []
    signal_kwargs: Dict[str, Any] = {}
    pos_mgmt: Dict[str, Any] = {}

    config_path = model_dir / "config.json"
    if not config_path.exists():
        return models, signal_kwargs, pos_mgmt

    with config_path.open() as f:
        mcfg = json.load(f)

    if mcfg.get("ensemble"):
        from alpha.models.ensemble import EnsembleAlphaModel

        sub_models: List[Any] = []
        weights = mcfg.get("ensemble_weights", [])
        for fname in mcfg.get("models", []):
            pkl_path = model_dir / fname
            if not pkl_path.exists():
                logger.warning("Ensemble member not found: %s", pkl_path)
                continue
            if "lgbm" in fname.lower():
                m = LGBMAlphaModel(name=fname)
                m.load(pkl_path)
            elif "xgb" in fname.lower():
                m = XGBAlphaModel(name=fname)
                m.load(pkl_path)
            else:
                m = LGBMAlphaModel(name=fname)
                m.load(pkl_path)
            sub_models.append(m)
            logger.info("Loaded ensemble member: %s [%s]", pkl_path, symbol)

        if sub_models:
            if len(weights) != len(sub_models):
                weights = [1.0 / len(sub_models)] * len(sub_models)
            ensemble = EnsembleAlphaModel(
                name=f"ensemble_{symbol}",
                sub_models=sub_models,
                weights=weights,
            )
            models.append(ensemble)
    else:
        for fname in mcfg.get("models", []):
            pkl_path = model_dir / fname
            if not pkl_path.exists():
                continue
            if "lgbm" in fname.lower():
                m = LGBMAlphaModel(name=fname)
                m.load(pkl_path)
            elif "xgb" in fname.lower():
                m = XGBAlphaModel(name=fname)
                m.load(pkl_path)
            else:
                m = LGBMAlphaModel(name=fname)
                m.load(pkl_path)
            models.append(m)
            logger.info("Loaded model: %s [%s]", pkl_path, symbol)

    # Signal constraints
    if mcfg.get("long_only"):
        signal_kwargs["long_only_symbols"] = {symbol}
    if "deadzone" in mcfg:
        signal_kwargs["deadzone"] = mcfg["deadzone"]
    if "min_hold" in mcfg:
        signal_kwargs["min_hold_bars"] = {symbol: mcfg["min_hold"]}
    if mcfg.get("monthly_gate", False):
        signal_kwargs["monthly_gate"] = True
        signal_kwargs["monthly_gate_window"] = mcfg.get("monthly_gate_window", 480)
    if mcfg.get("ensemble_weights"):
        signal_kwargs["ensemble_weights"] = mcfg["ensemble_weights"]

    # Position management
    pos_mgmt = mcfg.get("position_management", {})
    if pos_mgmt.get("bear_thresholds"):
        signal_kwargs["bear_thresholds"] = [tuple(x) for x in pos_mgmt["bear_thresholds"]]
    if pos_mgmt.get("vol_target") is not None:
        signal_kwargs["vol_target"] = pos_mgmt["vol_target"]
    if pos_mgmt.get("vol_feature"):
        signal_kwargs["vol_feature"] = pos_mgmt["vol_feature"]

    # Bear model
    bear_model_path = mcfg.get("bear_model_path")
    if bear_model_path:
        bear_dir = Path(bear_model_path)
        bear_cfg_path = bear_dir / "config.json"
        if bear_cfg_path.exists():
            with bear_cfg_path.open() as bf:
                bear_cfg = json.load(bf)
            bear_pkl = bear_dir / bear_cfg["models"][0]
            if bear_pkl.exists():
                bear_m = LGBMAlphaModel(name=f"bear_{symbol}")
                bear_m.load(bear_pkl)
                signal_kwargs["bear_model"] = bear_m
                logger.info("Loaded bear model: %s [%s]", bear_pkl, symbol)

    return models, signal_kwargs, pos_mgmt


def _try_build_unified_predictor(
    symbols: List[str],
    models_cfg: Dict[str, Any],
    strategy: Dict[str, Any],
) -> Optional[Any]:
    """Try to create a RustUnifiedPredictor for all symbols.

    Returns the predictor if all symbols have model JSON files, None otherwise.
    Falls back to legacy LiveInferenceBridge path if JSON files are missing.
    """
    try:
        from _quant_hotpath import RustUnifiedPredictor
    except ImportError:
        return None

    # Collect model JSON paths + configs per symbol
    all_model_jsons: Dict[str, List[str]] = {}
    all_configs: Dict[str, Dict[str, Any]] = {}

    for sym in symbols:
        sym_cfg = models_cfg.get(sym, {})
        model_path = sym_cfg.get("model_path")
        if not model_path:
            return None

        model_dir = Path(model_path)
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return None

        with config_path.open() as f:
            mcfg = json.load(f)

        # Find JSON model files for all models listed in config
        json_paths = []
        for fname in mcfg.get("models", []):
            json_name = fname.replace(".pkl", ".json")
            json_path = model_dir / json_name
            if not json_path.exists():
                logger.info("No JSON for %s in %s — falling back to legacy path", fname, model_dir)
                return None
            json_paths.append(str(json_path))

        if not json_paths:
            return None

        all_model_jsons[sym] = json_paths
        all_configs[sym] = mcfg

    # All symbols have JSON models — create unified predictor
    # Use the first symbol's models as the base (for shared model case)
    # For multi-symbol, each symbol may have different models
    # We need one predictor per symbol since models may differ
    # But the unified predictor already supports per-symbol engines internally
    # Limitation: currently one set of main models per predictor
    # Solution: create per-symbol predictors (same as per-symbol bridges)

    per_symbol_predictors: Dict[str, Any] = {}

    for sym in symbols:
        mcfg = all_configs[sym]
        json_paths = all_model_jsons[sym]
        sym_cfg = models_cfg.get(sym, {})
        model_dir = Path(sym_cfg["model_path"])

        # Ensemble weights
        weights = mcfg.get("ensemble_weights")
        if weights and len(weights) != len(json_paths):
            weights = None

        # Bear model
        bear_json = None
        bear_model_path = mcfg.get("bear_model_path")
        if bear_model_path:
            bear_dir = Path(bear_model_path)
            bear_cfg_path = bear_dir / "config.json"
            if bear_cfg_path.exists():
                with bear_cfg_path.open() as bf:
                    bear_cfg = json.load(bf)
                bear_fname = bear_cfg["models"][0].replace(".pkl", ".json")
                bear_json_path = bear_dir / bear_fname
                if bear_json_path.exists():
                    bear_json = str(bear_json_path)

        # Short model
        short_json = None
        short_dir = model_dir.parent / f"{sym}_short"
        short_cfg_path = short_dir / "config.json"
        if short_cfg_path.exists():
            with short_cfg_path.open() as sf:
                short_cfg = json.load(sf)
            for sfname in short_cfg.get("models", []):
                sjson = short_dir / sfname.replace(".pkl", ".json")
                if sjson.exists():
                    short_json = str(sjson)
                    break

        pred = RustUnifiedPredictor.create(
            json_paths,
            ensemble_weights=weights,
            bear_model_path=bear_json,
            short_model_path=short_json,
        )

        # Configure symbol constraints
        min_hold = mcfg.get("min_hold", 0)
        deadzone = mcfg.get("deadzone", 0.5)
        long_only = mcfg.get("long_only", False)
        monthly_gate = mcfg.get("monthly_gate", False)
        monthly_gate_window = mcfg.get("monthly_gate_window", 480)
        pos_mgmt = mcfg.get("position_management", {})
        bear_thresholds = None
        if pos_mgmt.get("bear_thresholds"):
            bear_thresholds = [tuple(x) for x in pos_mgmt["bear_thresholds"]]
        vol_target = pos_mgmt.get("vol_target")
        vol_feature = pos_mgmt.get("vol_feature", "atr_norm_14")

        pred.configure_symbol(
            sym,
            min_hold=min_hold,
            deadzone=deadzone,
            long_only=long_only,
            monthly_gate=monthly_gate,
            monthly_gate_window=monthly_gate_window,
            vol_target=vol_target,
            vol_feature=vol_feature,
            bear_thresholds=bear_thresholds,
        )

        per_symbol_predictors[sym] = pred
        logger.info(
            "Unified predictor for %s: %d models, bear=%s, short=%s",
            sym, len(json_paths), bear_json is not None, short_json is not None,
        )

    return per_symbol_predictors


def _try_build_tick_processors(
    symbols: List[str],
    models_cfg: Dict[str, Any],
    strategy: Dict[str, Any],
    currency: str = "USDT",
    balance: float = 10000.0,
) -> Optional[Dict[str, Any]]:
    """Try to create per-symbol RustTickProcessor objects.

    Returns dict of {symbol: RustTickProcessor} if all symbols have model JSON files.
    Falls back to None (legacy path) if JSON files are missing.
    """
    try:
        from _quant_hotpath import RustTickProcessor
    except ImportError:
        return None

    # Reuse the same model discovery logic as _try_build_unified_predictor
    all_model_jsons: Dict[str, List[str]] = {}
    all_configs: Dict[str, Dict[str, Any]] = {}

    for sym in symbols:
        sym_cfg = models_cfg.get(sym, {})
        model_path = sym_cfg.get("model_path")
        if not model_path:
            return None

        model_dir = Path(model_path)
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return None

        with config_path.open() as f:
            mcfg = json.load(f)

        json_paths = []
        for fname in mcfg.get("models", []):
            json_name = fname.replace(".pkl", ".json")
            json_path = model_dir / json_name
            if not json_path.exists():
                return None
            json_paths.append(str(json_path))

        if not json_paths:
            return None

        all_model_jsons[sym] = json_paths
        all_configs[sym] = mcfg

    per_symbol_tps: Dict[str, Any] = {}

    for sym in symbols:
        mcfg = all_configs[sym]
        json_paths = all_model_jsons[sym]
        sym_cfg = models_cfg.get(sym, {})
        model_dir = Path(sym_cfg["model_path"])

        weights = mcfg.get("ensemble_weights")
        if weights and len(weights) != len(json_paths):
            weights = None

        # Bear model
        bear_json = None
        bear_model_path = mcfg.get("bear_model_path")
        if bear_model_path:
            bear_dir = Path(bear_model_path)
            bear_cfg_path = bear_dir / "config.json"
            if bear_cfg_path.exists():
                with bear_cfg_path.open() as bf:
                    bear_cfg = json.load(bf)
                bear_fname = bear_cfg["models"][0].replace(".pkl", ".json")
                bear_json_path = bear_dir / bear_fname
                if bear_json_path.exists():
                    bear_json = str(bear_json_path)

        # Short model
        short_json = None
        short_dir = model_dir.parent / f"{sym}_short"
        short_cfg_path = short_dir / "config.json"
        if short_cfg_path.exists():
            with short_cfg_path.open() as sf:
                short_cfg = json.load(sf)
            for sfname in short_cfg.get("models", []):
                sjson = short_dir / sfname.replace(".pkl", ".json")
                if sjson.exists():
                    short_json = str(sjson)
                    break

        tp = RustTickProcessor.create(
            [sym],
            currency,
            balance,
            json_paths,
            ensemble_weights=weights,
            bear_model_path=bear_json,
            short_model_path=short_json,
        )

        # Configure symbol constraints
        min_hold = mcfg.get("min_hold", 0)
        deadzone = mcfg.get("deadzone", 0.5)
        long_only = mcfg.get("long_only", False)
        monthly_gate = mcfg.get("monthly_gate", False)
        monthly_gate_window = mcfg.get("monthly_gate_window", 480)
        pos_mgmt = mcfg.get("position_management", {})
        bear_thresholds = None
        if pos_mgmt.get("bear_thresholds"):
            bear_thresholds = [tuple(x) for x in pos_mgmt["bear_thresholds"]]
        vol_target = pos_mgmt.get("vol_target")
        vol_feature = pos_mgmt.get("vol_feature", "atr_norm_14")

        tp.configure_symbol(
            sym,
            min_hold=min_hold,
            deadzone=deadzone,
            long_only=long_only,
            monthly_gate=monthly_gate,
            monthly_gate_window=monthly_gate_window,
            vol_target=vol_target,
            vol_feature=vol_feature,
            bear_thresholds=bear_thresholds,
        )

        per_symbol_tps[sym] = tp
        logger.info(
            "Tick processor for %s: %d models, bear=%s, short=%s",
            sym, len(json_paths), bear_json is not None, short_json is not None,
        )

    return per_symbol_tps


def _build_ml_stack(
    raw: Dict[str, Any],
) -> Tuple[Optional[Any], List[Any], List[Any], Dict[str, Any]]:
    """Build ML pipeline (feature_computer, alpha_models, decision_modules, signal_kwargs) from config.

    Returns (None, [], [], {}) if no strategy.model_path configured or no models found.

    Supports two config layouts:
      1. Single-symbol: strategy.model_path (loads one model set)
      2. Multi-symbol: models.{SYMBOL}.model_path (loads per-symbol model sets)

    For multi-symbol, returns per-symbol inference bridges in signal_kwargs["per_symbol_bridges"].
    """
    strategy = raw.get("strategy", {})
    symbols = raw.get("trading", {}).get("symbols", ["BTCUSDT"])
    threshold = strategy.get("threshold", 0.002)
    threshold_short = strategy.get("threshold_short", 999.0)

    from features.enriched_computer import EnrichedFeatureComputer
    from decision.ml_decision import make_ml_decision

    # ── Multi-symbol layout: per-symbol model configs ─────
    models_cfg = raw.get("models", {})
    if models_cfg and isinstance(models_cfg, dict):
        # Try unified predictor path first (zero-copy Rust pipeline)
        unified_preds = _try_build_unified_predictor(symbols, models_cfg, strategy)

        all_dms: List[Any] = []
        any_loaded = False

        for sym in symbols:
            sym_cfg = models_cfg.get(sym, {})
            model_path = sym_cfg.get("model_path")
            if not model_path:
                logger.warning("No model_path for %s, skipping", sym)
                continue

            risk_pct = sym_cfg.get("risk_pct", strategy.get("risk_pct", 0.30))
            model_dir = Path(model_path)
            models, sig_kw, pos_mgmt = _load_symbol_models(model_dir, sym)
            if not models:
                logger.warning("No models loaded for %s", sym)
                continue

            any_loaded = True
            dd_limit = pos_mgmt.get("dd_limit") or 0.0
            dd_cooldown = pos_mgmt.get("dd_cooldown") or 48
            trailing_atr = pos_mgmt.get("trailing_atr") or 0.0
            atr_stop = pos_mgmt.get("atr_stop") or 0.0

            dm = make_ml_decision(
                symbol=sym,
                threshold=threshold,
                threshold_short=threshold_short,
                risk_pct=risk_pct,
                dd_limit=dd_limit,
                dd_cooldown=dd_cooldown,
                trailing_atr=trailing_atr,
                atr_stop=atr_stop,
            )
            all_dms.append(dm)

        if not any_loaded:
            logger.warning("No models loaded for any symbol — running without ML stack")
            return None, [], [], {}

        fc = EnrichedFeatureComputer()

        # Unified predictor path: skip LiveInferenceBridge entirely
        if unified_preds is not None:
            logger.info(
                "Multi-symbol unified predictor ready: %d symbols (zero-copy Rust pipeline)",
                len(unified_preds),
            )
            # Also try to build tick processors (full hot-path)
            balance = raw.get("trading", {}).get("starting_balance", 10000.0)
            currency = raw.get("trading", {}).get("currency", "USDT")
            tick_procs = _try_build_tick_processors(
                symbols, models_cfg, strategy, currency=currency, balance=balance,
            )
            result_kwargs: Dict[str, Any] = {"unified_predictors": unified_preds}
            if tick_procs is not None:
                result_kwargs["tick_processors"] = tick_procs
                logger.info("Tick processors ready: %d symbols (full Rust hot-path)", len(tick_procs))
            return fc, [], all_dms, result_kwargs

        # Legacy path: per-symbol LiveInferenceBridge
        from alpha.inference.bridge import LiveInferenceBridge
        per_symbol_bridges: Dict[str, Any] = {}
        for sym in symbols:
            sym_cfg = models_cfg.get(sym, {})
            model_path = sym_cfg.get("model_path")
            if not model_path:
                continue
            model_dir = Path(model_path)
            models, sig_kw, pos_mgmt = _load_symbol_models(model_dir, sym)
            if not models:
                continue
            bear_model = sig_kw.pop("bear_model", None)
            bridge = LiveInferenceBridge(
                models=models,
                bear_model=bear_model,
                **sig_kw,
            )
            per_symbol_bridges[sym] = bridge
            logger.info("ML stack for %s: models=%d, bridge ready (legacy path)", sym, len(models))

        logger.info(
            "Multi-symbol ML stack ready: %d symbols, %d decision modules",
            len(per_symbol_bridges), len(all_dms),
        )
        return fc, [], all_dms, {"per_symbol_bridges": per_symbol_bridges}

    # ── Single-symbol layout: strategy.model_path ─────────
    model_path = strategy.get("model_path")
    if not model_path:
        logger.info("No strategy.model_path — running without ML stack")
        return None, [], [], {}

    model_dir = Path(model_path)
    risk_pct = strategy.get("risk_pct", 0.30)
    models, signal_kwargs, pos_mgmt = _load_symbol_models(model_dir, symbols[0])

    # Legacy format fallback
    if not models:
        from alpha.models.lgbm_alpha import LGBMAlphaModel
        config_name = strategy.get("config_name", "mod_reg_1h")
        for sym in symbols:
            pkl = model_dir / sym / f"{config_name}.pkl"
            if pkl.exists():
                m = LGBMAlphaModel(name=f"{config_name}_{sym}")
                m.load(pkl)
                models.append(m)
                logger.info("Loaded model: %s", pkl)

    if not models:
        logger.warning("No models loaded — running without ML stack")
        return None, [], [], {}

    # Merge strategy-level overrides
    if strategy.get("monthly_gate", False) and "monthly_gate" not in signal_kwargs:
        signal_kwargs["monthly_gate"] = True
        signal_kwargs["monthly_gate_window"] = strategy.get("monthly_gate_window", 480)
    if strategy.get("trend_follow", False):
        signal_kwargs["trend_follow"] = True
        signal_kwargs["trend_indicator"] = strategy.get("trend_indicator", "tf4h_close_vs_ma20")
        signal_kwargs["trend_threshold"] = strategy.get("trend_threshold", 0.0)
        signal_kwargs["max_hold"] = strategy.get("max_hold", 120)
    strategy_pm = strategy.get("position_management", {})
    if "vol_target" not in signal_kwargs and strategy_pm.get("vol_target") is not None:
        signal_kwargs["vol_target"] = strategy_pm["vol_target"]
    if "vol_feature" not in signal_kwargs and strategy_pm.get("vol_feature"):
        signal_kwargs["vol_feature"] = strategy_pm["vol_feature"]
    bear_model_path = strategy.get("bear_model_path")
    if bear_model_path and "bear_model" not in signal_kwargs:
        from alpha.models.lgbm_alpha import LGBMAlphaModel
        bear_dir = Path(bear_model_path)
        bear_cfg_path = bear_dir / "config.json"
        if bear_cfg_path.exists():
            with bear_cfg_path.open() as bf:
                bear_cfg = json.load(bf)
            bear_pkl = bear_dir / bear_cfg["models"][0]
            if bear_pkl.exists():
                bear_m = LGBMAlphaModel(name="bear_detector")
                bear_m.load(bear_pkl)
                signal_kwargs["bear_model"] = bear_m

    # Expand per-symbol constraints for all symbols
    if "long_only_symbols" in signal_kwargs:
        signal_kwargs["long_only_symbols"] = set(symbols)
    if "min_hold_bars" in signal_kwargs:
        mh = list(signal_kwargs["min_hold_bars"].values())[0]
        signal_kwargs["min_hold_bars"] = {s: mh for s in symbols}

    fc = EnrichedFeatureComputer()
    dd_limit = pos_mgmt.get("dd_limit") or 0.0
    dd_cooldown = pos_mgmt.get("dd_cooldown") or 48
    trailing_atr = pos_mgmt.get("trailing_atr") or 0.0
    atr_stop = pos_mgmt.get("atr_stop") or 0.0

    dms = [
        make_ml_decision(
            symbol=sym,
            threshold=threshold,
            threshold_short=threshold_short,
            risk_pct=risk_pct,
            dd_limit=dd_limit,
            dd_cooldown=dd_cooldown,
            trailing_atr=trailing_atr,
            atr_stop=atr_stop,
        )
        for sym in symbols
    ]

    # Try unified predictor for single-symbol path
    # Build a synthetic models_cfg from strategy-level config
    _unified_models_cfg = {
        symbols[0]: {"model_path": model_path},
    }
    unified_preds = _try_build_unified_predictor(symbols, _unified_models_cfg, strategy)
    if unified_preds is not None:
        logger.info(
            "Single-symbol unified predictor ready: %s (zero-copy Rust pipeline)",
            symbols[0],
        )
        balance = strategy.get("starting_balance", 10000.0)
        currency = strategy.get("currency", "USDT")
        tick_procs = _try_build_tick_processors(
            symbols, _unified_models_cfg, strategy, currency=currency, balance=balance,
        )
        result_kwargs: Dict[str, Any] = {"unified_predictors": unified_preds}
        if tick_procs is not None:
            result_kwargs["tick_processors"] = tick_procs
            logger.info("Single-symbol tick processor ready: %s (full Rust hot-path)", symbols[0])
        return fc, [], dms, result_kwargs

    logger.info(
        "ML stack ready: %d models, %d decision modules, threshold=%.4f, signal_kwargs=%s",
        len(models), len(dms), threshold, signal_kwargs,
    )
    return fc, models, dms, signal_kwargs


def _ensure_testnet(raw: Dict[str, Any]) -> None:
    """Safety check: refuse to run validation against production."""
    testnet = raw.get("trading", {}).get("testnet", False)
    if not testnet:
        print("SAFETY: config must have trading.testnet: true for validation.")
        print("Refusing to run validation against production endpoints.")
        sys.exit(1)


def _output_dir(config_path: Path) -> Path:
    d = config_path.parent / "validation_output"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_equity_csv(path: Path, fills: List[Dict[str, Any]], starting_balance: float) -> None:
    """Write a minimal equity CSV from fill records."""
    equity = Decimal(str(starting_balance))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", "equity", "realized", "unrealized"])
        writer.writeheader()
        writer.writerow({
            "ts": datetime.now(timezone.utc).isoformat(),
            "equity": str(equity),
            "realized": "0",
            "unrealized": "0",
        })
        for fill in fills:
            writer.writerow({
                "ts": fill.get("ts", datetime.now(timezone.utc).isoformat()),
                "equity": str(equity),
                "realized": "0",
                "unrealized": "0",
            })


def _start_pollers(symbols: Any, testnet: bool = True):
    """Start all data pollers for one or more symbols.

    Args:
        symbols: single symbol string or sequence of symbols.
        testnet: whether to use testnet endpoints.

    Returns dict with keys:
        funding_sources: Dict[str, Callable] — per-symbol funding rate
        oi_sources: Dict[str, Callable] — per-symbol open interest
        fgi_source: Callable — global Fear & Greed
        iv_sources: Dict[str, Callable] — per-symbol implied vol
        pcr_sources: Dict[str, Callable] — per-symbol put/call ratio
        onchain_sources: Dict[str, Callable] — per-symbol on-chain
        liquidation_sources: Dict[str, Callable] — per-symbol liquidation
        mempool_source: Callable — global mempool
        macro_source: Callable — global macro
        sentiment_source: Callable — global sentiment
        all_pollers: List — all pollers for cleanup
    """
    from execution.adapters.binance.funding_poller import BinanceFundingPoller
    from execution.adapters.binance.oi_poller import BinanceOIPoller
    from execution.adapters.fgi_poller import FGIPoller
    from execution.adapters.deribit_iv_poller import DeribitIVPoller
    from execution.adapters.onchain_poller import OnchainPoller
    from execution.adapters.binance.liquidation_poller import BinanceLiquidationPoller
    from execution.adapters.mempool_poller import MempoolPoller
    from execution.adapters.macro_poller import MacroPoller
    from execution.adapters.sentiment_poller import SentimentPoller

    if isinstance(symbols, str):
        symbols = [symbols]

    all_pollers: List[Any] = []
    funding_sources: Dict[str, Any] = {}
    oi_sources: Dict[str, Any] = {}
    iv_sources: Dict[str, Any] = {}
    pcr_sources: Dict[str, Any] = {}
    onchain_sources: Dict[str, Any] = {}
    liquidation_sources: Dict[str, Any] = {}

    for sym in symbols:
        currency = sym.replace("USDT", "")
        asset = currency.lower()

        funding = BinanceFundingPoller(symbol=sym, testnet=testnet)
        oi = BinanceOIPoller(symbol=sym, testnet=testnet)
        deribit_iv = DeribitIVPoller(currency=currency)
        onchain = OnchainPoller(asset=asset)
        liquidation = BinanceLiquidationPoller(symbol=sym, testnet=testnet)

        funding.start()
        oi.start()
        deribit_iv.start()
        onchain.start()
        liquidation.start()

        funding_sources[sym] = funding.get_rate
        oi_sources[sym] = oi.get_oi
        iv_sources[sym] = lambda d=deribit_iv: d.get_current()[0]
        pcr_sources[sym] = lambda d=deribit_iv: d.get_current()[1]
        onchain_sources[sym] = onchain.get_current
        liquidation_sources[sym] = liquidation.get_current

        all_pollers.extend([funding, oi, deribit_iv, onchain, liquidation])

    # Global pollers (shared across symbols)
    fgi = FGIPoller()
    mempool = MempoolPoller()
    macro = MacroPoller()
    sentiment = SentimentPoller()
    fgi.start()
    mempool.start()
    macro.start()
    sentiment.start()
    all_pollers.extend([fgi, mempool, macro, sentiment])

    return {
        "funding_sources": funding_sources,
        "oi_sources": oi_sources,
        "fgi_source": fgi.get_value,
        "iv_sources": iv_sources,
        "pcr_sources": pcr_sources,
        "onchain_sources": onchain_sources,
        "liquidation_sources": liquidation_sources,
        "mempool_source": mempool.get_current,
        "macro_source": macro.get_current,
        "sentiment_source": sentiment.get_current,
        "all_pollers": all_pollers,
    }


def _stop_pollers(*pollers: Any) -> None:
    for p in pollers:
        try:
            p.stop()
        except Exception:
            pass


def run_paper(config_path: Path, duration: int) -> None:
    """Phase 1: Paper trading with testnet market data."""
    from infra.config.loader import load_config_secure
    from runner.live_paper_runner import LivePaperRunner, LivePaperConfig

    raw = load_config_secure(config_path)
    _ensure_testnet(raw)

    trading = raw.get("trading", {})
    symbol = trading.get("symbol", "BTCUSDT")
    symbols = tuple(trading["symbols"]) if "symbols" in trading else (symbol,)

    config = LivePaperConfig(
        symbols=symbols,
        starting_balance=10000.0,
        testnet=True,
    )

    fc, models, dms, signal_kwargs = _build_ml_stack(raw)
    pollers = _start_pollers(symbols, testnet=True)

    # Per-symbol inference bridges, unified predictors, or tick processors
    per_symbol_bridges = signal_kwargs.pop("per_symbol_bridges", None)
    unified_predictors = signal_kwargs.pop("unified_predictors", None)
    tick_processors = signal_kwargs.pop("tick_processors", None)

    # Cross-asset computer for altcoins (BTC-lead features)
    cross_asset = None
    btc_kline_poller = None
    has_altcoins = any(s != "BTCUSDT" for s in symbols)
    if has_altcoins:
        from features.cross_asset_computer import CrossAssetComputer
        from execution.adapters.binance.btc_kline_poller import BtcKlinePoller

        cross_asset = CrossAssetComputer()
        # Use per-symbol BTC funding if available, otherwise create dedicated poller
        btc_funding_fn = pollers["funding_sources"].get("BTCUSDT")
        if btc_funding_fn is None:
            from execution.adapters.binance.funding_poller import BinanceFundingPoller
            btc_funding = BinanceFundingPoller(symbol="BTCUSDT", testnet=True)
            btc_funding.start()
            pollers["all_pollers"].append(btc_funding)
            btc_funding_fn = btc_funding.get_rate

        btc_kline_poller = BtcKlinePoller(
            cross_asset, testnet=True,
            funding_source=btc_funding_fn,
        )
        btc_kline_poller.start()
        logger.info("Cross-asset enabled: BTC kline poller feeding CrossAssetComputer")

    output_dir = _output_dir(config_path)
    runner = LivePaperRunner.build(
        config,
        feature_computer=fc,
        alpha_models=models or None,
        decision_modules=dms or None,
        inference_bridges=per_symbol_bridges,
        unified_predictors=unified_predictors,
        tick_processors=tick_processors,
        funding_rate_source=pollers["funding_sources"],
        oi_source=pollers["oi_sources"],
        fgi_source=pollers["fgi_source"],
        implied_vol_source=pollers["iv_sources"],
        put_call_ratio_source=pollers["pcr_sources"],
        onchain_source=pollers["onchain_sources"],
        liquidation_source=pollers["liquidation_sources"],
        mempool_source=pollers["mempool_source"],
        macro_source=pollers["macro_source"],
        sentiment_source=pollers["sentiment_source"],
        cross_asset_computer=cross_asset,
        checkpoint_dir=output_dir,
        **signal_kwargs,
    )

    # Wire Telegram alerts if configured
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if tg_token and tg_chat:
        from monitoring.alerts.telegram import TelegramAlertSink
        from monitoring.alerts.base import CompositeAlertSink, DedupAlertSink
        from monitoring.alerts.console import ConsoleAlertSink
        from monitoring.health import SystemHealthMonitor

        tg_sink = DedupAlertSink(
            delegate=CompositeAlertSink(sinks=[
                ConsoleAlertSink(),
                TelegramAlertSink(tg_token, tg_chat),
            ]),
        )
        health_monitor = SystemHealthMonitor(sink=tg_sink)
        health_monitor.start()
        logger.info("Telegram alerts enabled (chat_id=%s)", tg_chat)

    def _timeout(*_: Any) -> None:
        logger.info("Paper phase duration reached (%ds), stopping...", duration)
        runner.stop()

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(duration)

    logger.info("Starting PAPER phase for %ds with %d symbols...", duration, len(symbols))
    try:
        runner.start()
    except KeyboardInterrupt:
        runner.stop()
    finally:
        _stop_pollers(*pollers["all_pollers"])
        if btc_kline_poller is not None:
            btc_kline_poller.stop()

    out = _output_dir(config_path)
    _write_equity_csv(out / "paper_equity.csv", runner.fills, 10000.0)
    logger.info("Paper phase complete. Fills: %d. Output: %s", len(runner.fills), out)


def run_shadow(config_path: Path, duration: int) -> None:
    """Phase 2: Shadow mode — signals recorded, no execution."""
    from infra.config.loader import load_config_secure
    from runner.live_runner import LiveRunner, LiveRunnerConfig

    raw = load_config_secure(config_path)
    _ensure_testnet(raw)
    # Shadow phase records signals only — no API keys needed

    trading = raw.get("trading", {})
    symbol = trading.get("symbol", "BTCUSDT")
    symbols = tuple(trading["symbols"]) if "symbols" in trading else (symbol,)

    # Shadow mode needs a venue client that won't be called
    class _NoOpClient:
        def send_order(self, order_event: Any) -> list:
            return []

    fc, models, dms, signal_kwargs = _build_ml_stack(raw)
    per_symbol_bridges = signal_kwargs.pop("per_symbol_bridges", None)
    unified_predictors = signal_kwargs.pop("unified_predictors", None)
    tick_processors = signal_kwargs.pop("tick_processors", None)
    bear_model = signal_kwargs.pop("bear_model", None)
    config = LiveRunnerConfig(
        symbols=symbols,
        testnet=True,
        shadow_mode=True,
        enable_preflight=False,
        enable_persistent_stores=False,
        **signal_kwargs,
    )

    pollers = _start_pollers(symbols, testnet=True)

    runner = LiveRunner.build(
        config,
        venue_clients={"binance": _NoOpClient()},
        feature_computer=fc,
        alpha_models=models or None,
        decision_modules=dms or None,
        bear_model=bear_model,
        inference_bridges=per_symbol_bridges,
        unified_predictors=unified_predictors,
        tick_processors=tick_processors,
        funding_rate_source=pollers["funding_sources"],
        oi_source=pollers["oi_sources"],
        fgi_source=pollers["fgi_source"],
        implied_vol_source=pollers["iv_sources"],
        put_call_ratio_source=pollers["pcr_sources"],
        onchain_source=pollers["onchain_sources"],
        liquidation_source=pollers["liquidation_sources"],
        mempool_source=pollers["mempool_source"],
        macro_source=pollers["macro_source"],
        sentiment_source=pollers["sentiment_source"],
    )

    def _timeout(*_: Any) -> None:
        logger.info("Shadow phase duration reached (%ds), stopping...", duration)
        runner.stop()

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(duration)

    logger.info("Starting SHADOW phase for %ds with testnet data...", duration)
    try:
        runner.start()
    except KeyboardInterrupt:
        runner.stop()
    finally:
        _stop_pollers(*pollers["all_pollers"])

    out = _output_dir(config_path)
    with (out / "shadow_events.json").open("w") as f:
        json.dump({"fills": runner.fills, "event_index": runner.event_index}, f, indent=2)
    logger.info("Shadow phase complete. Events: %d. Output: %s", runner.event_index, out)


def run_live(config_path: Path, duration: int) -> None:
    """Phase 3: Live testnet trading — real orders on testnet."""
    from infra.config.loader import load_config_secure, resolve_credentials
    from execution.adapters.binance.rest import BinanceRestClient, BinanceRestConfig
    from execution.adapters.binance.urls import resolve_binance_urls
    from runner.live_runner import LiveRunner, LiveRunnerConfig
    raw = load_config_secure(config_path)
    _ensure_testnet(raw)
    resolve_credentials(raw)

    trading = raw.get("trading", {})
    symbol = trading.get("symbol", "BTCUSDT")
    symbols = tuple(trading["symbols"]) if "symbols" in trading else (symbol,)

    creds = raw.get("credentials", {})
    key_env = creds.get("api_key_env") or "BINANCE_TESTNET_API_KEY"
    secret_env = creds.get("api_secret_env") or "BINANCE_TESTNET_API_SECRET"
    api_key = os.environ.get(key_env, "")
    api_secret = os.environ.get(secret_env, "")

    if not api_key or not api_secret:
        print(f"Missing testnet API credentials.")
        print(f"  1. Register at https://testnet.binancefuture.com/")
        print(f"  2. Generate API key/secret")
        print(f"  3. Export env vars:")
        print(f"     export {key_env}=<your_api_key>")
        print(f"     export {secret_env}=<your_api_secret>")
        sys.exit(1)

    urls = resolve_binance_urls(testnet=True)
    client = BinanceRestClient(
        cfg=BinanceRestConfig(
            base_url=urls.rest_base,
            api_key=api_key,
            api_secret=api_secret,
        )
    )

    fc, models, dms, signal_kwargs = _build_ml_stack(raw)
    per_symbol_bridges = signal_kwargs.pop("per_symbol_bridges", None)
    unified_predictors = signal_kwargs.pop("unified_predictors", None)
    tick_processors = signal_kwargs.pop("tick_processors", None)
    bear_model = signal_kwargs.pop("bear_model", None)
    config = LiveRunnerConfig(
        symbols=symbols,
        testnet=True,
        enable_persistent_stores=False,
        **signal_kwargs,
    )

    pollers = _start_pollers(symbols, testnet=True)

    runner = LiveRunner.build(
        config,
        venue_clients={"binance": client},
        feature_computer=fc,
        alpha_models=models or None,
        decision_modules=dms or None,
        bear_model=bear_model,
        inference_bridges=per_symbol_bridges,
        unified_predictors=unified_predictors,
        tick_processors=tick_processors,
        funding_rate_source=pollers["funding_sources"],
        oi_source=pollers["oi_sources"],
        fgi_source=pollers["fgi_source"],
        implied_vol_source=pollers["iv_sources"],
        put_call_ratio_source=pollers["pcr_sources"],
        onchain_source=pollers["onchain_sources"],
        liquidation_source=pollers["liquidation_sources"],
        mempool_source=pollers["mempool_source"],
        macro_source=pollers["macro_source"],
        sentiment_source=pollers["sentiment_source"],
    )

    if runner.user_stream is not None:
        us_url = getattr(runner.user_stream.cfg, "ws_base_url", "unknown")
        logger.info("User stream wired: base_url=%s", us_url)
    else:
        logger.info("User stream not wired (shadow or non-Binance)")

    def _timeout(*_: Any) -> None:
        logger.info("Live testnet phase duration reached (%ds), stopping...", duration)
        runner.stop()

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(duration)

    logger.info("Starting LIVE TESTNET phase for %ds...", duration)
    try:
        runner.start()
    except KeyboardInterrupt:
        runner.stop()
    finally:
        _stop_pollers(*pollers["all_pollers"])

    out = _output_dir(config_path)
    _write_equity_csv(out / "live_equity.csv", runner.fills, 10000.0)
    logger.info("Live testnet phase complete. Fills: %d. Output: %s", len(runner.fills), out)


def _start_status_logger(
    runner: Any,
    ws_transport: Any,
    stop_event: threading.Event,
    interval: float = 60.0,
) -> threading.Thread:
    """Daemon thread that prints periodic status lines for longrun mode."""
    start_time = time.monotonic()

    def _loop() -> None:
        while not stop_event.wait(timeout=interval):
            uptime = int(time.monotonic() - start_time)
            events = runner.event_index
            n_fills = len(runner.fills)

            # Feature completeness from coordinator state
            view = runner.coordinator.get_state_view()
            features = view.get("features", {})
            total = len(features)
            valid = sum(1 for v in features.values() if v is not None and not (isinstance(v, float) and math.isnan(v)))

            ws_state = ws_transport.state.value if hasattr(ws_transport, "state") else "unknown"

            logger.info(
                "LONGRUN STATUS | uptime=%ds events=%d fills=%d features=%d/%d ws=%s",
                uptime, events, n_fills, valid, total, ws_state,
            )

    t = threading.Thread(target=_loop, name="longrun-status", daemon=True)
    t.start()
    return t


def run_longrun(config_path: Path, duration: int) -> None:
    """Long-running testnet mode with WS reconnection and state persistence.

    Runs indefinitely (ignores duration). Stop with SIGTERM/SIGINT (Ctrl+C).
    Uses ReconnectingWsTransport for WS resilience and SQLite state checkpointing.
    """
    from infra.config.loader import load_config_secure, resolve_credentials
    from execution.adapters.binance.rest import BinanceRestClient, BinanceRestConfig
    from execution.adapters.binance.urls import resolve_binance_urls
    from execution.adapters.binance.transport_factory import create_ws_transport
    from execution.adapters.binance.reconnecting_ws_transport import ReconnectingWsTransport
    from runner.live_runner import LiveRunner, LiveRunnerConfig

    raw = load_config_secure(config_path)
    _ensure_testnet(raw)
    resolve_credentials(raw)

    trading = raw.get("trading", {})
    symbol = trading.get("symbol", "BTCUSDT")
    symbols = tuple(trading["symbols"]) if "symbols" in trading else (symbol,)

    creds = raw.get("credentials", {})
    key_env = creds.get("api_key_env") or "BINANCE_TESTNET_API_KEY"
    secret_env = creds.get("api_secret_env") or "BINANCE_TESTNET_API_SECRET"
    api_key = os.environ.get(key_env, "")
    api_secret = os.environ.get(secret_env, "")

    if not api_key or not api_secret:
        print(f"Missing testnet API credentials.")
        print(f"  1. Register at https://testnet.binancefuture.com/")
        print(f"  2. Generate API key/secret")
        print(f"  3. Export env vars:")
        print(f"     export {key_env}=<your_api_key>")
        print(f"     export {secret_env}=<your_api_secret>")
        sys.exit(1)

    urls = resolve_binance_urls(testnet=True)
    client = BinanceRestClient(
        cfg=BinanceRestConfig(
            base_url=urls.rest_base,
            api_key=api_key,
            api_secret=api_secret,
        )
    )

    fc, models, dms, signal_kwargs = _build_ml_stack(raw)
    per_symbol_bridges = signal_kwargs.pop("per_symbol_bridges", None)
    unified_predictors = signal_kwargs.pop("unified_predictors", None)
    tick_processors = signal_kwargs.pop("tick_processors", None)
    bear_model = signal_kwargs.pop("bear_model", None)

    output_dir = _output_dir(config_path)
    config = LiveRunnerConfig(
        symbols=symbols,
        testnet=True,
        enable_persistent_stores=True,
        data_dir=str(output_dir),
        **signal_kwargs,
    )

    pollers = _start_pollers(symbols, testnet=True)

    ws_transport = ReconnectingWsTransport(
        inner=create_ws_transport(),
        max_retries=20,
        max_delay_s=120.0,
    )

    runner = LiveRunner.build(
        config,
        venue_clients={"binance": client},
        transport=ws_transport,
        feature_computer=fc,
        alpha_models=models or None,
        decision_modules=dms or None,
        bear_model=bear_model,
        inference_bridges=per_symbol_bridges,
        unified_predictors=unified_predictors,
        tick_processors=tick_processors,
        funding_rate_source=pollers["funding_sources"],
        oi_source=pollers["oi_sources"],
        fgi_source=pollers["fgi_source"],
        implied_vol_source=pollers["iv_sources"],
        put_call_ratio_source=pollers["pcr_sources"],
        onchain_source=pollers["onchain_sources"],
        liquidation_source=pollers["liquidation_sources"],
        mempool_source=pollers["mempool_source"],
        macro_source=pollers["macro_source"],
        sentiment_source=pollers["sentiment_source"],
    )

    status_stop = threading.Event()
    _start_status_logger(runner, ws_transport, status_stop)

    # Bridge/tick_processor checkpoint: restore on startup, periodic save
    checkpoint_path = output_dir / "bridge_checkpoint.json"
    bridge = getattr(runner, "inference_bridge", None)

    # Tick processors have their own checkpoint/restore (signal state: z-score, hold_counter)
    _ckpt_tps = tick_processors

    def _ckpt_save() -> None:
        try:
            if _ckpt_tps is not None:
                data = {sym: tp.checkpoint() for sym, tp in _ckpt_tps.items()}
            elif bridge is not None:
                data = bridge.checkpoint()
            else:
                return
            with checkpoint_path.open("w") as cf:
                json.dump(data, cf)
        except Exception:
            logger.warning("Checkpoint save failed")

    def _ckpt_restore() -> None:
        if not checkpoint_path.exists():
            return
        try:
            with checkpoint_path.open() as cf:
                data = json.load(cf)
            if _ckpt_tps is not None:
                for sym, tp in _ckpt_tps.items():
                    sym_data = data.get(sym)
                    if sym_data:
                        tp.restore(sym_data)
            elif bridge is not None:
                bridge.restore(data)
            logger.info("Restored checkpoint from %s", checkpoint_path)
        except Exception:
            logger.warning("Failed to restore checkpoint, starting fresh")

    _ckpt_restore()

    def _checkpoint_loop(stop_ev: threading.Event) -> None:
        while not stop_ev.wait(timeout=60.0):
            _ckpt_save()

    ckpt_thread = threading.Thread(target=_checkpoint_loop, args=(status_stop,), daemon=True)
    ckpt_thread.start()

    logger.info("Starting LONGRUN mode (Ctrl+C or SIGTERM to stop)...")
    try:
        runner.start()
    except KeyboardInterrupt:
        runner.stop()
    finally:
        status_stop.set()
        _stop_pollers(*pollers["all_pollers"])

    _write_equity_csv(output_dir / "longrun_equity.csv", runner.fills, 10000.0)
    logger.info(
        "Longrun stopped. Fills: %d, Events: %d. Output: %s",
        len(runner.fills), runner.event_index, output_dir,
    )


def run_compare(config_path: Path) -> None:
    """Compare paper vs live equity curves."""
    from runner.backtest.pnl_compare import compare_from_files

    out = _output_dir(config_path)
    paper_csv = out / "paper_equity.csv"
    live_csv = out / "live_equity.csv"

    if not paper_csv.exists() or not live_csv.exists():
        print(f"Missing files. Run paper and live phases first.")
        print(f"  Expected: {paper_csv}")
        print(f"  Expected: {live_csv}")
        sys.exit(1)

    result = compare_from_files(paper_csv, live_csv)

    print("=" * 60)
    print("TESTNET VALIDATION — PnL COMPARISON")
    print("=" * 60)
    print(f"Paper final equity:  {result.backtest_final_equity}")
    print(f"Live final equity:   {result.live_final_equity}")
    print(f"Paper return:        {result.backtest_return_pct:.2f}%")
    print(f"Live return:         {result.live_return_pct:.2f}%")
    print(f"Return divergence:   {result.return_divergence_pct:.2f}%")
    print(f"Correlation:         {result.correlation:.4f}")
    print(f"Tracking error:      {result.tracking_error_pct:.4f}%")
    print(f"Paper max drawdown:  {result.backtest_max_dd_pct:.2f}%")
    print(f"Live max drawdown:   {result.live_max_dd_pct:.2f}%")
    print(f"Aligned points:      {result.aligned_points}")
    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")
    print("=" * 60)


def _pin_and_lock() -> None:
    """Pin to isolated CPU1 + mlock all memory to prevent page faults."""
    try:
        os.sched_setaffinity(0, {1})
    except OSError:
        pass
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        MCL_CURRENT_FUTURE = 3  # MCL_CURRENT | MCL_FUTURE
        libc.mlockall(MCL_CURRENT_FUTURE)
    except OSError:
        pass


def main() -> None:
    # Tune GC: raise gen0 threshold to reduce pause frequency in hot loop
    import gc
    gc.set_threshold(50_000, 50, 10)

    _pin_and_lock()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Testnet validation workflow")
    parser.add_argument("--config", type=Path, required=True, help="Testnet config YAML")
    parser.add_argument(
        "--phase",
        choices=["paper", "shadow", "live", "longrun", "compare"],
        required=True,
        help="Validation phase to run",
    )
    parser.add_argument("--duration", type=int, default=300, help="Phase duration in seconds")
    args = parser.parse_args()

    if args.phase == "paper":
        run_paper(args.config, args.duration)
    elif args.phase == "shadow":
        run_shadow(args.config, args.duration)
    elif args.phase == "live":
        run_live(args.config, args.duration)
    elif args.phase == "longrun":
        run_longrun(args.config, args.duration)
    elif args.phase == "compare":
        run_compare(args.config)


if __name__ == "__main__":
    main()
