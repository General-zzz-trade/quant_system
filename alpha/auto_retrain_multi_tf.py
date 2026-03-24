# alpha/auto_retrain_multi_tf.py
"""Multi-timeframe retrain functions (15m and 4h) for auto_retrain pipeline.

Extracted from auto_retrain.py to reduce file size.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from alpha.auto_retrain import (
    MIN_IC,
    MODEL_DIR_15M_TEMPLATE,
    MODEL_DIR_4H_TEMPLATE,
    DEFAULT_HORIZONS_15M,
    log_retrain_event,
)

logger = logging.getLogger(__name__)


def download_15m_data(symbols: List[str]) -> Dict[str, int]:
    """Download fresh 15m kline data before retraining.

    Returns dict of symbol -> new_bars_count.
    """
    logger.info("Downloading 15m kline data for %s...", symbols)
    try:
        from scripts.data.download_15m_klines import download_symbol
        results = {}
        for symbol in symbols:
            try:
                n = download_symbol(symbol)
                results[symbol] = n
                logger.info("Downloaded %d new 15m bars for %s", n, symbol)
            except Exception as e:
                logger.warning("15m download failed for %s: %s", symbol, e)
                results[symbol] = -1
        return results
    except ImportError as e:
        logger.warning("15m download script not available: %s", e)
        return {}


def retrain_15m_symbols(
    symbols: List[str],
    dry_run: bool = False,
    force: bool = False,
    max_age_days: int = 90,
) -> Dict[str, Dict[str, Any]]:
    """Retrain 15m models for the given symbols.

    Downloads fresh data, trains LightGBM + XGBoost, validates, and saves.
    Returns dict of symbol -> result.
    """
    results = {}

    for symbol in symbols:
        model_dir = Path(MODEL_DIR_15M_TEMPLATE.format(symbol=symbol))
        data_path = Path(f"data_files/{symbol}_15m.csv")

        result = {
            "symbol": symbol,
            "timeframe": "15m",
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "dry_run": dry_run,
        }

        # Check if retrain needed (reuse logic with 15m model dir)
        if not force:
            config_path = model_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                train_date_str = cfg.get("train_date", "")
                if train_date_str:
                    try:
                        train_date = datetime.strptime(train_date_str.split(" ")[0], "%Y-%m-%d")
                        age_days = (datetime.now() - train_date).days
                        if age_days <= max_age_days:
                            avg_ic = cfg.get("metrics", {}).get("avg_ic", 0)
                            if avg_ic >= MIN_IC:
                                logger.info("15m %s: model healthy (age=%dd, IC=%.4f), skipping",
                                           symbol, age_days, avg_ic)
                                result["skipped"] = True
                                result["reason"] = f"model healthy (age={age_days}d, IC={avg_ic:.4f})"
                                results[symbol] = result
                                continue
                    except ValueError as e:
                        logger.warning("Failed to parse 15m model train_date for %s: %s", symbol, e)

        if not data_path.exists():
            logger.warning("15m %s: no data file at %s", symbol, data_path)
            result["error"] = f"data file not found: {data_path}"
            results[symbol] = result
            continue

        # Backup
        if not dry_run and model_dir.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = model_dir.parent / f"{model_dir.name}_backup_{timestamp}"
            shutil.copytree(model_dir, backup_dir)
            result["backup_dir"] = str(backup_dir)
            logger.info("Backed up 15m %s -> %s", model_dir, backup_dir)

        # Train
        horizons = DEFAULT_HORIZONS_15M.get(symbol, [4, 8])
        logger.info("Training 15m %s with horizons %s...", symbol, horizons)
        t0 = time.time()

        try:
            from scripts.training.train_15m import train_symbol_15m
            success = train_symbol_15m(symbol, horizons)
        except Exception as e:
            logger.error("15m training failed for %s: %s", symbol, e)
            result["error"] = str(e)
            results[symbol] = result
            log_retrain_event(result)
            continue

        train_time = time.time() - t0
        result["train_time_sec"] = train_time

        if not success:
            logger.warning("15m training failed production checks for %s", symbol)
            result["error"] = "failed production checks"

            # Restore backup
            if not dry_run and "backup_dir" in result:
                backup_path = Path(result["backup_dir"])
                if backup_path.exists() and model_dir.exists():
                    shutil.rmtree(model_dir)
                    shutil.copytree(backup_path, model_dir)
                    logger.info("Restored 15m backup for %s", symbol)
        else:
            result["success"] = True
            # Load new metrics
            config_path = model_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    new_cfg = json.load(f)
                new_metrics = new_cfg.get("metrics", {})
                result["new_sharpe"] = new_metrics.get("sharpe", 0)
                result["new_avg_ic"] = new_metrics.get("avg_ic", 0)
                result["new_trades"] = new_metrics.get("trades", 0)

            if dry_run:
                logger.info("DRY RUN 15m %s: would deploy (Sharpe %.2f, IC %.4f)",
                           symbol, result.get("new_sharpe", 0), result.get("new_avg_ic", 0))
            else:
                logger.info("DEPLOYED 15m %s: Sharpe %.2f, IC %.4f in %.0fs",
                           symbol, result.get("new_sharpe", 0), result.get("new_avg_ic", 0), train_time)

        results[symbol] = result
        log_retrain_event(result)

    return results


def retrain_4h_symbols(
    symbols: List[str],
    dry_run: bool = False,
    force: bool = False,
    max_age_days: int = 90,
) -> Dict[str, Dict[str, Any]]:
    """Retrain 4h models for the given symbols.

    Trains via train_4h_daily, validates with IC/Sharpe gates, and saves.
    Returns dict of symbol -> result.
    """
    results = {}

    for symbol in symbols:
        model_dir = Path(MODEL_DIR_4H_TEMPLATE.format(symbol=symbol))
        data_path = Path(f"data_files/{symbol}_1h.csv")

        result = {
            "symbol": symbol,
            "timeframe": "4h",
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "dry_run": dry_run,
        }

        # Check if retrain needed (reuse logic with 4h model dir)
        if not force:
            config_path = model_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                train_date_str = cfg.get("train_date", "")
                if train_date_str:
                    try:
                        train_date = datetime.strptime(train_date_str.split(" ")[0], "%Y-%m-%d")
                        age_days = (datetime.now() - train_date).days
                        if age_days <= max_age_days:
                            avg_ic = cfg.get("metrics", {}).get("avg_ic", 0)
                            if avg_ic >= MIN_IC:
                                logger.info("4h %s: model healthy (age=%dd, IC=%.4f), skipping",
                                           symbol, age_days, avg_ic)
                                result["skipped"] = True
                                result["reason"] = f"model healthy (age={age_days}d, IC={avg_ic:.4f})"
                                results[symbol] = result
                                continue
                    except ValueError as e:
                        logger.warning("Failed to parse 4h model train_date for %s: %s", symbol, e)

        if not data_path.exists():
            logger.warning("4h %s: no data file at %s", symbol, data_path)
            result["error"] = f"data file not found: {data_path}"
            results[symbol] = result
            continue

        # Backup
        if not dry_run and model_dir.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = model_dir.parent / f"{model_dir.name}_backup_{timestamp}"
            shutil.copytree(model_dir, backup_dir)
            result["backup_dir"] = str(backup_dir)
            logger.info("Backed up 4h %s -> %s", model_dir, backup_dir)

        # Train
        logger.info("Training 4h %s...", symbol)
        t0 = time.time()

        try:
            from scripts.training.train_4h_daily import train_symbol
            success = train_symbol(symbol, interval="4h")
        except Exception as e:
            logger.error("4h training failed for %s: %s", symbol, e)
            result["error"] = str(e)
            results[symbol] = result
            log_retrain_event(result)
            continue

        train_time = time.time() - t0
        result["train_time_sec"] = train_time

        if not success:
            logger.warning("4h training failed production checks for %s", symbol)
            result["error"] = "failed production checks"

            # Restore backup
            if not dry_run and "backup_dir" in result:
                backup_path = Path(result["backup_dir"])
                if backup_path.exists() and model_dir.exists():
                    shutil.rmtree(model_dir)
                    shutil.copytree(backup_path, model_dir)
                    logger.info("Restored 4h backup for %s", symbol)
        else:
            result["success"] = True
            # Load new metrics
            config_path = model_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    new_cfg = json.load(f)
                new_metrics = new_cfg.get("metrics", {})
                result["new_sharpe"] = new_metrics.get("sharpe", 0)
                result["new_avg_ic"] = new_metrics.get("avg_ic", 0)
                result["new_trades"] = new_metrics.get("trades", 0)

            if dry_run:
                logger.info("DRY RUN 4h %s: would deploy (Sharpe %.2f, IC %.4f)",
                           symbol, result.get("new_sharpe", 0), result.get("new_avg_ic", 0))
            else:
                logger.info("DEPLOYED 4h %s: Sharpe %.2f, IC %.4f in %.0fs",
                           symbol, result.get("new_sharpe", 0), result.get("new_avg_ic", 0), train_time)

        results[symbol] = result
        log_retrain_event(result)

    return results
