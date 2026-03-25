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

from alpha.retrain.pipeline import (
    MIN_IC,
    MODEL_DIR_15M_TEMPLATE,
    MODEL_DIR_4H_TEMPLATE,
    DEFAULT_HORIZONS_15M,
    log_retrain_event,
)

logger = logging.getLogger(__name__)

# ── 15m cross-TF feature injection ──
# Extra features to inject into 15m training data beyond standard bar features.
# These are computed by batch_feature_engine but need explicit inclusion for 15m.

# V14 dominance features (already in batch_feature_engine via _add_dominance_features)
_15M_DOMINANCE_FEATURES = [
    "btc_dom_dev_20",   # BTC/ETH ratio deviation from 20-bar MA
    "btc_dom_dev_50",   # BTC/ETH ratio deviation from 50-bar MA
    "btc_dom_ret_24",   # BTC/ETH ratio 24-bar pct change
    "btc_dom_ret_72",   # BTC/ETH ratio 72-bar pct change
]

# Cross-TF: 1h regime label injected as 15m feature
_15M_CROSS_TF_FEATURES = [
    "regime_1h_label",  # from RustCompositeRegimeDetector on 1h bars, ffill to 15m
]


def prepare_15m_extra_features(symbol: str, df_15m: Any) -> Any:
    """Inject dominance and cross-TF regime features into 15m training DataFrame.

    Called before model training to enrich 15m bars with:
    1. V14 dominance features (BTC/ETH ratio metrics) — already computed if
       batch_feature_engine is used, but added explicitly for standalone 15m data.
    2. 1h regime label forward-filled to 15m resolution.

    Args:
        symbol: Trading symbol (e.g. "BTCUSDT").
        df_15m: 15m OHLCV DataFrame with timestamp/open_time column.

    Returns:
        df_15m with extra columns added in-place (also returned for chaining).
    """
    import numpy as np
    import pandas as pd

    ts_col = "open_time" if "open_time" in df_15m.columns else "timestamp"

    # --- V14: Dominance features ---
    # Load 1h ETH data for BTC/ETH ratio (only meaningful for BTCUSDT)
    if symbol == "BTCUSDT":
        eth_path = Path("data_files/ETHUSDT_15m.csv")
        if eth_path.exists():
            eth_df = pd.read_csv(eth_path)
            eth_ts_col = "open_time" if "open_time" in eth_df.columns else "timestamp"
            # Merge on timestamp for aligned ratio computation
            merged = df_15m[[ts_col, "close"]].merge(
                eth_df[[eth_ts_col, "close"]].rename(
                    columns={eth_ts_col: ts_col, "close": "eth_close"}
                ),
                on=ts_col, how="left",
            )
            eth_close = merged["eth_close"].values
            btc_close = merged["close"].values
            ratio = np.where(eth_close > 0, btc_close / eth_close, np.nan)
            ratio_s = pd.Series(ratio)
            ma20 = ratio_s.rolling(20).mean().values
            ma50 = ratio_s.rolling(50).mean().values
            df_15m["btc_dom_dev_20"] = ratio / np.where(ma20 > 0, ma20, np.nan) - 1
            df_15m["btc_dom_dev_50"] = ratio / np.where(ma50 > 0, ma50, np.nan) - 1
            df_15m["btc_dom_ret_24"] = ratio_s.pct_change(24).values
            df_15m["btc_dom_ret_72"] = ratio_s.pct_change(72).values
            logger.info("15m %s: injected V14 dominance features (4 cols)", symbol)
        else:
            for col in _15M_DOMINANCE_FEATURES:
                df_15m[col] = np.nan
            logger.warning("15m %s: ETH 15m data not found, dominance features = NaN", symbol)
    else:
        # For ETH, dominance features are less meaningful (self-referential)
        for col in _15M_DOMINANCE_FEATURES:
            df_15m[col] = np.nan

    # --- Cross-TF: 1h regime label ---
    regime_path = Path(f"data_files/{symbol}_1h.csv")
    if regime_path.exists():
        try:
            df_1h = pd.read_csv(regime_path)
            h1_ts_col = "open_time" if "open_time" in df_1h.columns else "timestamp"

            # Try to compute regime labels from 1h data
            try:
                from regime.composite_regime import CompositeRegimeDetector
                detector = CompositeRegimeDetector()
                regime_labels = []
                for _, row in df_1h.iterrows():
                    label = detector.detect(row.to_dict())
                    regime_labels.append(label)
                df_1h["regime_1h_label"] = regime_labels
            except (ImportError, Exception) as e:
                # Fallback: use simple volatility regime (0=low, 1=mid, 2=high)
                logger.debug("Regime detector unavailable (%s), using vol proxy", e)
                returns = df_1h["close"].pct_change()
                vol_20 = returns.rolling(20).std()
                vol_median = vol_20.median()
                df_1h["regime_1h_label"] = np.where(
                    vol_20 > vol_median * 1.5, 2,
                    np.where(vol_20 > vol_median * 0.5, 1, 0)
                )

            # Merge 1h regime into 15m via asof join (forward-fill)
            df_1h_regime = df_1h[[h1_ts_col, "regime_1h_label"]].rename(
                columns={h1_ts_col: ts_col}
            )
            df_15m = pd.merge_asof(
                df_15m.sort_values(ts_col),
                df_1h_regime.sort_values(ts_col),
                on=ts_col,
                direction="backward",
            )
            logger.info("15m %s: injected cross-TF regime_1h_label", symbol)
        except Exception as e:
            logger.warning("15m %s: cross-TF regime injection failed: %s", symbol, e)
            df_15m["regime_1h_label"] = np.nan
    else:
        import numpy as np
        df_15m["regime_1h_label"] = np.nan
        logger.warning("15m %s: 1h data not found for cross-TF regime", symbol)

    return df_15m


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
