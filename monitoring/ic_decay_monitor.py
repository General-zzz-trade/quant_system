"""IC Decay Monitor — detect model alpha decay via rolling IC comparison.

Computes rolling Spearman IC for each active model against realized forward
returns and compares with training IC to detect decay. Three-level alerting:

  GREEN:  rolling IC >= 50% of training IC
  YELLOW: rolling IC is 25-50% of training IC (WARNING)
  RED:    rolling IC < 25% of training IC or negative (CRITICAL, retrain needed)

Usage:
    python3 -m monitoring.ic_decay_monitor              # Print status table
    python3 -m monitoring.ic_decay_monitor --alert       # + send Telegram alerts
    python3 -m monitoring.ic_decay_monitor --json        # JSON output only

Designed for daily cron/systemd timer execution. Saves state to
data/runtime/ic_health.json for dashboard integration.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle  # noqa: S403 — trusted local model files produced by our training pipeline
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_DIR = Path("models_v8")
DATA_DIR = Path("data_files")
OUTPUT_PATH = Path("data/runtime/ic_health.json")

# Active models to monitor: (dir_name, symbol, interval)
ACTIVE_MODELS = [
    ("BTCUSDT_gate_v2", "BTCUSDT", "1h"),
    ("ETHUSDT_gate_v2", "ETHUSDT", "1h"),
    ("BTCUSDT_4h", "BTCUSDT", "4h"),
    ("ETHUSDT_4h", "ETHUSDT", "4h"),
]

# Rolling IC windows (in days)
IC_WINDOWS_DAYS = [30, 60, 90]

# Decay thresholds (fraction of training IC)
THRESHOLD_GREEN = 0.50   # >= 50% of training IC
THRESHOLD_YELLOW = 0.25  # >= 25% of training IC

# Auto-retrain on RED
RETRAIN_COOLDOWN_SECONDS = 24 * 3600  # 24 hours between auto-retrains
LAST_IC_RETRAIN_PATH = Path("data/runtime/last_ic_retrain.txt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resample_1h_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h OHLCV to 4h bars."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("datetime")

    agg = {
        "open_time": "first", "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }
    for col in ["quote_volume", "taker_buy_volume", "taker_buy_quote_volume", "trades"]:
        if col in df.columns:
            agg[col] = "sum"

    out = df.resample("4h").agg(agg).dropna(subset=["close"])
    return out.reset_index(drop=True)


def _load_data(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    """Load price data for a symbol/interval."""
    path = DATA_DIR / f"{symbol}_1h.csv"
    if not path.exists():
        logger.warning("Data file not found: %s", path)
        return None

    df = pd.read_csv(path)
    if interval == "4h":
        df = _resample_1h_to_4h(df)
    return df


def _load_model_config(model_dir: Path) -> Optional[Dict[str, Any]]:
    """Load model config.json."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        logger.warning("No config.json in %s", model_dir)
        return None
    with open(config_path) as f:
        return json.load(f)


def _load_model_pkl(model_dir: Path, filename: str):
    """Load a pickled model file.

    Uses pickle for LightGBM/XGBoost/Ridge models — these are trusted
    local artifacts produced by our own training pipeline.
    """
    path = model_dir / filename
    if not path.exists():
        return None
    with open(path, "rb") as f:
        raw = pickle.load(f)  # noqa: S301 — trusted local artifact
    return raw["model"] if isinstance(raw, dict) else raw


def _compute_features(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """Compute features using batch feature engine."""
    try:
        from features.batch_feature_engine import compute_features_batch
        feat_df = compute_features_batch(
            symbol, df,
            include_iv=True,
            include_onchain=True,
            include_v11=True,
        )
        return feat_df
    except Exception as e:
        logger.error("Feature computation failed for %s: %s", symbol, e)
        return pd.DataFrame()


def _predict_horizon(
    model, features: List[str], feat_df: pd.DataFrame,
    model_type: str = "lgbm",
) -> np.ndarray:
    """Generate predictions for a single horizon model."""
    # Subset to model features, fill missing with NaN
    X = pd.DataFrame(index=feat_df.index)
    for f in features:
        if f in feat_df.columns:
            X[f] = feat_df[f]
        else:
            X[f] = np.nan

    X_arr = X.values.astype(np.float64)

    # Ridge needs NaN replaced; LGBM/XGB handle NaN natively
    if model_type == "ridge":
        X_arr = np.nan_to_num(X_arr, nan=0.0)

    try:
        preds = model.predict(X_arr)
    except Exception as e:
        logger.debug("Prediction failed (%s): %s", model_type, e)
        preds = np.full(len(X_arr), np.nan)

    return preds


def _rolling_spearman_ic(
    preds: np.ndarray,
    returns: np.ndarray,
    window: int,
) -> float:
    """Compute Spearman IC over the last `window` bars."""
    n = min(len(preds), len(returns))
    if n < window:
        return np.nan

    p = preds[-window:]
    r = returns[-window:]

    # Remove NaN pairs
    mask = ~(np.isnan(p) | np.isnan(r))
    p, r = p[mask], r[mask]

    if len(p) < 30:
        return np.nan

    ic, _ = spearmanr(p, r)
    return float(ic) if np.isfinite(ic) else np.nan


def _classify_decay(rolling_ic: float, training_ic: float) -> str:
    """Classify IC decay level: GREEN, YELLOW, or RED."""
    if np.isnan(rolling_ic) or np.isnan(training_ic) or training_ic <= 0:
        return "RED"

    if rolling_ic < 0:
        return "RED"

    ratio = rolling_ic / training_ic
    if ratio >= THRESHOLD_GREEN:
        return "GREEN"
    if ratio >= THRESHOLD_YELLOW:
        return "YELLOW"
    return "RED"


# ---------------------------------------------------------------------------
# Core monitor
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    symbol: str,
    interval: str,
) -> Dict[str, Any]:
    """Evaluate IC decay for a single model.

    Returns a dict with per-horizon IC status.
    """
    model_dir = MODELS_DIR / model_name
    config = _load_model_config(model_dir)
    if config is None:
        return {"model": model_name, "error": "config not found"}

    # Load data
    df = _load_data(symbol, interval)
    if df is None or len(df) < 200:
        return {"model": model_name, "error": "insufficient data"}

    # Compute features
    feat_df = _compute_features(symbol, df)
    if feat_df.empty:
        return {"model": model_name, "error": "feature computation failed"}

    # Bars per day
    bars_per_day = 24 if interval == "1h" else 6

    # Get training metrics
    metrics = config.get("metrics", {})
    training_avg_ic = metrics.get("avg_ic", 0.0)
    per_horizon_ic = metrics.get("per_horizon_ic", {})

    horizon_results = []

    for hm in config.get("horizon_models", []):
        h = hm["horizon"]
        features = hm["features"]
        h_training_ic = hm.get("ic", per_horizon_ic.get(str(h), training_avg_ic))

        # Load LGBM model
        lgbm = _load_model_pkl(model_dir, hm["lgbm"])
        if lgbm is None:
            horizon_results.append({
                "horizon": h,
                "error": f"model file not found: {hm['lgbm']}",
            })
            continue

        # Generate predictions
        preds = _predict_horizon(lgbm, features, feat_df, model_type="lgbm")

        # Compute forward returns (h-bar ahead)
        closes = df["close"].values.astype(np.float64)
        fwd_returns = np.full(len(closes), np.nan)
        if len(closes) > h:
            fwd_returns[:-h] = closes[h:] / closes[:-h] - 1.0

        # Rolling IC for each window
        window_results = {}
        for days in IC_WINDOWS_DAYS:
            window = days * bars_per_day
            ic = _rolling_spearman_ic(preds, fwd_returns, window)
            status = _classify_decay(ic, h_training_ic)
            window_results[f"{days}d"] = {
                "ic": round(ic, 6) if np.isfinite(ic) else None,
                "status": status,
            }

        # Use 60-day window as the primary signal
        primary_ic = window_results.get("60d", {}).get("ic")
        primary_status = window_results.get("60d", {}).get("status", "RED")

        # Decay ratio
        decay_ratio = None
        if primary_ic is not None and h_training_ic > 0:
            decay_ratio = round(primary_ic / h_training_ic, 3)

        horizon_results.append({
            "horizon": h,
            "training_ic": round(h_training_ic, 6),
            "windows": window_results,
            "primary_ic": primary_ic,
            "primary_status": primary_status,
            "decay_ratio": decay_ratio,
        })

    # Overall model status = worst horizon status
    statuses = [hr["primary_status"] for hr in horizon_results if "primary_status" in hr]
    if "RED" in statuses:
        overall_status = "RED"
    elif "YELLOW" in statuses:
        overall_status = "YELLOW"
    elif statuses:
        overall_status = "GREEN"
    else:
        overall_status = "UNKNOWN"

    return {
        "model": model_name,
        "symbol": symbol,
        "interval": interval,
        "training_avg_ic": round(training_avg_ic, 6) if training_avg_ic else None,
        "overall_status": overall_status,
        "horizons": horizon_results,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "data_bars": len(df),
    }


def run_monitor() -> List[Dict[str, Any]]:
    """Evaluate all active models and return results."""
    results = []
    for model_name, symbol, interval in ACTIVE_MODELS:
        model_dir = MODELS_DIR / model_name
        if not model_dir.exists():
            logger.info("Skipping %s — directory not found", model_name)
            continue
        logger.info("Evaluating %s (%s %s)...", model_name, symbol, interval)
        try:
            result = evaluate_model(model_name, symbol, interval)
            results.append(result)
        except Exception as e:
            logger.error("Failed to evaluate %s: %s", model_name, e)
            results.append({"model": model_name, "error": str(e)})
    return results


def save_results(results: List[Dict[str, Any]]) -> None:
    """Save results to data/runtime/ic_health.json."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Saved IC health to %s", OUTPUT_PATH)


def print_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted status table."""
    STATUS_COLORS = {"GREEN": "\033[92m", "YELLOW": "\033[93m", "RED": "\033[91m"}
    RESET = "\033[0m"

    print()
    print("=" * 88)
    print(f"{'IC Decay Monitor':^88}")
    print(f"{'(' + datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC') + ')':^88}")
    print("=" * 88)
    print(
        f"{'Model':<22} {'H':>3} {'Train IC':>10} "
        f"{'30d IC':>9} {'60d IC':>9} {'90d IC':>9} "
        f"{'Decay%':>8} {'Status':>8}"
    )
    print("-" * 88)

    for r in results:
        if "error" in r and "horizons" not in r:
            print(f"{r['model']:<22}  {'ERROR: ' + r['error']}")
            continue

        for hr in r.get("horizons", []):
            if "error" in hr:
                print(f"{r['model']:<22} {hr['horizon']:>3}  {hr['error']}")
                continue

            train_ic = hr.get("training_ic", 0)
            windows = hr.get("windows", {})
            ic_30 = windows.get("30d", {}).get("ic")
            ic_60 = windows.get("60d", {}).get("ic")
            ic_90 = windows.get("90d", {}).get("ic")
            decay = hr.get("decay_ratio")
            status = hr.get("primary_status", "UNKNOWN")

            ic_30_s = f"{ic_30:.4f}" if ic_30 is not None else "N/A"
            ic_60_s = f"{ic_60:.4f}" if ic_60 is not None else "N/A"
            ic_90_s = f"{ic_90:.4f}" if ic_90 is not None else "N/A"
            decay_s = f"{decay * 100:.0f}%" if decay is not None else "N/A"

            color = STATUS_COLORS.get(status, "")
            print(
                f"{r['model']:<22} {hr['horizon']:>3} {train_ic:>10.4f} "
                f"{ic_30_s:>9} {ic_60_s:>9} {ic_90_s:>9} "
                f"{decay_s:>8} {color}{status:>8}{RESET}"
            )

    print("=" * 88)

    # Summary
    statuses = [r.get("overall_status") for r in results if "overall_status" in r]
    n_green = statuses.count("GREEN")
    n_yellow = statuses.count("YELLOW")
    n_red = statuses.count("RED")
    print(
        f"\nSummary: {n_green} GREEN, {n_yellow} YELLOW, {n_red} RED "
        f"out of {len(statuses)} models"
    )
    if n_red > 0:
        red_models = [r["model"] for r in results if r.get("overall_status") == "RED"]
        print(f"  RETRAIN NEEDED: {', '.join(red_models)}")
    print()


def send_alerts(results: List[Dict[str, Any]]) -> None:
    """Send Telegram alerts for YELLOW/RED models."""
    try:
        from monitoring.notify import send_alert, AlertLevel
    except ImportError:
        logger.warning("monitoring.notify not available — skipping alerts")
        return

    for r in results:
        status = r.get("overall_status")
        if status not in ("YELLOW", "RED"):
            continue

        model = r.get("model", "unknown")
        symbol = r.get("symbol", "")

        # Build details from horizon data
        details: Dict[str, str] = {"symbol": symbol, "model": model}
        for hr in r.get("horizons", []):
            h = hr.get("horizon", "?")
            pic = hr.get("primary_ic")
            tic = hr.get("training_ic", 0)
            details[f"h{h}_ic"] = f"{pic:.4f}" if pic is not None else "N/A"
            details[f"h{h}_train_ic"] = f"{tic:.4f}"
            dr = hr.get("decay_ratio")
            if dr is not None:
                details[f"h{h}_decay"] = f"{dr * 100:.0f}%"

        if status == "RED":
            send_alert(
                AlertLevel.CRITICAL,
                f"IC DECAY: {model} — retrain needed",
                details=details,
                source="ic_decay_monitor",
            )
        else:
            send_alert(
                AlertLevel.WARNING,
                f"IC DECAY: {model} — alpha weakening",
                details=details,
                source="ic_decay_monitor",
            )


# ---------------------------------------------------------------------------
# Auto-retrain trigger
# ---------------------------------------------------------------------------

def _should_retrain() -> bool:
    """Check if enough time has passed since last IC-triggered retrain."""
    if not LAST_IC_RETRAIN_PATH.exists():
        return True
    try:
        ts = float(LAST_IC_RETRAIN_PATH.read_text().strip())
        elapsed = datetime.now(timezone.utc).timestamp() - ts
        return elapsed > RETRAIN_COOLDOWN_SECONDS
    except (ValueError, OSError):
        return True


def _run_retrain() -> None:
    """Run auto_retrain in a subprocess (called from background thread)."""
    try:
        logger.info("Starting IC-triggered auto-retrain subprocess...")
        subprocess.run(
            [sys.executable, "-m", "scripts.auto_retrain", "--force", "--sighup"],
            cwd="/quant_system",
            timeout=1800,
        )
        logger.info("IC-triggered auto-retrain completed.")
    except subprocess.TimeoutExpired:
        logger.error("IC-triggered auto-retrain timed out after 1800s.")
    except Exception as e:
        logger.error("IC-triggered auto-retrain failed: %s", e)


def maybe_trigger_retrain(results: List[Dict[str, Any]]) -> None:
    """If any model is RED and cooldown has passed, trigger retrain in background."""
    red_models = [
        r["model"] for r in results
        if r.get("overall_status") == "RED"
    ]
    if not red_models:
        return

    if not _should_retrain():
        logger.info(
            "IC RED detected for %s but retrain cooldown not elapsed — skipping.",
            ", ".join(red_models),
        )
        return

    for model in red_models:
        logger.warning("IC RED detected for %s, triggering auto-retrain", model)

    # Write timestamp to prevent re-triggering within 24h
    LAST_IC_RETRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_IC_RETRAIN_PATH.write_text(str(datetime.now(timezone.utc).timestamp()))

    # Launch retrain in background thread so monitor can finish promptly
    t = threading.Thread(target=_run_retrain, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IC Decay Monitor — detect model alpha decay",
    )
    parser.add_argument(
        "--alert", action="store_true",
        help="Send Telegram alerts for YELLOW/RED models",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="JSON output only (no table)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Change to project root for relative paths
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    results = run_monitor()
    save_results(results)

    if args.json:
        print(json.dumps({"models": results}, indent=2, default=str))
    else:
        print_table(results)

    if args.alert:
        send_alerts(results)

    # Auto-retrain if any model hit RED
    maybe_trigger_retrain(results)

    # Exit code: 2 if any RED, 1 if any YELLOW, 0 if all GREEN
    statuses = [r.get("overall_status") for r in results if "overall_status" in r]
    if "RED" in statuses:
        sys.exit(2)
    elif "YELLOW" in statuses:
        sys.exit(1)


if __name__ == "__main__":
    main()
