#!/usr/bin/env python3
"""Automated Walk-Forward Retraining Pipeline.

Implements periodic model retraining with IC validation before deployment.
Designed to run as a cron job or manual trigger.

Features:
- Walk-forward: trains on expanding window, validates on recent data
- IC validation gate: new model must beat minimum IC threshold
- Comparison gate: new model Sharpe must be >= old model × decay_tolerance
- Automatic backup of old models before replacement
- Supports horizon removal (drops h48 if IC consistently negative)
- Logs all retrain events for audit

Usage:
    # Retrain all symbols
    python3 -m scripts.auto_retrain

    # Retrain specific symbol
    python3 -m scripts.auto_retrain --symbol ETHUSDT

    # Dry run (validate but don't replace)
    python3 -m scripts.auto_retrain --dry-run

    # Force retrain even if IC is still healthy
    python3 -m scripts.auto_retrain --force

    # Custom horizons (e.g., drop h48)
    python3 -m scripts.auto_retrain --horizons 12,24

    # Cron setup (retrain every Sunday at 2am UTC):
    # 0 2 * * 0 cd /quant_system && python3 -m scripts.auto_retrain >> logs/retrain.log 2>&1
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


sys.path.insert(0, "/quant_system")

from _quant_hotpath import cpp_greedy_ic_select  # type: ignore[import-untyped]  # noqa: E402

# Rust-accelerated greedy IC feature selection — used during walk-forward
# retrain to select optimal feature subsets in O(n*k) instead of O(n^2).
_rust_greedy_ic_select = cpp_greedy_ic_select

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Configuration (extracted to alpha/auto_retrain_config.py) ──
from alpha.auto_retrain_config import (  # noqa: E402, F401
    SYMBOLS, DEFAULT_HORIZONS, MODEL_DIR_TEMPLATE, MODEL_DIR_OVERRIDES,
    DATA_DIR_TEMPLATE, RETRAIN_LOG,
    SYMBOLS_15M, DEFAULT_HORIZONS_15M, MODEL_DIR_15M_TEMPLATE,
    SYMBOLS_4H, MODEL_DIR_4H_TEMPLATE,
    MIN_IC, MIN_SHARPE, DECAY_TOLERANCE, MIN_TRADES, BOOTSTRAP_P5_MIN,
    MIN_FINAL_SHARPE, MIN_FINAL_AVG_NET_BPS,
    DAILY_MAX_AGE_HOURS, DAILY_IC_TOLERANCE, DAILY_VALIDATION_MONTHS,
)


def calibrate_ensemble_weights(model_dir: Path, shrinkage: float = 0.3) -> Optional[Dict[str, float]]:
    """Calibrate Ridge/LGBM ensemble weights using recent IC and save to config.

    Loads recent prediction scores and returns from the model directory,
    calls rust_adaptive_ensemble_calibrate to compute optimal weights,
    and saves them to config.json under "ensemble_weights".

    Returns calibrated weights dict or None if calibration failed.
    Best-effort: never raises.
    """
    try:
        from _quant_hotpath import rust_adaptive_ensemble_calibrate  # type: ignore[import-untyped]

        config_path = model_dir / "config.json"
        if not config_path.exists():
            logger.debug("calibrate_ensemble_weights: no config.json in %s", model_dir)
            return None

        with open(config_path) as f:
            cfg = json.load(f)

        # Load score history (per-model predictions) and return history from
        # validation artifacts saved by train_v11
        scores_path = model_dir / "val_scores.json"
        returns_path = model_dir / "val_returns.json"

        if scores_path.exists() and returns_path.exists():
            with open(scores_path) as f:
                score_history = json.load(f)
            with open(returns_path) as f:
                return_history = json.load(f)
        else:
            # Fallback: construct from metrics if validation artifacts not available
            metrics = cfg.get("metrics", {})
            per_h_ic = metrics.get("per_horizon_ic", {})
            if not per_h_ic:
                logger.debug("calibrate_ensemble_weights: no score/return history for %s", model_dir)
                return None
            # Use per-horizon IC as a proxy for score quality
            score_history = {
                "ridge": [float(ic) for ic in per_h_ic.values()],
                "lgbm": [float(ic) * 0.9 for ic in per_h_ic.values()],
            }
            return_history = [float(ic) for ic in per_h_ic.values()]

        result = rust_adaptive_ensemble_calibrate(
            "ic_weighted", score_history, return_history, shrinkage,
        )
        if result is None:
            logger.debug("rust_adaptive_ensemble_calibrate returned None for %s", model_dir)
            return None

        ridge_w = result.get("ridge", 0.6)
        lgbm_w = result.get("lgbm", 0.4)
        total = ridge_w + lgbm_w
        if total <= 0:
            logger.warning("calibrate_ensemble_weights: invalid weights sum for %s", model_dir)
            return None
        weights = {"ridge": round(ridge_w / total, 4), "lgbm": round(lgbm_w / total, 4)}

        # Save to config.json
        cfg["ensemble_weights"] = weights
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        logger.info("Ensemble weights calibrated for %s: %s (shrinkage=%.2f)",
                     model_dir.name, weights, shrinkage)
        return weights

    except Exception as e:
        logger.warning("calibrate_ensemble_weights failed (non-blocking): %s", e)
        return None


def save_experiment_metadata(
    model_dir: Path,
    symbol: str,
    horizons: List[int],
    old_config: Optional[Dict[str, Any]],
    retrain_trigger: str = "scheduled",
) -> bool:
    """Save experiment tracking metadata alongside the trained model.

    Writes {model_dir}/experiment_meta.json with training details, metrics,
    and lineage info. Best-effort: never raises.

    Returns True if metadata was saved, False otherwise.
    """
    try:
        config_path = model_dir / "config.json"
        if not config_path.exists():
            logger.debug("save_experiment_metadata: no config.json in %s", model_dir)
            return False

        with open(config_path) as f:
            cfg = json.load(f)

        metrics = cfg.get("metrics", {})
        features = cfg.get("features", [])

        # Determine parent model name from backup or old config
        parent_model = None
        if old_config:
            old_train_date = old_config.get("train_date", "")
            if old_train_date:
                date_part = old_train_date.split(" ")[0].replace("-", "")
                parent_model = f"{model_dir.name}_backup_{date_part}"

        # Compute train/val row counts from metrics if available
        train_rows = metrics.get("train_rows", metrics.get("n_train", 0))
        val_rows = metrics.get("val_rows", metrics.get("n_val", metrics.get("oos_bars", 0)))

        meta = {
            "trained_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": symbol,
            "horizons": horizons,
            "n_features": len(features),
            "feature_names": list(features),
            "train_rows": train_rows,
            "val_rows": val_rows,
            "train_ic": metrics.get("train_ic", metrics.get("is_avg_ic", 0)),
            "val_ic": metrics.get("avg_ic", 0),
            "val_sharpe": metrics.get("sharpe", 0),
            "ensemble_weights": cfg.get("ensemble_weights", {
                "ridge": cfg.get("ridge_weight", 0.6),
                "lgbm": cfg.get("lgbm_weight", 0.4),
            }),
            "parent_model": parent_model,
            "retrain_trigger": retrain_trigger,
        }

        meta_path = model_dir / "experiment_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Experiment metadata saved to %s", meta_path)
        return True

    except Exception as e:
        logger.warning("save_experiment_metadata failed (non-blocking): %s", e)
        return False


def _check_model_age_hours(symbol: str, model_dir: Optional[Path] = None) -> Optional[float]:
    """Return model age in hours, or None if no model/date found."""
    if model_dir is None:
        model_dir = _model_dir_for(symbol)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        train_date_str = cfg.get("train_date", "")
        if not train_date_str:
            return None
        train_date = datetime.strptime(train_date_str.split(" ")[0], "%Y-%m-%d")
        return (datetime.now() - train_date).total_seconds() / 3600
    except (ValueError, KeyError):
        return None


def _daily_retrain_needed(symbol: str, model_dir: Optional[Path] = None) -> Tuple[bool, str]:
    """Check if daily retrain is needed (model older than DAILY_MAX_AGE_HOURS).

    Returns (needs_retrain, reason).
    """
    age_hours = _check_model_age_hours(symbol, model_dir)
    if age_hours is None:
        return True, "no existing model or missing train_date"
    if age_hours < DAILY_MAX_AGE_HOURS:
        return False, f"model is {age_hours:.1f}h old (< {DAILY_MAX_AGE_HOURS}h)"
    return True, f"model is {age_hours:.1f}h old (>= {DAILY_MAX_AGE_HOURS}h)"


def _daily_ic_gate(old_config: Optional[Dict[str, Any]], new_config: Dict[str, Any]) -> Tuple[bool, str]:
    """Check if new model IC passes daily tolerance gate.

    New IC must be >= old IC * DAILY_IC_TOLERANCE (5% tolerance).
    If no old model, always passes.

    Returns (passes, reason).
    """
    if old_config is None:
        return True, "no old model to compare"
    old_ic = old_config.get("metrics", {}).get("avg_ic", 0)
    new_ic = new_config.get("metrics", {}).get("avg_ic", 0)
    threshold = old_ic * DAILY_IC_TOLERANCE
    if new_ic >= threshold:
        return True, f"new IC {new_ic:.4f} >= {threshold:.4f} (old {old_ic:.4f} * {DAILY_IC_TOLERANCE})"
    return False, f"new IC {new_ic:.4f} < {threshold:.4f} (old {old_ic:.4f} * {DAILY_IC_TOLERANCE})"


def load_current_config(symbol: str) -> Optional[Dict[str, Any]]:
    """Load the current model's config.json."""
    config_path = _model_dir_for(symbol) / "config.json"
    if not config_path.exists():
        # Fallback to default template path
        config_path = Path(MODEL_DIR_TEMPLATE.format(symbol=symbol)) / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return json.load(f)


def _model_dir_for(symbol: str) -> Path:
    """Get model directory for a symbol, handling overrides."""
    if symbol in MODEL_DIR_OVERRIDES:
        return Path(MODEL_DIR_OVERRIDES[symbol])
    return Path(MODEL_DIR_TEMPLATE.format(symbol=symbol))


def check_data_freshness(symbol: str, max_stale_hours: int = 26) -> Tuple[bool, str]:
    """Check if data file is fresh enough for retraining.

    Returns (is_fresh, message).
    """
    data_path = Path(DATA_DIR_TEMPLATE.format(symbol=symbol))
    if not data_path.exists():
        return False, f"data file not found: {data_path}"

    import os
    mtime = os.path.getmtime(data_path)
    age_hours = (time.time() - mtime) / 3600
    if age_hours > max_stale_hours:
        return False, f"data is {age_hours:.0f}h old (max {max_stale_hours}h)"
    return True, f"data is {age_hours:.1f}h old"


def check_needs_retrain(
    symbol: str,
    force: bool = False,
    max_age_days: int = 90,
) -> Tuple[bool, str]:
    """Check if a symbol needs retraining.

    Returns (needs_retrain, reason).
    """
    if force:
        return True, "forced"

    cfg = load_current_config(symbol)
    if cfg is None:
        return True, "no existing model"

    # Check model age
    train_date_str = cfg.get("train_date", "")
    if train_date_str:
        try:
            train_date = datetime.strptime(train_date_str.split(" ")[0], "%Y-%m-%d")
            age_days = (datetime.now() - train_date).days
            if age_days > max_age_days:
                return True, f"model is {age_days} days old (max {max_age_days})"
        except ValueError as e:
            logger.warning("Failed to parse model train_date '%s': %s", train_date_str, e)

    # Check IC decay in metrics
    metrics = cfg.get("metrics", {})
    per_h_ic = metrics.get("per_horizon_ic", {})
    negative_horizons = [h for h, ic in per_h_ic.items() if float(ic) < 0]
    if len(negative_horizons) >= len(per_h_ic) // 2:
        return True, f"IC negative for horizons: {negative_horizons}"

    # Check if average IC is low
    avg_ic = metrics.get("avg_ic", 0)
    if avg_ic < MIN_IC:
        return True, f"avg IC={avg_ic:.4f} below threshold {MIN_IC}"

    return False, "model is healthy"


def retrain_symbol(
    symbol: str,
    horizons: List[int],
    dry_run: bool = False,
    retrain_trigger: str = "scheduled",
) -> Dict[str, Any]:
    """Retrain a symbol and validate the new model.

    Returns a result dict with success status and metrics.
    """
    from scripts.training.train_v11 import train_symbol_v11

    result = {
        "symbol": symbol,
        "horizons": horizons,
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "dry_run": dry_run,
    }

    # Data freshness gate: don't retrain on stale data
    fresh, fresh_msg = check_data_freshness(symbol)
    if not fresh:
        logger.warning("SKIP %s: %s", symbol, fresh_msg)
        result["error"] = f"stale data: {fresh_msg}"
        return result
    logger.info("%s data freshness: %s", symbol, fresh_msg)

    model_dir = _model_dir_for(symbol)
    old_config = load_current_config(symbol)

    # Record old metrics for comparison
    if old_config:
        old_metrics = old_config.get("metrics", {})
        result["old_sharpe"] = old_metrics.get("sharpe", 0)
        result["old_avg_ic"] = old_metrics.get("avg_ic", 0)
        result["old_train_date"] = old_config.get("train_date", "unknown")
        result["old_horizons"] = old_config.get("horizons", [])

    # Backup before retrain
    if not dry_run and model_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = model_dir.parent / f"{model_dir.name}_backup_{timestamp}"
        shutil.copytree(model_dir, backup_dir)
        result["backup_dir"] = str(backup_dir)
        logger.info("Backed up %s → %s", model_dir, backup_dir)

    # Train
    logger.info("Training %s with horizons %s...", symbol, horizons)
    t0 = time.time()

    try:
        success = train_symbol_v11(
            symbol,
            horizons=horizons,
            trailing_stop_pct=0.0,
            zscore_cap=0.0,
            regime_gate_enabled=False,
            lgbm_xgb_weight=0.5,
        )
        # Post-train config fixup: restore ensemble method + preserve manual overrides
        if success:
            cfg_path = model_dir / "config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = json.load(f)
                cfg["ensemble_method"] = "ic_weighted"
                cfg["ic_ema_span"] = 720
                cfg["ic_min_threshold"] = -0.01

                # Preserve manual overrides from old config (long_only, ridge_weight, etc.)
                # These were optimized via config sweep and should survive retraining.
                _PRESERVE_KEYS = ["long_only", "ridge_weight", "lgbm_weight"]
                if old_config:
                    for key in _PRESERVE_KEYS:
                        if key in old_config and key not in cfg:
                            cfg[key] = old_config[key]
                            logger.info("%s: preserved %s=%s from old config", symbol, key, old_config[key])

                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)
    except Exception as e:
        logger.error("Training failed for %s: %s", symbol, e)
        result["error"] = str(e)
        return result

    train_time = time.time() - t0
    result["train_time_sec"] = train_time

    if not success:
        logger.warning("Training failed production checks for %s", symbol)
        result["error"] = "failed production checks"
        return result

    # train_v11 always saves to {symbol}_gate_v2/, but some symbols use
    # a different model_dir (e.g., SUI → models_v8/SUIUSDT). Sync if needed.
    train_output_dir = Path(f"models_v8/{symbol}_gate_v2")
    if model_dir != train_output_dir and train_output_dir.exists():
        # Copy trained model to the correct model_dir
        if model_dir.exists():
            shutil.rmtree(model_dir)
        shutil.copytree(train_output_dir, model_dir)
        logger.info("Synced train output %s → %s", train_output_dir, model_dir)

    # Load new config and validate
    new_config = load_current_config(symbol)
    if new_config is None:
        result["error"] = "no config after training"
        return result

    new_metrics = new_config.get("metrics", {})
    new_sharpe = new_metrics.get("sharpe", 0)
    new_avg_ic = new_metrics.get("avg_ic", 0)
    new_trades = new_metrics.get("trades", 0)
    new_p5 = new_metrics.get("bootstrap_sharpe_p5", 0)
    new_final_sharpe = new_metrics.get("final_sharpe", None)
    new_final_avg_net_bps = new_metrics.get("final_avg_net_bps", None)

    result["new_sharpe"] = new_sharpe
    result["new_avg_ic"] = new_avg_ic
    result["new_trades"] = new_trades

    # Validation gates
    gates = {
        "ic_gate": new_avg_ic >= MIN_IC,
        "sharpe_gate": new_sharpe >= MIN_SHARPE,
        "trades_gate": new_trades >= MIN_TRADES,
        "bootstrap_gate": new_p5 >= BOOTSTRAP_P5_MIN,
        "final_sharpe_gate": new_final_sharpe is None or new_final_sharpe >= MIN_FINAL_SHARPE,
        "final_net_bps_gate": new_final_avg_net_bps is None or new_final_avg_net_bps >= MIN_FINAL_AVG_NET_BPS,
    }

    # Comparison gate: new model should not be drastically worse
    if old_config:
        old_sharpe = result.get("old_sharpe", 0)
        if old_sharpe > 0:
            gates["comparison_gate"] = new_sharpe >= old_sharpe * DECAY_TOLERANCE
        else:
            gates["comparison_gate"] = True
    else:
        gates["comparison_gate"] = True

    result["gates"] = gates
    all_pass = all(gates.values())

    if not all_pass:
        failed = [k for k, v in gates.items() if not v]
        logger.warning(
            "Retrain validation FAILED for %s: %s", symbol, failed
        )
        result["error"] = f"failed gates: {failed}"

        # Restore backup if we wrote new files
        if not dry_run and "backup_dir" in result:
            backup_dir = Path(result["backup_dir"])
            if backup_dir.exists():
                shutil.rmtree(model_dir)
                shutil.copytree(backup_dir, model_dir)
                logger.info("Restored backup for %s", symbol)
        return result

    result["success"] = True

    # ── Sign model artifacts (HMAC-SHA256) ──
    if not dry_run:
        from infra.model_signing import sign_model_dir
        n_signed = sign_model_dir(model_dir)
        result["files_signed"] = n_signed

    # ── Task A: Adaptive ensemble weight calibration ──
    if not dry_run:
        cal_weights = calibrate_ensemble_weights(model_dir, shrinkage=0.3)
        if cal_weights:
            result["ensemble_weights"] = cal_weights

    # ── Task B: Save experiment tracking metadata ──
    if not dry_run:
        save_experiment_metadata(
            model_dir, symbol, horizons, old_config,
            retrain_trigger=retrain_trigger,
        )

    # Register with ModelRegistry so metadata stays in sync
    if not dry_run:
        try:
            from research.model_registry.registry import ModelRegistry

            registry_db = Path("data") / "model_registry.db"
            registry = ModelRegistry(registry_db)

            mv = registry.register(
                name=f"alpha_{symbol.lower()}",
                params=new_config,
                features=list(new_config.get("features", [])),
                metrics={k: float(v) for k, v in new_metrics.items() if isinstance(v, (int, float))},
                tags=["auto_retrain", "v11"],
            )
            registry.promote(mv.model_id, reason="auto_retrain", actor="auto_retrain")
            logger.info("Registered and promoted model %s for %s", mv.model_id, symbol)
        except Exception as e:
            logger.warning("Registry integration failed (non-blocking): %s", e)

    if dry_run:
        logger.info(
            "DRY RUN %s: would deploy (Sharpe %.2f→%.2f, IC %.4f→%.4f)",
            symbol,
            result.get("old_sharpe", 0), new_sharpe,
            result.get("old_avg_ic", 0), new_avg_ic,
        )
    else:
        logger.info(
            "DEPLOYED %s: Sharpe %.2f→%.2f, IC %.4f→%.4f, %d trades in %.0fs",
            symbol,
            result.get("old_sharpe", 0), new_sharpe,
            result.get("old_avg_ic", 0), new_avg_ic,
            new_trades, train_time,
        )

    return result


def log_retrain_event(result: Dict[str, Any]) -> None:
    """Append retrain result to JSONL log."""
    RETRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RETRAIN_LOG, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


def send_sighup_to_runner() -> bool:
    """Send SIGHUP to the live runner process to trigger model hot-reload.

    Finds the runner PID from the pidfile or by process name match.
    Returns True if signal was sent successfully.
    """
    import os
    import signal

    # Try pidfile first
    pidfile = Path("data/live/runner.pid")
    if pidfile.exists():
        try:
            pid = int(pidfile.read_text().strip())
            os.kill(pid, signal.SIGHUP)
            logger.info("Sent SIGHUP to runner (pid=%d from pidfile)", pid)
            return True
        except (ValueError, ProcessLookupError, PermissionError) as e:
            logger.warning("Pidfile SIGHUP failed: %s", e)

    # Fallback: find runner by process name
    try:
        import subprocess
        result = subprocess.run(
            ["pgrep", "-f", "live_runner"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                pid = int(line.strip())
                if pid != os.getpid():  # Don't signal ourselves
                    os.kill(pid, signal.SIGHUP)
                    logger.info("Sent SIGHUP to runner (pid=%d from pgrep)", pid)
                    return True
    except Exception as e:
        logger.warning("pgrep SIGHUP fallback failed: %s", e)

    logger.warning("No runner process found for SIGHUP")
    return False


def send_alert(message: str, *, severity: str = "info") -> None:
    """Send alert via unified notification system (Telegram + console)."""
    try:
        from monitoring.notify import send_alert as _send, AlertLevel
        level_map = {"info": AlertLevel.INFO, "warning": AlertLevel.WARNING,
                     "error": AlertLevel.CRITICAL, "critical": AlertLevel.CRITICAL}
        _send(level_map.get(severity, AlertLevel.INFO), message, source="auto_retrain")
    except ImportError:
        logger.info("ALERT [%s]: %s", severity, message)


def cleanup_old_backups(symbol: str, keep: int = 3) -> None:
    """Keep only the N most recent backups for a symbol."""
    model_dir = _model_dir_for(symbol)
    parent = model_dir.parent
    pattern = f"{model_dir.name}_backup_*"

    backups = sorted(parent.glob(pattern), key=lambda p: p.stat().st_mtime)
    if len(backups) > keep:
        for old in backups[:-keep]:
            shutil.rmtree(old)
            logger.info("Cleaned up old backup: %s", old)


# Re-export for backward compatibility
from alpha.auto_retrain_multi_tf import download_15m_data, retrain_15m_symbols, retrain_4h_symbols  # noqa: F401, E402


def main():
    from alpha.auto_retrain_main import main as _main
    return _main()


if __name__ == "__main__":
    sys.exit(main())
