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
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


sys.path.insert(0, "/quant_system")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Configuration ──

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DEFAULT_HORIZONS = [12, 24]          # h48 dropped by default (negative IC)
MODEL_DIR_TEMPLATE = "models_v8/{symbol}_gate_v2"
DATA_DIR_TEMPLATE = "data_files/{symbol}_1h.csv"
RETRAIN_LOG = Path("logs/retrain_history.jsonl")

# ── 15m Configuration ──
SYMBOLS_15M = ["ETHUSDT", "SOLUSDT"]
DEFAULT_HORIZONS_15M = {
    "ETHUSDT": [4, 8],       # 1h, 2h — high frequency
    "SOLUSDT": [4, 8, 16],   # 1h, 2h, 4h
}
MODEL_DIR_15M_TEMPLATE = "models_v8/{symbol}_15m"

# Validation thresholds
MIN_IC = 0.02                         # minimum IC for new model to deploy
MIN_SHARPE = 1.0                      # minimum Sharpe for new model
DECAY_TOLERANCE = 0.7                 # new Sharpe >= old × this (30% decay OK)
MIN_TRADES = 15                       # minimum OOS trades
BOOTSTRAP_P5_MIN = 0.0               # bootstrap p5 must be positive
MIN_FINAL_SHARPE = 0.5               # final fold Sharpe must be > this
MIN_FINAL_AVG_NET_BPS = 2.0          # final fold avg net bps must be > this


def load_current_config(symbol: str) -> Optional[Dict[str, Any]]:
    """Load the current model's config.json."""
    config_path = Path(MODEL_DIR_TEMPLATE.format(symbol=symbol)) / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return json.load(f)


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
        except ValueError:
            pass

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
) -> Dict[str, Any]:
    """Retrain a symbol and validate the new model.

    Returns a result dict with success status and metrics.
    """
    from scripts.train_v11 import train_symbol_v11

    result = {
        "symbol": symbol,
        "horizons": horizons,
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "dry_run": dry_run,
    }

    model_dir = Path(MODEL_DIR_TEMPLATE.format(symbol=symbol))
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
        # Restore ic_weighted ensemble (train_v11 defaults to mean_zscore)
        if success:
            cfg_path = model_dir / "config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = json.load(f)
                cfg["ensemble_method"] = "ic_weighted"
                cfg["ic_ema_span"] = 720
                cfg["ic_min_threshold"] = -0.01
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
    """Send alert via configured channels (Telegram, webhook, or log-only).

    Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from env vars.
    Falls back to structured logging if no alert sink configured.
    """
    import os

    logger.info("ALERT [%s]: %s", severity, message)

    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if tg_token and tg_chat:
        try:
            import urllib.request
            import urllib.parse
            url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": tg_chat,
                "text": f"[auto_retrain] [{severity.upper()}] {message}",
                "parse_mode": "HTML",
            }).encode()
            req = urllib.request.Request(url, data=data, method="POST")
            urllib.request.urlopen(req, timeout=10)
            logger.info("Telegram alert sent")
        except Exception as e:
            logger.warning("Telegram alert failed: %s", e)

    webhook_url = os.environ.get("RETRAIN_WEBHOOK_URL", "")
    if webhook_url:
        try:
            import urllib.request
            data = json.dumps({
                "text": f"[auto_retrain] [{severity}] {message}",
                "severity": severity,
            }).encode()
            req = urllib.request.Request(
                webhook_url, data=data, method="POST",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.warning("Webhook alert failed: %s", e)


def cleanup_old_backups(symbol: str, keep: int = 3) -> None:
    """Keep only the N most recent backups for a symbol."""
    model_dir = Path(MODEL_DIR_TEMPLATE.format(symbol=symbol))
    parent = model_dir.parent
    pattern = f"{model_dir.name}_backup_*"

    backups = sorted(parent.glob(pattern), key=lambda p: p.stat().st_mtime)
    if len(backups) > keep:
        for old in backups[:-keep]:
            shutil.rmtree(old)
            logger.info("Cleaned up old backup: %s", old)


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
                    except ValueError:
                        pass

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
            logger.info("Backed up 15m %s → %s", model_dir, backup_dir)

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


def main():
    parser = argparse.ArgumentParser(description="Automated Walk-Forward Retraining")
    parser.add_argument("--symbol", default=None,
                        help="Comma-separated symbols (default: all)")
    parser.add_argument("--horizons", default="12,24",
                        help="Comma-separated horizons (default: 12,24 — h48 dropped)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate but don't replace models")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if model is healthy")
    parser.add_argument("--max-age-days", type=int, default=90,
                        help="Max model age before forced retrain (default: 90)")
    parser.add_argument("--notify-runner", action="store_true",
                        help="Send SIGHUP to runner after successful retrain")
    parser.add_argument("--alert", action="store_true",
                        help="Send alerts on success/failure (Telegram/webhook)")
    parser.add_argument("--include-15m", action="store_true",
                        help="Also retrain 15m models (ETHUSDT, SOLUSDT)")
    parser.add_argument("--only-15m", action="store_true",
                        help="Only retrain 15m models (skip 1h)")
    parser.add_argument("--symbols-15m", default=None,
                        help="Comma-separated 15m symbols (default: ETHUSDT,SOLUSDT)")
    args = parser.parse_args()

    symbols = (
        [s.strip().upper() for s in args.symbol.split(",")]
        if args.symbol
        else SYMBOLS
    )
    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    # Determine 15m symbols
    symbols_15m = (
        [s.strip().upper() for s in args.symbols_15m.split(",")]
        if args.symbols_15m
        else SYMBOLS_15M
    )
    run_1h = not args.only_15m
    run_15m = args.include_15m or args.only_15m

    print("=" * 70)
    print("  AUTOMATED WALK-FORWARD RETRAINING")
    print(f"  Date:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if run_1h:
        print(f"  1h Symbols:  {symbols}")
        print(f"  1h Horizons: {horizons}")
    if run_15m:
        print(f"  15m Symbols: {symbols_15m}")
    print(f"  Dry run:  {args.dry_run}")
    print(f"  Force:    {args.force}")
    print("=" * 70)

    results = {}

    # ── 15m: Download fresh data first ──
    if run_15m:
        print(f"\n{'=' * 70}")
        print("  15m DATA DOWNLOAD")
        print(f"{'=' * 70}")
        dl_results = download_15m_data(symbols_15m)
        for sym, n in dl_results.items():
            if n >= 0:
                print(f"  {sym}: {n} new bars")
            else:
                print(f"  {sym}: download FAILED")

    # ── 1h Retrain ──
    if not run_1h:
        print("\n  Skipping 1h retrain (--only-15m)")

    for symbol in (symbols if run_1h else []):
        print(f"\n{'─' * 70}")
        print(f"  {symbol}")
        print(f"{'─' * 70}")

        # Check if retrain needed
        needs, reason = check_needs_retrain(
            symbol, force=args.force, max_age_days=args.max_age_days
        )
        if not needs:
            print(f"  SKIP: {reason}")
            results[symbol] = {"symbol": symbol, "skipped": True, "reason": reason}
            continue

        print(f"  Retrain needed: {reason}")
        result = retrain_symbol(symbol, horizons=horizons, dry_run=args.dry_run)
        results[symbol] = result

        # Log event
        log_retrain_event(result)

        # Cleanup old backups
        if result.get("success") and not args.dry_run:
            cleanup_old_backups(symbol, keep=3)

    # ── 15m Retrain ──
    results_15m = {}
    if run_15m:
        print(f"\n{'=' * 70}")
        print("  15m MODEL RETRAIN")
        print(f"{'=' * 70}")
        results_15m = retrain_15m_symbols(
            symbols_15m,
            dry_run=args.dry_run,
            force=args.force,
            max_age_days=args.max_age_days,
        )

    # Summary
    print(f"\n{'=' * 70}")
    print("  RETRAIN SUMMARY")
    print(f"{'=' * 70}")

    succeeded = []
    for symbol, result in results.items():
        if result.get("skipped"):
            print(f"  {symbol}: SKIPPED ({result.get('reason', '')})")
        elif result.get("success"):
            succeeded.append(symbol)
            print(
                f"  {symbol}: SUCCESS "
                f"(Sharpe {result.get('old_sharpe', 0):.2f}→{result.get('new_sharpe', 0):.2f}, "
                f"IC {result.get('old_avg_ic', 0):.4f}→{result.get('new_avg_ic', 0):.4f})"
            )
        else:
            print(f"  {symbol}: FAILED ({result.get('error', 'unknown')})")

    # 15m summary
    if results_15m:
        print("\n  -- 15m Models --")
        for symbol, result in results_15m.items():
            if result.get("skipped"):
                print(f"  {symbol} (15m): SKIPPED ({result.get('reason', '')})")
            elif result.get("success"):
                succeeded.append(f"{symbol}_15m")
                print(
                    f"  {symbol} (15m): SUCCESS "
                    f"(Sharpe {result.get('new_sharpe', 0):.2f}, "
                    f"IC {result.get('new_avg_ic', 0):.4f})"
                )
            else:
                print(f"  {symbol} (15m): FAILED ({result.get('error', 'unknown')})")

    # CRITICAL: Parity check MUST run before SIGHUP to prevent live divergence.
    # If parity fails, restore backup and skip hot-reload.
    parity_ok = True
    if succeeded and not args.dry_run:
        logger.info("Running pre-deploy parity check...")
        try:
            import subprocess
            parity_result = subprocess.run(
                [
                    sys.executable, "-m", "pytest",
                    "tests/integration/test_live_backtest_parity.py",
                    "-x", "-q", "--tb=short",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd="/quant_system",
            )
            if parity_result.returncode != 0:
                parity_ok = False
                logger.warning(
                    "Pre-deploy parity check FAILED:\n%s",
                    parity_result.stdout + parity_result.stderr,
                )
                # Restore backups for all succeeded symbols
                for sym in succeeded:
                    r = results[sym]
                    backup_dir = r.get("backup_dir")
                    if backup_dir:
                        backup_path = Path(backup_dir)
                        model_dir = Path(MODEL_DIR_TEMPLATE.format(symbol=sym))
                        if backup_path.exists():
                            shutil.rmtree(model_dir)
                            shutil.copytree(backup_path, model_dir)
                            logger.info("Restored backup for %s (parity failed)", sym)
                    r["success"] = False
                    r["error"] = "parity check failed — backup restored"
                if args.alert:
                    send_alert(
                        f"Pre-deploy parity check FAILED for {succeeded} — backups restored, SIGHUP skipped",
                        severity="warning",
                    )
                succeeded.clear()
            else:
                logger.info("Pre-deploy parity check PASSED")
        except Exception as e:
            logger.warning("Pre-deploy parity check error: %s", e)

    # Send SIGHUP to runner only after parity check passes
    if args.notify_runner and succeeded and not args.dry_run and parity_ok:
        sighup_ok = send_sighup_to_runner()
        if sighup_ok:
            print("  → SIGHUP sent to runner for model hot-reload")
        else:
            print("  → WARNING: SIGHUP failed — manual model reload required")

    # Send alerts — include both 1h and 15m failures
    all_results = dict(results)
    for sym, r in results_15m.items():
        all_results[f"{sym}_15m"] = r

    failed = [s for s, r in all_results.items()
              if not r.get("success") and not r.get("skipped")]

    if args.alert:
        if succeeded:
            summary_parts = []
            for sym in succeeded:
                r = all_results.get(sym, {})
                summary_parts.append(
                    f"{sym}: Sharpe {r.get('new_sharpe', 0):.2f}"
                )
            send_alert(
                f"Retrain SUCCESS: {', '.join(summary_parts)}",
                severity="info",
            )
        if failed:
            fail_parts = []
            for sym in failed:
                r = all_results.get(sym, {})
                fail_parts.append(f"{sym}: {r.get('error', 'unknown')}")
            send_alert(
                f"Retrain FAILED: {', '.join(fail_parts)}",
                severity="error",
            )

    # Return exit code
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
