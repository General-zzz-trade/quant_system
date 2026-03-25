"""CLI and orchestration for auto_retrain.

Extracted from alpha/auto_retrain.py to keep file sizes manageable.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

from alpha.retrain.pipeline import (
    SYMBOLS, SYMBOLS_15M, SYMBOLS_4H,
    MODEL_DIR_TEMPLATE,
    check_data_freshness, check_needs_retrain,
    _daily_retrain_needed, _daily_ic_gate,
    load_current_config, _model_dir_for,
    retrain_symbol, log_retrain_event,
    cleanup_old_backups, send_sighup_to_runner, send_alert,
    download_15m_data, retrain_15m_symbols, retrain_4h_symbols,
    calibrate_ensemble_weights, save_experiment_metadata,  # noqa: F401
)

logger = logging.getLogger(__name__)


def _retrain_1h_symbols(symbols, horizons, args, retrain_mode):
    """Run 1h retrain loop. Returns results dict."""
    results: Dict[str, dict] = {}
    for symbol in symbols:
        print(f"\n{'_' * 70}")
        print(f"  {symbol}")
        print(f"{'_' * 70}")

        if args.daily:
            daily_needed, daily_reason = _daily_retrain_needed(symbol)
            if not daily_needed and not args.force:
                print(f"  SKIP (daily): {daily_reason}")
                results[symbol] = {"symbol": symbol, "skipped": True, "reason": daily_reason,
                                   "retrain_mode": retrain_mode}
                continue
            print(f"  Daily retrain needed: {daily_reason}")

        if not args.daily:
            needs, reason = check_needs_retrain(
                symbol, force=args.force, max_age_days=args.max_age_days
            )
            if not needs:
                print(f"  SKIP: {reason}")
                results[symbol] = {"symbol": symbol, "skipped": True, "reason": reason}
                continue
            print(f"  Retrain needed: {reason}")

        fresh, fresh_msg = check_data_freshness(symbol)
        if not fresh:
            print(f"  SKIP: stale data -- {fresh_msg}")
            results[symbol] = {"symbol": symbol, "skipped": True, "reason": f"stale data: {fresh_msg}"}
            continue

        old_config_for_daily = load_current_config(symbol) if args.daily else None
        trigger = "scheduled" if not args.force else "manual"
        if args.daily:
            trigger = "scheduled"
        result = retrain_symbol(symbol, horizons=horizons, dry_run=args.dry_run,
                                retrain_trigger=trigger)
        result["retrain_mode"] = retrain_mode

        if args.daily and result.get("success") and not args.dry_run:
            new_config = load_current_config(symbol)
            if new_config is not None:
                ic_pass, ic_reason = _daily_ic_gate(old_config_for_daily, new_config)
                if not ic_pass:
                    logger.warning("Daily IC gate FAILED for %s: %s", symbol, ic_reason)
                    result["success"] = False
                    result["error"] = f"daily IC gate failed: {ic_reason}"
                    if "backup_dir" in result:
                        backup_path = Path(result["backup_dir"])
                        model_dir = _model_dir_for(symbol)
                        if backup_path.exists():
                            shutil.rmtree(model_dir)
                            shutil.copytree(backup_path, model_dir)
                            logger.info("Restored backup for %s (daily IC gate failed)", symbol)
                else:
                    logger.info("Daily IC gate PASSED for %s: %s", symbol, ic_reason)

        results[symbol] = result
        log_retrain_event(result)
        if result.get("success") and not args.dry_run:
            cleanup_old_backups(symbol, keep=3)

    return results


def _run_parity_check_and_sighup(results, results_4h, succeeded, args):
    """Run parity check and send SIGHUP if all pass."""
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
                capture_output=True, text=True, timeout=120, cwd="/quant_system",
            )
            if parity_result.returncode != 0:
                parity_ok = False
                logger.warning("Pre-deploy parity check FAILED:\n%s",
                               parity_result.stdout + parity_result.stderr)
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
                    r["error"] = "parity check failed -- backup restored"
                if args.alert:
                    send_alert(
                        f"Pre-deploy parity check FAILED for {succeeded} -- backups restored",
                        severity="warning",
                    )
                succeeded.clear()
            else:
                logger.info("Pre-deploy parity check PASSED")
        except Exception as e:
            logger.warning("Pre-deploy parity check error: %s", e)

    if args.notify_runner and succeeded and not args.dry_run and parity_ok:
        sighup_ok = send_sighup_to_runner()
        if sighup_ok:
            print("  -> SIGHUP sent to runner for model hot-reload")
        else:
            print("  -> WARNING: SIGHUP failed -- manual model reload required")

    return parity_ok


def _refresh_ic_health(results, results_4h, succeeded):
    """Refresh IC health status for retrained models."""
    try:
        from datetime import timezone as _tz

        ic_path = Path("data/runtime/ic_health.json")
        models_status = []
        for sym in list(results.keys()) + [f"{s}_4h" for s in results_4h.keys()]:
            r = results.get(sym) or results_4h.get(sym.replace("_4h", ""), {})
            if r.get("success"):
                model_map = {
                    "BTCUSDT": "BTCUSDT_gate_v2", "ETHUSDT": "ETHUSDT_gate_v2",
                    "BTCUSDT_4h": "BTCUSDT_4h", "ETHUSDT_4h": "ETHUSDT_4h",
                }
                model_name = model_map.get(sym, sym)
                models_status.append({
                    "model": model_name,
                    "overall_status": "GREEN",
                    "note": f"retrained {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                })
        if models_status:
            existing = []
            if ic_path.exists():
                try:
                    existing = json.loads(ic_path.read_text()).get("models", [])
                except Exception as exc:
                    logger.debug("IC health read failed: %s", exc)
            retrained_names = {m["model"] for m in models_status}
            merged = [m for m in existing if m.get("model") not in retrained_names]
            merged.extend(models_status)
            ic_path.write_text(json.dumps({
                "timestamp": datetime.now(_tz.utc).isoformat(),
                "models": merged,
            }, indent=2))
            logger.info("IC health refreshed to GREEN for %d retrained models",
                        len(models_status))
    except Exception as e:
        logger.warning("IC health refresh failed (non-blocking): %s", e)


def main():
    """Main entry point for auto_retrain CLI."""
    parser = argparse.ArgumentParser(description="Automated Walk-Forward Retraining")
    parser.add_argument("--symbol", default=None, help="Comma-separated symbols")
    parser.add_argument("--horizons", default="12,24", help="Comma-separated horizons")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-age-days", type=int, default=90)
    parser.add_argument("--notify-runner", action="store_true")
    parser.add_argument("--alert", action="store_true")
    parser.add_argument("--include-15m", action="store_true")
    parser.add_argument("--only-15m", action="store_true")
    parser.add_argument("--symbols-15m", default=None)
    parser.add_argument("--include-4h", action="store_true")
    parser.add_argument("--only-4h", action="store_true")
    parser.add_argument("--symbols-4h", default=None)
    parser.add_argument("--daily", action="store_true")
    parser.add_argument("--sighup", action="store_true",
                        help="Alias for --notify-runner")
    args = parser.parse_args()

    # --sighup is an alias for --notify-runner
    if args.sighup:
        args.notify_runner = True

    symbols = ([s.strip().upper() for s in args.symbol.split(",")]
               if args.symbol else SYMBOLS)
    horizons = [int(h.strip()) for h in args.horizons.split(",")]
    symbols_15m = ([s.strip().upper() for s in args.symbols_15m.split(",")]
                   if args.symbols_15m else SYMBOLS_15M)
    symbols_4h = ([s.strip().upper() for s in args.symbols_4h.split(",")]
                  if args.symbols_4h else SYMBOLS_4H)

    run_1h = not args.only_15m and not args.only_4h
    run_15m = args.include_15m or args.only_15m
    run_4h = args.include_4h or args.only_4h
    retrain_mode = "daily_retrain" if args.daily else "weekly_retrain"

    print("=" * 70)
    print(f"  AUTOMATED WALK-FORWARD RETRAINING ({retrain_mode})")
    print(f"  Date:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if run_1h:
        print(f"  1h Symbols:  {symbols}")
        print(f"  1h Horizons: {horizons}")
    if run_15m:
        print(f"  15m Symbols: {symbols_15m}")
    if run_4h:
        print(f"  4h Symbols:  {symbols_4h}")
    print(f"  Dry run:  {args.dry_run}")
    print(f"  Force:    {args.force}")
    print("=" * 70)

    # Download 15m data
    if run_15m:
        print(f"\n{'=' * 70}\n  15m DATA DOWNLOAD\n{'=' * 70}")
        dl_results = download_15m_data(symbols_15m)
        for sym, n in dl_results.items():
            print(f"  {sym}: {n} new bars" if n >= 0 else f"  {sym}: download FAILED")

    # 1h retrain
    if run_1h:
        results = _retrain_1h_symbols(symbols, horizons, args, retrain_mode)
    else:
        print("\n  Skipping 1h retrain (--only-15m or --only-4h)")
        results = {}

    # 15m retrain
    results_15m = {}
    if run_15m:
        print(f"\n{'=' * 70}\n  15m MODEL RETRAIN\n{'=' * 70}")
        results_15m = retrain_15m_symbols(symbols_15m, dry_run=args.dry_run,
                                           force=args.force, max_age_days=args.max_age_days)

    # 4h retrain
    results_4h = {}
    if run_4h:
        print(f"\n{'=' * 70}\n  4h MODEL RETRAIN\n{'=' * 70}")
        results_4h = retrain_4h_symbols(symbols_4h, dry_run=args.dry_run,
                                         force=args.force, max_age_days=args.max_age_days)

    # Summary
    print(f"\n{'=' * 70}\n  RETRAIN SUMMARY\n{'=' * 70}")
    succeeded = []
    for symbol, result in results.items():
        if result.get("skipped"):
            print(f"  {symbol}: SKIPPED ({result.get('reason', '')})")
        elif result.get("success"):
            succeeded.append(symbol)
            print(f"  {symbol}: SUCCESS (Sharpe {result.get('old_sharpe', 0):.2f}->"
                  f"{result.get('new_sharpe', 0):.2f}, IC {result.get('old_avg_ic', 0):.4f}->"
                  f"{result.get('new_avg_ic', 0):.4f})")
        else:
            print(f"  {symbol}: FAILED ({result.get('error', 'unknown')})")

    for label, res_map in [("15m", results_15m), ("4h", results_4h)]:
        if res_map:
            print(f"\n  -- {label} Models --")
            for symbol, result in res_map.items():
                if result.get("skipped"):
                    print(f"  {symbol} ({label}): SKIPPED ({result.get('reason', '')})")
                elif result.get("success"):
                    succeeded.append(f"{symbol}_{label}")
                    print(f"  {symbol} ({label}): SUCCESS (Sharpe {result.get('new_sharpe', 0):.2f}, "
                          f"IC {result.get('new_avg_ic', 0):.4f})")
                else:
                    print(f"  {symbol} ({label}): FAILED ({result.get('error', 'unknown')})")

    _run_parity_check_and_sighup(results, results_4h, succeeded, args)

    if succeeded and not args.dry_run:
        _refresh_ic_health(results, results_4h, succeeded)

    # Alerts
    all_results = dict(results)
    for sym, r in results_15m.items():
        all_results[f"{sym}_15m"] = r
    for sym, r in results_4h.items():
        all_results[f"{sym}_4h"] = r
    failed = [s for s, r in all_results.items()
              if not r.get("success") and not r.get("skipped")]

    if args.alert:
        if succeeded:
            parts = [f"{sym}: Sharpe {all_results.get(sym, {}).get('new_sharpe', 0):.2f}"
                     for sym in succeeded]
            send_alert(f"Retrain SUCCESS: {', '.join(parts)}", severity="info")
        if failed:
            parts = [f"{sym}: {all_results.get(sym, {}).get('error', 'unknown')}"
                     for sym in failed]
            send_alert(f"Retrain FAILED: {', '.join(parts)}", severity="error")

    return 1 if failed else 0
