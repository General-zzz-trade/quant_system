#!/usr/bin/env python3
"""Train/validate all production models.

Orchestrates training for both 1h and 15m alphas across all production symbols.
Optionally performs walk-forward validation or forces retraining.

Usage:
    python3 -m scripts.training.train_all_production --dry-run   # validate only (config check)
    python3 -m scripts.training.train_all_production --force      # force retrain all
    python3 -m scripts.training.train_all_production              # retrain if needed
    python3 -m scripts.training.train_all_production --symbol ETHUSDT  # single symbol only
"""
from __future__ import annotations

import json
import sys
import argparse
import logging
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "/quant_system")

from scripts.ops.config import SYMBOL_CONFIG, MODEL_BASE
from scripts.ops.auto_retrain import (
    check_needs_retrain,
    retrain_symbol,
    retrain_15m_symbols,
    log_retrain_event,
    send_alert,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────


def check_model_status(key: str, cfg: dict) -> dict:
    """Check if model dir has valid config.json. Returns status dict.

    Args:
        key: Symbol key (e.g., "BTCUSDT", "ETHUSDT_15m")
        cfg: SYMBOL_CONFIG entry

    Returns:
        Dict with keys: status, sharpe, train_date, model_dir
    """
    model_dir = MODEL_BASE / cfg["model_dir"]
    config_path = model_dir / "config.json"

    result = {
        "model_dir": cfg["model_dir"],
        "status": "MISSING",
        "sharpe": None,
        "train_date": None,
    }

    if not model_dir.exists():
        result["status"] = "MISSING"
        return result

    if not config_path.exists():
        result["status"] = "NO_CONFIG"
        return result

    try:
        d = json.loads(config_path.read_text())
        metrics = d.get("metrics", {})
        sharpe = metrics.get("sharpe") or d.get("sharpe")
        train_date = d.get("train_date", "unknown")

        result["status"] = "OK"
        result["sharpe"] = sharpe
        result["train_date"] = train_date
        return result
    except Exception as e:
        result["status"] = f"ERROR: {e}"
        return result


def separate_symbols() -> Tuple[List[str], List[str]]:
    """Separate SYMBOL_CONFIG into 1h and 15m symbol groups.

    Returns:
        (symbols_1h, symbols_15m) where each is a list of SYMBOL_CONFIG keys
    """
    symbols_1h = []
    symbols_15m = []

    for key, cfg in SYMBOL_CONFIG.items():
        interval = cfg.get("interval", "60")
        if interval == "15":
            symbols_15m.append(key)
        else:
            symbols_1h.append(key)

    return symbols_1h, symbols_15m


def print_status_table(statuses: Dict[str, dict]) -> None:
    """Print a formatted status table for all models.

    Args:
        statuses: Dict of symbol -> status dict (from check_model_status)
    """
    print("\n" + "=" * 90)
    print(f"{'Symbol':<15} | {'Model Dir':<25} | {'Status':<10} | {'Sharpe':<8} | {'Train Date':<19}")
    print("-" * 90)

    for symbol, status in statuses.items():
        sharpe_str = f"{status['sharpe']:.2f}" if status['sharpe'] is not None else "N/A"
        train_date = status['train_date'] or "N/A"
        print(
            f"{symbol:<15} | {status['model_dir']:<25} | {status['status']:<10} | "
            f"{sharpe_str:<8} | {train_date:<19}"
        )

    print("=" * 90 + "\n")


def dry_run_mode(symbols_to_check: List[str]) -> int:
    """Validate mode: check config files exist without retraining.

    Args:
        symbols_to_check: List of symbols to validate

    Returns:
        Exit code (0 = all OK, 1 = some missing)
    """
    logger.info("DRY RUN MODE: validating model configs exist")

    statuses = {}
    missing_count = 0

    # Check both 1h and 15m symbols
    all_symbols = list(SYMBOL_CONFIG.keys())
    if symbols_to_check:
        all_symbols = [s for s in all_symbols if s in symbols_to_check or any(
            s.startswith(base) for base in symbols_to_check
        )]

    for symbol in all_symbols:
        if symbol not in SYMBOL_CONFIG:
            continue
        status = check_model_status(symbol, SYMBOL_CONFIG[symbol])
        statuses[symbol] = status
        if status["status"] != "OK":
            missing_count += 1

    print_status_table(statuses)

    if missing_count == 0:
        logger.info("✓ All %d models have valid configs", len(statuses))
        return 0
    else:
        logger.warning("✗ %d models missing or invalid configs", missing_count)
        return 1


def retrain_mode(
    symbols_to_train: Optional[List[str]] = None,
    force: bool = False,
) -> int:
    """Retrain mode: check each symbol and retrain if needed.

    Args:
        symbols_to_train: Optional list of specific symbols to retrain. If None, retrain all.
        force: Force retraining even if model is healthy

    Returns:
        Exit code (0 = success, 1 = failures)
    """
    symbols_1h, symbols_15m = separate_symbols()

    # Filter to requested symbols if provided
    if symbols_to_train:
        symbols_1h = [s for s in symbols_1h if s in symbols_to_train]
        symbols_15m = [s for s in symbols_15m if s in symbols_to_train]

    logger.info("Retraining %d 1h + %d 15m symbols (force=%s)", len(symbols_1h), len(symbols_15m), force)

    results = {}
    failed_count = 0

    # ── 1h symbols ────────────────────────────────────────────
    for symbol in symbols_1h:
        needs_retrain, reason = check_needs_retrain(symbol, force=force)

        if not needs_retrain:
            logger.info("%s: skipping (model healthy, %s)", symbol, reason)
            results[symbol] = {"skipped": True, "reason": reason}
            continue

        logger.info("%s: retraining (%s)", symbol, reason)
        result = retrain_symbol(symbol, horizons=[12, 24], dry_run=False)
        results[symbol] = result

        if not result.get("success", False):
            failed_count += 1
            logger.error("%s: retrain FAILED - %s", symbol, result.get("error", "unknown"))
            send_alert(f"Retrain FAILED for {symbol}: {result.get('error')}", severity="error")
        else:
            logger.info("%s: retrain SUCCESS (Sharpe %.2f, IC %.4f)",
                       symbol, result.get("new_sharpe", 0), result.get("new_avg_ic", 0))
            send_alert(f"Retrain SUCCESS for {symbol}: Sharpe {result.get('new_sharpe', 0):.2f}")

        # Log to retrain history
        log_retrain_event(result)

    # ── 15m symbols ───────────────────────────────────────────
    if symbols_15m:
        logger.info("Retraining 15m models: %s", symbols_15m)
        # Extract base symbols for retrain_15m_symbols (e.g., "ETHUSDT" from "ETHUSDT_15m")
        base_symbols = [s.split("_")[0] for s in symbols_15m]
        result_15m = retrain_15m_symbols(base_symbols, dry_run=False, force=force)
        results.update(result_15m)

        for symbol, result in result_15m.items():
            if result.get("skipped"):
                logger.info("%s_15m: skipping (%s)", symbol, result.get("reason", "unknown"))
            elif not result.get("success", False):
                failed_count += 1
                logger.error("%s_15m: retrain FAILED - %s", symbol, result.get("error", "unknown"))
                send_alert(f"Retrain FAILED for {symbol}_15m: {result.get('error')}", severity="error")
            else:
                logger.info("%s_15m: retrain SUCCESS", symbol)
                send_alert(f"Retrain SUCCESS for {symbol}_15m")

    # ── Summary ────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("RETRAIN SUMMARY")
    logger.info("=" * 60)
    for symbol, result in results.items():
        if result.get("skipped"):
            status = "SKIPPED"
        elif result.get("success"):
            status = "OK"
        else:
            status = "FAILED"
        logger.info(f"{symbol:<20} {status:<10} {result.get('error', '')}")
    logger.info("=" * 60)

    if failed_count > 0:
        logger.error("Retrain completed with %d failures", failed_count)
        return 1

    logger.info("Retrain completed successfully")
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train/validate all production models"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs only (no retraining)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if models are healthy"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Train only this symbol (e.g., ETHUSDT, ETHUSDT_15m)"
    )

    args = parser.parse_args()

    logger.info("Starting train_all_production (dry_run=%s, force=%s, symbol=%s)",
               args.dry_run, args.force, args.symbol or "all")

    # Validate requested symbol exists
    symbols_to_process = None
    if args.symbol:
        if args.symbol not in SYMBOL_CONFIG:
            logger.error("Symbol '%s' not in SYMBOL_CONFIG", args.symbol)
            return 1
        symbols_to_process = [args.symbol]

    # ── Dry run mode: validate configs exist ────
    if args.dry_run:
        return dry_run_mode(symbols_to_process or list(SYMBOL_CONFIG.keys()))

    # ── Retrain mode: check and retrain ────
    return retrain_mode(symbols_to_process, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
