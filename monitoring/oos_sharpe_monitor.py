"""OOS Sharpe Monitor — detect strategy decay via out-of-sample backtest Sharpe.

Runs a 3-month OOS backtest for each active model and checks the resulting
Sharpe ratio against safety thresholds:

  GREEN:  Sharpe >= 0.5
  YELLOW: Sharpe >= 0 (WARNING)
  RED:    Sharpe < 0  (CRITICAL)
  CRITICAL: Sharpe < -2

Usage:
    python3 -m monitoring.oos_sharpe_monitor              # Print status table
    python3 -m monitoring.oos_sharpe_monitor --alert       # + send Telegram alerts
    python3 -m monitoring.oos_sharpe_monitor --json        # JSON output only

Designed for weekly cron/systemd timer execution. Saves state to
data/runtime/oos_sharpe_health.json for dashboard integration.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_DIR = Path("models_v8")
OUTPUT_PATH = Path("data/runtime/oos_sharpe_health.json")
OOS_BARS = 2190  # ~3 months of 1h bars

# Active models: (dir_name, symbol)
ACTIVE_MODELS = [
    ("BTCUSDT_gate_v2", "BTCUSDT"),
    ("ETHUSDT_gate_v2", "ETHUSDT"),
]

# Sharpe thresholds
THRESHOLD_GREEN = 0.5
THRESHOLD_YELLOW = 0.0
THRESHOLD_CRITICAL = -2.0


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _classify_sharpe(sharpe: float) -> str:
    """Classify Sharpe ratio: GREEN, YELLOW, or RED."""
    if sharpe >= THRESHOLD_GREEN:
        return "GREEN"
    if sharpe >= THRESHOLD_YELLOW:
        return "YELLOW"
    return "RED"


def evaluate_model(model_name: str, symbol: str) -> Dict[str, Any]:
    """Run OOS backtest for a single model and return results."""
    model_dir = MODELS_DIR / model_name
    config_path = model_dir / "config.json"
    out_dir = Path("data/runtime/oos_backtest") / model_name

    if not model_dir.exists():
        return {"model": model_name, "symbol": symbol, "error": "model dir not found"}
    if not config_path.exists():
        return {"model": model_name, "symbol": symbol, "error": "config.json not found"}

    out_dir.mkdir(parents=True, exist_ok=True)

    # Allow unsigned models within this process
    os.environ["QUANT_ALLOW_UNSIGNED_MODELS"] = "1"

    try:
        from scripts.backtest_alpha_v8 import run_backtest

        summary = run_backtest(
            symbol=symbol,
            model_path=model_dir,
            config_path=config_path,
            out_dir=out_dir,
            long_only=True,
            oos_bars=OOS_BARS,
        )
    except Exception as e:
        logger.error("Backtest failed for %s: %s", model_name, e)
        return {"model": model_name, "symbol": symbol, "error": str(e)}

    if not summary:
        return {"model": model_name, "symbol": symbol, "error": "empty backtest result"}

    sharpe = summary.get("sharpe", float("nan"))
    status = _classify_sharpe(sharpe)

    return {
        "model": model_name,
        "symbol": symbol,
        "sharpe": round(sharpe, 3),
        "total_return": round(summary.get("total_return", 0.0), 4),
        "max_drawdown": round(summary.get("max_drawdown", 0.0), 4),
        "n_trades": summary.get("n_trades", 0),
        "n_bars": summary.get("n_bars", 0),
        "period_start": summary.get("period_start"),
        "period_end": summary.get("period_end"),
        "status": status,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def run_monitor() -> List[Dict[str, Any]]:
    """Evaluate all active models and return results."""
    results = []
    for model_name, symbol in ACTIVE_MODELS:
        logger.info("Running OOS backtest for %s (%s)...", model_name, symbol)
        result = evaluate_model(model_name, symbol)
        results.append(result)
    return results


def save_results(results: List[Dict[str, Any]]) -> None:
    """Save results to data/runtime/oos_sharpe_health.json."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Saved OOS Sharpe health to %s", OUTPUT_PATH)


# ---------------------------------------------------------------------------
# Display & alerting
# ---------------------------------------------------------------------------

def print_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted status table."""
    STATUS_COLORS = {"GREEN": "\033[92m", "YELLOW": "\033[93m", "RED": "\033[91m"}
    RESET = "\033[0m"

    print()
    print("=" * 80)
    print(f"{'OOS Sharpe Monitor':^80}")
    print(f"{'(' + datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC') + ')':^80}")
    print("=" * 80)
    print(
        f"{'Model':<22} {'Symbol':<10} {'Sharpe':>8} "
        f"{'Return':>9} {'MaxDD':>9} {'Trades':>7} {'Status':>8}"
    )
    print("-" * 80)

    for r in results:
        if "error" in r and "sharpe" not in r:
            print(f"{r['model']:<22} {r.get('symbol', ''):<10}  ERROR: {r['error']}")
            continue

        sharpe = r.get("sharpe", 0)
        total_ret = r.get("total_return", 0)
        max_dd = r.get("max_drawdown", 0)
        n_trades = r.get("n_trades", 0)
        status = r.get("status", "UNKNOWN")

        color = STATUS_COLORS.get(status, "")
        print(
            f"{r['model']:<22} {r.get('symbol', ''):<10} {sharpe:>8.3f} "
            f"{total_ret * 100:>8.2f}% {max_dd * 100:>8.2f}% {n_trades:>7} "
            f"{color}{status:>8}{RESET}"
        )

    print("=" * 80)

    statuses = [r.get("status") for r in results if "status" in r]
    n_green = statuses.count("GREEN")
    n_yellow = statuses.count("YELLOW")
    n_red = statuses.count("RED")
    print(
        f"\nSummary: {n_green} GREEN, {n_yellow} YELLOW, {n_red} RED "
        f"out of {len(statuses)} models"
    )
    print()


def send_alerts(results: List[Dict[str, Any]]) -> None:
    """Send Telegram alerts for YELLOW/RED models."""
    try:
        from monitoring.notify import send_alert, AlertLevel
    except ImportError:
        logger.warning("monitoring.notify not available — skipping alerts")
        return

    for r in results:
        status = r.get("status")
        if status not in ("YELLOW", "RED"):
            continue

        model = r.get("model", "unknown")
        symbol = r.get("symbol", "")
        sharpe = r.get("sharpe", 0)

        details: Dict[str, str] = {
            "symbol": symbol,
            "model": model,
            "sharpe": f"{sharpe:.3f}",
            "return": f"{r.get('total_return', 0) * 100:.2f}%",
            "max_dd": f"{r.get('max_drawdown', 0) * 100:.2f}%",
            "trades": str(r.get("n_trades", 0)),
        }

        if sharpe < THRESHOLD_CRITICAL:
            send_alert(
                AlertLevel.CRITICAL,
                f"OOS SHARPE CRITICAL: {model} Sharpe={sharpe:.2f}",
                details=details,
                source="oos_sharpe_monitor",
            )
        elif status == "RED":
            send_alert(
                AlertLevel.WARNING,
                f"OOS SHARPE WARNING: {model} Sharpe={sharpe:.2f} < 0",
                details=details,
                source="oos_sharpe_monitor",
            )
        else:
            send_alert(
                AlertLevel.WARNING,
                f"OOS SHARPE WEAK: {model} Sharpe={sharpe:.2f}",
                details=details,
                source="oos_sharpe_monitor",
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OOS Sharpe Monitor — detect strategy decay via backtest",
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

    statuses = [r.get("status") for r in results if "status" in r]
    if "RED" in statuses:
        sys.exit(2)
    elif "YELLOW" in statuses:
        sys.exit(1)


if __name__ == "__main__":
    main()
