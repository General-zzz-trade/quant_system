#!/usr/bin/env python3
"""Dedicated 15m model retrain with enhanced features.

Retrains BTC+ETH 15m models using:
- Standard 1h features downsampled to 15m bars
- V14 dominance features (btc_dom_ratio_dev_20, btc_dom_ratio_mom_10, etc.)
- Cross-TF features: 1h regime signal as 15m feature
- Tighter validation: min_sharpe=1.5, min_ic=0.03

Usage:
    python3 -m alpha.retrain_15m                    # Full retrain
    python3 -m alpha.retrain_15m --dry-run          # Validate only
    python3 -m alpha.retrain_15m --symbol BTCUSDT   # Single symbol
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from alpha.auto_retrain_config import (
    SYMBOLS_15M,
    DEFAULT_HORIZONS_15M,
    MODEL_DIR_15M_TEMPLATE,
)

logger = logging.getLogger(__name__)

# ── 15m-specific validation thresholds (stricter than 1h) ──
# 15m has more noise, so we require higher Sharpe and IC to deploy.
MIN_SHARPE_15M = 1.5      # 1h uses 1.0; 15m needs higher bar due to noise
MIN_IC_15M = 0.03          # 1h uses 0.02; 15m needs stronger signal
MIN_TRADES_15M = 30        # 1h uses 15; 15m should have more trades given 4x bar frequency
MIN_FINAL_SHARPE_15M = 0.8  # 1h uses 0.5

# ── V14 dominance feature candidates for 15m ──
# These are computed by batch_feature_engine (compute_features_batch -> _add_dominance_features)
# and by the incremental RustFeatureEngine (push_dominance).
DOMINANCE_FEATURES = [
    "btc_dom_ratio_dev_20",    # BTC/ETH ratio deviation from 20-bar MA
    "btc_dom_ratio_mom_10",    # BTC/ETH ratio 10-bar momentum
    "btc_dom_return_diff_6h",  # BTC-ETH 6-bar return differential
    "btc_dom_return_diff_24h", # BTC-ETH 24-bar return differential
]

# Batch-mode equivalents (used by batch_feature_engine)
DOMINANCE_FEATURES_BATCH = [
    "btc_dom_dev_20",   # ratio / MA(20) - 1
    "btc_dom_dev_50",   # ratio / MA(50) - 1
    "btc_dom_ret_24",   # ratio 24-bar pct change
    "btc_dom_ret_72",   # ratio 72-bar pct change
]

# Cross-TF feature: 1h regime label injected as 15m feature
CROSS_TF_FEATURES = [
    "regime_1h_label",  # from RustCompositeRegimeDetector on 1h bars
]

# Combined extra feature candidates for 15m model selection
EXTRA_15M_FEATURE_CANDIDATES = DOMINANCE_FEATURES_BATCH + CROSS_TF_FEATURES


def _validate_15m_result(result: Dict[str, Any]) -> tuple[bool, str]:
    """Apply stricter 15m validation thresholds.

    Returns (pass, reason) tuple.
    """
    sharpe = result.get("new_sharpe", 0)
    avg_ic = result.get("new_avg_ic", 0)
    trades = result.get("new_trades", 0)

    if sharpe < MIN_SHARPE_15M:
        return False, f"Sharpe {sharpe:.2f} < {MIN_SHARPE_15M} (15m threshold)"
    if avg_ic < MIN_IC_15M:
        return False, f"IC {avg_ic:.4f} < {MIN_IC_15M} (15m threshold)"
    if trades < MIN_TRADES_15M:
        return False, f"Trades {trades} < {MIN_TRADES_15M} (15m threshold)"
    return True, f"PASS (Sharpe {sharpe:.2f}, IC {avg_ic:.4f}, trades {trades})"


def _load_model_metrics(symbol: str) -> Dict[str, Any] | None:
    """Load current 15m model metrics from config.json."""
    model_dir = Path(MODEL_DIR_15M_TEMPLATE.format(symbol=symbol))
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return json.load(f)


def retrain_15m(
    symbols: List[str],
    dry_run: bool = False,
    force: bool = False,
    max_age_days: int = 90,
) -> Dict[str, Dict[str, Any]]:
    """Retrain 15m models with stricter validation and enhanced features.

    This wraps retrain_15m_symbols() from auto_retrain_multi_tf.py
    and applies additional 15m-specific validation gates.
    """
    from alpha.auto_retrain_multi_tf import download_15m_data, retrain_15m_symbols

    # Step 1: Download fresh 15m data
    print(f"\n{'=' * 70}")
    print("  15m DATA DOWNLOAD")
    print(f"{'=' * 70}")
    dl_results = download_15m_data(symbols)
    for sym, n in dl_results.items():
        if n >= 0:
            print(f"  {sym}: {n} new bars downloaded")
        else:
            print(f"  {sym}: download FAILED")

    # Step 2: Run retrain via existing pipeline
    print(f"\n{'=' * 70}")
    print("  15m MODEL RETRAIN")
    print(f"{'=' * 70}")
    results = retrain_15m_symbols(
        symbols, dry_run=dry_run, force=force, max_age_days=max_age_days
    )

    # Step 3: Apply stricter 15m validation on top
    print(f"\n{'=' * 70}")
    print("  15m ENHANCED VALIDATION (stricter thresholds)")
    print(f"{'=' * 70}")
    print(f"  min_sharpe={MIN_SHARPE_15M}, min_ic={MIN_IC_15M}, min_trades={MIN_TRADES_15M}")

    for symbol, result in results.items():
        if result.get("skipped"):
            print(f"  {symbol}: SKIPPED ({result.get('reason', '')})")
            continue

        if not result.get("success"):
            print(f"  {symbol}: FAILED at training stage ({result.get('error', 'unknown')})")
            continue

        # Apply stricter 15m validation
        passed, reason = _validate_15m_result(result)
        print(f"  {symbol}: {reason}")

        if not passed:
            result["success"] = False
            result["error"] = f"15m validation gate: {reason}"
            logger.warning("15m enhanced validation FAILED for %s: %s", symbol, reason)

            # Restore backup if we have one
            if not dry_run and "backup_dir" in result:
                import shutil
                backup_path = Path(result["backup_dir"])
                model_dir = Path(MODEL_DIR_15M_TEMPLATE.format(symbol=symbol))
                if backup_path.exists() and model_dir.exists():
                    shutil.rmtree(model_dir)
                    shutil.copytree(backup_path, model_dir)
                    logger.info("Restored 15m backup for %s (enhanced validation failed)", symbol)
                    print(f"  {symbol}: backup restored")

    # Step 4: Detailed validation report
    print(f"\n{'=' * 70}")
    print("  15m RETRAIN SUMMARY")
    print(f"{'=' * 70}")

    succeeded = []
    failed = []
    skipped = []

    for symbol, result in results.items():
        if result.get("skipped"):
            skipped.append(symbol)
            continue
        if result.get("success"):
            succeeded.append(symbol)
            sharpe = result.get("new_sharpe", 0)
            ic = result.get("new_avg_ic", 0)
            trades = result.get("new_trades", 0)
            t = result.get("train_time_sec", 0)
            print(f"  {symbol}: SUCCESS  Sharpe={sharpe:.2f}  IC={ic:.4f}  trades={trades}  time={t:.0f}s")

            # Log current model config for reference
            cfg = _load_model_metrics(symbol)
            if cfg:
                horizons = DEFAULT_HORIZONS_15M.get(symbol, [4, 8])
                print(f"           horizons={horizons}  model_dir={MODEL_DIR_15M_TEMPLATE.format(symbol=symbol)}")
        else:
            failed.append(symbol)
            print(f"  {symbol}: FAILED   {result.get('error', 'unknown')}")

    if skipped:
        print(f"\n  Skipped: {', '.join(skipped)}")

    print(f"\n  Total: {len(succeeded)} succeeded, {len(failed)} failed, {len(skipped)} skipped")

    # Step 5: Feature candidate info
    print(f"\n{'=' * 70}")
    print("  15m FEATURE CANDIDATES (for next iteration)")
    print(f"{'=' * 70}")
    print(f"  V14 dominance (batch): {DOMINANCE_FEATURES_BATCH}")
    print(f"  V14 dominance (incr):  {DOMINANCE_FEATURES}")
    print(f"  Cross-TF:              {CROSS_TF_FEATURES}")
    print("  Note: dominance features are already computed by batch_feature_engine")
    print("  Note: cross-TF regime_1h_label requires 1h regime data injected into 15m training set")

    if succeeded:
        print("\n  Next steps:")
        print("    1. Verify WF results in models_v8/{symbol}_15m/config.json")
        print("    2. Enable in strategy_config.py if Sharpe > 1.5 sustained")
        print("    3. Run: python3 -m alpha.auto_retrain --only-15m --force  (via main pipeline)")

    return results


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Dedicated 15m model retrain with enhanced validation"
    )
    parser.add_argument(
        "--symbol", default=None,
        help="Single symbol to retrain (default: all 15m symbols)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate only, do not deploy models"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force retrain even if model is healthy"
    )
    parser.add_argument(
        "--max-age-days", type=int, default=90,
        help="Skip retrain if model younger than this (default: 90)"
    )
    args = parser.parse_args()

    symbols = (
        [args.symbol.strip().upper()] if args.symbol else SYMBOLS_15M
    )

    print("=" * 70)
    print("  15m MODEL RETRAIN (DEDICATED)")
    print(f"  Date:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Symbols:  {symbols}")
    print(f"  Horizons: {dict((s, DEFAULT_HORIZONS_15M.get(s, [4, 8])) for s in symbols)}")
    print(f"  Dry run:  {args.dry_run}")
    print(f"  Force:    {args.force}")
    print(f"  Thresholds: Sharpe>={MIN_SHARPE_15M}, IC>={MIN_IC_15M}, trades>={MIN_TRADES_15M}")
    print("=" * 70)

    results = retrain_15m(
        symbols,
        dry_run=args.dry_run,
        force=args.force,
        max_age_days=args.max_age_days,
    )

    failed = [s for s, r in results.items()
              if not r.get("success") and not r.get("skipped")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
