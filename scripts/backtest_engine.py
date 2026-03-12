#!/usr/bin/env python3
"""Event-driven ML backtest using the real engine infrastructure.

Uses the same event-driven engine as production (EngineCoordinator, DecisionBridge,
EmbargoExecutionAdapter) with pluggable MLSignalDecisionModule.

Key properties:
  - OOS-only: skip training period + embargo gap
  - Next-bar execution: EmbargoExecutionAdapter fills at next bar's OPEN price
  - Realistic costs: taker fees + slippage applied per fill
  - Pre-committed params: model config.json drives deadzone/hold/leverage
  - Features computed with same batch engine as training (causal, no leakage)
  - Multi-symbol support via run_multi_backtest()

Usage:
    python3 -m scripts.backtest_engine
    python3 -m scripts.backtest_engine --symbols BTCUSDT
    python3 -m scripts.backtest_engine --symbols BTCUSDT,ETHUSDT --fee-bps 4
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from runner.backtest_runner import (
    run_backtest,
    run_multi_backtest,
    _build_trades_from_fills,
    _max_drawdown,
    EquityPoint,
)
from decision.backtest_module import MLSignalDecisionModule
from decision.precomputed_hook import PrecomputedFeatureHook


# ── Cost parameters (match production) ──
FEE_BPS = Decimal("4")         # 4bp taker per side
SLIPPAGE_BPS = Decimal("1")    # 1bp slippage per side
EMBARGO_BARS = 1               # Fill at next bar's OPEN (engine handles this)

# ── OOS config ──
OOS_MONTHS = 18                # Last 18 months = OOS period
EMBARGO_GAP = 48               # 48-bar gap between train and OOS boundary

# ── Portfolio ──
EQUITY = 10_000.0
RISK_FRACTION = 0.05           # 5% per position (split across coins)

# ── Dynamic leverage ──
LEV_DEFAULT = 2.0              # Fixed leverage for event-driven module

SYMBOLS_DEFAULT = ["BTCUSDT", "ETHUSDT"]

DATA_DIR = Path("/quant_system/data_files")
MODEL_DIR = Path("/quant_system/models_v8")


def prepare_oos_data(
    symbol: str,
    oos_months: int = OOS_MONTHS,
    embargo_gap: int = EMBARGO_GAP,
) -> Tuple[Optional[Path], Optional[PrecomputedFeatureHook], Dict[str, Any]]:
    """Prepare OOS CSV + precomputed features.

    Returns (csv_path, feature_hook, info_dict). csv_path is None if data/model missing.
    """
    data_path = DATA_DIR / f"{symbol}_1h.csv"
    model_dir = MODEL_DIR / f"{symbol}_gate_v2"

    if not data_path.exists():
        print(f"  SKIP {symbol}: no data at {data_path}")
        return None, None, {}
    if not (model_dir / "lgbm_v8.pkl").exists():
        print(f"  SKIP {symbol}: no model at {model_dir}")
        return None, None, {}

    df = pd.read_csv(data_path)
    n_total = len(df)

    # OOS boundary: last oos_months of data
    oos_bars = 24 * 30 * oos_months
    oos_start = max(0, n_total - oos_bars + embargo_gap)

    ts_col = "open_time" if "open_time" in df.columns else "timestamp"
    timestamps = df[ts_col].values

    oos_start_date = pd.Timestamp(timestamps[oos_start], unit="ms").strftime("%Y-%m-%d")
    oos_end_date = pd.Timestamp(timestamps[-1], unit="ms").strftime("%Y-%m-%d")

    info = {
        "n_total": n_total,
        "oos_start": oos_start,
        "oos_start_date": oos_start_date,
        "oos_end_date": oos_end_date,
        "oos_bars": n_total - oos_start,
    }

    # Pre-compute features on FULL data (features are causal, no leakage)
    # Then slice to OOS period for the hook
    print(f"  Computing features for {symbol}...", end=" ", flush=True)
    t0 = time.time()
    hook = PrecomputedFeatureHook.from_dataframe(symbol, df, include_4h=True)
    print(f"({time.time() - t0:.1f}s)")

    # Write OOS slice to temp CSV
    oos_df = df.iloc[oos_start:].copy()
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{symbol}_oos.csv", delete=False, prefix="bt_",
    )
    oos_df.to_csv(tmp.name, index=False)
    tmp.close()

    return Path(tmp.name), hook, info


def build_decision_module(
    symbol: str,
    equity_share: float,
    n_coins: int = 2,
    leverage: float = LEV_DEFAULT,
) -> Optional[MLSignalDecisionModule]:
    """Build MLSignalDecisionModule from model dir, using pre-committed config."""
    model_dir = MODEL_DIR / f"{symbol}_gate_v2"
    if not (model_dir / "lgbm_v8.pkl").exists():
        return None

    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    risk_per_coin = RISK_FRACTION / max(n_coins, 1)

    # For small accounts, scale up risk fraction so notional >= $5
    # (ensures meaningful position sizes even with $100 equity)
    min_notional = 5.0
    notional_check = equity_share * risk_per_coin * leverage
    if notional_check < min_notional and equity_share > 0:
        risk_per_coin = min_notional / (equity_share * leverage)
        risk_per_coin = min(risk_per_coin, 1.0)  # Cap at 100%

    return MLSignalDecisionModule(
        symbol=symbol,
        model_dir=model_dir,
        equity=equity_share,
        risk_fraction=risk_per_coin,
        deadzone=cfg.get("deadzone", 1.0),
        min_hold=cfg.get("min_hold", 24),
        max_hold=cfg.get("max_hold", 120),
        long_only=cfg.get("long_only", False),
        zscore_window=720,
        leverage=leverage,
    )


def run_single_symbol(
    symbol: str,
    equity: float = EQUITY,
    fee_bps: Decimal = FEE_BPS,
    slippage_bps: Decimal = SLIPPAGE_BPS,
    out_dir: Optional[Path] = None,
    n_coins: int = 1,
) -> Tuple[List[EquityPoint], List[Dict[str, Any]], Dict[str, Any]]:
    """Run single-symbol OOS backtest through the event-driven engine."""
    csv_path, feature_hook, info = prepare_oos_data(symbol)
    if csv_path is None:
        return [], [], {}

    module = build_decision_module(symbol, equity_share=equity, n_coins=n_coins)
    if module is None:
        return [], [], {}

    print(f"  {symbol}: {info['oos_bars']:,} OOS bars "
          f"({info['oos_start_date']} → {info['oos_end_date']})")

    # Funding CSV (if available)
    funding_csv_path = DATA_DIR / f"{symbol}_funding.csv"
    funding_csv = str(funding_csv_path) if funding_csv_path.exists() else None
    if funding_csv:
        print(f"  Funding: {funding_csv_path.name}")

    t0 = time.time()
    equity_curve, fills = run_backtest(
        csv_path=csv_path,
        symbol=symbol,
        starting_balance=Decimal(str(equity)),
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        embargo_bars=EMBARGO_BARS,
        decision_modules=[module],
        feature_hook=feature_hook,
        funding_csv=funding_csv,
        out_dir=out_dir,
    )
    elapsed = time.time() - t0

    # Clean up temp file
    try:
        csv_path.unlink()
    except OSError:
        pass

    trades = _build_trades_from_fills(fills)
    info["elapsed_sec"] = elapsed
    info["n_fills"] = len(fills)
    info["n_trades"] = len(trades)

    return equity_curve, fills, info


def analyze_results(
    equity: List[EquityPoint],
    fills: List[Dict[str, Any]],
    label: str,
    initial_equity: float,
) -> Dict[str, Any]:
    """Compute and print summary statistics."""
    if not equity:
        print(f"\n  [{label}] No equity points.")
        return {}

    trades = _build_trades_from_fills(fills)
    eq_values = [float(e.equity) for e in equity]

    start_eq = eq_values[0]
    end_eq = eq_values[-1]
    total_return = (end_eq - start_eq) / start_eq if start_eq > 0 else 0
    mdd = float(_max_drawdown([e.equity for e in equity]))

    # Sharpe from equity curve (hourly returns)
    eq_arr = np.array(eq_values)
    rets = np.diff(eq_arr) / eq_arr[:-1]
    rets = rets[np.isfinite(rets)]
    if len(rets) > 10:
        sharpe = float(np.mean(rets) / max(np.std(rets, ddof=1), 1e-12) * np.sqrt(8760))
    else:
        sharpe = 0.0

    # Trade stats
    n_trades = len(trades)
    wins = sum(1 for t in trades if float(t.get("net_pnl", 0)) > 0)
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0

    total_fees = sum(float(f.get("fee", 0) or 0) for f in fills)

    # Long/short breakdown
    n_longs = sum(1 for t in trades if t.get("side") == "long")
    n_shorts = sum(1 for t in trades if t.get("side") == "short")

    print(f"\n  [{label}]")
    print(f"  {'─' * 65}")
    print(f"  Bars: {len(equity):,}  Trades: {n_trades}  (L:{n_longs} S:{n_shorts})")
    print(f"  Win Rate:    {win_rate:.1f}%")
    print(f"  Return:      {total_return*100:+.2f}%  "
          f"(${end_eq - start_eq:+,.0f})")
    print(f"  Max DD:      {mdd*100:.2f}%")
    print(f"  Sharpe:      {sharpe:.2f}")
    print(f"  Total Fees:  ${total_fees:,.0f}")

    return {
        "label": label,
        "bars": len(equity),
        "trades": n_trades,
        "longs": n_longs,
        "shorts": n_shorts,
        "win_rate": win_rate,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_dd": mdd,
        "total_fees": total_fees,
        "start_equity": start_eq,
        "end_equity": end_eq,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Event-driven ML backtest (OOS)")
    parser.add_argument("--symbols", default=",".join(SYMBOLS_DEFAULT),
                        help="Comma-separated symbols")
    parser.add_argument("--equity", type=float, default=EQUITY)
    parser.add_argument("--fee-bps", type=int, default=4, help="Fee bps per side")
    parser.add_argument("--slippage-bps", type=int, default=1, help="Slippage bps per side")
    parser.add_argument("--out", default=None, help="Output directory")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    fee_bps = Decimal(str(args.fee_bps))
    slippage_bps = Decimal(str(args.slippage_bps))
    out_dir = Path(args.out) if args.out else None

    print("=" * 70)
    print("EVENT-DRIVEN ML BACKTEST (OOS Only)")
    print("=" * 70)
    print(f"  Engine:     EngineCoordinator + EmbargoExecutionAdapter")
    print(f"  Decision:   MLSignalDecisionModule (LGBM+XGB ensemble)")
    print(f"  Execution:  Next-bar OPEN fill (embargo_bars={EMBARGO_BARS})")
    print(f"  Costs:      {fee_bps}bp fee/side + {slippage_bps}bp slippage/side "
          f"= {(fee_bps + slippage_bps) * 2}bp RT")
    print(f"  OOS:        Last {OOS_MONTHS} months, {EMBARGO_GAP}-bar embargo gap")
    print(f"  Equity:     ${args.equity:,.0f}")
    print(f"  Symbols:    {', '.join(symbols)}")

    # Count valid coins for portfolio split
    n_coins = sum(
        1 for s in symbols
        if (DATA_DIR / f"{s}_1h.csv").exists()
        and (MODEL_DIR / f"{s}_gate_v2" / "lgbm_v8.pkl").exists()
    )
    equity_per = args.equity / max(n_coins, 1)

    all_stats = {}

    # Run each symbol independently (portfolio = sum of independent runs)
    print(f"\n  Portfolio: {n_coins} coins, ${equity_per:,.0f} each")

    for symbol in symbols:
        eq, fills, info = run_single_symbol(
            symbol, equity=equity_per,
            fee_bps=fee_bps, slippage_bps=slippage_bps,
            out_dir=(out_dir / symbol.lower()) if out_dir else None,
            n_coins=n_coins,
        )
        stats = analyze_results(eq, fills, symbol, equity_per)
        if stats:
            all_stats[symbol] = stats
            print(f"  Time: {info.get('elapsed_sec', 0):.1f}s")

    # Portfolio aggregate
    if len(all_stats) > 1:
        total_ret_dollar = sum(
            s["end_equity"] - s["start_equity"] for s in all_stats.values()
        )
        total_trades = sum(s["trades"] for s in all_stats.values())
        total_fees = sum(s["total_fees"] for s in all_stats.values())
        port_return = total_ret_dollar / args.equity
        # Approximate portfolio Sharpe (average of per-coin)
        port_sharpe = np.mean([s["sharpe"] for s in all_stats.values()])
        port_mdd = max(s["max_dd"] for s in all_stats.values())

        print(f"\n  [PORTFOLIO]")
        print(f"  {'─' * 65}")
        print(f"  Trades: {total_trades}  Return: {port_return*100:+.2f}%  "
              f"(${total_ret_dollar:+,.0f})")
        print(f"  Max DD:  {port_mdd*100:.2f}%  Avg Sharpe: {port_sharpe:.2f}")
        print(f"  Fees:    ${total_fees:,.0f}")

    # Comparison
    print(f"\n{'=' * 70}")
    print("COMPARISON WITH HONEST NUMPY BACKTEST")
    print("=" * 70)
    print("  Honest backtest (backtest_honest.py):")
    print("    Portfolio: Sharpe 0.98, +3.41%, MaxDD 2.27%")
    print("    BTC OOS:   Sharpe 0.86, +2.89%, 360 trades, 46.9% WR")
    print("    ETH OOS:   Sharpe 1.16, +3.92%, 267 trades, 50.6% WR")
    print()
    print("  Event-driven engine:")
    for label, stats in all_stats.items():
        print(f"    {label}: Sharpe {stats['sharpe']:.2f}, "
              f"{stats['total_return']*100:+.2f}%, "
              f"MaxDD {stats['max_dd']*100:.2f}%, "
              f"{stats['trades']} trades, "
              f"{stats['win_rate']:.1f}% WR")


if __name__ == "__main__":
    main()
