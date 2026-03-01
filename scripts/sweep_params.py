"""Parameter sweep for threshold and risk_pct optimization.

Scans threshold x risk_pct grid, runs backtest for each combo,
and reports Calmar-sorted results to find optimal parameters.

Usage:
    python3 -m scripts.sweep_params --symbol BTCUSDT
    python3 -m scripts.sweep_params --all
    python3 -m scripts.sweep_params --all --threshold 0.01 --risk-pct 0.10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from scripts.run_lgbm_backtest import run_one

DEFAULT_THRESHOLDS = [0.001, 0.002, 0.005, 0.008, 0.01, 0.02]
DEFAULT_RISK_PCTS = [0.05, 0.10, 0.15, 0.20, 0.30]
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def _sf(v: str, default: float = 0.0) -> float:
    try:
        return float(v) if v else default
    except (ValueError, TypeError):
        return default


def sweep_one(
    symbol: str, threshold: float, risk_pct: float, out_base: Path,
    *,
    threshold_short: Optional[float] = None,
    atr_stop: float = 0.0,
    trailing_atr: float = 0.0,
    min_hold_bars: int = 0,
    vol_target: float = 0.0,
    enable_regime_gate: bool = False,
) -> Optional[dict]:
    out_dir = out_base / symbol / f"th{threshold}_rp{risk_pct}"
    summary = run_one(
        symbol, threshold, risk_pct, out_dir, enriched=True,
        threshold_short=threshold_short,
        atr_stop=atr_stop,
        trailing_atr=trailing_atr,
        min_hold_bars=min_hold_bars,
        vol_target=vol_target,
        enable_regime_gate=enable_regime_gate,
    )
    if summary is None:
        return None
    summary["threshold"] = threshold
    summary["risk_pct"] = risk_pct
    return summary


def sweep_symbol(
    symbol: str,
    thresholds: List[float],
    risk_pcts: List[float],
    out_base: Path,
    *,
    threshold_short: Optional[float] = None,
    atr_stop: float = 0.0,
    trailing_atr: float = 0.0,
    min_hold_bars: int = 0,
    vol_target: float = 0.0,
    enable_regime_gate: bool = False,
) -> List[dict]:
    results = []
    total = len(thresholds) * len(risk_pcts)
    idx = 0
    for th in thresholds:
        for rp in risk_pcts:
            idx += 1
            print(f"[{idx}/{total}] {symbol} th={th} rp={rp}")
            s = sweep_one(
                symbol, th, rp, out_base,
                threshold_short=threshold_short,
                atr_stop=atr_stop,
                trailing_atr=trailing_atr,
                min_hold_bars=min_hold_bars,
                vol_target=vol_target,
                enable_regime_gate=enable_regime_gate,
            )
            if s:
                results.append(s)
    return results


def print_sweep_table(results: List[dict]) -> None:
    if not results:
        print("No results.")
        return

    # Sort by calmar descending
    results.sort(key=lambda r: _sf(r.get("calmar_ratio", "")), reverse=True)

    hdr = (
        f"{'Th':>7} {'Rp':>5} {'Calmar':>8} {'Sharpe':>8} "
        f"{'CAGR%':>8} {'MaxDD%':>8} {'Ret%':>8} {'PF':>6} "
        f"{'Trd/d':>6} {'Trades':>7} {'Fee%Grs':>8} {'WR%':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        th = r.get("threshold", 0)
        rp = r.get("risk_pct", 0)
        calmar = _sf(r.get("calmar_ratio", ""))
        sharpe = _sf(r.get("sharpe_ratio", ""))
        cagr = _sf(r.get("cagr", "")) * 100
        mdd = _sf(r.get("max_drawdown", "")) * 100
        ret = _sf(r.get("return", "")) * 100
        pf = _sf(r.get("profit_factor", ""))
        tpd = _sf(r.get("trades_per_day", ""))
        trades = int(_sf(r.get("trades", "")))
        wr = _sf(r.get("win_rate", "")) * 100

        # Fee as % of gross profit
        total_fees = _sf(r.get("total_fees", ""))
        gross = _sf(r.get("return", "")) * 10000  # starting_balance=10000
        fee_pct_gross = (total_fees / gross * 100) if gross > 0 else 999

        print(
            f"{th:>7.3f} {rp:>5.2f} {calmar:>8.3f} {sharpe:>8.3f} "
            f"{cagr:>8.1f} {mdd:>8.1f} {ret:>8.2f} {pf:>6.2f} "
            f"{tpd:>6.2f} {trades:>7} {fee_pct_gross:>8.1f} {wr:>6.1f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Parameter sweep: threshold x risk_pct")
    p.add_argument("--symbol", help="Single symbol to sweep")
    p.add_argument("--all", action="store_true", help="Sweep all symbols")
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Single threshold (skip sweep, use this value)",
    )
    p.add_argument(
        "--risk-pct", type=float, default=None,
        help="Single risk_pct (skip sweep, use this value)",
    )
    p.add_argument("--out", default="output/sweep", help="Output base directory")
    p.add_argument("--threshold-short", type=float, default=None, help="Short threshold (constant across sweep)")
    p.add_argument("--atr-stop", type=float, default=0.0, help="ATR hard stop multiplier (constant)")
    p.add_argument("--trailing-atr", type=float, default=0.0, help="ATR trailing stop multiplier (constant)")
    p.add_argument("--min-hold", type=int, default=0, help="Min bars to hold (constant)")
    p.add_argument("--vol-target", type=float, default=0.0, help="Vol-target sizing (constant)")
    p.add_argument("--regime-gate", action="store_true", help="Enable RegimeGate")
    args = p.parse_args()

    if not args.symbol and not args.all:
        p.print_help()
        return

    symbols = ALL_SYMBOLS if args.all else [args.symbol.upper()]
    thresholds = [args.threshold] if args.threshold is not None else DEFAULT_THRESHOLDS
    risk_pcts = [args.risk_pct] if args.risk_pct is not None else DEFAULT_RISK_PCTS
    out_base = Path(args.out)

    print(f"Parameter Sweep")
    print(f"  Symbols:    {symbols}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Risk pcts:  {risk_pcts}")
    print(f"  Combos:     {len(thresholds) * len(risk_pcts)} per symbol")
    print()

    all_results: Dict[str, List[dict]] = {}
    for sym in symbols:
        print(f"=== {sym} ===")
        results = sweep_symbol(
            sym, thresholds, risk_pcts, out_base,
            threshold_short=args.threshold_short,
            atr_stop=args.atr_stop,
            trailing_atr=args.trailing_atr,
            min_hold_bars=args.min_hold,
            vol_target=args.vol_target,
            enable_regime_gate=args.regime_gate,
        )
        all_results[sym] = results
        print()
        print_sweep_table(results)
        print()

    # Save combined results
    out_base.mkdir(parents=True, exist_ok=True)
    combined_path = out_base / "sweep_results.json"
    with combined_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {combined_path}")


if __name__ == "__main__":
    main()
