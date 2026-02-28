"""Multi-factor trend/mean-reversion hybrid strategy backtest.

Usage:
    python3 -m scripts.run_multi_factor_backtest
    python3 -m scripts.run_multi_factor_backtest --csv data_files/BTCUSDT_1h.csv --risk 0.02
"""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-factor strategy backtest")
    p.add_argument("--csv", default="data_files/BTCUSDT_1h.csv", help="OHLCV CSV path")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--starting-balance", type=float, default=10000)
    p.add_argument("--fee-bps", type=float, default=4)
    p.add_argument("--slippage-bps", type=float, default=2)
    p.add_argument("--risk-per-trade", type=float, default=0.02)
    p.add_argument("--atr-stop", type=float, default=3.0)
    p.add_argument("--max-position-pct", type=float, default=0.80)
    p.add_argument("--trend-threshold", type=float, default=0.30)
    p.add_argument("--range-threshold", type=float, default=0.99)
    p.add_argument("--out", default="output/multi_factor")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = (root / csv_path).resolve()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()

    from strategies.multi_factor.decision_module import MultiFactorConfig, MultiFactorDecisionModule
    from runner.backtest_runner import run_backtest

    config = MultiFactorConfig(
        symbol=args.symbol,
        risk_per_trade=args.risk_per_trade,
        atr_stop_multiple=args.atr_stop,
        max_position_pct=args.max_position_pct,
        trend_threshold=args.trend_threshold,
        range_threshold=args.range_threshold,
    )
    module = MultiFactorDecisionModule(config=config)

    print(f"Running multi-factor backtest...")
    print(f"  CSV: {csv_path}")
    print(f"  Symbol: {args.symbol}")
    print(f"  Balance: {args.starting_balance}")
    print(f"  Fee: {args.fee_bps} bps, Slippage: {args.slippage_bps} bps")
    print(f"  Risk/trade: {args.risk_per_trade}, ATR stop: {args.atr_stop}x")
    print()

    equity, fills = run_backtest(
        csv_path=csv_path,
        symbol=args.symbol,
        starting_balance=Decimal(str(args.starting_balance)),
        fee_bps=Decimal(str(args.fee_bps)),
        slippage_bps=Decimal(str(args.slippage_bps)),
        out_dir=out_dir,
        decision_modules=[module],
    )

    if not equity:
        print("No equity points produced.")
        return

    start_eq = float(equity[0].equity)
    end_eq = float(equity[-1].equity)
    ret = (end_eq - start_eq) / start_eq if start_eq != 0 else 0

    # Max drawdown
    peak = 0.0
    max_dd = 0.0
    for ep in equity:
        eq = float(ep.equity)
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    print(f"=== Results ===")
    print(f"bars={len(equity)}")
    print(f"start_equity={start_eq:.2f}")
    print(f"end_equity={end_eq:.2f}")
    print(f"return={ret:.4%}")
    print(f"max_drawdown={max_dd:.4%}")
    print(f"total_fills={len(fills)}")

    # Load summary if written
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"trades={summary.get('trades')}")
        print(f"trades_per_day={summary.get('trades_per_day')}")
        print(f"win_rate={summary.get('win_rate')}")
        print(f"profit_factor={summary.get('profit_factor')}")
        sharpe = summary.get("sharpe")
        print(f"sharpe={sharpe}")
        print(f"avg_trade_pnl={summary.get('avg_trade_pnl')}")
        print(f"max_consecutive_losses={summary.get('max_consecutive_losses')}")
        print(f"total_fees={summary.get('total_fees')}")
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
