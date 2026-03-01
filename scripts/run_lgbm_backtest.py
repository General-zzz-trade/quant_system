"""Run backtest using trained LGBM alpha model as signal source.

Usage:
    python3 -m scripts.run_lgbm_backtest --symbol BTCUSDT
    python3 -m scripts.run_lgbm_backtest --symbol ETHUSDT --threshold 0.001
    python3 -m scripts.run_lgbm_backtest --all
    python3 -m scripts.run_lgbm_backtest --all --risk-pct 0.5 --threshold 0.0005
"""
from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Optional

from decision.ml_decision import MLDecisionModule


def run_one(
    symbol: str, threshold: float, risk_pct: float, out_base: Path,
    *, enriched: bool = False,
    threshold_short: Optional[float] = None,
    atr_stop: float = 0.0,
    trailing_atr: float = 0.0,
    min_hold_bars: int = 0,
    vol_target: float = 0.0,
    enable_regime_gate: bool = False,
) -> Optional[dict]:
    from runner.backtest_runner import run_backtest
    from alpha.models.lgbm_alpha import LGBMAlphaModel

    csv_path = Path(f"data_files/{symbol}_1h.csv")
    model_path = Path(f"models/{symbol}/lgbm_alpha_final.pkl")

    if not csv_path.exists():
        print(f"  SKIP {symbol}: CSV not found at {csv_path}")
        return None
    if not model_path.exists():
        print(f"  SKIP {symbol}: Model not found at {model_path}")
        return None

    model = LGBMAlphaModel(name="lgbm_alpha")
    model.load(model_path)

    if enriched:
        from features.enriched_computer import EnrichedFeatureComputer
        feature_computer = EnrichedFeatureComputer()
    else:
        from features.live_computer import LiveFeatureComputer
        feature_computer = LiveFeatureComputer(fast_ma=10, slow_ma=30, vol_window=20)

    module = MLDecisionModule(
        symbol=symbol, risk_pct=risk_pct, threshold=threshold,
        threshold_short=threshold_short,
        atr_stop=atr_stop,
        trailing_atr=trailing_atr,
        min_hold_bars=min_hold_bars,
        vol_target=vol_target,
    )
    out_dir = out_base / symbol

    print(f"  Running {symbol} (threshold={threshold}, risk_pct={risk_pct})...")

    equity, fills = run_backtest(
        csv_path=csv_path,
        symbol=symbol,
        starting_balance=Decimal("10000"),
        fee_bps=Decimal("4"),
        slippage_bps=Decimal("2"),
        out_dir=out_dir,
        decision_modules=[module],
        feature_computer=feature_computer,
        alpha_models=[model],
        enable_regime_gate=enable_regime_gate,
    )

    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="LGBM alpha backtest")
    p.add_argument("--symbol", help="Single symbol to test")
    p.add_argument("--threshold", type=float, default=0.005, help="ML score threshold")
    p.add_argument("--risk-pct", type=float, default=0.30, help="Equity % per position (0.3 = 30%%)")
    p.add_argument("--out", default="output/lgbm", help="Output directory")
    p.add_argument("--all", action="store_true", help="Run all 3 symbols")
    p.add_argument("--enriched", action="store_true", help="Use enriched feature set")
    p.add_argument("--threshold-short", type=float, default=None, help="Short threshold (None=same as --threshold)")
    p.add_argument("--atr-stop", type=float, default=0.0, help="ATR hard stop multiplier (0=disabled)")
    p.add_argument("--trailing-atr", type=float, default=0.0, help="ATR trailing stop multiplier (0=disabled)")
    p.add_argument("--min-hold", type=int, default=0, help="Min bars to hold (0=disabled)")
    p.add_argument("--vol-target", type=float, default=0.0, help="Vol-target sizing (0=fixed risk_pct)")
    p.add_argument("--regime-gate", action="store_true", help="Enable RegimeGate in backtest")
    args = p.parse_args()

    out_base = Path(args.out)

    symbols = []
    if args.all:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        p.print_help()
        return

    print(f"LGBM Alpha Backtest (equity-% sizing)")
    print(f"  Threshold: {args.threshold}")
    print(f"  Risk per position: {args.risk_pct*100:.0f}% of equity")
    print()

    results = {}
    for sym in symbols:
        summary = run_one(
            sym, args.threshold, args.risk_pct, out_base,
            enriched=args.enriched,
            threshold_short=args.threshold_short,
            atr_stop=args.atr_stop,
            trailing_atr=args.trailing_atr,
            min_hold_bars=args.min_hold,
            vol_target=args.vol_target,
            enable_regime_gate=args.regime_gate,
        )
        if summary:
            results[sym] = summary

    if results:
        print(f"\n{'='*75}")
        print(f"{'Symbol':<10} {'Return':>10} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>10} {'WinRate':>8} {'Trades':>7}")
        print(f"{'-'*75}")
        for sym, s in results.items():
            ret = float(s.get("return", 0)) * 100
            cagr = float(s.get("cagr", 0)) * 100
            sharpe = s.get("sharpe_ratio", "N/A")
            maxdd = float(s.get("max_drawdown", 0)) * 100
            wr = float(s.get("win_rate", 0)) * 100
            trades = s.get("trades", 0)
            print(f"{sym:<10} {ret:>9.2f}% {cagr:>7.1f}% {sharpe:>8} {maxdd:>9.2f}% {wr:>7.1f}% {trades:>7}")
        print(f"{'='*75}")


if __name__ == "__main__":
    main()
