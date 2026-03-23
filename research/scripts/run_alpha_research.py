"""End-to-end alpha research CLI.

Usage:
    python3 -m scripts.run_alpha_research --factors momentum_20,rsi_14,macd_hist
    python3 -m scripts.run_alpha_research --factors momentum_20 --backtest --walk-forward
    python3 -m scripts.run_alpha_research --factors momentum_20,rsi_14 --compare
    python3 -m scripts.run_alpha_research --custom-factor my_factors.py:my_alpha
"""

from __future__ import annotations

import argparse
import importlib.util
from decimal import Decimal
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from research.alpha_factor import AlphaFactor, ComparisonReport, FactorReport, evaluate_factor, compare_factors
from runner.backtest.csv_io import OhlcvBar


# ---------------------------------------------------------------------------
# Built-in Factor Library
# ---------------------------------------------------------------------------

def _momentum(window: int) -> Callable[[Sequence[OhlcvBar]], List[Optional[float]]]:
    def compute(bars: Sequence[OhlcvBar]) -> List[Optional[float]]:
        result: List[Optional[float]] = [None] * len(bars)
        for i in range(window, len(bars)):
            prev = float(bars[i - window].c)
            if prev != 0:
                result[i] = float(bars[i].c) / prev - 1.0
        return result
    return compute


def _rsi_factor(window: int) -> Callable[[Sequence[OhlcvBar]], List[Optional[float]]]:
    def compute(bars: Sequence[OhlcvBar]) -> List[Optional[float]]:
        result: List[Optional[float]] = [None] * len(bars)
        if len(bars) < window + 1:
            return result
        gains: List[float] = []
        losses: List[float] = []
        for i in range(1, len(bars)):
            diff = float(bars[i].c) - float(bars[i - 1].c)
            gains.append(max(diff, 0.0))
            losses.append(max(-diff, 0.0))
        if len(gains) < window:
            return result
        avg_gain = sum(gains[:window]) / window
        avg_loss = sum(losses[:window]) / window
        alpha = 2.0 / (window + 1)
        for i in range(window, len(gains)):
            avg_gain = avg_gain * (1 - alpha) + gains[i] * alpha
            avg_loss = avg_loss * (1 - alpha) + losses[i] * alpha
            if avg_loss < 1e-12:
                result[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i + 1] = 100.0 - 100.0 / (1.0 + rs)
        return result
    return compute


def _macd_hist_factor() -> Callable[[Sequence[OhlcvBar]], List[Optional[float]]]:
    def compute(bars: Sequence[OhlcvBar]) -> List[Optional[float]]:
        result: List[Optional[float]] = [None] * len(bars)
        if len(bars) < 35:
            return result
        fast_ema = float(bars[0].c)
        slow_ema = float(bars[0].c)
        signal_ema = 0.0
        a_fast = 2.0 / 13.0
        a_slow = 2.0 / 27.0
        a_sig = 2.0 / 10.0
        atr_vals: List[float] = []
        for i in range(1, len(bars)):
            c = float(bars[i].c)
            h = float(bars[i].h)
            lo = float(bars[i].l)
            prev_c = float(bars[i - 1].c)
            fast_ema = fast_ema * (1 - a_fast) + c * a_fast
            slow_ema = slow_ema * (1 - a_slow) + c * a_slow
            macd_line = fast_ema - slow_ema
            tr = max(h - lo, abs(h - prev_c), abs(lo - prev_c))
            atr_vals.append(tr)
            if i < 26:
                signal_ema = macd_line
                continue
            signal_ema = signal_ema * (1 - a_sig) + macd_line * a_sig
            hist = macd_line - signal_ema
            atr_window = min(14, len(atr_vals))
            atr = sum(atr_vals[-atr_window:]) / atr_window
            result[i] = hist / atr if atr > 1e-12 else None
        return result
    return compute


def _bb_pct_factor() -> Callable[[Sequence[OhlcvBar]], List[Optional[float]]]:
    def compute(bars: Sequence[OhlcvBar]) -> List[Optional[float]]:
        import math
        result: List[Optional[float]] = [None] * len(bars)
        window = 20
        for i in range(window - 1, len(bars)):
            closes = [float(bars[j].c) for j in range(i - window + 1, i + 1)]
            mean = sum(closes) / window
            var = sum((c - mean) ** 2 for c in closes) / window
            std = math.sqrt(var) if var > 0 else 0.0
            if std < 1e-12:
                continue
            upper = mean + 2.0 * std
            lower = mean - 2.0 * std
            band_width = upper - lower
            if band_width < 1e-12:
                continue
            result[i] = (float(bars[i].c) - lower) / band_width
        return result
    return compute


def _vol_momentum(window: int) -> Callable[[Sequence[OhlcvBar]], List[Optional[float]]]:
    def compute(bars: Sequence[OhlcvBar]) -> List[Optional[float]]:
        result: List[Optional[float]] = [None] * len(bars)
        for i in range(window, len(bars)):
            vols = [float(bars[j].v or 0) for j in range(i - window, i)]
            mean_vol = sum(vols) / window
            if mean_vol < 1e-12:
                continue
            result[i] = float(bars[i].v or 0) / mean_vol
        return result
    return compute


def _price_accel() -> Callable[[Sequence[OhlcvBar]], List[Optional[float]]]:
    def compute(bars: Sequence[OhlcvBar]) -> List[Optional[float]]:
        result: List[Optional[float]] = [None] * len(bars)
        for i in range(20, len(bars)):
            c = float(bars[i].c)
            c5 = float(bars[i - 5].c)
            c20 = float(bars[i - 20].c)
            if c5 == 0 or c20 == 0:
                continue
            mom5 = c / c5 - 1.0
            mom20 = c / c20 - 1.0
            result[i] = mom5 - mom20
        return result
    return compute


BUILTIN_FACTORS: Dict[str, AlphaFactor] = {
    "momentum_20": AlphaFactor("momentum_20", _momentum(20), "momentum"),
    "momentum_50": AlphaFactor("momentum_50", _momentum(50), "momentum"),
    "rsi_14": AlphaFactor("rsi_14", _rsi_factor(14), "technical"),
    "macd_hist": AlphaFactor("macd_hist", _macd_hist_factor(), "technical"),
    "bb_pct": AlphaFactor("bb_pct", _bb_pct_factor(), "technical"),
    "vol_mom_20": AlphaFactor("vol_mom_20", _vol_momentum(20), "volume"),
    "price_accel": AlphaFactor("price_accel", _price_accel(), "composite"),
}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_factor_report(report: FactorReport) -> None:
    print(f"\n=== Factor: {report.name} ===")
    print(f"IC mean:       {report.ic_mean:+.4f}    Rank IC mean:  {report.rank_ic_mean:+.4f}")
    print(f"IC std:        {report.ic_std:.4f}    Rank IC std:   {report.rank_ic_std:.4f}")
    print(f"IC IR:         {report.ic_ir:+.2f}      Rank IC IR:    {report.rank_ic_ir:+.2f}")
    print(f"IC t-stat:     {report.ic_t_stat:+.2f}      Pct IC > 0:    {report.pct_positive_ic:.1%}")
    print(f"Turnover:      {report.avg_turnover:.2f}      Autocorr:      {report.factor_autocorr:.2f}")
    print(f"Observations:  {report.n_observations}")
    if report.decay_profile:
        parts = "  ".join(f"h={h}: {ic:.3f}" for h, ic in zip([1, 5, 10, 24], report.decay_profile))
        print(f"Decay: {parts}")


def print_comparison(report: ComparisonReport) -> None:
    print("\n=== Factor Comparison ===")
    for fr in report.factor_reports:
        print_factor_report(fr)

    names = [fr.name for fr in report.factor_reports]
    print("\n--- Correlation Matrix ---")
    header = f"{'':>15s}" + "".join(f"{n:>15s}" for n in names)
    print(header)
    for a in names:
        row = f"{a:>15s}" + "".join(f"{report.correlation_matrix[a][b]:>15.3f}" for b in names)
        print(row)

    print("\n--- Marginal IC ---")
    for name, mic in report.marginal_ic.items():
        print(f"  {name:>20s}: {mic:+.4f}")


def print_backtest_summary(summary: Dict) -> None:
    print(f"\n=== Backtest: {summary.get('factor_name', summary.get('symbol', '?'))} ===")
    print(f"Bars:          {summary.get('bars', 0)}")
    print(f"Return:        {summary.get('return', '?')}")
    print(f"Max Drawdown:  {summary.get('max_drawdown', '?')}")
    print(f"Sharpe:        {summary.get('sharpe_ratio', '?')}")
    print(f"Trades:        {summary.get('trades', 0)}")
    print(f"Win Rate:      {summary.get('win_rate', '?')}")
    print(f"Profit Factor: {summary.get('profit_factor', '?')}")


# ---------------------------------------------------------------------------
# Custom factor loading
# ---------------------------------------------------------------------------

def load_custom_factor(spec: str) -> AlphaFactor:
    """Load a custom factor from 'path/to/file.py:function_name'."""
    if ":" not in spec:
        raise ValueError(f"Custom factor spec must be 'path:function', got: {spec}")
    path_str, func_name = spec.rsplit(":", 1)
    path = Path(path_str).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Custom factor file not found: {path}")

    module_spec = importlib.util.spec_from_file_location("custom_factor", path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    func = getattr(module, func_name, None)
    if func is None:
        raise AttributeError(f"Function '{func_name}' not found in {path}")

    return AlphaFactor(name=func_name, compute_fn=func, category="custom")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_csv() -> Path:
    """Find a default CSV data file."""
    candidates = [
        Path("data_files/BTCUSDT_1h_ohlcv.csv"),
        Path("data_files/BTCUSDT_1m_ohlcv.csv"),
        Path("data/binance/ohlcv/BTCUSDT_1m_ohlcv.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    # Search for any CSV in data_files/
    data_dir = Path("data_files")
    if data_dir.exists():
        csvs = sorted(data_dir.glob("*.csv"))
        if csvs:
            return csvs[0]
    raise FileNotFoundError(
        "No CSV data found. Provide --csv or place data in data_files/"
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Alpha factor research pipeline")
    parser.add_argument(
        "--factors", type=str, default="momentum_20",
        help="Comma-separated factor names from built-in library",
    )
    parser.add_argument("--custom-factor", type=str, help="Custom factor as path:function")
    parser.add_argument("--csv", type=str, help="Path to OHLCV CSV")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--horizons", type=str, default="1,5,10,24",
                        help="Comma-separated horizons for IC decay")
    parser.add_argument("--compare", action="store_true", help="Run factor comparison")
    parser.add_argument("--backtest", action="store_true", help="Run factor backtest")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation")
    parser.add_argument("--sweep", type=str, default=None,
                        help="Factor family to sweep (e.g. 'momentum', 'all')")
    parser.add_argument("--screen", action="store_true", help="Screen swept factors for quality")
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--out-dir", type=str, help="Output directory")

    args = parser.parse_args(argv)

    # Resolve CSV
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = _find_csv()
    print(f"Data: {csv_path}")

    # Build factor list
    factors: List[AlphaFactor] = []
    if args.custom_factor:
        factors.append(load_custom_factor(args.custom_factor))

    for name in args.factors.split(","):
        name = name.strip()
        if not name:
            continue
        if name in BUILTIN_FACTORS:
            factors.append(BUILTIN_FACTORS[name])
        else:
            print(f"WARNING: Unknown factor '{name}', skipping. Available: {list(BUILTIN_FACTORS.keys())}")

    if not factors:
        print("No factors to evaluate.")
        return

    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    # Load bars
    from runner.backtest.csv_io import iter_ohlcv_csv
    bars = list(iter_ohlcv_csv(csv_path))
    print(f"Loaded {len(bars)} bars")

    # Sweep & screen mode
    if args.sweep:
        from research.factor_factory import FactorFactory

        _SWEEP_GRIDS = {
            "momentum": {"window": [5, 10, 20, 50, 100]},
            "rsi": {"window": [7, 14, 21, 28]},
            "vol_mom": {"window": [5, 10, 20, 50]},
        }

        factory = FactorFactory({
            "momentum": _momentum,
            "rsi": _rsi_factor,
            "vol_mom": _vol_momentum,
        })

        families = list(_SWEEP_GRIDS.keys()) if args.sweep == "all" else [args.sweep]
        sweep_factors = []
        for fam in families:
            grid = _SWEEP_GRIDS.get(fam)
            if grid is None:
                print(f"WARNING: Unknown sweep family '{fam}'")
                continue
            sweep_factors.extend(factory.generate_sweep(fam, grid))

        print(f"Swept {len(sweep_factors)} factor variants")

        if args.screen:
            results = factory.screen(sweep_factors, bars)
            passed = [r for r in results if r.passed]
            print(f"Screening: {len(passed)}/{len(results)} passed")
            for r in results[:10]:
                status = "PASS" if r.passed else "FAIL"
                print(f"  [{status}] {r.factor.name:>20s}  IC_IR={r.report.ic_ir:+.2f}  "
                      f"IC={r.report.ic_mean:+.4f}  autocorr={r.report.factor_autocorr:.2f}")
                if not r.passed:
                    print(f"           reasons: {', '.join(r.reject_reasons)}")

            if passed:
                selected = factory.select_uncorrelated(results, bars)
                print(f"\nUncorrelated selection: {len(selected)} factors")
                for r in selected:
                    print(f"  {r.factor.name:>20s}  IC_IR={r.report.ic_ir:+.2f}")
        else:
            for f in sweep_factors:
                report = evaluate_factor(f, bars, horizons)
                print_factor_report(report)
        return

    # Evaluate
    if args.compare and len(factors) > 1:
        report = compare_factors(factors, bars, horizons)
        print_comparison(report)
    else:
        for factor in factors:
            report = evaluate_factor(factor, bars, horizons)
            print_factor_report(report)

    # Backtest
    if args.backtest:
        from research.factor_backtest import backtest_factor, FactorStrategyConfig

        out = Path(args.out_dir) if args.out_dir else None
        for factor in factors:
            cfg = FactorStrategyConfig(symbol=args.symbol)
            summary = backtest_factor(
                factor, csv_path,
                symbol=args.symbol,
                starting_balance=Decimal(str(args.balance)),
                fee_bps=Decimal(str(args.fee_bps)),
                config=cfg,
                out_dir=out / factor.name if out else None,
            )
            print_backtest_summary(summary)

    # Walk-forward
    if args.walk_forward:
        from research.factor_backtest import walk_forward_factor, FactorStrategyConfig

        for factor in factors:
            cfg = FactorStrategyConfig(symbol=args.symbol)
            folds = walk_forward_factor(
                factor, csv_path,
                symbol=args.symbol,
                starting_balance=Decimal(str(args.balance)),
                fee_bps=Decimal(str(args.fee_bps)),
                config=cfg,
                train_size=args.train_size,
                test_size=args.test_size,
            )
            print(f"\n=== Walk-Forward: {factor.name} ({len(folds)} folds) ===")
            for fold in folds:
                sharpe = fold.get("sharpe_ratio", "?")
                ret = fold.get("return", "?")
                dd = fold.get("max_drawdown", "?")
                print(f"  Fold {fold.get('window_idx', '?')}: "
                      f"return={ret}  dd={dd}  sharpe={sharpe}")


if __name__ == "__main__":
    main()
