"""Risk-managed long BTC/ETH: ONE predeclared spec, ONE predeclared gate, development data only.

    python -m lab.overlay_study --data-dir data_files/binance_public --out research/results/overlay_<date>

Everything that decides the outcome is fixed in this file and was committed
before any result on real data was computed (see
research/protocols/2026-11-13-overlay-and-ml-recheck.md):

- **Product.** A drawdown-control overlay on a long BTC/ETH holding, not an
  alpha. Long-only is part of the product definition, not a tuned choice.
- **Spec, no selection.** The daily trend score is the mean sign of the
  trailing return over the union of every lookback in the 2026-11-13 lab grid
  (10, 20, 30, 60, 90, 120, 180 days), equally weighted. Every other parameter
  is the lab ``BASE`` value, unchanged.
- **Window.** Development data only, 2021-01-10 to 2025-11-16. Bars after
  ``DEV_END`` never enter a simulation or a statistic. That later window has
  been viewed several times; the final test is forward paper trading.
- **Benchmark.** 50/50 BTC/ETH at a constant gross exposure equal to the
  strategy's own average exposure, rebalanced daily, without costs. It asks
  whether trend timing beats simply holding less.
- **Gate.** All of ``GATE`` must hold (see ``gate_checks``).

Caveat recorded in the output: the development window and the lab grid were
seen before this spec was written, so this is a screen, not an independent
test. ``ROBUSTNESS`` specs are reported only and never change the outcome.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from lab.data import DAY, funding_series, load_symbol
from lab.metrics import daily_equity, exposure
from lab.simulate import simulate
from lab.strategy import StrategySpec
from lab.strategy import compute_targets
from lab.study import BASE, DEV_END, EVAL_START, STAGE_A, SYMBOLS, _manifest

SLEEVE_BALANCE = "5000"                      # per symbol; the portfolio starts at 10,000
LOOKBACKS = tuple(sorted({lb for spec in STAGE_A for lb in spec.lookbacks}))
SPEC: StrategySpec = replace(BASE, lookbacks=LOOKBACKS, long_only=True)
ROBUSTNESS = {f"lookbacks_{'_'.join(map(str, spec.lookbacks))}": replace(SPEC, lookbacks=spec.lookbacks)
              for spec in STAGE_A if spec.signal == "sign" and spec.long_only}

GATE = {
    "calmar_vs_matched_benchmark_min_ratio": 1.5,    # CAGR / max drawdown, strategy vs matched benchmark
    "max_drawdown_at_most_matched_benchmark": True,
    "sharpe_shortfall_vs_benchmark_max": 0.10,       # daily Sharpe may trail the benchmark by at most this
    "cost_x2_total_return_min_exclusive": 0.0,
    "fills_per_symbol_year_max": 150,
}


def stats(equity: pd.Series) -> dict:
    """Daily equity (first point = starting capital) -> summary statistics."""
    returns = equity.pct_change().dropna()
    years = len(returns) / 365.0
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0
    peak = equity.cummax()
    mdd = float(((peak - equity) / peak).max())
    std = float(returns.std())
    return {"days": int(len(returns)), "total_return": total, "cagr": cagr, "max_drawdown": mdd,
            "sharpe": float(returns.mean() / std * math.sqrt(365)) if std > 0 else float("nan"),
            "vol": float(std * math.sqrt(365)), "calmar": cagr / mdd if mdd > 0 else float("nan")}


def yearly_returns(equity: pd.Series) -> dict[int, float]:
    years = pd.to_datetime(equity.index.to_numpy() * DAY, unit="ms", utc=True).year
    out, previous = {}, float(equity.iloc[0])
    for year in sorted(set(years)):
        last = float(equity[years == year].iloc[-1])
        out[int(year)] = last / previous - 1
        previous = last
    return out


def portfolio_equity(sleeves, first_day: int, initial: float) -> pd.Series:
    """Hourly sleeve curves -> daily portfolio equity, prefixed with the starting capital."""
    daily = daily_equity([s.curve for s in sleeves])
    daily = daily[daily.index >= first_day]
    return pd.concat([pd.Series({first_day - 1: initial}), daily])


def matched_benchmark(frames: dict, days: pd.Index, level: float, initial: float) -> pd.Series:
    """50/50 BTC/ETH at constant gross exposure ``level``, rebalanced daily, no costs."""
    returns = {}
    for symbol, frame in frames.items():
        close = frame.set_index("available_at")["close"]
        returns[symbol] = close.groupby((close.index.to_numpy() - 1) // DAY).last().pct_change()
    mean = pd.concat(returns, axis=1).mean(axis=1).reindex(days[1:])
    if mean.isna().any():
        raise ValueError("benchmark has gaps on strategy days")
    curve = initial * (1 + level * mean).cumprod()
    return pd.concat([pd.Series({days[0]: initial}), curve])


def gate_checks(strategy: dict, benchmark: dict, cost_x2: dict, fills_per_year: dict) -> dict:
    worst_fills = max(fills_per_year.values())
    checks = {
        "calmar_vs_matched_benchmark": (
            strategy["calmar"], benchmark["calmar"],
            bool(math.isfinite(strategy["calmar"]) and math.isfinite(benchmark["calmar"])
                 and strategy["calmar"] >= GATE["calmar_vs_matched_benchmark_min_ratio"] * benchmark["calmar"]
                 and strategy["calmar"] > 0)),
        "max_drawdown_vs_matched_benchmark": (
            strategy["max_drawdown"], benchmark["max_drawdown"],
            bool(strategy["max_drawdown"] <= benchmark["max_drawdown"])),
        "sharpe_vs_benchmark": (
            strategy["sharpe"], benchmark["sharpe"],
            bool(strategy["sharpe"] >= benchmark["sharpe"] - GATE["sharpe_shortfall_vs_benchmark_max"])),
        "cost_x2_total_return": (
            cost_x2["total_return"], GATE["cost_x2_total_return_min_exclusive"],
            bool(cost_x2["total_return"] > GATE["cost_x2_total_return_min_exclusive"])),
        "fills_per_symbol_year": (
            worst_fills, GATE["fills_per_symbol_year_max"], bool(worst_fills <= GATE["fills_per_symbol_year_max"])),
    }
    return {name: {"value": v, "reference": ref, "ok": ok} for name, (v, ref, ok) in checks.items()}


def _code_identity() -> dict:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "lab", "runner/offline"],
                                             text=True).strip())
    except Exception:
        revision, dirty = "unavailable", True
    return {"git_revision": revision, "lab_or_offline_dirty": dirty}


def run_study(data_dir: Path, out: Path, *, eval_start: int = EVAL_START, dev_end: int = DEV_END) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("use an empty output directory; preserve previous studies")
    if eval_start % DAY or dev_end % DAY or not eval_start < dev_end:
        raise ValueError("evaluation window must be whole UTC days")
    out.mkdir(parents=True, exist_ok=True)
    bars, funding, frames, fseries = {}, {}, {}, {}
    for symbol in SYMBOLS:
        bars[symbol], funding[symbol], frames[symbol] = load_symbol(data_dir, symbol)
        fseries[symbol] = funding_series(funding[symbol])
    first_day = eval_start // DAY
    initial = float(len(SYMBOLS)) * float(SLEEVE_BALANCE)

    def run(spec: StrategySpec, cost_multiplier: int = 1):
        sleeves = [simulate(s, bars[s], compute_targets(frames[s], fseries[s], spec), funding[s], spec,
                            start_ms=eval_start, end_ms=dev_end, cost_multiplier=cost_multiplier,
                            starting_balance=SLEEVE_BALANCE) for s in SYMBOLS]
        return sleeves, portfolio_equity(sleeves, first_day, initial)

    sleeves, equity = run(SPEC)
    if equity.index[-1] >= dev_end // DAY:
        raise RuntimeError("development statistics reached the held-out window")
    level = exposure([s.curve for s in sleeves])
    bench = matched_benchmark(frames, equity.index, level, initial)
    full_hold = matched_benchmark(frames, equity.index, 1.0, initial)
    _, equity_x2 = run(SPEC, cost_multiplier=2)
    days = len(equity) - 1
    fills_per_year = {s.symbol: len(s.fills) / days * 365.0 for s in sleeves}
    strategy_stats, bench_stats, x2_stats = stats(equity), stats(bench), stats(equity_x2)
    checks = gate_checks(strategy_stats, bench_stats, x2_stats, fills_per_year)
    passed = all(c["ok"] for c in checks.values())
    robustness = {}
    for name, spec in ROBUSTNESS.items():
        r_sleeves, r_equity = run(spec)
        r_level = exposure([s.curve for s in r_sleeves])
        robustness[name] = {"strategy": stats(r_equity), "average_exposure": r_level,
                            "matched_benchmark": stats(matched_benchmark(frames, r_equity.index, r_level, initial))}
    window = {"eval_start_ms": eval_start, "dev_end_ms": dev_end,
              "holdout": "not computed; the final test is forward paper trading"}
    frozen = {
        "schema": 1, "kind": "btc_eth_multitimeframe_target", "symbols": list(SYMBOLS), "spec": SPEC.record(),
        "study": "lab.overlay_study", "evaluation": window,
        "selection": "none: one predeclared spec", "trials": 1, "gate": GATE,
        "gate_result": checks, "screen_passed": passed,
        # ``passed`` is what runner.binance_target reads as the mainnet gate. A development-window
        # screen can only earn forward paper trading, so it never sets it.
        "passed": False,
        "mainnet": "refused: requires an accepted forward paper test, recorded by a person",
        "code": _code_identity(), "data": _manifest(data_dir),
        "caveat": ("The development window and the 2026-11-13 lab grid were seen before this spec was "
                   "written; this is a screen, not an independent test."),
    }
    summary = {
        "window": window, "spec": SPEC.record(), "average_exposure": level,
        "strategy": strategy_stats, "strategy_cost_x2": x2_stats,
        "matched_benchmark": bench_stats, "benchmark_50_50_full_exposure": stats(full_hold),
        "fills_per_symbol_year": fills_per_year,
        "yearly": {"strategy": yearly_returns(equity), "matched_benchmark": yearly_returns(bench)},
        "gate": {"passed": passed, "checks": checks},
        "robustness_report_only": robustness,
    }
    (out / "final_spec.json").write_text(json.dumps(frozen, indent=2, default=float) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data_files/binance_public"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_study(args.data_dir, args.out)
    print(json.dumps(summary["gate"], indent=2, default=float))


if __name__ == "__main__":
    main()


__all__ = ["SPEC", "GATE", "ROBUSTNESS", "run_study", "stats", "matched_benchmark", "gate_checks", "np"]
