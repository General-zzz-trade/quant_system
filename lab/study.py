"""BTC/ETH multi-timeframe study: predeclared candidates, staged selection,
walk-forward estimate of the selection procedure, robustness, frozen spec.

    python -m lab.study --data-dir data_files/binance_public --out research/results/lab_<date>

Everything that decides the outcome is fixed in this file BEFORE running:
candidate grids, the staged selection rule, the walk-forward folds, the gate.
Data after DEV_END (the secondary holdout, already viewed in aggregate in the
2026-11-13 study) is never used for selection; it is reported once, after the
spec is frozen. The primary final test is forward paper trading.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lab.data import DAY, funding_series, load_symbol
from lab.metrics import daily_equity, deflated_sharpe_probability, exposure, summarize, turnover, yearly
from lab.simulate import simulate
from lab.strategy import StrategySpec, compute_targets

SYMBOLS = ("BTCUSDT", "ETHUSDT")


def _ms(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp() * 1000)


EVAL_START = _ms("2021-01-10")   # every candidate (incl. 1-year funding bands) is warm
DEV_END = _ms("2025-11-16")      # selection never sees data from here on
HOLDOUT_END = _ms("2026-11-12")
OOS_YEARS = (2022, 2023, 2024, 2025)

BASE = StrategySpec(vol_lookback=30, target_vol=0.30, max_weight=0.9, band=0.10, execution="market", ttl_bars=1)
STAGE_A = [replace(BASE, lookbacks=lb, signal=sig, long_only=lo)
           for lb in ((10, 30, 90), (20, 60, 120), (30, 90, 180))
           for sig in ("sign", "scaled")
           for lo in (False, True)]
LAYERS = [(f, v, c) for f in (False, True) for v in (False, True) for c in (False, True)]
EXECUTIONS = [("market", 1), ("maker", 1), ("maker", 4)]
MIN_IMPROVEMENT = 0.05   # a layer / execution variant must add this much Sharpe to be adopted

GATE = {
    "walk_forward_sharpe_min": 0.8,
    "walk_forward_cost_x2_sharpe_min": 0.4,
    "walk_forward_max_drawdown_max": 0.35,
    "deflated_sharpe_probability_min": 0.90,
    "fills_per_symbol_year_max": 150,
    "positive_oos_years_min": 3,
}


class Lab:
    def __init__(self, data_dir: Path) -> None:
        self.bars, self.funding, self.frames, self.fseries = {}, {}, {}, {}
        for symbol in SYMBOLS:
            bars, funding, frame = load_symbol(data_dir, symbol)
            self.bars[symbol], self.funding[symbol], self.frames[symbol] = bars, funding, frame
            self.fseries[symbol] = funding_series(funding)
        self._targets: dict = {}
        self._sims: dict = {}

    def targets(self, symbol: str, spec: StrategySpec) -> pd.DataFrame:
        key = (symbol, replace(spec, band=0.0, execution="market", ttl_bars=1))
        if key not in self._targets:
            self._targets[key] = compute_targets(self.frames[symbol], self.fseries[symbol], spec)
        return self._targets[key]

    def run(self, spec: StrategySpec, cost_multiplier: int = 1) -> dict:
        key = (spec, cost_multiplier)
        if key not in self._sims:
            sleeves = [simulate(s, self.bars[s], self.targets(s, spec), self.funding[s], spec,
                                start_ms=EVAL_START, end_ms=HOLDOUT_END, cost_multiplier=cost_multiplier)
                       for s in SYMBOLS]
            self._sims[key] = {"sleeves": sleeves, "daily": daily_equity([r.curve for r in sleeves])}
        return self._sims[key]

    def sharpe(self, spec: StrategySpec, start_day: int, end_day: int) -> float:
        return summarize(self.run(spec)["daily"], start_day=start_day, end_day=end_day)["sharpe"]

    @property
    def trials(self) -> int:
        return len({k for k in self._sims if k[1] == 1})


def select(lab: Lab, start_day: int, end_day: int) -> tuple[StrategySpec, dict]:
    """Staged selection on [start_day, end_day): daily layer, then 4h/funding layers, then execution."""
    log = {}
    scores = [(lab.sharpe(spec, start_day, end_day), i, spec) for i, spec in enumerate(STAGE_A)]
    log["stage_a"] = [{"spec": s.record(), "sharpe": sh} for sh, _, s in scores]
    best_a = max((x for x in scores if math.isfinite(x[0])), key=lambda x: (x[0], -x[1]))[2]
    chosen, chosen_sharpe = best_a, lab.sharpe(best_a, start_day, end_day)
    stage_b = []
    for f, v, c in LAYERS[1:]:
        spec = replace(best_a, funding_filter=f, vol_spike_cut=v, confirm_4h=c)
        sh = lab.sharpe(spec, start_day, end_day)
        stage_b.append({"spec": spec.record(), "sharpe": sh})
        if math.isfinite(sh) and sh > chosen_sharpe + MIN_IMPROVEMENT:
            chosen, chosen_sharpe = spec, sh
    log["stage_b"] = stage_b
    base_c, base_sharpe = chosen, chosen_sharpe
    stage_c = []
    for execution, ttl in EXECUTIONS[1:]:
        spec = replace(base_c, execution=execution, ttl_bars=ttl)
        sh = lab.sharpe(spec, start_day, end_day)
        stage_c.append({"spec": spec.record(), "sharpe": sh})
        if math.isfinite(sh) and sh > base_sharpe + MIN_IMPROVEMENT and sh > chosen_sharpe:
            chosen, chosen_sharpe = spec, sh
    log["stage_c"] = stage_c
    log["chosen"] = {"spec": chosen.record(), "sharpe": chosen_sharpe}
    return chosen, log


def _returns(daily: pd.Series, start_day: int, end_day: int) -> pd.Series:
    s = daily[(daily.index >= start_day - 1) & (daily.index < end_day)]
    return s.pct_change().dropna()


def _stats(returns: pd.Series) -> dict:
    eq = (1 + returns).cumprod()
    sharpe = float(returns.mean() / returns.std() * math.sqrt(365)) if returns.std() > 0 else float("nan")
    peak = eq.cummax()
    return {"days": int(len(returns)), "sharpe": sharpe, "total_return": float(eq.iloc[-1] - 1),
            "max_drawdown": float(((peak - eq) / peak).max())}


def _manifest(data_dir: Path) -> dict:
    path = Path(data_dir) / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def day_of(ms: int) -> int:
    return ms // DAY


def run_study(data_dir: Path, out: Path) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("use an empty output directory; preserve previous studies")
    out.mkdir(parents=True, exist_ok=True)
    lab = Lab(data_dir)
    first_day = day_of(EVAL_START) + 1
    # 1) Walk-forward: re-run the whole staged selection on data before each OOS year.
    folds, oos, oos_x2 = [], [], []
    for year in OOS_YEARS:
        is_end = day_of(_ms(f"{year}-01-01"))
        oos_end = min(day_of(_ms(f"{year + 1}-01-01")), day_of(DEV_END))
        spec, log = select(lab, first_day, is_end)
        r = _returns(lab.run(spec)["daily"], is_end, oos_end)
        r2 = _returns(lab.run(spec, cost_multiplier=2)["daily"], is_end, oos_end)
        oos.append(r)
        oos_x2.append(r2)
        folds.append({"oos_year": year, "chosen": spec.record(), "in_sample_sharpe": log["chosen"]["sharpe"],
                      "oos": _stats(r), "oos_cost_x2": _stats(r2)})
    wf, wf2 = _stats(pd.concat(oos)), _stats(pd.concat(oos_x2))
    # 2) Final selection on the whole development period, then freeze.
    final, final_log = select(lab, first_day, day_of(DEV_END))
    dev = lab.run(final)
    dev_returns = _returns(dev["daily"], first_day, day_of(DEV_END))
    trial_sharpes = [lab.sharpe(spec, first_day, day_of(DEV_END)) for (spec, m) in list(lab._sims) if m == 1]
    dsr = deflated_sharpe_probability(dev_returns, trial_sharpes)
    dev_days = int(len(dev_returns))
    fills = {r.symbol: turnover([f for f in r.fills if f["at"] < DEV_END], dev_days) for r in dev["sleeves"]}
    per_symbol = {}
    for r in dev["sleeves"]:
        d = daily_equity([r.curve])
        per_symbol[r.symbol] = summarize(d, start_day=first_day, end_day=day_of(DEV_END))
    bench = {}
    for s in SYMBOLS:
        c = lab.frames[s].set_index("available_at")["close"]
        bench[s] = c.groupby((c.index.to_numpy() - 1) // DAY).last()
    bench_r = (pd.concat(bench, axis=1).pct_change().mean(axis=1)).dropna()
    bench_dev = bench_r[(bench_r.index >= first_day) & (bench_r.index < day_of(DEV_END))]
    corr = float(dev_returns.corr(bench_dev.reindex(dev_returns.index)))
    positive_years = sum(1 for f in folds if f["oos"]["total_return"] > 0)
    gate_checks = {
        "walk_forward_sharpe": (wf["sharpe"], wf["sharpe"] >= GATE["walk_forward_sharpe_min"]),
        "walk_forward_cost_x2_sharpe": (wf2["sharpe"], wf2["sharpe"] >= GATE["walk_forward_cost_x2_sharpe_min"]),
        "walk_forward_max_drawdown": (wf["max_drawdown"], wf["max_drawdown"] <= GATE["walk_forward_max_drawdown_max"]),
        "deflated_sharpe_probability": (dsr, math.isfinite(dsr) and dsr >= GATE["deflated_sharpe_probability_min"]),
        "fills_per_symbol_year": (max(fills.values()), max(fills.values()) <= GATE["fills_per_symbol_year_max"]),
        "positive_oos_years": (positive_years, positive_years >= GATE["positive_oos_years_min"]),
    }
    passed = all(ok for _, ok in gate_checks.values())
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "lab", "runner/offline"],
                                             text=True).strip())
    except Exception:
        revision, dirty = "unavailable", True
    frozen = {
        "schema": 1, "kind": "btc_eth_multitimeframe_target", "symbols": list(SYMBOLS), "spec": final.record(),
        "evaluation": {"eval_start_ms": EVAL_START, "dev_end_ms": DEV_END, "holdout_end_ms": HOLDOUT_END},
        "selection": final_log, "trials": lab.trials, "gate": GATE,
        "gate_result": {k: {"value": v, "ok": ok} for k, (v, ok) in gate_checks.items()}, "passed": passed,
        "code": {"git_revision": revision, "lab_or_offline_dirty": dirty},
        "data": _manifest(data_dir),
    }
    # Freeze BEFORE looking at the secondary holdout.
    (out / "final_spec.json").write_text(json.dumps(frozen, indent=2, default=float) + "\n")
    hold = _stats(_returns(dev["daily"], day_of(DEV_END), day_of(HOLDOUT_END)))
    hold_bench = _stats(bench_r[(bench_r.index >= day_of(DEV_END)) & (bench_r.index < day_of(HOLDOUT_END))])
    summary = {
        "walk_forward": {"folds": folds, "oos": wf, "oos_cost_x2": wf2},
        "development": {"portfolio": summarize(dev["daily"], start_day=first_day, end_day=day_of(DEV_END)),
                        "per_symbol": per_symbol, "fills_per_year": fills,
                        "exposure": exposure([r.curve[r.curve["at"] < DEV_END] for r in dev["sleeves"]]),
                        "yearly": {k: v for k, v in yearly(dev["daily"]).items() if k < 2026},
                        "benchmark_50_50_buy_hold": _stats(bench_dev), "correlation_to_benchmark": corr,
                        "deflated_sharpe_probability": dsr, "trials": lab.trials},
        "secondary_holdout": {"strategy": hold, "benchmark_50_50_buy_hold": hold_bench,
                              "note": "viewed once after freezing; not used for selection or gating"},
        "gate": {"passed": passed, "checks": {k: {"value": v, "ok": ok} for k, (v, ok) in gate_checks.items()}},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    (out / "candidates.json").write_text(json.dumps(
        {"stage_a": final_log["stage_a"], "stage_b": final_log["stage_b"], "stage_c": final_log["stage_c"],
         "walk_forward_folds": folds}, indent=2, default=float) + "\n")
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


__all__ = ["run_study", "select", "STAGE_A", "LAYERS", "EXECUTIONS", "GATE", "np"]
