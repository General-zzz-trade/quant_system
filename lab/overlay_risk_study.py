"""Risk management of the frozen long overlay: FOUR predeclared variants, ONE predeclared gate, development data.

    python -m lab.overlay_risk_study --data-dir data_files/binance_public --out research/results/overlay_risk_<date>

Everything that decides the outcome is fixed in this file and in
research/protocols/2026-11-15-overlay-risk.md, committed before any result on real data was computed.

- **Question.** Does another way of controlling risk cut the overlay's drawdown clearly without giving up its
  return per unit of risk? The frozen overlay (``lab.overlay_study.SPEC``) stays in its forward test unchanged;
  it is the reference here.
- **Variants (parameters fixed, no search).**
  - ``ewma_vol``: daily volatility from an exponentially weighted estimate, half-life 10 days (30 days of
    history before the first value), instead of the 30-day simple estimate.
  - ``portfolio_vol``: a correlation-aware cap on both symbols. The portfolio volatility predicted with the
    current 30-day correlation is compared with the same weights at the long-run average correlation (the
    expanding mean of the 30-day correlation, known at the time); when the current correlation is higher,
    both weights are scaled down by the ratio. Never scaled up.
  - ``drawdown_brake``: the whole portfolio's weights are multiplied by 1 while the strategy's own equity is
    at most 5 % below its running peak, falling linearly to 0.25 at 15 % and staying 0.25 beyond.
  - ``combined``: all three together.
- **Gate (all must hold, each against the frozen overlay on the same run).** See ``GATE``.
- **Window.** Development data only, 2021-01-10 to 2025-11-16. Bars after ``DEV_END`` never enter a
  simulation or a statistic.
- **Selection.** If more than one variant passes, the highest Calmar wins. Each variant is one trial in
  the ``directional_btc_eth`` family; the deflated Sharpe over that family is reported only.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lab.data import DAY, funding_series, load_symbol
from lab.metrics import exposure
from lab.overlay_study import SLEEVE_BALANCE, SPEC, _code_identity, portfolio_equity, stats
from lab.simulate import SYMBOL_RULES, Sleeve, SleeveResult, broker_config
from lab.study import DEV_END, EVAL_START, SYMBOLS, _manifest
from trading.domain.market import group_closes

FAMILY = "directional_btc_eth"
STUDY = "overlay_risk_2026-11-15"
EWMA_HALFLIFE_DAYS = 10
CORR_LOOKBACK_DAYS = 30
BRAKE = (0.05, 0.15, 0.25)          # full exposure up to 5 % drawdown, linear to 0.25 at 15 %, 0.25 beyond
VARIANTS = {
    "ewma_vol": {"vol": "ewma", "portfolio": False, "brake": False},
    "portfolio_vol": {"vol": "simple", "portfolio": True, "brake": False},
    "drawdown_brake": {"vol": "simple", "portfolio": False, "brake": True},
    "combined": {"vol": "ewma", "portfolio": True, "brake": True},
}
GATE = {
    "max_drawdown_at_most_ratio_of_overlay": 0.85,
    "sharpe_shortfall_vs_overlay_max": 0.05,
    "cost_x2_total_return_min_exclusive": 0.0,
    "fills_per_symbol_year_max": 150,
    "years_drawdown_not_worse_min": 4,          # of the 5 calendar years 2021-2025
}


# -- target weights -------------------------------------------------------------------------------------------
def daily_frame(hourly: pd.DataFrame, vol: str) -> pd.DataFrame:
    """The overlay's daily weight, with the simple or the EWMA volatility (same score, same caps)."""
    daily = group_closes(hourly, 24)
    close = daily["close"]
    logret = np.log(close).diff()
    if vol == "ewma":
        vol_d = logret.ewm(halflife=EWMA_HALFLIFE_DAYS, min_periods=SPEC.vol_lookback).std()
    elif vol == "simple":
        vol_d = logret.rolling(SPEC.vol_lookback, min_periods=SPEC.vol_lookback).std()
    else:
        raise ValueError(vol)
    score = pd.concat([np.sign(close / close.shift(lb) - 1.0) for lb in SPEC.lookbacks], axis=1)
    score = score.mean(axis=1, skipna=False).clip(lower=0.0)
    weight = (score * SPEC.target_vol / (vol_d * math.sqrt(365.0))).clip(-SPEC.max_weight, SPEC.max_weight)
    return pd.DataFrame({"available_at": daily["available_at"].to_numpy(dtype=np.int64),
                         "weight": weight.to_numpy(dtype=float), "logret": logret.to_numpy(dtype=float)})


def portfolio_multiplier(frames: dict[str, pd.DataFrame]) -> pd.Series:
    """Per daily close: min(1, predicted portfolio vol at the average correlation / at the current one)."""
    a, b = (frames[s].set_index("available_at") for s in SYMBOLS)
    joined = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
    ra, rb = joined["logret_a"], joined["logret_b"]
    sa = ra.rolling(CORR_LOOKBACK_DAYS, min_periods=CORR_LOOKBACK_DAYS).std()
    sb = rb.rolling(CORR_LOOKBACK_DAYS, min_periods=CORR_LOOKBACK_DAYS).std()
    rho = ra.rolling(CORR_LOOKBACK_DAYS, min_periods=CORR_LOOKBACK_DAYS).corr(rb)
    rho_bar = rho.expanding(min_periods=1).mean()          # long-run average known at each close
    xa, xb = joined["weight_a"].fillna(0.0) * sa, joined["weight_b"].fillna(0.0) * sb
    now = xa ** 2 + xb ** 2 + 2 * xa * xb * rho
    usual = xa ** 2 + xb ** 2 + 2 * xa * xb * rho_bar
    ratio = np.sqrt(usual / now).where(now > 0, 1.0)
    return ratio.clip(upper=1.0).where(rho.notna(), np.nan)


def brake_multiplier(drawdown: float) -> float:
    start, full, floor = BRAKE
    if not math.isfinite(drawdown) or drawdown < 0:
        raise ValueError("drawdown must be a finite non-negative fraction")
    if drawdown <= start:
        return 1.0
    if drawdown >= full:
        return floor
    return 1.0 - (1.0 - floor) * (drawdown - start) / (full - start)


def _asof(times: np.ndarray, at: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The last value known at or before each time (NaN before the first)."""
    idx = np.searchsorted(at, times, side="right") - 1
    out = np.full(len(times), np.nan)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def hourly_targets(hourly: dict[str, pd.DataFrame], variant: dict) -> dict[str, pd.DataFrame]:
    """Per symbol: available_at, target (before the drawdown brake), ready."""
    frames = {s: daily_frame(hourly[s], variant["vol"]) for s in SYMBOLS}
    mult = portfolio_multiplier(frames) if variant["portfolio"] else None
    out = {}
    for s in SYMBOLS:
        times = hourly[s]["available_at"].to_numpy(dtype=np.int64)
        day_at = frames[s]["available_at"].to_numpy(dtype=np.int64)
        w = _asof(times, day_at, frames[s]["weight"].to_numpy(dtype=float))
        if mult is not None:
            m = _asof(times, mult.index.to_numpy(dtype=np.int64), mult.to_numpy(dtype=float))
            w = w * m
        ready = np.isfinite(w)
        out[s] = pd.DataFrame({"available_at": times, "target": np.where(ready, w, 0.0), "ready": ready})
    return out


# -- one joint simulation of both sleeves ----------------------------------------------------------------------
@dataclass
class Run:
    sleeves: list[SleeveResult]
    equity: pd.Series


def simulate_portfolio(bars: dict, targets: dict, funding: dict, *, brake: bool, start_ms: int, end_ms: int,
                       cost_multiplier: int = 1) -> list[SleeveResult]:
    """Both sleeves bar by bar, exactly as ``lab.simulate.simulate`` steps one; the brake scales both targets
    by the portfolio's own drawdown at the decision close (known then: equity after this bar)."""
    sleeves = {s: Sleeve(s, SPEC, SYMBOL_RULES[s], broker_config(SPEC, SYMBOL_RULES[s],
                                                                  starting_balance=SLEEVE_BALANCE,
                                                                  cost_multiplier=cost_multiplier))
               for s in SYMBOLS}
    target_maps = {s: dict(zip(targets[s]["available_at"].tolist(),
                               zip(targets[s]["target"].tolist(), targets[s]["ready"].tolist()))) for s in SYMBOLS}
    by_time: dict[int, dict] = {}
    for s in SYMBOLS:
        for bar in bars[s]:
            if bar.available_at <= start_ms or bar.open_time >= end_ms:
                continue
            by_time.setdefault(bar.available_at, {})[s] = bar
    rows: dict[str, list] = {s: [] for s in SYMBOLS}
    peak = 0.0
    for at in sorted(by_time):
        step = by_time[at]
        if set(step) != set(SYMBOLS):
            raise ValueError(f"bars of both symbols are required at every close ({at})")
        for s in SYMBOLS:
            bar = step[s]
            rate = funding[s][bar.open_time][0] if bar.open_time in funding[s] else None
            sleeves[s].on_bar(bar, rate)
        total = sum(float(sleeves[s].broker.equity) for s in SYMBOLS)
        peak = max(peak, total)
        scale = brake_multiplier((peak - total) / peak) if brake else 1.0
        for s in SYMBOLS:
            bar = step[s]
            target, ready = target_maps[s].get(at, (0.0, False))
            sleeves[s].decide(bar, target * scale, ready)
            b = sleeves[s].broker
            rows[s].append((at, float(b.equity), float(b.position.qty), float(b.mark), float(b.fees),
                            float(b.funding)))
    return [SleeveResult(s, pd.DataFrame(rows[s], columns=["at", "equity", "qty", "mark", "fees", "funding"]),
                         list(sleeves[s].broker.fills), sleeves[s].orders) for s in SYMBOLS]


# -- gate ------------------------------------------------------------------------------------------------------
def yearly_max_drawdown(equity: pd.Series) -> dict[int, float]:
    """Max drawdown inside each calendar year (the peak restarts at the year's first close)."""
    years = pd.to_datetime(equity.index.to_numpy() * DAY, unit="ms", utc=True).year
    out = {}
    for year in sorted(set(years)):
        part = equity[years == year]
        peak = part.cummax()
        out[int(year)] = float(((peak - part) / peak).max())
    return out


def gate_checks(variant: dict, overlay: dict, cost_x2: dict, fills_per_year: dict,
                years_variant: dict, years_overlay: dict) -> dict:
    worst_fills = max(fills_per_year.values())
    common = sorted(set(years_variant) & set(years_overlay))
    not_worse = sum(years_variant[y] <= years_overlay[y] for y in common)
    checks = {
        "max_drawdown_vs_overlay": (
            variant["max_drawdown"], overlay["max_drawdown"],
            bool(variant["max_drawdown"] <= GATE["max_drawdown_at_most_ratio_of_overlay"] * overlay["max_drawdown"])),
        "sharpe_vs_overlay": (
            variant["sharpe"], overlay["sharpe"],
            bool(variant["sharpe"] >= overlay["sharpe"] - GATE["sharpe_shortfall_vs_overlay_max"])),
        "cost_x2_total_return": (
            cost_x2["total_return"], GATE["cost_x2_total_return_min_exclusive"],
            bool(cost_x2["total_return"] > GATE["cost_x2_total_return_min_exclusive"])),
        "fills_per_symbol_year": (
            worst_fills, GATE["fills_per_symbol_year_max"], bool(worst_fills <= GATE["fills_per_symbol_year_max"])),
        "years_drawdown_not_worse": (
            not_worse, GATE["years_drawdown_not_worse_min"], bool(not_worse >= GATE["years_drawdown_not_worse_min"])),
    }
    return {name: {"value": v, "reference": ref, "ok": ok} for name, (v, ref, ok) in checks.items()}


def select(results: dict[str, dict]) -> str | None:
    """The passing variant with the highest Calmar, or None."""
    passing = [(r["strategy"]["calmar"], name) for name, r in results.items() if r["gate"]["passed"]]
    return max(passing)[1] if passing else None


# -- the study -------------------------------------------------------------------------------------------------
def run_study(data_dir: Path, out: Path, *, eval_start: int = EVAL_START, dev_end: int = DEV_END,
              registry: Path | None = None) -> dict:
    from lab import trials
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("use an empty output directory; preserve previous studies")
    if eval_start % DAY or dev_end % DAY or not eval_start < dev_end:
        raise ValueError("evaluation window must be whole UTC days")
    out.mkdir(parents=True, exist_ok=True)
    bars, funding, hourly = {}, {}, {}
    for s in SYMBOLS:
        bars[s], funding[s], hourly[s] = load_symbol(data_dir, s)
        hourly[s] = hourly[s][hourly[s]["open_time"] < dev_end].reset_index(drop=True)
        funding_series(funding[s])                     # validates the settlements
    first_day, initial = eval_start // DAY, float(len(SYMBOLS)) * float(SLEEVE_BALANCE)

    def run(variant: dict, cost_multiplier: int = 1) -> Run:
        sleeves = simulate_portfolio(bars, hourly_targets(hourly, variant), funding, brake=variant["brake"],
                                     start_ms=eval_start, end_ms=dev_end, cost_multiplier=cost_multiplier)
        equity = portfolio_equity(sleeves, first_day, initial)
        if equity.index[-1] >= dev_end // DAY:
            raise RuntimeError("development statistics reached the held-out window")
        return Run(sleeves, equity)

    reference = {"vol": "simple", "portfolio": False, "brake": False}
    base = run(reference)
    base_stats, base_years = stats(base.equity), yearly_max_drawdown(base.equity)
    days = len(base.equity) - 1
    results = {}
    for name, variant in VARIANTS.items():
        r = run(variant)
        x2 = run(variant, cost_multiplier=2)
        fills = {s.symbol: len(s.fills) / days * 365.0 for s in r.sleeves}
        s_stats, years = stats(r.equity), yearly_max_drawdown(r.equity)
        checks = gate_checks(s_stats, base_stats, stats(x2.equity), fills, years, base_years)
        results[name] = {"variant": variant, "strategy": s_stats, "strategy_cost_x2": stats(x2.equity),
                         "average_exposure": exposure([s.curve for s in r.sleeves]), "fills_per_symbol_year": fills,
                         "yearly_max_drawdown": years,
                         "gate": {"passed": all(c["ok"] for c in checks.values()), "checks": checks},
                         "daily_returns": r.equity.pct_change().dropna()}
    rows = [{"family": FAMILY, "study": STUDY, "trial": name, "config": r["variant"],
             "window": [eval_start, dev_end], "sharpe": r["strategy"]["sharpe"], "count": 1}
            for name, r in results.items()]
    for name, r in results.items():
        r["deflated_sharpe_report_only"] = trials.deflated(r.pop("daily_returns"), FAMILY, extra=rows,
                                                           **({"path": registry} if registry else {}))
    trials.record(rows, **({"path": registry} if registry else {}))
    winner = select(results)
    summary = {
        "study": STUDY, "window": {"eval_start_ms": eval_start, "dev_end_ms": dev_end,
                                   "holdout": "not computed; the final test is forward data at registered looks"},
        "reference_overlay": {"spec": SPEC.record(), "strategy": base_stats, "yearly_max_drawdown": base_years,
                              "average_exposure": exposure([s.curve for s in base.sleeves])},
        "gate": GATE, "variants": results, "selected": winner,
        "parameters": {"ewma_halflife_days": EWMA_HALFLIFE_DAYS, "corr_lookback_days": CORR_LOOKBACK_DAYS,
                       "brake": BRAKE},
        "code": _code_identity(), "data": _manifest(data_dir),
        "caveat": ("The overlay's development results, its yearly returns and the lab's funding, 4h-volatility "
                   "and 4h-confirmation layers were seen before these variants were written; this is a screen. "
                   "A passing variant only earns a forward test; the frozen overlay is unchanged."),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data_files/binance_public"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_study(args.data_dir, args.out)
    print(json.dumps({"selected": summary["selected"],
                      "gate": {k: v["gate"]["passed"] for k, v in summary["variants"].items()}}, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["VARIANTS", "GATE", "BRAKE", "run_study", "simulate_portfolio", "brake_multiplier",
           "portfolio_multiplier", "daily_frame", "hourly_targets", "gate_checks", "select", "yearly_max_drawdown"]
