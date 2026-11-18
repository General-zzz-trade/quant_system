"""The weekend gap of the US-stock perpetuals.

Protocol: research/protocols/2026-11-18-us-weekend.md, committed before any result.

    python research/results/us_weekend_2026-11-18/run.py screen   # 6 configurations on the selection weekends
    python research/results/us_weekend_2026-11-18/run.py final    # the survivor only: the holdout, once

Writes results.json next to this file and appends the 6 trials to research/trials.jsonl.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from lab import trials as registry  # noqa: E402
from lab import um_lab  # noqa: E402
from lab import us_weekend as wk  # noqa: E402

PERPS = ROOT / "data_files" / "binance_usdt_perps_2020_2026-11-12"
STOCKS = ROOT / "data_files" / "us_equity_daily_2026-11-12"
TRADFI = ROOT / "research" / "lab" / "runs" / "2026-11-17b" / "tradfi_contracts.json"
START, SPLIT, END = date(2025, 12, 1), date(2026, 8, 13), date(2026, 11, 13)
CONFIGS = [{"threshold": x, "exit": e} for x in (0.01, 0.02, 0.03) for e in wk.EXIT_RULES]
MIN_QUOTE, DRAWS = 5e6, 10_000      # costs: the shared broker's 5 + 3 bp a side; stress: times 2, funding times 2
FAMILY, STUDY = "us_weekend_2026-11-18", "us_weekend_2026-11-18"


def label(c: dict) -> str:
    return f"fade{int(c['threshold'] * 100)}_{c['exit']}"


def tickers() -> list[str]:
    underlying = json.loads(TRADFI.read_text())["underlying"]
    return sorted(s[:-4] for s, kind in underlying.items() if kind == "EQUITY"
                  and (STOCKS / f"{s[:-4]}_1d.csv").is_file() and (PERPS / f"{s}_1h.csv.gz").is_file())


def events(holdout: bool) -> list[dict]:
    """Every event of every ticker in the selection (or the holdout), for both exit rules."""
    out = []
    for t in tickers():
        bars = pd.read_csv(PERPS / f"{t}USDT_1h.csv.gz").set_index("open_time").sort_index()
        path = PERPS / f"{t}USDT_funding.csv.gz"
        fund = pd.read_csv(path) if path.is_file() else pd.DataFrame({"funding_time": [], "rate": []})
        funding = pd.Series(fund["rate"].to_numpy(float), index=fund["funding_time"].to_numpy(np.int64)).sort_index()
        stock = pd.read_csv(STOCKS / f"{t}_1d.csv")
        stock["day"] = pd.to_datetime(stock["date"]).dt.date
        stock = stock.set_index("day")
        rules = um_lab.rules(f"{t}USDT", PERPS)
        for d0, d1 in wk.breaks(list(stock.index)):
            if d0 < START or not ((SPLIT <= d1 < END) if holdout else d1 < SPLIT):
                continue
            for rule in wk.EXIT_RULES:
                e = wk.event(bars, funding, d0, d1, exit_rule=rule)
                if e is None or not wk.tracks(e["reference_price"], float(stock.at[d0, "close"])):
                    continue
                if wk.median_daily_quote(bars, e["decision"]) < MIN_QUOTE:
                    continue
                gap = float(stock.at[d1, "open"]) / float(stock.at[d0, "close"]) - 1.0
                base = wk.replay(bars, funding, e, rules=rules)
                stress = wk.replay(bars, funding, e, rules=rules, cost_multiplier=2, funding_multiplier=2.0)
                out.append({**e, "ticker": t, "exit_rule": rule, "stock_gap": gap, "net_base": base,
                            "net_stress": stress})
    return out


def trades(evts: list[dict], c: dict, *, stress: bool = False) -> pd.DataFrame:
    """The configuration's trades: the events of its exit rule with a move of at least its threshold, at the shared
    broker's result (base or stress costs); an event the broker could not fill is not a trade."""
    rows = []
    for e in evts:
        r = e["net_stress"] if stress else e["net_base"]
        if e["exit_rule"] == c["exit"] and wk.trades_at(e, threshold=c["threshold"]) and r is not None:
            rows.append({"weekend": e["d1"], "ticker": e["ticker"], "net": r})
    return pd.DataFrame(rows, columns=["weekend", "ticker", "net"])


def stats(tr: pd.DataFrame, weekends: list[str]) -> dict:
    """Per event and per weekend (an equal-weight mean of the weekend's trades; a weekend without one is 0)."""
    if tr.empty:
        return {"events": 0, "weekends_traded": 0, "mean_net": None, "t_weekend": None, "hit_rate": None}
    by = tr.groupby("weekend")["net"].mean()
    n = len(by)
    t = float(by.mean() / by.std(ddof=1) * math.sqrt(n)) if n > 1 and by.std(ddof=1) > 0 else 0.0
    series = by.reindex(weekends).fillna(0.0)
    sharpe = float(series.mean() / series.std(ddof=1) * math.sqrt(52)) if series.std(ddof=1) > 0 else 0.0
    return {"events": int(len(tr)), "weekends_traded": n, "mean_net": float(tr["net"].mean()),
            "median_net": float(tr["net"].median()), "hit_rate": float((tr["net"] > 0).mean()), "t_weekend": t,
            "weekly_sharpe": sharpe, "mean_weekend": float(by.mean())}


def permutation_p(tr: pd.DataFrame, draws: int = DRAWS, seed: int = 0) -> float:
    """Flip each weekend's sign at random: the share of draws whose mean weekend return is at least the observed."""
    by = tr.groupby("weekend")["net"].mean().to_numpy()
    observed = by.mean()
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(draws, len(by)))
    return float(((signs * by).mean(axis=1) >= observed).sum() + 1) / (draws + 1)


def commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def screen() -> dict:
    evts = events(holdout=False)
    weekends = sorted({e["d1"] for e in evts})
    rows = {label(c): {"base": stats(trades(evts, c), weekends),
                       "stress": stats(trades(evts, c, stress=True), weekends)} for c in CONFIGS}
    best = max(CONFIGS, key=lambda c: rows[label(c)]["base"]["t_weekend"] or -9.0)
    tr = trades(evts, best)
    s, st = rows[label(best)]["base"], rows[label(best)]["stress"]
    top = tr.groupby("ticker")["net"].sum().idxmax() if not tr.empty else None
    without = tr[tr["ticker"] != top]["net"].mean() if not tr.empty else float("nan")
    half = weekends[len(weekends) // 2] if weekends else ""
    halves = [tr[tr["weekend"] < half]["net"].mean(), tr[tr["weekend"] >= half]["net"].mean()] if not tr.empty \
        else [float("nan")] * 2
    p = permutation_p(tr) if not tr.empty else 1.0
    checks = {"mean_net_above_0": (s["mean_net"] or 0) > 0, "t_at_least_2": (s["t_weekend"] or 0) >= 2.0,
              "permutation_p_at_most_0.05_over_6": p <= 0.05 / len(CONFIGS),
              "stress_mean_above_0": (st["mean_net"] or 0) > 0,
              "at_least_100_events_15_weekends": s["events"] >= 100 and s["weekends_traded"] >= 15,
              "without_top_ticker_above_0": bool(without > 0), "both_halves_above_0": bool(min(halves) > 0)}
    moves = pd.DataFrame([{"move": e["move"], "gap": e["stock_gap"]} for e in evts if e["exit_rule"] == "close"])
    done = {(r["study"], r["trial"]) for r in registry.load(ROOT / "research/trials.jsonl", FAMILY)}
    registry.record([{"family": FAMILY, "study": STUDY, "trial": label(c), "config": c,
                      "window": [START.isoformat(), SPLIT.isoformat()],
                      "sharpe": rows[label(c)]["base"]["weekly_sharpe"],
                      "count": 1, "note": "selection weekends, base costs; weekly Sharpe; holdout not read"}
                     for c in CONFIGS if (STUDY, label(c)) not in done], path=ROOT / "research/trials.jsonl")
    return {"events_selection": len({(e["ticker"], e["d1"]) for e in evts}), "weekends_selection": len(weekends),
            "tickers": len({e["ticker"] for e in evts}), "configs": rows, "chosen": label(best),
            "permutation_p": p, "top_ticker": top, "mean_without_top_ticker": float(without),
            "halves_mean_net": [float(h) for h in halves],
            "report_only": {"corr_weekend_move_vs_stock_gap": float(moves["move"].corr(moves["gap"]))
                            if len(moves) > 2 else None, "events_for_corr": int(len(moves))},
            "checks": checks, "survives": all(checks.values())}


def final(screened: dict) -> dict:
    if not screened["survives"]:
        return {"opened_holdout": False, "reason": "no configuration passed the line"}
    c = next(c for c in CONFIGS if label(c) == screened["chosen"])
    evts = events(holdout=True)
    weekends = sorted({e["d1"] for e in evts})
    s = stats(trades(evts, c), weekends)
    checks = {"holdout_mean_above_0": (s["mean_net"] or 0) > 0, "holdout_t_at_least_1": (s["t_weekend"] or 0) >= 1.0}
    return {"opened_holdout": True, "choice": label(c), "holdout": s,
            "holdout_stress": stats(trades(evts, c, stress=True), weekends),
            "checks": checks, "passes": all(checks.values())}


def main(step: str) -> None:
    path = HERE / "results.json"
    if step == "screen":
        path.write_text(json.dumps({"code_commit": commit(), "screen": screen()}, indent=1, default=str))
    elif step == "final":
        out = json.loads(path.read_text())
        out["final"], out["final_commit"] = final(out["screen"]), commit()
        path.write_text(json.dumps(out, indent=1, default=str))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
