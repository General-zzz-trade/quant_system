"""The alpha check of the current strategies (plan.md, committed before any result): a diagnostic, nothing selected.

    python research/results/alpha_check_2026-11-18/run.py      # -> results.json

The two lab runs from their caches (no new replay); the weak portfolio from its history check; the three forward
candidates of 2026-11-14 replayed once with their own code, settings and development windows (not new trials).
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from lab import candidates_study as cs  # noqa: E402
from lab import gates, trials  # noqa: E402
from lab import pair_study  # noqa: E402
from lab import um_judge as judge  # noqa: E402
from lab import um_lab as lab  # noqa: E402
from lab import um_monthly  # noqa: E402
from lab import um_portfolio as portfolio  # noqa: E402
from lab.candidates import flush_targets  # noqa: E402
from lab.pair import SYMBOLS, build, load_inputs  # noqa: E402
from lab.study import DEV_END  # noqa: E402

SNAP = ROOT / "data_files" / "binance_usdt_perps_2020_2026-11-12"
DATA, PREMIUM = ROOT / "data_files" / "binance_public", ROOT / "data_files" / "binance_premium_2025-11-16"
RUNS = ("2026-11-17c", "2026-11-17")
DAY = lab.DAY


def commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def run_module(name: str):
    spec = importlib.util.spec_from_file_location(f"judge_{name.replace('-', '_')}",
                                                  ROOT / "research" / "lab" / "runs" / name / "judge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lab_run(name: str) -> dict:
    """``lab.um_monthly.line`` on a run's cache, plus each idea's walk-forward, beta and the hold's drawdown."""
    m = run_module(name)
    j = m.make()
    screened = json.loads((ROOT / "research" / "lab" / "runs" / name / "screen.json").read_text())
    verdict = um_monthly.line(j, screened, variants=getattr(m, "VARIANTS", judge.STOP_VARIANTS))
    ideas = {}
    for idea, v in verdict["ideas"].items():
        wf = screened["ideas"][idea]["walk_forward"]
        ideas[idea] = {"walk_forward": {k: wf.get(k) for k in ("days", "total_return", "sharpe", "max_drawdown")},
                       "regression_alpha": v["regression_alpha"], "portfolio_gain": v["portfolio_gain"],
                       "family_deflated": v["family_deflated"], "power": v["power"], **classify(v)}
    return {"benchmarks": verdict["benchmarks"], "ideas": ideas}


def classify(v: dict) -> dict:
    evidence = bool(v["regression_alpha"]["passes"] and v["portfolio_gain"]["passes"])
    verified = bool(evidence and v["family_deflated"] is not None and v["family_deflated"]["passes"])
    return {"alpha_evidence": evidence, "verified_alpha": verified,
            "class": "验证过的 alpha" if verified else ("有 alpha 证据" if evidence else "没有 alpha 证据")}


def benchmarks(start: int, end: int) -> pd.DataFrame:
    hold = lab.buy_and_hold(judge.BTC, start, end, SNAP)
    vol = portfolio.vol_target_hold(SNAP, [(start, end)], capital="150")[0]
    return pd.DataFrame({"btc_hold": hold, "btc_vol_target": vol}).fillna(0.0)


def family_sharpes(family: str) -> list[float]:
    return [float(r["sharpe"]) for r in trials.load(trials.REGISTRY, family)
            if r.get("sharpe") is not None and math.isfinite(float(r["sharpe"]))]


def measure(daily: pd.Series, start: int, end: int, family: str, recorded_sharpe: float) -> dict:
    daily = daily.copy()
    daily.index = daily.index.astype(np.int64)
    bench = benchmarks(start, end)
    s = lab.summary(daily)
    sd, n = float(daily.std(ddof=1)), len(daily)
    v = {"walk_forward": {k: s.get(k) for k in ("days", "total_return", "sharpe", "max_drawdown")},
         "recorded_sharpe": recorded_sharpe, "regression_alpha": gates.regression_alpha(daily, bench),
         "portfolio_gain": gates.portfolio_gain(daily, bench["btc_hold"]),
         "family_deflated": gates.family_deflated(daily, family_sharpes(family)),
         "power": {"at_sharpe_0.5": gates.power(0.5 / math.sqrt(365) * sd, sd, n) if sd > 0 else 0.0},
         "btc_hold": lab.summary(bench["btc_hold"].reindex(daily.index).fillna(0.0))}
    return {**v, **classify(v)}


def candidates() -> dict:
    inputs = load_inputs(DATA, PREMIUM, DEV_END)
    recorded = json.loads((ROOT / "research/results/forward_candidates_2026-11-14/summary.json").read_text())
    data = cs.b_dataset(inputs)
    preds, _ = cs.b_walk_forward(data, first_test=cs.B_START, limit=DEV_END)
    b = cs.run_sleeves(inputs, cs.b_targets(inputs, data, preds, cs.B_START), start=cs.B_START, end=DEV_END)
    signals = {s: flush_targets(inputs["hourly"][s], inputs["metrics"][s]) for s in SYMBOLS}
    c = cs.run_sleeves(inputs, signals, start=cs.C_START, end=DEV_END)
    closes, feats = build(inputs)
    cand = next(x for x in pair_study.CANDIDATES if x["id"] == "ridge_PD_7d")
    signal, _ = pair_study.ml_predictions(feats, closes, cand, oos_start=pair_study.OOS_START, limit=DEV_END)
    sides = pair_study.decisions(signal, cand["horizon"], start=pair_study.OOS_START, end=DEV_END)
    run = pair_study.simulate_pair(pair_study.merged_hourly(inputs["hourly"]), inputs["funding"], sides,
                                   start=pair_study.OOS_START, end=DEV_END, capital=pair_study.CAPITAL)
    pair = pair_study.describe(run, pair_study.OOS_START, pair_study.CAPITAL)
    pair_recorded = json.loads((ROOT / "research/results/pair_2026-11-14/summary.json").read_text())
    return {"dir_ridge_7d": measure(b["_daily_returns"], cs.B_START, DEV_END, cs.FAMILY,
                                    recorded["dir_ridge_7d"]["base"]["sharpe"]),
            "flush_reversal_72h": measure(c["_daily_returns"], cs.C_START, DEV_END, cs.FAMILY,
                                          recorded["flush_reversal_72h"]["base"]["sharpe"]),
            "pair_ridge_PD_7d": measure(pair["_daily_returns"], pair_study.OOS_START, DEV_END, pair_study.FAMILY,
                                        _pair_sharpe(pair_recorded))}


def _pair_sharpe(summary: dict) -> float:
    return float(summary["candidates"]["ridge_PD_7d"]["base"]["sharpe"])


def weak_portfolio() -> dict:
    h = json.loads((ROOT / "research/results/weak_portfolio_2026-11-18/history.json").read_text())
    v = {"walk_forward": h["walk_forward"], "regression_alpha": h["regression_alpha"],
         "portfolio_gain": h["portfolio_gain"], "family_deflated": None, "power": h["power"],
         "mean_member_correlation": h["mean_member_correlation"]}
    return {**v, **classify(v)}


def main() -> None:
    out = {"code_commit": commit(), "plan": "research/results/alpha_check_2026-11-18/plan.md"}
    out["candidates_2026-11-14"] = candidates()
    print("candidates done", flush=True)
    out["weak_portfolio_2026-11-18"] = weak_portfolio()
    for name in RUNS:
        out[f"lab_{name}"] = lab_run(name)
        print(name, "done", flush=True)
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
