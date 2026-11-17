"""The BTC/ETH feature-pool study (README.md, registered before any result): can the pool tell a trend is coming,
how volatile the next days will be, or which way the coin goes; and, where a trend can be told, does filtering the
frozen overlay with it pay. Everything known before 2025-11-16 only."""
import json
import sys
from decimal import Decimal as D
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lab import feature_pool as fp  # noqa: E402
from lab import feature_study as st  # noqa: E402
from lab import pilot_params as pp  # noqa: E402
from lab import trials  # noqa: E402
from lab import um_judge  # noqa: E402
from lab.data import funding_series, load_symbol  # noqa: E402
from lab.metrics import daily_equity  # noqa: E402
from lab.overlay_study import SPEC  # noqa: E402
from lab.strategy import compute_targets  # noqa: E402
from lab.study import DEV_END, EVAL_START  # noqa: E402
from trading.adapters.simulated.bar_broker import SymbolRules  # noqa: E402

PUBLIC = ROOT / "data_files" / "binance_public"
FEATURES = ROOT / "data_files" / "btc_eth_features_2025-11-16"
MARKET = ROOT / "data_files" / "binance_usdt_perps_2020_2026-11-12"
TRADFI = json.loads((ROOT / "research/lab/runs/2026-11-17b/tradfi_contracts.json").read_text())["symbols"]
OUT = Path(__file__).with_name("results.json")
DAY = fp.DAY
LIMIT = DEV_END
YEARS = (2021, 2022, 2023, 2024, 2025)
TARGETS = (("trend", 14, "binary"), ("up", 14, "binary"), ("fwd_vol", 7, "regression"), ("fwd_vol", 14, "regression"))
MODELS = {"binary": ("logistic", "lgbm"), "regression": ("ridge", "lgbm")}
NULL_RUNS, BLOCK, PCT = 40, 30, 97.5
MIN_AUC, MIN_GAIN, MIN_YEARS = 0.55, 0.02, 3
RULES = {"BTCUSDT": SymbolRules(D("0.001"), D("0.001"), D("50")), "ETHUSDT": SymbolRules(D("0.001"), D("0.001"),
                                                                                         D("20"))}
FAMILY, STUDY = "feature_pool_btc_eth", "feature_pool_2026-11-17"


def screen(x: pd.DataFrame, y: pd.Series) -> dict:
    """Each feature's rank correlation with the target in each year, and how many keep one sign in 4 of 5 years."""
    years = pd.to_datetime(x.index.to_numpy(dtype=np.int64), unit="ms").year
    ic = {}
    for col in x.columns:
        per = []
        for year in YEARS:
            m = (years == year) & x[col].notna().to_numpy() & y.notna().to_numpy()
            per.append(float(spearmanr(x[col][m], y[m]).statistic) if m.sum() > 30 else np.nan)
        ic[col] = per
    stable = [c for c, v in ic.items() if max(np.nansum(np.sign(v) > 0), np.nansum(np.sign(v) < 0)) >= 4]
    return {"ic_by_year": ic, "stable": stable}


def main():
    closes_m, funding_m = fp.load_market(MARKET, LIMIT, exclude=frozenset(TRADFI) | frozenset(um_judge.STABLE))
    breadth = fp.market_breadth(closes_m, funding_m)
    inputs = fp.load_inputs(PUBLIC, FEATURES, LIMIT, breadth=breadth)
    times = np.arange(EVAL_START - EVAL_START % DAY + DAY, LIMIT + 1, DAY, dtype=np.int64)
    out = {"limit": "2025-11-16", "market_contracts": int(closes_m.shape[1]), "assets": {}}
    rows = []
    preds = {}
    for asset in fp.SYMBOLS:
        x = fp.build(inputs, asset, times)
        x["log_rv_30"] = np.log(x["rv_30"])
        closes = fp.daily(inputs["hourly"][asset])["close"]
        result = {"coverage": {c: float(x[c].notna().mean()) for c in x.columns}, "targets": {}}
        for name, horizon, kind in TARGETS:
            y_all = st.targets(closes, horizon).reindex(times)
            y, label_end = y_all[name], y_all["label_end"].fillna(pd.Series(times + horizon * DAY, index=times))
            key = f"{name}_{horizon}"
            shuffled = st._shuffled(y.dropna(), BLOCK, np.random.default_rng(1)).reindex(y.index)
            entry = {"kind": kind, "screen": screen(x, y), "screen_shuffled_stable": len(screen(x, shuffled)["stable"]),
                     "models": {}}
            for model in MODELS[kind]:
                naive = "log_rv_30" if kind == "regression" else None
                ev = st.evaluate(x, y, label_end, YEARS, kind=kind, model=model, naive=naive)
                null = st.null_scores(x, y, label_end, YEARS, kind=kind, model=model, runs=NULL_RUNS, block=BLOCK,
                                      seed=7, naive=naive)
                score = ev["auc"] if kind == "binary" else ev["r2_gain"]
                floor, threshold = (MIN_AUC, 0.5) if kind == "binary" else (MIN_GAIN, 0.0)
                good_years = sum(v is not None and v > threshold for v in ev["by_year"].values())
                checks = {"score_floor": score >= floor, "above_null": score > float(np.percentile(null, PCT)),
                          "years": good_years >= MIN_YEARS}
                entry["models"][model] = {"score": score, "by_year": ev["by_year"], "null_pct": PCT,
                                          "null_threshold": float(np.percentile(null, PCT)),
                                          "null_mean": float(np.mean(null)), "checks": checks,
                                          "skill": all(checks.values()), "n_test": ev["n_test"],
                                          **({"base_rate": ev["base_rate"]} if kind == "binary" else
                                             {"r2": ev["r2"], "r2_naive": ev["r2_naive"]})}
                preds[(asset, key, model)] = (ev["predictions"], y, label_end)
                rows.append({"family": FAMILY, "study": STUDY, "trial": f"{asset} {key} {model}",
                             "config": {"asset": asset, "target": key, "model": model, "score": round(score, 4),
                                        "measure": "auc" if kind == "binary" else "r2_gain"},
                             "window": [int(EVAL_START), int(LIMIT)], "sharpe": None, "count": 1,
                             "note": "development window only; features lab.feature_pool.FEATURES + log_rv_30"})
                print(asset, key, model, json.dumps({k: entry["models"][model][k] for k in ("score", "checks")}),
                      flush=True)
            entry["skill"] = any(m["skill"] for m in entry["models"].values())
            result["targets"][key] = entry
        out["assets"][asset] = result
    out["economic"] = economic(out, preds, rows)
    trials.record(rows, path=ROOT / "research/trials.jsonl")
    OUT.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({a: {k: v["skill"] for k, v in r["targets"].items()} for a, r in out["assets"].items()},
                     indent=1))
    print(json.dumps(out["economic"], indent=1, default=str)[:3000])


def economic(out: dict, preds: dict, rows: list) -> dict:
    """The frozen overlay filtered by the trend probability, on each asset whose trend target showed skill."""
    chosen = {}
    for asset, result in out["assets"].items():
        models = result["targets"]["trend_14"]["models"]
        for model in ("logistic", "lgbm"):
            if models[model]["skill"]:
                chosen[asset] = model
                break
    if not chosen:
        return {"run": False, "reason": "no asset's trend target showed skill"}
    bars, funding, frames, targets = {}, {}, {}, {}
    for s in fp.SYMBOLS:
        bars[s], funding[s], frames[s] = load_symbol(PUBLIC, s)
        targets[s] = compute_targets(frames[s], funding_series(funding[s]), SPEC)
    start = st.year_start(YEARS[1])
    filtered = {s: t.copy() for s, t in targets.items()}
    gates = {}
    for asset, model in chosen.items():
        p, y, label_end = preds[(asset, "trend_14", model)]
        base = {}
        for fold in st.year_folds(y.dropna().index, label_end.loc[y.dropna().index], YEARS):
            base[fold["year"]] = float(y.dropna().iloc[fold["train"]].mean())
        years = pd.to_datetime(p.index.to_numpy(dtype=np.int64), unit="ms").year
        on = pd.Series((p.to_numpy() >= np.array([base[yr] for yr in years])).astype(float), index=p.index)
        days = np.arange(int(p.index.min()), int(LIMIT) + DAY, DAY, dtype=np.int64)
        on = on.reindex(days).ffill()                                   # the last prediction known at each day
        decided = filtered[asset]["available_at"].to_numpy(dtype=np.int64) // DAY * DAY   # p(t) is known at t
        keep = on.reindex(decided).fillna(0.0).to_numpy() > 0           # before the first test year: unfiltered
        keep = np.where(decided < start, True, keep)
        filtered[asset]["target"] = filtered[asset]["target"].to_numpy() * keep
        gates[asset] = {"model": model, "share_on": float(on.mean())}

    def run(t, cost):
        curve, counts, _ = pp.run(bars, t, funding, pp.BASELINE, rules=RULES, spec=SPEC, start=start, end=LIMIT,
                                  capital=150.0, cost=cost)
        daily = daily_equity([curve])
        equity = pd.concat([pd.Series({start // DAY - 1: 150.0}), daily])
        return pp.stats(equity.pct_change().dropna())
    plain, with_filter, with_filter2 = run(targets, 1), run(filtered, 1), run(filtered, 2)
    for label, s in (("filtered cost x1", with_filter), ("filtered cost x2", with_filter2), ("unfiltered", plain)):
        rows.append({"family": "directional_btc_eth", "study": STUDY, "trial": f"overlay {label}",
                     "config": {"filter": gates, "risk": pp.BASELINE.key}, "window": [int(start), int(LIMIT)],
                     "sharpe": round(s["sharpe"], 4), "count": 1, "note": "feature-pool trend filter"})
    checks = {"sharpe_at_least_plain_plus_0.10": with_filter["sharpe"] >= plain["sharpe"] + 0.10,
              "double_costs_return_above_0": with_filter2["total_return"] > 0}
    return {"run": True, "filters": gates, "plain": plain, "filtered": with_filter,
            "filtered_double_costs": with_filter2, "checks": checks, "adopt": all(checks.values())}


if __name__ == "__main__":
    main()
