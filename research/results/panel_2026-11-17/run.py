"""The whole-market panel study (README.md, registered before any result): one pooled model over the 100 most traded
crypto USDT perpetuals of each day; direction (daily cross-sectional rank IC), trend coming and volatility; every
replay registered. The money test of the README is written only if the direction target passes."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lab import feature_pool as fp  # noqa: E402
from lab import panel_study as ps  # noqa: E402
from lab import trials  # noqa: E402
from lab import um_judge  # noqa: E402
from lab.study import DEV_END, EVAL_START  # noqa: E402

SNAP = ROOT / "data_files" / "binance_usdt_perps_2020_2026-11-12"
TRADFI = json.loads((ROOT / "research/lab/runs/2026-11-17b/tradfi_contracts.json").read_text())["symbols"]
EXCLUDE = frozenset(TRADFI) | frozenset(um_judge.STABLE)
OUT = Path(__file__).with_name("results.json")
DAY = ps.DAY
LIMIT, TOP = DEV_END, 100
YEARS = (2021, 2022, 2023, 2024, 2025)
NULL_RUNS, PCT = 20, 97.5
TARGETS = {"ret_14_scaled": ("regression", "end_14", "within_day"), "trend_14": ("binary", "end_14", "blocks"),
           "fwd_vol_7": ("regression", "end_7", "blocks"), "fwd_vol_14": ("regression", "end_14", "blocks")}
MODELS = {"binary": ("logistic", "lgbm"), "regression": ("ridge", "lgbm")}
FAMILY, STUDY = "panel_usdt_perps", "panel_2026-11-17"


def build_panel(times: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    symbols = [p.name[:-len("_1h.csv.gz")] for p in sorted(SNAP.glob("*_1h.csv.gz"))]
    symbols = [s for s in symbols if s not in EXCLUDE]
    volumes, first = {}, {}
    for s in symbols:
        h = pd.read_csv(SNAP / f"{s}_1h.csv.gz", usecols=["open_time", "quote_volume"])
        h = h[h["open_time"] + ps.H <= LIMIT]
        if len(h):
            volumes[s], first[s] = ps.daily_volume(h), int(h["open_time"].iloc[0])
    member = ps.liquid_universe(volumes, first, times, top=TOP)
    needed = [s for s in member.columns if member[s].any()]
    rows, labels, btc = [], [], None
    for s in needed:
        h, funding = ps.load_symbol(SNAP, s, LIMIT)
        feats = ps.symbol_features(h, funding, times)
        if s == "BTCUSDT":
            btc = feats
        keep = member[s].to_numpy()
        rows.append(feats[keep].assign(symbol=s))
        labels.append(ps.symbol_targets(fp.daily(h)["close"], times)[keep].assign(symbol=s))
    closes_m, funding_m = fp.load_market(SNAP, LIMIT, exclude=EXCLUDE)
    market = ps.market_state(closes_m, funding_m, btc, times)
    panel = ps.cross_section(pd.concat(rows).sort_index(kind="stable"), market)
    y = pd.concat(labels).sort_index(kind="stable")
    info = {"contracts_seen": len(volumes), "contracts_ever_in_universe": len(needed), "rows": int(len(panel)),
            "mean_names_per_day": float(member.sum(axis=1).mean())}
    return panel, y, info


def main():
    times = np.arange(EVAL_START + DAY, LIMIT + 1, DAY, dtype=np.int64)
    panel, y, info = build_panel(times)
    assert (panel["symbol"].to_numpy() == y["symbol"].to_numpy()).all()
    panel["log_rv_30"] = np.log(panel["rv_30"])
    x = panel[list(ps.PANEL_FEATURES) + ["log_rv_30"]].astype(np.float32)
    t = panel.index.to_numpy(dtype=np.int64)
    sym = panel["symbol"].to_numpy()
    out = {"info": info, "targets": {}}
    rows = []
    rng = np.random.default_rng(11)
    for name, (kind, end, null_how) in TARGETS.items():
        target = y[name].to_numpy(float)
        label_end = y[end].fillna(pd.Series(t + 14 * DAY, index=y.index)).to_numpy(dtype=np.int64)
        entry = {"kind": kind, "null": null_how, "models": {}}
        for model in MODELS[kind]:
            naive = "log_rv_30" if name.startswith("fwd_vol") else None
            ev = ps.evaluate(x, target, t, label_end, YEARS, kind=kind, model=model, naive=naive)
            nulls = []
            for _ in range(NULL_RUNS):
                fake = ps.shuffled(target, t, sym, how=null_how, rng=rng)
                nv = ps.evaluate(x, fake, t, label_end, YEARS, kind=kind, model=model, naive=naive)
                nulls.append(nv["auc"] if kind == "binary" else (nv["ic"]["mean_ic"] if name == "ret_14_scaled"
                                                                 else nv["r2_gain"]))
            threshold = float(np.percentile(nulls, PCT))
            if name == "ret_14_scaled":
                ic = ev["ic"]
                score = ic["mean_ic"]
                checks = {"mean_ic_above_0": score > 0, "t_spaced_at_least_3": ic["t_spaced"] >= 3,
                          "years": sum(v > 0 for k, v in ic["by_year"].items() if k >= YEARS[1]) >= 3,
                          "above_null": score > threshold}
            elif kind == "binary":
                score = ev["auc"]
                checks = {"auc_at_least_0.55": score >= 0.55, "above_null": score > threshold,
                          "years": sum(v > 0.5 for v in ev["by_year"].values()) >= 3}
            else:
                score = ev["r2_gain"]
                checks = {"gain_at_least_0.02": score >= 0.02, "above_null": score > threshold,
                          "years": sum(v > 0 for v in ev["by_year"].values()) >= 3}
            btc_eth = np.isin(sym[np.isfinite(target)], ps.ALWAYS)
            pred = ev["predictions"]
            entry["models"][model] = {"score": score, "checks": checks, "skill": all(checks.values()),
                                      "null_threshold": threshold, "null_mean": float(np.mean(nulls)),
                                      "by_year": ev.get("by_year"), "ic": ev.get("ic"), "n_test": ev["n_test"],
                                      "btc_eth_rows_with_prediction": int(np.isfinite(pred[btc_eth]).sum())}
            rows.append({"family": FAMILY, "study": STUDY, "trial": f"{name} {model}",
                         "config": {"target": name, "model": model, "score": round(float(score), 4),
                                    "universe": f"top {TOP} by 30-day median quote volume, crypto USDT perpetuals"},
                         "window": [int(EVAL_START), int(LIMIT)], "sharpe": None, "count": 1,
                         "note": "development window only; lab.panel_study.PANEL_FEATURES + log_rv_30"})
            print(name, model, json.dumps({"score": score, "checks": checks, "threshold": threshold}), flush=True)
        entry["skill"] = any(m["skill"] for m in entry["models"].values())
        out["targets"][name] = entry
    trials.record(rows, path=ROOT / "research/trials.jsonl")
    OUT.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({k: v["skill"] for k, v in out["targets"].items()}, indent=1))


if __name__ == "__main__":
    main()
