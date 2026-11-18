"""The weak-signal portfolio (research/protocols/2026-11-18-weak-signal-portfolio.md, committed before any result).

    python research/results/weak_portfolio_2026-11-18/run.py history   # the folds, from the 17c cache (report only)
    python research/results/weak_portfolio_2026-11-18/run.py holdout   # opens 17c's holdout once; freezes; registers

The history needs no new replay; the holdout step replays only the members' holdout period (registered in
research/lab/trials.jsonl) and, when both totals are above 0, writes forward_spec.json and registers its 24 looks.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
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
from lab import gates  # noqa: E402
from lab import um_judge as judge  # noqa: E402
from lab import um_lab as lab  # noqa: E402
from lab import um_portfolio as portfolio  # noqa: E402

RUN = ROOT / "research" / "lab" / "runs" / "2026-11-17c"
K, EFFECT_SHARPE, LOOKS = 5, 1.5, 24
STUDY = "weak_portfolio_2026-11-18:frozen"
DAY = lab.DAY


def run17c():
    spec = importlib.util.spec_from_file_location("judge_2026_11_17c", RUN / "judge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def setup():
    m = run17c()
    j = m.make()
    names = list(json.loads((RUN / "screen.json").read_text())["ideas"])
    runs = {n: [(c, j.load(c)) for c in j.candidates(n, m.VARIANTS)] for n in names}
    hold = {k: lab.buy_and_hold(judge.BTC, a, b, m.SNAP) for k, (a, b) in enumerate(j.periods)
            if k != j.holdout}
    vol = portfolio.vol_target_hold(m.SNAP, j.periods[:-1], capital=m.TERMS.capital)
    return m, j, runs, hold, vol


def bench(hold: dict, vol: dict, ks) -> pd.DataFrame:
    return pd.DataFrame({"btc_hold": judge.joined({k: {"daily": hold[k]} for k in ks}, ks),
                         "btc_vol_target": judge.joined({k: {"daily": vol[k]} for k in ks}, ks)}).fillna(0.0)


def mean_correlation(dailies: list[pd.Series]) -> float | None:
    if len(dailies) < 2:
        return None
    panel = pd.concat(dailies, axis=1).dropna()
    c = panel.corr().to_numpy()
    return float(np.mean([c[a, b] for a, b in itertools.combinations(range(len(dailies)), 2)]))


def history() -> dict:
    m, j, runs, hold, vol = setup()
    folds = list(j.folds)
    pf = portfolio.walk_forward_portfolio(runs, j.periods, folds, [hold, vol], eligible=lambda n: True, k=K)
    daily = pf["daily"]
    s = lab.summary(daily)
    b = bench(hold, vol, folds)
    sd, n = float(daily.std(ddof=1)), len(daily)
    by_key = {name: {c.key: r for c, r in configs} for name, configs in runs.items()}
    chosen = {name: judge.walk_forward(configs, folds)["chosen"] for name, configs in runs.items()}
    correlation = {}
    for i, fold in enumerate(folds):
        members = pf["members"][i]
        dailies = [by_key[n][chosen[n][i].key][fold]["daily"] for n in members]
        correlation[str(fold)] = mean_correlation(dailies)
    return {"code_commit": commit(), "members_by_fold": pf["members"], "walk_forward": s,
            "regression_alpha": gates.regression_alpha(daily, b),
            "portfolio_gain": gates.portfolio_gain(daily, b["btc_hold"]),
            "benchmarks": {k: lab.summary(v) for k, v in b.items()}, "mean_member_correlation": correlation,
            "daily_sd": sd, "days": n,
            "power": {"at_sharpe_1.0": gates.power(1.0 / math.sqrt(365) * sd, sd, n),
                      "at_sharpe_1.5": gates.power(1.5 / math.sqrt(365) * sd, sd, n)}}


def holdout(hist: dict) -> dict:
    m, j, runs, hold, vol = setup()
    before = list(range(j.holdout))
    bars = [lab.summary(judge.joined({k: {"daily": x[k]} for k in before}, before)).get("sharpe", float("inf"))
            for x in (hold, vol)]
    ranked = []
    for name, configs in runs.items():
        final = judge.walk_forward(configs, list(j.folds))["final"]
        results = next(r for c, r in configs if c.key == final.key)
        prior = lab.summary(judge.joined(results, before))
        if prior.get("days") and prior["total_return"] > 0 and all(prior["sharpe"] > bar for bar in bars):
            ranked.append((-prior["sharpe"], name, final))
    members = sorted(ranked, key=lambda r: (r[0], r[1]))[:K]
    out = {"code_commit": commit(), "opened_holdout": True,
           "members": [{"idea": name, "params": cfg.params, "stop": cfg.label["stop"], "prior_sharpe": -neg}
                       for neg, name, cfg in members]}
    if not members:
        return {**out, "opened_holdout": False, "frozen": False, "reason": "no idea qualifies before the holdout"}
    j.run_all([(cfg, (j.holdout,)) for _, _, cfg in members], processes=2, holdout=True)
    a, b = j.periods[j.holdout]
    days = pd.RangeIndex(a // DAY, b // DAY)
    daily = portfolio.combine([j.load(cfg)[j.holdout]["daily"] for _, _, cfg in members]).reindex(days).fillna(0.0)
    hold_h = lab.buy_and_hold(judge.BTC, a, b, m.SNAP).reindex(days).fillna(0.0)
    s = lab.summary(daily)
    out.update({"holdout": s, "holdout_btc_hold": lab.summary(hold_h),
                "holdout_regression_alpha": gates.regression_alpha(daily, pd.DataFrame({"btc_hold": hold_h}))})
    freeze = hist["walk_forward"]["total_return"] > 0 and s["total_return"] > 0
    out["frozen"] = bool(freeze)
    if freeze:
        spec = {"study": STUDY, "registered": "2026-11-18", "passed": False,
                "protocol": "research/protocols/2026-11-18-weak-signal-portfolio.md",
                "source_run": str(RUN.relative_to(ROOT)),
                "terms": {"capital": m.TERMS.capital, "leverage": m.TERMS.leverage, "symbols": list(m.TERMS.symbols)},
                "members": out["members"], "weights": "equal capital at the forward start, not rebalanced",
                "forward_start": "2026-11-13",
                "sequential": {"effect_sharpe": EFFECT_SHARPE, "daily_sd": hist["daily_sd"], "alpha": 0.05, "beta": 0.2,
                               "max_looks": LOOKS}}
        path = HERE / "forward_spec.json"
        path.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ends = [f"{2026 + (11 + i) // 12}-{(11 + i) % 12 + 1:02d}-01" for i in range(LOOKS)]
        with (ROOT / "research" / "forward_looks.jsonl").open("a") as handle:
            for i, end in enumerate(ends, start=1):
                handle.write(json.dumps({"end": end, "purpose": f"monthly look {i} of {LOOKS} (sequential test); "
                                         "criterion in research/protocols/2026-11-18-weak-signal-portfolio.md",
                                         "registered": "2026-11-18", "spec": str(path.relative_to(ROOT)),
                                         "spec_sha256": digest, "study": STUDY}, sort_keys=True) + "\n")
        out.update({"forward_spec_sha256": digest, "looks": ends})
    return out


def main(step: str) -> None:
    if step == "history":
        (HERE / "history.json").write_text(json.dumps(history(), indent=1, default=str))
    elif step == "holdout":
        hist = json.loads((HERE / "history.json").read_text())
        (HERE / "holdout.json").write_text(json.dumps(holdout(hist), indent=1, default=str))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
