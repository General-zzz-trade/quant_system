"""The 2026-11-17d comparison (plan.md, registered before any result): maker against market entries for five fixed
BTC/ETH strategies at lab.LIVE_BTC_ETH, on development data only (periods 0-5; the holdout period is never replayed).

    python research/lab/runs/2026-11-17d/compare.py run      # replays, registry rows -> compare.json

Every replay is registered in research/lab/trials.jsonl and cached under /root/lab_cache/2026-11-17d/<execution>.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from lab import um_judge as judge  # noqa: E402
from lab import um_lab as lab  # noqa: E402
from trading.domain.strategy.base import OverlayStrategy  # noqa: E402
from trading.domain.strategy.spec import PARAMS  # noqa: E402

SNAP = ROOT / "data_files" / "binance_usdt_perps_2020_2026-11-12"
START, END = lab.ms("2020-01-01"), lab.ms("2026-11-13")
FOLDS = (*((lab.ms(f"{y}-05-13"), lab.ms(f"{y + 1}-05-13")) for y in range(2021, 2025)),
         (lab.ms("2025-05-13"), lab.ms("2025-11-16")))
PERIODS = judge.periods(START, FOLDS, END)
CACHE = Path("/root/lab_cache/2026-11-17d")
NOTE = ("2026-11-17d maker vs market on the 2020+ snapshot, BTCUSDT and ETHUSDT at LIVE_BTC_ETH "
        "(runs/2026-11-17d/plan.md)")
TERMS = lab.LIVE_BTC_ETH
OVERLAY = json.loads((ROOT / "research/results/overlay_2026-11-13/final_spec.json").read_text())["spec"]
# 2026-11-17c screen.json final_choice, fixed before this study
STRATEGIES = (("F1_btc_trend_beta", {"n": 40}, judge.StopVariant("0.08", 4, "pause_then_reset")),
              ("D2_taker_imbalance", {"x": 0.01}, judge.StopVariant("0.12", 4, "pause")),
              ("E3_vol_target_trend", {"target": 0.02}, judge.StopVariant("0.08", 4, "pause_then_reset")),
              ("B3_dip_in_uptrend", {"z": 2.0}, judge.StopVariant("0.15", 4, "pause")),
              ("overlay", OVERLAY, judge.StopVariant("0.15", 4, "pause_then_reset")))
EXECUTIONS = (("market", 1), ("maker", 1), ("maker", 3))
COSTS = (1, 2)
RUN = tuple(range(0, len(PERIODS) - 1))                  # 0..5; the holdout (6) is never replayed
MAIN = tuple(range(1, len(PERIODS) - 1))                 # 1..5: 2021-05-13 to 2025-11-16
AT_LEAST, MEDIAN = 4, 0.05


def ideas() -> dict:
    found = lab.lab_ideas()
    found["overlay"] = (lambda params: OverlayStrategy(PARAMS["overlay"](params)), [OVERLAY])
    return found


def make(execution: str, ttl: int) -> judge.Judge:
    lists = [list(TERMS.symbols)] * len(PERIODS)
    return judge.Judge(ideas=ideas(), lists=lists, cache=CACHE / f"{execution}_ttl{ttl}", run_date="2026-11-17d",
                       snapshot=SNAP, periods=PERIODS, note=NOTE, terms=TERMS, slots=len(TERMS.symbols),
                       execution=execution, ttl_bars=ttl)


def commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def measured(j: judge.Judge, config: judge.Config) -> dict:
    periods = j.load(config)
    s = lab.summary(judge.joined(periods, MAIN))
    return {"main": {k: s.get(k) for k in ("total_return", "sharpe", "max_drawdown", "days")},
            "period_0": {k: lab.summary(periods[0]["daily"]).get(k) for k in ("total_return", "sharpe")},
            "round_trips": sum(len(periods[k]["trips"]) for k in MAIN),
            "entry_orders": sum(len(periods[k]["entries"]) for k in MAIN),
            "fills": sum(periods[k]["fills"] for k in MAIN), "exchange_stops": sum(periods[k]["stops"] for k in MAIN)}


def verdict(table: dict) -> dict:
    """plan.md's rule, at double costs: each maker lifetime beats market on at least 4 of 5 Sharpes, with a median
    difference of at least +0.05; both lifetimes must."""
    out = {}
    for execution, ttl in EXECUTIONS[1:]:
        diffs = [table[name][f"{execution}_ttl{ttl}"]["x2"]["main"]["sharpe"]
                 - table[name]["market_ttl1"]["x2"]["main"]["sharpe"] for name, _, _ in STRATEGIES]
        out[f"{execution}_ttl{ttl}"] = {"sharpe_differences": diffs, "better": sum(d > 0 for d in diffs),
                                        "median": statistics.median(diffs),
                                        "passes": sum(d > 0 for d in diffs) >= AT_LEAST
                                        and statistics.median(diffs) >= MEDIAN}
    out["maker_better"] = all(v["passes"] for k, v in out.items())
    return out


def main(step: str) -> None:
    if step != "run":
        raise SystemExit(__doc__)
    table: dict = {name: {} for name, _, _ in STRATEGIES}
    for execution, ttl in EXECUTIONS:
        j = make(execution, ttl)
        configs = [judge.Config(name, params, stop, cost) for name, params, stop in STRATEGIES for cost in COSTS]
        j.run_all([(config, RUN) for config in configs], processes=2)
        for config in configs:
            table[config.idea].setdefault(f"{execution}_ttl{ttl}", {})[f"x{config.cost}"] = measured(j, config)
    out = {"code_commit": commit(), "periods": [[judge._date(a), judge._date(b)] for a, b in PERIODS[:-1]],
           "main_periods": list(MAIN), "table": table, "verdict": verdict(table)}
    (HERE / "compare.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps(out["verdict"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
