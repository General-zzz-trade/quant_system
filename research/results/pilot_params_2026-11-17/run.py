"""The pilot's risk and execution settings (README.md, registered before any result): the frozen overlay at 150 USDT
over BTC and ETH, the 60 settings of lab.pilot_params.GRID on the development window, each year chosen from the years
before it by the plateau rule, the final choice from all years, and the adoption rule against today's settings."""
import json
import sys
from dataclasses import asdict
from decimal import Decimal as D
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lab import pilot_params as pp  # noqa: E402
from lab.data import DAY, funding_series, load_symbol  # noqa: E402
from lab.metrics import daily_equity, deflated_sharpe_probability_k  # noqa: E402
from lab.overlay_study import SPEC  # noqa: E402
from lab.strategy import compute_targets  # noqa: E402
from lab.study import DEV_END, EVAL_START, SYMBOLS  # noqa: E402
from trading.adapters.simulated.bar_broker import SymbolRules  # noqa: E402

DATA = ROOT / "data_files" / "binance_public"
OUT = Path(__file__).with_name("results.json")
LEVERAGE_STUDY = ROOT / "research/results/overlay_pilot_leverage_2026-11-17/results.json"
RULES = {"BTCUSDT": SymbolRules(D("0.001"), D("0.001"), D("50")),        # Binance exchangeInfo, checked 2026-11-17
         "ETHUSDT": SymbolRules(D("0.001"), D("0.001"), D("20"))}
CAPITAL = 150.0
YEARS = (2021, 2022, 2023, 2024, 2025)


def returns(curve):
    """Daily returns from the window's start, the first day included (as the leverage studies count them)."""
    daily = daily_equity([curve])
    daily = daily[daily.index >= EVAL_START // DAY]
    equity = pd.concat([pd.Series({EVAL_START // DAY - 1: CAPITAL}), daily])
    return equity.pct_change().dropna()


def by_year(daily):
    years = pd.to_datetime(np.asarray(daily.index, dtype=np.int64) * DAY, unit="ms").year
    return {int(y): pp.stats(daily[years == y]) for y in YEARS}


def main():
    bars, funding, targets = {}, {}, {}
    for symbol in SYMBOLS:
        bars[symbol], funding[symbol], frame = load_symbol(DATA, symbol)
        targets[symbol] = compute_targets(frame, funding_series(funding[symbol]), SPEC)

    def replay(setting, cost=1):
        curve, counts, _ = pp.run(bars, targets, funding, setting, rules=RULES, spec=SPEC, start=EVAL_START,
                                  end=DEV_END, capital=CAPITAL, cost=cost)
        return returns(curve), counts

    out = {"spec": SPEC.record(), "capital_usdt": CAPITAL, "window": "2021-01-10..2025-11-16 (development only)",
           "baseline": pp.BASELINE.key, "settings": {}}
    dailies = {}
    for setting in pp.GRID:
        daily, counts = replay(setting)
        dailies[setting.key] = daily
        out["settings"][setting.key] = {"setting": {**asdict(setting), "margin": setting.margin},
                                        "whole": pp.stats(daily), "years": by_year(daily), "counts": counts}
        print(setting.key, json.dumps(out["settings"][setting.key]["whole"]), flush=True)

    # the consistency check against the leverage study (same simulation)
    reference = json.loads(LEVERAGE_STUDY.read_text())["runs"]["2x 10x/0.045 3/12 x1"]["stats"]
    mine = out["settings"][pp.BASELINE.key]["whole"]
    out["consistency"] = {"reference_total_return": reference["total_return"], "reference_sharpe": reference["sharpe"],
                          "total_return": round(mine["total_return"], 4), "sharpe": round(mine["sharpe"], 3)}
    out["consistency"]["same"] = (out["consistency"]["total_return"] == reference["total_return"]
                                  and out["consistency"]["sharpe"] == reference["sharpe"])
    if not out["consistency"]["same"]:
        OUT.write_text(json.dumps(out, indent=1, default=str) + "\n")
        raise SystemExit(f"consistency check failed: {out['consistency']}")

    wf = pp.walk_forward(dailies, YEARS)
    base = dailies[pp.BASELINE.key]
    base_wf = base[np.isin(pd.to_datetime(np.asarray(base.index, dtype=np.int64) * DAY, unit="ms").year,
                           list(YEARS[1:]))]
    doubled = {}
    for key in set(wf["choices"].values()):
        setting = next(s for s in pp.GRID if s.key == key)
        doubled[key] = replay(setting, cost=2)[0]
    wf2 = pd.concat([doubled[key][pd.to_datetime(np.asarray(doubled[key].index, dtype=np.int64) * DAY,
                                                 unit="ms").year == year] for year, key in wf["choices"].items()])
    whole = {k: v["whole"] for k, v in out["settings"].items()}
    final = pp.plateau_choice(whole) or pp.BASELINE
    final2 = pp.stats(replay(final, cost=2)[0])
    sharpes = [v["sharpe"] for v in whole.values()]
    checks = {"walk_forward_sharpe_at_least_baseline": pp.stats(wf["daily"])["sharpe"] >= pp.stats(base_wf)["sharpe"],
              "walk_forward_double_costs_return_above_0": pp.stats(wf2)["total_return"] > 0,
              "final_double_costs_return_above_0": final2["total_return"] > 0}
    out["walk_forward"] = {"choices": wf["choices"], "stats": pp.stats(wf["daily"]), "double_costs": pp.stats(wf2),
                           "baseline_same_years": pp.stats(base_wf),
                           "deflated_probability": deflated_sharpe_probability_k(wf["daily"], float(np.std(sharpes,
                                                                                                          ddof=1)), 60)}
    out["final"] = {"choice": final.key, "whole": whole[final.key], "double_costs": final2,
                    "neighbours": {n.key: whole[n.key] for n in pp.neighbours(final)}}
    out["adoption"] = {"checks": checks, "adopt": all(checks.values()),
                       "settings_to_track": final.key if all(checks.values()) else pp.BASELINE.key}
    OUT.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({k: out[k] for k in ("consistency", "walk_forward", "final", "adoption")}, indent=1,
                     default=str))


if __name__ == "__main__":
    main()
