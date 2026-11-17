"""The overlay's signal settings (README.md, registered before any result): the 72 settings of
lab.pilot_params.SIGNAL_GRID at the pilot's risk settings today, on the development window, each year chosen from
the years before it by the plateau rule, the final choice from all years, and the adoption rule against the frozen
spec. Every replay is registered in research/trials.jsonl (family directional_btc_eth)."""
import json
import sys
from decimal import Decimal as D
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lab import pilot_params as pp  # noqa: E402
from lab import trials  # noqa: E402
from lab.data import DAY, funding_series, load_symbol  # noqa: E402
from lab.metrics import daily_equity, deflated_sharpe_probability_k  # noqa: E402
from lab.overlay_study import SPEC  # noqa: E402
from lab.strategy import compute_targets  # noqa: E402
from lab.study import DEV_END, EVAL_START, SYMBOLS  # noqa: E402
from trading.adapters.simulated.bar_broker import SymbolRules  # noqa: E402

DATA = ROOT / "data_files" / "binance_public"
OUT = Path(__file__).with_name("results.json")
RISK = ROOT / "research/results/pilot_params_2026-11-17/results.json"
RULES = {"BTCUSDT": SymbolRules(D("0.001"), D("0.001"), D("50")),        # Binance exchangeInfo, checked 2026-11-17
         "ETHUSDT": SymbolRules(D("0.001"), D("0.001"), D("20"))}
CAPITAL = 150.0
YEARS = (2021, 2022, 2023, 2024, 2025)
FAMILY, STUDY = "directional_btc_eth", "overlay_signal_2026-11-17"
MARGIN = 0.10                                                            # gate 1: at least 0.10 above the frozen spec


def returns(curve):
    daily = daily_equity([curve])
    daily = daily[daily.index >= EVAL_START // DAY]
    equity = pd.concat([pd.Series({EVAL_START // DAY - 1: CAPITAL}), daily])
    return equity.pct_change().dropna()


def years_of(daily):
    return pd.to_datetime(np.asarray(daily.index, dtype=np.int64) * DAY, unit="ms").year


def main():
    bars, funding, frames, fseries = {}, {}, {}, {}
    for symbol in SYMBOLS:
        bars[symbol], funding[symbol], frames[symbol] = load_symbol(DATA, symbol)
        fseries[symbol] = funding_series(funding[symbol])

    def replay(setting, cost=1):
        spec = setting.spec(SPEC)
        targets = {s: compute_targets(frames[s], fseries[s], spec) for s in SYMBOLS}
        curve, counts, _ = pp.run(bars, targets, funding, pp.BASELINE, rules=RULES, spec=spec, start=EVAL_START,
                                  end=DEV_END, capital=CAPITAL, cost=cost)
        return returns(curve), counts

    out = {"risk_setting": pp.BASELINE.key, "frozen": pp.FROZEN.key, "capital_usdt": CAPITAL,
           "window": "2021-01-10..2025-11-16 (development only)", "settings": {}}
    dailies, rows = {}, []
    window = [EVAL_START, DEV_END]
    for setting in pp.SIGNAL_GRID:
        daily, counts = replay(setting)
        dailies[setting.key] = daily
        whole = pp.stats(daily)
        out["settings"][setting.key] = {"setting": {**setting.__dict__, "lookbacks": list(setting.lookbacks)},
                                        "whole": whole, "counts": counts}
        rows.append({"family": FAMILY, "study": STUDY, "trial": f"{setting.key} cost x1",
                     "config": out["settings"][setting.key]["setting"], "window": window,
                     "sharpe": round(whole["sharpe"], 4), "count": 1, "note": f"risk {pp.BASELINE.key}"})
        print(setting.key, json.dumps(whole), flush=True)

    reference = json.loads(RISK.read_text())["settings"][pp.BASELINE.key]["whole"]
    mine = out["settings"][pp.FROZEN.key]["whole"]
    out["consistency"] = {"same": round(mine["total_return"], 6) == round(reference["total_return"], 6)
                          and round(mine["sharpe"], 6) == round(reference["sharpe"], 6),
                          "frozen": mine, "reference": reference}
    if not out["consistency"]["same"]:
        OUT.write_text(json.dumps(out, indent=1, default=str) + "\n")
        raise SystemExit("consistency check failed")

    wf = pp.walk_forward(dailies, YEARS, pp.SIGNAL_GRID, neighbours_of=pp.signal_neighbours, fallback=pp.FROZEN)
    frozen = dailies[pp.FROZEN.key]
    frozen_wf = frozen[np.isin(years_of(frozen), list(YEARS[1:]))]
    doubled = {}
    for key in set(wf["choices"].values()):
        setting = next(s for s in pp.SIGNAL_GRID if s.key == key)
        doubled[key] = replay(setting, cost=2)[0]
        rows.append({"family": FAMILY, "study": STUDY, "trial": f"{key} cost x2", "config": {"setting": key, "cost": 2},
                     "window": window, "sharpe": round(pp.stats(doubled[key])["sharpe"], 4), "count": 1,
                     "note": f"risk {pp.BASELINE.key}"})
    wf2 = pd.concat([doubled[key][years_of(doubled[key]) == year] for year, key in wf["choices"].items()])
    whole = {k: v["whole"] for k, v in out["settings"].items()}
    final = pp.plateau_choice(whole, pp.SIGNAL_GRID, neighbours_of=pp.signal_neighbours) or pp.FROZEN
    final_daily2 = doubled.get(final.key)
    if final_daily2 is None:
        final_daily2 = replay(final, cost=2)[0]
        rows.append({"family": FAMILY, "study": STUDY, "trial": f"{final.key} cost x2",
                     "config": {"setting": final.key, "cost": 2}, "window": window,
                     "sharpe": round(pp.stats(final_daily2)["sharpe"], 4), "count": 1,
                     "note": f"risk {pp.BASELINE.key}"})
    final2 = pp.stats(final_daily2)
    wf_stats, frozen_stats = pp.stats(wf["daily"]), pp.stats(frozen_wf)
    family = trials.family_stats(FAMILY, path=ROOT / "research/trials.jsonl", extra=rows)
    checks = {"walk_forward_sharpe_at_least_frozen_plus_0.10": wf_stats["sharpe"] >= frozen_stats["sharpe"] + MARGIN,
              "walk_forward_double_costs_return_above_0": pp.stats(wf2)["total_return"] > 0,
              "final_double_costs_return_above_0": final2["total_return"] > 0}
    out["walk_forward"] = {"choices": wf["choices"], "stats": wf_stats, "double_costs": pp.stats(wf2),
                           "frozen_same_years": frozen_stats, "family": family,
                           "deflated_probability": deflated_sharpe_probability_k(wf["daily"], family["sharpe_std"],
                                                                                 family["n_trials"])}
    out["final"] = {"choice": final.key, "whole": whole[final.key], "double_costs": final2,
                    "neighbours": {n.key: whole[n.key] for n in pp.signal_neighbours(final)}}
    out["adoption"] = {"checks": checks, "adopt": all(checks.values())}
    trials.record(rows, path=ROOT / "research/trials.jsonl")
    OUT.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({k: out[k] for k in ("consistency", "walk_forward", "final", "adoption")}, indent=1,
                     default=str))


if __name__ == "__main__":
    main()
