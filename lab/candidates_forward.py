"""Forward replay of the three frozen candidates and their registered conditions.

Protocol: research/protocols/2026-11-14-forward-candidates.md. ``evaluate_forward`` cuts every raw input at
the look's end date, replays the frozen spec over [start, end) with the functions the development run used
(lab/candidates_study.py) and evaluates ``FORWARD_GATE``. Called by lab/forward.py at a registered look.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from lab.candidates import FLUSH_HOLD, HORIZON_DAYS, flush_targets, predict_frozen
from lab.candidates_study import (FORWARD_GATE, PAIR_SMALL_CAPITAL, _above, _at_most, _checks, _ic, b_dataset,
                                  b_targets, flush_segments, run_sleeves)
from lab.data import DAY, HOUR
from lab.pair import SYMBOLS, build, labels, load_inputs
from lab.pair_study import REPORT_CAPITAL, decisions, describe, merged_hourly, public, simulate_pair


def evaluate_pair(frozen: dict, inputs: dict, *, start: int, end: int) -> dict:
    closes, feats = build(inputs)
    prediction = pd.Series(predict_frozen(frozen["model"], feats[list(frozen["features"])].to_numpy(float)),
                           index=feats.index)
    sides = decisions(prediction, HORIZON_DAYS, start=start, end=end)
    bars, funding = merged_hourly(inputs["hourly"]), inputs["funding"]

    def run(side_map, capital, cost_multiplier=1):
        return describe(simulate_pair(bars, funding, side_map, start=start, end=end, capital=capital,
                                      cost_multiplier=cost_multiplier), start, capital)

    base, cost_x2 = run(sides, REPORT_CAPITAL), run(sides, REPORT_CAPITAL, 2)
    first = min(sides) if sides else start
    statics = {"static_long_pair": run({first: 1}, REPORT_CAPITAL), "static_short_pair": run({first: -1},
                                                                                             REPORT_CAPITAL)}
    lab = labels(closes, HORIZON_DAYS)
    scored = (feats.index >= start) & (lab["label_end"].to_numpy(dtype=np.int64) <= end)
    realized = (lab["y"] / (feats["sigma_pair"] * math.sqrt(HORIZON_DAYS))).to_numpy(float)
    ic = _ic(prediction.to_numpy()[scored], realized[scored])
    best_static = max(v["sharpe"] if math.isfinite(v["sharpe"]) else -math.inf for v in statics.values())
    g = FORWARD_GATE
    checks = {
        "cost_x2_total_return": _above(cost_x2["total_return"], g["cost_x2_return_above"]),
        "sharpe_vs_best_static_pair": _above(base["sharpe"], best_static),
        "ic": _above(ic, g["ic_above"]),
        "max_drawdown_hourly": _at_most(base["max_drawdown_hourly"], g["pair_drawdown_hourly_max"]),
        "round_trips_per_year": _at_most(base["round_trips_per_year"], g["pair_round_trips_max"]),
        "no_wipeout": (base["wiped"] or cost_x2["wiped"], not (base["wiped"] or cost_x2["wiped"])),
    }
    return {"strategy": public(base), "strategy_cost_x2": public(cost_x2), "decisions": len(sides),
            "static_pairs": {k: public(v) for k, v in statics.items()}, "best_static_sharpe": best_static,
            "capital_167": public(run(sides, PAIR_SMALL_CAPITAL)), "ic": ic, "checks": _checks(checks),
            "_daily_returns": base["_daily_returns"]}


def evaluate_directional(frozen: dict, inputs: dict, *, start: int, end: int) -> dict:
    data = b_dataset(inputs)
    preds = {s: pd.Series(predict_frozen(frozen["model"], data[s]["x"]), index=data[s]["feats"].index)
             for s in SYMBOLS}
    targets = b_targets(inputs, data, preds, start)
    base, cost_x2 = (run_sleeves(inputs, targets, start=start, end=end, cost_multiplier=m) for m in (1, 2))
    p, y = [], []
    for s in SYMBOLS:
        d = data[s]
        scored = (d["times"] >= start) & (d["label_end"] <= end)
        p.append(preds[s].to_numpy()[scored])
        y.append(d["y"][scored])
    ic = _ic(np.concatenate(p), np.concatenate(y))
    g = FORWARD_GATE
    worst_fills = max(base["fills_per_symbol_year"].values())
    checks = {
        "cost_x2_total_return": _above(cost_x2["total_return"], g["cost_x2_return_above"]),
        "ic": _above(ic, g["ic_above"]),
        "max_drawdown_daily": _at_most(base["max_drawdown"], g["drawdown_daily_max"]),
        "fills_per_symbol_year": _at_most(worst_fills, g["fills_per_symbol_year_max"]),
    }
    return {"strategy": public(base), "strategy_cost_x2": public(cost_x2), "ic": ic, "checks": _checks(checks),
            "_daily_returns": base["_daily_returns"]}


def evaluate_flush(frozen: dict, inputs: dict, *, start: int, end: int) -> dict:
    signals = {s: flush_targets(inputs["hourly"][s], inputs["metrics"][s]) for s in SYMBOLS}
    base, cost_x2 = (run_sleeves(inputs, signals, start=start, end=end, cost_multiplier=m) for m in (1, 2))
    segments = {s.symbol: flush_segments(s, signals[s.symbol], start=start, end=end) for s in base["_sleeves"]}
    returns = [seg["return"] for rows in segments.values() for seg in rows]
    events = len(returns)
    winning = float(np.mean([r > 0 for r in returns])) if returns else float("nan")
    g = FORWARD_GATE
    checks = {
        "events": (events, events >= g["flush_events_min"]),
        "cost_x2_total_return": _above(cost_x2["total_return"], g["cost_x2_return_above"]),
        "winning_share": (winning, math.isfinite(winning) and winning >= g["flush_winning_share_min"]),
        "max_drawdown_daily": _at_most(base["max_drawdown"], g["drawdown_daily_max"]),
    }
    return {"strategy": public(base), "strategy_cost_x2": public(cost_x2), "events": events,
            "winning_share": winning, "segments": segments, "hold_hours": FLUSH_HOLD // HOUR,
            "checks": _checks(checks), "_daily_returns": base["_daily_returns"]}


def verdict(kind: str, result: dict) -> str:
    """The registered forward verdict: continue, stop, or (C only) too few events to judge."""
    if kind == "flush_reversal_rule" and not result["checks"]["events"]["ok"]:
        return "insufficient_sample"
    return "continue" if all(c["ok"] for c in result["checks"].values()) else "stop"


FORWARD = {"eth_btc_pair_frozen": evaluate_pair, "directional_ridge_frozen": evaluate_directional,
           "flush_reversal_rule": evaluate_flush}


def evaluate_forward(frozen: dict, data_dir: Path, premium_dir: Path | None, start: int, end: int) -> dict:
    """Replay a frozen candidate over [start, end); raw inputs are cut at ``end`` before anything is computed."""
    if premium_dir is None:
        raise ValueError("these candidates need a premium-index snapshot (--premium-dir)")
    inputs = load_inputs(data_dir, premium_dir, end)
    for s in SYMBOLS:
        if inputs["bars"][s][-1].open_time + HOUR < end:
            raise ValueError(f"{s}: data ends before the look's end date")
    result = FORWARD[frozen["kind"]](frozen, inputs, start=start, end=end)
    out = {k: v for k, v in result.items() if not k.startswith("_")}
    out.update(window={"start_ms": start, "end_ms": end, "days": int((end - start) // DAY)},
               verdict=verdict(frozen["kind"], result))
    return out


__all__ = ["FORWARD", "evaluate_forward", "verdict", "evaluate_pair", "evaluate_directional", "evaluate_flush"]
