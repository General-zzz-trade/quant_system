"""Three forward-test candidates: one development run that freezes them, and their forward replay.

    python -m lab.candidates_study --data-dir data_files/binance_public \
        --premium-dir data_files/binance_premium_2025-11-16 --out research/results/forward_candidates_2026-11-14

Fixed before any real-data run (research/protocols/2026-11-14-forward-candidates.md):

- **A pair_ridge_pd_7d:** the pair study's selected candidate, fitted once on every development sample
  whose label ended by 2025-11-16. No development performance is computed; it is registered as it is.
- **B dir_ridge_7d:** pooled Ridge on each symbol's own 15 features, 7-day horizon. Purged walk-forward
  from 2023-03-01 (91-day blocks) for the report, then one final fit. Weekly sign x volatility-targeted
  weight per symbol, shorts allowed.
- **C flush_reversal_72h:** long for 72 hours after a price and open-interest flush. No fitted parameter.
  Development run from 2022-07-01.
- **Registration floor (B and C):** base-cost portfolio return > 0, no wipeout, at most 150 fills per
  symbol-year, and for C at least 10 events. A candidate below the floor is closed, not re-run.
- **Forward:** lab/candidates_forward.py replays a frozen spec over [FORWARD_START, end) with the same
  functions and evaluates ``FORWARD_GATE``.

The study runs once: it refuses a non-empty output directory and a study id already in the registry.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lab import trials
from lab.candidates import (B_FEATURES, HORIZON_DAYS, daily_close, directional_labels, fit_ridge,
                            flush_targets, predict_frozen, symbol_features, weekly_targets)
from lab.data import DAY, HOUR
from lab.overlay_study import SLEEVE_BALANCE, portfolio_equity, stats
from lab.pair import D_FEATURES, P_FEATURES, SYMBOLS, build, labels, load_inputs
from lab.pair_study import (BLOCK, MIN_TRAIN_ROWS, REPORT_CAPITAL, _code_identity, public)
from lab.simulate import simulate
from lab.study import BASE, DEV_END, _manifest
from lab.validation import walk_forward

STUDY = "forward_candidates_2026-11-14"
FAMILY = "directional_btc_eth"
B_START = int(datetime(2023, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)
C_START = int(datetime(2022, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)
PAIR_FEATURES = tuple(P_FEATURES) + tuple(D_FEATURES)
PAIR_SMALL_CAPITAL = 167.0
FLOOR = {"base_total_return_min_exclusive": 0.0, "fills_per_symbol_year_max": 150, "flush_events_min": 10}
FORWARD_GATE = {"cost_x2_return_above": 0.0, "ic_above": 0.0, "pair_drawdown_hourly_max": 0.60,
                "pair_round_trips_max": 150, "drawdown_daily_max": 0.25, "fills_per_symbol_year_max": 150,
                "flush_events_min": 10, "flush_winning_share_min": 0.5}
REFUSED = "refused: requires an accepted forward test, recorded by a person"


# -- shared: per-symbol sleeves ------------------------------------------------------------------------
def run_sleeves(inputs: dict, targets: dict, *, start: int, end: int, cost_multiplier: int = 1) -> dict:
    """Both sleeves through the lab simulator; portfolio statistics plus the sleeves themselves."""
    sleeves = [simulate(s, [b for b in inputs["bars"][s] if b.open_time < end], targets[s], inputs["funding"][s],
                        BASE, start_ms=start, end_ms=end, starting_balance=SLEEVE_BALANCE,
                        cost_multiplier=cost_multiplier) for s in SYMBOLS]
    initial = float(len(SYMBOLS)) * float(SLEEVE_BALANCE)
    equity = portfolio_equity(sleeves, start // DAY, initial)
    out = stats(equity)
    out["fills_per_symbol_year"] = {s.symbol: len(s.fills) / max(out["days"], 1) * 365.0 for s in sleeves}
    out["wiped"] = bool(any((s.curve["equity"] <= 0).any() for s in sleeves))
    out["_daily_returns"], out["_sleeves"] = equity.pct_change().dropna(), sleeves
    return out


def _above(value: float, reference: float) -> tuple[float, bool]:
    return value, bool(math.isfinite(value) and value > reference)


def _at_most(value: float, reference: float) -> tuple[float, bool]:
    return value, bool(math.isfinite(value) and value <= reference)


def _ic(prediction: np.ndarray, realized: np.ndarray) -> float:
    ok = np.isfinite(prediction) & np.isfinite(realized)
    return float(spearmanr(prediction[ok], realized[ok]).statistic) if ok.sum() >= 10 else float("nan")


# -- A: frozen pair model ------------------------------------------------------------------------------
def fit_pair(inputs: dict, limit: int) -> dict:
    """One Ridge fit of the pair candidate on every sample whose 7-day label ended by ``limit``."""
    closes, feats = build(inputs)
    lab = labels(closes, HORIZON_DAYS)
    x = feats[list(PAIR_FEATURES)].to_numpy(float)
    y = (lab["y"] / (feats["sigma_pair"] * math.sqrt(HORIZON_DAYS))).to_numpy(float)
    rows = np.isfinite(x).all(axis=1) & np.isfinite(y) & (lab["label_end"].to_numpy(dtype=np.int64) <= limit)
    return fit_ridge(x[rows], y[rows])


# -- B: weekly directional Ridge ------------------------------------------------------------------------
def b_dataset(inputs: dict) -> dict:
    data = {}
    for s in SYMBOLS:
        feats = symbol_features(inputs, s)
        lab = directional_labels(daily_close(inputs["hourly"][s]))
        y = (lab["y"] / (feats["sigma"] * math.sqrt(HORIZON_DAYS))).to_numpy(float)
        x = feats[list(B_FEATURES)].to_numpy(float)
        data[s] = {"feats": feats, "x": x, "y": y, "has_x": np.isfinite(x).all(axis=1),
                   "times": feats.index.to_numpy(dtype=np.int64),
                   "label_end": lab["label_end"].to_numpy(dtype=np.int64)}
    return data


def _rows(data: dict, mask_of) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for s in SYMBOLS:
        d = data[s]
        keep = mask_of(d)
        xs.append(d["x"][keep])
        ys.append(d["y"][keep])
    return np.concatenate(xs), np.concatenate(ys)


def b_walk_forward(data: dict, *, first_test: int, limit: int) -> tuple[dict, list[dict]]:
    """Out-of-sample predictions per symbol and per-block IC; each block trains on labels ended before it."""
    dates = np.unique(np.concatenate([data[s]["times"] for s in SYMBOLS]))
    preds = {s: pd.Series(np.nan, index=data[s]["feats"].index) for s in SYMBOLS}
    blocks = []
    for fold in walk_forward(dates, dates + HORIZON_DAYS * DAY, first_test=first_test, block=BLOCK, limit=limit):
        a, b = fold.test_start, fold.test_end
        x_train, y_train = _rows(data, lambda d: d["has_x"] & np.isfinite(d["y"]) & (d["label_end"] < a))
        if len(x_train) < MIN_TRAIN_ROWS:
            continue
        model = fit_ridge(x_train, y_train)
        p_all, y_all = [], []
        for s in SYMBOLS:
            d = data[s]
            target = (d["times"] >= a) & (d["times"] < b) & d["has_x"]
            p = predict_frozen(model, d["x"][target])
            preds[s].iloc[np.flatnonzero(target)] = p
            scored = d["label_end"][target] <= limit
            p_all.append(p[scored])
            y_all.append(d["y"][target][scored])
        blocks.append({"test_start": int(a), "train_rows": int(len(x_train)),
                       "predicted": int(sum(len(p) for p in p_all)),
                       "ic": _ic(np.concatenate(p_all), np.concatenate(y_all))})
    return preds, blocks


def fit_directional(data: dict, limit: int) -> dict:
    """One pooled Ridge fit on every sample whose 7-day label ended by ``limit``."""
    return fit_ridge(*_rows(data, lambda d: d["has_x"] & np.isfinite(d["y"]) & (d["label_end"] <= limit)))


def b_targets(inputs: dict, data: dict, preds: dict, start: int) -> dict:
    return {s: weekly_targets(inputs["hourly"][s], preds[s], data[s]["feats"]["sigma"], start=start)
            for s in SYMBOLS}


# -- C: flush reversal ----------------------------------------------------------------------------------
def flush_segments(sleeve, signal: pd.DataFrame, *, start: int, end: int) -> list[dict]:
    """Each holding period that began in [start, end): entry time and the sleeve's return over it."""
    curve = sleeve.curve.set_index("at")["equity"]
    times = signal["available_at"].to_numpy(dtype=np.int64)
    target = signal["target"].to_numpy(float)
    segments = []
    for i in np.flatnonzero(signal["entry"].to_numpy(dtype=bool) & (times >= start) & (times < end)):
        flat = np.flatnonzero(target[i:] == 0)
        closed_at = int(times[i + flat[0]]) + HOUR if len(flat) else int(curve.index[-1])
        closed_at = min(closed_at, int(curve.index[-1]))
        entry_at = int(times[i])
        if entry_at in curve.index and closed_at in curve.index and curve.loc[entry_at] > 0:
            segments.append({"entry": entry_at, "closed": closed_at,
                             "return": float(curve.loc[closed_at] / curve.loc[entry_at] - 1.0)})
    return segments


# -- checks ----------------------------------------------------------------------------------------------
def _checks(checks: dict) -> dict:
    return {name: {"value": value, "ok": bool(ok)} for name, (value, ok) in checks.items()}


def floor(result: dict, *, events: int | None = None) -> dict:
    """The development-run registration floor for B and C."""
    worst_fills = max(result["base"]["fills_per_symbol_year"].values())
    wiped = result["base"]["wiped"] or result["cost_x2"]["wiped"]
    checks = {
        "base_total_return": _above(result["base"]["total_return"], FLOOR["base_total_return_min_exclusive"]),
        "no_wipeout": (wiped, not wiped),
        "fills_per_symbol_year": _at_most(worst_fills, FLOOR["fills_per_symbol_year_max"]),
    }
    if events is not None:
        checks["events"] = (events, events >= FLOOR["flush_events_min"])
    return _checks(checks)


# -- the development run -----------------------------------------------------------------------------------
def run_study(data_dir: Path, premium_dir: Path, out: Path, *, b_start: int = B_START, c_start: int = C_START,
              limit: int = DEV_END, registry: Path = trials.REGISTRY, study: str = STUDY) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("use an empty output directory; preserve previous studies")
    if any(r["study"] == study for r in trials.load(registry)):
        raise ValueError(f"{study} already recorded its trials; a study runs once")
    if b_start % DAY or c_start % DAY or limit % DAY or not max(b_start, c_start) < limit:
        raise ValueError("windows must be whole UTC days")
    inputs = load_inputs(data_dir, premium_dir, limit)
    common = {"schema": 1, "study": study, "symbols": list(SYMBOLS), "passed": False, "mainnet": REFUSED,
              "fit_limit_ms": limit, "code": _code_identity(),
              "data": {"market": _manifest(data_dir), "premium": _manifest(premium_dir)},
              "forward_gate": FORWARD_GATE}

    pair_model = fit_pair(inputs, limit)
    spec_a = {**common, "kind": "eth_btc_pair_frozen", "candidate": "pair_ridge_pd_7d",
              "features": list(PAIR_FEATURES), "horizon_days": HORIZON_DAYS, "model": pair_model,
              "capital": REPORT_CAPITAL, "registered": True,
              "source": "research/results/pair_2026-11-14 (ridge_PD_7d); no development performance recomputed"}

    data = b_dataset(inputs)
    preds, blocks = b_walk_forward(data, first_test=b_start, limit=limit)
    targets = b_targets(inputs, data, preds, b_start)
    b = {"base": run_sleeves(inputs, targets, start=b_start, end=limit),
         "cost_x2": run_sleeves(inputs, targets, start=b_start, end=limit, cost_multiplier=2)}
    b_floor = floor(b)
    final = fit_directional(data, limit)
    spec_b = {**common, "kind": "directional_ridge_frozen", "candidate": "dir_ridge_7d",
              "features": list(B_FEATURES), "horizon_days": HORIZON_DAYS, "model": final,
              "sizing": {"target_vol": 0.30, "max_weight": 0.9, "vol_lookback_days": 30, "band": BASE.band},
              "registered": all(c["ok"] for c in b_floor.values())}

    signals = {s: flush_targets(inputs["hourly"][s], inputs["metrics"][s]) for s in SYMBOLS}
    c = {"base": run_sleeves(inputs, signals, start=c_start, end=limit),
         "cost_x2": run_sleeves(inputs, signals, start=c_start, end=limit, cost_multiplier=2)}
    segments = {s.symbol: flush_segments(s, signals[s.symbol], start=c_start, end=limit)
                for s in c["base"]["_sleeves"]}
    returns = [seg["return"] for rows in segments.values() for seg in rows]
    c_floor = floor(c, events=len(returns))
    spec_c = {**common, "kind": "flush_reversal_rule", "candidate": "flush_reversal_72h",
              "rule": {"quantile": 0.05, "window_hours": 8760, "min_history_hours": 4320, "hold_hours": 72,
                       "target_vol": 0.30, "max_weight": 0.9, "vol_lookback_days": 30, "band": BASE.band},
              "registered": all(x["ok"] for x in c_floor.values())}

    def trial(name, result, window, config):
        sharpe = result["base"]["sharpe"]
        return {"family": FAMILY, "study": study, "trial": name, "config": config, "window": window,
                "sharpe": float(sharpe) if math.isfinite(sharpe) else None, "count": 1,
                "note": "daily Sharpe, 2 x 5,000 USDT sleeves, base costs, development run of a frozen candidate"}

    rows = [trial("dir_ridge_7d", b, [b_start, limit], {"kind": spec_b["kind"], "features": list(B_FEATURES),
                                                       "horizon_days": HORIZON_DAYS, "sizing": spec_b["sizing"]}),
            trial("flush_reversal_72h", c, [c_start, limit], {"kind": spec_c["kind"], "rule": spec_c["rule"]})]
    summary = {
        "pair_ridge_pd_7d": {"registered": True, "fit_rows": pair_model["rows"], "note": spec_a["source"]},
        "dir_ridge_7d": {"registered": spec_b["registered"], "floor": b_floor, "window": [b_start, limit],
                         "base": public(b["base"]), "cost_x2": public(b["cost_x2"]), "blocks": blocks,
                         "mean_ic": float(np.nanmean([k["ic"] for k in blocks])) if blocks else None,
                         "fit_rows": final["rows"]},
        "flush_reversal_72h": {"registered": spec_c["registered"], "floor": c_floor, "window": [c_start, limit],
                               "base": public(c["base"]), "cost_x2": public(c["cost_x2"]), "events": len(returns),
                               "winning_share": float(np.mean([r > 0 for r in returns])) if returns else None,
                               "mean_segment_return": float(np.mean(returns)) if returns else None,
                               "segments": segments},
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in (("pair_ridge_pd_7d", spec_a), ("dir_ridge_7d", spec_b), ("flush_reversal_72h", spec_c)):
        (out / f"{name}.json").write_text(json.dumps(spec, indent=2, default=float) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    trials.record(rows, path=registry)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data_files/binance_public"))
    parser.add_argument("--premium-dir", type=Path, default=Path("data_files/binance_premium_2025-11-16"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_study(args.data_dir, args.premium_dir, args.out)
    print(json.dumps({name: {k: v for k, v in entry.items() if k in ("registered", "floor", "events", "mean_ic")}
                      for name, entry in summary.items()}, indent=2, default=float))


if __name__ == "__main__":
    main()


__all__ = ["STUDY", "FLOOR", "FORWARD_GATE", "run_study", "floor", "fit_pair", "fit_directional", "run_sleeves",
           "b_dataset", "b_walk_forward", "b_targets", "flush_segments"]
