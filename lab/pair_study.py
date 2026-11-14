"""ETH vs BTC relative value: ten predeclared candidates, one gate, development data only.

    python -m lab.pair_study --data-dir data_files/binance_public \
        --premium-dir data_files/binance_premium_2025-11-16 --out research/results/pair_2026-11-14

Fixed before any real-data run (research/protocols/2026-11-14-eth-btc-relative.md):

- **Candidates (10, all counted as trials):** {Ridge, LightGBM} x {P, P+D
  features} x {3, 7-day horizon}, plus two rules on a 7-day calendar: 30-day
  ratio momentum and 30-day mean reversion.
- **Validation:** purged expanding walk-forward, 91-day test blocks from
  2023-03-01 to 2025-11-16; models refit per block on samples whose labels
  ended before the block. Features standardized on the training block and
  clipped at +/-5; Ridge alpha = number of training rows; LightGBM fixed
  (``LGBM_PARAMS``). Target: pair return / (30-day pair volatility x sqrt(h)),
  clipped at +/-5.
- **Trading:** decisions every h days from 2023-03-01: side = sign of the
  prediction (no prediction: keep the position). A side change closes both legs
  and opens the new pair at the next 1h open; legs are sized once at entry to
  1x equity on the BTC leg (0.001 BTC lots, >= 100 USDT) with the ETH leg
  matched in notional. Taker 5 bp + slippage 3 bp per fill; actual funding.
  167 USDT. Cost x2 doubles fee and slippage.
- **Selection:** the highest daily Sharpe among the ten.
- **Gate:** ``GATE``; deflated Sharpe over the family's trials.
- **Report only:** every candidate, 10,000 USDT runs, per-block IC, how often
  a leg moved 9.5% against its entry (a 10x isolated liquidation).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lab import trials
from lab.data import DAY, HOUR
from lab.metrics import daily_equity
from lab.overlay_study import stats, yearly_returns
from lab.pair import D_FEATURES, P_FEATURES, build, labels, load_inputs
from lab.study import DEV_END, _manifest
from lab.validation import walk_forward

STUDY = "pair_2026-11-14"
FAMILY = "eth_btc_relative"
OOS_START = int(datetime(2023, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)
BLOCK = 91 * DAY
CAPITAL = 167.0
REPORT_CAPITAL = 10_000.0
LEG_EXPOSURE = 1.0
TAKER_BPS, SLIPPAGE_BPS = 5.0, 3.0
LOT = {"BTCUSDT": 0.001, "ETHUSDT": 0.001}
MIN_NOTIONAL = {"BTCUSDT": 100.0, "ETHUSDT": 20.0}
ISOLATED_10X_LIQUIDATION_MOVE = 0.095
MIN_TRAIN_ROWS = 100
LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.02, num_leaves=4, max_depth=2, min_child_samples=50,
                   subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, random_state=7,
                   n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1)
CANDIDATES = [{"id": f"{model}_{fs}_{h}d", "kind": "ml", "model": model, "features": fs, "horizon": h}
              for model in ("ridge", "lgbm") for fs in ("P", "PD") for h in (3, 7)]
CANDIDATES += [{"id": "rule_mom30_7d", "kind": "rule", "rule": "mom", "horizon": 7},
               {"id": "rule_rev30_7d", "kind": "rule", "rule": "rev", "horizon": 7}]
GATE = {
    "deflated_sharpe_probability_min": 0.95,
    "sharpe_must_exceed_best_static_pair": True,
    "cost_x2_total_return_min_exclusive": 0.0,
    "max_drawdown_hourly_max": 0.60,
    "round_trips_per_year_max": 150,
    "wipeout_allowed": False,
}


# -- predictions ---------------------------------------------------------------------------------
def fit_predict(model: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    mu, sd = x_train.mean(axis=0), x_train.std(axis=0)
    sd[sd == 0] = 1.0
    z_train = np.clip((x_train - mu) / sd, -5, 5)
    z_test = np.clip((x_test - mu) / sd, -5, 5)
    y = np.clip(y_train, -5, 5)
    if model == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=float(len(z_train))).fit(z_train, y).predict(z_test)
    if model == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**LGBM_PARAMS).fit(z_train, y).predict(z_test)
    raise ValueError(f"unknown model {model}")


def ml_predictions(feats: pd.DataFrame, closes: pd.DataFrame, cand: dict, *, oos_start: int,
                   limit: int) -> tuple[pd.Series, list[dict]]:
    """Out-of-sample predictions for every decision time in the walk-forward blocks, and per-block IC."""
    h = cand["horizon"]
    cols = list(P_FEATURES) + (list(D_FEATURES) if cand["features"] == "PD" else [])
    lab = labels(closes, h)
    times = feats.index.to_numpy(dtype=np.int64)
    x = feats[cols].to_numpy(float)
    y = (lab["y"] / (feats["sigma_pair"] * math.sqrt(h))).to_numpy(float)
    has_x = np.isfinite(x).all(axis=1)
    usable = has_x & np.isfinite(y)
    preds = pd.Series(np.nan, index=feats.index)
    blocks = []
    for fold in walk_forward(times, lab["label_end"].to_numpy(dtype=np.int64), first_test=oos_start, block=BLOCK,
                             limit=limit):
        train = fold.train[usable[fold.train]]
        target = np.flatnonzero((times >= fold.test_start) & (times < fold.test_end) & has_x)
        if len(train) < MIN_TRAIN_ROWS or not len(target):
            continue
        p = fit_predict(cand["model"], x[train], y[train], x[target])
        preds.iloc[target] = p
        scored = fold.test[usable[fold.test]]
        ic = float(spearmanr(preds.iloc[scored], y[scored]).statistic) if len(scored) >= 10 else float("nan")
        blocks.append({"test_start": fold.test_start, "train_rows": int(len(train)), "predicted": int(len(target)),
                       "ic": ic})
    return preds, blocks


def rule_signal(feats: pd.DataFrame, rule: str) -> pd.Series:
    if rule == "mom":
        return np.sign(feats["mom_30"])
    if rule == "rev":
        return -np.sign(feats["dev_30"])
    raise ValueError(rule)


def decisions(signal: pd.Series, horizon: int, *, start: int, end: int) -> dict[int, int]:
    """Side at each decision time of the h-day calendar starting at ``start``; NaN signals are skipped."""
    out = {}
    for t, value in signal.items():
        t = int(t)
        if start <= t < end and (t - start) // DAY % horizon == 0 and (t - start) % DAY == 0 and np.isfinite(value):
            out[t] = int(np.sign(value))
    return out


# -- pair simulation -----------------------------------------------------------------------------
@dataclass
class PairRun:
    curve: pd.DataFrame        # at, equity (hourly)
    round_trips: int
    fees: float
    funding: float
    wiped: bool
    positions: int
    isolated_breaches: int     # positions in which a leg moved >= 9.5% against its entry
    worst_leg_move: float


def merged_hourly(hourly: dict) -> pd.DataFrame:
    b = hourly["BTCUSDT"][["open_time", "open", "high", "low", "close"]].add_suffix("_b")
    e = hourly["ETHUSDT"][["open_time", "open", "high", "low", "close"]].add_suffix("_e")
    return b.merge(e, left_on="open_time_b", right_on="open_time_e").rename(columns={"open_time_b": "open_time"})


def simulate_pair(bars: pd.DataFrame, funding: dict, sides: dict[int, int], *, start: int, end: int,
                  capital: float, cost_multiplier: int = 1) -> PairRun:
    taker, slip = TAKER_BPS * cost_multiplier / 1e4, SLIPPAGE_BPS * cost_multiplier / 1e4
    legs = ("BTCUSDT", "ETHUSDT")
    suffix = {"BTCUSDT": "_b", "ETHUSDT": "_e"}
    frame = bars[(bars["open_time"] >= start - HOUR) & (bars["open_time"] < end)].reset_index(drop=True)
    if frame.empty or int(frame["open_time"].iloc[0]) != start - HOUR:
        raise ValueError("hourly bars must cover the hour before the start")
    cols = {s: {k: frame[k + suffix[s]].to_numpy(float) for k in ("open", "high", "low", "close")} for s in legs}
    times = frame["open_time"].to_numpy(dtype=np.int64)
    qty = dict.fromkeys(legs, 0.0)
    entry = dict.fromkeys(legs, 0.0)
    cash, fees, paid, side = capital, 0.0, 0.0, 0
    trips = positions = breaches = 0
    worst = worst_pos = 0.0
    wiped = False
    curve = []

    def close_all(i):
        nonlocal cash, fees, trips, breaches, worst_pos, side
        for s in legs:
            if qty[s]:
                px = cols[s]["open"][i] * (1 - math.copysign(slip, qty[s]))
                fee = abs(qty[s]) * px * taker
                cash += qty[s] * (px - entry[s]) - fee
                fees += fee
                qty[s] = 0.0
        trips += 1
        breaches += worst_pos >= ISOLATED_10X_LIQUIDATION_MOVE
        worst_pos, side = 0.0, 0

    for i in range(1, len(frame)):
        t = int(times[i])
        for s in legs:
            if qty[s] and t in funding[s]:
                rate, mark = funding[s][t]
                payment = -qty[s] * float(mark) * float(rate)
                cash += payment
                paid += payment
        want = sides.get(t)
        if not wiped and want is not None and want != side:
            if side != 0:
                close_all(i)
            if want != 0:
                btc_px, eth_px = cols["BTCUSDT"]["close"][i - 1], cols["ETHUSDT"]["close"][i - 1]
                q_b = math.floor(cash * LEG_EXPOSURE / btc_px / LOT["BTCUSDT"] + 1e-9) * LOT["BTCUSDT"]
                q_e = math.floor(q_b * btc_px / eth_px / LOT["ETHUSDT"] + 1e-9) * LOT["ETHUSDT"]
                if q_b * btc_px >= MIN_NOTIONAL["BTCUSDT"] and q_e * eth_px >= MIN_NOTIONAL["ETHUSDT"]:
                    for s, q, sign in (("BTCUSDT", q_b, -want), ("ETHUSDT", q_e, want)):
                        px = cols[s]["open"][i] * (1 + sign * slip)
                        fee = q * px * taker
                        cash -= fee
                        fees += fee
                        qty[s], entry[s] = sign * q, px
                    side = want
                    positions += 1
        for s in legs:
            if qty[s]:
                move = ((entry[s] - cols[s]["low"][i]) if qty[s] > 0 else (cols[s]["high"][i] - entry[s])) / entry[s]
                worst_pos = max(worst_pos, move)
                worst = max(worst, move)
        equity = cash + sum(qty[s] * (cols[s]["close"][i] - entry[s]) for s in legs)
        if equity <= 0 and not wiped:
            wiped, equity = True, 0.0
            cash, qty = 0.0, dict.fromkeys(legs, 0.0)
        curve.append((t + HOUR, equity))
    if side != 0:
        breaches += worst_pos >= ISOLATED_10X_LIQUIDATION_MOVE
    return PairRun(pd.DataFrame(curve, columns=["at", "equity"]), trips, fees, paid, wiped, positions, breaches,
                   worst)


# -- statistics and gate -------------------------------------------------------------------------
def describe(run: PairRun, start: int, capital: float) -> dict:
    first_day = start // DAY
    daily = daily_equity([run.curve])
    daily = pd.concat([pd.Series({first_day - 1: capital}), daily[daily.index >= first_day]])
    out = stats(daily)
    equity = np.concatenate([[capital], run.curve["equity"].to_numpy()])
    peak = np.maximum.accumulate(equity)
    years = out["days"] / 365.0
    out.update({"max_drawdown_hourly": float(((peak - equity) / peak).max()), "final_equity": float(equity[-1]),
                "round_trips": run.round_trips, "round_trips_per_year": run.round_trips / years,
                "positions": run.positions, "fees": run.fees, "funding": run.funding, "wiped": run.wiped,
                "isolated_10x_breaches": run.isolated_breaches, "worst_leg_move": run.worst_leg_move,
                "yearly": yearly_returns(daily)})
    out["_daily_returns"] = daily.pct_change().dropna()
    return out


def public(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def gate_checks(selected: dict, cost_x2: dict, statics: dict, deflated: dict) -> dict:
    best_static = max(statics["static_long_pair"]["sharpe"], statics["static_short_pair"]["sharpe"])
    checks = {
        "deflated_sharpe_probability": (deflated["probability"], GATE["deflated_sharpe_probability_min"],
                                        math.isfinite(deflated["probability"])
                                        and deflated["probability"] >= GATE["deflated_sharpe_probability_min"]),
        "sharpe_vs_best_static_pair": (selected["sharpe"], best_static,
                                       math.isfinite(selected["sharpe"]) and selected["sharpe"] > best_static),
        "cost_x2_total_return": (cost_x2["total_return"], GATE["cost_x2_total_return_min_exclusive"],
                                 cost_x2["total_return"] > GATE["cost_x2_total_return_min_exclusive"]),
        "max_drawdown_hourly": (selected["max_drawdown_hourly"], GATE["max_drawdown_hourly_max"],
                                selected["max_drawdown_hourly"] <= GATE["max_drawdown_hourly_max"]),
        "round_trips_per_year": (selected["round_trips_per_year"], GATE["round_trips_per_year_max"],
                                 selected["round_trips_per_year"] <= GATE["round_trips_per_year_max"]),
        "no_wipeout": (selected["wiped"] or cost_x2["wiped"], False, not (selected["wiped"] or cost_x2["wiped"])),
    }
    return {name: {"value": v, "reference": ref, "ok": bool(ok)} for name, (v, ref, ok) in checks.items()}


def _code_identity() -> dict:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "lab", "runner/offline"],
                                             text=True).strip())
    except Exception:
        revision, dirty = "unavailable", True
    return {"git_revision": revision, "lab_or_offline_dirty": dirty}


def run_study(data_dir: Path, premium_dir: Path, out: Path, *, oos_start: int = OOS_START, limit: int = DEV_END,
              registry: Path = trials.REGISTRY, study: str = STUDY) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("use an empty output directory; preserve previous studies")
    if oos_start % DAY or limit % DAY or not oos_start < limit:
        raise ValueError("window must be whole UTC days")
    if trials.load(registry, FAMILY) and any(r["study"] == study for r in trials.load(registry, FAMILY)):
        raise ValueError(f"{study} already recorded its trials; a study runs once")
    inputs = load_inputs(data_dir, premium_dir, limit)
    closes, feats = build(inputs)
    bars = merged_hourly(inputs["hourly"])
    funding = inputs["funding"]

    def run(sides, capital=CAPITAL, cost_multiplier=1):
        return describe(simulate_pair(bars, funding, sides, start=oos_start, end=limit, capital=capital,
                                      cost_multiplier=cost_multiplier), oos_start, capital)

    candidates = {}
    for cand in CANDIDATES:
        if cand["kind"] == "ml":
            signal, blocks = ml_predictions(feats, closes, cand, oos_start=oos_start, limit=limit)
        else:
            signal, blocks = rule_signal(feats, cand["rule"]), []
        sides = decisions(signal, cand["horizon"], start=oos_start, end=limit)
        candidates[cand["id"]] = {"spec": cand, "sides": sides, "blocks": blocks, "base": run(sides),
                                  "capital_10k": run(sides, capital=REPORT_CAPITAL)}
    first = min(t for c in candidates.values() for t in c["sides"])
    statics = {"static_long_pair": run({first: 1}), "static_short_pair": run({first: -1})}
    selected_id = max(candidates, key=lambda k: (np.nan_to_num(candidates[k]["base"]["sharpe"], nan=-np.inf), k))
    chosen = candidates[selected_id]
    cost_x2 = run(chosen["sides"], cost_multiplier=2)
    window = [oos_start, limit]
    rows = [{"family": FAMILY, "study": study, "trial": cid, "config": c["spec"], "window": window,
             "sharpe": float(c["base"]["sharpe"]) if math.isfinite(c["base"]["sharpe"]) else None, "count": 1,
             "note": "daily Sharpe, 167 USDT, base costs, purged walk-forward OOS"} for cid, c in candidates.items()]
    deflated = trials.deflated(chosen["base"]["_daily_returns"], FAMILY, path=registry, extra=rows)
    checks = gate_checks(chosen["base"], cost_x2, statics, deflated)
    passed = all(c["ok"] for c in checks.values())
    summary = {
        "window": {"oos_start_ms": oos_start, "limit_ms": limit, "holdout": "not computed"},
        "selected": selected_id, "selected_spec": chosen["spec"],
        "strategy": public(chosen["base"]), "strategy_cost_x2": public(cost_x2), "deflated_sharpe": deflated,
        "static_pairs": {k: public(v) for k, v in statics.items()},
        "gate": {"passed": passed, "checks": checks},
        "candidates": {cid: {"base": public(c["base"]), "capital_10k": public(c["capital_10k"]),
                             "decisions": len(c["sides"]), "blocks": c["blocks"],
                             "mean_ic": float(np.nanmean([b["ic"] for b in c["blocks"]])) if c["blocks"] else None}
                       for cid, c in candidates.items()},
    }
    frozen = {
        "schema": 1, "kind": "eth_btc_pair", "symbols": ["BTCUSDT", "ETHUSDT"], "study": "lab.pair_study",
        "selected": chosen["spec"], "candidates": CANDIDATES, "lgbm_params": LGBM_PARAMS,
        "evaluation": summary["window"], "trials": len(CANDIDATES), "gate": GATE, "gate_result": checks,
        "screen_passed": passed, "passed": False,
        "mainnet": "refused: requires an accepted forward test, recorded by a person",
        "code": _code_identity(), "data": {"market": _manifest(data_dir), "premium": _manifest(premium_dir)},
        "caveat": ("The registrant knew that ETH/BTC fell over 2022-2025; the gate therefore requires beating "
                   "both static pair positions. The development window was used heavily for directional studies."),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    (out / "final_spec.json").write_text(json.dumps(frozen, indent=2, default=float) + "\n")
    trials.record(rows, path=registry)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data_files/binance_public"))
    parser.add_argument("--premium-dir", type=Path, default=Path("data_files/binance_premium_2025-11-16"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_study(args.data_dir, args.premium_dir, args.out)
    print(json.dumps({"selected": summary["selected"], "gate": summary["gate"]}, indent=2, default=float))


if __name__ == "__main__":
    main()


__all__ = ["CANDIDATES", "GATE", "run_study", "simulate_pair", "decisions", "ml_predictions", "gate_checks"]
