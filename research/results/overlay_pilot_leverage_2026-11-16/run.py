"""The pilot's strategy at the leverage the tech lead chose (2026-11-16 evening): the frozen overlay spec
(lab/overlay_study.py SPEC, identical to research/results/overlay_pilot_2026-11-16/final_spec.json), positions of 1,
2, 2.5 and 3 times the strategy capital, 10x isolated margin with a 4.5 % stop at the average entry (the runtime's
exchange stop), 150 USDT split over BTC and ETH (75 each) with Binance's real lot rules.

Development window only (2021-01-10 .. 2025-11-16, lab/study.py); nothing after DEV_END is read. Costs are the
shared broker's (taker 5 bp, 3 bp market slippage, 10 bp stop slippage, real funding); a second run doubles them.
The two symbols are stepped together with the 4-hour entry pause after a stop and the daily 5 % / weekly 12 %
drawdown latches (as in the runtime; checked on bar closes). Reported: return, Sharpe, drawdown, worst day, stops,
liquidations, latch trips, and how often the 15 % loss stop would have ended a 30- or 90-day pilot. Not a selection:
the owner chose the leverage; these numbers show what it means.
"""
import json
import math
import sys
from dataclasses import asdict
from decimal import Decimal as D
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lab import um_lab  # noqa: E402
from lab.data import DAY, funding_series, load_symbol  # noqa: E402
from lab.metrics import daily_equity  # noqa: E402
from lab.overlay_study import SPEC  # noqa: E402
from lab.simulate import Sleeve  # noqa: E402
from lab.strategy import compute_targets  # noqa: E402
from lab.study import DEV_END, EVAL_START, SYMBOLS  # noqa: E402
from trading.adapters.simulated.bar_broker import ReplayConfig, SymbolRules  # noqa: E402

DATA = ROOT / "data_files" / "binance_public"
OUT = Path(__file__).with_name("results.json")
RULES = {"BTCUSDT": SymbolRules(D("0.001"), D("0.001"), D("50")),        # Binance, checked 2026-11-16
         "ETHUSDT": SymbolRules(D("0.001"), D("0.001"), D("20"))}
CAPITAL, LEVERAGES, STOP, MARGIN_X = 150.0, (1.0, 2.0, 2.5, 3.0), "0.045", "10"


PAUSE_MS = 4 * 3_600_000                 # the live entry pause after a stop (STOP_PAUSE_SECONDS)
DAILY_LATCH, WEEKLY_LATCH = 0.05, 0.12    # the runtime's default drawdown latches, on the whole account (owner)


def config_for(symbol, cost):
    rules = RULES[symbol]
    m = D(cost)
    return rules, ReplayConfig(starting_balance=str(CAPITAL / len(SYMBOLS)), maker_bps=str(2 * m),
                               taker_bps=str(5 * m), slippage_bps=str(3 * m), stop_slippage_bps=str(10 * m),
                               participation="0.01", qty_step=str(rules.step), min_notional=str(rules.min_notional),
                               ttl_bars=SPEC.ttl_bars, entry_mode="market", isolated_leverage=MARGIN_X,
                               max_exposure="3", maintenance_margin_rate="0.005", stop_pct=STOP)


def run(bars, targets, funding, leverage, cost, latches=True):
    """Both symbols stepped together, hour by hour, with the live rules a single sleeve cannot see: the 4-hour entry
    pause after a stop (per symbol) and the daily / weekly drawdown latches on the whole account, which close every
    position and refuse entries until the UTC day (week) ends. Latches are checked on bar closes (live checks the
    equity about every minute, so live trips earlier, nearer the limit)."""
    sleeves, maps, by_time = {}, {}, {}
    for symbol in SYMBOLS:
        rules, config = config_for(symbol, cost)
        sleeves[symbol] = Sleeve(symbol, SPEC, rules, config)
        t = targets[symbol]
        maps[symbol] = dict(zip(t["available_at"].astype(np.int64).tolist(),
                                zip((t["target"] * leverage).tolist(), t["ready"].tolist())))
        for bar in bars[symbol]:
            if EVAL_START < bar.available_at and bar.open_time < DEV_END:
                by_time.setdefault(bar.open_time, {})[symbol] = bar
    paused = {s: 0 for s in SYMBOLS}
    day_base = week_base = CAPITAL
    day = week = None
    latched_until, latch_days, rows = 0, 0, []
    for open_time in sorted(by_time):
        step = by_time[open_time]
        if len(step) < len(SYMBOLS):
            continue                                     # both symbols on the same hour, as the runtime has them
        for symbol, bar in step.items():
            sleeve = sleeves[symbol]
            before = len(sleeve.broker.fills)
            rate = funding[symbol][bar.open_time][0] if bar.open_time in funding[symbol] else None
            sleeve.on_bar(bar, rate)
            if any(f["order"] == "protective_stop" for f in sleeve.broker.fills[before:]):
                paused[symbol] = bar.available_at + PAUSE_MS
        close = next(iter(step.values())).available_at
        equity = sum(float(s.broker.equity) for s in sleeves.values())
        this_day, this_week = (close - 1) // DAY, ((close - 1) // DAY + 3) // 7       # UTC day; week from Monday
        if this_day != day:
            day, day_base = this_day, equity if day is not None else CAPITAL
        if this_week != week:
            week, week_base = this_week, equity if week is not None else CAPITAL
        if latches and close >= latched_until and (equity <= day_base * (1 - DAILY_LATCH)
                                                   or equity <= week_base * (1 - WEEKLY_LATCH)):
            weekly = equity <= week_base * (1 - WEEKLY_LATCH)
            latched_until = ((this_week + 1) * 7 - 3) * DAY + 1 if weekly else (this_day + 1) * DAY + 1
            latch_days += 1
        for symbol, bar in step.items():
            target, ready = maps[symbol].get(bar.available_at, (0.0, False))
            if close < latched_until:
                target, ready = 0.0, True                # latched: close everything, no entry
            elif bar.available_at < paused[symbol]:
                target = 0.0 if target * float(sleeves[symbol].broker.position.qty) <= 0 else target
                if sleeves[symbol].broker.position.qty == 0:
                    target = 0.0
            sleeves[symbol].decide(bar, target, ready)
        rows.append((close, equity, sum(abs(float(s.broker.position.qty) * float(s.broker.mark))
                                        for s in sleeves.values())))
    curve = pd.DataFrame(rows, columns=["at", "equity", "notional"])
    counts = {s: {"fills": len(v.broker.fills),
                  "stops": sum(1 for f in v.broker.fills if f["order"] == "protective_stop"),
                  "liquidations": len(v.broker.liquidations),
                  "entries": sum(1 for o in v.broker.orders if not o["reduce_only"])} for s, v in sleeves.items()}
    counts["latch_trips"] = latch_days
    return curve, counts


def stats(curves, initial):
    daily = daily_equity(curves)
    daily = daily[daily.index >= EVAL_START // DAY]
    equity = pd.concat([pd.Series({EVAL_START // DAY - 1: initial}), daily])
    r = equity.pct_change().dropna()
    years = len(r) / 365
    gross = sum(c.set_index("at")["notional"].abs() for c in curves)
    eq = sum(c.set_index("at")["equity"] for c in curves)
    lev = (gross / eq).replace([np.inf, -np.inf], np.nan).dropna()
    week = (1 + r).groupby(pd.to_datetime(r.index * DAY, unit="ms").to_period("W")).prod() - 1
    return {"final_usdt": round(float(equity.iloc[-1]), 2), "total_return": round(float(equity.iloc[-1] / initial - 1), 4),  # noqa: E501
            "cagr": round(float((equity.iloc[-1] / initial) ** (1 / years) - 1), 4) if equity.iloc[-1] > 0 else -1.0,
            "sharpe": round(float(r.mean() / r.std() * math.sqrt(365)), 3),
            "max_drawdown": round(float((1 - equity / equity.cummax()).max()), 4),
            "worst_day": round(float(r.min()), 4), "days_below_minus_5pct": int((r < -0.05).sum()),
            "weeks_below_minus_12pct": int((week < -0.12).sum()),
            "loss_stop_15pct_share_of_90_day_pilots": round(float(um_lab.loss_stop_share(r, 0.15, 90)), 3),
            "loss_stop_15pct_share_of_30_day_pilots": round(float(um_lab.loss_stop_share(r, 0.15, 30)), 3),
            "exposure_mean": round(float(lev.mean()), 3), "exposure_max": round(float(lev.max()), 3),
            "days": int(len(r))}


bars, funding, frames, fseries, targets = {}, {}, {}, {}, {}
for symbol in SYMBOLS:
    bars[symbol], funding[symbol], frames[symbol] = load_symbol(DATA, symbol)
    fseries[symbol] = funding_series(funding[symbol])
    targets[symbol] = compute_targets(frames[symbol], fseries[symbol], SPEC)
out = {"spec": SPEC.record(), "capital_usdt": CAPITAL, "isolated_leverage": MARGIN_X, "stop_pct": STOP,
       "window": "2021-01-10..2025-11-16 (development only)", "rules": {k: asdict(v) for k, v in RULES.items()},
       "runs": {}}
for leverage in LEVERAGES:
    for cost, latches in ((1, True), (2, True), (1, False)):
        curve, counts = run(bars, targets, funding, leverage, cost, latches)
        key = f"{leverage:g}x cost x{cost}" + ("" if latches else " no latches")
        out["runs"][key] = {"stats": stats([curve], CAPITAL), "counts": counts}
        print(key, json.dumps(out["runs"][key]["stats"]), json.dumps(counts), flush=True)
OUT.write_text(json.dumps(out, indent=1, default=str) + "\n")
