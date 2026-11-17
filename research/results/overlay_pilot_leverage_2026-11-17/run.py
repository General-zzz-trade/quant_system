"""The pilot's strategy at 2-10x positions (owner 2026-11-17: "改为资金的10倍", "10 倍，止损线一起重定"): what each
setting would have done, before the runtime's caps are reset. Not a selection: the owner chose 10x.

As research/results/overlay_pilot_leverage_2026-11-16/run.py (the frozen overlay, 150 USDT over BTC and ETH, real lot
rules, isolated margin with liquidation, the stop at the average entry, the 4-hour pause after a stop, both symbols
stepped together, development window only), over: position multiples 2-10; exchange leverage and stop (10x 4.5 %,
15x 3 %, 20x 2.25 %), only where the exchange leverage is at least the multiple; drawdown latches none, 3/12, 5/12 and
10/25 (daily/weekly %, whole account); costs x1, and x2 without latches.
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
RULES = {"BTCUSDT": SymbolRules(D("0.001"), D("0.001"), D("50")),        # Binance exchangeInfo, checked 2026-11-17
         "ETHUSDT": SymbolRules(D("0.001"), D("0.001"), D("20"))}
CAPITAL = 150.0
MULTIPLES = (2, 3, 5, 7, 10)
MARGINS = ((10, "0.045"), (15, "0.03"), (20, "0.0225"))                  # exchange leverage, stop
LATCHES = {"none": None, "3/12": (0.03, 0.12), "5/12": (0.05, 0.12), "10/25": (0.10, 0.25)}
PAUSE_MS = 4 * 3_600_000                                                 # STOP_PAUSE_SECONDS


def config_for(symbol, cost, multiple, margin, stop):
    rules = RULES[symbol]
    m = D(cost)
    return rules, ReplayConfig(starting_balance=str(CAPITAL / len(SYMBOLS)), maker_bps=str(2 * m),
                               taker_bps=str(5 * m), slippage_bps=str(3 * m), stop_slippage_bps=str(10 * m),
                               participation="0.01", qty_step=str(rules.step), min_notional=str(rules.min_notional),
                               ttl_bars=SPEC.ttl_bars, entry_mode="market", isolated_leverage=str(margin),
                               max_exposure=str(multiple), maintenance_margin_rate="0.005", stop_pct=stop)


def run(bars, targets, funding, multiple, margin, stop, cost, latch):
    """As the 2026-11-16 run: both symbols hour by hour, the pause after a stop per symbol, the drawdown latches on
    the whole account (close everything, no entry until the UTC day or week ends; checked on bar closes)."""
    sleeves, maps, by_time = {}, {}, {}
    for symbol in SYMBOLS:
        rules, config = config_for(symbol, cost, multiple, margin, stop)
        sleeves[symbol] = Sleeve(symbol, SPEC, rules, config)
        t = targets[symbol]
        maps[symbol] = dict(zip(t["available_at"].astype(np.int64).tolist(),
                                zip((t["target"] * multiple).tolist(), t["ready"].tolist())))
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
            continue
        for symbol, bar in step.items():
            sleeve = sleeves[symbol]
            before = len(sleeve.broker.fills)
            rate = funding[symbol][bar.open_time][0] if bar.open_time in funding[symbol] else None
            sleeve.on_bar(bar, rate)
            if any(f["order"] == "protective_stop" for f in sleeve.broker.fills[before:]):
                paused[symbol] = bar.available_at + PAUSE_MS
        close = next(iter(step.values())).available_at
        equity = sum(float(s.broker.equity) for s in sleeves.values())
        this_day, this_week = (close - 1) // DAY, ((close - 1) // DAY + 3) // 7
        if this_day != day:
            day, day_base = this_day, equity if day is not None else CAPITAL
        if this_week != week:
            week, week_base = this_week, equity if week is not None else CAPITAL
        if latch and close >= latched_until and (equity <= day_base * (1 - latch[0])
                                                 or equity <= week_base * (1 - latch[1])):
            weekly = equity <= week_base * (1 - latch[1])
            latched_until = ((this_week + 1) * 7 - 3) * DAY + 1 if weekly else (this_day + 1) * DAY + 1
            latch_days += 1
        for symbol, bar in step.items():
            target, ready = maps[symbol].get(bar.available_at, (0.0, False))
            if close < latched_until:
                target, ready = 0.0, True
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


def stats(curve, initial):
    daily = daily_equity([curve])
    daily = daily[daily.index >= EVAL_START // DAY]
    equity = pd.concat([pd.Series({EVAL_START // DAY - 1: initial}), daily])
    r = equity.pct_change().dropna()
    years = len(r) / 365
    lev = (curve.set_index("at")["notional"].abs() / curve.set_index("at")["equity"]).replace(
        [np.inf, -np.inf], np.nan).dropna()
    final = float(equity.iloc[-1])
    return {"final_usdt": round(final, 2), "total_return": round(final / initial - 1, 4),
            "cagr": round((final / initial) ** (1 / years) - 1, 4) if final > 0 else -1.0,
            "sharpe": round(float(r.mean() / r.std() * math.sqrt(365)), 3) if r.std() > 0 else 0.0,
            "max_drawdown": round(float((1 - equity / equity.cummax()).max()), 4),
            "worst_day": round(float(r.min()), 4),
            **{f"loss_stop_{int(x * 100)}pct_share_of_90_day_pilots": round(float(um_lab.loss_stop_share(r, x, 90)), 3)
               for x in (0.15, 0.30, 0.50)},
            "exposure_mean": round(float(lev.mean()), 3), "exposure_max": round(float(lev.max()), 3),
            "days": int(len(r))}


def main():
    bars, funding, frames, fseries, targets = {}, {}, {}, {}, {}
    for symbol in SYMBOLS:
        bars[symbol], funding[symbol], frames[symbol] = load_symbol(DATA, symbol)
        fseries[symbol] = funding_series(funding[symbol])
        targets[symbol] = compute_targets(frames[symbol], fseries[symbol], SPEC)
    out = {"spec": SPEC.record(), "capital_usdt": CAPITAL, "window": "2021-01-10..2025-11-16 (development only)",
           "rules": {k: asdict(v) for k, v in RULES.items()}, "runs": {}}
    for multiple in MULTIPLES:
        for margin, stop in MARGINS:
            if margin < multiple:
                continue
            for latch_name, latch in [*LATCHES.items(), ("none", None)]:
                cost = 2 if (latch is None and f"{multiple}x {margin}x/{stop} none x1" in out["runs"]) else 1
                key = f"{multiple}x {margin}x/{stop} {latch_name} x{cost}"
                curve, counts = run(bars, targets, funding, multiple, margin, stop, cost, latch)
                out["runs"][key] = {"stats": stats(curve, CAPITAL), "counts": counts}
                print(key, json.dumps(out["runs"][key]["stats"]), json.dumps(counts), flush=True)
                OUT.write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
