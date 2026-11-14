"""ETH 4h breakout, 10x isolated, 1-3x sizing: ONE predeclared spec and gate, development data only.

    python -m lab.breakout_study --data-dir data_files/binance_public --out research/results/breakout_<date>

Everything that decides the outcome is fixed here and in ``lab/breakout.py`` and
was committed before any result on real data was computed (see
research/protocols/2026-11-14-eth-4h-breakout.md):

- **Spec.** ``lab.breakout.BreakoutSpec()`` defaults (20/10-bar channels on 4h
  bars, stop 1.5 x ATR(14), sizing volatility over 42 bars) and
  ``SizingRule(target_vol=1.0, max_trade_loss=0.06)`` (1-3x, 10x isolated,
  entry refused when the stop is beyond half the liquidation distance).
- **Account.** ETHUSDT only, 167 USDT, isolated margin at 10x.
- **Costs.** Market orders at the next 1h open: taker 5 bp + slippage 3 bp;
  protective stops: taker 5 bp + 10 bp slippage; actual funding. ``cost x2``
  doubles fees and both slippages (funding unchanged).
- **Window.** 2021-01-10 to 2025-11-16. Bars from ``DEV_END`` on are cut off
  before features are computed; they never enter a simulation or a statistic.
- **Gate.** All of ``GATE`` must hold (``gate_checks``).
- **Report only.** A daily two-sided trend control with the same sizing, stop
  and costs; ETH buy-and-hold at 1x without costs.

The development window has been used heavily, and 4h features were tried in
the 2026-11-13 lab study, so passing earns forward paper trading only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from lab.breakout import BreakoutSpec, desired_side, features, stop_distance
from lab.data import DAY, aggregate, load_symbol
from lab.metrics import daily_equity
from lab.orders import make_order
from lab.overlay_study import stats, yearly_returns
from lab.rebalance import floor_step
from lab.simulate import SYMBOL_RULES, SymbolRules
from lab.sizing import SizingRule
from lab.study import DEV_END, EVAL_START, _manifest
from runner.offline.broker import BarBroker, ReplayConfig

D = Decimal
HOUR = 3_600_000
SYMBOL = "ETHUSDT"
START_BALANCE = "167"
SPEC = BreakoutSpec()
RULE = SizingRule(target_vol=1.0, max_trade_loss=0.06)
CONTROL_LOOKBACKS = (20, 60, 120)        # days; the lab BASE lookbacks

GATE = {
    "cost_x2_total_return_min_exclusive": 0.0,
    "daily_sharpe_min": 0.5,
    "max_drawdown_hourly_max": 0.60,
    "liquidations_max": 0,                  # base and cost x2 runs
    "round_trips_per_year_max": 150,
    "mean_net_per_round_trip_min": 0.003,   # fraction of entry notional, after fees, slippage and funding
}


def broker_config(rules: SymbolRules, cost_multiplier: int = 1) -> ReplayConfig:
    m = D(cost_multiplier)
    return ReplayConfig(starting_balance=START_BALANCE, maker_bps=str(D(2) * m), taker_bps=str(D(5) * m),
                        slippage_bps=str(D(3) * m), limit_offset_bps="2", participation="0.01",
                        max_exposure=str(RULE.max_exposure), qty_step=str(rules.step),
                        min_notional=str(rules.min_notional), ttl_bars=1, entry_mode="market",
                        isolated_leverage=str(RULE.isolated_leverage),
                        maintenance_margin_rate=str(RULE.maintenance_margin_rate),
                        stop_slippage_bps=str(D(10) * m))


@dataclass
class RunResult:
    curve: pd.DataFrame                       # at, equity, qty, close (hourly, at each bar close)
    trades: list[dict]
    open_trade: dict | None
    liquidations: list[dict]
    entries: list[dict] = field(default_factory=list)
    refusals: Counter = field(default_factory=Counter)
    broker_rejections: Counter = field(default_factory=Counter)
    fees: float = 0.0
    funding: float = 0.0


def _sign(qty) -> int:
    return 1 if qty > 0 else -1 if qty < 0 else 0


def trade_ledger(fills: list[dict], orders: list[dict], funding_log: list[dict]) -> tuple[list[dict], dict | None]:
    """Fills -> round trips (flat -> position -> flat), with fees and funding attributed.

    Funding settled at T belongs to the trade open just before T: entry_at < T <= exit_at.
    """
    reasons = {row["id"]: row.get("reason") for row in orders}
    trades, pos, cur = [], D(0), None
    for f in fills:
        qty, price, fee = D(f["qty"]), D(f["price"]), D(f["fee"])
        signed = qty if f["side"] == "buy" else -qty
        if pos == 0:
            cur = {"side": _sign(signed), "entry_at": f["at"], "entry_price": price, "notional": D(0),
                   "cash": D(0), "fees": D(0), "max_qty": D(0)}
        if abs(pos + signed) > abs(pos):
            cur["notional"] += qty * price
        cur["cash"] -= signed * price
        cur["fees"] += fee
        pos += signed
        cur["max_qty"] = max(cur["max_qty"], abs(pos))
        if pos == 0:
            kind = f["order"] if isinstance(f["order"], str) else f"signal_{reasons.get(f['order'])}"
            cur.update(exit_at=f["at"], exit_price=price, exit_kind=kind)
            trades.append(cur)
            cur = None
    for t in trades:
        t["funding"] = sum((D(e["cashflow"]) for e in funding_log if t["entry_at"] < e["at"] <= t["exit_at"]), D(0))
        t["net"] = t["cash"] - t["fees"] + t["funding"]
        t["net_over_notional"] = t["net"] / t["notional"]
    return trades, cur


def run(bars: list, funding: dict, rows: pd.DataFrame, decide, *, start_ms: int, end_ms: int,
        cost_multiplier: int = 1, rules: SymbolRules | None = None) -> RunResult:
    """Hourly loop: settle funding, process the bar (fills, stop, liquidation), decide at feature rows."""
    rules = rules or SYMBOL_RULES[SYMBOL]
    broker = BarBroker(broker_config(rules, cost_multiplier))
    by_time = {int(r.available_at): r for r in rows.itertuples(index=False)}
    result = RunResult(curve=pd.DataFrame(), trades=[], open_trade=None, liquidations=[])
    curve = []
    for bar in bars:
        if bar.available_at <= start_ms or bar.open_time >= end_ms:
            continue
        broker.now, broker.mark = bar.open_time, bar.open
        if bar.open_time in funding:
            broker.settle_funding(bar.open_time, funding[bar.open_time][0], bar.open)
        broker.process_bar(bar)
        row = by_time.get(bar.available_at)
        if row is not None and row.ready:
            _act(broker, bar, row, decide, rules, result)
        curve.append((bar.available_at, float(broker.equity), float(broker.position.qty), float(bar.close)))
    result.curve = pd.DataFrame(curve, columns=["at", "equity", "qty", "close"])
    result.trades, result.open_trade = trade_ledger(broker.fills, broker.orders, broker.funding_log)
    result.liquidations = list(broker.liquidations)
    result.broker_rejections = Counter(r["rejection"] for r in broker.orders if r.get("rejection"))
    result.fees, result.funding = float(broker.fees), float(broker.funding)
    return result


def _act(broker: BarBroker, bar, row, decide, rules: SymbolRules, result: RunResult) -> None:
    side = _sign(broker.position.qty)
    want = decide(side, row)
    if want == side:
        return
    at, close = bar.available_at, bar.close
    if side != 0:
        broker.send_order(make_order(order_id=f"exit:{at}", source="breakout", symbol=SYMBOL,
                                     side="sell" if side > 0 else "buy", qty=abs(broker.position.qty), price=close,
                                     reduce_only=True, reason="exit"))
        broker.clear_stop()
    if want == 0:
        return
    distance = stop_distance(row, SPEC)
    exposure, reason = RULE.exposure(want, float(row.vol), distance)
    if exposure <= 0:
        result.refusals["stop beyond half the liquidation distance" if "liquidation" in reason
                        else "volatility or stop unavailable"] += 1
        return
    qty = floor_step(D(str(exposure * SPEC.size_headroom)) * broker.equity / close, rules.step)
    if qty < rules.min_qty or qty * close < rules.min_notional:
        result.refusals["below exchange minimum"] += 1
        return
    broker.send_order(make_order(order_id=f"entry:{at}", source="breakout", symbol=SYMBOL,
                                 side="buy" if want > 0 else "sell", qty=qty, price=close,
                                 reduce_only=False, reason="entry"))
    stop = close * (1 - want * D(str(distance)))
    broker.set_stop(stop)
    result.entries.append({"at": at, "side": want, "exposure": exposure, "capped_by_trade_loss": "max trade loss"
                           in reason, "stop_distance": distance, "qty": str(qty), "stop": str(stop)})


def control_rows(hourly: pd.DataFrame, four_hour: pd.DataFrame) -> pd.DataFrame:
    """Daily two-sided trend (mean sign of 20/60/120-day returns) at each daily close, with the
    same 4h ATR / volatility for sizing and the stop. Report only."""
    daily = aggregate(hourly, 24)
    close = daily["close"]
    score = pd.concat([np.sign(close / close.shift(lb) - 1.0) for lb in CONTROL_LOOKBACKS], axis=1).mean(
        axis=1, skipna=False)
    trend = pd.DataFrame({"available_at": daily["available_at"], "trend": np.sign(score)})
    rows = four_hour.merge(trend, on="available_at", how="inner")
    rows["ready"] = rows["ready"] & np.isfinite(rows["trend"])
    return rows


def control_side(side: int, row) -> int:
    return int(row.trend)


def hourly_max_drawdown(curve: pd.DataFrame, initial: float) -> float:
    equity = np.concatenate([[initial], curve["equity"].to_numpy()])
    peak = np.maximum.accumulate(equity)
    return float(((peak - equity) / peak).max())


def daily_curve(curve: pd.DataFrame, first_day: int, initial: float) -> pd.Series:
    daily = daily_equity([curve])
    daily = daily[daily.index >= first_day]
    return pd.concat([pd.Series({first_day - 1: initial}), daily])


def describe(result: RunResult, first_day: int, initial: float) -> dict:
    equity = daily_curve(result.curve, first_day, initial)
    out = stats(equity)
    years = out["days"] / 365.0
    nets = [float(t["net_over_notional"]) for t in result.trades]
    exposures = [e["exposure"] for e in result.entries]
    out.update({
        "max_drawdown_hourly": hourly_max_drawdown(result.curve, initial),
        "final_equity": float(result.curve["equity"].iloc[-1]),
        "round_trips": len(result.trades), "round_trips_per_year": len(result.trades) / years,
        "mean_net_per_round_trip": float(np.mean(nets)) if nets else float("nan"),
        "median_net_per_round_trip": float(np.median(nets)) if nets else float("nan"),
        "win_rate": float(np.mean([n > 0 for n in nets])) if nets else float("nan"),
        "liquidations": len(result.liquidations),
        "exit_kinds": dict(Counter(t["exit_kind"] for t in result.trades)),
        "net_by_side": {side: float(sum(t["net"] for t in result.trades if t["side"] == s))
                        for side, s in (("long", 1), ("short", -1))},
        "fees": result.fees, "funding": result.funding,
        "entries": len(result.entries),
        "entry_exposure": ({"median": float(np.median(exposures)), "p10": float(np.quantile(exposures, 0.1)),
                            "p90": float(np.quantile(exposures, 0.9)),
                            "capped_by_trade_loss": float(np.mean([e["capped_by_trade_loss"]
                                                                   for e in result.entries]))}
                           if exposures else {}),
        "refused_entries": dict(result.refusals), "broker_rejections": dict(result.broker_rejections),
        "open_at_end": result.open_trade is not None,
    })
    return out


def gate_checks(base: dict, cost_x2: dict) -> dict:
    checks = {
        "cost_x2_total_return": (cost_x2["total_return"], GATE["cost_x2_total_return_min_exclusive"],
                                 cost_x2["total_return"] > GATE["cost_x2_total_return_min_exclusive"]),
        "daily_sharpe": (base["sharpe"], GATE["daily_sharpe_min"],
                         math.isfinite(base["sharpe"]) and base["sharpe"] >= GATE["daily_sharpe_min"]),
        "max_drawdown_hourly": (base["max_drawdown_hourly"], GATE["max_drawdown_hourly_max"],
                                base["max_drawdown_hourly"] <= GATE["max_drawdown_hourly_max"]),
        "liquidations": (base["liquidations"] + cost_x2["liquidations"], GATE["liquidations_max"],
                         base["liquidations"] + cost_x2["liquidations"] <= GATE["liquidations_max"]),
        "round_trips_per_year": (base["round_trips_per_year"], GATE["round_trips_per_year_max"],
                                 base["round_trips_per_year"] <= GATE["round_trips_per_year_max"]),
        "mean_net_per_round_trip": (base["mean_net_per_round_trip"], GATE["mean_net_per_round_trip_min"],
                                    math.isfinite(base["mean_net_per_round_trip"])
                                    and base["mean_net_per_round_trip"] >= GATE["mean_net_per_round_trip_min"]),
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


def _write_trades(path: Path, trades: list[dict]) -> None:
    columns = ["side", "entry_at", "exit_at", "entry_price", "exit_price", "max_qty", "notional", "fees", "funding",
               "net", "net_over_notional", "exit_kind"]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for t in trades:
            writer.writerow([str(t[c]) for c in columns])


def run_study(data_dir: Path, out: Path, *, eval_start: int = EVAL_START, dev_end: int = DEV_END) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("use an empty output directory; preserve previous studies")
    if eval_start % DAY or dev_end % DAY or not eval_start < dev_end:
        raise ValueError("evaluation window must be whole UTC days")
    out.mkdir(parents=True, exist_ok=True)
    bars, funding, hourly = load_symbol(data_dir, SYMBOL)
    # Data from dev_end on never reaches a feature, a simulation or a statistic.
    bars = [b for b in bars if b.open_time < dev_end]
    hourly = hourly[hourly["open_time"] < dev_end].reset_index(drop=True)
    funding = {t: v for t, v in funding.items() if t < dev_end}
    rows = features(hourly, SPEC)
    first_day, initial = eval_start // DAY, float(START_BALANCE)

    def simulate(decide, feature_rows, cost_multiplier=1):
        return run(bars, funding, feature_rows, decide, start_ms=eval_start, end_ms=dev_end,
                   cost_multiplier=cost_multiplier)

    base = simulate(desired_side, rows)
    if base.curve["at"].iloc[-1] > dev_end:
        raise RuntimeError("development statistics reached the held-out window")
    x2 = simulate(desired_side, rows, cost_multiplier=2)
    base_stats, x2_stats = describe(base, first_day, initial), describe(x2, first_day, initial)
    checks = gate_checks(base_stats, x2_stats)
    passed = all(c["ok"] for c in checks.values())

    control = simulate(control_side, control_rows(hourly, rows))
    closes = hourly.groupby((hourly["available_at"].to_numpy() - 1) // DAY)["close"].last()
    closes = closes[closes.index >= first_day - 1]
    if closes.index[0] != first_day - 1:
        raise ValueError("no close for the day before the evaluation window")
    hold = initial * closes / closes.iloc[0]
    window = {"eval_start_ms": eval_start, "dev_end_ms": dev_end,
              "holdout": "not computed; the final test is forward paper trading"}
    config = broker_config(SYMBOL_RULES[SYMBOL]).record()
    frozen = {
        "schema": 1, "kind": "eth_4h_breakout_isolated", "symbols": [SYMBOL], "spec": SPEC.record(),
        "sizing": RULE.record(), "broker": config, "study": "lab.breakout_study", "evaluation": window,
        "selection": "none: one predeclared spec", "trials": 1, "gate": GATE, "gate_result": checks,
        "screen_passed": passed,
        # A development-window screen earns forward paper trading at most; never the mainnet flag.
        "passed": False,
        "mainnet": "refused: requires an accepted forward paper test, recorded by a person",
        "code": _code_identity(), "data": _manifest(data_dir),
        "caveat": ("The development window was used heavily before, and 4h features were tried in the "
                   "2026-11-13 lab study; this is a screen, not an independent test. The runtime's daily/weekly "
                   "drawdown latches are not modelled."),
    }
    summary = {
        "window": window, "spec": SPEC.record(), "sizing": RULE.record(), "broker": config,
        "strategy": base_stats, "strategy_cost_x2": x2_stats,
        "yearly": {"strategy": yearly_returns(daily_curve(base.curve, first_day, initial)),
                   "control": yearly_returns(daily_curve(control.curve, first_day, initial)),
                   "eth_hold_1x": yearly_returns(hold)},
        "gate": {"passed": passed, "checks": checks},
        "report_only": {"control_daily_trend": describe(control, first_day, initial),
                        "eth_hold_1x_no_costs": stats(hold)},
    }
    _write_trades(out / "trades.csv", base.trades)
    (out / "final_spec.json").write_text(json.dumps(frozen, indent=2, default=float) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data_files/binance_public"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_study(args.data_dir, args.out)
    print(json.dumps(summary["gate"], indent=2, default=float))


if __name__ == "__main__":
    main()


__all__ = ["SPEC", "RULE", "GATE", "run", "run_study", "trade_ledger", "gate_checks", "control_rows"]
