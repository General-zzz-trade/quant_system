"""Earnings drift on the US-stock perpetuals' underlyings.

Protocol: research/protocols/2026-11-18-us-earnings-drift.md, committed before any result (the 2026-11-18 standards).

    python research/results/us_earnings_drift_2026-11-18/run.py explore   # 4 configurations; choose one; the power
    python research/results/us_earnings_drift_2026-11-18/run.py confirm   # the chosen one, once (power >= 80 % only)
    python research/results/us_earnings_drift_2026-11-18/run.py final     # the holdout, once (after a pass only)

Each step writes its JSON next to this file; the explore step appends the 4 trials to research/trials.jsonl.
"""
from __future__ import annotations

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
from lab import trials as registry  # noqa: E402
from lab import um_lab as lab  # noqa: E402
from lab import us_earnings as ue  # noqa: E402
from lab import us_equity as us  # noqa: E402

DAY = ue.DAY
DATA = ROOT / "data_files"
CALENDARS = ("us_earnings_2016-11-17_2019-11-17", "us_earnings_2019-11-17_2022-11-17",
             "us_earnings_2022-11-17_2025-12-01", "us_earnings_2026-11-12")
TRADFI = ROOT / "research" / "lab" / "runs" / "2026-11-17b" / "tradfi_contracts.json"
WINDOWS = {"explore": ("2016-11-17", "2021-11-17"), "confirm": ("2021-11-17", "2024-11-17"),
           "final": ("2024-11-17", "2026-11-13")}
CONFIGS = [{"signal": s, "hold": h} for s in ("surprise", "reaction") for h in (5, 20)]
LITERATURE_EFFECT = {5: 0.003, 20: 0.005}          # an estimate per event for large caps (the protocol)
EXCLUDED, DKNG_FROM, MAX_REACTION = {"SHAZ"}, "2020-04-24", 0.5
FAMILY, STUDY = "us_earnings_drift_2026-11-18", "us_earnings_drift_2026-11-18"


def ms(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").value // 10**6)


def label(c: dict) -> str:
    return f"{c['signal']}_h{c['hold']}"


def calendar() -> pd.DataFrame:
    parts = [pd.read_csv(DATA / d / "earnings.csv", dtype=str).fillna("") for d in CALENDARS]
    return pd.concat(parts).drop_duplicates(["symbol", "date"]).reset_index(drop=True)


def tickers() -> list[str]:
    underlying = json.loads(TRADFI.read_text())["underlying"]
    return sorted(s[:-4] for s, k in underlying.items() if k == "EQUITY" and s[:-4] not in EXCLUDED
                  and (us.US_SNAPSHOT / f"{s[:-4]}_1d.csv").is_file())


def window_events(window: str) -> tuple[dict, dict, dict]:
    """(ticker -> events whose decision falls in the window, ticker -> frame, counts)."""
    lo, hi = (ms(d) for d in WINDOWS[window])
    cal, qqq = calendar(), us.daily_frame("QQQ")
    found, frames, counts = {}, {}, {"skipped_not_a_trading_day": 0, "excluded_big_move": 0, "excluded_dkng": 0}
    for t in tickers():
        frame = us.daily_frame(t)
        evts, skipped = ue.events(t, cal, frame, qqq)
        counts["skipped_not_a_trading_day"] += skipped
        kept = []
        for e in evts:
            if not lo <= e["decide"] < hi:
                continue
            if abs(e["reaction"]) > MAX_REACTION:
                counts["excluded_big_move"] += 1
                continue
            if t == "DKNG" and e["day"] < DKNG_FROM:
                counts["excluded_dkng"] += 1
                continue
            kept.append(e)
        if kept:
            found[t], frames[t] = kept, frame
    return found, frames, counts


def run_config(c: dict, found: dict, frames: dict, window: str, *, stress: bool = False) -> dict:
    """Every event of the configuration through the shared broker; per-event results and the portfolio's days."""
    lo, hi = (ms(d) for d in WINDOWS[window])
    results, sleeves, overlaps = [], [], 0
    for t, evts in found.items():
        frame = frames[t]
        held, end = [], -1
        for e in sorted(evts, key=lambda e: e["decide"]):
            s = ue.side(e, c["signal"])
            j = frame.index.get_loc(e["decide"])
            if s == 0:
                continue
            if j <= end:
                overlaps += 1
                continue
            held.append({**e, "side": s})
            end = j + c["hold"]
        if not held:
            continue
        measured = abs(us.daily_funding(us.perpetual(t)) or 0.0)
        rate = 0.10 / 365 if stress else measured
        longs, shorts = ue.split(ue.weights(frame.index, held, hold=c["hold"]))
        for sign, w in ((1, longs), (-1, shorts)):
            mine = [e for e in held if e["side"] == sign]
            if not mine:
                continue
            r = us.replay_daily(t, w, frame=frame, starting_balance="1000", start=lo - 10 * DAY,
                                end=hi + 60 * DAY, daily_rate=sign * rate, cost_multiplier=2 if stress else 1)
            for e in mine:
                results.append({"ticker": t, "decide": e["decide"], "side": sign,
                                "net": ue.event_return(r["curve"], decide=e["decide"], hold=c["hold"])})
            daily = r["curve"].pct_change().fillna(0.0)
            sleeves.append(pd.Series(daily.to_numpy(), index=(r["curve"].index // DAY).astype(int)))
    port = ue.portfolio(sleeves) if sleeves else pd.Series(dtype=float)
    days = range(lo // DAY, hi // DAY)
    port = port.groupby(level=0).sum().reindex(days).fillna(0.0)
    return {"events": pd.DataFrame(results, columns=["ticker", "decide", "side", "net"]), "daily": port,
            "overlaps_skipped": overlaps}


def stats(ev: pd.DataFrame) -> dict:
    if ev.empty:
        return {"events": 0, "mean_net": None, "t_week": None}
    weeks = ev.assign(week=[ue.week(int(d)) for d in ev["decide"]]).groupby("week")["net"].mean()
    t = float(weeks.mean() / weeks.std(ddof=1) * math.sqrt(len(weeks))) if len(weeks) > 1 else 0.0
    top = ev.groupby("ticker")["net"].sum().idxmax()
    return {"events": int(len(ev)), "weeks": int(len(weeks)), "mean_net": float(ev["net"].mean()),
            "median_net": float(ev["net"].median()), "hit_rate": float((ev["net"] > 0).mean()), "t_week": t,
            "weekly_mean_sd": float(weeks.std(ddof=1)) if len(weeks) > 1 else None,
            "top_ticker": str(top), "mean_without_top": float(ev[ev["ticker"] != top]["net"].mean()),
            "longs": int((ev["side"] > 0).sum()), "shorts": int((ev["side"] < 0).sum())}


def benchmarks(window: str) -> pd.DataFrame:
    lo, hi = (ms(d) for d in WINDOWS[window])
    out = {}
    for t in ("SPY", "QQQ"):
        close = us.daily_frame(t)["close"]
        r = close.pct_change().dropna()
        r.index = ((r.index + DAY) // DAY).astype(int)
        out[t] = r.reindex(range(lo // DAY, hi // DAY)).fillna(0.0)
    return pd.DataFrame(out)


def commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def explore() -> dict:
    found, frames, counts = window_events("explore")
    rows, sharpes = {}, {}
    for c in CONFIGS:
        base, stress = run_config(c, found, frames, "explore"), run_config(c, found, frames, "explore", stress=True)
        rows[label(c)] = {"base": stats(base["events"]), "stress": stats(stress["events"]),
                          "portfolio": lab.summary(base["daily"]), "overlaps_skipped": base["overlaps_skipped"]}
        sharpes[label(c)] = rows[label(c)]["portfolio"].get("sharpe", 0.0)
    chosen = max(CONFIGS, key=lambda c: rows[label(c)]["base"]["t_week"] or -9.0)
    s = rows[label(chosen)]["base"]
    clusters = int(round(s["weeks"] * 3 / 5))      # the confirmation is 3 years to the exploration's 5
    sd = s["weekly_mean_sd"]
    lit = LITERATURE_EFFECT[chosen["hold"]]
    power = {"confirm_weeks_assumed": clusters, "weekly_mean_sd": sd, "literature_effect": lit,
             "power_at_literature": gates.power(lit, sd, clusters), "measured_effect": s["mean_net"],
             "power_at_measured": gates.power(s["mean_net"], sd, clusters),
             "weeks_needed_at_literature": gates.samples_needed(lit, sd)}
    done = {(r["study"], r["trial"]) for r in registry.load(ROOT / "research/trials.jsonl", FAMILY)}
    registry.record([{"family": FAMILY, "study": STUDY, "trial": label(c), "config": c,
                      "window": list(WINDOWS["explore"]), "sharpe": float(sharpes[label(c)]), "count": 1,
                      "note": "exploration window, base costs"}
                     for c in CONFIGS if (STUDY, label(c)) not in done], path=ROOT / "research/trials.jsonl")
    return {"code_commit": commit(), "counts": counts, "configs": rows, "family_sharpes": sharpes,
            "chosen": label(chosen), "power": power, "confirm_runs": power["power_at_literature"] >= 0.8}


def confirm(explored: dict) -> dict:
    if not explored["confirm_runs"]:
        return {"ran": False, "reason": "power at the literature's effect below 80 %: frozen for forward data"}
    c = next(c for c in CONFIGS if label(c) == explored["chosen"])
    found, frames, counts = window_events("confirm")
    base, stress = run_config(c, found, frames, "confirm"), run_config(c, found, frames, "confirm", stress=True)
    s, st = stats(base["events"]), stats(stress["events"])
    bench = benchmarks("confirm")
    alpha = gates.regression_alpha(base["daily"], bench)
    gain = gates.portfolio_gain(base["daily"], bench["SPY"])
    deflated = gates.family_deflated(base["daily"], list(explored["family_sharpes"].values()))
    checks = {"mean_above_0_and_t_at_least_2": (s["mean_net"] or 0) > 0 and (s["t_week"] or 0) >= 2.0,
              "regression_alpha": alpha["passes"], "portfolio_gain": gain["passes"],
              "family_deflated_at_least_0.9": deflated["passes"], "stress_mean_above_0": (st["mean_net"] or 0) > 0,
              "at_least_200_events_and_not_one_stock": s["events"] >= 200 and s["mean_without_top"] > 0}
    return {"ran": True, "code_commit": commit(), "choice": label(c), "counts": counts, "base": s, "stress": st,
            "portfolio": lab.summary(base["daily"]), "regression_alpha": alpha, "portfolio_gain": gain,
            "family_deflated": deflated, "checks": checks, "passes": all(checks.values())}


def final(confirmed: dict) -> dict:
    if not confirmed.get("passes"):
        return {"opened_holdout": False, "reason": "the confirmation did not pass (or did not run)"}
    c = next(c for c in CONFIGS if label(c) == confirmed["choice"])
    found, frames, _ = window_events("final")
    base = run_config(c, found, frames, "final")
    s = stats(base["events"])
    return {"opened_holdout": True, "choice": label(c), "holdout": s,
            "checks": {"mean_above_0": (s["mean_net"] or 0) > 0, "t_at_least_1": (s["t_week"] or 0) >= 1.0}}


def main(step: str) -> None:
    def write(name: str, data: dict) -> None:
        (HERE / name).write_text(json.dumps(data, indent=1, default=lambda o: float(o) if isinstance(o, np.floating)
                                            else str(o)))
    if step == "explore":
        write("explore.json", explore())
    elif step == "confirm":
        write("confirm.json", confirm(json.loads((HERE / "explore.json").read_text())))
    elif step == "final":
        write("final.json", final(json.loads((HERE / "confirm.json").read_text())))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
