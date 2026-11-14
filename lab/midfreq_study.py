"""Mid-frequency feasibility study: seven predeclared candidates, one gate, development data only.

    python -m lab.midfreq_study --data-dir data_files/binance_midfreq_2025-11-16 \
        --out research/results/midfreq_2026-11-14

Fixed before any real-data run (research/protocols/2026-11-14-midfreq-feasibility.md):

- **Universe:** the 15 USDT perpetuals with the highest quote volume in 2022-Q4 (delisted ones included).
- **Candidates (7, all counted as trials):** cross-sectional reversal at 15 minutes and 1 hour, 24-hour
  momentum held 4 hours, 1-hour taker-flow imbalance, and a pooled Ridge at 15 minutes, 1 hour and 4 hours.
  Each is long the top 3 and short the bottom 3 at 1x gross, decided on its horizon clock and filled at the
  next 5-minute open.
- **Costs:** taker (5 bp fee + 2 bp slippage per side) decides; taker x2 decides; maker (2 bp, no slippage,
  every order filled) is an optimistic bound and is only reported.
- **Capital:** 200 USDT decides; 10,000 USDT is reported.
- **Selection:** the highest daily Sharpe at 200 USDT and taker costs.
- **Gate:** ``GATE``. The conclusion is ``feasible``, ``maker_only`` or ``not_feasible``.

The study runs once: it refuses a non-empty output directory and a study id already in the registry.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lab import trials
from lab.midfreq import DAY, decision_rows, features, load_panel, ridge_scores, rule_scores, simulate
from lab.overlay_study import stats
from lab.pair_study import _code_identity
from lab.study import DEV_END, _manifest

STUDY = "midfreq_2026-11-14"
FAMILY = "midfreq_usdm"
START = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
BLOCK = 91 * DAY
CAPITAL, REPORT_CAPITAL = 200.0, 10_000.0
CANDIDATES = [{"id": name, "kind": kind, "horizon_bars": bars} for name, kind, bars in (
    ("xs_rev_15m", "rule", 3), ("xs_rev_1h", "rule", 12), ("xs_mom_4h", "rule", 48), ("flow_1h", "rule", 12),
    ("ridge_15m", "ridge", 3), ("ridge_1h", "ridge", 12), ("ridge_4h", "ridge", 48))]
COSTS = {"taker": (5e-4, 2e-4), "taker_x2": (1e-3, 4e-4), "maker": (2e-4, 0.0), "zero": (0.0, 0.0)}
GATE = {"taker_total_return_above": 0.0, "deflated_sharpe_probability_min": 0.95, "taker_x2_total_return_above": 0.0,
        "max_drawdown_daily_max": 0.30, "wipeout_allowed": False}
REFUSED = "refused: a feasibility result is not a tradable strategy"


def describe(run) -> dict:
    out = stats(run.equity)
    out.update(turnover_per_year=run.turnover, skipped_legs=run.skipped_legs, opened_legs=run.opened_legs,
               wiped=run.wiped, average_legs=run.average_legs)
    out["_daily_returns"] = run.equity.pct_change().dropna()
    return out


def public(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _above(value: float, reference: float) -> dict:
    return {"value": value, "reference": reference, "ok": bool(math.isfinite(value) and value > reference)}


def gate_checks(taker: dict, taker_x2: dict, deflated: dict) -> dict:
    wiped = taker["wiped"] or taker_x2["wiped"]
    probability = deflated["probability"]
    return {
        "taker_total_return": _above(taker["total_return"], GATE["taker_total_return_above"]),
        "deflated_sharpe_probability": {"value": probability, "reference": GATE["deflated_sharpe_probability_min"],
                                        "ok": bool(math.isfinite(probability)
                                                   and probability >= GATE["deflated_sharpe_probability_min"])},
        "taker_x2_total_return": _above(taker_x2["total_return"], GATE["taker_x2_total_return_above"]),
        "max_drawdown_daily": {"value": taker["max_drawdown"], "reference": GATE["max_drawdown_daily_max"],
                               "ok": bool(taker["max_drawdown"] <= GATE["max_drawdown_daily_max"])},
        "no_wipeout": {"value": wiped, "reference": False, "ok": not wiped},
    }


def conclusion(checks: dict, maker: dict) -> str:
    """feasible: the gate passed. maker_only: it failed, but the optimistic maker bound makes money within the
    drawdown limit. not_feasible: neither."""
    if all(c["ok"] for c in checks.values()):
        return "feasible"
    maker_ok = (math.isfinite(maker["total_return"]) and maker["total_return"] > 0 and not maker["wiped"]
                and maker["max_drawdown"] <= GATE["max_drawdown_daily_max"])
    return "maker_only" if maker_ok else "not_feasible"


def run_study(data_dir: Path, out: Path, *, start: int = START, limit: int = DEV_END,
              registry: Path = trials.REGISTRY, study: str = STUDY, block: int = BLOCK) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("use an empty output directory; preserve previous studies")
    if any(r["study"] == study for r in trials.load(registry)):
        raise ValueError(f"{study} already recorded its trials; a study runs once")
    if start % DAY or limit % DAY or not start < limit:
        raise ValueError("window must be whole UTC days")
    panel = load_panel(data_dir, end_ms=limit)
    feats = features(panel)
    results = {}
    for cand in CANDIDATES:
        rows = decision_rows(panel, cand["horizon_bars"], start=start, end=limit)
        blocks: list[dict] = []
        if cand["kind"] == "ridge":
            scores, blocks = ridge_scores(panel, feats, rows, cand["horizon_bars"], first_test=start, block=block,
                                          limit=limit)
        else:
            scores = rule_scores(feats, rows, cand["id"])
        runs = {name: describe(simulate(panel, rows, scores, capital=CAPITAL, fee=fee, slippage=slip))
                for name, (fee, slip) in COSTS.items()}
        runs["taker_10k"] = describe(simulate(panel, rows, scores, capital=REPORT_CAPITAL, fee=COSTS["taker"][0],
                                              slippage=COSTS["taker"][1]))
        results[cand["id"]] = {"spec": cand, "runs": runs, "blocks": blocks, "decisions": int(len(rows))}
    selected = max(results, key=lambda k: (np.nan_to_num(results[k]["runs"]["taker"]["sharpe"], nan=-np.inf), k))
    chosen = results[selected]["runs"]
    rows_out = [{"family": FAMILY, "study": study, "trial": cid, "config": r["spec"], "window": [start, limit],
                 "sharpe": float(r["runs"]["taker"]["sharpe"]) if math.isfinite(r["runs"]["taker"]["sharpe"]) else None,
                 "count": 1, "note": "daily Sharpe, 200 USDT, taker costs, 15 symbols, long 3 / short 3"}
                for cid, r in results.items()]
    deflated = trials.deflated(chosen["taker"]["_daily_returns"], FAMILY, path=registry, extra=rows_out)
    checks = gate_checks(chosen["taker"], chosen["taker_x2"], deflated)
    verdict = conclusion(checks, chosen["maker"])
    summary = {
        "window": {"start_ms": start, "limit_ms": limit}, "symbols": panel.symbols, "selected": selected,
        "conclusion": verdict, "gate": {"passed": verdict == "feasible", "checks": checks},
        "deflated_sharpe": deflated,
        "candidates": {cid: {"decisions": r["decisions"], "runs": {n: public(v) for n, v in r["runs"].items()},
                             "blocks": r["blocks"],
                             "mean_ic": float(np.nanmean([b["ic"] for b in r["blocks"]])) if r["blocks"] else None}
                       for cid, r in results.items()},
    }
    frozen = {"schema": 1, "kind": "midfreq_feasibility", "study": study, "symbols": panel.symbols,
              "selected": results[selected]["spec"], "candidates": CANDIDATES, "costs": COSTS, "gate": GATE,
              "gate_result": checks, "conclusion": verdict, "passed": False, "mainnet": REFUSED,
              "code": _code_identity(), "data": _manifest(data_dir)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    (out / "final_spec.json").write_text(json.dumps(frozen, indent=2, default=float) + "\n")
    trials.record(rows_out, path=registry)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_study(args.data_dir, args.out)
    print(json.dumps({"selected": summary["selected"], "conclusion": summary["conclusion"],
                      "gate": summary["gate"]}, indent=2, default=float))


if __name__ == "__main__":
    main()


__all__ = ["CANDIDATES", "COSTS", "GATE", "run_study", "gate_checks", "conclusion"]
