#!/usr/bin/env python3
"""Compare a paper run of runner.binance_main with an offline replay.

    python3 -m scripts.compare_paper_replay --events data/runtime/binance/binance_events.jsonl \
        --models-dir models_binance --data-dir data_files/binance_public --symbol BTCUSDT

Rebuilds the same period from the downloaded public data (refresh it first),
replays it with the same frozen model and config, and reports every fill that
differs. Identical inputs must give identical fills; a difference means live
inputs (bars, funding, positioning) diverged from the research data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runner.binance_main import WARMUP_BARS, load_model_dir

HOUR = 3_600_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()

    from data.quality.causal_bars import load_bars
    from runner.offline.broker import ReplayConfig
    from runner.offline.context import DerivativesContext
    from runner.offline.replay import run_replay
    from runner.offline.report import load_funding

    rows = [json.loads(line) for line in args.events.read_text().splitlines() if line.strip()]
    equity = [r for r in rows if r["kind"] == "paper_equity" and r["symbol"] == args.symbol]
    paper_fills = [r for r in rows if r["kind"] == "paper_fill" and r["symbol"] == args.symbol]
    if not equity:
        print("no paper_equity rows for this symbol", file=sys.stderr)
        return 2
    first_live, last_live = equity[0]["bar_open"], equity[-1]["bar_open"]
    bars = load_bars(args.data_dir / f"{args.symbol}_1h.csv", extras=True,
                     start_ms=first_live - WARMUP_BARS * HOUR)
    bars = [b for b in bars if b.open_time <= last_live]
    if bars[-1].open_time != last_live:
        print("downloaded data does not yet cover the paper period; refresh it", file=sys.stderr)
        return 2
    bundle, final_config, _ = load_model_dir(args.models_dir, args.symbol, first_live)
    context = DerivativesContext.from_files(args.data_dir / f"{args.symbol}_funding.csv",
                                            args.data_dir / f"{args.symbol}_metrics.csv")
    result = run_replay(bars, bundle, load_funding(args.data_dir / f"{args.symbol}_funding.csv"),
                        start_ms=first_live, config=ReplayConfig(**final_config), symbol=args.symbol,
                        context=context)

    def key(f):
        return (f["at"], f["side"], f["qty"], f["price"], f["fee"])
    paper, replay = [key(f) for f in paper_fills], [key(f) for f in result["fills"]]
    mismatches = [(i, p, r) for i, (p, r) in enumerate(zip(paper, replay)) if p != r]
    report = {"symbol": args.symbol, "bars": len(equity), "paper_fills": len(paper), "replay_fills": len(replay),
              "first_mismatch": mismatches[0] if mismatches else None,
              "paper_final_equity": equity[-1]["equity"], "replay_final_equity": result["final"]["equity"],
              "identical": not mismatches and len(paper) == len(replay)}
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
