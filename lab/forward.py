"""The forward window (data from 2026-11-13 on) is the final test; it is read only at registered looks.

    python -m lab.forward --study overlay_2026-11-13 --spec research/results/overlay_2026-11-13/final_spec.json \
        --data-dir data_files/binance_public_<date after end> --end 2027-05-13 --out research/results/forward_...

Rules enforced here:
- ``FORWARD_START`` is 2026-11-13 00:00 UTC. Development studies stop at
  ``lab.study.DEV_END`` (2025-11-16); 2025-11-16..2026-11-13 has been viewed
  and is used for nothing.
- A look is a (study, spec SHA-256, end date) triple appended to
  ``research/forward_looks.jsonl`` by a commit made BEFORE the end date. An
  unregistered look, a changed spec or an end date not yet passed is refused.
- Bars from the end date on never enter the evaluation; bars before
  ``FORWARD_START`` are used only as feature warm-up, never as returns.
- Each look writes a new result directory; a look is evaluated once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from lab.data import DAY, funding_series, load_symbol
from lab.metrics import exposure
from lab.overlay_study import SLEEVE_BALANCE, matched_benchmark, portfolio_equity, stats
from lab.simulate import simulate
from lab.strategy import StrategySpec, compute_targets
from lab.study import _manifest

FORWARD_START = int(datetime(2026, 11, 13, tzinfo=timezone.utc).timestamp() * 1000)
LOOKS = Path("research/forward_looks.jsonl")


def spec_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def day_ms(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp() * 1000)


def load_looks(path: Path = LOOKS) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def registered_look(study: str, digest: str, end: str, *, path: Path = LOOKS, now_ms: int | None = None) -> dict:
    """The registered look for this triple; refuses unregistered, premature or late-registered looks."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000) if now_ms is None else now_ms
    matches = [r for r in load_looks(path) if (r["study"], r["spec_sha256"], r["end"]) == (study, digest, end)]
    if not matches:
        raise PermissionError(f"no registered forward look for {study} at {end} with this spec")
    look = matches[0]
    end_ms = day_ms(end)
    if day_ms(look["registered"]) >= end_ms:
        raise PermissionError("the look was registered on or after its end date")
    if end_ms <= FORWARD_START or end_ms % DAY:
        raise ValueError("a look must end on a whole UTC day after FORWARD_START")
    if now_ms < end_ms + DAY:
        raise PermissionError("the look's end date has not fully passed")
    return look


def evaluate_target_spec(spec_path: Path, data_dir: Path, end_ms: int) -> dict:
    """Replay a frozen target-weight spec over [FORWARD_START, end) exactly as its study did."""
    frozen = json.loads(Path(spec_path).read_text())
    if frozen.get("kind") != "btc_eth_multitimeframe_target":
        raise ValueError("only target-weight lab specs are supported")
    spec = StrategySpec.from_record(frozen["spec"])
    symbols = frozen["symbols"]
    bars, funding, frames, fseries = {}, {}, {}, {}
    for symbol in symbols:
        b, f, frame = load_symbol(data_dir, symbol)
        if b[-1].open_time + 3_600_000 < end_ms:
            raise ValueError(f"{symbol}: data ends before the look's end date")
        bars[symbol] = [x for x in b if x.open_time < end_ms]
        funding[symbol] = {t: v for t, v in f.items() if t < end_ms}
        frames[symbol] = frame[frame["open_time"] < end_ms].reset_index(drop=True)
        fseries[symbol] = funding_series(funding[symbol])
    sleeves = [simulate(s, bars[s], compute_targets(frames[s], fseries[s], spec), funding[s], spec,
                        start_ms=FORWARD_START, end_ms=end_ms, starting_balance=SLEEVE_BALANCE) for s in symbols]
    initial = float(len(symbols)) * float(SLEEVE_BALANCE)
    equity = portfolio_equity(sleeves, FORWARD_START // DAY, initial)
    level = exposure([s.curve for s in sleeves])
    days = len(equity) - 1
    return {
        "window": {"start_ms": FORWARD_START, "end_ms": end_ms, "days": days},
        "strategy": stats(equity), "average_exposure": level,
        "matched_benchmark": stats(matched_benchmark(frames, equity.index, level, initial)),
        "benchmark_50_50_full_exposure": stats(matched_benchmark(frames, equity.index, 1.0, initial)),
        "fills_per_symbol_year": {s.symbol: len(s.fills) / max(days, 1) * 365.0 for s in sleeves},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--end", required=True, help="YYYY-MM-DD, exclusive")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("use an empty output directory; each look is evaluated once")
    digest = spec_digest(args.spec)
    look = registered_look(args.study, digest, args.end)
    result = evaluate_target_spec(args.spec, args.data_dir, day_ms(args.end))
    result.update(look=look, spec_sha256=digest, data=_manifest(args.data_dir))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps({k: result[k] for k in ("window", "strategy", "matched_benchmark")}, indent=2, default=float))


if __name__ == "__main__":
    main()


__all__ = ["FORWARD_START", "LOOKS", "registered_look", "evaluate_target_spec", "spec_digest"]
