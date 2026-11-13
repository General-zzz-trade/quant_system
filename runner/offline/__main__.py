"""python -m runner.offline --help"""
import argparse
import json
import logging
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


@contextmanager
def no_network():
    """Fail immediately if a replay dependency attempts a socket connection."""
    old_connect, old_connect_ex = socket.socket.connect, socket.socket.connect_ex
    old_create = socket.create_connection
    def blocked(*args, **kwargs):
        raise RuntimeError("network is forbidden in offline validation")
    socket.socket.connect = socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex = old_connect, old_connect_ex
        socket.create_connection = old_create


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline 1h+4h validation; no exchange keys or network")
    sub = parser.add_subparsers(dest="command", required=True)
    research = sub.add_parser("research", help="Predeclared candidate selection + one frozen final test")
    research.add_argument("--out", required=True, type=Path)
    research.add_argument("--config", type=Path)
    research.add_argument("--data-dir", required=True, type=Path,
                          help="data.downloads.binance_public output (<SYM>_1h.csv, _funding.csv, _metrics.csv)")
    research.add_argument("--symbol", required=True)
    research.add_argument("--start", help="First bar date (UTC, YYYY-MM-DD); default: first metrics day")
    research.add_argument("--validation-only", action="store_true",
                          help="Stop after candidate selection: no final model and no final-test replay")
    for name in ("demo", "study", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--config", type=Path)
        if name != "demo":
            p.add_argument("--bars", type=Path, required=True)
            p.add_argument("--funding", type=Path, required=True)
        if name == "replay":
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--start-ms", type=int, required=True)
            p.add_argument("--checkpoint", type=Path)
            p.add_argument("--stop-after", type=int)
            p.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)
    with no_network():
        from data.quality.causal_bars import load_bars
        from .broker import ReplayConfig
        from .fixtures import synthetic_bars, synthetic_funding
        from .report import checkpoint_replay, load_funding, metrics, write_result
        from .replay import run_replay
        from .study import run_study
        config = ReplayConfig(**json.loads(args.config.read_text())) if args.config else ReplayConfig()
        if args.command == "research":
            from .context import DerivativesContext
            from .research import run_research
            context = DerivativesContext.from_files(args.data_dir / f"{args.symbol}_funding.csv",
                                                    args.data_dir / f"{args.symbol}_metrics.csv")
            if args.start:
                start_ms = int(datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp() * 1000)
            else:
                first = context.metrics_start_ms
                start_ms = None if first is None else (first // 86_400_000 + 1) * 86_400_000
            bars = load_bars(args.data_dir / f"{args.symbol}_1h.csv", start_ms=start_ms, extras=True,
                             as_of_ms=int(datetime.now(timezone.utc).timestamp() * 1000))
            funding = load_funding(args.data_dir / f"{args.symbol}_funding.csv")
            summary = run_research(bars, funding, args.out, symbol=args.symbol, context=context, config=config,
                                   validation_only=args.validation_only)
            print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
            return
        if args.command == "demo":
            bars = synthetic_bars()
            summary = run_study(bars, synthetic_funding(bars), args.out, synthetic=True, config=config)
        else:
            bars = load_bars(args.bars, as_of_ms=int(datetime.now(timezone.utc).timestamp() * 1000))
            funding = load_funding(args.funding)
            if args.command == "study":
                summary = run_study(bars, funding, args.out, config=config)
            else:
                if args.out.exists() and any(args.out.iterdir()):
                    parser.error("use an empty --out directory")
                bundle = json.loads(args.model.read_text())
                if args.resume and not args.checkpoint:
                    parser.error("--resume requires --checkpoint")
                if args.checkpoint:
                    result = checkpoint_replay(bars, bundle, funding, start_ms=args.start_ms, config=config,
                        checkpoint=args.checkpoint, stop_after=args.stop_after, resume=args.resume)
                else:
                    result = run_replay(bars, bundle, funding, start_ms=args.start_ms, config=config,
                                        stop_after=args.stop_after)
                write_result(args.out, result)
                summary = metrics(result)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
