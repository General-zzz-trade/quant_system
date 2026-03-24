# runner/backtest_cli.py
"""CLI entry point and argument parsing for backtest_runner.

Extracted from backtest_runner.py to reduce file size.
"""
from __future__ import annotations

import argparse
import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pick_default_csv(root: Path) -> Optional[Path]:
    fixed = root / "data" / "binance" / "ohlcv" / "BTCUSDT_1m_ohlcv.csv"
    if fixed.exists() and fixed.stat().st_size > 0:
        return fixed

    ohlcv_dir = root / "data" / "binance" / "ohlcv"
    if not ohlcv_dir.exists():
        return None

    candidates = [p for p in ohlcv_dir.glob("*_ohlcv.csv") if p.is_file() and p.stat().st_size > 0]
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _infer_symbol_from_csv_name(csv_path: Path) -> Optional[str]:
    name = csv_path.stem
    if "_1m_" in name:
        return name.split("_")[0]
    if "-1m-" in name:
        return name.split("-")[0]
    if "_" in name:
        return name.split("_")[0]
    return None


def _default_out_dir(root: Path, symbol: str) -> Path:
    return root / "out" / f"{symbol.lower()}_default"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")

    # Default backtest command (also works without subcommand)
    p.add_argument("--csv", default=None, help="Path to OHLCV CSV")
    p.add_argument("--symbol", default=None, help="Symbol, e.g. BTCUSDT")
    p.add_argument("--starting-balance", default="10000", help="Starting balance")
    p.add_argument("--ma", type=int, default=20, help="Moving average window")
    p.add_argument("--qty", default="0.01", help="Order quantity")
    p.add_argument("--fee-bps", default="0", help="Fee bps per fill (e.g. 4 = 0.04%%)")
    p.add_argument("--slippage-bps", default="0", help="Slippage bps per fill (e.g. 2 = 0.02%%)")
    p.add_argument("--out", default=None, help="Output directory for csv logs")
    p.add_argument("--multi-csv", default=None, help="Multi-asset: BTCUSDT:path1,ETHUSDT:path2")

    # Replay subcommand
    replay_p = sub.add_parser("replay", help="Replay events from SQLite event log")
    replay_p.add_argument("--event-log", required=True, help="Path to SQLite event log")
    replay_p.add_argument("--symbol", default=None, help="Symbol filter")
    replay_p.add_argument("--out", default=None, help="Output directory")

    return p


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    root = _project_root()

    if args.command == "replay":
        return args

    if args.csv is None and args.multi_csv is None:
        picked = _pick_default_csv(root)
        if picked is None:
            print("Missing --csv and no default CSV found.")
            print(f"Expected: {root / 'data' / 'binance' / 'ohlcv' / 'BTCUSDT_1m_ohlcv.csv'}")
            print("Or put any *_ohlcv.csv under: data/binance/ohlcv/")
            raise SystemExit(2)
        args.csv = str(picked)

    if args.multi_csv is not None:
        return args

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = (root / csv_path).resolve()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        raise SystemExit(2)
    args.csv = str(csv_path)

    if args.symbol is None:
        args.symbol = _infer_symbol_from_csv_name(csv_path) or "BTCUSDT"

    if args.out is None or str(args.out).strip() == "":
        args.out = str(_default_out_dir(root, args.symbol))
    else:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            args.out = str((root / out_path).resolve())

    return args


def main() -> None:
    from runner.backtest_runner import run_backtest, run_multi_backtest, _max_drawdown

    args = parse_args()

    if getattr(args, "command", None) == "replay":
        from runner.replay_runner import run_replay
        run_replay(
            event_log_path=Path(args.event_log),
            symbol=args.symbol,
            out_dir=Path(args.out) if args.out else None,
        )
        return

    if getattr(args, "multi_csv", None) is not None:
        csv_paths: Dict[str, Path] = {}
        root = _project_root()
        for pair in args.multi_csv.split(","):
            sym, path_str = pair.split(":", 1)
            p = Path(path_str)
            if not p.is_absolute():
                p = (root / p).resolve()
            csv_paths[sym.upper()] = p

        out_dir = Path(args.out) if args.out else None
        eq, _ = run_multi_backtest(
            csv_paths=csv_paths,
            starting_balance=Decimal(str(args.starting_balance)),
            fee_bps=Decimal(str(args.fee_bps)),
            slippage_bps=Decimal(str(args.slippage_bps)),
            out_dir=out_dir,
        )
        if eq:
            start = eq[0].equity
            end = eq[-1].equity
            ret = (end - start) / start if start != 0 else Decimal("0")
            mdd = _max_drawdown([x.equity for x in eq])
            print(f"symbols={','.join(csv_paths.keys())}")
            print(f"bars={len(eq)}")
            print(f"start_equity={start}")
            print(f"end_equity={end}")
            print(f"return={ret}")
            print(f"max_drawdown={mdd}")
        return

    csv_path = Path(args.csv)
    out_dir = Path(args.out) if args.out else None

    eq, _ = run_backtest(
        csv_path=csv_path,
        symbol=args.symbol,
        starting_balance=Decimal(str(args.starting_balance)),
        ma_window=int(args.ma),
        order_qty=Decimal(str(args.qty)),
        fee_bps=Decimal(str(args.fee_bps)),
        slippage_bps=Decimal(str(args.slippage_bps)),
        out_dir=out_dir,
    )

    if not eq:
        print("No equity points produced. Check CSV columns and data.")
        return

    start = eq[0].equity
    end = eq[-1].equity
    ret = (end - start) / start if start != 0 else Decimal("0")
    mdd = _max_drawdown([x.equity for x in eq])

    summary_path = (out_dir / "summary.json") if out_dir else None
    summary = None
    if summary_path and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = None

    print(f"csv={csv_path}")
    print(f"symbol={args.symbol}")
    print(f"out={out_dir}")
    print(f"bars={len(eq)}")
    print(f"start_equity={start}")
    print(f"end_equity={end}")
    print(f"return={ret}")
    print(f"max_drawdown={mdd}")

    if isinstance(summary, dict):
        print(f"trades={summary.get('trades')}")
        print(f"trades_per_day={summary.get('trades_per_day')}")
        print(f"win_rate={summary.get('win_rate')}")
        print(f"profit_factor={summary.get('profit_factor')}")
        print(f"avg_trade_pnl={summary.get('avg_trade_pnl')}")
        print(f"median_trade_pnl={summary.get('median_trade_pnl')}")
        print(f"max_consecutive_losses={summary.get('max_consecutive_losses')}")
        print(f"total_fees={summary.get('total_fees')}")
        print(f"avg_duration_sec={summary.get('avg_duration_sec')}")
        print(f"p95_duration_sec={summary.get('p95_duration_sec')}")



if __name__ == "__main__":
    main()
