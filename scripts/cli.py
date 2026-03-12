# tools/cli.py
"""CLI entry point for quant system tools."""
from __future__ import annotations

import argparse
import sys

from scripts.catalog import render_catalog


def main() -> None:
    """主命令行入口。"""
    parser = argparse.ArgumentParser(
        prog="quant",
        description="Quant Trading System CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # backtest 子命令
    bt = sub.add_parser("backtest", help="Run backtest")
    bt.add_argument("--config", type=str, help="Config file path")
    bt.add_argument("--data", type=str, help="Data directory")
    bt.add_argument("--symbol", type=str, default="BTCUSDT")

    # sync 子命令
    sync = sub.add_parser("sync", help="Sync market data")
    sync.add_argument("--symbol", type=str, default="BTCUSDT")
    sync.add_argument("--interval", type=str, default="1h")

    catalog = sub.add_parser("catalog", help="Show curated scripts catalog")
    catalog.add_argument(
        "--scripts",
        action="store_true",
        help="List maintained scripts groups and primary entrypoints",
    )

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "backtest":
        from runner.backtest_runner import run_backtest
        run_backtest()
    elif args.command == "sync":
        print(f"Syncing {args.symbol} {args.interval}...")
    elif args.command == "catalog":
        print(render_catalog())


if __name__ == "__main__":
    main()
