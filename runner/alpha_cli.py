"""Argument parsing kept independent of models, exchange clients and Rust."""
from __future__ import annotations

import argparse
import os


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Framework-native alpha runner")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "BTCUSDT_4h", "ETHUSDT", "ETHUSDT_4h"],
                        help="Runner keys from SYMBOL_CONFIG")
    parser.add_argument("--ws", action="store_true", help="Use WebSocket for live bars")
    parser.add_argument("--check-startup", action="store_true",
                        help="Check native market-to-state wiring offline, then exit; no models or exchange access")
    parser.add_argument("--check-paper-contract", action="store_true",
                        help="Check the data-to-signal paper path with offline fixtures; no account access")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="dry_run", action="store_true", help="Signals only (default)")
    mode.add_argument("--enable-execution", dest="dry_run", action="store_false",
                      help="Explicitly enable exchange writes")
    parser.set_defaults(dry_run=True)
    parser.add_argument("--venue", choices=["bybit", "binance", "okx"], default="bybit")
    args = parser.parse_args(argv)
    if args.check_startup and args.check_paper_contract:
        parser.error("Select only one offline check")
    if (args.check_startup or args.check_paper_contract) and not args.dry_run:
        parser.error("Offline checks cannot be combined with --enable-execution")
    if not args.dry_run and not os.environ.get("RISK_SCOPE", "").strip():
        parser.error("--enable-execution requires RISK_SCOPE identifying venue, account and demo/live environment")
    return args
