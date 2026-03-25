#!/usr/bin/env python3
"""Entry script for Polymarket 5m BTC Up/Down market maker.

Usage:
    python3 scripts/run_polymarket_maker.py [--config config/polymarket.yaml] [--once]
    python3 scripts/run_polymarket_maker.py --gamma 0.15 --order-size 20

Environment:
    POLYMARKET_API_KEY      API key for Polymarket CLOB
    POLYMARKET_API_SECRET   API secret for Polymarket CLOB
    POLYMARKET_BASE_URL     Base URL (default: https://clob.polymarket.com)
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polymarket 5m BTC Up/Down market maker"
    )
    parser.add_argument("--config", default="config/polymarket.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--once", action="store_true",
                        help="Run single cycle then exit")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="A-S risk aversion (default: 0.1)")
    parser.add_argument("--kappa", type=float, default=1.5,
                        help="A-S order arrival intensity (default: 1.5)")
    parser.add_argument("--order-size", type=float, default=10.0,
                        help="Order size per side (default: 10)")
    parser.add_argument("--max-inventory", type=float, default=100.0,
                        help="Max net inventory (default: 100)")
    parser.add_argument("--refresh", type=float, default=30.0,
                        help="Quote refresh interval in seconds (default: 30)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from polymarket.config import PolymarketConfig
    from polymarket.runner import PolymarketMakerRunner

    # Load config from YAML if it exists, override with env vars
    if os.path.exists(args.config):
        config = PolymarketConfig.from_yaml(args.config)
    else:
        config = PolymarketConfig(
            api_key=os.environ.get("POLYMARKET_API_KEY", ""),
            api_secret=os.environ.get("POLYMARKET_API_SECRET", ""),
            base_url=os.environ.get("POLYMARKET_BASE_URL",
                                    "https://clob.polymarket.com"),
        )

    runner = PolymarketMakerRunner(
        config,
        gamma=args.gamma,
        kappa=args.kappa,
        order_size=args.order_size,
        max_inventory=args.max_inventory,
        refresh_interval=args.refresh,
    )

    # Graceful shutdown on SIGINT/SIGTERM
    def _shutdown(signum: int, frame: object) -> None:
        logging.getLogger(__name__).info("Received signal %d, shutting down...", signum)
        runner.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if args.once:
        summary = runner.run_once()
        print(f"Cycle result: {summary}")
    else:
        runner.start()


if __name__ == "__main__":
    main()
