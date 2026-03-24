# runner/live_runner_cli.py
"""CLI entry point for LiveRunner (standalone execution).

Extracted from live_runner.py to reduce file size.
"""
from __future__ import annotations

import gc
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for live runner CLI."""
    gc.set_threshold(50_000, 50, 10)

    # Pin to isolated CPU1 + mlock all memory
    try:
        os.sched_setaffinity(0, {1})
    except OSError as e:
        logger.debug("Could not pin to CPU1: %s", e)
    try:
        import ctypes
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _libc.mlockall(3)  # MCL_CURRENT | MCL_FUTURE
    except OSError as e:
        logger.debug("Could not lock memory pages: %s", e)

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Live trading runner")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML path")
    parser.add_argument("--shadow", action="store_true", help="Shadow mode -- simulate orders")
    args = parser.parse_args()

    # Venue clients must be constructed from config credentials
    from infra.config.loader import load_config_secure, resolve_credentials

    raw = load_config_secure(args.config)
    creds = resolve_credentials(raw)

    venue_clients: Dict[str, Any] = {}
    exchange = raw.get("venue", raw.get("trading", {}).get("exchange", "binance"))
    testnet = bool(raw.get("testnet", raw.get("trading", {}).get("testnet", False)))

    if exchange == "binance":
        from execution.adapters.binance.rest import BinanceRestClient, BinanceRestConfig
        from execution.adapters.binance.urls import resolve_binance_urls

        binance_urls = resolve_binance_urls(testnet)
        client = BinanceRestClient(
            cfg=BinanceRestConfig(
                base_url=binance_urls.rest_base,
                api_key=creds.get("api_key", ""),
                api_secret=creds.get("api_secret", ""),
            )
        )
        venue_clients["binance"] = client

    from runner.live_runner import LiveRunner

    runner = LiveRunner.from_config(
        args.config,
        venue_clients=venue_clients,
        shadow_mode=getattr(args, 'shadow', False),
    )
    runner.start()


if __name__ == "__main__":
    main()
