#!/usr/bin/env python3
"""Start live trading — transparent assembly, every step visible.

Replaces the 2,017-line LiveRunner.build() with explicit module wiring.
Each step creates one component, visible and testable.

Usage:
    python3 -m runner.run_trading --config config/production.yaml
    python3 -m runner.run_trading --symbols BTCUSDT --testnet
    python3 -m runner.run_trading --symbols BTCUSDT --testnet --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from runner.trading_config import TradingConfig
from runner.trading_engine import TradingEngine
from runner.risk_manager import RiskManager
from runner.order_manager import OrderManager
from runner.binance_executor import BinanceExecutor
from runner.recovery_manager import RecoveryManager
from runner.lifecycle_manager import LifecycleManager
from runner.runner_loop import RunnerLoop

logger = logging.getLogger(__name__)


def build_runner(config: TradingConfig, dry_run: bool = False):
    """Assemble all modules. Returns LifecycleManager."""

    # 1. Feature + ML engine
    # In production, these are RustFeatureEngine + RustInferenceBridge.
    # Here we use duck-typed objects so the assembly is testable without Rust.
    from engine.feature_hook import FeatureComputeHook

    feature_hook = FeatureComputeHook(symbols=list(config.symbols))
    inference_bridge = _load_inference_bridge(config)

    engine = TradingEngine(
        feature_hook=feature_hook,
        inference_bridge=inference_bridge,
        symbols=list(config.symbols),
        model_dir=config.model_dir,
    )

    # 2. Risk
    from risk.kill_switch import KillSwitch

    kill_switch = KillSwitch()
    risk = RiskManager(
        kill_switch=kill_switch,
        max_position=config.max_concentration,
        max_notional=config.max_gross_leverage * 10_000,
        max_open_orders=5,
    )

    # 3. Orders
    orders = OrderManager(timeout_sec=config.pending_order_timeout_sec)

    # 4. Execution
    venue_client = _create_venue_client(config) if not dry_run else None
    executor = BinanceExecutor(
        venue_client=venue_client,
        kill_switch=kill_switch,
        use_ws=config.use_ws_orders,
        shadow_mode=config.shadow_mode or dry_run,
    )

    # 5. Recovery
    recovery = RecoveryManager(
        state_dir=config.data_dir,
        engine=engine,
        risk=risk,
        orders=orders,
        interval_sec=config.checkpoint_interval_sec,
    )

    # 6. Event loop
    loop = RunnerLoop(engine, risk, orders, executor)

    # 7. Lifecycle (start/stop sequencing + signals)
    lifecycle = LifecycleManager(
        engine=engine,
        executor=executor,
        recovery=recovery,
        loop=loop,
    )

    return lifecycle


def _load_inference_bridge(config: TradingConfig):
    """Load inference bridge from model_dir. Returns duck-typed object."""
    try:
        from alpha.inference_bridge import LiveInferenceBridge

        bridge = LiveInferenceBridge(model_dir=config.model_dir)
        logger.info("Inference bridge loaded from %s", config.model_dir)
        return bridge
    except Exception as e:
        logger.warning("Could not load inference bridge: %s (using stub)", e)

        class _StubBridge:
            def predict(self, symbol, features):
                return 0.0

            def get_params(self):
                return {}

            def reload_model(self, symbol, path):
                pass

        return _StubBridge()


def _create_venue_client(config: TradingConfig):
    """Create Binance venue client from config."""
    import os

    if config.testnet:
        from execution.adapters.binance.venue_client_um import BinanceVenueClientUM

        return BinanceVenueClientUM(
            api_key=os.environ.get("BINANCE_TESTNET_API_KEY", ""),
            api_secret=os.environ.get("BINANCE_TESTNET_API_SECRET", ""),
            testnet=True,
        )
    raise NotImplementedError("Production venue client requires explicit configuration")


def main():
    parser = argparse.ArgumentParser(description="Decomposed live trader")
    parser.add_argument("--config", help="YAML config path")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    parser.add_argument("--testnet", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true", help="Assemble but don't connect")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.config:
        # TODO: YAML loading
        config = TradingConfig(symbols=tuple(args.symbols), testnet=args.testnet)
    else:
        config = TradingConfig(symbols=tuple(args.symbols), testnet=args.testnet)

    logger.info("Config: %d fields, symbols=%s, testnet=%s, shadow=%s",
                len(config.__dataclass_fields__), config.symbols,
                config.testnet, config.shadow_mode)

    lifecycle = build_runner(config, dry_run=args.dry_run)

    if args.dry_run:
        logger.info("Dry run: assembly OK, all modules constructed. Exiting.")
        return

    lifecycle.start()


if __name__ == "__main__":
    main()
