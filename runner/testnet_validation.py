# runner/testnet_validation.py
"""3-phase testnet validation workflow: paper → shadow → live → compare.

Usage:
    python -m runner.testnet_validation --config testnet_binance.yaml --phase paper --duration 300
    python -m runner.testnet_validation --config testnet_binance.yaml --phase shadow --duration 300
    python -m runner.testnet_validation --config testnet_binance.yaml --phase live --duration 300
    python -m runner.testnet_validation --config testnet_binance.yaml --phase compare
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _build_ml_stack(raw: Dict[str, Any]) -> Tuple[Optional[Any], List[Any], List[Any]]:
    """Build ML pipeline (feature_computer, alpha_models, decision_modules) from config.

    Returns (None, [], []) if no strategy.model_path configured or no models found.
    """
    strategy = raw.get("strategy", {})
    model_path = strategy.get("model_path")
    if not model_path:
        logger.info("No strategy.model_path — running without ML stack")
        return None, [], []

    from alpha.models.lgbm_alpha import LGBMAlphaModel
    from decision.ml_decision import MLDecisionModule
    from features.enriched_computer import EnrichedFeatureComputer

    model_dir = Path(model_path)
    config_name = strategy.get("config_name", "mod_reg_1h")
    symbols = raw.get("trading", {}).get("symbols", ["BTCUSDT"])
    threshold = strategy.get("threshold", 0.002)
    threshold_short = strategy.get("threshold_short", 999.0)
    risk_pct = strategy.get("risk_pct", 0.30)

    models: List[Any] = []
    for sym in symbols:
        pkl = model_dir / sym / f"{config_name}.pkl"
        if pkl.exists():
            m = LGBMAlphaModel(name=f"{config_name}_{sym}")
            m.load(pkl)
            models.append(m)
            logger.info("Loaded model: %s", pkl)
        else:
            logger.warning("Model not found: %s", pkl)

    if not models:
        logger.warning("No models loaded — running without ML stack")
        return None, [], []

    fc = EnrichedFeatureComputer()

    dms = [
        MLDecisionModule(
            symbol=sym,
            threshold=threshold,
            threshold_short=threshold_short,
            risk_pct=risk_pct,
        )
        for sym in symbols
    ]

    logger.info(
        "ML stack ready: %d models, %d decision modules, threshold=%.4f",
        len(models), len(dms), threshold,
    )
    return fc, models, dms


def _ensure_testnet(raw: Dict[str, Any]) -> None:
    """Safety check: refuse to run validation against production."""
    testnet = raw.get("trading", {}).get("testnet", False)
    if not testnet:
        print("SAFETY: config must have trading.testnet: true for validation.")
        print("Refusing to run validation against production endpoints.")
        sys.exit(1)


def _output_dir(config_path: Path) -> Path:
    d = config_path.parent / "validation_output"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_equity_csv(path: Path, fills: List[Dict[str, Any]], starting_balance: float) -> None:
    """Write a minimal equity CSV from fill records."""
    equity = Decimal(str(starting_balance))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", "equity", "realized", "unrealized"])
        writer.writeheader()
        writer.writerow({
            "ts": datetime.now(timezone.utc).isoformat(),
            "equity": str(equity),
            "realized": "0",
            "unrealized": "0",
        })
        for fill in fills:
            writer.writerow({
                "ts": fill.get("ts", datetime.now(timezone.utc).isoformat()),
                "equity": str(equity),
                "realized": "0",
                "unrealized": "0",
            })


def run_paper(config_path: Path, duration: int) -> None:
    """Phase 1: Paper trading with testnet market data."""
    from infra.config.loader import load_config_secure
    from runner.live_paper_runner import LivePaperRunner, LivePaperConfig

    raw = load_config_secure(config_path)
    _ensure_testnet(raw)
    # Paper phase uses WS market data only — no API keys needed

    trading = raw.get("trading", {})
    symbol = trading.get("symbol", "BTCUSDT")
    symbols = tuple(trading["symbols"]) if "symbols" in trading else (symbol,)

    config = LivePaperConfig(
        symbols=symbols,
        starting_balance=10000.0,
        testnet=True,
    )

    fc, models, dms = _build_ml_stack(raw)
    runner = LivePaperRunner.build(
        config,
        feature_computer=fc,
        alpha_models=models or None,
        decision_modules=dms or None,
    )

    def _timeout(*_: Any) -> None:
        logger.info("Paper phase duration reached (%ds), stopping...", duration)
        runner.stop()

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(duration)

    logger.info("Starting PAPER phase for %ds with testnet data...", duration)
    try:
        runner.start()
    except KeyboardInterrupt:
        runner.stop()

    out = _output_dir(config_path)
    _write_equity_csv(out / "paper_equity.csv", runner.fills, 10000.0)
    logger.info("Paper phase complete. Fills: %d. Output: %s", len(runner.fills), out)


def run_shadow(config_path: Path, duration: int) -> None:
    """Phase 2: Shadow mode — signals recorded, no execution."""
    from infra.config.loader import load_config_secure
    from runner.live_runner import LiveRunner, LiveRunnerConfig

    raw = load_config_secure(config_path)
    _ensure_testnet(raw)
    # Shadow phase records signals only — no API keys needed

    trading = raw.get("trading", {})
    symbol = trading.get("symbol", "BTCUSDT")
    symbols = tuple(trading["symbols"]) if "symbols" in trading else (symbol,)

    config = LiveRunnerConfig(
        symbols=symbols,
        testnet=True,
        shadow_mode=True,
        enable_preflight=False,
        enable_persistent_stores=False,
    )

    # Shadow mode needs a venue client that won't be called
    class _NoOpClient:
        def send_order(self, order_event: Any) -> list:
            return []

    fc, models, dms = _build_ml_stack(raw)
    runner = LiveRunner.build(
        config,
        venue_clients={"binance": _NoOpClient()},
        feature_computer=fc,
        alpha_models=models or None,
        decision_modules=dms or None,
    )

    def _timeout(*_: Any) -> None:
        logger.info("Shadow phase duration reached (%ds), stopping...", duration)
        runner.stop()

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(duration)

    logger.info("Starting SHADOW phase for %ds with testnet data...", duration)
    try:
        runner.start()
    except KeyboardInterrupt:
        runner.stop()

    out = _output_dir(config_path)
    with (out / "shadow_events.json").open("w") as f:
        json.dump({"fills": runner.fills, "event_index": runner.event_index}, f, indent=2)
    logger.info("Shadow phase complete. Events: %d. Output: %s", runner.event_index, out)


def run_live(config_path: Path, duration: int) -> None:
    """Phase 3: Live testnet trading — real orders on testnet."""
    from infra.config.loader import load_config_secure, resolve_credentials
    from execution.adapters.binance.rest import BinanceRestClient, BinanceRestConfig
    from execution.adapters.binance.urls import resolve_binance_urls
    from runner.live_runner import LiveRunner, LiveRunnerConfig

    raw = load_config_secure(config_path)
    _ensure_testnet(raw)
    resolve_credentials(raw)

    trading = raw.get("trading", {})
    symbol = trading.get("symbol", "BTCUSDT")
    symbols = tuple(trading["symbols"]) if "symbols" in trading else (symbol,)

    creds = raw.get("credentials", {})
    api_key = os.environ.get(creds.get("api_key_env", ""), "")
    api_secret = os.environ.get(creds.get("api_secret_env", ""), "")

    if not api_key or not api_secret:
        key_env = creds.get("api_key_env", "BINANCE_TESTNET_API_KEY")
        secret_env = creds.get("api_secret_env", "BINANCE_TESTNET_API_SECRET")
        print(f"Missing testnet API credentials.")
        print(f"  1. Register at https://testnet.binancefuture.com/")
        print(f"  2. Generate API key/secret")
        print(f"  3. Export env vars:")
        print(f"     export {key_env}=<your_api_key>")
        print(f"     export {secret_env}=<your_api_secret>")
        sys.exit(1)

    urls = resolve_binance_urls(testnet=True)
    client = BinanceRestClient(
        cfg=BinanceRestConfig(
            base_url=urls.rest_base,
            api_key=api_key,
            api_secret=api_secret,
        )
    )

    config = LiveRunnerConfig(
        symbols=symbols,
        testnet=True,
        enable_persistent_stores=False,
    )

    fc, models, dms = _build_ml_stack(raw)
    runner = LiveRunner.build(
        config,
        venue_clients={"binance": client},
        feature_computer=fc,
        alpha_models=models or None,
        decision_modules=dms or None,
    )

    if runner.user_stream is not None:
        us_url = getattr(runner.user_stream.cfg, "ws_base_url", "unknown")
        logger.info("User stream wired: base_url=%s", us_url)
    else:
        logger.info("User stream not wired (shadow or non-Binance)")

    def _timeout(*_: Any) -> None:
        logger.info("Live testnet phase duration reached (%ds), stopping...", duration)
        runner.stop()

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(duration)

    logger.info("Starting LIVE TESTNET phase for %ds...", duration)
    try:
        runner.start()
    except KeyboardInterrupt:
        runner.stop()

    out = _output_dir(config_path)
    _write_equity_csv(out / "live_equity.csv", runner.fills, 10000.0)
    logger.info("Live testnet phase complete. Fills: %d. Output: %s", len(runner.fills), out)


def run_compare(config_path: Path) -> None:
    """Compare paper vs live equity curves."""
    from runner.backtest.pnl_compare import compare_from_files

    out = _output_dir(config_path)
    paper_csv = out / "paper_equity.csv"
    live_csv = out / "live_equity.csv"

    if not paper_csv.exists() or not live_csv.exists():
        print(f"Missing files. Run paper and live phases first.")
        print(f"  Expected: {paper_csv}")
        print(f"  Expected: {live_csv}")
        sys.exit(1)

    result = compare_from_files(paper_csv, live_csv)

    print("=" * 60)
    print("TESTNET VALIDATION — PnL COMPARISON")
    print("=" * 60)
    print(f"Paper final equity:  {result.backtest_final_equity}")
    print(f"Live final equity:   {result.live_final_equity}")
    print(f"Paper return:        {result.backtest_return_pct:.2f}%")
    print(f"Live return:         {result.live_return_pct:.2f}%")
    print(f"Return divergence:   {result.return_divergence_pct:.2f}%")
    print(f"Correlation:         {result.correlation:.4f}")
    print(f"Tracking error:      {result.tracking_error_pct:.4f}%")
    print(f"Paper max drawdown:  {result.backtest_max_dd_pct:.2f}%")
    print(f"Live max drawdown:   {result.live_max_dd_pct:.2f}%")
    print(f"Aligned points:      {result.aligned_points}")
    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")
    print("=" * 60)


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Testnet validation workflow")
    parser.add_argument("--config", type=Path, required=True, help="Testnet config YAML")
    parser.add_argument(
        "--phase",
        choices=["paper", "shadow", "live", "compare"],
        required=True,
        help="Validation phase to run",
    )
    parser.add_argument("--duration", type=int, default=300, help="Phase duration in seconds")
    args = parser.parse_args()

    if args.phase == "paper":
        run_paper(args.config, args.duration)
    elif args.phase == "shadow":
        run_shadow(args.config, args.duration)
    elif args.phase == "live":
        run_live(args.config, args.duration)
    elif args.phase == "compare":
        run_compare(args.config)


if __name__ == "__main__":
    main()
