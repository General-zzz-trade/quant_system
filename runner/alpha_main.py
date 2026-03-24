"""Framework-native alpha entry point.

Replaces the AlphaRunner-based entry point (scripts/ops/run_bybit_alpha.py)
with EngineCoordinator + DecisionBridge + ExecutionBridge + Bybit WS.

Usage::

    python3 -m runner.alpha_main --symbols BTCUSDT BTCUSDT_4h ETHUSDT ETHUSDT_4h --ws
    python3 -m runner.alpha_main --dry-run  # no execution, just signals
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from _quant_hotpath import RustInferenceBridge

from alpha.model_loader_prod import create_adapter, load_model
from data.quality.validators import BarValidator
from data.quality.gaps import GapDetector
from decision.modules.alpha import AlphaDecisionModule
from decision.signals.alpha_signal import EnsemblePredictor, SignalDiscretizer
from decision.sizing.adaptive import AdaptivePositionSizer
from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from engine.feature_hook import FeatureComputeHook
from event.header import EventHeader
from event.types import EventType, ControlEvent, FundingEvent, MarketEvent
from execution.adapters.bybit.execution_adapter import BybitExecutionAdapter
from execution.adapters.bybit.ws_client import BybitWsClient
from runner.strategy_config import SYMBOL_CONFIG

logger = logging.getLogger(__name__)

MODEL_BASE = Path("models_v8")


# ── Builder ─────────────────────────────────────────────────


def _build_coordinator(
    symbol: str,
    runner_key: str,
    model_info: dict,
    adapter: Any,
    dry_run: bool = False,
) -> tuple[EngineCoordinator, AlphaDecisionModule]:
    """Build a full coordinator pipeline for one runner.

    Returns (coordinator, alpha_module) so callers can wire consensus
    and warmup independently.
    """
    cfg = SYMBOL_CONFIG.get(runner_key, {})
    is_4h = "4h" in runner_key

    # Feature engine (per-symbol Rust instance created lazily by hook)
    feature_hook = FeatureComputeHook(
        computer=None,
        warmup_bars=cfg.get("warmup", 300 if is_4h else 800),
    )

    # Inference bridge for z-score normalization + constraints
    bridge = RustInferenceBridge(
        model_info["zscore_window"],
        model_info["zscore_warmup"],
    )

    # Signal pipeline components
    predictor = EnsemblePredictor(
        model_info["horizon_models"],
        model_info["config"],
    )
    discretizer = SignalDiscretizer(
        bridge,
        symbol=symbol,
        deadzone=model_info["deadzone"],
        min_hold=model_info["min_hold"],
        max_hold=model_info["max_hold"],
        long_only=model_info.get("long_only", False),
    )
    sizer = AdaptivePositionSizer(
        runner_key=runner_key,
        step_size=cfg.get("step", 0.001),
        min_size=cfg.get("size", 0.001),
        max_qty=cfg.get("max_qty", 0),
    )

    # Decision module
    alpha_module = AlphaDecisionModule(
        symbol=symbol,
        runner_key=runner_key,
        predictor=predictor,
        discretizer=discretizer,
        sizer=sizer,
    )

    # Coordinator config
    coordinator_cfg = CoordinatorConfig(
        symbol_default=symbol,
        symbols=(symbol,),
        currency="USDT",
        feature_hook=feature_hook,
    )

    # Assemble coordinator
    coordinator = EngineCoordinator(cfg=coordinator_cfg)

    # Attach decision bridge
    decision_bridge = DecisionBridge(
        dispatcher_emit=coordinator.emit,
        modules=[alpha_module],
    )
    coordinator.attach_decision_bridge(decision_bridge)

    # Attach execution bridge (live only)
    if not dry_run:
        exec_adapter = BybitExecutionAdapter(adapter)
        execution_bridge = ExecutionBridge(
            adapter=exec_adapter,
            dispatcher_emit=coordinator.emit,
        )
        coordinator.attach_execution_bridge(execution_bridge)

    logger.info(
        "Built coordinator: runner_key=%s symbol=%s dry_run=%s warmup=%d",
        runner_key, symbol, dry_run,
        cfg.get("warmup", 300 if is_4h else 800),
    )
    return coordinator, alpha_module


# ── Warmup ──────────────────────────────────────────────────


def _validate_warmup_bars(
    bars: list[dict],
    symbol: str,
    interval: str,
) -> None:
    """Run data quality checks on warmup bars and log results.

    Converts adapter dicts to Bar objects for the validators, then logs
    any errors/warnings. Never blocks warmup — quality issues are warnings only.
    """
    from data.store import Bar as StoreBar

    if not bars:
        return

    interval_seconds = 14400 if interval == "240" else 3600

    # Convert adapter dicts to Bar objects
    store_bars: list[StoreBar] = []
    for raw in bars:
        ts_ms = raw.get("time") or raw.get("start") or 0
        ts = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
        try:
            store_bars.append(StoreBar(
                ts=ts,
                open=Decimal(str(raw["open"])),
                high=Decimal(str(raw["high"])),
                low=Decimal(str(raw["low"])),
                close=Decimal(str(raw["close"])),
                volume=Decimal(str(raw["volume"])),
                symbol=symbol,
            ))
        except (KeyError, ValueError) as exc:
            logger.warning("Warmup bar conversion failed at ts=%s: %s", ts_ms, exc)

    if not store_bars:
        return

    # Bar validation (OHLC consistency, time continuity, anomaly detection)
    validator = BarValidator(
        zscore_threshold=5.0,
        max_gap_seconds=interval_seconds * 2,
    )
    result = validator.validate(store_bars)

    if not result.valid:
        logger.warning(
            "Warmup data quality ERRORS for %s (interval=%s): %d errors — %s",
            symbol, interval, len(result.errors),
            "; ".join(result.errors[:5]),
        )
    if result.warnings:
        logger.info(
            "Warmup data quality warnings for %s (interval=%s): %d warnings — %s",
            symbol, interval, len(result.warnings),
            "; ".join(result.warnings[:5]),
        )

    # Gap detection
    gap_detector = GapDetector(interval_seconds=interval_seconds)
    gap_report = gap_detector.detect(
        store_bars,
        start=store_bars[0].ts,
        end=store_bars[-1].ts,
    )
    if gap_report.gaps:
        logger.warning(
            "Warmup gap report for %s (interval=%s): %d gaps, %.1f%% complete",
            symbol, interval, len(gap_report.gaps), gap_report.completeness_pct,
        )
    else:
        logger.info(
            "Warmup quality OK for %s (interval=%s): %d bars, 0 gaps, %d anomalies",
            symbol, interval, result.stats["total_bars"], result.stats["anomalies"],
        )


def _warmup(
    coordinator: EngineCoordinator,
    adapter: Any,
    symbol: str,
    interval: str,
    limit: int,
) -> int:
    """Fetch historical bars and push through the coordinator.

    Returns the number of bars processed.
    """
    logger.info("Warmup: fetching %d bars for %s interval=%s", limit, symbol, interval)
    bars = adapter.get_klines(symbol, interval=interval, limit=limit)

    # Bybit returns newest first — reverse for chronological order
    bars = list(reversed(bars))

    # Data quality validation (log-only, never blocks warmup)
    try:
        _validate_warmup_bars(bars, symbol, interval)
    except Exception:
        logger.exception("Warmup data quality check failed (non-fatal)")

    count = 0
    for bar in bars:
        ts_ms = bar.get("time") or bar.get("start") or 0
        ts = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)

        header = EventHeader.new_root(
            event_type=EventType.MARKET,
            version=1,
            source="warmup",
        )
        event = MarketEvent(
            header=header,
            ts=ts,
            symbol=symbol,
            open=Decimal(str(bar["open"])),
            high=Decimal(str(bar["high"])),
            low=Decimal(str(bar["low"])),
            close=Decimal(str(bar["close"])),
            volume=Decimal(str(bar["volume"])),
        )
        coordinator.emit(event, actor="warmup")
        count += 1

    logger.info("Warmup complete: %s %d bars", symbol, count)
    return count


# ── Main ────────────────────────────────────────────────────


def main() -> None:
    """Framework-native alpha entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Framework-native alpha runner")
    parser.add_argument(
        "--symbols", nargs="+",
        default=["BTCUSDT", "BTCUSDT_4h", "ETHUSDT", "ETHUSDT_4h"],
        help="Runner keys from SYMBOL_CONFIG",
    )
    parser.add_argument("--ws", action="store_true", help="Use WebSocket for live bars")
    parser.add_argument("--dry-run", action="store_true", help="No execution (signals only)")
    args = parser.parse_args()

    # Adapter (Bybit REST)
    adapter = create_adapter()
    logger.info("Bybit adapter connected")

    # Build coordinators
    coordinators: dict[str, EngineCoordinator] = {}
    modules: dict[str, AlphaDecisionModule] = {}

    for runner_key in args.symbols:
        cfg = SYMBOL_CONFIG.get(runner_key)
        if cfg is None:
            logger.warning("Unknown runner_key %s — skipping", runner_key)
            continue

        model_dir = MODEL_BASE / cfg["model_dir"]
        model_info = load_model(model_dir)
        symbol = cfg.get("symbol", runner_key)

        coord, module = _build_coordinator(
            symbol=symbol,
            runner_key=runner_key,
            model_info=model_info,
            adapter=adapter,
            dry_run=args.dry_run,
        )
        coordinators[runner_key] = coord
        modules[runner_key] = module

    if not coordinators:
        logger.error("No coordinators built — exiting")
        sys.exit(1)

    # Share consensus dict across all modules
    consensus: dict[str, int] = {}
    for module in modules.values():
        module.set_consensus(consensus)

    # Warmup each coordinator
    for runner_key, coord in coordinators.items():
        cfg = SYMBOL_CONFIG[runner_key]
        symbol = cfg.get("symbol", runner_key)
        interval = cfg.get("interval", "60")
        is_4h = "4h" in runner_key
        warmup_limit = cfg.get("warmup", 300 if is_4h else 800)
        _warmup(coord, adapter, symbol, interval, warmup_limit)

    # Start all coordinators
    for coord in coordinators.values():
        coord.start()
    logger.info("All %d coordinators started", len(coordinators))

    # Graceful shutdown
    shutdown = False

    def _handle_signal(signum: int, frame: Any) -> None:
        nonlocal shutdown
        logger.info("Signal %d received — shutting down", signum)
        shutdown = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # WS on_bar callback: route bar to matching coordinators
    def _on_bar(ws_symbol: str, bar: dict) -> None:
        if not bar.get("confirm", True):
            return

        ts_ms = bar.get("time") or bar.get("start") or 0
        ts = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)

        header = EventHeader.new_root(
            event_type=EventType.MARKET,
            version=1,
            source="bybit_ws",
        )
        event = MarketEvent(
            header=header,
            ts=ts,
            symbol=ws_symbol,
            open=Decimal(str(bar["open"])),
            high=Decimal(str(bar["high"])),
            low=Decimal(str(bar["low"])),
            close=Decimal(str(bar["close"])),
            volume=Decimal(str(bar["volume"])),
        )

        # Emit FundingEvent when funding rate data is available in the bar
        funding_rate_raw = bar.get("funding_rate")
        if funding_rate_raw is not None:
            try:
                fr_val = float(funding_rate_raw)
                if fr_val == fr_val:  # not NaN
                    funding_header = EventHeader.new_root(
                        event_type=EventType.FUNDING,
                        version=1,
                        source="bybit_ws",
                    )
                    funding_event = FundingEvent(
                        header=funding_header,
                        ts=ts,
                        symbol=ws_symbol,
                        funding_rate=Decimal(str(funding_rate_raw)),
                        mark_price=Decimal(str(bar["close"])),
                    )
                    # Route funding event to all matching coordinators
                    bar_interval = str(bar.get("interval", "60"))
                    for runner_key, coord in coordinators.items():
                        cfg = SYMBOL_CONFIG[runner_key]
                        runner_symbol = cfg.get("symbol", runner_key)
                        runner_interval = cfg.get("interval", "60")
                        if runner_symbol == ws_symbol and runner_interval == bar_interval:
                            try:
                                coord.emit(funding_event, actor="live")
                            except Exception:
                                logger.debug("Error emitting funding to %s", runner_key)
            except (ValueError, TypeError):
                pass

        # Route to all coordinators that handle this symbol + interval
        bar_interval = str(bar.get("interval", "60"))
        for runner_key, coord in coordinators.items():
            cfg = SYMBOL_CONFIG[runner_key]
            runner_symbol = cfg.get("symbol", runner_key)
            runner_interval = cfg.get("interval", "60")
            if runner_symbol == ws_symbol and runner_interval == bar_interval:
                try:
                    coord.emit(event, actor="live")
                except Exception:
                    logger.exception("Error emitting bar to %s", runner_key)

    # WebSocket setup
    ws_clients: list[BybitWsClient] = []
    if args.ws:
        # Group symbols by interval for separate WS connections
        interval_symbols: dict[str, list[str]] = {}
        for runner_key in coordinators:
            cfg = SYMBOL_CONFIG[runner_key]
            symbol = cfg.get("symbol", runner_key)
            interval = cfg.get("interval", "60")
            interval_symbols.setdefault(interval, [])
            if symbol not in interval_symbols[interval]:
                interval_symbols[interval].append(symbol)

        for interval, symbols in interval_symbols.items():
            ws = BybitWsClient(
                symbols=symbols,
                interval=interval,
                on_bar=_on_bar,
            )
            ws.start()
            ws_clients.append(ws)
            logger.info("WS started: interval=%s symbols=%s", interval, symbols)

    # Main loop
    try:
        while not shutdown:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down...")
        # Emit ControlEvent to signal shutdown to all coordinators
        for runner_key, coord in coordinators.items():
            try:
                control_header = EventHeader.new_root(
                    event_type=EventType.CONTROL,
                    version=1,
                    source="system",
                )
                shutdown_event = ControlEvent(
                    header=control_header,
                    command="shutdown",
                    reason="process termination",
                )
                logger.info(
                    "Emitting shutdown ControlEvent to %s: %s",
                    runner_key, shutdown_event.command,
                )
            except Exception:
                logger.debug("Failed to create shutdown ControlEvent for %s", runner_key)
        for ws in ws_clients:
            ws.stop()
        for coord in coordinators.values():
            coord.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
