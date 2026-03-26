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
import os
import signal
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from _quant_hotpath import (
    RustKillSwitch,
    rust_event_types,
    rust_sides,
    rust_signal_sides,
    rust_venues,
    rust_order_types,
    rust_time_in_force,
)

from alpha.model_loader_prod import create_adapter, load_model
from decision.modules.alpha import AlphaDecisionModule
from decision.signals.alpha_signal import EnsemblePredictor
from engine.coordinator import EngineCoordinator
from event.header import EventHeader
from event.types import EventType, ControlEvent, FundingEvent, MarketEvent
from execution.adapters.bybit.ws_client import BybitWsClient
from runner.builders.alpha_builder import build_coordinator as _build_coordinator
from strategy.config import SYMBOL_CONFIG
from runner.warmup import warmup as _warmup

# PnL tracking (graceful degradation if attribution module unavailable)
try:
    from attribution.pnl_tracker import PnLTracker as _PnLTracker
    _HAS_PNL_TRACKER = True
except Exception:
    _HAS_PNL_TRACKER = False

logger = logging.getLogger(__name__)

MODEL_BASE = Path("models_v8")


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

    # Validate Rust event configuration at startup
    VALID_EVENT_TYPES = set(rust_event_types())
    VALID_SIDES = set(rust_sides())
    VALID_SIGNAL_SIDES = set(rust_signal_sides())
    VALID_VENUES = set(rust_venues())
    VALID_ORDER_TYPES = set(rust_order_types())
    VALID_TIF = set(rust_time_in_force())
    logger.info(
        "Event config validated: %d event types, %d sides, %d signal sides, "
        "%d venues, %d order types, %d TIF values",
        len(VALID_EVENT_TYPES), len(VALID_SIDES), len(VALID_SIGNAL_SIDES),
        len(VALID_VENUES), len(VALID_ORDER_TYPES), len(VALID_TIF),
    )

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

    # ── PnL Tracker (observer mode — read-only, never affects trading) ──
    pnl_tracker: Any = None
    if _HAS_PNL_TRACKER:
        try:
            pnl_tracker = _PnLTracker()
            # Track open positions: {symbol: {"side": int, "entry_price": float, "qty": float}}
            _pnl_positions: dict[str, dict[str, Any]] = {}

            def _on_fill_for_pnl(event: Any) -> None:
                """Observer: record fills into PnLTracker. Never raises."""
                try:
                    # Only handle FillEvent (has fill_id attribute)
                    if not hasattr(event, "fill_id"):
                        return
                    symbol = str(getattr(event, "symbol", ""))
                    if not symbol:
                        return
                    fill_price = float(getattr(event, "price", 0))
                    fill_qty = float(getattr(event, "qty", 0))
                    fill_side = str(getattr(event, "side", "buy"))
                    if fill_price <= 0 or fill_qty <= 0:
                        return

                    # Determine side as int: +1 for long, -1 for short
                    side_int = 1 if fill_side == "buy" else -1

                    pos = _pnl_positions.get(symbol)
                    if pos is not None and pos["side"] != side_int:
                        # Closing fill (opposite side to existing position)
                        pnl_tracker.record_close(
                            symbol=symbol,
                            side=pos["side"],
                            entry_price=pos["entry_price"],
                            exit_price=fill_price,
                            size=min(fill_qty, pos["qty"]),
                            reason="fill",
                        )
                        remaining = pos["qty"] - fill_qty
                        if remaining <= 1e-12:
                            _pnl_positions.pop(symbol, None)
                        else:
                            pos["qty"] = remaining
                    else:
                        # Opening or adding to position
                        if pos is None:
                            _pnl_positions[symbol] = {
                                "side": side_int,
                                "entry_price": fill_price,
                                "qty": fill_qty,
                            }
                        else:
                            # Average entry price
                            total_qty = pos["qty"] + fill_qty
                            pos["entry_price"] = (
                                (pos["entry_price"] * pos["qty"] + fill_price * fill_qty)
                                / total_qty
                            )
                            pos["qty"] = total_qty
                except Exception:
                    logger.debug("PnL fill observer error", exc_info=True)

            # Register fill observer on each coordinator's dispatcher
            from engine.dispatcher import Route
            for coord in coordinators.values():
                coord.dispatcher.register(
                    route=Route.PIPELINE, handler=_on_fill_for_pnl,
                )
            logger.info("PnL tracker initialized and attached to %d coordinators", len(coordinators))
        except Exception:
            logger.warning("PnL tracker initialization failed — continuing without PnL tracking", exc_info=True)
            pnl_tracker = None
    else:
        logger.warning("attribution module unavailable — PnL tracking disabled")

    # Warmup each coordinator (execution bridge detached — no real orders)
    for runner_key, coord in coordinators.items():
        cfg = SYMBOL_CONFIG[runner_key]
        symbol = cfg.get("symbol", runner_key)
        interval = cfg.get("interval", "60")
        is_4h = "4h" in runner_key
        warmup_limit = cfg.get("warmup", 300 if is_4h else 800)
        _warmup(coord, adapter, symbol, interval, warmup_limit)

    # Reset alpha module signal state after warmup so live starts clean
    for runner_key, alpha_mod in modules.items():
        alpha_mod._signal = 0
        alpha_mod._current_qty = __import__("decimal").Decimal("0")
        alpha_mod._entry_price = 0.0
        alpha_mod._trade_peak = 0.0
        alpha_mod._last_trade_bar = alpha_mod._bars_processed  # cooldown from warmup end
        # Reset Rust-side hold counter via discretizer bridge
        try:
            alpha_mod._discretizer._bridge.reset_hold(alpha_mod._symbol)
        except Exception:
            pass  # bridge may not support reset_hold
    logger.info("Post-warmup reset: all alpha modules signal=0, hold counters reset")

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

    # SIGHUP: hot-reload models without restart
    def _handle_sighup(signum: int, frame: Any) -> None:
        """Hot-reload models from disk for all runners."""
        logger.info("SIGHUP received — reloading models...")
        t0 = time.monotonic()
        reloaded = 0
        failed = 0
        for runner_key, alpha_mod in modules.items():
            try:
                cfg = SYMBOL_CONFIG[runner_key]
                model_dir = MODEL_BASE / cfg["model_dir"]
                model_info = load_model(model_dir)
                new_predictor = EnsemblePredictor(
                    model_info["horizon_models"],
                    model_info["config"],
                )
                alpha_mod.update_predictor(new_predictor)
                reloaded += 1
                logger.info("Reloaded model for %s", runner_key)
            except Exception:
                failed += 1
                logger.exception(
                    "Failed to reload model for %s — keeping current", runner_key,
                )
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "Hot-reload complete: %d reloaded, %d failed, %.1fms",
            reloaded, failed, elapsed_ms,
        )

    signal.signal(signal.SIGHUP, _handle_sighup)

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

    # Daily drawdown kill switch
    _kill_switch = RustKillSwitch()
    _daily_start_equity: float | None = None
    _MAX_DAILY_DRAWDOWN_PCT = float(os.environ.get("MAX_DAILY_DRAWDOWN_PCT", "5.0"))
    _SCALE = 100_000_000
    _loop_iter = 0

    def _get_equity() -> float | None:
        """Read equity from the first coordinator's state store (Fd8 → float)."""
        for coord in coordinators.values():
            try:
                account = coord._store.get_account()
                return account.balance / _SCALE
            except Exception:
                pass
        return None

    # Main loop
    try:
        while not shutdown:
            time.sleep(1)
            _loop_iter += 1

            # Check daily drawdown every 60 iterations (~60s)
            if _loop_iter % 60 == 0:
                try:
                    equity = _get_equity()
                    if equity is not None and equity > 0:
                        if _daily_start_equity is None:
                            _daily_start_equity = equity
                            logger.info(
                                "Daily drawdown tracker initialized: start_equity=%.2f",
                                _daily_start_equity,
                            )
                        elif _daily_start_equity > 0:
                            dd_pct = (1.0 - equity / _daily_start_equity) * 100
                            if dd_pct > _MAX_DAILY_DRAWDOWN_PCT:
                                logger.critical(
                                    "DAILY DRAWDOWN %.1f%% exceeds limit %.1f%% — killing trading",
                                    dd_pct, _MAX_DAILY_DRAWDOWN_PCT,
                                )
                                _kill_switch.arm(
                                    "global", "daily_dd", "hard_kill",
                                    f"daily drawdown {dd_pct:.1f}%",
                                    ttl_seconds=3600.0,
                                )
                                for coord in coordinators.values():
                                    coord.halt_trading(
                                        reason=f"daily drawdown {dd_pct:.1f}%"
                                    )
                except Exception:
                    pass  # never crash the main loop

                # PnL status (observer-only, never crashes main loop)
                if pnl_tracker is not None:
                    try:
                        tc = pnl_tracker.trade_count
                        if tc > 0:
                            logger.info(
                                "PnL status: trades=%d pnl=%.2f win_rate=%.1f%% dd=%.2f%% | %s",
                                tc, pnl_tracker.total_pnl,
                                pnl_tracker.win_rate * 100,
                                pnl_tracker.drawdown_pct,
                                pnl_tracker.pnl_by_symbol,
                            )
                    except Exception:
                        pass
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down...")
        # PnL summary at shutdown
        if pnl_tracker is not None:
            try:
                summary = pnl_tracker.summary()
                logger.info("PnL summary at shutdown: %s", summary)
            except Exception:
                logger.debug("Failed to generate PnL shutdown summary", exc_info=True)
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
