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
import signal  # noqa: F401 — used in run() for SIGINT/SIGTERM/SIGHUP handlers
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
from event.types import EventType, ControlEvent, FillEvent, FundingEvent, MarketEvent
from execution.adapters.bybit.ws_client import BybitWsClient
from runner.builders.alpha_builder import build_coordinator as _build_coordinator
from strategy.config import SYMBOL_CONFIG
from monitoring.notify import send_alert, AlertLevel
from monitoring.rolling_sharpe import RollingSharpeTracker
from runner.limit_order_manager import LimitOrderManager
from runner.warmup import warmup as _warmup
from data.oi_cache import BinanceOICache

# Z-score buffer checkpoint paths
_ZSCORE_CHECKPOINT_DIR = Path("data/runtime/zscore_checkpoints")

def _save_zscore_checkpoint(runner_key: str, bridge) -> None:
    """Save InferenceBridge z-score buffer to disk for fast restart."""
    try:
        _ZSCORE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        data = bridge.checkpoint()
        path = _ZSCORE_CHECKPOINT_DIR / f"{runner_key}.json"
        import json
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass  # best-effort

def _restore_zscore_checkpoint(runner_key: str, bridge) -> bool:
    """Restore InferenceBridge z-score buffer from disk. Returns True if restored."""
    try:
        import json
        path = _ZSCORE_CHECKPOINT_DIR / f"{runner_key}.json"
        if not path.exists():
            return False
        with open(path) as f:
            data = json.load(f)
        bridge.restore(data)
        return True
    except Exception:
        return False

# PnL tracking (graceful degradation if attribution module unavailable)
try:
    from attribution.pnl_tracker import PnLTracker as _PnLTracker
    _HAS_PNL_TRACKER = True
except Exception:
    _HAS_PNL_TRACKER = False

logger = logging.getLogger(__name__)

MODEL_BASE = Path("models_v8")


def _create_binance_adapter():
    """Create Binance Futures adapter from environment variables."""
    from execution.adapters.binance import BinanceAdapter, BinanceConfig

    api_key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
    api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")
    if not api_key or not api_secret:
        raise RuntimeError(
            "BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET "
            "environment variables are required for --venue binance."
        )
    testnet = os.environ.get("BINANCE_TESTNET", "1").lower() in ("1", "true", "yes")
    config = BinanceConfig(api_key=api_key, api_secret=api_secret, testnet=testnet)
    adapter = BinanceAdapter(config)
    if not adapter.connect():
        raise RuntimeError("Failed to connect to Binance Futures")
    return adapter


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
    parser.add_argument(
        "--venue", choices=["bybit", "binance"], default="bybit",
        help="Exchange venue (default: bybit)",
    )
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

    # Adapter (venue-specific REST)
    if args.venue == "binance":
        adapter = _create_binance_adapter()
        logger.info("Binance adapter connected")
    else:
        adapter = create_adapter()
        logger.info("Bybit adapter connected")

    # Limit order pre-placement manager (reduces slippage vs market orders)
    _limit_mgr = LimitOrderManager(
        adapter=adapter,
        offset_bps=float(os.environ.get("LIMIT_OFFSET_BPS", "30")),
        post_only=True,
        ttl_seconds=float(os.environ.get("LIMIT_TTL_S", "300")),
        qty_scale=float(os.environ.get("LIMIT_QTY_SCALE", "0.5")),
    )

    # Build coordinators
    coordinators: dict[str, EngineCoordinator] = {}
    modules: dict[str, AlphaDecisionModule] = {}

    # Live OI/LS caches — one per unique symbol (shared across 1h/4h runners)
    oi_caches: dict[str, BinanceOICache] = {}
    for runner_key in args.symbols:
        cfg = SYMBOL_CONFIG.get(runner_key)
        if cfg is None:
            continue
        sym = cfg.get("symbol", runner_key)
        if sym not in oi_caches:
            oi_caches[sym] = BinanceOICache(sym)
    for sym, cache in oi_caches.items():
        cache.start()
        logger.info("Started BinanceOICache for %s (55s refresh)", sym)

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
            oi_cache=oi_caches.get(symbol),
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

    # Restore z-score checkpoints BEFORE warmup (preserves historical variance)
    restored_count = 0
    for runner_key, alpha_mod in modules.items():
        try:
            bridge = alpha_mod._discretizer._bridge
            if _restore_zscore_checkpoint(runner_key, bridge):
                restored_count += 1
                logger.info("Restored z-score checkpoint for %s", runner_key)
        except Exception:
            pass
    if restored_count > 0:
        logger.info("Restored %d/%d z-score checkpoints — signals ready immediately", restored_count, len(modules))

    # Seed cross-symbol close prices so dominance features work during warmup
    try:
        from engine.feature_hook import _last_closes
        for sym in ["BTCUSDT", "ETHUSDT"]:
            ticker = adapter.get_ticker(sym)
            if ticker:
                price = float(getattr(ticker, "last_price", 0) or 0)
                if price > 0:
                    _last_closes[sym] = price
                    logger.info("Seeded %s close=%.2f for dominance", sym, price)
    except Exception:
        logger.debug("Cross-symbol close seeding failed (non-fatal)", exc_info=True)

    # Disable audit logging during warmup (prevents fake entries in decision_audit.jsonl)
    for alpha_mod in modules.values():
        alpha_mod._audit_enabled = False

    # Parallel warmup — each coordinator warms up in its own thread
    # (execution bridge detached — no real orders)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _warmup_one(runner_key: str, coord):
        cfg = SYMBOL_CONFIG[runner_key]
        symbol = cfg.get("symbol", runner_key)
        interval = cfg.get("interval", "60")
        is_4h = "4h" in runner_key
        warmup_limit = cfg.get("warmup", 300 if is_4h else 800)
        _warmup(coord, adapter, symbol, interval, warmup_limit)
        return runner_key

    t0_warmup = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_warmup_one, rk, co): rk for rk, co in coordinators.items()}
        for fut in as_completed(futures):
            try:
                rk = fut.result()
                logger.debug("Warmup done: %s", rk)
            except Exception:
                logger.exception("Warmup failed for %s", futures[fut])
    logger.info("Parallel warmup complete: %d runners in %.1fs",
                len(coordinators), time.monotonic() - t0_warmup)

    # RE-LOAD batch z-score checkpoints AFTER warmup to override
    # the incorrect incremental predictions that warmup pushed into the buffer.
    # This is the critical fix for the batch/incremental feature divergence.
    for runner_key, alpha_mod in modules.items():
        try:
            bridge = alpha_mod._discretizer._bridge
            if _restore_zscore_checkpoint(runner_key, bridge):
                logger.info("Re-loaded batch z-score checkpoint for %s (post-warmup)", runner_key)
        except Exception:
            pass

    # Re-enable audit logging after warmup
    for alpha_mod in modules.values():
        alpha_mod._audit_enabled = True

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

    # Sync positions from exchange — if exchange has an open position,
    # restore alpha module state so stop-loss/quick_loss work correctly.
    try:
        exchange_positions = adapter.get_positions()
        if exchange_positions:
            for pos in exchange_positions:
                sym = pos.symbol
                side = pos.side  # "Buy" or "Sell"
                qty = float(pos.qty)
                entry = float(pos.entry_price)
                if qty <= 0 or entry <= 0:
                    continue
                # Find matching 1h runner (not 4h — exchange position is shared)
                for rk, am in modules.items():
                    cfg = SYMBOL_CONFIG[rk]
                    if cfg.get("symbol", rk) == sym and "4h" not in rk and "15m" not in rk:
                        pos_signal = 1 if side.lower() == "buy" else -1
                        am._signal = pos_signal
                        am._entry_price = entry
                        am._trade_peak = entry
                        am._current_qty = __import__("decimal").Decimal(str(qty))
                        am._last_trade_bar = -9999  # allow immediate exit if needed
                        logger.info(
                            "POSITION SYNC %s: %s qty=%.4f entry=%.2f → pos_signal=%+d",
                            sym, side, qty, entry, pos_signal,
                        )
                        break
    except Exception:
        logger.debug("Position sync failed (non-fatal)", exc_info=True)

    # Start all coordinators
    for coord in coordinators.values():
        coord.start()
    logger.info("All %d coordinators started", len(coordinators))

    # Allow instant signal to trigger immediately after restart.
    # Z-clamp (|z|>3.5 → dz+0.5) prevents extreme post-warmup z-scores
    # from causing wrong-direction trades. Existing positions are already
    # synced via POSITION SYNC above.
    for runner_key, alpha_mod in modules.items():
        alpha_mod._last_trade_bar = -9999  # allow first trade immediately

    # Push batch predictions into z-score buffer BEFORE instant signal
    # This ensures the first signal uses batch-calibrated predictions
    for runner_key, alpha_mod in modules.items():
        try:
            cfg = SYMBOL_CONFIG[runner_key]
            if "4h" in runner_key or "15m" in runner_key:
                continue
            symbol = cfg.get("symbol", runner_key)
            model_dir = cfg.get("model_dir", runner_key)
            from runner.batch_predictor import predict_latest
            batch_pred = predict_latest(symbol, model_dir)
            if batch_pred is not None:
                bridge = alpha_mod._discretizer._bridge
                bridge.zscore_normalize(symbol, batch_pred, alpha_mod._bars_processed)
                logger.info("Batch prediction pushed: %s pred=%+.6f", symbol, batch_pred)
        except Exception:
            pass

    # Instant signal via BATCH prediction (not incremental features).
    # Pushes batch prediction into z-score buffer and checks for signal.
    # If z > deadzone → places order directly. Skips incremental feature pipeline.
    for runner_key, alpha_mod in modules.items():
        try:
            cfg = SYMBOL_CONFIG[runner_key]
            if "4h" in runner_key or "15m" in runner_key:
                continue
            symbol = cfg.get("symbol", runner_key)
            model_dir = cfg.get("model_dir", runner_key)

            from runner.batch_predictor import predict_latest
            batch_pred = predict_latest(symbol, model_dir)
            if batch_pred is None:
                continue

            bridge = alpha_mod._discretizer._bridge
            hour_key = alpha_mod._bars_processed + 1
            z_val = bridge.zscore_normalize(symbol, batch_pred, hour_key)

            if z_val is None:
                continue
            z = max(-5.0, min(5.0, z_val))
            dz = alpha_mod._discretizer.deadzone

            if abs(z) > dz and alpha_mod._signal == 0:
                # Signal detected — get current price and place order
                bars = adapter.get_klines(symbol, interval="60", limit=1)
                if not bars:
                    continue
                price = float(bars[0].get("close", 0))
                if price <= 0:
                    continue

                new_signal = 1 if z > 0 else -1
                side = "buy" if new_signal == 1 else "sell"
                direction = "LONG" if new_signal == 1 else "SHORT"

                # Size the position
                equity = 0.0
                try:
                    bal = adapter.get_balances().get("USDT")
                    if bal:
                        equity = float(bal.total)
                except Exception:
                    pass
                if equity <= 0:
                    continue

                # Build minimal snapshot for sizer
                _acct = type("A", (), {
                    "balance_f": equity,
                    "balance": int(equity * 100_000_000),
                })()
                _mkt = type("M", (), {
                    "close_f": price,
                    "close": int(price * 100_000_000),
                })()
                _snap = type("S", (), {
                    "account": _acct,
                    "markets": {symbol: _mkt},
                    "features": {},
                    "portfolio": None,
                    "risk": None,
                })()
                qty = alpha_mod._sizer.target_qty(
                    _snap, symbol,
                    leverage=alpha_mod._leverage,
                    ic_scale=alpha_mod._ic_scale,
                )
                if float(qty) <= 0:
                    continue

                # Place order
                resp = adapter.send_market_order(symbol, side, float(qty))
                status = resp.get("status", "")
                if status not in ("error", "failed"):
                    alpha_mod._signal = new_signal
                    alpha_mod._entry_price = price
                    alpha_mod._trade_peak = price
                    alpha_mod._current_qty = qty
                    logger.info(
                        "BATCH INSTANT %s: z=%+.2f > dz=%.1f → %s %.4f @ $%.2f",
                        symbol, z, dz, direction, float(qty), price,
                    )
                else:
                    logger.warning("BATCH INSTANT %s order failed: %s", symbol, resp)
            else:
                logger.info(
                    "Batch instant %s: z=%+.2f (dz=%.1f) → %s",
                    symbol, z, dz, "FLAT" if abs(z) <= dz else "HOLD",
                )
        except Exception:
            logger.debug("Batch instant failed for %s", runner_key, exc_info=True)

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

    # Live funding rate cache — updated by WS ticker, consumed by feature hooks
    _live_funding: dict[str, float] = {}

    # WS on_bar callback: route bar to matching coordinators
    def _on_bar(ws_symbol: str, bar: dict) -> None:
        if not bar.get("confirm", True):
            return

        # Bar closed: cancel any stale pre-placed limit orders for this symbol
        # and check if any were filled (so we don't double-enter)
        try:
            if _limit_mgr.has_pending(ws_symbol):
                fill_info = _limit_mgr.check_fill(ws_symbol)
                if fill_info:
                    # Limit order was filled — update alpha module + state store
                    for rk, am in modules.items():
                        cfg = SYMBOL_CONFIG[rk]
                        if cfg.get("symbol", rk) != ws_symbol:
                            continue
                        if "4h" in rk or "15m" in rk:
                            continue
                        if am._signal != 0:
                            continue  # already in a position
                        # Mark the position as entered via limit fill
                        side = fill_info["side"]
                        am._signal = 1 if side == "buy" else -1
                        am._entry_price = fill_info["price"]
                        am._trade_peak = fill_info["price"]
                        am._current_qty = Decimal(str(fill_info["qty"]))
                        am._last_trade_bar = am._bars_processed
                        am._consensus[rk] = am._signal
                        logger.info(
                            "LIMIT FILL applied %s: signal=%d entry=$%.2f qty=%.4f",
                            rk, am._signal, am._entry_price, fill_info["qty"],
                        )
                        # Emit FillEvent so coordinator state store tracks the position
                        coord = coordinators.get(rk)
                        if coord is not None:
                            fill_header = EventHeader.new_root(
                                event_type=EventType.FILL,
                                version=1,
                                source=f"limit_fill.{rk}",
                            )
                            fill_ev = FillEvent(
                                header=fill_header,
                                fill_id=fill_info.get("orderId", fill_header.event_id),
                                order_id=fill_info.get("orderId", ""),
                                symbol=ws_symbol,
                                qty=Decimal(str(fill_info["qty"])),
                                price=Decimal(str(fill_info["price"])),
                                side=side,
                            )
                            coord.emit(fill_ev, actor="limit_fill")
                            logger.info(
                                "LIMIT FillEvent emitted %s → coordinator %s",
                                ws_symbol, rk,
                            )
                else:
                    # Not filled — cancel stale order
                    _limit_mgr.cancel_stale(ws_symbol)
        except Exception:
            logger.debug("Limit order bar-close check failed for %s", ws_symbol, exc_info=True)

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

        # Push live funding rates from WS ticker to feature hooks before bar emit
        for ws in ws_clients:
            fr = ws.get_last_funding_rate(ws_symbol)
            if fr is not None:
                _live_funding[ws_symbol] = fr
                break
        if ws_symbol in _live_funding:
            for runner_key, coord in coordinators.items():
                cfg = SYMBOL_CONFIG[runner_key]
                if cfg.get("symbol", runner_key) == ws_symbol:
                    hook = getattr(coord, "_feature_hook", None)
                    if hook is not None:
                        hook.update_live_funding_rate(ws_symbol, _live_funding[ws_symbol])

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

                # Override z-score buffer with batch prediction (fixes incremental divergence)
                if "4h" not in runner_key and "15m" not in runner_key:
                    try:
                        from runner.batch_predictor import predict_latest
                        model_dir = cfg.get("model_dir", runner_key)
                        batch_pred = predict_latest(ws_symbol, model_dir)
                        if batch_pred is not None:
                            am = modules.get(runner_key)
                            if am is not None:
                                bridge = am._discretizer._bridge
                                hour_key = am._bars_processed
                                bridge.zscore_normalize(ws_symbol, batch_pred, hour_key)
                    except Exception:
                        pass  # fallback to incremental prediction

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

        # Real-time price tracker + wick detector
        _rt_prices: dict[str, float] = {}
        _rt_highs: dict[str, float] = {}   # rolling 5-min high
        _rt_lows: dict[str, float] = {}    # rolling 5-min low
        _rt_history: dict[str, list] = {}  # (ts, price) for 5 min
        _wick_cooldown: dict[str, float] = {}  # last wick trade ts

        _WICK_THRESHOLD = 0.008   # 0.8% move = potential wick
        _WICK_BOUNCE = 0.003      # 0.3% bounce from extreme = confirmed wick
        _WICK_COOLDOWN_S = 300.0  # 5 min between wick trades
        _WICK_WINDOW_S = 60.0     # look back 60s for extreme

        def _on_tick(ws_symbol: str, price: float) -> None:
            _rt_prices[ws_symbol] = price
            now = time.time()

            # Maintain 60s price history
            hist = _rt_history.setdefault(ws_symbol, [])
            hist.append((now, price))
            # Trim to 60s window
            cutoff = now - _WICK_WINDOW_S
            while hist and hist[0][0] < cutoff:
                hist.pop(0)

            if len(hist) < 3:
                return

            # Find min/max in window
            prices = [p for _, p in hist]
            win_high = max(prices)
            win_low = min(prices)
            first_price = hist[0][1]

            # Detect wick: price dropped >threshold then bounced >bounce
            # or price spiked >threshold then dropped >bounce
            if first_price <= 0:
                return

            # Check cooldown
            last_wick = _wick_cooldown.get(ws_symbol, 0)
            if now - last_wick < _WICK_COOLDOWN_S:
                return

            # Wick DOWN (buy opportunity): dropped from first, now bouncing
            drop = (first_price - win_low) / first_price
            bounce_from_low = (price - win_low) / win_low if win_low > 0 else 0
            if drop > _WICK_THRESHOLD and bounce_from_low > _WICK_BOUNCE:
                # Check: do we have a long signal or no position?
                for rk, am in modules.items():
                    cfg = SYMBOL_CONFIG[rk]
                    if cfg.get("symbol", rk) != ws_symbol:
                        continue
                    if "4h" in rk or "15m" in rk:
                        continue
                    if am._signal != 0:
                        continue  # already in position
                    _wick_cooldown[ws_symbol] = now
                    logger.info(
                        "WICK DOWN %s: drop=%.2f%% bounce=%.2f%% "
                        "($%.0f→$%.0f→$%.0f) — triggering early buy",
                        ws_symbol, drop * 100, bounce_from_low * 100,
                        first_price, win_low, price,
                    )
                    try:
                        wh = EventHeader.new_root(
                            event_type=EventType.MARKET,
                            version=1, source="wick_buy",
                        )
                        we = MarketEvent(
                            header=wh,
                            ts=datetime.now(timezone.utc),
                            symbol=ws_symbol,
                            open=Decimal(str(first_price)),
                            high=Decimal(str(win_high)),
                            low=Decimal(str(win_low)),
                            close=Decimal(str(price)),
                            volume=Decimal("0"),
                        )
                        coordinators[rk].emit(we, actor="live")
                    except Exception:
                        logger.debug("Wick buy failed", exc_info=True)
                    break

            # Wick UP (sell opportunity): spiked from first, now dropping
            spike = (win_high - first_price) / first_price
            drop_from_high = (win_high - price) / win_high if win_high > 0 else 0
            if spike > _WICK_THRESHOLD and drop_from_high > _WICK_BOUNCE:
                for rk, am in modules.items():
                    cfg = SYMBOL_CONFIG[rk]
                    if cfg.get("symbol", rk) != ws_symbol:
                        continue
                    if "4h" in rk or "15m" in rk:
                        continue
                    if am._signal != 0:
                        continue
                    _wick_cooldown[ws_symbol] = now
                    logger.info(
                        "WICK UP %s: spike=%.2f%% drop=%.2f%% "
                        "($%.0f→$%.0f→$%.0f) — triggering early sell",
                        ws_symbol, spike * 100, drop_from_high * 100,
                        first_price, win_high, price,
                    )
                    try:
                        wh = EventHeader.new_root(
                            event_type=EventType.MARKET,
                            version=1, source="wick_sell",
                        )
                        we = MarketEvent(
                            header=wh,
                            ts=datetime.now(timezone.utc),
                            symbol=ws_symbol,
                            open=Decimal(str(first_price)),
                            high=Decimal(str(win_high)),
                            low=Decimal(str(win_low)),
                            close=Decimal(str(price)),
                            volume=Decimal("0"),
                        )
                        coordinators[rk].emit(we, actor="live")
                    except Exception:
                        logger.debug("Wick sell failed", exc_info=True)
                    break

        for interval, symbols in interval_symbols.items():
            if args.venue == "binance":
                from execution.adapters.binance.ws_kline_client import BinanceWsClient
                ws = BinanceWsClient(
                    symbols=symbols,
                    interval=interval,
                    on_bar=_on_bar,
                    on_tick=_on_tick,
                    testnet=True,
                )
            else:
                ws = BybitWsClient(
                    symbols=symbols,
                    interval=interval,
                    on_bar=_on_bar,
                    on_tick=_on_tick,
                )
            ws.start()
            ws_clients.append(ws)
            logger.info("WS started: interval=%s symbols=%s venue=%s",
                        interval, symbols, args.venue)

    # Daily drawdown kill switch
    _kill_switch = RustKillSwitch()
    _daily_start_equity: float | None = None
    _MAX_DAILY_DRAWDOWN_PCT = float(os.environ.get("MAX_DAILY_DRAWDOWN_PCT", "5.0"))
    _SCALE = 100_000_000
    _loop_iter = 0

    # Live Sharpe tracker — alerts when Sharpe < 0 for 3 consecutive checks
    _sharpe_tracker = RollingSharpeTracker(window=720, interval="1h")
    _sharpe_neg_streak: dict[str, int] = {}  # symbol -> consecutive negative count
    _SHARPE_NEG_THRESHOLD = 3  # alert after N consecutive negative checks
    _last_pnl_trade_count: int = 0  # track how many trades we've already fed

    def _get_equity() -> float | None:
        """Read equity from the first coordinator's state store (Fd8 → float)."""
        for coord in coordinators.values():
            try:
                account = coord._store.get_account()
                return account.balance / _SCALE
            except Exception:
                pass
        return None

    # Auto-reload: watch model config.json files for changes
    _model_mtimes: dict[str, float] = {}
    for rk in modules:
        cfg = SYMBOL_CONFIG[rk]
        cfg_path = MODEL_BASE / cfg["model_dir"] / "config.json"
        if cfg_path.exists():
            _model_mtimes[rk] = cfg_path.stat().st_mtime

    # Main loop
    try:
        while not shutdown:
            time.sleep(1)
            _loop_iter += 1

            # Auto-reload: check model files every 30s for changes
            if _loop_iter % 30 == 0:
                for rk, alpha_mod in modules.items():
                    try:
                        cfg = SYMBOL_CONFIG[rk]
                        cfg_path = MODEL_BASE / cfg["model_dir"] / "config.json"
                        if not cfg_path.exists():
                            continue
                        mtime = cfg_path.stat().st_mtime
                        old_mtime = _model_mtimes.get(rk, 0)
                        if mtime > old_mtime and old_mtime > 0:
                            # Config changed — auto-reload this model
                            model_dir = MODEL_BASE / cfg["model_dir"]
                            model_info = load_model(model_dir)
                            new_predictor = EnsemblePredictor(
                                model_info["horizon_models"],
                                model_info["config"],
                            )
                            alpha_mod.update_predictor(new_predictor)
                            alpha_mod._discretizer.deadzone = model_info["deadzone"]
                            alpha_mod._discretizer.min_hold = model_info["min_hold"]
                            alpha_mod._discretizer.max_hold = model_info["max_hold"]
                            alpha_mod._discretizer._long_only = model_info.get("long_only", False)
                            logger.info(
                                "AUTO-RELOAD %s: config.json changed, model reloaded "
                                "(dz=%.1f mh=%d lo=%s)",
                                rk, model_info["deadzone"], model_info["min_hold"],
                                model_info.get("long_only", False),
                            )
                        _model_mtimes[rk] = mtime
                    except Exception:
                        logger.debug("Auto-reload check failed for %s", rk, exc_info=True)

            # Save z-score checkpoints every 300 iterations (~5min)
            if _loop_iter % 300 == 0:
                for rk, am in modules.items():
                    try:
                        _save_zscore_checkpoint(rk, am._discretizer._bridge)
                    except Exception:
                        pass

            # Real-time market monitor + intra-bar signal preview
            # Every 60s: preview z-score as if bar closed NOW at current tick price
            if _loop_iter % 60 == 0 and _rt_prices:
                for runner_key, alpha_mod in modules.items():
                    try:
                        cfg = SYMBOL_CONFIG[runner_key]
                        sym = cfg.get("symbol", runner_key)
                        if "4h" in runner_key or "15m" in runner_key:
                            continue  # only monitor 1h runners
                        price = _rt_prices.get(sym, 0)
                        if price <= 0:
                            continue
                        last_close = alpha_mod._closes[-1] if alpha_mod._closes else 0
                        if last_close <= 0:
                            continue
                        move_pct = (price / last_close - 1) * 100

                        # Position info
                        current_signal = alpha_mod._signal
                        pos_info = ""
                        if current_signal != 0:
                            entry = alpha_mod._entry_price
                            if entry > 0:
                                pnl_pct = current_signal * (price / entry - 1) * 100
                                pos_info = f" | pos={'LONG' if current_signal > 0 else 'SHORT'} pnl={pnl_pct:+.2f}%"

                        # Intra-bar signal preview: predict with current price as if bar closed
                        preview_z = ""
                        try:
                            # Build mock features with current price
                            _lf = getattr(alpha_mod, '_last_features', None)
                            features = dict(_lf) if _lf else {}
                            # Update price-derived features
                            if last_close > 0:
                                features["ret_1"] = price / last_close - 1
                            # Predict
                            pred = alpha_mod._predictor.predict(features)
                            if pred is not None:
                                # Preview z-score WITHOUT mutating the bridge state
                                bridge = alpha_mod._discretizer._bridge
                                state = bridge.checkpoint()
                                # Use a fake hour_key that won't collide with real bars
                                preview_hour = alpha_mod._bars_processed * 100 + _loop_iter
                                z_val = bridge.zscore_normalize(sym, pred, preview_hour)
                                bridge.restore(state)  # restore original state
                                if z_val is not None:
                                    z = max(-5.0, min(5.0, z_val))
                                    dz = alpha_mod._discretizer.deadzone
                                    if abs(z) > dz:
                                        direction = "BUY" if z > 0 else "SELL"
                                        preview_z = f" | PREVIEW z={z:+.2f} → {direction} ***"
                                        # Cancel any pending limit — we're going market
                                        _limit_mgr.cancel_stale(sym)
                                        # EARLY ENTRY: trigger trade if no position
                                        if current_signal == 0:
                                            # Emit a synthetic bar at current price
                                            try:
                                                early_header = EventHeader.new_root(
                                                    event_type=EventType.MARKET,
                                                    version=1,
                                                    source="early_entry",
                                                )
                                                early_event = MarketEvent(
                                                    header=early_header,
                                                    ts=datetime.now(timezone.utc),
                                                    symbol=sym,
                                                    open=Decimal(str(last_close)),
                                                    high=Decimal(str(max(price, last_close))),
                                                    low=Decimal(str(min(price, last_close))),
                                                    close=Decimal(str(price)),
                                                    volume=Decimal("0"),
                                                )
                                                coord = coordinators[runner_key]
                                                coord.emit(early_event, actor="live")
                                                logger.info(
                                                    "EARLY ENTRY %s: z=%+.2f > dz=%.1f, "
                                                    "emitted synthetic bar at $%.2f",
                                                    sym, z, dz, price,
                                                )
                                            except Exception:
                                                logger.debug("Early entry failed for %s", sym, exc_info=True)
                                    elif abs(z) > dz * 0.7:
                                        preview_z = f" | PREVIEW z={z:+.2f} (approaching dz={dz})"
                                        # PRE-PLACE limit order at favorable price
                                        if current_signal == 0 and not args.dry_run:
                                            try:
                                                limit_side = "buy" if z > 0 else "sell"
                                                # Use sizer for proper qty
                                                snap = coordinators[runner_key]._state_store
                                                limit_qty = alpha_mod._sizer.target_qty(
                                                    snap, sym,
                                                    leverage=alpha_mod._leverage,
                                                    ic_scale=alpha_mod._ic_scale,
                                                )
                                                _limit_mgr.maybe_place(
                                                    symbol=sym,
                                                    side=limit_side,
                                                    qty=float(limit_qty),
                                                    current_price=price,
                                                    z_score=z,
                                                    deadzone=dz,
                                                )
                                            except Exception:
                                                logger.debug(
                                                    "Limit pre-place failed %s",
                                                    sym, exc_info=True,
                                                )
                                    else:
                                        preview_z = f" | z={z:+.2f}"
                                        # Signal faded: cancel any pending limit
                                        if abs(z) < dz * 0.5:
                                            _limit_mgr.cancel_stale(sym)
                        except Exception:
                            pass

                        logger.info(
                            "MONITOR %s: $%.2f (move=%+.2f%%)%s%s",
                            sym, price, move_pct, pos_info, preview_z,
                        )
                    except Exception:
                        pass

            # Cancel TTL-expired limit orders every 60s
            if _loop_iter % 60 == 0:
                try:
                    expired = _limit_mgr.cancel_expired()
                    if expired > 0:
                        logger.info("Cancelled %d TTL-expired limit orders", expired)
                except Exception:
                    pass

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

            # Live Sharpe check every 3600 iterations (~1 hour)
            if _loop_iter % 3600 == 0 and pnl_tracker is not None:
                try:
                    # Feed new trades into the sharpe tracker
                    all_trades = pnl_tracker.trades
                    new_trades = all_trades[_last_pnl_trade_count:]
                    _last_pnl_trade_count = len(all_trades)
                    for t in new_trades:
                        sym = t.get("symbol", "UNKNOWN")
                        pnl_pct = t.get("pnl_pct", 0.0)
                        if pnl_pct == 0.0:
                            # Derive from USD pnl if pnl_pct not available
                            entry = t.get("entry", 0)
                            size = t.get("size", 0)
                            pnl_usd = t.get("pnl_usd", 0)
                            notional = entry * size if entry and size else 0
                            pnl_pct = (pnl_usd / notional * 100) if notional > 0 else 0.0
                        _sharpe_tracker.update(sym, pnl_pct)

                    # Check Sharpe per symbol
                    report = _sharpe_tracker.report()
                    statuses = _sharpe_tracker.status()
                    for sym, sharpe_val in report.items():
                        status = statuses.get(sym, "?")
                        logger.info("SHARPE %s: %.3f [%s]", sym, sharpe_val, status)
                        if sharpe_val < 0:
                            _sharpe_neg_streak[sym] = _sharpe_neg_streak.get(sym, 0) + 1
                        else:
                            _sharpe_neg_streak[sym] = 0

                        if _sharpe_neg_streak.get(sym, 0) >= _SHARPE_NEG_THRESHOLD:
                            send_alert(
                                AlertLevel.WARNING,
                                f"Live Sharpe negative {_sharpe_neg_streak[sym]}x: {sym}",
                                details={
                                    "symbol": sym,
                                    "sharpe": f"{sharpe_val:.3f}",
                                    "streak": str(_sharpe_neg_streak[sym]),
                                    "trades": str(len(all_trades)),
                                },
                                source="alpha_main.sharpe_monitor",
                            )
                            logger.warning(
                                "SHARPE ALERT %s: %.3f negative for %d consecutive hours",
                                sym, sharpe_val, _sharpe_neg_streak[sym],
                            )
                except Exception:
                    pass  # never crash the main loop
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down...")
        # Cancel any pending limit orders
        try:
            _limit_mgr.cancel_all_pending()
            logger.info("All pending limit orders cancelled")
        except Exception:
            logger.debug("Failed to cancel pending limit orders", exc_info=True)
        # Save z-score checkpoints for fast restart
        for rk, am in modules.items():
            try:
                _save_zscore_checkpoint(rk, am._discretizer._bridge)
            except Exception:
                pass
        logger.info("Z-score checkpoints saved for %d runners", len(modules))
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
        # Stop OI caches
        for sym, cache in oi_caches.items():
            cache.stop()
            logger.info("Stopped BinanceOICache for %s", sym)
        for ws in ws_clients:
            ws.stop()
        for coord in coordinators.values():
            coord.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
