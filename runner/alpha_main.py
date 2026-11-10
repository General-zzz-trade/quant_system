"""Framework-native alpha entry point.

Replaces the AlphaRunner-based entry point (scripts/ops/run_bybit_alpha.py)
with EngineCoordinator + DecisionBridge + ExecutionBridge + Bybit WS.

Usage::

    python3 -m runner.alpha_main --symbols BTCUSDT BTCUSDT_4h ETHUSDT ETHUSDT_4h --ws
    python3 -m runner.alpha_main --dry-run  # no execution, just signals
"""
from __future__ import annotations

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
    RustCircuitBreaker,
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
from runner.builders.alpha_builder import build_coordinator as _build_coordinator, _build_composite_gate
from shared.drawdown_guard import DrawdownGuard
from strategy.config import SYMBOL_CONFIG
from monitoring.notify import send_alert, AlertLevel
from monitoring.rolling_sharpe import RollingSharpeTracker
from runner.warmup import warmup as _warmup
from runner.alpha_cli import parse_args
from monitoring.live_ic_killswitch import is_runner_paused as _is_runner_paused

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


# OnlineRidge checkpoint paths — same dir as z-score, separate file so
# retraining can invalidate one without the other.
_ONLINE_RIDGE_CHECKPOINT_DIR = Path("data/runtime/online_ridge_checkpoints")


def _save_online_ridge_checkpoint(runner_key: str, online_ridge) -> None:
    """Save OnlineRidge RLS state (weights + P matrix + n_updates).

    Without this, every alpha_main restart throws away whatever
    incremental learning accumulated since the last weekly retrain.
    The live z-score buffer is still continuous but the raw predictions
    feeding it collapse back to the static sklearn weights, which causes
    a visible z-score discontinuity (observed 2026-04-11: pre-restart
    z=+2.33 → post-restart z=+0.73 for ETH 1h).
    """
    if online_ridge is None:
        return
    try:
        _ONLINE_RIDGE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        data = online_ridge.to_dict()
        path = _ONLINE_RIDGE_CHECKPOINT_DIR / f"{runner_key}.json"
        import json
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass  # best-effort, never blocks main loop


def _restore_online_ridge_checkpoint(runner_key: str, online_ridge) -> bool:
    """Restore OnlineRidge state from disk. Returns True if restored.

    Safe across retrain: if feature count changed, from_dict returns
    False and the caller keeps the freshly-loaded static weights.
    """
    if online_ridge is None:
        return False
    try:
        import json
        path = _ONLINE_RIDGE_CHECKPOINT_DIR / f"{runner_key}.json"
        if not path.exists():
            return False
        with open(path) as f:
            data = json.load(f)
        return online_ridge.from_dict(data)
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

    testnet = os.environ.get("BINANCE_TESTNET", "1").lower() in ("1", "true", "yes")
    if testnet:
        api_key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
        api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")
    else:
        api_key = os.environ.get("BINANCE_API_KEY", "")
        api_secret = os.environ.get("BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        env_prefix = "BINANCE_TESTNET" if testnet else "BINANCE"
        raise RuntimeError(
            f"{env_prefix}_API_KEY and {env_prefix}_API_SECRET "
            "environment variables are required for --venue binance."
        )
    config = BinanceConfig(api_key=api_key, api_secret=api_secret, testnet=testnet)
    adapter = BinanceAdapter(config)
    if not adapter.connect():
        raise RuntimeError("Failed to connect to Binance Futures")
    return adapter


def _create_okx_adapter(*, dry_run: bool = True):
    """Create OKX SWAP adapter from environment variables.

    Requires OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE. Reads optional
    OKX_BASE_URL, OKX_SIMULATED, OKX_MAX_ORDER_NOTIONAL, OKX_MAX_DAILY_NOTIONAL.
    """
    from execution.adapters.okx.adapter import OkxAdapter
    from execution.adapters.okx.config import OkxConfig

    config = OkxConfig.from_env()
    adapter = OkxAdapter(config)
    if not adapter.connect(configure_leverage=not dry_run):
        raise RuntimeError("Failed to connect to OKX SWAP")
    logger.info("OKX adapter connected: %s", repr(config))
    return adapter


# ── Main ────────────────────────────────────────────────────


def main() -> None:
    """Framework-native alpha entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    if args.check_paper_contract:
        import json
        from runner.paper_check import check_paper_contract
        print(json.dumps(check_paper_contract(), sort_keys=True, allow_nan=False))
        return
    if args.check_startup:
        import json
        from runner.startup_check import check_startup
        print(json.dumps(check_startup(), sort_keys=True))
        return

    if not args.symbols or any(key not in SYMBOL_CONFIG for key in args.symbols):
        raise RuntimeError("Startup blocked: every requested runner must have a known configuration")

    _portfolio_binding = None
    _account_ledger = None
    if not args.dry_run:
        from runner.account_reconciliation import account_ledger_preflight
        from runner.portfolio_risk import execution_risk_preflight
        _portfolio_binding = execution_risk_preflight(args.venue)
        _account_ledger = account_ledger_preflight()

    # Managed state is mandatory for execution and opt-in for paper. Reject
    # lost/mismatched journals before account access and hold the writer lease.
    managed_state = os.environ.get("RUNTIME_STATE_DIR")
    if not args.dry_run and not managed_state:
        raise RuntimeError("Execution requires persistent managed runtime state")
    if managed_state is not None:
        import atexit
        from shared.runtime_state import RuntimeStateLease, validate_execution_state
        if not managed_state:
            raise RuntimeError("RUNTIME_STATE_DIR must not be empty")
        state_dir = Path(managed_state)
        if state_dir.absolute() != Path("data/runtime").absolute():
            raise RuntimeError("Managed state must mount the runner's data/runtime directory")
        expected_drawdown = state_dir / "drawdown.sqlite"
        actual_drawdown = Path(os.environ.get("DRAWDOWN_STATE_PATH", "data/runtime/drawdown.sqlite"))
        if actual_drawdown.absolute() != expected_drawdown.absolute() or actual_drawdown.is_symlink():
            raise RuntimeError("Drawdown state must use the managed runtime directory")
        lease = RuntimeStateLease(
            state_dir, os.environ["RISK_SCOPE"],
            journal_path=Path(os.environ.get("ORDER_JOURNAL_PATH", "data/runtime/orders.sqlite")),
        )
        if not args.dry_run:
            try:
                validate_execution_state(state_dir, os.environ["RISK_SCOPE"], journal_path=lease.journal_path)
            except BaseException:
                lease.close()
                raise
        atexit.register(lease.close)

    # Online input is required. Never silently replace missing OI/LS data.
    # Resolve it before constructing an adapter or opening any account connection.
    try:
        from data.oi_cache import BinanceOICache
    except ModuleNotFoundError as exc:
        if exc.name != "data.oi_cache":
            raise
        raise RuntimeError(
            "Online startup blocked: required module data.oi_cache is unavailable; "
            "--check-startup validates offline engine wiring only"
        ) from exc

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
    elif args.venue == "okx":
        adapter = _create_okx_adapter(dry_run=args.dry_run)
        logger.info("OKX adapter connected")
    else:
        adapter = create_adapter()
        logger.info("Bybit adapter connected")

    # Circuit breaker: blocks orders after repeated exchange failures
    _circuit_breaker = RustCircuitBreaker()

    # Manual kill switch blocks new exposure at the execution boundary
    _kill_switch = RustKillSwitch()

    # Shared across all coordinators in this account, including rate limits.
    _drawdown = None
    _risk_gate = None
    if not args.dry_run:
        _drawdown = DrawdownGuard(
            Path(os.environ.get("DRAWDOWN_STATE_PATH", "data/runtime/drawdown.sqlite")),
            os.environ["RISK_SCOPE"],
            daily_pct=float(os.environ.get("MAX_DAILY_DRAWDOWN_PCT", "3.0")),
            weekly_pct=float(os.environ.get("MAX_WEEKLY_DRAWDOWN_PCT", "10.0")),
        )
        _risk_gate = _build_composite_gate(
            kill_switch=_kill_switch, venue=args.venue, drawdown_guard=_drawdown,
            reservation=_portfolio_binding, account_ledger=_account_ledger,
            paused=lambda sym: any(
                _is_runner_paused(rk) for rk in args.symbols
                if SYMBOL_CONFIG.get(rk, {}).get("symbol", rk) == sym
                and "4h" not in rk and "15m" not in rk
            ),
        )

    def _refresh_equity() -> None:
        if _drawdown is None:
            return
        try:
            allowed, reason = _drawdown.observe(adapter.get_account_equity())
            if not allowed:
                logger.warning("New entries paused: %s", reason)
        except Exception:
            _drawdown.invalidate("exchange equity unavailable")
            logger.warning("Equity refresh failed; entries blocked", exc_info=True)

    _refresh_equity()  # before any coordinator can send an order

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
            circuit_breaker=_circuit_breaker,
            kill_switch=_kill_switch,
            risk_gate=_risk_gate,
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

            def _restore_pnl_position(symbol, position):
                if position.qty:
                    _pnl_positions[symbol] = {
                        "side": 1 if position.qty > 0 else -1,
                        "qty": float(abs(position.qty)), "entry_price": float(position.entry_price),
                    }
                else:
                    _pnl_positions.pop(symbol, None)

            # Durable execution observers run after the receipt commits.
            from engine.dispatcher import Route
            for coord in coordinators.values():
                consumer = getattr(coord, "_durable_fill_consumer", None)
                if consumer is not None:
                    consumer.observers.append(_on_fill_for_pnl)
                    consumer.restore_observers.append(_restore_pnl_position)
                else:
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

    # Seed cross-symbol close prices so dominance features work during warmup.
    # Adapters return a dict: {"lastPrice": float, ...}. Earlier code did
    # `getattr(ticker, "last_price", 0)` which silently returned 0 for every
    # call (dicts don't have attribute access), so seeding never worked on
    # any venue. Use dict access with the real key name instead.
    try:
        from engine.feature_hook import _last_closes
        for sym in ["BTCUSDT", "ETHUSDT"]:
            try:
                ticker = adapter.get_ticker(sym)
            except (AttributeError, NotImplementedError):
                continue  # adapter may not implement get_ticker
            if not ticker:
                continue
            price = float(ticker.get("lastPrice", 0)) if isinstance(ticker, dict) \
                else float(getattr(ticker, "last_price", 0) or 0)
            if price > 0:
                _last_closes[sym] = price
                logger.info("Seeded %s close=%.2f for dominance", sym, price)
    except Exception:
        logger.debug("Cross-symbol close seeding failed (non-fatal)", exc_info=True)

    # Mark session boundary in audit log — append a separator instead of
    # truncating, so previous entries (today's trades) survive restarts.
    try:
        from monitoring.decision_audit import audit_path_for
        import json as _json
        audit_path = audit_path_for(args.venue)
        with open(audit_path, "a") as _af:
            _af.write(_json.dumps({
                "ts": time.time(), "type": "session_start",
                "venue": args.venue, "pid": os.getpid(),
            }) + "\n")
        logger.info("Session marker appended to %s", audit_path.name)
    except Exception:
        pass

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
        count = _warmup(coord, adapter, symbol, interval, warmup_limit)
        if count < warmup_limit:
            raise RuntimeError(f"Warmup incomplete for {runner_key}: {count}/{warmup_limit} bars")
        return runner_key

    t0_warmup = time.monotonic()
    warmup_failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_warmup_one, rk, co): rk for rk, co in coordinators.items()}
        for fut in as_completed(futures):
            try:
                rk = fut.result()
                logger.debug("Warmup done: %s", rk)
            except Exception:
                warmup_failures.append(futures[fut])
                logger.exception("Warmup failed for %s", futures[fut])
    if warmup_failures:
        for cache in oi_caches.values():
            cache.stop()
        raise RuntimeError("Startup blocked: incomplete warmup for " + ", ".join(warmup_failures))
    for runner_key, coord in coordinators.items():
        coord._data_readiness.set_live()
        if "4h" not in runner_key and "15m" not in runner_key:
            coord._data_readiness.set_external_status("batch_inputs", False, "awaiting a validated closed bar")
    logger.info("Parallel warmup complete: %d runners in %.1fs",
                len(coordinators), time.monotonic() - t0_warmup)

    # RE-LOAD batch z-score checkpoints AFTER warmup to override
    # the incorrect incremental predictions that warmup pushed into the buffer.
    # DO NOT save checkpoints here — preserve the batch-synced version.
    for runner_key, alpha_mod in modules.items():
        try:
            bridge = alpha_mod._discretizer._bridge
            if _restore_zscore_checkpoint(runner_key, bridge):
                logger.info("Re-loaded batch z-score checkpoint for %s (post-warmup)", runner_key)
        except Exception:
            pass
        # Restore OnlineRidge RLS state — preserves drifted weights + P
        # matrix across restarts so z-score buffer stays continuous and
        # online learning doesn't reset to n_updates=0 every launch.
        try:
            _or = getattr(alpha_mod._predictor, "_online_ridge", None)
            if _or is not None and _restore_online_ridge_checkpoint(runner_key, _or):
                logger.info(
                    "Re-loaded OnlineRidge checkpoint for %s: n_updates=%d drift=%.4f",
                    runner_key, _or.n_updates, _or.weight_drift,
                )
        except Exception:
            logger.debug("OnlineRidge restore failed for %s", runner_key, exc_info=True)
    # NOTE: intentionally NOT saving z-score checkpoints here.
    # The batch-synced checkpoint must survive until next sync_zscore_from_batch.py run.

    # Re-enable audit logging after warmup
    for alpha_mod in modules.values():
        alpha_mod._audit_enabled = True

    # Reset alpha module signal state after warmup so live starts clean
    for runner_key, alpha_mod in modules.items():
        alpha_mod._pending_orders.clear()
        alpha_mod._pending_reductions.clear()
        alpha_mod._pending_exit_reasons.clear()
        alpha_mod._signal = 0
        alpha_mod._current_qty = __import__("decimal").Decimal("0")
        alpha_mod._entry_price = 0.0
        alpha_mod._trade_peak = 0.0
        alpha_mod._entry_bar = 0   # no active entry after reset
        alpha_mod._last_trade_bar = alpha_mod._bars_processed  # cooldown from warmup end
        alpha_mod._consecutive_stops = 0   # warmup stops must not leak into live
        alpha_mod._stop_pause_until = 0    # clear any warmup-triggered pause
        # Reset Rust-side hold counter via discretizer bridge
        try:
            alpha_mod._discretizer._bridge.reset_hold(alpha_mod._symbol)
        except Exception:
            pass  # bridge may not support reset_hold
    logger.info("Post-warmup reset: all alpha modules signal=0, hold counters reset")

    # Restore committed fill state before consulting the exchange or enabling entries.
    from runner.execution_recovery import reconcile_execution
    for rk, coord in coordinators.items():
        if getattr(coord, "_durable_fill_consumer", None) is not None:
            reconcile_execution(coord, modules[rk], restore=True)

    def _refresh_portfolio() -> None:
        if _portfolio_binding is None:
            return
        store = _portfolio_binding.store
        # Multi-account groups require the dedicated collector with all accounts.
        # A partial local observation must never overwrite its complete baseline.
        if len(store.registrations) != 1:
            return
        from runner.portfolio_risk import refresh_portfolio
        try:
            refresh_portfolio(store, {os.environ["RISK_SCOPE"]: (adapter, Path(managed_state))})
        except Exception:
            logger.exception("Portfolio reconciliation failed; new exposure blocked")

    _refresh_portfolio()

    def _refresh_wallet() -> None:
        if _account_ledger is None:
            return
        from runner.account_reconciliation import reconcile_bybit_wallet
        try:
            reconcile_bybit_wallet(_account_ledger, adapter)
        except Exception:
            logger.exception("Wallet reconciliation failed; new exposure blocked")

    _refresh_wallet()

    # Sync positions from exchange — if exchange has an open position,
    # restore alpha module state so stop-loss/quick_loss work correctly.
    #
    # Earlier this loop did `if qty <= 0: continue`, but `pos.qty` is signed
    # (positive=long, negative=short). That filter silently skipped EVERY
    # short position, so on restart the runner would start with am._signal=0
    # while the exchange held a short, and the next strong signal would open
    # a fresh short on top — turning a 1.59-contract intended short into a
    # 9.7-contract actual short (2026-04-18 ETH incident).
    try:
        exchange_positions = adapter.get_positions()
        if exchange_positions:
            for pos in exchange_positions:
                sym = pos.symbol
                qty_signed = float(pos.qty)
                entry = float(pos.entry_price)
                if abs(qty_signed) < 1e-8 or entry <= 0:
                    continue
                pos_signal = 1 if qty_signed > 0 else -1
                qty_abs = abs(qty_signed)
                # Find matching 1h runner (not 4h — exchange position is shared)
                for rk, am in modules.items():
                    if getattr(coordinators[rk], "_durable_fill_consumer", None) is not None:
                        continue
                    cfg = SYMBOL_CONFIG[rk]
                    if cfg.get("symbol", rk) == sym and "4h" not in rk and "15m" not in rk:
                        am._signal = pos_signal
                        am._entry_price = entry
                        am._trade_peak = entry
                        am._current_qty = Decimal(str(qty_abs))
                        am._last_trade_bar = am._bars_processed  # treat sync as trade start for max_hold
                        # Critical: set _entry_bar so min-hold guard (alpha.py:497)
                        # protects the synced position from immediate signal_change
                        # exit. Post-warmup z-score window is cold → fresh |z| often
                        # drops below dz for first few bars; without this the sync'd
                        # position would be liquidated on the very next bar.
                        am._entry_bar = am._bars_processed
                        mh_protect = int(getattr(am._discretizer, "min_hold", 6))
                        logger.info(
                            "POSITION SYNC %s: qty=%+.4f entry=%.2f → "
                            "pos_signal=%+d (entry_bar=%d, protected %d bars)",
                            sym, qty_signed, entry, pos_signal,
                            am._bars_processed, mh_protect,
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
    # synced via POSITION SYNC above — preserve their _last_trade_bar.
    for runner_key, alpha_mod in modules.items():
        if alpha_mod._signal == 0:
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

    # Entries originate only from confirmed bars through ExecutionBridge.

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
                # Match the auto-reload path: discretizer params can change
                # alongside the model (a retrain may shift dz/min_hold for
                # the new IC profile). SIGHUP path used to skip this and
                # leave the discretizer at the OLD thresholds until the
                # next 30s auto-reload tick caught up.
                alpha_mod._discretizer.deadzone = model_info["deadzone"]
                alpha_mod._discretizer.min_hold = model_info["min_hold"]
                alpha_mod._discretizer.max_hold = model_info["max_hold"]
                alpha_mod._discretizer._long_only = model_info.get("long_only", False)
                reloaded += 1
                logger.info(
                    "Reloaded model for %s (dz=%.1f mh=%d lo=%s)",
                    runner_key, model_info["deadzone"], model_info["min_hold"],
                    model_info.get("long_only", False),
                )
            except Exception:
                failed += 1
                logger.exception(
                    "Failed to reload model for %s — keeping current", runner_key,
                )
        # Resync z-score buffers after model change — prevents z-score
        # discontinuity when new model produces different prediction scale.
        # Runs as subprocess to avoid ~400MB in-process memory spike (OOM fix).
        if reloaded > 0:
            try:
                import subprocess as _sp
                _sp.Popen(
                    [sys.executable, "-m", "scripts.sync_zscore_from_batch"],
                    cwd="/quant_system",
                    stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                )
                logger.info("Z-score buffers resynced after model reload")
            except Exception:
                logger.warning("Z-score resync failed — buffers may be stale",
                               exc_info=True)

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "Hot-reload complete: %d reloaded, %d failed, %.1fms",
            reloaded, failed, elapsed_ms,
        )

    signal.signal(signal.SIGHUP, _handle_sighup)

    # Live funding rate cache — updated by WS ticker, consumed by feature hooks
    _live_funding: dict[str, float] = {}
    # Track consecutive emit failures per runner for alerting
    _emit_fail_count: dict[str, int] = {}
    # Cached exchange position snapshot — refreshed by reconciliation loop,
    # checked on each bar to detect divergence early (no extra REST calls).
    _exchange_pos_cache: dict[str, float] = {}  # symbol → signed qty

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

        # Invalidate batch prediction cache for this symbol (new bar = new data)
        try:
            from runner.batch_predictor import invalidate_cache
            invalidate_cache(ws_symbol)
        except Exception:
            pass

        # Route to all coordinators that handle this symbol + interval
        bar_interval = str(bar.get("interval", "60"))
        for runner_key, coord in coordinators.items():
            cfg = SYMBOL_CONFIG[runner_key]
            runner_symbol = cfg.get("symbol", runner_key)
            runner_interval = cfg.get("interval", "60")
            if runner_symbol == ws_symbol and runner_interval == bar_interval:
                # Inject batch prediction BEFORE emit so decide() uses it
                # instead of incremental features (prevents z-score buffer contamination)
                if "4h" not in runner_key and "15m" not in runner_key:
                    try:
                        from runner.batch_predictor import predict_latest
                        model_dir = cfg.get("model_dir", runner_key)
                        batch_pred = predict_latest(ws_symbol, model_dir, as_of_ms=int(ts_ms))
                        coord._data_readiness.set_external_status(
                            "batch_inputs", batch_pred is not None,
                            "batch prediction inputs or model unavailable" if batch_pred is None else "",
                        )
                        if batch_pred is not None:
                            am = modules.get(runner_key)
                            if am is not None:
                                am._batch_pred_override = batch_pred
                    except Exception:
                        coord._data_readiness.set_external_status("batch_inputs", False, "batch validation failed")

                # Entry pause is checked at execution; bars and exits keep flowing.
                try:
                    coord.emit(event, actor="live")
                    _emit_fail_count.pop(runner_key, None)
                except Exception:
                    _emit_fail_count[runner_key] = _emit_fail_count.get(runner_key, 0) + 1
                    logger.error(
                        "Error emitting bar to %s (consecutive=%d)",
                        runner_key, _emit_fail_count[runner_key], exc_info=True,
                    )

                # Bar-level divergence check: module vs cached exchange position.
                # No REST call — uses cache refreshed every 5 min by reconciliation.
                if "4h" not in runner_key and "15m" not in runner_key:
                    am = modules.get(runner_key)
                    if am is not None:
                        ex_qty = _exchange_pos_cache.get(ws_symbol, 0.0)
                        mod_signal = am._signal
                        # system=flat but exchange=position → will be caught by reconciliation
                        # system=position but exchange=flat → order may have been rejected
                        if mod_signal != 0 and abs(ex_qty) == 0 and am._bars_processed > 810:
                            logger.warning(
                                "DIVERGENCE %s: module=%+d qty=%.4f but exchange cache=FLAT",
                                ws_symbol, mod_signal, float(am._current_qty),
                            )

    # WebSocket setup
    ws_clients: list[BybitWsClient] = []
    _rt_prices: dict[str, float] = {}
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

        # Tick prices are observation only; never manufacture a closed candle.
        def _on_tick(ws_symbol: str, price: float) -> None:
            _rt_prices[ws_symbol] = price

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
            elif args.venue == "okx":
                from execution.adapters.okx.ws_kline_client import OkxWsClient
                ws = OkxWsClient(
                    symbols=symbols,
                    interval=interval,
                    on_bar=_on_bar,
                    on_tick=_on_tick,
                    demo=os.environ.get("OKX_SIMULATED", "0").lower()
                        in ("1", "true", "yes"),
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

    _loop_iter = 0

    # Live Sharpe tracker — alerts when Sharpe < 0 for 3 consecutive checks
    _sharpe_tracker = RollingSharpeTracker(window=720, interval="1h")
    _sharpe_neg_streak: dict[str, int] = {}  # symbol -> consecutive negative count
    _SHARPE_NEG_THRESHOLD = 3  # alert after N consecutive negative checks
    _last_pnl_trade_count: int = 0  # track how many trades we've already fed

    # Reconcile alert dedup: only alert once per (sym, state) per hour
    _last_recon_alert: dict[str, tuple[str, float]] = {}  # sym -> (state_key, ts)
    _RECON_ALERT_COOLDOWN_S = 3600  # 1 hour

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

            # Z-score checkpoint save DISABLED — batch-synced checkpoints
            # must not be overwritten by incremental predictions.
            # Run scripts/sync_zscore_from_batch.py to update checkpoints.

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

                        # Intra-bar signal preview using BATCH prediction
                        # (incremental features diverge ~100x, making preview unreliable)
                        preview_z = ""
                        try:
                            from runner.batch_predictor import predict_latest
                            cfg_m = SYMBOL_CONFIG.get(runner_key, {})
                            model_dir = cfg_m.get("model_dir", runner_key)
                            batch_pred = predict_latest(sym, model_dir)
                            if batch_pred is not None:
                                bridge = alpha_mod._discretizer._bridge
                                state = bridge.checkpoint()
                                preview_hour = alpha_mod._bars_processed * 100 + _loop_iter
                                z_val = bridge.zscore_normalize(sym, batch_pred, preview_hour)
                                bridge.restore(state)
                                if z_val is not None:
                                    z = max(-5.0, min(5.0, z_val))
                                    dz = alpha_mod._discretizer.deadzone
                                    if abs(z) > dz:
                                        direction = "BUY" if z > 0 else "SELL"
                                        preview_z = f" | PREVIEW z={z:+.2f} → {direction} ***"
                                    elif abs(z) > dz * 0.85:
                                        preview_z = f" | PREVIEW z={z:+.2f} (approaching dz={dz})"
                                    else:
                                        preview_z = f" | z={z:+.2f}"
                        except Exception:
                            pass

                        logger.info(
                            "MONITOR %s: $%.2f (move=%+.2f%%)%s%s",
                            sym, price, move_pct, pos_info, preview_z,
                        )
                    except Exception:
                        pass

            # Reconcile delayed fills through the same dispatcher, including
            # while entry risk gates are closed. No retry of order placement.
            if _loop_iter % 5 == 0 and not args.dry_run:
                for rk, coord in coordinators.items():
                    if getattr(coord, "_durable_fill_consumer", None) is not None:
                        try:
                            reconcile_execution(coord, modules[rk])
                        except Exception:
                            modules[rk]._execution_ready = False
                            logger.exception("Pending fill reconciliation failed; entries blocked")

            if _loop_iter % 60 == 0:
                _refresh_equity()
                _refresh_portfolio()
                _refresh_wallet()

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

            # Position reconciliation every 60 iterations (~1 min).
            # Exchange is the single source of truth for position state.
            # This loop detects divergence between module state and exchange,
            # syncing module ← exchange when they disagree.
            #
            # Tightened from 5 min → 1 min on 2026-04-18 after the audit/fill
            # reconcile audit found multi-hour drift windows post-incident
            # (audit_fill_reconcile.py reports). Tighter cadence trades 5
            # extra REST calls per minute for << 1 minute of drift exposure.
            if _loop_iter % 60 == 0 and _loop_iter > 0:
                try:
                    exchange_positions = adapter.get_positions()
                    ex_map: dict[str, float] = {}
                    for pos in exchange_positions:
                        if not pos.is_flat:
                            # pos.qty is already signed (positive=long, negative=short).
                            # An earlier version did `-float(pos.qty)` for shorts,
                            # which sign-flipped them in the cache and triggered
                            # spurious DIRECTION FLIP alerts on every short trade.
                            ex_map[pos.symbol] = float(pos.qty)
                    _exchange_pos_cache.clear()
                    _exchange_pos_cache.update(ex_map)
                    for rk, am in modules.items():
                        if getattr(coordinators[rk], "_durable_fill_consumer", None) is not None:
                            continue  # durable ledger never heals by inventing fills
                        cfg_r = SYMBOL_CONFIG.get(rk, {})
                        sym = cfg_r.get("symbol", rk)
                        if "4h" in rk or "15m" in rk:
                            continue
                        if am._signal != 0:
                            internal_qty = float(am._current_qty) * (1 if am._signal > 0 else -1)
                        else:
                            internal_qty = 0.0
                        exchange_qty = ex_map.get(sym, 0.0)

                        # Best price: prefer tick (realtime) > bar close > REST
                        _mkt_price = _rt_prices.get(sym, 0.0)
                        if _mkt_price <= 0:
                            from engine.feature_hook import _last_closes
                            _mkt_price = _last_closes.get(sym, 0.0)
                        if _mkt_price <= 0:
                            try:
                                _tk = adapter.get_ticker(sym)
                                _mkt_price = (
                                    float(_tk.get("lastPrice", 0))
                                    if isinstance(_tk, dict)
                                    else float(getattr(_tk, "last_price", 0) or 0)
                                )
                            except Exception:
                                _mkt_price = 0.0

                        # Detect dust: exchange has position but system thinks flat
                        if am._signal == 0 and abs(exchange_qty) > 0:
                            if _mkt_price <= 0:
                                # Cannot determine price — sync as real position
                                logger.warning(
                                    "RECONCILE %s: no market price for dust check, syncing %.4f",
                                    sym, exchange_qty,
                                )
                                am._signal = 1 if exchange_qty > 0 else -1
                                am._current_qty = Decimal(str(abs(exchange_qty)))
                                am._last_trade_bar = am._bars_processed
                                continue
                            logger.warning("RECONCILE %s: syncing exchange position %.4f", sym, exchange_qty)
                            am._signal = 1 if exchange_qty > 0 else -1
                            am._current_qty = Decimal(str(abs(exchange_qty)))
                            am._entry_price = _mkt_price
                            am._trade_peak = _mkt_price
                            am._last_trade_bar = am._bars_processed
                        # Detect: system thinks has position but exchange is flat
                        elif am._signal != 0 and abs(exchange_qty) == 0:
                            logger.warning(
                                "RECONCILE %s: system=%+d qty=%.4f but exchange=FLAT — resetting to flat",
                                sym, am._signal, float(am._current_qty),
                            )
                            am._signal = 0
                            am._current_qty = Decimal("0")
                            am._entry_price = 0.0
                            am._trade_peak = 0.0
                        # Detect qty mismatch: both have position but different size
                        elif am._signal != 0 and abs(exchange_qty) > 0:
                            diff_pct = abs(abs(internal_qty) - abs(exchange_qty)) / max(abs(exchange_qty), 1e-8) * 100
                            if diff_pct > 10:
                                logger.warning(
                                    "RECONCILE %s: qty mismatch internal=%.4f exchange=%.4f (%.1f%%)",
                                    sym, internal_qty, exchange_qty, diff_pct,
                                )
                                old_internal_qty = float(am._current_qty)
                                am._current_qty = Decimal(str(abs(exchange_qty)))
                                # Audit the drift correction so audit_fill_reconcile.py
                                # accounts for the silent state change. Without this,
                                # downstream PnL trackers think the position was
                                # always at the corrected size.
                                if getattr(am, "_audit_enabled", False):
                                    try:
                                        am._audit.log_entry(
                                            symbol=sym,
                                            side="reconcile",
                                            qty=float(abs(exchange_qty)),
                                            price=_mkt_price if _mkt_price > 0 else 0.0,
                                            reason="reconcile_qty_correction",
                                            runner_key=rk,
                                            old_internal_qty=old_internal_qty,
                                            new_qty=float(abs(exchange_qty)),
                                            diff_pct=round(diff_pct, 2),
                                        )
                                    except Exception:
                                        pass
                                # Serious (>20%) divergence deserves a Telegram
                                # alert — prevents silent state drift on real money.
                                # Dedup: only alert once per state per hour.
                                if diff_pct > 20:
                                    state_key = (
                                        f"DIV_{sym}_{int(internal_qty*100)}"
                                        f"_{int(exchange_qty*100)}"
                                    )
                                    last = _last_recon_alert.get(sym)
                                    cooldown_ok = (
                                        last is None
                                        or last[0] != state_key
                                        or (time.time() - last[1]) > _RECON_ALERT_COOLDOWN_S
                                    )
                                    if cooldown_ok:
                                        try:
                                            send_alert(
                                                AlertLevel.WARNING,
                                                f"Position DIVERGENCE on {sym}",
                                                details={
                                                    "venue": getattr(adapter, "venue", "?"),
                                                    "symbol": sym,
                                                    "internal_qty": f"{internal_qty:+.4f}",
                                                    "exchange_qty": f"{exchange_qty:+.4f}",
                                                    "diff_pct": f"{diff_pct:.1f}%",
                                                },
                                                source="reconcile",
                                            )
                                            _last_recon_alert[sym] = (state_key, time.time())
                                        except Exception:
                                            pass
                        # Direction flip: system says LONG but exchange says SHORT
                        # (or vice versa) — critical bug, immediate alert.
                        if (am._signal != 0 and abs(exchange_qty) > 0
                                and (am._signal > 0) != (exchange_qty > 0)):
                            logger.error(
                                "RECONCILE %s: DIRECTION FLIP internal=%+d exchange=%+.4f — forcing flat",
                                sym, am._signal, exchange_qty,
                            )
                            state_key = (
                                f"FLIP_{sym}_{am._signal}"
                                f"_{1 if exchange_qty > 0 else -1}"
                            )
                            last = _last_recon_alert.get(sym)
                            cooldown_ok = (
                                last is None
                                or last[0] != state_key
                                or (time.time() - last[1]) > _RECON_ALERT_COOLDOWN_S
                            )
                            if cooldown_ok:
                                try:
                                    send_alert(
                                        AlertLevel.CRITICAL,
                                        f"DIRECTION FLIP on {sym}",
                                        details={
                                            "venue": getattr(adapter, "venue", "?"),
                                            "symbol": sym,
                                            "internal_signal": str(am._signal),
                                            "exchange_qty": f"{exchange_qty:+.4f}",
                                        },
                                        source="reconcile",
                                    )
                                    _last_recon_alert[sym] = (state_key, time.time())
                                except Exception:
                                    pass
                except Exception:
                    logger.warning("Position reconciliation failed", exc_info=True)

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
        # Best-effort final reconciliation/cancellation; unresolved journal
        # reservations survive shutdown and continue to block new exposure.
        for coord in coordinators.values():
            reconcile = getattr(getattr(coord._execution_bridge, "adapter", None), "reconcile_pending", None)
            if reconcile is not None:
                try:
                    for fill in reconcile():
                        coord.emit(fill, actor="execution_reconcile")
                except Exception:
                    logger.exception("Shutdown reconciliation incomplete")
        # Z-score checkpoint save DISABLED on shutdown — batch-synced
        # checkpoints preserved. Use sync_zscore_from_batch.py to update.
        logger.info("Z-score checkpoints preserved (batch-synced)")

        # OnlineRidge checkpoint SAVE — preserves drifted RLS weights
        # + P matrix so next restart continues online learning instead
        # of resetting to the static sklearn coefficients (which was
        # causing the observed post-restart z-score discontinuity).
        for runner_key, alpha_mod in modules.items():
            try:
                _or = getattr(alpha_mod._predictor, "_online_ridge", None)
                if _or is not None:
                    _save_online_ridge_checkpoint(runner_key, _or)
            except Exception:
                logger.debug("OnlineRidge save failed for %s",
                             runner_key, exc_info=True)
        logger.info("OnlineRidge checkpoints saved")
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
