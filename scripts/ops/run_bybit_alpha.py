#!/usr/bin/env python3
"""Run the current directional alpha strategy on Bybit.

This is the active multi-symbol alpha service used by ``bybit-alpha.service``.
The current runtime path is AlphaRunner/PortfolioManager/PortfolioCombiner.
This CLI does not switch to ``LiveRunner``.

Usage:
    python3 -m scripts.run_bybit_alpha                    # live demo trading
    python3 -m scripts.run_bybit_alpha --dry-run          # signal only, no orders
    python3 -m scripts.run_bybit_alpha --once              # single bar then exit
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

from scripts.ops.config import (
    INTERVAL, MODEL_BASE, POLL_INTERVAL, SYMBOL_CONFIG, WARMUP_BARS,
)
from scripts.ops.model_loader import create_adapter, load_model
from scripts.ops.alpha_runner import AlphaRunner
from scripts.ops.order_utils import reliable_close_position
from scripts.ops.portfolio_manager import PortfolioManager
from scripts.ops.portfolio_combiner import PortfolioCombiner
from scripts.ops.hedge_runner import HedgeRunner
from scripts.ops.runtime_ownership import claim_bybit_symbol_lease
from scripts.ops.runtime_kill_latch import (
    PersistentKillSwitch,
    STARTUP_GUARD_EXIT_CODE,
    build_bybit_kill_latch,
    require_kill_latch_clear,
)

logger = logging.getLogger(__name__)

# Shared correlation tracker — updated on each bar, read by AlphaRunner sizing
_correlation_computer = None

try:
    from _quant_hotpath import RustCorrelationComputer
    _correlation_computer = RustCorrelationComputer(window=120)  # 5-day rolling
except ImportError:
    pass

def _resolve_runner_target(
    runner_key: str,
    runner_intervals: dict | None = None,
) -> tuple[str, str]:
    if runner_intervals is None:
        return runner_key, INTERVAL
    return runner_intervals.get(runner_key, (runner_key, INTERVAL))


def _combo_entry_price(combo_trade: dict | None, fallback_price: float) -> float:
    """Prefer combiner-reported fill price when syncing PM entry truth."""
    if not combo_trade:
        return fallback_price
    fill_price = combo_trade.get("fill_price")
    try:
        return float(fill_price) if fill_price is not None else fallback_price
    except (TypeError, ValueError):
        return fallback_price


def _runtime_symbols(symbols: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    claimed: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        real_symbol = str(SYMBOL_CONFIG.get(symbol, {}).get("symbol", symbol)).upper()
        if real_symbol and real_symbol not in seen:
            seen.add(real_symbol)
            claimed.append(real_symbol)
    return tuple(claimed)


def _claim_runtime_symbols(
    adapter: Any,
    symbols: list[str] | tuple[str, ...],
    *,
    dry_run: bool,
):
    if dry_run:
        return None
    return claim_bybit_symbol_lease(
        adapter=adapter,
        service_name="bybit-alpha.service",
        symbols=_runtime_symbols(symbols),
    )


def _build_runtime_kill_latch(adapter: Any, *, dry_run: bool):
    if dry_run:
        return None
    return build_bybit_kill_latch(
        adapter=adapter,
        service_name="bybit-alpha.service",
        scope_name="portfolio",
    )


def _build_portfolio_combiners(
    runners: dict[str, AlphaRunner],
    adapter: Any,
    *,
    dry_run: bool,
    runner_intervals: dict[str, tuple[str, str]] | None = None,
) -> dict[str, PortfolioCombiner]:
    if runner_intervals is None:
        runner_intervals = {s: (s, INTERVAL) for s in runners}

    symbol_runners: dict[str, list[str]] = {}
    for runner_key, (real_symbol, _) in runner_intervals.items():
        symbol_runners.setdefault(real_symbol, []).append(runner_key)

    combo_state_store = None
    for runner in runners.values():
        if runner._state_store is not None:
            combo_state_store = runner._state_store
            break

    combiners: dict[str, PortfolioCombiner] = {}
    for real_symbol, runner_keys in symbol_runners.items():
        if len(runner_keys) <= 1:
            continue
        weights = {runner_key: 0.5 for runner_key in runner_keys}
        combiners[real_symbol] = PortfolioCombiner(
            adapter=adapter,
            symbol=real_symbol,
            weights=weights,
            threshold=0.3,
            dry_run=dry_run,
            state_store=combo_state_store,
        )
        for runner_key in runner_keys:
            runners[runner_key]._dry_run = True
        logger.info("COMBO mode: %s runners=%s weights=%s", real_symbol, runner_keys, weights)

    return combiners


def _enforce_portfolio_kill(
    runners: dict[str, AlphaRunner],
    combiners: dict[str, PortfolioCombiner],
    portfolio_manager: PortfolioManager | None,
    last_prices: dict[str, float],
) -> dict[str, dict]:
    """Flatten active alpha runtime state once the shared kill switch is armed."""
    if portfolio_manager is None or not portfolio_manager.is_killed:
        return {}

    for runner in runners.values():
        runner.force_flat_local_state()

    forced: dict[str, dict] = {}
    for symbol, combiner in combiners.items():
        price = last_prices.get(symbol)
        if price is None:
            continue
        trade = combiner.force_flat(price, reason="portfolio_killed")
        if trade is None:
            continue
        forced[symbol] = trade
        if combiner._current_position == 0:
            portfolio_manager.record_position(symbol, 0, 0, "COMBO_KILLED")
            logger.warning(
                "PM kill enforced for %s: forced combo flat at $%.2f -> %s",
                symbol, price, trade,
            )
        else:
            logger.error(
                "PM kill failed to flatten %s at $%.2f; combiner still has position=%s trade=%s",
                symbol, price, combiner._current_position, trade,
            )

    return forced


def _process_alpha_bar(
    runner_key: str,
    bar: dict,
    *,
    runners: dict[str, AlphaRunner],
    runner_intervals: dict[str, tuple[str, str]] | None,
    combiners: dict[str, PortfolioCombiner],
    last_prices: dict[str, float],
    portfolio_manager: PortfolioManager | None,
    hedge_runner: HedgeRunner | None = None,
    mode_label: str,
) -> dict:
    real_symbol, _interval = _resolve_runner_target(runner_key, runner_intervals)
    last_prices[real_symbol] = bar["close"]

    if hedge_runner is not None:
        hedge_result = hedge_runner.on_bar(real_symbol, bar["close"])
        if hedge_result and hedge_result.get("trade"):
            logger.info(
                "HEDGE %s: ratio=%.6f ma=%.6f",
                hedge_result["trade"],
                hedge_result.get("ratio", 0),
                hedge_result.get("ratio_ma", 0),
            )

    if _correlation_computer is not None:
        try:
            _correlation_computer.update(real_symbol, bar["close"])
        except Exception:
            logger.debug("correlation tracker update failed for %s", real_symbol, exc_info=True)

    runner = runners[runner_key]
    result = runner.process_bar(bar)
    forced_combo_trades = _enforce_portfolio_kill(
        runners, combiners, portfolio_manager, last_prices,
    )

    if result.get("action") != "signal":
        return result

    regime = result.get("regime", "?")
    label = runner_key if runner_key != real_symbol else real_symbol
    trade_str = ""

    if portfolio_manager is not None and portfolio_manager.is_killed:
        result["signal"] = 0
        result["hold_count"] = 0
        killed_trade = forced_combo_trades.get(real_symbol)
        if killed_trade is not None:
            trade_str = f" KILL={killed_trade}"
    elif real_symbol in combiners:
        combo_trade = combiners[real_symbol].update_signal(
            runner_key, result["signal"], result["close"],
        )
        if combo_trade:
            trade_str = f" COMBO={combo_trade}"
            if portfolio_manager is not None:
                desired = combo_trade.get("to", 0)
                if desired != 0:
                    entry_price = _combo_entry_price(combo_trade, result["close"])
                    portfolio_manager.record_position(
                        real_symbol,
                        combiners[real_symbol]._position_size * desired,
                        entry_price,
                        "COMBO",
                    )
                else:
                    portfolio_manager.record_position(real_symbol, 0, 0, "COMBO")
    elif "trade" in result:
        trade_str = f" TRADE={result['trade']}"

    z_scale = result.get("z_scale", 1.0)
    z_scale_str = f" zs={z_scale:.1f}" if z_scale != 1.0 else ""
    logger.info(
        "%s %s bar %d: $%.1f z=%+.3f sig=%d hold=%d regime=%s dz=%.3f%s%s",
        mode_label,
        label,
        result["bar"],
        result["close"],
        result["z"],
        result["signal"],
        result["hold_count"],
        regime,
        result.get("dz", 0),
        z_scale_str,
        trade_str,
    )
    return result


def _stop_runners(runners: dict[str, AlphaRunner]) -> None:
    for runner in runners.values():
        try:
            runner.stop()
        except Exception:
            logger.exception("Failed to stop runner for %s", getattr(runner, "_symbol", "?"))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bybit alpha strategy runner")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"],
                        help="Symbols to trade (default: BTCUSDT ETHUSDT)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--ws", action="store_true", help="Use WebSocket instead of REST polling")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.legacy:
        parser.error(
            "--legacy was removed; scripts.run_bybit_alpha always uses AlphaRunner. "
            "Use runner.live_runner directly for framework runtime experiments."
        )
    return args


def _run_ws_mode(runners: dict, adapter: Any, dry_run: bool,
                 runner_intervals: dict | None = None,
                 hedge_runner: HedgeRunner | None = None,
                 portfolio_manager: PortfolioManager | None = None,
                 combiners: dict[str, PortfolioCombiner] | None = None,
                 last_prices: dict[str, float] | None = None) -> None:
    """WebSocket-based event loop — processes bars on confirmed kline push.

    Supports multiple intervals (e.g. 1h + 15m) via separate WS clients.
    runner_intervals maps runner_key -> (real_symbol, interval).
    """
    from execution.adapters.bybit.ws_client import BybitWsClient

    if runner_intervals is None:
        runner_intervals = {s: (s, INTERVAL) for s in runners}

    # Group runners by interval -> separate WS clients
    interval_groups: dict[str, dict[str, str]] = {}  # interval -> {real_symbol: runner_key}
    for runner_key, (real_symbol, interval) in runner_intervals.items():
        interval_groups.setdefault(interval, {})[real_symbol] = runner_key

    if combiners is None:
        combiners = _build_portfolio_combiners(
            runners, adapter, dry_run=dry_run, runner_intervals=runner_intervals,
        )
    if last_prices is None:
        last_prices = {}

    def make_bar_handler(group: dict[str, str]):
        def on_ws_bar(symbol: str, bar: dict) -> None:
            runner_key = group.get(symbol)
            if not runner_key:
                return
            _process_alpha_bar(
                runner_key,
                bar,
                runners=runners,
                runner_intervals=runner_intervals,
                combiners=combiners,
                last_prices=last_prices,
                portfolio_manager=portfolio_manager,
                hedge_runner=hedge_runner,
                mode_label="WS",
            )
        return on_ws_bar

    def on_ws_tick(symbol: str, price: float) -> None:
        """Real-time stop-loss check — routes to all runners for this symbol."""
        for rkey, (rsym, _) in runner_intervals.items():
            if rsym == symbol:
                runner = runners.get(rkey)
                if runner:
                    runner.check_realtime_stoploss(price)

    # Start one WS client per interval
    ws_clients = []
    for interval, group in interval_groups.items():
        real_symbols = list(group.keys())
        ws = BybitWsClient(
            symbols=real_symbols, interval=interval,
            on_bar=make_bar_handler(group), on_tick=on_ws_tick, demo=True,
        )
        ws_clients.append(ws)

    logger.info(
        "Starting multi-symbol alpha (WebSocket + realtime stop): %s, dry=%s",
        list(runners.keys()), dry_run,
    )
    for ws in ws_clients:
        ws.start()

    # Capture state_store from runners for heartbeat logging
    _ws_state_store = None
    for r in runners.values():
        if r._state_store is not None:
            _ws_state_store = r._state_store
            break

    # WS stale detection: if no message for WS_STALE_THRESHOLD_S, attempt reconnect.
    # After WS_MAX_RECONNECT_ATTEMPTS consecutive failures, arm kill switch.
    WS_STALE_THRESHOLD_S = 120  # 2 minutes
    WS_MAX_RECONNECT_ATTEMPTS = 3

    try:
        while True:
            time.sleep(300)

            # ── WS health check: detect silent disconnects ──
            for ws in ws_clients:
                stale_s = ws.seconds_since_last_message
                if stale_s > WS_STALE_THRESHOLD_S:
                    logger.critical(
                        "WS STALE: no data for %.0fs (threshold=%ds), attempting reconnect",
                        stale_s, WS_STALE_THRESHOLD_S,
                    )
                    ws.stop()
                    time.sleep(2)
                    ws.start()
                    ws._reconnect_count += 1
                    if ws._reconnect_count >= WS_MAX_RECONNECT_ATTEMPTS:
                        logger.critical(
                            "WS STALE: %d consecutive reconnect failures, arming kill switch",
                            ws._reconnect_count,
                        )
                        # Find kill_switch from any runner
                        for r in runners.values():
                            if r._kill_switch is not None:
                                r._kill_switch.arm(
                                    "global", "*", "halt",
                                    f"WS stale {stale_s:.0f}s, {ws._reconnect_count} reconnects failed",
                                    source="ws_health_check",
                                )
                                break
                else:
                    ws._reconnect_count = 0  # reset on healthy check

            sigs = {s: r._current_signal for s, r in runners.items()}
            pm_status = portfolio_manager.get_status() if portfolio_manager else None
            hedge_status = hedge_runner.get_status() if hedge_runner else None
            # RustStateStore portfolio snapshot (exposure, unrealized PnL)
            store_status = None
            if _ws_state_store is not None:
                try:
                    port = _ws_state_store.get_portfolio()
                    store_status = {
                        "equity": port.total_equity,
                        "exposure": port.gross_exposure,
                        "unrealized": port.unrealized_pnl,
                        "symbols": port.symbols,
                    }
                except Exception as e:
                    logger.error("Failed to get WS state store portfolio: %s", e, exc_info=True)
            logger.info("WS HEARTBEAT sigs=%s pm=%s hedge=%s store=%s",
                        sigs, pm_status, hedge_status, store_status)
    except KeyboardInterrupt:
        logger.info("Stopped")
        for ws in ws_clients:
            ws.stop()
        if not dry_run:
            for key, runner in runners.items():
                if runner._current_signal != 0:
                    real_symbol = runner._symbol  # use real exchange symbol, not dict key
                    logger.info("Closing %s (key=%s) on exit...", real_symbol, key)
                    reliable_close_position(adapter, real_symbol, verify=False)

def main(argv: list[str] | None = None):
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    adapter = create_adapter()
    try:
        kill_latch = _build_runtime_kill_latch(adapter, dry_run=args.dry_run)
        if kill_latch is not None:
            require_kill_latch_clear(kill_latch, runtime_name="bybit-alpha.service")
    except RuntimeError as exc:
        logger.critical("Persistent runtime kill latch armed: %s", exc)
        raise SystemExit(STARTUP_GUARD_EXIT_CODE) from exc
    try:
        lease = _claim_runtime_symbols(adapter, args.symbols, dry_run=args.dry_run)
    except RuntimeError as exc:
        logger.critical("Runtime symbol ownership conflict: %s", exc)
        raise SystemExit(STARTUP_GUARD_EXIT_CODE) from exc
    bal = adapter.get_balances()
    usdt = bal.get("USDT")
    logger.info("USDT balance: %s", usdt.total if usdt else "?")

    # ── Rust components: 12/12 integrated ──
    # Active (6): RustFeatureEngine, RustInferenceBridge, RustRiskEvaluator,
    #   RustKillSwitch, RustOrderStateMachine, RustCircuitBreaker
    # Newly integrated (6): RustStateStore, rust_pipeline_apply,
    #   RustUnifiedPredictor, RustTickProcessor, RustWsClient, RustWsOrderGateway
    from _quant_hotpath import (
        RustRiskEvaluator, RustKillSwitch,
        RustStateStore,
        rust_pipeline_apply,       # noqa: F401 — atomic state updates (used internally by StateStore)
        RustUnifiedPredictor,      # noqa: F401 — requires JSON model export, our models are pickle
        RustTickProcessor,         # noqa: F401 — standalone binary hot-path, not used in Python runner
        RustWsClient,              # noqa: F401 — generic WS transport, Bybit uses own WS client
        RustWsOrderGateway,        # noqa: F401 — Binance WS-API only, Bybit uses REST orders
    )

    risk_eval = RustRiskEvaluator(max_drawdown_pct=0.15)
    kill_switch = PersistentKillSwitch(
        RustKillSwitch(),
        latch=kill_latch,
        service_name="bybit-alpha.service",
        scope_name="portfolio",
    )

    # RustStateStore: authoritative position truth across all symbols.
    # Keeps market, position, account, portfolio, and risk state on the Rust
    # heap. Updated via RustFillEvent/RustMarketEvent (zero-copy fast path).
    # Reconciliation compares store.get_position(sym) with exchange REST.
    all_symbols = []
    for s in args.symbols:
        real_sym = SYMBOL_CONFIG.get(s, {}).get("symbol", s)
        if real_sym not in all_symbols:
            all_symbols.append(real_sym)
    # Bug fix: pass real balance so total_equity = balance + unrealized_pnl is non-zero
    initial_balance_fd8 = int(float(usdt.total) * 100_000_000) if usdt else 0
    state_store = RustStateStore(all_symbols, "USDT", initial_balance_fd8)
    logger.info(
        "Rust 12/12: StateStore(symbols=%s) + RiskEvaluator(max_dd=15%%) + KillSwitch "
        "| Available but unused: RustUnifiedPredictor (requires JSON models, our models "
        "are pickle Ridge/LGBM), RustTickProcessor (standalone binary path), "
        "RustWsClient (generic transport, Bybit uses own WS client), "
        "RustWsOrderGateway (Binance WS-API only, Bybit uses REST)",
        all_symbols,
    )
    # Note: rust_pipeline_apply is the free-function equivalent of
    # StateStore.process_event() — called internally by the store's reducers.
    # We import it for completeness; direct use is not needed when using StateStore.

    # Build per-symbol runners
    runners: dict[str, AlphaRunner] = {}
    last_bar_times: dict[str, int] = {}

    runner_intervals: dict[str, tuple[str, str]] = {}  # runner_key -> (real_symbol, interval)

    for symbol in args.symbols:
        sym_cfg = SYMBOL_CONFIG.get(symbol, {"size": 0.001, "model_dir": f"{symbol}_gate_v2"})
        model_dir = MODEL_BASE / sym_cfg["model_dir"]
        if not (model_dir / "config.json").exists():
            logger.warning("No model for %s at %s, skipping", symbol, model_dir)
            continue

        model_info = load_model(model_dir)
        # Pass composite regime flag from SYMBOL_CONFIG to model_info
        if sym_cfg.get("use_composite_regime"):
            model_info["use_composite_regime"] = True
        real_symbol = sym_cfg.get("symbol", symbol)  # e.g. ETHUSDT_15m -> ETHUSDT
        interval = sym_cfg.get("interval", INTERVAL)
        warmup = sym_cfg.get("warmup", WARMUP_BARS)

        logger.info(
            "%s: model v%s, %d features, dz=%.1f, hold=%d-%d, size=%.4f",
            symbol, model_info["config"]["version"],
            len(model_info["features"]), model_info["deadzone"],
            model_info["min_hold"], model_info["max_hold"], sym_cfg["size"],
        )

        runner = AlphaRunner(
            adapter=adapter, model_info=model_info, symbol=real_symbol,
            dry_run=args.dry_run, position_size=sym_cfg["size"],
            max_qty=sym_cfg.get("max_qty", 0),
            step_size=sym_cfg.get("step", 0.01),
            risk_evaluator=risk_eval, kill_switch=kill_switch,
            state_store=state_store,
        )
        runner._runner_key = symbol  # for cross-symbol consensus scaling
        logger.info("%s: warming up %d bars...", symbol, warmup)
        runner.warmup(limit=warmup, interval=interval)
        runners[symbol] = runner
        runner_intervals[symbol] = (real_symbol, interval)
        last_bar_times[symbol] = 0

    if not runners:
        logger.error("No symbols with models. Exiting.")
        if lease is not None:
            lease.release()
        return

    combiners = _build_portfolio_combiners(
        runners, adapter, dry_run=args.dry_run, runner_intervals=runner_intervals,
    )
    last_prices: dict[str, float] = {}

    # Initialize portfolio manager (unified position + risk)
    pm = PortfolioManager(adapter, dry_run=args.dry_run,
                          risk_evaluator=risk_eval, kill_switch=kill_switch)
    logger.info("PM: PortfolioManager enabled (max_exposure=140%%, max_per_sym=30%%, max_dd=15%%, Rust risk)")

    # Hedge disabled: 8 cycles over 3 days, 0 wins, -$45 in fees, ratio
    # oscillates in 6th decimal (0.000048-0.000051) = pure noise trading.
    # Keeping code for future re-evaluation with longer MA or different basket.
    hedge = None

    if args.once:
        try:
            for symbol, runner in runners.items():
                real_symbol, interval = _resolve_runner_target(symbol, runner_intervals)
                bars = adapter.get_klines(real_symbol, interval=interval, limit=2)
                if bars:
                    result = _process_alpha_bar(
                        symbol,
                        bars[0],
                        runners=runners,
                        runner_intervals=runner_intervals,
                        combiners=combiners,
                        last_prices=last_prices,
                        portfolio_manager=pm,
                        hedge_runner=hedge,
                        mode_label="ONCE",
                    )
                    logger.info("%s: %s", symbol, json.dumps(result, default=str))
        finally:
            _stop_runners(runners)
            if lease is not None:
                lease.release()
        return

    # WebSocket mode: push-based, low latency
    if args.ws:
        try:
            _run_ws_mode(runners, adapter, args.dry_run,
                         runner_intervals=runner_intervals, hedge_runner=hedge,
                         portfolio_manager=pm, combiners=combiners,
                         last_prices=last_prices)
        finally:
            _stop_runners(runners)
            if lease is not None:
                lease.release()
        return

    logger.info(
        "Starting multi-symbol alpha (REST poll): %s, poll=%ds, dry=%s",
        list(runners.keys()), POLL_INTERVAL, args.dry_run,
    )

    heartbeat_interval = 300  # log heartbeat every 5 minutes
    last_heartbeat = time.time()
    cycle_count = 0

    try:
        while True:
            cycle_count += 1
            for symbol, runner in runners.items():
                try:
                    real_symbol, interval = _resolve_runner_target(symbol, runner_intervals)
                    bars = adapter.get_klines(real_symbol, interval=interval, limit=2)
                    if not bars:
                        continue

                    latest = bars[0]
                    bar_time = latest["time"]

                    if bar_time > last_bar_times[symbol]:
                        last_bar_times[symbol] = bar_time
                        _process_alpha_bar(
                            symbol,
                            latest,
                            runners=runners,
                            runner_intervals=runner_intervals,
                            combiners=combiners,
                            last_prices=last_prices,
                            portfolio_manager=pm,
                            hedge_runner=hedge,
                            mode_label="REST",
                        )
                except Exception:
                    logger.exception("Error processing %s", symbol)

            # Heartbeat: log status every 5 minutes
            now = time.time()
            if now - last_heartbeat >= heartbeat_interval:
                last_heartbeat = now
                sigs = {s: r._current_signal for s, r in runners.items()}
                holds = {s: r._hold_count for s, r in runners.items()}
                regimes = {s: "active" if r._regime_active else "FILTERED" for s, r in runners.items()}
                pnls = {s: f"${r._total_pnl:.2f}" for s, r in runners.items()}
                trades = {s: f"{r._win_count}/{r._trade_count}" for s, r in runners.items()}
                sizes = {s: f"{r._position_size:.2f}" for s, r in runners.items()}
                # RustStateStore portfolio snapshot
                store_info = ""
                if state_store is not None:
                    try:
                        port = state_store.get_portfolio()
                        store_info = (f" store_equity={port.total_equity}"
                                      f" exposure={port.gross_exposure}"
                                      f" unrealized={port.unrealized_pnl}")
                    except Exception as e:
                        logger.error("Failed to get state store portfolio: %s", e, exc_info=True)
                logger.info(
                    "HEARTBEAT cycle=%d sigs=%s holds=%s regimes=%s pnl=%s trades=%s size=%s%s",
                    cycle_count, sigs, holds, regimes, pnls, trades, sizes, store_info,
                )

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Stopped")
        if not args.dry_run:
            for symbol, runner in runners.items():
                if runner._current_signal != 0:
                    logger.info("Closing %s position on exit...", symbol)
                    reliable_close_position(adapter, symbol, verify=False)
    finally:
        _stop_runners(runners)
        if lease is not None:
            lease.release()


if __name__ == "__main__":
    main()
