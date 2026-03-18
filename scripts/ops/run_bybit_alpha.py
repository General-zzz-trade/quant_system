#!/usr/bin/env python3
"""Run alpha strategy on Bybit demo trading.

Connects to Bybit demo API, fetches 1h klines, computes features via
RustFeatureEngine, runs LightGBM inference, and trades BTCUSDT perpetual.

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

logger = logging.getLogger(__name__)


def _select_runner_class(legacy: bool = False):
    """Return AlphaRunner (legacy) or LiveRunner (default) class."""
    if legacy:
        from scripts.ops.alpha_runner import AlphaRunner  # noqa: F811
        return AlphaRunner
    from runner.live_runner import LiveRunner
    return LiveRunner


def _build_live_config(symbols, ws=False, dry_run=False):
    """Map SYMBOL_CONFIG entries to a LiveRunnerConfig instance."""
    from runner.config import LiveRunnerConfig

    composite_symbols = tuple(
        s for s in symbols
        if SYMBOL_CONFIG.get(s, {}).get("use_composite_regime", False)
    )

    multi_interval: dict = {}
    for s in symbols:
        sc = SYMBOL_CONFIG.get(s, {})
        if sc.get("interval") == "15":
            base = sc.get("symbol", s)
            if base not in multi_interval:
                multi_interval[base] = ["60"]
            multi_interval[base].append("15")

    # Map multi-interval symbols to LiveRunnerConfig's multi_tf_models format
    multi_tf_models = None
    if multi_interval:
        multi_tf_models = {sym: intervals for sym, intervals in multi_interval.items()}

    # Enable regime sizing for symbols that use composite regime (e.g. BTC)
    enable_regime_sizing = len(composite_symbols) > 0

    return LiveRunnerConfig(
        symbols=tuple(symbols),
        shadow_mode=dry_run,
        enable_regime_sizing=enable_regime_sizing,
        enable_multi_tf_ensemble=bool(multi_tf_models),
        multi_tf_models=multi_tf_models,
    )


def _run_ws_mode(runners: dict, adapter: Any, dry_run: bool,
                 runner_intervals: dict | None = None,
                 hedge_runner: HedgeRunner | None = None,
                 portfolio_manager: PortfolioManager | None = None) -> None:
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

    # Build portfolio combiner for symbols with multiple alphas
    # Group runner_keys by real_symbol
    symbol_runners: dict[str, list[str]] = {}
    for rkey, (rsym, _) in runner_intervals.items():
        symbol_runners.setdefault(rsym, []).append(rkey)

    # Get state_store from any runner (all share the same instance)
    _combo_state_store = None
    for r in runners.values():
        if r._state_store is not None:
            _combo_state_store = r._state_store
            break

    combiners: dict[str, PortfolioCombiner] = {}
    for rsym, rkeys in symbol_runners.items():
        if len(rkeys) > 1:
            # Multiple alphas on same symbol -> use combiner
            weights = {k: 0.5 for k in rkeys}
            combiners[rsym] = PortfolioCombiner(
                adapter=adapter, symbol=rsym, weights=weights,
                threshold=0.3, dry_run=dry_run,
                state_store=_combo_state_store,
            )
            # Disable direct trading in individual runners
            for rk in rkeys:
                runners[rk]._dry_run = True  # signals only, combiner executes
            logger.info("COMBO mode: %s runners=%s weights=%s", rsym, rkeys, weights)

    def make_bar_handler(group: dict[str, str]):
        def on_ws_bar(symbol: str, bar: dict) -> None:
            # Feed hedge runner (all 1h symbols)
            if hedge_runner is not None:
                hr = hedge_runner.on_bar(symbol, bar["close"])
                if hr and hr.get("trade"):
                    logger.info("HEDGE %s: ratio=%.6f ma=%.6f",
                                hr["trade"], hr.get("ratio", 0), hr.get("ratio_ma", 0))

            runner_key = group.get(symbol)
            if not runner_key:
                return
            runner = runners.get(runner_key)
            if not runner:
                return
            result = runner.process_bar(bar)
            if result.get("action") == "signal":
                regime = result.get("regime", "?")
                label = runner_key if runner_key != symbol else symbol
                trade_str = ""

                # Route signal through combiner if available
                real_sym = runner_intervals.get(runner_key, (symbol, ""))[0]
                if real_sym in combiners:
                    combo_trade = combiners[real_sym].update_signal(
                        runner_key, result["signal"], result["close"],
                    )
                    if combo_trade:
                        trade_str = f" COMBO={combo_trade}"
                        # Sync PM position tracking after COMBO fill
                        if portfolio_manager is not None:
                            desired = combo_trade.get("to", 0)
                            if desired != 0:
                                portfolio_manager.record_position(
                                    real_sym,
                                    combiners[real_sym]._position_size * desired,
                                    result["close"],
                                    "COMBO",
                                )
                            else:
                                portfolio_manager.record_position(real_sym, 0, 0, "COMBO")
                elif "trade" in result:
                    trade_str = f" TRADE={result['trade']}"

                z_sc = result.get("z_scale", 1.0)
                z_sc_str = f" zs={z_sc:.1f}" if z_sc != 1.0 else ""
                logger.info(
                    "WS %s bar %d: $%.1f z=%+.3f sig=%d hold=%d regime=%s dz=%.3f%s%s",
                    label, result["bar"], result["close"],
                    result["z"], result["signal"],
                    result["hold_count"], regime, result.get("dz", 0),
                    z_sc_str, trade_str,
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
            for symbol, runner in runners.items():
                if runner._current_signal != 0:
                    logger.info("Closing %s on exit...", symbol)
                    reliable_close_position(adapter, symbol, verify=False)


def main():
    parser = argparse.ArgumentParser(description="Bybit alpha strategy runner")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"],
                        help="Symbols to trade (default: BTCUSDT ETHUSDT)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--ws", action="store_true", help="Use WebSocket instead of REST polling")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--legacy", action="store_true",
        help="Use legacy AlphaRunner instead of LiveRunner (deprecated path)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    adapter = create_adapter()
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
    kill_switch = RustKillSwitch()

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
        return

    if args.once:
        for symbol, runner in runners.items():
            bars = adapter.get_klines(symbol, interval=INTERVAL, limit=2)
            if bars:
                result = runner.process_bar(bars[0])
                logger.info("%s: %s", symbol, json.dumps(result, default=str))
        return

    # Initialize portfolio manager (unified position + risk)
    pm = PortfolioManager(adapter, dry_run=args.dry_run,
                          risk_evaluator=risk_eval, kill_switch=kill_switch)
    logger.info("PM: PortfolioManager enabled (max_exposure=140%%, max_per_sym=30%%, max_dd=15%%, Rust risk)")

    # Hedge disabled: 8 cycles over 3 days, 0 wins, -$45 in fees, ratio
    # oscillates in 6th decimal (0.000048-0.000051) = pure noise trading.
    # Keeping code for future re-evaluation with longer MA or different basket.
    hedge = None

    # WebSocket mode: push-based, low latency
    if args.ws:
        _run_ws_mode(runners, adapter, args.dry_run,
                     runner_intervals=runner_intervals, hedge_runner=hedge,
                     portfolio_manager=pm)
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
                    bars = adapter.get_klines(symbol, interval=INTERVAL, limit=2)
                    if not bars:
                        continue

                    latest = bars[0]
                    bar_time = latest["time"]

                    if bar_time > last_bar_times[symbol]:
                        last_bar_times[symbol] = bar_time
                        result = runner.process_bar(latest)
                        if result.get("action") == "signal":
                            regime = result.get("regime", "?")
                            logger.info(
                                "%s bar %d: $%.1f z=%+.3f sig=%d hold=%d regime=%s dz=%.3f%s",
                                symbol, result["bar"], result["close"],
                                result["z"], result["signal"],
                                result["hold_count"], regime,
                                result.get("dz", 0),
                                f" TRADE={result['trade']}" if "trade" in result else "",
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


if __name__ == "__main__":
    main()
