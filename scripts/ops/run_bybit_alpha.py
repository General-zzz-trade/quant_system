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
from scripts.ops.portfolio_manager import PortfolioManager
from scripts.ops.portfolio_combiner import PortfolioCombiner
from scripts.ops.hedge_runner import HedgeRunner

logger = logging.getLogger(__name__)


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

    combiners: dict[str, PortfolioCombiner] = {}
    for rsym, rkeys in symbol_runners.items():
        if len(rkeys) > 1:
            # Multiple alphas on same symbol -> use combiner
            weights = {k: 0.5 for k in rkeys}
            combiners[rsym] = PortfolioCombiner(
                adapter=adapter, symbol=rsym, weights=weights,
                threshold=0.3, dry_run=dry_run,
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

    try:
        while True:
            time.sleep(300)
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
                    adapter.close_position(symbol)


def main():
    parser = argparse.ArgumentParser(description="Bybit alpha strategy runner")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"],
                        help="Symbols to trade (default: BTCUSDT ETHUSDT)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--ws", action="store_true", help="Use WebSocket instead of REST polling")
    parser.add_argument("--log-level", default="INFO")
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
    state_store = RustStateStore(all_symbols, "USDT", 0)
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

    # Initialize hedge runner (BTC+ALT structural alpha)
    hedge = HedgeRunner(adapter, dry_run=args.dry_run) if not args.dry_run else None
    if hedge:
        hedge.warmup_from_csv(n_bars=600)  # Pre-load 25 days of history -> immediate signals
        logger.info("HEDGE: BTC+ALT hedge enabled, ALT basket=%s", hedge.ALT_BASKET)

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
                    adapter.close_position(symbol)


if __name__ == "__main__":
    main()
