# runner/testnet_validation.py
"""3-phase testnet validation workflow: paper → shadow → live → longrun → compare.

Usage:
    python -m runner.testnet_validation --config testnet_binance.yaml --phase paper --duration 300
    python -m runner.testnet_validation --config testnet_binance.yaml --phase shadow --duration 300
    python -m runner.testnet_validation --config testnet_binance.yaml --phase live --duration 300
    python -m runner.testnet_validation --config testnet_binance.yaml --phase longrun
    python -m runner.testnet_validation --config testnet_binance.yaml --phase compare
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _build_ml_stack(
    raw: Dict[str, Any],
) -> Tuple[Optional[Any], List[Any], List[Any], Dict[str, Any]]:
    """Build ML pipeline (feature_computer, alpha_models, decision_modules, signal_kwargs) from config.

    Returns (None, [], [], {}) if no strategy.model_path configured or no models found.

    When model_dir contains a config.json (V8+ format), loads ensemble models and
    extracts signal constraints automatically. Falls back to legacy per-symbol layout.
    """
    strategy = raw.get("strategy", {})
    model_path = strategy.get("model_path")
    if not model_path:
        logger.info("No strategy.model_path — running without ML stack")
        return None, [], [], {}

    from alpha.models.lgbm_alpha import LGBMAlphaModel
    from alpha.models.xgb_alpha import XGBAlphaModel
    from decision.ml_decision import MLDecisionModule
    from features.enriched_computer import EnrichedFeatureComputer

    model_dir = Path(model_path)
    symbols = raw.get("trading", {}).get("symbols", ["BTCUSDT"])
    threshold = strategy.get("threshold", 0.002)
    threshold_short = strategy.get("threshold_short", 999.0)
    risk_pct = strategy.get("risk_pct", 0.30)

    signal_kwargs: Dict[str, Any] = {}
    models: List[Any] = []
    pos_mgmt: Dict[str, Any] = {}

    # ── V8+ format: config.json in model_dir ──────────────
    model_config_path = model_dir / "config.json"
    if model_config_path.exists():
        with model_config_path.open() as f:
            mcfg = json.load(f)

        if mcfg.get("ensemble"):
            from alpha.models.ensemble import EnsembleAlphaModel

            sub_models: List[Any] = []
            weights = mcfg.get("ensemble_weights", [])
            for i, fname in enumerate(mcfg.get("models", [])):
                pkl_path = model_dir / fname
                if not pkl_path.exists():
                    logger.warning("Ensemble member not found: %s", pkl_path)
                    continue
                if "lgbm" in fname.lower():
                    m = LGBMAlphaModel(name=fname)
                    m.load(pkl_path)
                elif "xgb" in fname.lower():
                    m = XGBAlphaModel(name=fname)
                    m.load(pkl_path)
                else:
                    logger.warning("Unknown model type for %s, trying LGBM", fname)
                    m = LGBMAlphaModel(name=fname)
                    m.load(pkl_path)
                sub_models.append(m)
                logger.info("Loaded ensemble member: %s", pkl_path)

            if not sub_models:
                logger.warning("No ensemble members loaded")
                return None, [], [], {}

            # Adjust weights to match loaded sub_models (some may be missing)
            if len(weights) != len(sub_models):
                weights = [1.0 / len(sub_models)] * len(sub_models)

            ensemble = EnsembleAlphaModel(
                name=f"ensemble_{mcfg.get('symbol', 'unknown')}",
                sub_models=sub_models,
                weights=weights,
            )
            models.append(ensemble)
        else:
            # Single model in V8 dir
            for fname in mcfg.get("models", []):
                pkl_path = model_dir / fname
                if not pkl_path.exists():
                    continue
                if "lgbm" in fname.lower():
                    m = LGBMAlphaModel(name=fname)
                    m.load(pkl_path)
                elif "xgb" in fname.lower():
                    m = XGBAlphaModel(name=fname)
                    m.load(pkl_path)
                else:
                    m = LGBMAlphaModel(name=fname)
                    m.load(pkl_path)
                models.append(m)
                logger.info("Loaded model: %s", pkl_path)

        # Extract signal constraints from config.json
        if mcfg.get("long_only"):
            signal_kwargs["long_only_symbols"] = set(symbols)
        if "deadzone" in mcfg:
            signal_kwargs["deadzone"] = mcfg["deadzone"]
        if "min_hold" in mcfg:
            signal_kwargs["min_hold_bars"] = {s: mcfg["min_hold"] for s in symbols}
        if mcfg.get("monthly_gate", False) or strategy.get("monthly_gate", False):
            signal_kwargs["monthly_gate"] = True
            signal_kwargs["monthly_gate_window"] = mcfg.get(
                "monthly_gate_window", strategy.get("monthly_gate_window", 480)
            )

        # Position management: bear_thresholds, vol_target, vol_feature
        pos_mgmt = mcfg.get("position_management", {})
        if pos_mgmt.get("bear_thresholds"):
            signal_kwargs["bear_thresholds"] = [tuple(x) for x in pos_mgmt["bear_thresholds"]]
        if pos_mgmt.get("vol_target") is not None:
            signal_kwargs["vol_target"] = pos_mgmt["vol_target"]
        if pos_mgmt.get("vol_feature"):
            signal_kwargs["vol_feature"] = pos_mgmt["vol_feature"]

        # Bear model for regime-switch (Strategy F)
        bear_model_path = mcfg.get("bear_model_path")
        if bear_model_path:
            bear_dir = Path(bear_model_path)
            bear_cfg_path = bear_dir / "config.json"
            if bear_cfg_path.exists():
                with bear_cfg_path.open() as bf:
                    bear_cfg = json.load(bf)
                bear_pkl = bear_dir / bear_cfg["models"][0]
                if bear_pkl.exists():
                    bear_m = LGBMAlphaModel(name="bear_detector")
                    bear_m.load(bear_pkl)
                    signal_kwargs["bear_model"] = bear_m
                    logger.info("Loaded bear model: %s", bear_pkl)

    # ── Legacy format: model_dir/SYM/config_name.pkl ──────
    if not models:
        config_name = strategy.get("config_name", "mod_reg_1h")
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
        return None, [], [], {}

    fc = EnrichedFeatureComputer()

    dd_limit = pos_mgmt.get("dd_limit") or 0.0
    dd_cooldown = pos_mgmt.get("dd_cooldown") or 48

    dms = [
        MLDecisionModule(
            symbol=sym,
            threshold=threshold,
            threshold_short=threshold_short,
            risk_pct=risk_pct,
            dd_limit=dd_limit,
            dd_cooldown=dd_cooldown,
        )
        for sym in symbols
    ]

    logger.info(
        "ML stack ready: %d models, %d decision modules, threshold=%.4f, signal_kwargs=%s",
        len(models), len(dms), threshold, signal_kwargs,
    )
    return fc, models, dms, signal_kwargs


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


def _start_pollers(symbol: str, testnet: bool = True):
    """Start all data pollers. Returns dict of pollers by name."""
    from execution.adapters.binance.funding_poller import BinanceFundingPoller
    from execution.adapters.binance.oi_poller import BinanceOIPoller
    from execution.adapters.fgi_poller import FGIPoller
    from execution.adapters.deribit_iv_poller import DeribitIVPoller
    from execution.adapters.onchain_poller import OnchainPoller
    from execution.adapters.binance.liquidation_poller import BinanceLiquidationPoller
    from execution.adapters.mempool_poller import MempoolPoller
    from execution.adapters.macro_poller import MacroPoller
    from execution.adapters.sentiment_poller import SentimentPoller

    currency = symbol.replace("USDT", "")
    asset = currency.lower()
    funding = BinanceFundingPoller(symbol=symbol, testnet=testnet)
    oi = BinanceOIPoller(symbol=symbol, testnet=testnet)
    fgi = FGIPoller()
    deribit_iv = DeribitIVPoller(currency=currency)
    onchain = OnchainPoller(asset=asset)
    liquidation = BinanceLiquidationPoller(symbol=symbol, testnet=testnet)
    mempool = MempoolPoller()
    macro = MacroPoller()
    sentiment = SentimentPoller()
    funding.start()
    oi.start()
    fgi.start()
    deribit_iv.start()
    onchain.start()
    liquidation.start()
    mempool.start()
    macro.start()
    sentiment.start()
    return funding, oi, fgi, deribit_iv, onchain, liquidation, mempool, macro, sentiment


def _stop_pollers(*pollers: Any) -> None:
    for p in pollers:
        try:
            p.stop()
        except Exception:
            pass


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

    fc, models, dms, signal_kwargs = _build_ml_stack(raw)
    funding, oi, fgi, deribit_iv, onchain, liquidation, mempool, macro, sentiment = _start_pollers(symbols[0], testnet=True)

    # Cross-asset computer for non-BTC symbols (BTC-lead features)
    cross_asset = None
    btc_kline_poller = None
    if symbols[0] != "BTCUSDT":
        from features.cross_asset_computer import CrossAssetComputer
        from execution.adapters.binance.btc_kline_poller import BtcKlinePoller
        from execution.adapters.binance.funding_poller import BinanceFundingPoller

        cross_asset = CrossAssetComputer()
        btc_funding = BinanceFundingPoller(symbol="BTCUSDT", testnet=True)
        btc_funding.start()
        btc_kline_poller = BtcKlinePoller(
            cross_asset, testnet=True,
            funding_source=btc_funding.get_rate,
        )
        btc_kline_poller.start()
        logger.info("Cross-asset enabled: BTC kline poller feeding CrossAssetComputer")

    runner = LivePaperRunner.build(
        config,
        feature_computer=fc,
        alpha_models=models or None,
        decision_modules=dms or None,
        funding_rate_source=funding.get_rate,
        oi_source=oi.get_oi,
        fgi_source=fgi.get_value,
        implied_vol_source=lambda: deribit_iv.get_current()[0],
        put_call_ratio_source=lambda: deribit_iv.get_current()[1],
        onchain_source=onchain.get_current,
        liquidation_source=liquidation.get_current,
        mempool_source=mempool.get_current,
        macro_source=macro.get_current,
        sentiment_source=sentiment.get_current,
        cross_asset_computer=cross_asset,
        **signal_kwargs,
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
    finally:
        _stop_pollers(funding, oi, fgi, deribit_iv, onchain, liquidation, mempool, macro, sentiment)
        if btc_kline_poller is not None:
            btc_kline_poller.stop()

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

    # Shadow mode needs a venue client that won't be called
    class _NoOpClient:
        def send_order(self, order_event: Any) -> list:
            return []

    fc, models, dms, signal_kwargs = _build_ml_stack(raw)
    bear_model = signal_kwargs.pop("bear_model", None)
    config = LiveRunnerConfig(
        symbols=symbols,
        testnet=True,
        shadow_mode=True,
        enable_preflight=False,
        enable_persistent_stores=False,
        **signal_kwargs,
    )

    funding, oi, fgi, deribit_iv, onchain, liquidation, mempool, macro, sentiment = _start_pollers(symbols[0], testnet=True)

    runner = LiveRunner.build(
        config,
        venue_clients={"binance": _NoOpClient()},
        feature_computer=fc,
        alpha_models=models or None,
        decision_modules=dms or None,
        bear_model=bear_model,
        funding_rate_source=funding.get_rate,
        oi_source=oi.get_oi,
        fgi_source=fgi.get_value,
        implied_vol_source=lambda: deribit_iv.get_current()[0],
        put_call_ratio_source=lambda: deribit_iv.get_current()[1],
        onchain_source=onchain.get_current,
        liquidation_source=liquidation.get_current,
        mempool_source=mempool.get_current,
        macro_source=macro.get_current,
        sentiment_source=sentiment.get_current,
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
    finally:
        _stop_pollers(funding, oi, fgi, deribit_iv, onchain, liquidation, mempool, macro, sentiment)

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
    key_env = creds.get("api_key_env") or "BINANCE_TESTNET_API_KEY"
    secret_env = creds.get("api_secret_env") or "BINANCE_TESTNET_API_SECRET"
    api_key = os.environ.get(key_env, "")
    api_secret = os.environ.get(secret_env, "")

    if not api_key or not api_secret:
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

    fc, models, dms, signal_kwargs = _build_ml_stack(raw)
    bear_model = signal_kwargs.pop("bear_model", None)
    config = LiveRunnerConfig(
        symbols=symbols,
        testnet=True,
        enable_persistent_stores=False,
        **signal_kwargs,
    )

    funding, oi, fgi, deribit_iv, onchain, liquidation, mempool, macro, sentiment = _start_pollers(symbols[0], testnet=True)

    runner = LiveRunner.build(
        config,
        venue_clients={"binance": client},
        feature_computer=fc,
        alpha_models=models or None,
        decision_modules=dms or None,
        bear_model=bear_model,
        funding_rate_source=funding.get_rate,
        oi_source=oi.get_oi,
        fgi_source=fgi.get_value,
        implied_vol_source=lambda: deribit_iv.get_current()[0],
        put_call_ratio_source=lambda: deribit_iv.get_current()[1],
        onchain_source=onchain.get_current,
        liquidation_source=liquidation.get_current,
        mempool_source=mempool.get_current,
        macro_source=macro.get_current,
        sentiment_source=sentiment.get_current,
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
    finally:
        _stop_pollers(funding, oi, fgi, deribit_iv, onchain, liquidation, mempool, macro, sentiment)

    out = _output_dir(config_path)
    _write_equity_csv(out / "live_equity.csv", runner.fills, 10000.0)
    logger.info("Live testnet phase complete. Fills: %d. Output: %s", len(runner.fills), out)


def _start_status_logger(
    runner: Any,
    ws_transport: Any,
    stop_event: threading.Event,
    interval: float = 60.0,
) -> threading.Thread:
    """Daemon thread that prints periodic status lines for longrun mode."""
    start_time = time.monotonic()

    def _loop() -> None:
        while not stop_event.wait(timeout=interval):
            uptime = int(time.monotonic() - start_time)
            events = runner.event_index
            n_fills = len(runner.fills)

            # Feature completeness from coordinator state
            view = runner.coordinator.get_state_view()
            features = view.get("features", {})
            total = len(features)
            valid = sum(1 for v in features.values() if v is not None and not (isinstance(v, float) and math.isnan(v)))

            ws_state = ws_transport.state.value if hasattr(ws_transport, "state") else "unknown"

            logger.info(
                "LONGRUN STATUS | uptime=%ds events=%d fills=%d features=%d/%d ws=%s",
                uptime, events, n_fills, valid, total, ws_state,
            )

    t = threading.Thread(target=_loop, name="longrun-status", daemon=True)
    t.start()
    return t


def run_longrun(config_path: Path, duration: int) -> None:
    """Long-running testnet mode with WS reconnection and state persistence.

    Runs indefinitely (ignores duration). Stop with SIGTERM/SIGINT (Ctrl+C).
    Uses ReconnectingWsTransport for WS resilience and SQLite state checkpointing.
    """
    from infra.config.loader import load_config_secure, resolve_credentials
    from execution.adapters.binance.rest import BinanceRestClient, BinanceRestConfig
    from execution.adapters.binance.urls import resolve_binance_urls
    from execution.adapters.binance.ws_transport_websocket_client import WebsocketClientTransport
    from execution.adapters.binance.reconnecting_ws_transport import ReconnectingWsTransport
    from runner.live_runner import LiveRunner, LiveRunnerConfig

    raw = load_config_secure(config_path)
    _ensure_testnet(raw)
    resolve_credentials(raw)

    trading = raw.get("trading", {})
    symbol = trading.get("symbol", "BTCUSDT")
    symbols = tuple(trading["symbols"]) if "symbols" in trading else (symbol,)

    creds = raw.get("credentials", {})
    key_env = creds.get("api_key_env") or "BINANCE_TESTNET_API_KEY"
    secret_env = creds.get("api_secret_env") or "BINANCE_TESTNET_API_SECRET"
    api_key = os.environ.get(key_env, "")
    api_secret = os.environ.get(secret_env, "")

    if not api_key or not api_secret:
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

    fc, models, dms, signal_kwargs = _build_ml_stack(raw)
    bear_model = signal_kwargs.pop("bear_model", None)

    output_dir = _output_dir(config_path)
    config = LiveRunnerConfig(
        symbols=symbols,
        testnet=True,
        enable_persistent_stores=True,
        data_dir=str(output_dir),
        **signal_kwargs,
    )

    funding, oi, fgi, deribit_iv, onchain, liquidation, mempool, macro, sentiment = _start_pollers(symbols[0], testnet=True)

    ws_transport = ReconnectingWsTransport(
        inner=WebsocketClientTransport(),
        max_retries=20,
        max_delay_s=120.0,
    )

    runner = LiveRunner.build(
        config,
        venue_clients={"binance": client},
        transport=ws_transport,
        feature_computer=fc,
        alpha_models=models or None,
        decision_modules=dms or None,
        bear_model=bear_model,
        funding_rate_source=funding.get_rate,
        oi_source=oi.get_oi,
        fgi_source=fgi.get_value,
        implied_vol_source=lambda: deribit_iv.get_current()[0],
        put_call_ratio_source=lambda: deribit_iv.get_current()[1],
        onchain_source=onchain.get_current,
        liquidation_source=liquidation.get_current,
        mempool_source=mempool.get_current,
        macro_source=macro.get_current,
        sentiment_source=sentiment.get_current,
    )

    status_stop = threading.Event()
    _start_status_logger(runner, ws_transport, status_stop)

    logger.info("Starting LONGRUN mode (Ctrl+C or SIGTERM to stop)...")
    try:
        runner.start()
    except KeyboardInterrupt:
        runner.stop()
    finally:
        status_stop.set()
        _stop_pollers(funding, oi, fgi, deribit_iv, onchain, liquidation, mempool, macro, sentiment)

    _write_equity_csv(output_dir / "longrun_equity.csv", runner.fills, 10000.0)
    logger.info(
        "Longrun stopped. Fills: %d, Events: %d. Output: %s",
        len(runner.fills), runner.event_index, output_dir,
    )


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
        choices=["paper", "shadow", "live", "longrun", "compare"],
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
    elif args.phase == "longrun":
        run_longrun(args.config, args.duration)
    elif args.phase == "compare":
        run_compare(args.config)


if __name__ == "__main__":
    main()
