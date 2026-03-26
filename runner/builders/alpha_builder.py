"""Builder and balance helpers for alpha_main coordinator construction."""
from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any

from _quant_hotpath import RustInferenceBridge

from decision.modules.alpha import AlphaDecisionModule
from decision.signals.alpha_signal import EnsemblePredictor, SignalDiscretizer
from decision.sizing.adaptive import AdaptivePositionSizer
from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from engine.feature_hook import FeatureComputeHook
from execution.adapters.bybit.execution_adapter import BybitExecutionAdapter
from strategy.config import SYMBOL_CONFIG, LEVERAGE_LADDER

logger = logging.getLogger(__name__)

MODEL_BASE = Path("models_v8")
DATA_DIR = Path("data_files")


# ── Static data source loaders (from CSV files refreshed by data-refresh timer) ──

def _load_latest_csv_value(path: Path, ts_col: str = "timestamp", val_col: str = None) -> float:
    """Load the latest value from a 2-column CSV (timestamp, value)."""
    if not path.exists():
        return math.nan
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            last_row = None
            for row in reader:
                last_row = row
            if last_row is None:
                return math.nan
            if val_col:
                return float(last_row[val_col])
            # Auto-detect: use second column
            cols = list(last_row.keys())
            return float(last_row[cols[1]])
    except Exception:
        return math.nan


def _load_latest_macro(path: Path) -> dict:
    """Load latest macro data from macro_daily.csv."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            last_row = None
            for row in reader:
                last_row = row
            if last_row is None:
                return {}
            return {
                "dxy": float(last_row.get("dxy", "nan")),
                "spx": float(last_row.get("spx", last_row.get("spy_close", "nan"))),
                "vix": float(last_row.get("vix", "nan")),
                "date": last_row.get("date", ""),
            }
    except Exception:
        return {}


def _load_latest_onchain(path: Path) -> dict:
    """Load latest on-chain metrics."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            last_row = None
            for row in reader:
                last_row = row
            return dict(last_row) if last_row else {}
    except Exception:
        return {}


def _build_data_sources(symbol: str) -> dict:
    """Build all external data source callables for FeatureComputeHook.

    Each source returns the latest value from CSV files that are refreshed
    every 6 hours by the data-refresh systemd timer. Values are cached in
    memory and refreshed every 60 seconds (not on every bar push).
    """
    _cache: dict[str, Any] = {}
    _cache_ts: dict[str, float] = {}
    _CACHE_TTL = 60.0  # seconds

    import time as _time

    def _cached(key: str, loader):
        """Return cached value, refresh if older than TTL."""
        now = _time.monotonic()
        if key not in _cache or (now - _cache_ts.get(key, 0)) > _CACHE_TTL:
            _cache[key] = loader()
            _cache_ts[key] = now
        return _cache[key]

    sources: dict[str, Any] = {}

    # Funding rate
    funding_path = DATA_DIR / f"{symbol}_funding.csv"
    if funding_path.exists():
        sources["funding_rate_source"] = lambda _p=funding_path: _cached(
            "funding", lambda: _load_latest_csv_value(_p, val_col="funding_rate"))

    # Open interest
    oi_path = DATA_DIR / f"{symbol}_open_interest.csv"
    if oi_path.exists():
        sources["oi_source"] = lambda _p=oi_path: _cached(
            "oi", lambda: _load_latest_csv_value(_p, val_col="sum_open_interest"))

    # Long/short ratio
    ls_path = DATA_DIR / f"{symbol}_ls_ratio.csv"
    if ls_path.exists():
        sources["ls_ratio_source"] = lambda _p=ls_path: _cached(
            "ls", lambda: _load_latest_csv_value(_p, val_col="long_short_ratio"))

    # Spot close (for basis calculation)
    spot_path = DATA_DIR / f"{symbol}_spot_1h.csv"
    if spot_path.exists():
        sources["spot_close_source"] = lambda _p=spot_path: _cached(
            "spot", lambda: _load_latest_csv_value(_p, val_col="close"))

    # Fear & Greed Index (symbol-independent)
    fgi_path = DATA_DIR / "fear_greed_index.csv"
    if fgi_path.exists():
        sources["fgi_source"] = lambda _p=fgi_path: _cached(
            "fgi", lambda: _load_latest_csv_value(_p, val_col="value"))

    # Implied volatility (Deribit) — column is 'implied_vol'
    iv_path = DATA_DIR / f"{symbol}_deribit_iv.csv"
    if iv_path.exists():
        sources["implied_vol_source"] = lambda _p=iv_path: _cached(
            "iv", lambda: _load_latest_csv_value(_p, val_col="implied_vol"))

    # Put/call ratio
    pcr_path = DATA_DIR / f"{symbol}_deribit_pcr.csv"
    if not pcr_path.exists():
        pcr_path = DATA_DIR / f"{symbol}_deribit_iv.csv"
    if pcr_path.exists():
        sources["put_call_ratio_source"] = lambda _p=pcr_path: _cached(
            "pcr", lambda: _load_latest_csv_value(_p, val_col="put_call_ratio"))

    # On-chain metrics (file naming: btc_onchain_daily.csv, eth_onchain_daily.csv)
    # CSV columns: exchange_reserve, exchange_inflow, exchange_outflow, exchange_netflow
    # FeatureHook expects: FlowInExUSD, FlowOutExUSD, SplyExNtv, AdrActCnt, TxTfrCnt, HashRate
    sym_lower = symbol.replace("USDT", "").lower()  # BTCUSDT → btc
    onchain_path = DATA_DIR / f"{sym_lower}_onchain_daily.csv"
    if onchain_path.exists():
        def _onchain_mapped(_p=onchain_path):
            raw = _cached("onchain_raw", lambda: _load_latest_onchain(_p))
            return {
                "FlowInExUSD": float(raw.get("exchange_inflow", "nan")),
                "FlowOutExUSD": float(raw.get("exchange_outflow", "nan")),
                "SplyExNtv": float(raw.get("exchange_reserve", "nan")),
                "AdrActCnt": math.nan,  # not in this CSV
                "TxTfrCnt": math.nan,   # not in this CSV
                "HashRate": math.nan,   # not in this CSV
            }
        sources["onchain_source"] = _onchain_mapped

    # Liquidation proxy
    # CSV columns: ts, liq_proxy_volume, liq_proxy_buy, liq_proxy_sell, liq_proxy_imbalance, liq_proxy_cluster
    # FeatureHook expects: liq_total_volume, liq_buy_volume, liq_sell_volume, liq_count
    liq_path = DATA_DIR / f"{symbol}_liquidation_proxy.csv"
    if liq_path.exists():
        def _liq_mapped(_p=liq_path):
            raw = _cached("liq_raw", lambda: _load_latest_onchain(_p))
            return {
                "liq_total_volume": float(raw.get("liq_proxy_volume", "nan")),
                "liq_buy_volume": float(raw.get("liq_proxy_buy", "nan")),
                "liq_sell_volume": float(raw.get("liq_proxy_sell", "nan")),
                "liq_count": 0.0,  # proxy doesn't have count
            }
        sources["liquidation_source"] = _liq_mapped

    # Macro from fred_macro.csv or individual ETF files
    macro_path = DATA_DIR / "fred_macro.csv"
    if not macro_path.exists() or macro_path.stat().st_size < 50:
        # Try assembling from individual macro files
        spy_path = DATA_DIR / "macro" / "SPY_daily.csv"
        vix_path = DATA_DIR / "macro" / "VIX_daily.csv"
        if spy_path.exists():
            def _macro_from_etfs(_spy=spy_path, _vix=vix_path):
                result = {"dxy": math.nan, "spx": math.nan, "vix": math.nan, "date": ""}
                try:
                    result["spx"] = _load_latest_csv_value(_spy, val_col="close")
                except Exception:
                    pass
                try:
                    result["vix"] = _load_latest_csv_value(_vix, val_col="close")
                except Exception:
                    pass
                return result
            sources["macro_source"] = lambda: _cached("macro", _macro_from_etfs)
    else:
        sources["macro_source"] = lambda _p=macro_path: _cached(
            "macro", lambda: _load_latest_macro(_p))

    return sources


def get_initial_balance(adapter: Any) -> float:
    """Fetch USDT equity from Bybit adapter for tick processor initialization.

    Returns 0.0 on any failure (tick processor will be initialized with zero
    balance and updated from exchange on first bar).
    """
    try:
        snapshot = adapter.get_balances()
        bal = snapshot.get("USDT")
        if bal is not None:
            return float(bal.total)
    except Exception:
        logger.debug("Could not fetch initial balance for tick processor", exc_info=True)
    return 0.0


def build_coordinator(
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

    # External data sources (CSV files refreshed by data-refresh timer)
    data_sources = _build_data_sources(symbol)
    n_sources = len(data_sources)
    logger.info("Data sources for %s: %d connected (%s)",
                runner_key, n_sources, ", ".join(sorted(data_sources.keys())))

    # Feature engine (per-symbol Rust instance created lazily by hook)
    feature_hook = FeatureComputeHook(
        computer=None,
        warmup_bars=cfg.get("warmup", 300 if is_4h else 800),
        **data_sources,
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

    # Leverage from strategy_config (auto-detects live vs demo)
    leverage = LEVERAGE_LADDER[0][1] if LEVERAGE_LADDER else 10.0

    # Decision module
    alpha_module = AlphaDecisionModule(
        symbol=symbol,
        runner_key=runner_key,
        predictor=predictor,
        discretizer=discretizer,
        sizer=sizer,
        leverage=leverage,
    )

    # Fetch exchange balance for state store initialization
    balance = get_initial_balance(adapter)
    logger.info("Initial balance for %s: $%.2f", runner_key, balance)

    # RustTickProcessor: full hot-path (~80μs vs ~1ms Python pipeline)
    # DISABLED: tick processor has its own internal z-score buffer that doesn't
    # sync with Python-side InferenceBridge. This causes z=0 in production
    # because decide() reads from the Python bridge (empty) instead of the
    # tick processor's Rust buffer. Re-enable once signal routing is unified.
    tick_proc = None

    # Coordinator config
    coordinator_cfg = CoordinatorConfig(
        symbol_default=symbol,
        symbols=(symbol,),
        currency="USDT",
        feature_hook=feature_hook,
        tick_processor=tick_proc,
        starting_balance=balance,
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

    fast_path = "RustTickProcessor ENABLED" if tick_proc is not None else "Python pipeline (tick processor unavailable)"
    logger.info(
        "Built coordinator: runner_key=%s symbol=%s dry_run=%s warmup=%d path=%s",
        runner_key, symbol, dry_run,
        cfg.get("warmup", 300 if is_4h else 800),
        fast_path,
    )
    return coordinator, alpha_module
