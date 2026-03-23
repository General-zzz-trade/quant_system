#!/usr/bin/env python3
"""Kernel-aware backtester — replays historical bars through the production pipeline.

Uses the same code path as live trading:
  FeatureComputeHook → RustFeatureEngine → LiveInferenceBridge → signal

This eliminates the parity gap between research backtests (batch Python) and
production (incremental Rust kernel).

Usage:
    python3 -m scripts.backtest_kernel
    python3 -m scripts.backtest_kernel --symbols BTCUSDT ETHUSDT
    python3 -m scripts.backtest_kernel --full --alloc inverse_vol
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────

FEE_BPS = 4e-4
SLIPPAGE_BPS = 2e-4
COST_PER_TRADE = FEE_BPS + SLIPPAGE_BPS
INITIAL_CAPITAL = 10000.0
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
WARMUP_BARS = 65


# ── BarEvent — synthetic event matching production interface ─────

@dataclass
class BarEvent:
    """Synthetic bar event that matches the production event interface.

    FeatureComputeHook.on_event() uses getattr() to extract fields,
    so this dataclass must have the same attribute names as the live event.
    """
    symbol: str
    ts: datetime
    close: float
    open: float
    high: float
    low: float
    volume: float
    quote_volume: float = 0.0
    trades: int = 0
    taker_buy_volume: float = 0.0
    taker_buy_quote_volume: float = 0.0
    event_type: str = "market_bar"


# ── Auxiliary data sources ───────────────────────────────────────

class TimeseriesCursor:
    """Forward-only cursor over a sorted (timestamp, value) schedule.

    Returns the most recent value at or before the query timestamp.
    """

    def __init__(self, timestamps: np.ndarray, values: np.ndarray):
        self._ts = timestamps
        self._vals = values
        self._idx = 0
        self._cur: Optional[float] = None

    def advance_to(self, ts_ms: int) -> Optional[float]:
        while self._idx < len(self._ts) and self._ts[self._idx] <= ts_ms:
            self._cur = float(self._vals[self._idx])
            self._idx += 1
        return self._cur


class DictCursor:
    """Forward-only cursor returning a dict of columns at each timestamp."""

    def __init__(self, df: pd.DataFrame, ts_col: str = "timestamp"):
        self._ts = df[ts_col].values.astype(np.int64)
        self._cols = {c: df[c].values for c in df.columns if c != ts_col}
        self._idx = 0
        self._cur: Optional[Dict[str, float]] = None

    def advance_to(self, ts_ms: int) -> Optional[Dict[str, float]]:
        while self._idx < len(self._ts) and self._ts[self._idx] <= ts_ms:
            row = {}
            for k, v in self._cols.items():
                val = v[self._idx]
                try:
                    row[k] = float(val)
                except (ValueError, TypeError):
                    row[k] = val  # keep strings as-is (e.g. date)
            self._cur = row
            self._idx += 1
        return self._cur


def _load_funding_cursor(symbol: str) -> Optional[TimeseriesCursor]:
    path = Path(f"data_files/{symbol}_funding.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return TimeseriesCursor(
        df["timestamp"].values.astype(np.int64),
        df["funding_rate"].values.astype(np.float64),
    )


def _load_oi_cursor(symbol: str) -> Optional[TimeseriesCursor]:
    path = Path(f"data_files/{symbol}_open_interest.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    col = "sum_open_interest_value" if "sum_open_interest_value" in df.columns else "sum_open_interest"
    return TimeseriesCursor(
        df["timestamp"].values.astype(np.int64),
        df[col].values.astype(np.float64),
    )


def _load_ls_cursor(symbol: str) -> Optional[TimeseriesCursor]:
    path = Path(f"data_files/{symbol}_ls_ratio.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return TimeseriesCursor(
        df["timestamp"].values.astype(np.int64),
        df["long_short_ratio"].values.astype(np.float64),
    )


def _load_fgi_cursor() -> Optional[TimeseriesCursor]:
    path = Path("data_files/fear_greed_index.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return TimeseriesCursor(
        df["timestamp"].values.astype(np.int64),
        df["value"].values.astype(np.float64),
    )


def _load_macro_cursor() -> Optional[DictCursor]:
    path = Path("data_files/macro_daily.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    ts_col = "timestamp_ms" if "timestamp_ms" in df.columns else "timestamp"
    cols = ["dxy", "spx", "vix", "date"]
    keep = [c for c in cols if c in df.columns]
    sub = df[[ts_col] + keep].rename(columns={ts_col: "timestamp"})
    return DictCursor(sub)


def _load_mempool_cursor() -> Optional[DictCursor]:
    path = Path("data_files/btc_mempool_fees.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    rename = {}
    if "avg_fee" in df.columns:
        rename["avg_fee"] = "fastest_fee"
    if "min_fee" in df.columns:
        rename["min_fee"] = "economy_fee"
    # mempool_size not in this file, but RustFeatureEngine handles NaN
    df = df.rename(columns=rename)
    cols = [c for c in ["fastest_fee", "economy_fee", "fee_urgency"] if c in df.columns]
    sub = df[["timestamp"] + cols]
    return DictCursor(sub)


def _load_spot_close_cursor(symbol: str) -> Optional[TimeseriesCursor]:
    path = Path(f"data_files/{symbol}_spot_1h.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    ts_col = "open_time" if "open_time" in df.columns else "timestamp"
    return TimeseriesCursor(
        df[ts_col].values.astype(np.int64),
        df["close"].values.astype(np.float64),
    )


def _load_iv_cursor(symbol: str) -> Optional[TimeseriesCursor]:
    path = Path(f"data_files/{symbol}_deribit_iv.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "implied_vol" not in df.columns:
        return None
    # Convert ISO timestamp to ms
    ts = pd.to_datetime(df["timestamp"]).astype(np.int64) // 10**6
    return TimeseriesCursor(ts.values, df["implied_vol"].values.astype(np.float64))


def _load_pcr_cursor(symbol: str) -> Optional[TimeseriesCursor]:
    path = Path(f"data_files/{symbol}_deribit_iv.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "put_call_ratio" not in df.columns:
        return None
    ts = pd.to_datetime(df["timestamp"]).astype(np.int64) // 10**6
    return TimeseriesCursor(ts.values, df["put_call_ratio"].values.astype(np.float64))


def _load_onchain_cursor(symbol: str) -> Optional[DictCursor]:
    path = Path(f"data_files/{symbol}_onchain.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    ts_col = "timestamp"
    cols = [c for c in ["FlowInExUSD", "FlowOutExUSD", "SplyExNtv",
                         "AdrActCnt", "TxTfrCnt", "HashRate"] if c in df.columns]
    if not cols:
        return None
    sub = df[[ts_col] + cols]
    return DictCursor(sub)


def _load_liquidation_cursor(symbol: str) -> Optional[DictCursor]:
    path = Path(f"data_files/{symbol}_liquidation_proxy.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    ts_col = "ts" if "ts" in df.columns else "timestamp"
    rename = {}
    if "liq_proxy_volume" in df.columns:
        rename["liq_proxy_volume"] = "liq_total_volume"
    if "liq_proxy_buy" in df.columns:
        rename["liq_proxy_buy"] = "liq_buy_volume"
    if "liq_proxy_sell" in df.columns:
        rename["liq_proxy_sell"] = "liq_sell_volume"
    if "liq_proxy_cluster" in df.columns:
        rename["liq_proxy_cluster"] = "liq_count"
    df = df.rename(columns=rename)
    cols = [c for c in ["liq_total_volume", "liq_buy_volume", "liq_sell_volume", "liq_count"]
            if c in df.columns]
    sub = df[[ts_col] + cols].rename(columns={ts_col: "timestamp"})
    return DictCursor(sub)


# ── Model loading ────────────────────────────────────────────────

def _load_symbol_config(symbol: str) -> Dict[str, Any]:
    """Load model config for a symbol."""
    config_path = Path(f"models_v8/{symbol}_gate_v2/config.json")
    with open(config_path) as f:
        return json.load(f)


def _load_models(symbol: str, cfg: Dict[str, Any]):
    """Load ensemble models (LGBMAlphaModel + XGBAlphaModel) for a symbol."""
    from alpha.models.lgbm_alpha import LGBMAlphaModel
    from alpha.models.xgb_alpha import XGBAlphaModel

    model_dir = Path(f"models_v8/{symbol}_gate_v2")
    feature_names = cfg["features"]
    models = []
    weights = cfg.get("ensemble_weights", [])

    for fname in cfg.get("models", []):
        pkl_path = model_dir / fname
        if not pkl_path.exists():
            continue
        if "lgbm" in fname.lower():
            m = LGBMAlphaModel(name=f"lgbm_{symbol}", feature_names=feature_names)
            m.load(pkl_path)
            models.append(m)
        elif "xgb" in fname.lower():
            m = XGBAlphaModel(name=f"xgb_{symbol}", feature_names=feature_names)
            m.load(pkl_path)
            models.append(m)

    if len(weights) < len(models):
        weights = [1.0 / len(models)] * len(models)

    return models, weights


def _load_bear_model(cfg: Dict[str, Any]):
    """Load bear model if configured."""
    bear_path = cfg.get("bear_model_path")
    if not bear_path:
        return None
    bear_dir = Path(bear_path)
    bear_cfg_path = bear_dir / "config.json"
    if not bear_cfg_path.exists():
        return None

    with open(bear_cfg_path) as f:
        bear_cfg = json.load(f)

    from alpha.models.lgbm_alpha import LGBMAlphaModel
    bear_pkl = bear_dir / bear_cfg["models"][0]
    if not bear_pkl.exists():
        return None

    m = LGBMAlphaModel(
        name="bear_classifier",
        feature_names=bear_cfg.get("features", []),
    )
    m.load(bear_pkl)
    m._is_classifier = True
    return m


# ── Per-symbol kernel backtest ───────────────────────────────────

def backtest_symbol_kernel(
    symbol: str,
    oos_bars: int,
    full: bool,
) -> Optional[Dict[str, Any]]:
    """Run kernel-aware backtest for a single symbol.

    Replays bars through FeatureComputeHook + LiveInferenceBridge,
    using the exact same code path as production.
    """
    from engine.feature_hook import FeatureComputeHook
    from alpha.inference.bridge import LiveInferenceBridge
    from features.cross_asset_computer import CrossAssetComputer

    cfg = _load_symbol_config(symbol)
    models, weights = _load_models(symbol, cfg)
    if not models:
        print(f"  [{symbol}] No models found")
        return None

    bear_model = _load_bear_model(cfg)
    pm = cfg.get("position_management", {})
    bear_thresholds = None
    if pm.get("bear_thresholds"):
        bear_thresholds = [tuple(x) for x in pm["bear_thresholds"]]

    # Build LiveInferenceBridge with production config
    min_hold = cfg.get("min_hold", 24)
    deadzone = cfg.get("deadzone", 0.5)
    long_only = cfg.get("long_only", False)
    monthly_gate_window = cfg.get("monthly_gate_window", 480)
    monthly_gate = cfg.get("monthly_gate", True)

    bridge = LiveInferenceBridge(
        models=models,
        min_hold_bars={symbol: min_hold},
        long_only_symbols={symbol} if long_only else set(),
        deadzone=deadzone,
        monthly_gate=monthly_gate,
        monthly_gate_window={symbol: monthly_gate_window},
        bear_model=bear_model,
        bear_thresholds=bear_thresholds,
        vol_target=pm.get("vol_target"),
        vol_feature=pm.get("vol_feature", "atr_norm_14"),
        ensemble_weights=weights,
    )

    # Load auxiliary data cursors
    funding_cursor = _load_funding_cursor(symbol)
    oi_cursor = _load_oi_cursor(symbol)
    ls_cursor = _load_ls_cursor(symbol)
    fgi_cursor = _load_fgi_cursor()
    macro_cursor = _load_macro_cursor()
    mempool_cursor = _load_mempool_cursor() if symbol == "BTCUSDT" else None
    liq_cursor = _load_liquidation_cursor(symbol)
    spot_cursor = _load_spot_close_cursor(symbol)
    iv_cursor = _load_iv_cursor(symbol)
    pcr_cursor = _load_pcr_cursor(symbol)
    onchain_cursor = _load_onchain_cursor(symbol)

    # Create mutable state for source callbacks
    _aux: Dict[str, Any] = {
        "funding": None, "oi": None, "ls": None, "fgi": None,
        "macro": None, "mempool": None, "liq": None,
        "spot_close": None, "iv": None, "pcr": None, "onchain": None,
    }

    # Cross-asset: only for non-BTC symbols
    cross_asset = CrossAssetComputer() if symbol != "BTCUSDT" else None

    # Build FeatureComputeHook (production wiring — all 12 sources)
    hook = FeatureComputeHook(
        computer=None,  # unused when _rust_push is called directly
        inference_bridge=bridge,
        warmup_bars=WARMUP_BARS,
        funding_rate_source=lambda: _aux["funding"],
        cross_asset_computer=cross_asset,
        oi_source=lambda: _aux["oi"],
        ls_ratio_source=lambda: _aux["ls"],
        fgi_source=lambda: _aux["fgi"],
        spot_close_source=lambda: _aux["spot_close"],
        implied_vol_source=lambda: _aux["iv"],
        put_call_ratio_source=lambda: _aux["pcr"],
        onchain_source=lambda: _aux["onchain"],
        mempool_source=lambda: _aux["mempool"],
        liquidation_source=lambda: _aux["liq"],
        macro_source=lambda: _aux["macro"],
    )

    # Load OHLCV data
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    if not csv_path.exists():
        print(f"  [{symbol}] Data file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    if not full:
        df = df.iloc[-oos_bars:].reset_index(drop=True)

    ts_col = "open_time" if "open_time" in df.columns else "timestamp"

    # If cross-asset needed, also load BTC data for benchmark
    btc_df = None
    if cross_asset is not None:
        btc_path = Path("data_files/BTCUSDT_1h.csv")
        if btc_path.exists():
            btc_full = pd.read_csv(btc_path)
            # Align BTC to same timestamp range
            btc_ts_col = "open_time" if "open_time" in btc_full.columns else "timestamp"
            min_ts = df[ts_col].iloc[0]
            btc_df = btc_full[btc_full[btc_ts_col] >= min_ts].reset_index(drop=True)
            # Build timestamp → index mapping for BTC
            btc_ts_to_idx = {}
            for i, t in enumerate(btc_df[btc_ts_col].values):
                btc_ts_to_idx[int(t)] = i

    # Replay bars
    n = len(df)
    timestamps = df[ts_col].values.astype(np.int64)
    signals = np.zeros(n)
    closes = np.zeros(n)
    funding_rates = np.zeros(n)

    btc_funding_cursor = _load_funding_cursor("BTCUSDT") if cross_asset else None

    for i in range(n):
        ts_ms = int(timestamps[i])
        row = df.iloc[i]

        # Advance auxiliary cursors
        if funding_cursor:
            _aux["funding"] = funding_cursor.advance_to(ts_ms)
            funding_rates[i] = _aux["funding"] if _aux["funding"] is not None else 0.0
        if oi_cursor:
            _aux["oi"] = oi_cursor.advance_to(ts_ms)
        if ls_cursor:
            _aux["ls"] = ls_cursor.advance_to(ts_ms)
        if fgi_cursor:
            _aux["fgi"] = fgi_cursor.advance_to(ts_ms)
        if spot_cursor:
            _aux["spot_close"] = spot_cursor.advance_to(ts_ms)
        if iv_cursor:
            _aux["iv"] = iv_cursor.advance_to(ts_ms)
        if pcr_cursor:
            _aux["pcr"] = pcr_cursor.advance_to(ts_ms)
        if onchain_cursor:
            _aux["onchain"] = onchain_cursor.advance_to(ts_ms)
        if macro_cursor:
            _aux["macro"] = macro_cursor.advance_to(ts_ms)
        if mempool_cursor:
            _aux["mempool"] = mempool_cursor.advance_to(ts_ms)
        if liq_cursor:
            _aux["liq"] = liq_cursor.advance_to(ts_ms)

        close_f = float(row["close"])
        closes[i] = close_f

        # Push BTC benchmark bar FIRST for cross-asset
        if cross_asset is not None and btc_df is not None:
            btc_idx = btc_ts_to_idx.get(ts_ms)
            if btc_idx is not None:
                btc_row = btc_df.iloc[btc_idx]
                btc_funding = None
                if btc_funding_cursor:
                    btc_funding = btc_funding_cursor.advance_to(ts_ms)
                cross_asset.on_bar(
                    "BTCUSDT",
                    close=float(btc_row["close"]),
                    high=float(btc_row["high"]),
                    low=float(btc_row["low"]),
                    funding_rate=btc_funding,
                )

        # Create synthetic bar event
        bar_ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        event = BarEvent(
            symbol=symbol,
            ts=bar_ts,
            close=close_f,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            volume=float(row["volume"]),
            quote_volume=float(row.get("quote_volume", 0) or 0),
            trades=int(row.get("trades", 0) or 0),
            taker_buy_volume=float(row.get("taker_buy_volume", 0) or 0),
            taker_buy_quote_volume=float(row.get("taker_buy_quote_volume", 0) or 0),
        )

        # Run production pipeline
        features = hook.on_event(event)

        if features is not None:
            score = features.get("ml_score", 0.0)
            if score is not None:
                signals[i] = float(score)

    # Apply per-symbol DD breaker from config
    dd_limit = pm.get("dd_limit")
    dd_cooldown = pm.get("dd_cooldown", 48)
    if dd_limit is not None:
        if dd_limit > 0:
            dd_limit = -dd_limit
        from research.backtesting.backtest_alpha_v8 import _apply_dd_breaker
        signals = _apply_dd_breaker(signals, closes, dd_limit, dd_cooldown)

    n_active = int(np.sum(signals != 0))
    print(f"  [{symbol}] {n} bars, active={n_active} ({n_active/n*100:.1f}%), "
          f"long={np.mean(signals > 0)*100:.1f}%, short={np.mean(signals < 0)*100:.1f}%")

    # Volatility for allocation
    vol_values = np.full(n, 0.02)

    return {
        "symbol": symbol,
        "timestamps": timestamps,
        "closes": closes,
        "signal": signals,
        "vol": vol_values,
        "funding_rates": funding_rates,
        "n": n,
    }


# ── Portfolio simulation (shared with backtest_portfolio.py) ─────

def _compute_weights(method: str, symbol_data: List[Dict], bar_idx: int) -> np.ndarray:
    n_sym = len(symbol_data)
    if method == "equal" or n_sym == 0:
        return np.ones(max(n_sym, 1)) / max(n_sym, 1)
    if method == "inverse_vol":
        vols = np.array([sd["vol"][bar_idx] if bar_idx < sd["n"] else 0.02
                         for sd in symbol_data])
        vols = np.clip(vols, 0.001, None)
        inv = 1.0 / vols
        return inv / inv.sum()
    return np.ones(n_sym) / n_sym


def run_kernel_backtest(
    symbols: List[str],
    alloc_method: str = "equal",
    max_leverage: float = 1.0,
    dd_limit: float = -0.15,
    dd_cooldown: int = 48,
    oos_bars: int = 13140,
    full: bool = False,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run multi-symbol kernel-aware portfolio backtest."""
    print(f"\n{'='*70}")
    print(f"  Kernel-Aware Portfolio Backtest: {' + '.join(symbols)}")
    print(f"  Allocation: {alloc_method}, Max leverage: {max_leverage}")
    print(f"  DD limit: {dd_limit*100:.1f}%, OOS bars: {oos_bars}")
    print(f"{'='*70}")

    # Generate per-symbol signals through production kernel
    symbol_data = []
    for sym in symbols:
        result = backtest_symbol_kernel(sym, oos_bars, full)
        if result is not None:
            symbol_data.append(result)

    if not symbol_data:
        print("  No valid symbols. Exiting.")
        return {}

    # Find common timestamp range
    all_ts = [set(sd["timestamps"]) for sd in symbol_data]
    common_ts = sorted(all_ts[0].intersection(*all_ts[1:]))
    if len(common_ts) < 100:
        print(f"  Too few common bars: {len(common_ts)}")
        return {}

    print(f"\n  Common bars: {len(common_ts)}")

    # Align all symbols to common timestamps
    for sd in symbol_data:
        ts_to_idx = {int(ts): i for i, ts in enumerate(sd["timestamps"])}
        idxs = [ts_to_idx[ts] for ts in common_ts]
        sd["closes"] = sd["closes"][idxs]
        sd["signal"] = sd["signal"][idxs]
        sd["vol"] = sd["vol"][idxs]
        sd["funding_rates"] = sd["funding_rates"][idxs]
        sd["timestamps"] = np.array(common_ts, dtype=np.int64)
        sd["n"] = len(common_ts)

    n_bars = len(common_ts)
    n_sym = len(symbol_data)
    timestamps = np.array(common_ts, dtype=np.int64)

    # ── Simulate portfolio ──
    equity = np.ones(n_bars + 1) * INITIAL_CAPITAL
    portfolio_signal = np.zeros((n_bars, n_sym))
    per_symbol_pnl = np.zeros((n_bars, n_sym))
    portfolio_pnl = np.zeros(n_bars)

    cool_remaining = 0
    current_weights = np.ones(n_sym) / n_sym
    dd_peak = INITIAL_CAPITAL

    for t in range(n_bars):
        if cool_remaining > 0:
            cool_remaining -= 1
            equity[t + 1] = equity[t]
            if cool_remaining == 0:
                dd_peak = equity[t + 1]
            continue

        if t % 24 == 0:
            current_weights = _compute_weights(alloc_method, symbol_data, t)

        gross_exposure = 0.0
        for s in range(n_sym):
            raw_sig = symbol_data[s]["signal"][t]
            portfolio_signal[t, s] = raw_sig * current_weights[s]
            gross_exposure += abs(portfolio_signal[t, s])

        if gross_exposure > max_leverage:
            scale = max_leverage / gross_exposure
            portfolio_signal[t] *= scale

        if t < n_bars - 1:
            bar_pnl = 0.0
            for s in range(n_sym):
                ret = (symbol_data[s]["closes"][t + 1] - symbol_data[s]["closes"][t]) / symbol_data[s]["closes"][t]
                sig = portfolio_signal[t, s]
                prev_sig = portfolio_signal[t - 1, s] if t > 0 else 0.0
                turnover = abs(sig - prev_sig)
                cost = turnover * COST_PER_TRADE
                funding_cost = sig * symbol_data[s]["funding_rates"][t] / 8.0
                sym_pnl = sig * ret - cost - abs(funding_cost)
                per_symbol_pnl[t, s] = sym_pnl
                bar_pnl += sym_pnl

            portfolio_pnl[t] = bar_pnl
            equity[t + 1] = equity[t] * (1 + bar_pnl)

            dd_peak = max(dd_peak, equity[t + 1])
            dd = (equity[t + 1] - dd_peak) / dd_peak
            if dd < dd_limit:
                cool_remaining = dd_cooldown

    equity[-1] = equity[-2]

    # ── Compute metrics ──
    final_equity = equity[-1]
    total_return = (final_equity / INITIAL_CAPITAL) - 1.0
    annual_return = (1 + total_return) ** (8760 / max(n_bars, 1)) - 1.0

    active = portfolio_pnl != 0
    n_active = int(active.sum())
    sharpe = 0.0
    if n_active > 1:
        active_pnl = portfolio_pnl[active]
        std_a = float(np.std(active_pnl, ddof=1))
        if std_a > 0:
            sharpe = float(np.mean(active_pnl)) / std_a * np.sqrt(8760)

    peak_eq = np.maximum.accumulate(equity)
    dd = (equity - peak_eq) / peak_eq
    max_dd = float(np.min(dd))

    # Print results
    print(f"\n{'='*70}")
    print("  KERNEL BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"  Period: {n_bars} bars")
    print(f"  Initial: ${INITIAL_CAPITAL:,.0f} -> Final: ${final_equity:,.2f}")
    print(f"  Total return: {total_return*100:+.2f}%")
    print(f"  Annual return: {annual_return*100:+.2f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Max DD: {max_dd*100:.2f}%")

    print("\n  --- Per-Symbol Attribution ---")
    print(f"  {'Symbol':<12} {'Return':>8} {'Contribution':>14}")
    print(f"  {'-'*36}")
    sym_summaries = {}
    for s in range(n_sym):
        sym = symbol_data[s]["symbol"]
        sym_total_pnl = float(np.sum(per_symbol_pnl[:, s]))
        print(f"  {sym:<12} {sym_total_pnl*100:>+7.2f}% {sym_total_pnl*100:>+13.4f}%")
        sym_summaries[sym] = {"contribution": sym_total_pnl}

    # Monthly breakdown
    dt_list = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts in timestamps[:n_bars]]
    month_keys = [f"{d.year}-{d.month:02d}" for d in dt_list]
    unique_months = sorted(set(month_keys))

    monthly = []
    print("\n  --- Monthly Breakdown ---")
    print(f"  {'Month':<10} {'Return':>8} {'Sharpe':>8}")
    print(f"  {'-'*28}")
    for mk in unique_months:
        mask = np.array([m == mk for m in month_keys])
        if mask.sum() < 10:
            continue
        m_pnl = portfolio_pnl[mask]
        m_ret = float(np.sum(m_pnl))
        m_active = m_pnl != 0
        m_sharpe = 0.0
        if m_active.sum() > 1:
            m_std = float(np.std(m_pnl[m_active], ddof=1))
            if m_std > 0:
                m_sharpe = float(np.mean(m_pnl[m_active])) / m_std * np.sqrt(8760)
        monthly.append({"month": mk, "return": m_ret, "sharpe": m_sharpe})
        print(f"  {mk:<10} {m_ret*100:>+7.2f}% {m_sharpe:>8.2f}")

    pos_months = sum(1 for m in monthly if m["return"] > 0)
    print(f"  {'-'*28}")
    print(f"  Positive months: {pos_months}/{len(monthly)}")

    # ── Save results ──
    if out_dir is None:
        out_dir = Path(f"results/kernel_backtest/{'_'.join(symbols)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    eq_df = pd.DataFrame({
        "timestamp": timestamps,
        "equity": equity[:-1],
        "portfolio_pnl": portfolio_pnl,
    })
    for s in range(n_sym):
        eq_df[f"{symbol_data[s]['symbol']}_pnl"] = per_symbol_pnl[:, s]
    eq_df.to_csv(out_dir / "equity_curve.csv", index=False)
    pd.DataFrame(monthly).to_csv(out_dir / "monthly.csv", index=False)

    summary = {
        "type": "kernel_backtest",
        "symbols": symbols,
        "allocation": alloc_method,
        "max_leverage": max_leverage,
        "dd_limit": dd_limit,
        "n_bars": n_bars,
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": float(final_equity),
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "pos_months": pos_months,
        "total_months": len(monthly),
        "per_symbol": sym_summaries,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to {out_dir}/")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Kernel-aware portfolio backtest")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--alloc", choices=["equal", "inverse_vol"], default="equal")
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--dd-limit", type=float, default=-0.15)
    parser.add_argument("--dd-cooldown", type=int, default=48)
    parser.add_argument("--oos-bars", type=int, default=13140)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    os.environ.setdefault("QUANT_ALLOW_UNSIGNED_MODELS", "1")

    out_dir = Path(args.out) if args.out else None
    run_kernel_backtest(
        symbols=[s.upper() for s in args.symbols],
        alloc_method=args.alloc,
        max_leverage=args.leverage,
        dd_limit=args.dd_limit,
        dd_cooldown=args.dd_cooldown,
        oos_bars=args.oos_bars,
        full=args.full,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
