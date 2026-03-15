"""Generate bridge checkpoint from historical data for z-score warmup.

Usage:
    python -m scripts.warm_bridge BTCUSDT
    python -m scripts.warm_bridge --all

Replays the last 820 hourly bars through the production pipeline
(FeatureComputeHook → RustFeatureEngine → LiveInferenceBridge)
to pre-fill the z-score buffer. Produces bridge_checkpoint.json
that the paper trading runner restores on startup, eliminating
the 168-hour warmup wait.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Extra bars beyond zscore_window for feature warmup
FEATURE_WARMUP = 100
ZSCORE_WINDOW = 720
WARMUP_TOTAL = ZSCORE_WINDOW + FEATURE_WARMUP


def warm_symbol(symbol: str) -> Path:
    """Generate and save checkpoint for a symbol's paper trading container."""
    from scripts.backtest_kernel import (
        _load_symbol_config, _load_models, _load_bear_model,
        _load_funding_cursor, _load_oi_cursor, _load_ls_cursor,
        _load_fgi_cursor, _load_macro_cursor, _load_mempool_cursor,
        _load_liquidation_cursor, _load_spot_close_cursor,
        _load_iv_cursor, _load_pcr_cursor, _load_onchain_cursor,
        WARMUP_BARS, BarEvent,
    )
    from engine.feature_hook import FeatureComputeHook
    from alpha.inference.bridge import LiveInferenceBridge
    from features.cross_asset_computer import CrossAssetComputer

    cfg = _load_symbol_config(symbol)
    models, weights = _load_models(symbol, cfg)
    if not models:
        raise RuntimeError(f"No models found for {symbol}")

    bear_model = _load_bear_model(cfg)
    pm = cfg.get("position_management", {})
    bear_thresholds = None
    if pm.get("bear_thresholds"):
        bear_thresholds = [tuple(x) for x in pm["bear_thresholds"]]

    bridge = LiveInferenceBridge(
        models=models,
        min_hold_bars={symbol: cfg.get("min_hold", 24)},
        long_only_symbols={symbol} if cfg.get("long_only") else set(),
        deadzone=cfg.get("deadzone", 0.5),
        monthly_gate=cfg.get("monthly_gate", True),
        monthly_gate_window={symbol: cfg.get("monthly_gate_window", 480)},
        bear_model=bear_model,
        bear_thresholds=bear_thresholds,
        vol_target=pm.get("vol_target"),
        vol_feature=pm.get("vol_feature", "atr_norm_14"),
        ensemble_weights=weights,
    )

    # Load cursors
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

    _aux: Dict[str, Any] = {
        "funding": None, "oi": None, "ls": None, "fgi": None,
        "macro": None, "mempool": None, "liq": None,
        "spot_close": None, "iv": None, "pcr": None, "onchain": None,
    }

    cross_asset = CrossAssetComputer() if symbol != "BTCUSDT" else None

    hook = FeatureComputeHook(
        computer=None,
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

    # Load OHLCV — only last WARMUP_TOTAL bars
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    df = pd.read_csv(csv_path)
    df = df.iloc[-WARMUP_TOTAL:].reset_index(drop=True)
    ts_col = "open_time" if "open_time" in df.columns else "timestamp"

    btc_df = None
    btc_ts_to_idx = {}
    btc_funding_cursor = None
    if cross_asset is not None:
        btc_path = Path("data_files/BTCUSDT_1h.csv")
        if btc_path.exists():
            btc_full = pd.read_csv(btc_path)
            btc_ts_col = "open_time" if "open_time" in btc_full.columns else "timestamp"
            min_ts = df[ts_col].iloc[0]
            btc_df = btc_full[btc_full[btc_ts_col] >= min_ts].reset_index(drop=True)
            for i, t in enumerate(btc_df[btc_ts_col].values):
                btc_ts_to_idx[int(t)] = i
        btc_funding_cursor = _load_funding_cursor("BTCUSDT")

    # Replay bars
    t0 = time.time()
    n = len(df)
    timestamps = df[ts_col].values.astype(np.int64)

    for i in range(n):
        ts_ms = int(timestamps[i])
        row = df.iloc[i]

        # Advance auxiliary cursors
        if funding_cursor:
            _aux["funding"] = funding_cursor.advance_to(ts_ms)
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

        # Push BTC benchmark bar for cross-asset
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

        bar_ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        event = BarEvent(
            symbol=symbol,
            ts=bar_ts,
            close=float(row["close"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            volume=float(row["volume"]),
            quote_volume=float(row.get("quote_volume", 0) or 0),
            trades=int(row.get("trades", 0) or 0),
            taker_buy_volume=float(row.get("taker_buy_volume", 0) or 0),
            taker_buy_quote_volume=float(row.get("taker_buy_quote_volume", 0) or 0),
        )

        hook.on_event(event)

    elapsed = time.time() - t0

    # Extract checkpoint from bridge
    ckpt = bridge.checkpoint()
    n_scores = len(ckpt.get("zscore_buf", {}).get(symbol, []))
    logger.info(
        "%s: %d z-score samples from %d bars in %.1fs",
        symbol, n_scores, n, elapsed,
    )

    # Save to the checkpoint path that run_paper() will read
    output_dir = Path("infra/config/examples/validation_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "bridge_checkpoint.json"

    # Merge with existing checkpoint (multi-symbol)
    if ckpt_path.exists():
        with ckpt_path.open() as f:
            existing = json.load(f)
        for key in ckpt:
            if isinstance(ckpt[key], dict) and isinstance(existing.get(key), dict):
                existing[key].update(ckpt[key])
            else:
                existing[key] = ckpt[key]
        ckpt = existing

    with ckpt_path.open("w") as f:
        json.dump(ckpt, f)

    return ckpt_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) < 2 or sys.argv[1] == "--all":
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    else:
        symbols = [sys.argv[1]]

    for sym in symbols:
        t0 = time.time()
        path = warm_symbol(sym)
        elapsed = time.time() - t0
        print(f"  {sym}: checkpoint -> {path} ({elapsed:.1f}s)")

    print("\nDone. Restart containers to apply warmup.")


if __name__ == "__main__":
    main()
