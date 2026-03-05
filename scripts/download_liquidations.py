#!/usr/bin/env python3
"""Generate liquidation proxy data from OI + volume data.

The Binance /fapi/v1/allForceOrders endpoint was deprecated.
For live data, the LiquidationPoller uses the WebSocket forceOrder stream.
For historical IC analysis, we generate proxy metrics from OI changes + volume
(large OI drops with volume spikes indicate liquidation cascades).

Output: data_files/{SYMBOL}_liquidation_proxy.csv

Usage:
    python scripts/download_liquidations.py --symbol BTCUSDT
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_liquidation_proxy(symbol: str, out_dir: str) -> Path:
    """Generate proxy liquidation metrics from OI + kline data."""
    out_path = Path(out_dir) / f"{symbol}_liquidation_proxy.csv"

    kline_path = Path(out_dir) / f"{symbol}_1h.csv"
    oi_path = Path(out_dir) / f"{symbol}_open_interest.csv"

    if not kline_path.exists():
        raise FileNotFoundError(f"Missing kline data: {kline_path}")
    if not oi_path.exists():
        raise FileNotFoundError(f"Missing OI data: {oi_path}")

    klines = pd.read_csv(kline_path)
    oi_df = pd.read_csv(oi_path)

    # Merge on timestamp
    ts_col_k = "timestamp" if "timestamp" in klines.columns else "open_time"
    ts_col_oi = "timestamp"

    klines = klines.rename(columns={ts_col_k: "ts"})
    oi_df = oi_df.rename(columns={ts_col_oi: "ts"})

    # Forward-fill OI into hourly klines
    merged = pd.merge_asof(
        klines.sort_values("ts"),
        oi_df[["ts", "sum_open_interest"]].sort_values("ts"),
        on="ts",
        direction="backward",
    )

    # Compute proxy metrics
    merged["oi"] = merged["sum_open_interest"].astype(float)
    merged["oi_change"] = merged["oi"].pct_change()
    merged["volume"] = merged["volume"].astype(float)
    merged["quote_volume"] = merged["quote_volume"].astype(float)

    # Proxy liquidation volume: |OI drop| * average trade size when OI drops sharply
    # Large OI decreases indicate forced closures
    oi_drop = merged["oi_change"].clip(upper=0).abs()  # only negative changes
    vol_spike = merged["quote_volume"] / merged["quote_volume"].rolling(24).mean()

    # Proxy: combine OI drop magnitude with volume spike
    merged["liq_proxy_volume"] = oi_drop * vol_spike * merged["quote_volume"]
    merged["liq_proxy_volume"] = merged["liq_proxy_volume"].fillna(0)

    # Direction proxy: if price went up during liquidation, shorts were squeezed (BUY side)
    price_change = merged["close"].astype(float).pct_change()
    merged["liq_proxy_buy"] = np.where(price_change > 0, merged["liq_proxy_volume"], 0)
    merged["liq_proxy_sell"] = np.where(price_change <= 0, merged["liq_proxy_volume"], 0)

    # Imbalance
    total = merged["liq_proxy_buy"] + merged["liq_proxy_sell"]
    merged["liq_proxy_imbalance"] = np.where(
        total > 0,
        (merged["liq_proxy_buy"] - merged["liq_proxy_sell"]) / total,
        0,
    )

    # Cluster: rolling 6-bar count of high-liquidation events (> 2 std)
    liq_mean = merged["liq_proxy_volume"].rolling(24).mean()
    liq_std = merged["liq_proxy_volume"].rolling(24).std()
    merged["liq_proxy_cluster"] = np.where(
        merged["liq_proxy_volume"] > liq_mean + 2 * liq_std, 1.0, 0.0
    )

    # Output
    out_cols = [
        "ts", "liq_proxy_volume", "liq_proxy_buy", "liq_proxy_sell",
        "liq_proxy_imbalance", "liq_proxy_cluster",
    ]
    merged[out_cols].to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(merged))
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate liquidation proxy data")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--out-dir", default="data_files")
    args = parser.parse_args()

    generate_liquidation_proxy(args.symbol, args.out_dir)


if __name__ == "__main__":
    main()
