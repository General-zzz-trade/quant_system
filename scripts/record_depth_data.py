#!/usr/bin/env python3
"""Record depth data — aggregates live OB snapshots into bar-level CSV.

Runs the depth stream and saves aggregated bar-level orderbook features.
Run for 2-4 weeks to accumulate data for backtesting.

Usage:
    python3 -m scripts.record_depth_data --symbol BTCUSDT --bar-interval 3600
    python3 -m scripts.record_depth_data --symbol BTCUSDT --testnet
"""
from __future__ import annotations

import argparse
import csv
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from execution.adapters.binance.depth_processor import DepthProcessor
from execution.adapters.binance.ws_depth_stream import BinanceDepthStreamClient, DepthStreamConfig
from features.live_orderbook_features import LiveOrderbookFeatureAggregator, ORDERBOOK_FEATURE_NAMES

logger = logging.getLogger(__name__)

_running = True


def _handle_signal(sig, frame):
    global _running
    _running = False
    print("\nShutting down...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record depth data")
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--bar-interval", type=int, default=3600, help="Bar interval in seconds")
    parser.add_argument("--out", default="data_files", help="Output directory")
    parser.add_argument("--testnet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    symbol = args.symbol.lower()
    symbol_upper = symbol.upper()

    # Setup
    stream_name = f"{symbol}@depth20@100ms"
    ws_base = ("wss://stream.binancefuture.com/stream" if args.testnet
               else "wss://fstream.binance.com/stream")

    cfg = DepthStreamConfig(ws_base=ws_base, recv_timeout_sec=30.0)
    processor = DepthProcessor(max_levels=20)
    client = BinanceDepthStreamClient(
        transport=None,  # Will be created on connect
        processor=processor,
        streams=(stream_name,),
        cfg=cfg,
    )

    aggregator = LiveOrderbookFeatureAggregator(depth_levels=10)

    # Output CSV
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol_upper}_depth_features.csv"

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(f"Recording depth data for {symbol_upper}")
    print(f"  Bar interval: {args.bar_interval}s")
    print(f"  Output: {out_path}")
    print(f"  Press Ctrl+C to stop\n")

    # Write header if new file
    write_header = not out_path.exists()
    csv_file = open(out_path, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(["timestamp", "symbol"] + list(ORDERBOOK_FEATURE_NAMES) + ["n_snapshots"])

    bar_start = int(time.time()) // args.bar_interval * args.bar_interval
    next_bar = bar_start + args.bar_interval
    snapshot_count = 0

    try:
        while _running:
            try:
                snapshot = client.step()
                if snapshot is not None:
                    aggregator.on_depth(snapshot)
                    snapshot_count += 1
            except Exception as e:
                logger.warning("Depth stream error: %s", e)
                time.sleep(1)
                continue

            now = int(time.time())
            if now >= next_bar:
                feats = aggregator.flush_bar(symbol_upper)
                ts_ms = next_bar * 1000
                row = [ts_ms, symbol_upper]
                row.extend(feats.get(name) for name in ORDERBOOK_FEATURE_NAMES)
                row.append(snapshot_count)
                writer.writerow(row)
                csv_file.flush()

                dt = datetime.fromtimestamp(next_bar, tz=timezone.utc)
                logger.info("Bar %s: %d snapshots, imb=%.4f spread=%.2f",
                            dt.strftime("%H:%M"),
                            snapshot_count,
                            feats.get("ob_imbalance_mean") or 0,
                            feats.get("ob_spread_mean_bps") or 0)

                snapshot_count = 0
                next_bar += args.bar_interval
    finally:
        csv_file.close()
        try:
            client.close()
        except Exception:
            pass

    print(f"\nDone. Data saved to {out_path}")


if __name__ == "__main__":
    main()
