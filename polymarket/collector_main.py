"""Entry point for Polymarket 5m BTC data collector.

Usage:
    python3 -m polymarket.collector [--db data/polymarket/collector.db] [--once]
    python3 -m polymarket.collector --stats
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys

sys.path.insert(0, "/quant_system")


def main():
    parser = argparse.ArgumentParser(description="Polymarket 5m BTC data collector")
    parser.add_argument("--db", default="data/polymarket/collector.db")
    parser.add_argument("--once", action="store_true", help="Collect once and exit")
    parser.add_argument("--stats", action="store_true", help="Show collection stats")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--mode",
        choices=["basic", "intra"],
        default="intra",
        help="basic=one sample per window, intra=30s intra-window sampling (default: intra)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from polymarket.collector import PolymarketCollector

    collector = PolymarketCollector(db_path=args.db)

    if args.stats:
        stats = collector.get_stats()
        print(f"Records:              {stats['total_records']}")
        print(f"Range:                {stats['first_record']} -> {stats['last_record']}")
        print(f"Results:              {stats['results']}")
        print(f"Intra-window samples: {stats['intra_window_samples']}")
        if stats['avg_pricing_delay'] is not None:
            print(f"Avg pricing delay:    {stats['avg_pricing_delay']:.4f}")
        else:
            print(f"Avg pricing delay:    N/A (no Polymarket data yet)")
        print(f"Current sigma (ann):  {stats['current_sigma_annual']:.4f}")
        return

    if args.once:
        if args.mode == "intra":
            collector.collect_intra_window()
        else:
            collector.collect_one()
        return

    # Graceful shutdown on SIGTERM / SIGINT
    def shutdown(signum, frame):
        collector.stop()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    collector.start(mode=args.mode)


if __name__ == "__main__":
    main()
