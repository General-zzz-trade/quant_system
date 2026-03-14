"""Analyze collected Polymarket 5m BTC Up/Down data.

Usage:
    python3 -m polymarket.collector_analyzer [--db data/polymarket/collector.db]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, "/quant_system")


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def overview(conn: sqlite3.Connection):
    """Print basic collection statistics."""
    row = conn.execute(
        "SELECT COUNT(*), MIN(timestamp_utc), MAX(timestamp_utc) FROM market_snapshots"
    ).fetchone()
    print("=" * 60)
    print("OVERVIEW")
    print("=" * 60)
    print(f"  Total records : {row[0]}")
    print(f"  First record  : {row[1]}")
    print(f"  Last record   : {row[2]}")
    print()


def win_ratio(conn: sqlite3.Connection):
    """Up/Down win ratio from both Polymarket and Binance results."""
    print("=" * 60)
    print("WIN RATIOS")
    print("=" * 60)
    for col, label in [
        ("polymarket_result", "Polymarket"),
        ("binance_result", "Binance"),
    ]:
        rows = conn.execute(
            f"SELECT {col}, COUNT(*) FROM market_snapshots "
            f"WHERE {col} IS NOT NULL GROUP BY {col}"
        ).fetchall()
        total = sum(r[1] for r in rows)
        if total == 0:
            print(f"  {label}: no data")
            continue
        print(f"  {label} (n={total}):")
        for r in rows:
            pct = r[1] / total * 100
            print(f"    {r[0]:>6s}: {r[1]:5d} ({pct:5.1f}%)")
    print()


def volume_stats(conn: sqlite3.Connection):
    """Average volume at snapshot and at settlement."""
    print("=" * 60)
    print("VOLUME")
    print("=" * 60)
    row = conn.execute(
        "SELECT AVG(volume), AVG(final_volume) FROM market_snapshots"
    ).fetchone()
    print(f"  Avg snapshot volume : ${row[0] or 0:,.0f}")
    print(f"  Avg final volume    : ${row[1] or 0:,.0f}")
    print()


def time_of_day_pattern(conn: sqlite3.Connection):
    """Show Up/Down frequency by hour of day (UTC)."""
    print("=" * 60)
    print("TIME-OF-DAY PATTERN (Polymarket result)")
    print("=" * 60)
    rows = conn.execute(
        "SELECT window_start_ts, polymarket_result FROM market_snapshots "
        "WHERE polymarket_result IS NOT NULL"
    ).fetchall()
    if not rows:
        print("  No settled records yet.")
        print()
        return

    hourly: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        hour = datetime.fromtimestamp(r[0], tz=timezone.utc).hour
        hourly[hour][r[1]] += 1

    print(f"  {'Hour':>4s}  {'Up':>5s}  {'Down':>5s}  {'Up%':>6s}")
    for h in range(24):
        up = hourly[h].get("Up", 0)
        down = hourly[h].get("Down", 0)
        total = up + down
        pct = (up / total * 100) if total else 0
        print(f"  {h:4d}  {up:5d}  {down:5d}  {pct:5.1f}%")
    print()


def pricing_bias(conn: sqlite3.Connection):
    """Check if initial Up/Down pricing is well-calibrated."""
    print("=" * 60)
    print("PRICING BIAS ANALYSIS")
    print("=" * 60)
    rows = conn.execute(
        "SELECT up_price, down_price, polymarket_result FROM market_snapshots "
        "WHERE up_price IS NOT NULL AND polymarket_result IS NOT NULL"
    ).fetchall()
    if not rows:
        print("  No settled records with pricing data yet.")
        print()
        return

    # Bin up_price into buckets and check actual win rate
    buckets: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        up_price = r[0]
        result = r[2]
        # 5% buckets
        bucket_lo = int(up_price * 20) * 5
        bucket_hi = bucket_lo + 5
        bucket_label = f"{bucket_lo}-{bucket_hi}%"
        buckets[bucket_label].append(result == "Up")

    print(f"  {'Up Price Bucket':>15s}  {'n':>5s}  {'Actual Up%':>10s}  {'Implied':>8s}")
    for label in sorted(buckets.keys()):
        wins = buckets[label]
        n = len(wins)
        actual_pct = sum(wins) / n * 100 if n else 0
        implied = label  # bucket range itself is the implied probability
        print(f"  {label:>15s}  {n:5d}  {actual_pct:9.1f}%  {implied:>8s}")
    print()


def binance_correlation(conn: sqlite3.Connection):
    """Correlation between Binance price movement and Polymarket result."""
    print("=" * 60)
    print("BINANCE vs POLYMARKET AGREEMENT")
    print("=" * 60)
    rows = conn.execute(
        "SELECT binance_result, polymarket_result FROM market_snapshots "
        "WHERE binance_result IS NOT NULL AND polymarket_result IS NOT NULL"
    ).fetchall()
    if not rows:
        print("  No records with both results yet.")
        print()
        return

    agree = sum(1 for r in rows if r[0] == r[1])
    total = len(rows)
    print(f"  Total compared : {total}")
    print(f"  Agreement      : {agree} ({agree / total * 100:.1f}%)")
    print(f"  Disagreement   : {total - agree} ({(total - agree) / total * 100:.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze Polymarket 5m BTC collector data")
    parser.add_argument("--db", default="data/polymarket/collector.db")
    args = parser.parse_args()

    conn = _connect(args.db)
    overview(conn)
    win_ratio(conn)
    volume_stats(conn)
    time_of_day_pattern(conn)
    pricing_bias(conn)
    binance_correlation(conn)
    conn.close()


if __name__ == "__main__":
    main()
