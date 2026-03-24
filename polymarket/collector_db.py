"""Database initialization and storage helpers for PolymarketCollector.

Extracted from collector.py to keep file sizes manageable.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)


def init_db(db_path: str) -> None:
    """Create all required tables and indexes."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_utc TEXT NOT NULL,
            window_start_ts INTEGER NOT NULL,
            slug TEXT NOT NULL,
            up_price REAL,
            down_price REAL,
            volume REAL,
            binance_btc_open REAL,
            binance_btc_close REAL,
            binance_result TEXT,
            polymarket_result TEXT,
            final_volume REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_window_ts
        ON market_snapshots(window_start_ts)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS intra_window_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            window_start_ts INTEGER NOT NULL,
            sample_time_utc TEXT NOT NULL,
            elapsed_sec INTEGER NOT NULL,
            binance_price REAL NOT NULL,
            strike_price REAL,
            move_bps REAL,
            fair_value_up REAL,
            fair_value_down REAL,
            polymarket_up_price REAL,
            polymarket_down_price REAL,
            pricing_delay REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # V2 schema: CLOB orderbook data + RSI signal
    conn.execute("""
        CREATE TABLE IF NOT EXISTS intra_window_samples_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            window_start_ts INTEGER NOT NULL,
            sample_time_utc TEXT NOT NULL,
            elapsed_sec INTEGER NOT NULL,
            binance_price REAL NOT NULL,
            strike_price REAL,
            move_bps REAL,
            fair_value_up REAL,
            fair_value_down REAL,
            clob_up_best_bid REAL,
            clob_up_best_ask REAL,
            clob_up_spread REAL,
            clob_up_bid_depth REAL,
            clob_up_ask_depth REAL,
            clob_down_best_bid REAL,
            clob_down_best_ask REAL,
            clob_up_mid REAL,
            pricing_delay REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Add RSI columns if not yet present (safe migration)
    try:
        conn.execute("ALTER TABLE intra_window_samples_v2 ADD COLUMN rsi_5 REAL")
    except sqlite3.OperationalError:
        pass  # column already exists
    try:
        conn.execute("ALTER TABLE intra_window_samples_v2 ADD COLUMN rsi_signal TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE intra_window_samples_v2 ADD COLUMN btc_ret_3bar REAL")
    except sqlite3.OperationalError:
        pass

    # 15m market samples table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS samples_15m (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            window_start_ts INTEGER NOT NULL,
            sample_time_utc TEXT NOT NULL,
            elapsed_sec INTEGER NOT NULL,
            binance_price REAL NOT NULL,
            strike_price REAL,
            move_bps REAL,
            fair_value_up REAL,
            fair_value_down REAL,
            clob_up_best_bid REAL,
            clob_up_best_ask REAL,
            clob_up_spread REAL,
            clob_up_bid_depth REAL,
            clob_up_ask_depth REAL,
            clob_down_best_bid REAL,
            clob_down_best_ask REAL,
            clob_up_mid REAL,
            pricing_delay REAL,
            rsi_5 REAL,
            rsi_signal TEXT,
            btc_ret_3bar REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_samples_15m_window_ts
        ON samples_15m(window_start_ts)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_intra_v2_window_ts
        ON intra_window_samples_v2(window_start_ts)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_intra_window_ts
        ON intra_window_samples(window_start_ts)
    """)
    conn.commit()
    conn.close()


def store_snapshot(db_path: str, record: dict) -> None:
    """Insert a market snapshot row."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO market_snapshots
        (timestamp_utc, window_start_ts, slug, up_price, down_price,
         volume, binance_btc_open)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["timestamp_utc"],
            record["window_start_ts"],
            record["slug"],
            record.get("up_price"),
            record.get("down_price"),
            record.get("volume", 0),
            record.get("binance_btc_open"),
        ),
    )
    conn.commit()
    conn.close()


def update_result(
    db_path: str,
    window_ts: int,
    polymarket_result: Optional[str] = None,
    binance_result: Optional[str] = None,
    btc_close: Optional[float] = None,
    final_volume: Optional[float] = None,
) -> None:
    """Update a market snapshot row with settlement results."""
    conn = sqlite3.connect(db_path)
    sets: list[str] = []
    params: list = []
    if polymarket_result is not None:
        sets.append("polymarket_result = ?")
        params.append(polymarket_result)
    if binance_result is not None:
        sets.append("binance_result = ?")
        params.append(binance_result)
    if btc_close is not None:
        sets.append("binance_btc_close = ?")
        params.append(btc_close)
    if final_volume is not None:
        sets.append("final_volume = ?")
        params.append(final_volume)
    if sets:
        params.append(window_ts)
        conn.execute(
            f"UPDATE market_snapshots SET {', '.join(sets)} WHERE window_start_ts = ?",
            params,
        )
        conn.commit()
    conn.close()


def store_intra_sample(db_path: str, sample: dict) -> None:
    """Store an intra-window sample to the legacy v1 table."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO intra_window_samples
        (window_start_ts, sample_time_utc, elapsed_sec, binance_price,
         strike_price, move_bps, fair_value_up, fair_value_down,
         polymarket_up_price, polymarket_down_price, pricing_delay)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sample["window_start_ts"],
            sample["sample_time_utc"],
            sample["elapsed_sec"],
            sample["binance_price"],
            sample.get("strike_price"),
            sample.get("move_bps"),
            sample.get("fair_value_up"),
            sample.get("fair_value_down"),
            sample.get("polymarket_up_price"),
            sample.get("polymarket_down_price"),
            sample.get("pricing_delay"),
        ),
    )
    conn.commit()
    conn.close()


def store_intra_sample_v2(db_path: str, sample: dict) -> None:
    """Store an intra-window sample with CLOB orderbook data + RSI."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO intra_window_samples_v2
        (window_start_ts, sample_time_utc, elapsed_sec, binance_price,
         strike_price, move_bps, fair_value_up, fair_value_down,
         clob_up_best_bid, clob_up_best_ask, clob_up_spread,
         clob_up_bid_depth, clob_up_ask_depth,
         clob_down_best_bid, clob_down_best_ask,
         clob_up_mid, pricing_delay,
         rsi_5, rsi_signal, btc_ret_3bar)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sample["window_start_ts"],
            sample["sample_time_utc"],
            sample["elapsed_sec"],
            sample["binance_price"],
            sample.get("strike_price"),
            sample.get("move_bps"),
            sample.get("fair_value_up"),
            sample.get("fair_value_down"),
            sample.get("clob_up_best_bid"),
            sample.get("clob_up_best_ask"),
            sample.get("clob_up_spread"),
            sample.get("clob_up_bid_depth"),
            sample.get("clob_up_ask_depth"),
            sample.get("clob_down_best_bid"),
            sample.get("clob_down_best_ask"),
            sample.get("clob_up_mid"),
            sample.get("pricing_delay"),
            sample.get("rsi_5"),
            sample.get("rsi_signal"),
            sample.get("btc_ret_3bar"),
        ),
    )
    conn.commit()
    conn.close()


def store_sample_15m(db_path: str, sample: dict) -> None:
    """Store a 15m window sample."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO samples_15m
        (window_start_ts, sample_time_utc, elapsed_sec, binance_price,
         strike_price, move_bps, fair_value_up, fair_value_down,
         clob_up_best_bid, clob_up_best_ask, clob_up_spread,
         clob_up_bid_depth, clob_up_ask_depth,
         clob_down_best_bid, clob_down_best_ask,
         clob_up_mid, pricing_delay,
         rsi_5, rsi_signal, btc_ret_3bar)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sample["window_start_ts"],
            sample["sample_time_utc"],
            sample["elapsed_sec"],
            sample["binance_price"],
            sample.get("strike_price"),
            sample.get("move_bps"),
            sample.get("fair_value_up"),
            sample.get("fair_value_down"),
            sample.get("clob_up_best_bid"),
            sample.get("clob_up_best_ask"),
            sample.get("clob_up_spread"),
            sample.get("clob_up_bid_depth"),
            sample.get("clob_up_ask_depth"),
            sample.get("clob_down_best_bid"),
            sample.get("clob_down_best_ask"),
            sample.get("clob_up_mid"),
            sample.get("pricing_delay"),
            sample.get("rsi_5"),
            sample.get("rsi_signal"),
            sample.get("btc_ret_3bar"),
        ),
    )
    conn.commit()
    conn.close()


def get_stats(db_path: str, sigma_annual: float) -> dict:
    """Return collection statistics."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT COUNT(*), MIN(timestamp_utc), MAX(timestamp_utc) FROM market_snapshots"
    ).fetchone()
    results = conn.execute(
        "SELECT polymarket_result, COUNT(*) FROM market_snapshots "
        "WHERE polymarket_result IS NOT NULL GROUP BY polymarket_result"
    ).fetchall()

    # Intra-window stats (v1 legacy)
    intra_row = conn.execute(
        "SELECT COUNT(*) FROM intra_window_samples"
    ).fetchone()
    intra_count = intra_row[0] if intra_row else 0

    # V2 CLOB stats
    intra_v2_row = conn.execute(
        "SELECT COUNT(*) FROM intra_window_samples_v2"
    ).fetchone()
    intra_v2_count = intra_v2_row[0] if intra_v2_row else 0

    # Average pricing delay from v2 CLOB data
    accuracy_row = conn.execute(
        "SELECT AVG(ABS(pricing_delay)), AVG(clob_up_spread) "
        "FROM intra_window_samples_v2 WHERE pricing_delay IS NOT NULL"
    ).fetchone()
    avg_pricing_delay = accuracy_row[0] if accuracy_row and accuracy_row[0] is not None else None
    avg_clob_spread = accuracy_row[1] if accuracy_row and accuracy_row[1] is not None else None

    conn.close()

    stats = {
        "total_records": row[0],
        "first_record": row[1],
        "last_record": row[2],
        "results": {r[0]: r[1] for r in results},
        "intra_window_samples_v1": intra_count,
        "intra_window_samples_v2": intra_v2_count,
        "avg_pricing_delay": avg_pricing_delay,
        "avg_clob_spread": avg_clob_spread,
        "current_sigma_annual": sigma_annual,
    }
    return stats
