"""Polymarket 5-minute BTC Up/Down data collector.

Collects real market data every 5 minutes:
- Polymarket market pricing (Up/Down shares)
- Settlement result (which side won)
- Concurrent Binance BTC/USDT price
- Volume and liquidity

Stores data in SQLite for later analysis.

Usage:
    python3 -m polymarket.collector [--db data/polymarket/collector.db] [--once]

Deployment:
    sudo cp infra/systemd/polymarket-collector.service /etc/systemd/system/
    sudo systemctl enable --now polymarket-collector
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_WINDOW_SEC = 300  # 5 minutes
_SETTLE_OFFSET = 5  # seconds after boundary to let market settle


class PolymarketCollector:
    """Collects Polymarket 5m BTC Up/Down market data into SQLite."""

    def __init__(self, db_path: str = "data/polymarket/collector.db"):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._running = False

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _init_db(self):
        conn = sqlite3.connect(self._db_path)
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
        conn.commit()
        conn.close()

    def _store(self, record: dict):
        conn = sqlite3.connect(self._db_path)
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

    def _update_result(
        self,
        window_ts: int,
        polymarket_result: str | None = None,
        binance_result: str | None = None,
        btc_close: float | None = None,
        final_volume: float | None = None,
    ):
        conn = sqlite3.connect(self._db_path)
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

    # ------------------------------------------------------------------
    # External API helpers
    # ------------------------------------------------------------------

    def _get_binance_price(self) -> float:
        """Get current BTC/USDT price from Binance public API (no auth)."""
        url = "https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT"
        req = Request(url, headers={"Accept": "application/json"})
        try:
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return float(data.get("price", 0))
        except Exception as e:
            logger.warning("Failed to get Binance price: %s", e)
            return 0.0

    def _get_current_5m_market(self, window_ts: int) -> dict:
        """Fetch a 5m BTC up/down market from the Gamma API by window timestamp."""
        slug = f"btc-updown-5m-{window_ts}"
        url = f"https://gamma-api.polymarket.com/events?slug={slug}"
        req = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "quant-collector/1.0",
            },
        )
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                if not (data and isinstance(data, list) and len(data) > 0):
                    return {}
                ev = data[0]
                result: dict = {
                    "slug": slug,
                    "title": ev.get("title", ""),
                    "volume": float(ev.get("volume", 0) or 0),
                    "closed": ev.get("closed", False),
                }
                for m in ev.get("markets", []):
                    for t in m.get("tokens", []):
                        if not isinstance(t, dict):
                            continue
                        outcome = t.get("outcome", "")
                        price = float(t.get("price", 0) or 0)
                        winner = t.get("winner", False)
                        if outcome == "Up":
                            result["up_price"] = price
                            if winner:
                                result["winner"] = "Up"
                        elif outcome == "Down":
                            result["down_price"] = price
                            if winner:
                                result["winner"] = "Down"
                return result
        except Exception as e:
            logger.warning("Failed to fetch market %s: %s", slug, e)
        return {}

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _current_window_ts() -> int:
        """Start timestamp of the current 5-minute window."""
        now = int(time.time())
        return (now // _WINDOW_SEC) * _WINDOW_SEC

    @staticmethod
    def _next_window_ts() -> int:
        """Timestamp of the next 5-minute boundary."""
        now = int(time.time())
        return ((now // _WINDOW_SEC) + 1) * _WINDOW_SEC

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect_one(self) -> dict:
        """Collect one data point for the current 5m window.

        Also back-fills result for the *previous* window if available.
        Returns the recorded data dict.
        """
        now = datetime.now(timezone.utc)
        window_ts = self._current_window_ts()

        # 1. Fetch current market prices
        market = self._get_current_5m_market(window_ts)

        # 2. Get Binance BTC price at window start
        btc_open = self._get_binance_price()

        # 3. Back-fill previous window result
        prev_ts = window_ts - _WINDOW_SEC
        prev_market = self._get_current_5m_market(prev_ts)
        prev_poly_result = prev_market.get("winner")

        # Also derive Binance-based result from stored open vs current price
        btc_close_prev = btc_open  # approximate (now ~ prev close)
        self._backfill_previous(prev_ts, prev_poly_result, prev_market, btc_close_prev)

        # 4. Store current window snapshot
        record = {
            "timestamp_utc": now.strftime("%Y-%m-%dT%H:%M:%S"),
            "window_start_ts": window_ts,
            "slug": market.get("slug", f"btc-updown-5m-{window_ts}"),
            "up_price": market.get("up_price"),
            "down_price": market.get("down_price"),
            "volume": market.get("volume", 0),
            "binance_btc_open": btc_open,
        }
        self._store(record)

        logger.info(
            "Collected: ts=%d up=%.3f down=%.3f btc=$%.0f vol=$%.0f prev_result=%s",
            window_ts,
            record.get("up_price") or 0,
            record.get("down_price") or 0,
            btc_open,
            record.get("volume") or 0,
            prev_poly_result or "pending",
        )
        return record

    def _backfill_previous(
        self,
        prev_ts: int,
        poly_result: str | None,
        prev_market: dict,
        btc_close: float,
    ):
        """Update the previous window row with settlement info."""
        # Derive Binance-based result from stored open price
        conn = sqlite3.connect(self._db_path)
        row = conn.execute(
            "SELECT binance_btc_open FROM market_snapshots WHERE window_start_ts = ?",
            (prev_ts,),
        ).fetchone()
        conn.close()

        binance_result = None
        if row and row[0] and btc_close:
            prev_open = row[0]
            if btc_close > prev_open:
                binance_result = "Up"
            elif btc_close < prev_open:
                binance_result = "Down"
            else:
                binance_result = "Flat"

        self._update_result(
            prev_ts,
            polymarket_result=poly_result,
            binance_result=binance_result,
            btc_close=btc_close,
            final_volume=prev_market.get("volume"),
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return collection statistics."""
        conn = sqlite3.connect(self._db_path)
        row = conn.execute(
            "SELECT COUNT(*), MIN(timestamp_utc), MAX(timestamp_utc) FROM market_snapshots"
        ).fetchone()
        results = conn.execute(
            "SELECT polymarket_result, COUNT(*) FROM market_snapshots "
            "WHERE polymarket_result IS NOT NULL GROUP BY polymarket_result"
        ).fetchall()
        conn.close()
        return {
            "total_records": row[0],
            "first_record": row[1],
            "last_record": row[2],
            "results": {r[0]: r[1] for r in results},
        }

    # ------------------------------------------------------------------
    # Continuous run
    # ------------------------------------------------------------------

    def start(self):
        """Run collector continuously, aligned to 5-minute boundaries."""
        self._running = True
        logger.info("Polymarket collector starting (db=%s)", self._db_path)

        while self._running:
            try:
                self.collect_one()
            except Exception:
                logger.exception("Collection cycle failed")

            # Sleep until next 5-minute boundary + small offset
            next_boundary = self._next_window_ts()
            sleep_sec = max(1, next_boundary - time.time() + _SETTLE_OFFSET)
            logger.debug("Sleeping %.0fs until next window", sleep_sec)

            # Interruptible sleep (1s ticks)
            end_time = time.time() + sleep_sec
            while self._running and time.time() < end_time:
                time.sleep(1)

    def stop(self):
        """Signal the collector to stop after the current cycle."""
        self._running = False
        logger.info("Polymarket collector stopping")


if __name__ == "__main__":
    from polymarket.collector_main import main

    main()
