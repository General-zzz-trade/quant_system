"""Tick data collector for market maker backtesting.

Captures aggTrade + L2 depth (100ms) from Binance Futures to SQLite.
Computes and stores VPIN/imbalance via RustStreamingMicrostructure in real-time.

Usage:
    collector = TickCollector("ETHUSDT", db_path="data/ticks/ETHUSDT.db")
    collector.start()   # blocks, Ctrl-C to stop
    collector.stop()

    # Or as background thread:
    collector.start_background()
    ...
    collector.stop()
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class TickCollectorConfig:
    """Collector configuration."""
    symbol: str = "ETHUSDT"
    db_path: str = "data/ticks/ETHUSDT.db"

    # Depth stream — use @depth20 (full snapshot) not @depth (diff)
    depth_levels: int = 20          # L2 levels to store
    depth_interval: str = "100ms"   # @100ms or @500ms

    # Microstructure
    vpin_bucket_volume: float = 100.0
    vpin_n_buckets: int = 50
    trade_buffer_size: int = 200

    # Persistence
    batch_size: int = 500           # flush every N records
    flush_interval_s: float = 5.0   # or every N seconds
    vacuum_interval_h: float = 24.0 # VACUUM every N hours

    # Stats
    stats_interval_s: float = 60.0  # log stats every N seconds


class TickCollector:
    """Collects aggTrade + depth ticks to SQLite with real-time VPIN."""

    def __init__(self, cfg: TickCollectorConfig | None = None) -> None:
        self._cfg = cfg or TickCollectorConfig()
        self._db: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # Batching
        self._trade_batch: list[tuple] = []
        self._depth_batch: list[tuple] = []
        self._last_flush = time.monotonic()

        # Stats
        self._trade_count = 0
        self._depth_count = 0
        self._start_time = 0.0
        self._last_stats = 0.0
        self._last_vacuum = 0.0

        # Microstructure
        self._micro = None
        self._last_vpin = 0.0
        self._last_imbalance = 0.0

    def _init_db(self) -> None:
        """Create database and tables."""
        path = Path(self._cfg.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA cache_size=-32768")  # 32MB cache

        self._db.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms INTEGER NOT NULL,
                trade_id INTEGER NOT NULL,
                price REAL NOT NULL,
                qty REAL NOT NULL,
                side TEXT NOT NULL,
                vpin REAL,
                ob_imbalance REAL,
                recv_lat_us INTEGER
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS depth_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms INTEGER NOT NULL,
                last_update_id INTEGER NOT NULL,
                best_bid REAL NOT NULL,
                best_ask REAL NOT NULL,
                mid_price REAL NOT NULL,
                spread_bps REAL NOT NULL,
                bid_depth_5 REAL,
                ask_depth_5 REAL,
                bid_depth_20 REAL,
                ask_depth_20 REAL,
                vpin REAL,
                ob_imbalance REAL,
                weighted_mid REAL,
                depth_ratio REAL,
                bids_json TEXT,
                asks_json TEXT
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS microstructure (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms INTEGER NOT NULL,
                vpin REAL NOT NULL,
                ob_imbalance REAL,
                spread_bps REAL,
                weighted_mid REAL,
                depth_ratio REAL,
                ob_signal TEXT,
                trade_count INTEGER
            )
        """)

        # Indexes for efficient time-range queries
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts_ms)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_depth_ts ON depth_snapshots(ts_ms)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_micro_ts ON microstructure(ts_ms)")
        self._db.commit()
        log.info("Database initialized: %s", self._cfg.db_path)

    def _init_microstructure(self) -> None:
        """Initialize Rust streaming microstructure computer."""
        try:
            from _quant_hotpath import RustStreamingMicrostructure
            self._micro = RustStreamingMicrostructure(
                trade_buffer_size=self._cfg.trade_buffer_size,
                vpin_bucket_volume=self._cfg.vpin_bucket_volume,
                vpin_n_buckets=self._cfg.vpin_n_buckets,
            )
            log.info("RustStreamingMicrostructure initialized")
        except ImportError:
            log.warning("RustStreamingMicrostructure not available, VPIN disabled")
            self._micro = None

    def _on_trade(self, raw: str) -> None:
        """Process an aggTrade message."""
        try:
            from _quant_hotpath import rust_parse_agg_trade
            parsed = rust_parse_agg_trade(raw)
        except ImportError:
            parsed = self._parse_trade_python(raw)

        if parsed is None:
            return

        recv_time = time.monotonic()
        ts_ms = parsed["ts_ms"]
        price = float(parsed["price"])
        qty = float(parsed["qty"])
        side = parsed["side"]
        trade_id = parsed.get("trade_id", 0)

        # Update microstructure
        vpin = None
        imbalance = None
        if self._micro is not None:
            try:
                result = self._micro.on_trade(price, qty, side)
                vpin = result.get("vpin", 0.0)
                imbalance = result.get("ob_imbalance", 0.0)
                self._last_vpin = vpin
                self._last_imbalance = imbalance
            except Exception:
                pass

        # Latency: exchange ts vs receive time (approximate)
        recv_lat_us = int((recv_time - self._start_time) * 1e6) if self._start_time else 0

        row = (ts_ms, trade_id, price, qty, side, vpin, imbalance, recv_lat_us)

        with self._lock:
            self._trade_batch.append(row)
            self._trade_count += 1
            if len(self._trade_batch) >= self._cfg.batch_size:
                self._flush_trades()

    def _on_depth(self, raw: str) -> None:
        """Process a depth update message."""
        try:
            from _quant_hotpath import rust_parse_depth
            parsed = rust_parse_depth(raw, self._cfg.depth_levels)
        except ImportError:
            parsed = self._parse_depth_python(raw)

        if parsed is None:
            return

        ts_ms = parsed.get("ts_ms", 0)
        last_update_id = parsed.get("last_update_id", 0)
        bids = parsed.get("bids", [])
        asks = parsed.get("asks", [])

        if not bids or not asks:
            return

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        if best_bid <= 0 or best_ask <= 0:
            return

        mid = (best_bid + best_ask) / 2.0
        spread_bps = (best_ask - best_bid) / mid * 10000

        # Depth sums
        bid_depth_5 = sum(float(b[0]) * float(b[1]) for b in bids[:5])
        ask_depth_5 = sum(float(a[0]) * float(a[1]) for a in asks[:5])
        bid_depth_20 = sum(float(b[0]) * float(b[1]) for b in bids[:20])
        ask_depth_20 = sum(float(a[0]) * float(a[1]) for a in asks[:20])

        # Update microstructure
        vpin = self._last_vpin
        imbalance = self._last_imbalance
        weighted_mid = mid
        depth_ratio = bid_depth_5 / ask_depth_5 if ask_depth_5 > 0 else 1.0

        if self._micro is not None:
            try:
                bid_tuples = [(float(b[0]), float(b[1])) for b in bids[:5]]
                ask_tuples = [(float(a[0]), float(a[1])) for a in asks[:5]]
                result = self._micro.on_depth(bid_tuples, ask_tuples)
                vpin = result.get("vpin", vpin)
                imbalance = result.get("ob_imbalance", imbalance)
                weighted_mid = result.get("weighted_mid", mid)
                depth_ratio = result.get("depth_ratio", depth_ratio)
                self._last_vpin = vpin
                self._last_imbalance = imbalance
            except Exception:
                pass

        # Compact JSON for top 5 levels only (saves space)
        bids_json = json.dumps([[b[0], b[1]] for b in bids[:5]])
        asks_json = json.dumps([[a[0], a[1]] for a in asks[:5]])

        row = (
            ts_ms, last_update_id, best_bid, best_ask, mid, spread_bps,
            bid_depth_5, ask_depth_5, bid_depth_20, ask_depth_20,
            vpin, imbalance, weighted_mid, depth_ratio,
            bids_json, asks_json,
        )

        with self._lock:
            self._depth_batch.append(row)
            self._depth_count += 1
            if len(self._depth_batch) >= self._cfg.batch_size:
                self._flush_depth()

    def _flush_trades(self) -> None:
        """Flush trade batch to database. Must hold self._lock."""
        if not self._trade_batch or self._db is None:
            return
        self._db.executemany(
            "INSERT INTO trades (ts_ms, trade_id, price, qty, side, vpin, ob_imbalance, recv_lat_us) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            self._trade_batch,
        )
        self._db.commit()
        self._trade_batch.clear()

    def _flush_depth(self) -> None:
        """Flush depth batch to database. Must hold self._lock."""
        if not self._depth_batch or self._db is None:
            return
        self._db.executemany(
            "INSERT INTO depth_snapshots "
            "(ts_ms, last_update_id, best_bid, best_ask, mid_price, spread_bps, "
            "bid_depth_5, ask_depth_5, bid_depth_20, ask_depth_20, "
            "vpin, ob_imbalance, weighted_mid, depth_ratio, bids_json, asks_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            self._depth_batch,
        )
        self._db.commit()
        self._depth_batch.clear()

    def _flush_all(self) -> None:
        """Flush all pending batches."""
        with self._lock:
            self._flush_trades()
            self._flush_depth()
        self._last_flush = time.monotonic()

    def _log_stats(self) -> None:
        """Log collection statistics."""
        now = time.monotonic()
        elapsed = now - self._start_time
        if elapsed <= 0:
            return
        tps = self._trade_count / elapsed
        dps = self._depth_count / elapsed

        # DB size
        try:
            db_size_mb = os.path.getsize(self._cfg.db_path) / 1e6
        except OSError:
            db_size_mb = 0

        log.info(
            "STATS elapsed=%.0fs trades=%d (%.1f/s) depth=%d (%.1f/s) "
            "vpin=%.3f imbalance=%.3f db=%.1fMB",
            elapsed, self._trade_count, tps, self._depth_count, dps,
            self._last_vpin, self._last_imbalance, db_size_mb,
        )
        self._last_stats = now

    def start(self) -> None:
        """Start collector (blocking). Ctrl-C to stop."""
        self._init_db()
        self._init_microstructure()
        self._running = True
        self._start_time = time.monotonic()
        self._last_stats = self._start_time
        self._last_vacuum = self._start_time

        log.info("Starting tick collector for %s", self._cfg.symbol)

        # Build streams
        sym_lower = self._cfg.symbol.lower()
        trade_stream = f"{sym_lower}@aggTrade"
        # Use @depth20 (full snapshot each message) not @depth (diff only)
        depth_stream = f"{sym_lower}@depth{self._cfg.depth_levels}@{self._cfg.depth_interval}"

        try:
            from execution.adapters.binance.ws_transport_websocket_client import (
                WebsocketClientTransport,
            )
        except ImportError:
            log.error("websocket-client not installed")
            return

        # Two separate WS connections: trades + depth
        trade_transport = WebsocketClientTransport()
        depth_transport = WebsocketClientTransport()

        base_url = "wss://fstream.binance.com/stream"
        trade_url = f"{base_url}?streams={trade_stream}"
        depth_url = f"{base_url}?streams={depth_stream}"

        log.info("Connecting trade stream: %s", trade_stream)
        trade_transport.connect(trade_url)

        log.info("Connecting depth stream: %s", depth_stream)
        depth_transport.connect(depth_url)

        # Trade receiver thread
        def _trade_loop():
            while self._running:
                try:
                    raw = trade_transport.recv(timeout_s=2.0)
                    if raw:
                        self._on_trade(raw)
                except Exception:
                    if self._running:
                        log.exception("Trade recv error")
                        time.sleep(1)

        trade_thread = threading.Thread(target=_trade_loop, daemon=True, name="tick-trades")
        trade_thread.start()

        # Main loop: depth + flush + stats
        log.info("Collector running. Ctrl-C to stop.")
        try:
            while self._running:
                try:
                    raw = depth_transport.recv(timeout_s=2.0)
                    if raw:
                        self._on_depth(raw)
                except Exception:
                    if self._running:
                        log.exception("Depth recv error")
                        time.sleep(1)

                now = time.monotonic()

                # Periodic flush
                if now - self._last_flush > self._cfg.flush_interval_s:
                    self._flush_all()

                # Periodic stats
                if now - self._last_stats > self._cfg.stats_interval_s:
                    self._log_stats()

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — stopping collector")
        finally:
            self._running = False
            self._flush_all()
            trade_transport.close()
            depth_transport.close()
            if self._db:
                self._db.close()
            self._log_stats()
            log.info("Collector stopped")

    def start_background(self) -> None:
        """Start collector in a background thread."""
        self._thread = threading.Thread(target=self.start, daemon=True, name="tick-collector")
        self._thread.start()

    def stop(self) -> None:
        """Stop collector gracefully."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    # ── Python fallback parsers ──────────────────────────────

    @staticmethod
    def _parse_trade_python(raw: str) -> dict | None:
        """Parse aggTrade JSON without Rust."""
        try:
            msg = json.loads(raw)
            data = msg.get("data", msg)
            if data.get("e") != "aggTrade":
                return None
            return {
                "ts_ms": data["T"],
                "trade_id": data["a"],
                "price": data["p"],
                "qty": data["q"],
                "side": "sell" if data.get("m", False) else "buy",
            }
        except (json.JSONDecodeError, KeyError):
            return None

    @staticmethod
    def _parse_depth_python(raw: str) -> dict | None:
        """Parse depth JSON without Rust."""
        try:
            msg = json.loads(raw)
            data = msg.get("data", msg)
            return {
                "ts_ms": data.get("T", data.get("E", 0)),
                "last_update_id": data.get("u", 0),
                "bids": data.get("b", []),
                "asks": data.get("a", []),
            }
        except (json.JSONDecodeError, KeyError):
            return None
