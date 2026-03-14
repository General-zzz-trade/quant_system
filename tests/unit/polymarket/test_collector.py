"""Tests for polymarket.collector — Polymarket 5m BTC data collector."""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
# importlib import mode may resolve tests/unit/polymarket as the 'polymarket'
# package, shadowing the real one.  Force-reload from project root.
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# Remove any cached reference to the test-directory 'polymarket' package
if "polymarket" in sys.modules:
    _cached = sys.modules["polymarket"]
    if "tests" in getattr(_cached, "__file__", ""):
        del sys.modules["polymarket"]
        for k in list(sys.modules):
            if k.startswith("polymarket."):
                del sys.modules[k]

import importlib
_pm = importlib.import_module("polymarket.collector")
PolymarketCollector = _pm.PolymarketCollector


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_collector.db")


@pytest.fixture
def collector(db_path):
    return PolymarketCollector(db_path=db_path)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_gamma_response(slug: str, up_price: float = 0.52, down_price: float = 0.48,
                         volume: float = 5000, winner: str | None = None) -> bytes:
    """Build a fake Gamma API response."""
    up_token = {"outcome": "Up", "price": up_price, "winner": winner == "Up"}
    down_token = {"outcome": "Down", "price": down_price, "winner": winner == "Down"}
    event = {
        "title": f"BTC 5m {slug}",
        "slug": slug,
        "volume": volume,
        "closed": winner is not None,
        "markets": [{"tokens": [up_token, down_token]}],
    }
    return json.dumps([event]).encode()


def _make_binance_response(price: float = 83000.0) -> bytes:
    return json.dumps({"symbol": "BTCUSDT", "price": str(price)}).encode()


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestInitCreatesDb:
    def test_init_creates_db(self, db_path, collector):
        """SQLite DB is created with the correct schema on init."""
        conn = sqlite3.connect(db_path)
        # Check table exists
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='market_snapshots'"
        ).fetchall()
        assert len(tables) == 1

        # Check columns
        cols = conn.execute("PRAGMA table_info(market_snapshots)").fetchall()
        col_names = {c[1] for c in cols}
        expected = {
            "id", "timestamp_utc", "window_start_ts", "slug",
            "up_price", "down_price", "volume",
            "binance_btc_open", "binance_btc_close",
            "binance_result", "polymarket_result",
            "final_volume", "created_at",
        }
        assert expected.issubset(col_names)
        conn.close()

    def test_init_creates_index(self, db_path, collector):
        conn = sqlite3.connect(db_path)
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_window_ts'"
        ).fetchall()
        assert len(indexes) == 1
        conn.close()


class TestStoreAndRetrieve:
    def test_store_and_retrieve(self, collector, db_path):
        """Record stored and queryable."""
        record = {
            "timestamp_utc": "2026-03-14T12:00:00",
            "window_start_ts": 1773496800,
            "slug": "btc-updown-5m-1773496800",
            "up_price": 0.52,
            "down_price": 0.48,
            "volume": 5000,
            "binance_btc_open": 83000.0,
        }
        collector._store(record)

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT * FROM market_snapshots WHERE window_start_ts = ?",
            (1773496800,),
        ).fetchone()
        conn.close()

        assert row is not None
        # Check values (columns by index: 0=id, 1=ts_utc, 2=window_ts, 3=slug, ...)
        assert row[2] == 1773496800
        assert row[3] == "btc-updown-5m-1773496800"
        assert abs(row[4] - 0.52) < 1e-6  # up_price
        assert abs(row[5] - 0.48) < 1e-6  # down_price


class TestCurrentWindowTsAligned:
    def test_current_window_ts_aligned(self, collector):
        """Timestamps are aligned to 5-minute boundaries."""
        ts = collector._current_window_ts()
        assert ts % 300 == 0

    def test_next_window_is_300s_ahead(self, collector):
        nxt = collector._next_window_ts()
        cur = collector._current_window_ts()
        # next should be either 300s ahead of current, or same if exactly on boundary
        assert nxt - cur in (300,)


class TestUpdateResult:
    def test_update_result(self, collector, db_path):
        """Result update modifies the correct row."""
        record = {
            "timestamp_utc": "2026-03-14T12:00:00",
            "window_start_ts": 1773496800,
            "slug": "btc-updown-5m-1773496800",
            "up_price": 0.52,
            "down_price": 0.48,
            "volume": 5000,
            "binance_btc_open": 83000.0,
        }
        collector._store(record)
        collector._update_result(
            1773496800,
            polymarket_result="Up",
            binance_result="Up",
            btc_close=83100.0,
            final_volume=8000.0,
        )

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT polymarket_result, binance_result, binance_btc_close, final_volume "
            "FROM market_snapshots WHERE window_start_ts = ?",
            (1773496800,),
        ).fetchone()
        conn.close()

        assert row[0] == "Up"
        assert row[1] == "Up"
        assert abs(row[2] - 83100.0) < 0.1
        assert abs(row[3] - 8000.0) < 0.1


class TestGetStats:
    def test_get_stats_empty(self, collector):
        stats = collector.get_stats()
        assert stats["total_records"] == 0
        assert stats["results"] == {}

    def test_get_stats_with_data(self, collector):
        for i, result in enumerate(["Up", "Up", "Down"]):
            ts = 1773496800 + i * 300
            collector._store({
                "timestamp_utc": f"2026-03-14T12:{i * 5:02d}:00",
                "window_start_ts": ts,
                "slug": f"btc-updown-5m-{ts}",
                "volume": 1000,
            })
            collector._update_result(ts, polymarket_result=result)

        stats = collector.get_stats()
        assert stats["total_records"] == 3
        assert stats["results"]["Up"] == 2
        assert stats["results"]["Down"] == 1


class TestCollectOneWithMockedApis:
    @patch("polymarket.collector.urlopen")
    def test_collect_one_full_cycle(self, mock_urlopen, collector, db_path):
        """Full collection cycle with mocked Gamma + Binance API."""
        window_ts = (int(time.time()) // 300) * 300
        prev_ts = window_ts - 300

        # urlopen is called multiple times:
        # 1. current market (gamma)
        # 2. binance price
        # 3. previous market (gamma) for backfill
        current_gamma = _make_gamma_response(
            f"btc-updown-5m-{window_ts}", up_price=0.51, down_price=0.49, volume=3000,
        )
        binance_resp = _make_binance_response(83500.0)
        prev_gamma = _make_gamma_response(
            f"btc-updown-5m-{prev_ts}", up_price=0.50, down_price=0.50,
            volume=7000, winner="Up",
        )

        # Mock urlopen to return different responses based on URL
        def fake_urlopen(req, **kwargs):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            resp = MagicMock()
            if "gamma-api" in url and str(window_ts) in url:
                resp.read.return_value = current_gamma
            elif "gamma-api" in url and str(prev_ts) in url:
                resp.read.return_value = prev_gamma
            elif "binance" in url:
                resp.read.return_value = binance_resp
            else:
                resp.read.return_value = b"[]"
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        mock_urlopen.side_effect = fake_urlopen

        record = collector.collect_one()

        assert record["window_start_ts"] == window_ts
        assert abs(record["up_price"] - 0.51) < 1e-6
        assert abs(record["down_price"] - 0.49) < 1e-6
        assert abs(record["binance_btc_open"] - 83500.0) < 0.1

        # Verify stored in DB
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT * FROM market_snapshots WHERE window_start_ts = ?",
            (window_ts,),
        ).fetchone()
        conn.close()
        assert row is not None

    @patch("polymarket.collector.urlopen")
    def test_collect_one_handles_api_failure(self, mock_urlopen, collector):
        """Collector handles API failures gracefully."""
        mock_urlopen.side_effect = Exception("Connection refused")

        # Should not raise
        record = collector.collect_one()
        assert record["window_start_ts"] % 300 == 0
        assert record["binance_btc_open"] == 0.0
        assert record.get("up_price") is None
