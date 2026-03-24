"""Polymarket 5-minute BTC Up/Down data collector.

Collects real market data every 5 minutes:
- Polymarket market pricing (Up/Down shares)
- Settlement result (which side won)
- Concurrent Binance BTC/USDT price
- Volume and liquidity

Supports two modes:
- basic: one sample per 5-minute window (original behavior)
- intra: 30-second intra-window sampling with Black-Scholes fair values

Stores data in SQLite for later analysis.

Usage:
    python3 -m polymarket.collector [--db data/polymarket/collector.db] [--once]
    python3 -m polymarket.collector --mode intra [--once]

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

from polymarket.collector_db import (
    init_db,
    store_snapshot,
    update_result,
    store_intra_sample_v2,
    get_stats as _get_stats_db,
)
from polymarket.collector_sampling import (
    current_window_ts_15m,
    next_window_ts_15m,
)

from polymarket.collector_signals import (  # noqa: E402
    VolatilityTracker,
    RSITracker,
    binary_call_fair_value,  # re-export for backward compat
)

logger = logging.getLogger(__name__)

_WINDOW_5M = 300   # 5 minutes
_WINDOW_15M = 900  # 15 minutes
_SETTLE_OFFSET = 5  # seconds after boundary to let market settle
_INTRA_INTERVAL = 30  # seconds between intra-window samples

# Keep backward compat alias
_WINDOW_SEC = _WINDOW_5M


class PolymarketCollector:
    """Collects Polymarket 5m BTC Up/Down market data into SQLite."""

    def __init__(self, db_path: str = "data/polymarket/collector.db"):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        init_db(db_path)
        self._running = False
        self._vol_tracker = VolatilityTracker(window=60)
        self._rsi_tracker = RSITracker(period=5)       # RSI on 5m closes
        self._rsi_tracker_15m = RSITracker(period=5)    # RSI on 15m closes
        self._token_cache_15m: dict = {"window_ts": None}

    # ------------------------------------------------------------------
    # Database delegates
    # ------------------------------------------------------------------

    def _store(self, record: dict):
        store_snapshot(self._db_path, record)

    def _update_result(self, window_ts, polymarket_result=None,
                       binance_result=None, btc_close=None, final_volume=None):
        update_result(self._db_path, window_ts, polymarket_result,
                      binance_result, btc_close, final_volume)

    def _store_intra_sample(self, sample: dict) -> None:
        from polymarket.collector_db import store_intra_sample
        store_intra_sample(self._db_path, sample)

    def _store_intra_sample_v2(self, sample: dict) -> None:
        store_intra_sample_v2(self._db_path, sample)

    def _store_sample_15m(self, sample: dict) -> None:
        from polymarket.collector_db import store_sample_15m
        store_sample_15m(self._db_path, sample)

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

    def _get_binance_5m_kline(self, window_ts: int) -> dict:
        """Get the Binance 5m kline for a specific window."""
        start_ms = window_ts * 1000
        end_ms = (window_ts + _WINDOW_SEC) * 1000 - 1
        url = (
            f"https://fapi.binance.com/fapi/v1/klines"
            f"?symbol=BTCUSDT&interval=5m&startTime={start_ms}&endTime={end_ms}&limit=1"
        )
        req = Request(url, headers={"Accept": "application/json"})
        try:
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                if data and isinstance(data, list) and len(data) > 0:
                    k = data[0]
                    return {
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    }
        except Exception as e:
            logger.warning("Failed to get Binance 5m kline for ts=%d: %s", window_ts, e)
        return {}

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

    # 15m time helpers (delegates)
    @staticmethod
    def _current_window_ts_15m() -> int:
        return current_window_ts_15m()

    @staticmethod
    def _next_window_ts_15m() -> int:
        return next_window_ts_15m()

    # ------------------------------------------------------------------
    # CLOB helpers
    # ------------------------------------------------------------------

    def _resolve_token_ids(self, window_ts: int) -> dict:
        """Resolve CLOB token IDs for a 5m window from Gamma /markets API."""
        slug = f"btc-updown-5m-{window_ts}"
        url = f"https://gamma-api.polymarket.com/markets?slug={slug}"
        req = Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "quant-collector/1.0"},
        )
        try:
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                if not data or not isinstance(data, list):
                    return {}
                m = data[0]
                tokens_raw = m.get("clobTokenIds", "[]")
                tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
                outcomes_raw = m.get("outcomes", "[]")
                outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
                result = {}
                for i, token_id in enumerate(tokens):
                    outcome = outcomes[i] if i < len(outcomes) else ("Up" if i == 0 else "Down")
                    if outcome == "Up":
                        result["up_token"] = token_id
                    elif outcome == "Down":
                        result["down_token"] = token_id
                return result
        except Exception as e:
            logger.debug("Failed to resolve tokens for %s: %s", slug, e)
        return {}

    def _get_clob_orderbook(self, token_id: str) -> dict:
        """Fetch real orderbook from CLOB API for a single token."""
        url = f"https://clob.polymarket.com/book?token_id={token_id}"
        req = Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "quant-collector/1.0"},
        )
        try:
            with urlopen(req, timeout=5) as resp:
                book = json.loads(resp.read())
                bids = book.get("bids", [])
                asks = book.get("asks", [])
                best_bid = max((float(b["price"]) for b in bids), default=0.0)
                best_ask = min((float(a["price"]) for a in asks), default=1.0)
                mid = (best_bid + best_ask) / 2 if bids and asks else None
                spread = best_ask - best_bid if bids and asks else None
                mid_ref = mid or 0.5
                bid_depth = sum(
                    float(b["size"]) * float(b["price"])
                    for b in bids
                    if float(b["price"]) >= mid_ref - 0.10
                )
                ask_depth = sum(
                    float(a["size"]) * float(a["price"])
                    for a in asks
                    if float(a["price"]) <= mid_ref + 0.10
                )
                return {
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "mid": mid,
                    "spread": spread,
                    "bid_depth": bid_depth,
                    "ask_depth": ask_depth,
                }
        except Exception as e:
            logger.debug("Failed to get CLOB book for %s: %s", token_id[:16], e)
        return {}

    def _get_polymarket_prices(self, window_ts: int) -> dict:
        """Fetch real-time Polymarket prices from CLOB orderbook."""
        if not hasattr(self, "_token_cache") or self._token_cache.get("window_ts") != window_ts:
            tokens = self._resolve_token_ids(window_ts)
            self._token_cache = {"window_ts": window_ts, **tokens}

        up_token = self._token_cache.get("up_token")
        down_token = self._token_cache.get("down_token")
        if not up_token:
            return {}

        result: dict = {"source": "clob"}
        up_book = self._get_clob_orderbook(up_token)
        if up_book:
            result["up_price"] = up_book.get("mid")
            result["up_best_bid"] = up_book.get("best_bid", 0)
            result["up_best_ask"] = up_book.get("best_ask", 1)
            result["up_spread"] = up_book.get("spread")
            result["up_bid_depth"] = up_book.get("bid_depth", 0)
            result["up_ask_depth"] = up_book.get("ask_depth", 0)
        if down_token:
            down_book = self._get_clob_orderbook(down_token)
            if down_book:
                result["down_price"] = down_book.get("mid")
                result["down_best_bid"] = down_book.get("best_bid", 0)
                result["down_best_ask"] = down_book.get("best_ask", 1)
        return result

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect_one(self) -> dict:
        """Collect one data point for the current 5m window."""
        now = datetime.now(timezone.utc)
        window_ts = self._current_window_ts()

        market = self._get_current_5m_market(window_ts)
        btc_now = self._get_binance_price()

        # Back-fill PREVIOUS window with accurate Binance 5m kline
        prev_ts = window_ts - _WINDOW_SEC
        prev_kline = self._get_binance_5m_kline(prev_ts)
        prev_market = self._get_current_5m_market(prev_ts)

        if prev_kline:
            kline_open = prev_kline["open"]
            kline_close = prev_kline["close"]
            binance_result = "Up" if kline_close >= kline_open else "Down"
            self._update_result(
                prev_ts,
                polymarket_result=prev_market.get("winner"),
                binance_result=binance_result,
                btc_close=kline_close,
                final_volume=prev_market.get("volume"),
            )
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "UPDATE market_snapshots SET binance_btc_open = ? "
                "WHERE window_start_ts = ? AND "
                "(binance_btc_open IS NULL OR binance_btc_open = 0)",
                (kline_open, prev_ts),
            )
            conn.commit()
            conn.close()
        else:
            self._backfill_previous(prev_ts, prev_market.get("winner"), prev_market, btc_now)

        record = {
            "timestamp_utc": now.strftime("%Y-%m-%dT%H:%M:%S"),
            "window_start_ts": window_ts,
            "slug": market.get("slug", f"btc-updown-5m-{window_ts}"),
            "up_price": market.get("up_price"),
            "down_price": market.get("down_price"),
            "volume": market.get("volume", 0),
            "binance_btc_open": btc_now,
        }
        self._store(record)

        prev_result = prev_kline and ("Up" if prev_kline["close"] >= prev_kline["open"] else "Down")
        logger.info(
            "Collected: ts=%d btc=$%.0f vol=$%.0f | prev: open=$%.0f close=$%.0f -> %s",
            window_ts, btc_now, record.get("volume") or 0,
            prev_kline.get("open", 0) if prev_kline else 0,
            prev_kline.get("close", 0) if prev_kline else 0,
            prev_result or "no_kline",
        )
        return record

    def _backfill_previous(self, prev_ts, poly_result, prev_market, btc_close):
        """Update the previous window row with settlement info."""
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
    # Intra-window sampling (5m)
    # ------------------------------------------------------------------

    def _build_intra_sample(self, window_ts, elapsed, price, strike, sigma,
                            rsi_at_open, rsi_signal, btc_ret_3bar):
        """Build a single 5m intra-window sample dict."""
        from polymarket.collector_intra_helpers import build_intra_sample
        return build_intra_sample(self, window_ts, elapsed, price, strike, sigma,
                                  rsi_at_open, rsi_signal, btc_ret_3bar)

    def _init_5m_window(self):
        """Initialize state at the start of a 5m window. Returns tuple."""
        from polymarket.collector_intra_helpers import init_5m_window
        return init_5m_window(self)

    def _sample_5m_tick(self, window_ts, strike, sigma, rsi_at_open, rsi_signal, btc_ret_3bar):
        """Take one 5m intra-window sample."""
        from polymarket.collector_intra_helpers import sample_5m_tick
        return sample_5m_tick(self, window_ts, strike, sigma, rsi_at_open, rsi_signal, btc_ret_3bar)

    def collect_intra_window(self) -> None:
        """Run 30-second sampling within a single 5-minute window."""
        from polymarket.collector_intra import collect_intra_window
        collect_intra_window(self)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return collection statistics."""
        return _get_stats_db(self._db_path, self._vol_tracker.sigma_annual)

    # ------------------------------------------------------------------
    # Continuous run
    # ------------------------------------------------------------------

    def start(self, mode: str = "basic"):
        """Run collector continuously, aligned to 5-minute boundaries."""
        from polymarket.collector_intra import run_continuous
        run_continuous(self, mode)

    def _collect_intra_with_15m(self, ts_15m, strike_15m, rsi_15m, rsi_sig_15m, ret3_15m):
        """Run 5m intra-window sampling, also sampling 15m market at each tick."""
        from polymarket.collector_intra import collect_intra_with_15m
        collect_intra_with_15m(self, ts_15m, strike_15m, rsi_15m, rsi_sig_15m, ret3_15m)

    def stop(self):
        """Signal the collector to stop after the current cycle."""
        self._running = False
        logger.info("Polymarket collector stopping")


if __name__ == "__main__":
    from polymarket.collector_main import main

    main()
