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
import math
import sqlite3
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_WINDOW_SEC = 300  # 5 minutes
_SETTLE_OFFSET = 5  # seconds after boundary to let market settle
_INTRA_INTERVAL = 30  # seconds between intra-window samples


def binary_call_fair_value(S: float, K: float, T_minutes: float, sigma_annual: float) -> float:
    """Fair value of a binary call option (digital call).

    Uses the Black-Scholes formula to compute the risk-neutral probability
    that the underlying finishes at or above the strike.

    Args:
        S: current price
        K: strike (window open price)
        T_minutes: time remaining in minutes
        sigma_annual: annualized volatility

    Returns:
        Probability that S_T >= K (0 to 1).
    """
    if T_minutes <= 0:
        return 1.0 if S >= K else 0.0
    if K <= 0 or S <= 0:
        return 0.5
    T = T_minutes / (365 * 24 * 60)
    d2 = (math.log(S / K) + (-0.5 * sigma_annual**2) * T) / (sigma_annual * math.sqrt(T))
    # norm.cdf(x) = 0.5 * (1 + erf(x / sqrt(2)))
    return 0.5 * (1.0 + math.erf(d2 / math.sqrt(2)))


class VolatilityTracker:
    """Track rolling 1-hour realized volatility from Binance 1-minute returns."""

    def __init__(self, window: int = 60):
        self._returns: list[float] = []
        self._window = window
        self._prev_price: float | None = None

    def update(self, price: float) -> None:
        """Update with a new price observation."""
        if self._prev_price is not None and self._prev_price > 0:
            ret = math.log(price / self._prev_price)
            self._returns.append(ret)
            if len(self._returns) > self._window:
                self._returns.pop(0)
        self._prev_price = price

    @property
    def sigma_annual(self) -> float:
        """Annualized volatility estimate.

        Falls back to 50% if fewer than 10 observations.
        """
        if len(self._returns) < 10:
            return 0.50  # default 50% annual vol
        std_1m = statistics.stdev(self._returns)
        return std_1m * math.sqrt(365 * 24 * 60)


class PolymarketCollector:
    """Collects Polymarket 5m BTC Up/Down market data into SQLite."""

    def __init__(self, db_path: str = "data/polymarket/collector.db"):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._running = False
        self._vol_tracker = VolatilityTracker(window=60)

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
        # V2 schema: CLOB orderbook data
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

    def _get_binance_5m_kline(self, window_ts: int) -> dict:
        """Get the Binance 5m kline (open, close) for a specific window.

        Uses the Binance klines REST API (public, no auth) to get the exact
        open and close prices for the 5-minute window starting at window_ts.
        This is more accurate than spot price snapshots.
        """
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

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect_one(self) -> dict:
        """Collect one data point for the current 5m window.

        Also back-fills the *previous* window with accurate Binance kline
        open/close data and derives the Up/Down result.
        Returns the recorded data dict.
        """
        now = datetime.now(timezone.utc)
        window_ts = self._current_window_ts()

        # 1. Fetch current market metadata from Polymarket
        market = self._get_current_5m_market(window_ts)

        # 2. Get current Binance spot price
        btc_now = self._get_binance_price()

        # 3. Back-fill PREVIOUS window with accurate Binance 5m kline
        prev_ts = window_ts - _WINDOW_SEC
        prev_kline = self._get_binance_5m_kline(prev_ts)
        prev_market = self._get_current_5m_market(prev_ts)

        if prev_kline:
            # Derive result from Binance kline open vs close
            # This matches Polymarket's rule: "Up if close >= open"
            kline_open = prev_kline["open"]
            kline_close = prev_kline["close"]
            if kline_close >= kline_open:
                binance_result = "Up"
            else:
                binance_result = "Down"

            self._update_result(
                prev_ts,
                polymarket_result=prev_market.get("winner"),
                binance_result=binance_result,
                btc_close=kline_close,
                final_volume=prev_market.get("volume"),
            )

            # Also update the open price if we stored it without kline data
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
            # Fallback: use spot price approximation
            self._backfill_previous(prev_ts, prev_market.get("winner"), prev_market, btc_now)

        # 4. Store current window snapshot
        record = {
            "timestamp_utc": now.strftime("%Y-%m-%dT%H:%M:%S"),
            "window_start_ts": window_ts,
            "slug": market.get("slug", f"btc-updown-5m-{window_ts}"),
            "up_price": market.get("up_price"),
            "down_price": market.get("down_price"),
            "volume": market.get("volume", 0),
            "binance_btc_open": btc_now,  # approximate; will be corrected by kline in next cycle
        }
        self._store(record)

        prev_result = prev_kline and ("Up" if prev_kline["close"] >= prev_kline["open"] else "Down")
        logger.info(
            "Collected: ts=%d btc=$%.0f vol=$%.0f | prev: open=$%.0f close=$%.0f → %s",
            window_ts,
            btc_now,
            record.get("volume") or 0,
            prev_kline.get("open", 0) if prev_kline else 0,
            prev_kline.get("close", 0) if prev_kline else 0,
            prev_result or "no_kline",
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
    # Intra-window sampling
    # ------------------------------------------------------------------

    def _store_intra_sample(self, sample: dict) -> None:
        """Store an intra-window sample to the legacy v1 table."""
        conn = sqlite3.connect(self._db_path)
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

    def _store_intra_sample_v2(self, sample: dict) -> None:
        """Store an intra-window sample with CLOB orderbook data."""
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            """
            INSERT INTO intra_window_samples_v2
            (window_start_ts, sample_time_utc, elapsed_sec, binance_price,
             strike_price, move_bps, fair_value_up, fair_value_down,
             clob_up_best_bid, clob_up_best_ask, clob_up_spread,
             clob_up_bid_depth, clob_up_ask_depth,
             clob_down_best_bid, clob_down_best_ask,
             clob_up_mid, pricing_delay)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        conn.commit()
        conn.close()

    def _resolve_token_ids(self, window_ts: int) -> dict:
        """Resolve CLOB token IDs for a 5m window from Gamma /markets API.

        Called once per window. Returns {"up_token": str, "down_token": str}
        or empty dict on failure.
        """
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
        """Fetch real orderbook from CLOB API for a single token.

        Returns {"best_bid", "best_ask", "mid", "spread", "bid_depth", "ask_depth"}
        with prices from actual live orders (not cached Gamma data).
        """
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
                # Find true best bid (highest) and best ask (lowest)
                best_bid = max((float(b["price"]) for b in bids), default=0.0)
                best_ask = min((float(a["price"]) for a in asks), default=1.0)
                mid = (best_bid + best_ask) / 2 if bids and asks else None
                spread = best_ask - best_bid if bids and asks else None
                # Depth: total notional within 10 cents of mid
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
        """Fetch real-time Polymarket prices from CLOB orderbook.

        V2: Uses CLOB API directly for real orderbook data instead of
        cached Gamma API outcomePrices. Resolves token IDs from Gamma
        once per window, then queries CLOB for live bid/ask.
        """
        # Resolve token IDs (cached per window via caller)
        if not hasattr(self, "_token_cache") or self._token_cache.get("window_ts") != window_ts:
            tokens = self._resolve_token_ids(window_ts)
            self._token_cache = {"window_ts": window_ts, **tokens}

        up_token = self._token_cache.get("up_token")
        down_token = self._token_cache.get("down_token")

        if not up_token:
            return {}

        result: dict = {"source": "clob"}

        # Fetch Up orderbook
        up_book = self._get_clob_orderbook(up_token)
        if up_book:
            result["up_price"] = up_book.get("mid")
            result["up_best_bid"] = up_book.get("best_bid", 0)
            result["up_best_ask"] = up_book.get("best_ask", 1)
            result["up_spread"] = up_book.get("spread")
            result["up_bid_depth"] = up_book.get("bid_depth", 0)
            result["up_ask_depth"] = up_book.get("ask_depth", 0)

        # Fetch Down orderbook
        if down_token:
            down_book = self._get_clob_orderbook(down_token)
            if down_book:
                result["down_price"] = down_book.get("mid")
                result["down_best_bid"] = down_book.get("best_bid", 0)
                result["down_best_ask"] = down_book.get("best_ask", 1)

        return result

    def collect_intra_window(self) -> None:
        """Run 30-second sampling within a single 5-minute window.

        V2: Fetches real CLOB orderbook data (not cached Gamma API).
        Samples Binance price + CLOB best bid/ask every 30 seconds,
        computes Black-Scholes binary option fair values, measures pricing delay
        vs CLOB mid-price, and stores results.
        """
        window_ts = self._current_window_ts()

        # Get strike price (Binance spot at window open)
        strike = self._get_binance_price()
        if strike > 0:
            self._vol_tracker.update(strike)

        sigma = self._vol_tracker.sigma_annual

        # Resolve token IDs once per window
        self._token_cache = {"window_ts": None}  # force refresh
        tokens = self._resolve_token_ids(window_ts)
        self._token_cache = {"window_ts": window_ts, **tokens}
        has_tokens = bool(tokens.get("up_token"))

        logger.info(
            "Intra-window start: ts=%d strike=$%.0f sigma=%.2f tokens=%s",
            window_ts, strike, sigma, "yes" if has_tokens else "NO",
        )

        # Sample every 30 seconds for the remainder of the window
        while self._running:
            now = time.time()
            elapsed = int(now - window_ts)

            # If we've passed the window boundary, break out
            if elapsed >= _WINDOW_SEC:
                break

            # Get current Binance price
            price = self._get_binance_price()
            if price <= 0:
                time.sleep(_INTRA_INTERVAL)
                continue

            # Get CLOB orderbook data
            pm = self._get_polymarket_prices(window_ts)
            up_mid = pm.get("up_price")
            up_bid = pm.get("up_best_bid", 0)
            up_ask = pm.get("up_best_ask", 1)
            up_spread = pm.get("up_spread")
            up_bid_depth = pm.get("up_bid_depth", 0)
            up_ask_depth = pm.get("up_ask_depth", 0)
            down_bid = pm.get("down_best_bid", 0)
            down_ask = pm.get("down_best_ask", 1)

            # Compute fair values
            remaining_min = max(0, (_WINDOW_SEC - elapsed)) / 60.0
            move_bps = ((price - strike) / strike * 10000) if strike > 0 else 0.0
            fair_up = binary_call_fair_value(price, strike, remaining_min, sigma)
            fair_down = 1.0 - fair_up

            # Pricing delay: CLOB mid vs BS fair value
            pricing_delay = None
            if up_mid is not None:
                pricing_delay = up_mid - fair_up

            sample = {
                "window_start_ts": window_ts,
                "sample_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_sec": elapsed,
                "binance_price": price,
                "strike_price": strike,
                "move_bps": move_bps,
                "fair_value_up": fair_up,
                "fair_value_down": fair_down,
                "clob_up_best_bid": up_bid if pm else None,
                "clob_up_best_ask": up_ask if pm else None,
                "clob_up_spread": up_spread,
                "clob_up_bid_depth": up_bid_depth if pm else None,
                "clob_up_ask_depth": up_ask_depth if pm else None,
                "clob_down_best_bid": down_bid if pm else None,
                "clob_down_best_ask": down_ask if pm else None,
                "clob_up_mid": up_mid,
                "pricing_delay": pricing_delay,
            }
            self._store_intra_sample_v2(sample)

            # Log with CLOB orderbook data
            delay_str = f"delay={pricing_delay:+.3f}" if pricing_delay is not None else "no_clob"
            logger.info(
                "  t+%ds: BTC=$%.0f move=%+.1fbps fair=%.3f "
                "clob=%.3f bid=%.2f ask=%.2f spd=%.2f d=$%.0f/$%.0f %s",
                elapsed, price, move_bps, fair_up,
                up_mid or 0, up_bid, up_ask, up_spread or 0,
                up_bid_depth, up_ask_depth, delay_str,
            )

            # Update vol tracker
            self._vol_tracker.update(price)

            # Sleep until next 30-second mark
            next_sample = window_ts + ((elapsed // _INTRA_INTERVAL) + 1) * _INTRA_INTERVAL
            sleep_sec = max(0, next_sample - time.time())
            end_time = time.time() + sleep_sec
            while self._running and time.time() < end_time:
                time.sleep(min(1, max(0, end_time - time.time())))

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
            "current_sigma_annual": self._vol_tracker.sigma_annual,
        }
        return stats

    # ------------------------------------------------------------------
    # Continuous run
    # ------------------------------------------------------------------

    def start(self, mode: str = "basic"):
        """Run collector continuously, aligned to 5-minute boundaries.

        Args:
            mode: 'basic' for one sample per window (original),
                  'intra' for 30-second intra-window sampling with fair values.
        """
        self._running = True
        logger.info("Polymarket collector starting (db=%s, mode=%s)", self._db_path, mode)

        while self._running:
            try:
                if mode == "intra":
                    # Wait until next window boundary, then run intra-window sampling
                    next_boundary = self._next_window_ts()
                    wait_sec = max(0, next_boundary - time.time() + _SETTLE_OFFSET)
                    end_wait = time.time() + wait_sec
                    while self._running and time.time() < end_wait:
                        time.sleep(1)
                    if self._running:
                        self.collect_intra_window()
                else:
                    self.collect_one()
            except Exception:
                logger.exception("Collection cycle failed")

            if mode != "intra":
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
