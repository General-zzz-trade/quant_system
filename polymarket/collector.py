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
    sample_15m_once,
)

logger = logging.getLogger(__name__)

_WINDOW_5M = 300   # 5 minutes
_WINDOW_15M = 900  # 15 minutes
_SETTLE_OFFSET = 5  # seconds after boundary to let market settle
_INTRA_INTERVAL = 30  # seconds between intra-window samples

# Keep backward compat alias
_WINDOW_SEC = _WINDOW_5M


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


class RSITracker:
    """Track rolling RSI(5) on 5-minute BTC closes for signal annotation."""

    def __init__(self, period: int = 5):
        self._period = period
        self._closes: list[float] = []
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self._rsi: float = 50.0

    def update(self, close: float) -> float:
        """Feed a 5-minute close price. Returns current RSI."""
        self._closes.append(close)
        n = len(self._closes)
        if n < 2:
            return 50.0

        change = self._closes[-1] - self._closes[-2]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        if n <= self._period + 1:
            # Initial SMA phase
            if n == self._period + 1:
                gains = [max(self._closes[i] - self._closes[i - 1], 0)
                         for i in range(1, n)]
                losses = [max(self._closes[i - 1] - self._closes[i], 0)
                          for i in range(1, n)]
                self._avg_gain = sum(gains[:self._period]) / self._period
                self._avg_loss = sum(losses[:self._period]) / self._period
            else:
                return 50.0
        else:
            self._avg_gain = (self._avg_gain * (self._period - 1) + gain) / self._period
            self._avg_loss = (self._avg_loss * (self._period - 1) + loss) / self._period

        if self._avg_loss < 1e-10:
            self._rsi = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            self._rsi = 100.0 - 100.0 / (1.0 + rs)
        return self._rsi

    @property
    def value(self) -> float:
        return self._rsi

    @property
    def signal(self) -> str:
        """Return 'up', 'down', or 'neutral' based on RSI thresholds."""
        if self._rsi < 25:
            return "up"
        elif self._rsi > 75:
            return "down"
        return "neutral"

    @property
    def bar_count(self) -> int:
        return len(self._closes)


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
        pm = self._get_polymarket_prices(window_ts)
        up_mid = pm.get("up_price")
        up_bid = pm.get("up_best_bid", 0)
        up_ask = pm.get("up_best_ask", 1)
        up_spread = pm.get("up_spread")
        up_bid_depth = pm.get("up_bid_depth", 0)
        up_ask_depth = pm.get("up_ask_depth", 0)
        down_bid = pm.get("down_best_bid", 0)
        down_ask = pm.get("down_best_ask", 1)

        remaining_min = max(0, (_WINDOW_SEC - elapsed)) / 60.0
        move_bps = ((price - strike) / strike * 10000) if strike > 0 else 0.0
        fair_up = binary_call_fair_value(price, strike, remaining_min, sigma)
        pricing_delay = (up_mid - fair_up) if up_mid is not None else None

        sample = {
            "window_start_ts": window_ts,
            "sample_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_sec": elapsed,
            "binance_price": price,
            "strike_price": strike,
            "move_bps": move_bps,
            "fair_value_up": fair_up,
            "fair_value_down": 1.0 - fair_up,
            "clob_up_best_bid": up_bid if pm else None,
            "clob_up_best_ask": up_ask if pm else None,
            "clob_up_spread": up_spread,
            "clob_up_bid_depth": up_bid_depth if pm else None,
            "clob_up_ask_depth": up_ask_depth if pm else None,
            "clob_down_best_bid": down_bid if pm else None,
            "clob_down_best_ask": down_ask if pm else None,
            "clob_up_mid": up_mid,
            "pricing_delay": pricing_delay,
            "rsi_5": rsi_at_open,
            "rsi_signal": rsi_signal,
            "btc_ret_3bar": btc_ret_3bar,
        }
        return sample, up_mid, up_bid, up_ask, up_spread, fair_up, move_bps, pricing_delay

    def _init_5m_window(self):
        """Initialize state at the start of a 5m window. Returns tuple."""
        window_ts = self._current_window_ts()
        strike = self._get_binance_price()
        if strike > 0:
            self._vol_tracker.update(strike)
            self._rsi_tracker.update(strike)

        sigma = self._vol_tracker.sigma_annual
        rsi_at_open = self._rsi_tracker.value
        rsi_signal = self._rsi_tracker.signal
        closes = self._rsi_tracker._closes
        btc_ret_3bar = None
        if len(closes) >= 4:
            btc_ret_3bar = (closes[-1] - closes[-4]) / closes[-4]

        self._token_cache = {"window_ts": None}
        tokens = self._resolve_token_ids(window_ts)
        self._token_cache = {"window_ts": window_ts, **tokens}
        has_tokens = bool(tokens.get("up_token"))

        logger.info(
            "Intra-window start: ts=%d strike=$%.0f sigma=%.2f tokens=%s RSI=%.0f(%s) ret3=%s",
            window_ts, strike, sigma, "yes" if has_tokens else "NO",
            rsi_at_open, rsi_signal,
            f"{btc_ret_3bar*100:+.2f}%" if btc_ret_3bar is not None else "N/A",
        )
        return window_ts, strike, sigma, rsi_at_open, rsi_signal, btc_ret_3bar

    def _sample_5m_tick(self, window_ts, strike, sigma, rsi_at_open, rsi_signal, btc_ret_3bar):
        """Take one 5m intra-window sample."""
        now = time.time()
        elapsed = int(now - window_ts)
        if elapsed >= _WINDOW_SEC:
            return False

        price = self._get_binance_price()
        if price <= 0:
            return True  # continue, just skip

        sample, up_mid, up_bid, up_ask, up_spread, fair_up, move_bps, pricing_delay = (
            self._build_intra_sample(
                window_ts, elapsed, price, strike, sigma,
                rsi_at_open, rsi_signal, btc_ret_3bar,
            )
        )
        self._store_intra_sample_v2(sample)

        delay_str = f"delay={pricing_delay:+.3f}" if pricing_delay is not None else "no_clob"
        rsi_str = f"RSI={rsi_at_open:.0f}({rsi_signal})"
        logger.info(
            "  t+%ds: BTC=$%.0f move=%+.1fbps fair=%.3f "
            "clob=%.3f bid=%.2f ask=%.2f spd=%.2f %s %s",
            elapsed, price, move_bps, fair_up,
            up_mid or 0, up_bid, up_ask, up_spread or 0,
            delay_str, rsi_str,
        )
        self._vol_tracker.update(price)
        return True

    def collect_intra_window(self) -> None:
        """Run 30-second sampling within a single 5-minute window."""
        window_ts, strike, sigma, rsi_at_open, rsi_signal, btc_ret_3bar = (
            self._init_5m_window()
        )

        while self._running:
            now = time.time()
            elapsed = int(now - window_ts)
            if elapsed >= _WINDOW_SEC:
                break

            price = self._get_binance_price()
            if price > 0:
                sample, up_mid, up_bid, up_ask, up_spread, fair_up, move_bps, pricing_delay = (
                    self._build_intra_sample(
                        window_ts, elapsed, price, strike, sigma,
                        rsi_at_open, rsi_signal, btc_ret_3bar,
                    )
                )
                self._store_intra_sample_v2(sample)

                delay_str = f"delay={pricing_delay:+.3f}" if pricing_delay is not None else "no_clob"
                rsi_str = f"RSI={rsi_at_open:.0f}({rsi_signal})"
                logger.info(
                    "  t+%ds: BTC=$%.0f move=%+.1fbps fair=%.3f "
                    "clob=%.3f bid=%.2f ask=%.2f spd=%.2f %s %s",
                    elapsed, price, move_bps, fair_up,
                    up_mid or 0, up_bid, up_ask, up_spread or 0,
                    delay_str, rsi_str,
                )
                self._vol_tracker.update(price)

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
        return _get_stats_db(self._db_path, self._vol_tracker.sigma_annual)

    # ------------------------------------------------------------------
    # Continuous run
    # ------------------------------------------------------------------

    def start(self, mode: str = "basic"):
        """Run collector continuously, aligned to 5-minute boundaries."""
        self._running = True
        logger.info("Polymarket collector starting (db=%s, mode=%s)", self._db_path, mode)

        # 15m window state (persists across 5m cycles)
        cur_15m_ts: int = 0
        strike_15m: float = 0.0
        rsi_15m_val: float = 50.0
        rsi_15m_sig: str = "neutral"
        ret_3bar_15m: float | None = None

        while self._running:
            try:
                if mode == "intra":
                    next_boundary = self._next_window_ts()
                    wait_sec = max(0, next_boundary - time.time() + _SETTLE_OFFSET)
                    end_wait = time.time() + wait_sec
                    while self._running and time.time() < end_wait:
                        time.sleep(1)
                    if not self._running:
                        break

                    # Check if a new 15m window just started
                    new_15m_ts = self._current_window_ts_15m()
                    if new_15m_ts != cur_15m_ts:
                        cur_15m_ts = new_15m_ts
                        strike_15m = self._get_binance_price()
                        if strike_15m > 0:
                            self._rsi_tracker_15m.update(strike_15m)
                        rsi_15m_val = self._rsi_tracker_15m.value
                        rsi_15m_sig = self._rsi_tracker_15m.signal
                        closes_15m = self._rsi_tracker_15m._closes
                        ret_3bar_15m = None
                        if len(closes_15m) >= 4:
                            ret_3bar_15m = (closes_15m[-1] - closes_15m[-4]) / closes_15m[-4]
                        self._token_cache_15m = {"window_ts": None}
                        logger.info(
                            "15m window start: ts=%d strike=$%.0f RSI=%.0f(%s) ret3=%s",
                            cur_15m_ts, strike_15m, rsi_15m_val, rsi_15m_sig,
                            f"{ret_3bar_15m*100:+.2f}%" if ret_3bar_15m is not None else "N/A",
                        )

                    self._collect_intra_with_15m(
                        cur_15m_ts, strike_15m, rsi_15m_val, rsi_15m_sig, ret_3bar_15m,
                    )
                else:
                    self.collect_one()
            except Exception:
                logger.exception("Collection cycle failed")

            if mode != "intra":
                next_boundary = self._next_window_ts()
                sleep_sec = max(1, next_boundary - time.time() + _SETTLE_OFFSET)
                logger.debug("Sleeping %.0fs until next window", sleep_sec)
                end_time = time.time() + sleep_sec
                while self._running and time.time() < end_time:
                    time.sleep(1)

    def _collect_intra_with_15m(self, ts_15m, strike_15m, rsi_15m, rsi_sig_15m, ret3_15m):
        """Run 5m intra-window sampling, also sampling 15m market at each tick."""
        window_ts, strike, sigma, rsi_at_open, rsi_signal, btc_ret_3bar = (
            self._init_5m_window()
        )

        while self._running:
            now = time.time()
            elapsed = int(now - window_ts)
            if elapsed >= _WINDOW_SEC:
                break

            price = self._get_binance_price()
            if price <= 0:
                time.sleep(_INTRA_INTERVAL)
                continue

            # --- 5m sample ---
            sample, up_mid, up_bid, up_ask, up_spread, fair_up, move_bps, pricing_delay = (
                self._build_intra_sample(
                    window_ts, elapsed, price, strike, sigma,
                    rsi_at_open, rsi_signal, btc_ret_3bar,
                )
            )
            self._store_intra_sample_v2(sample)

            delay_str = f"delay={pricing_delay:+.3f}" if pricing_delay is not None else "no_clob"
            rsi_str = f"RSI={rsi_at_open:.0f}({rsi_signal})"
            logger.info(
                "  t+%ds: BTC=$%.0f move=%+.1fbps fair=%.3f "
                "clob=%.3f bid=%.2f ask=%.2f spd=%.2f %s %s",
                elapsed, price, move_bps, fair_up,
                up_mid or 0, up_bid, up_ask, up_spread or 0,
                delay_str, rsi_str,
            )

            # --- 15m sample (piggyback on same tick) ---
            try:
                sample_15m_once(
                    self._db_path, ts_15m, strike_15m, sigma,
                    rsi_15m, rsi_sig_15m, ret3_15m,
                    self._get_binance_price, self._get_clob_orderbook,
                    self._token_cache_15m, binary_call_fair_value,
                )
            except Exception:
                logger.debug("15m sample failed", exc_info=True)

            self._vol_tracker.update(price)

            next_sample = window_ts + ((elapsed // _INTRA_INTERVAL) + 1) * _INTRA_INTERVAL
            sleep_sec = max(0, next_sample - time.time())
            end_time = time.time() + sleep_sec
            while self._running and time.time() < end_time:
                time.sleep(min(1, max(0, end_time - time.time())))

    def stop(self):
        """Signal the collector to stop after the current cycle."""
        self._running = False
        logger.info("Polymarket collector stopping")


if __name__ == "__main__":
    from polymarket.collector_main import main

    main()
