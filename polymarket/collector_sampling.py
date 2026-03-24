"""15-minute window sampling helpers for PolymarketCollector.

Extracted from collector.py to keep file sizes manageable.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen

from polymarket.collector_db import store_sample_15m

logger = logging.getLogger(__name__)

_WINDOW_15M = 900  # 15 minutes


def current_window_ts_15m() -> int:
    """Start timestamp of the current 15-minute window."""
    now = int(time.time())
    return (now // _WINDOW_15M) * _WINDOW_15M


def next_window_ts_15m() -> int:
    """Timestamp of the next 15-minute boundary."""
    now = int(time.time())
    return ((now // _WINDOW_15M) + 1) * _WINDOW_15M


def resolve_token_ids_15m(window_ts: int) -> dict:
    """Resolve CLOB token IDs for a 15m window."""
    slug = f"btc-updown-15m-{window_ts}"
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
        logger.debug("Failed to resolve 15m tokens for ts=%d: %s", window_ts, e)
    return {}


def get_polymarket_prices_15m(
    window_ts: int,
    token_cache_15m: dict,
    get_clob_orderbook_fn,
) -> dict:
    """Fetch CLOB orderbook for a 15m window.

    Args:
        window_ts: 15m window start timestamp.
        token_cache_15m: Mutable cache dict; updated in-place.
        get_clob_orderbook_fn: Callable(token_id) -> dict for CLOB orderbook fetch.
    """
    if token_cache_15m.get("window_ts") != window_ts:
        tokens = resolve_token_ids_15m(window_ts)
        token_cache_15m.clear()
        token_cache_15m.update({"window_ts": window_ts, **tokens})

    up_token = token_cache_15m.get("up_token")
    down_token = token_cache_15m.get("down_token")
    if not up_token:
        return {}

    result: dict = {"source": "clob"}
    up_book = get_clob_orderbook_fn(up_token)
    if up_book:
        result["up_price"] = up_book.get("mid")
        result["up_best_bid"] = up_book.get("best_bid", 0)
        result["up_best_ask"] = up_book.get("best_ask", 1)
        result["up_spread"] = up_book.get("spread")
        result["up_bid_depth"] = up_book.get("bid_depth", 0)
        result["up_ask_depth"] = up_book.get("ask_depth", 0)
    if down_token:
        down_book = get_clob_orderbook_fn(down_token)
        if down_book:
            result["down_best_bid"] = down_book.get("best_bid", 0)
            result["down_best_ask"] = down_book.get("best_ask", 1)
    return result


def sample_15m_once(
    db_path: str,
    window_ts: int,
    strike: float,
    sigma: float,
    rsi_val: float,
    rsi_sig: str,
    ret_3bar: float | None,
    get_binance_price_fn,
    get_clob_orderbook_fn,
    token_cache_15m: dict,
    binary_call_fair_value_fn,
) -> None:
    """Take one 15m CLOB sample and store it.

    Args:
        db_path: SQLite database path.
        window_ts: 15m window start timestamp.
        strike: Strike price at window open.
        sigma: Annualized volatility.
        rsi_val: RSI value at window open.
        rsi_sig: RSI signal string.
        ret_3bar: 3-bar return or None.
        get_binance_price_fn: Callable() -> float.
        get_clob_orderbook_fn: Callable(token_id) -> dict.
        token_cache_15m: Mutable cache dict.
        binary_call_fair_value_fn: Callable(S, K, T_min, sigma) -> float.
    """
    now = time.time()
    elapsed = int(now - window_ts)
    if elapsed >= _WINDOW_15M:
        return

    price = get_binance_price_fn()
    if price <= 0:
        return

    pm = get_polymarket_prices_15m(window_ts, token_cache_15m, get_clob_orderbook_fn)
    up_mid = pm.get("up_price")
    up_bid = pm.get("up_best_bid", 0)
    up_ask = pm.get("up_best_ask", 1)
    up_spread = pm.get("up_spread")
    up_bid_depth = pm.get("up_bid_depth", 0)
    up_ask_depth = pm.get("up_ask_depth", 0)
    down_bid = pm.get("down_best_bid", 0)
    down_ask = pm.get("down_best_ask", 1)

    remaining_min = max(0, (_WINDOW_15M - elapsed)) / 60.0
    move_bps = ((price - strike) / strike * 10000) if strike > 0 else 0.0
    fair_up = binary_call_fair_value_fn(price, strike, remaining_min, sigma)
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
        "rsi_5": rsi_val,
        "rsi_signal": rsi_sig,
        "btc_ret_3bar": ret_3bar,
    }
    store_sample_15m(db_path, sample)

    delay_str = f"delay={pricing_delay:+.3f}" if pricing_delay is not None else "no_clob"
    logger.info(
        "  15m t+%ds: BTC=$%.0f move=%+.1fbps fair=%.3f "
        "clob=%.3f bid=%.2f ask=%.2f %s RSI=%.0f(%s)",
        elapsed, price, move_bps, fair_up,
        up_mid or 0, up_bid, up_ask, delay_str,
        rsi_val, rsi_sig,
    )
