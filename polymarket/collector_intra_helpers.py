"""Intra-window sampling helpers for PolymarketCollector.

Extracted from collector.py to keep it under 500 lines.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from polymarket.collector_signals import binary_call_fair_value

logger = logging.getLogger(__name__)

_WINDOW_SEC = 300  # 5 minutes
_INTRA_INTERVAL = 30  # seconds between intra-window samples


def build_intra_sample(collector: Any, window_ts: int, elapsed: int,
                       price: float, strike: float, sigma: float,
                       rsi_at_open: float, rsi_signal: str, btc_ret_3bar: Any):
    """Build a single 5m intra-window sample dict."""
    pm = collector._get_polymarket_prices(window_ts)
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


def init_5m_window(collector: Any):
    """Initialize state at the start of a 5m window. Returns tuple."""
    window_ts = collector._current_window_ts()
    strike = collector._get_binance_price()
    if strike > 0:
        collector._vol_tracker.update(strike)
        collector._rsi_tracker.update(strike)

    sigma = collector._vol_tracker.sigma_annual
    rsi_at_open = collector._rsi_tracker.value
    rsi_signal = collector._rsi_tracker.signal
    closes = collector._rsi_tracker._closes
    btc_ret_3bar = None
    if len(closes) >= 4:
        btc_ret_3bar = (closes[-1] - closes[-4]) / closes[-4]

    collector._token_cache = {"window_ts": None}
    tokens = collector._resolve_token_ids(window_ts)
    collector._token_cache = {"window_ts": window_ts, **tokens}
    has_tokens = bool(tokens.get("up_token"))

    logger.info(
        "Intra-window start: ts=%d strike=$%.0f sigma=%.2f tokens=%s RSI=%.0f(%s) ret3=%s",
        window_ts, strike, sigma, "yes" if has_tokens else "NO",
        rsi_at_open, rsi_signal,
        f"{btc_ret_3bar*100:+.2f}%" if btc_ret_3bar is not None else "N/A",
    )
    return window_ts, strike, sigma, rsi_at_open, rsi_signal, btc_ret_3bar


def sample_5m_tick(collector: Any, window_ts: int, strike: float, sigma: float,
                   rsi_at_open: float, rsi_signal: str, btc_ret_3bar: Any):
    """Take one 5m intra-window sample."""
    now = time.time()
    elapsed = int(now - window_ts)
    if elapsed >= _WINDOW_SEC:
        return False

    price = collector._get_binance_price()
    if price <= 0:
        return True  # continue, just skip

    sample, up_mid, up_bid, up_ask, up_spread, fair_up, move_bps, pricing_delay = (
        build_intra_sample(
            collector, window_ts, elapsed, price, strike, sigma,
            rsi_at_open, rsi_signal, btc_ret_3bar,
        )
    )
    collector._store_intra_sample_v2(sample)

    delay_str = f"delay={pricing_delay:+.3f}" if pricing_delay is not None else "no_clob"
    rsi_str = f"RSI={rsi_at_open:.0f}({rsi_signal})"
    logger.info(
        "  t+%ds: BTC=$%.0f move=%+.1fbps fair=%.3f "
        "clob=%.3f bid=%.2f ask=%.2f spd=%.2f %s %s",
        elapsed, price, move_bps, fair_up,
        up_mid or 0, up_bid, up_ask, up_spread or 0,
        delay_str, rsi_str,
    )
    collector._vol_tracker.update(price)
    return True
