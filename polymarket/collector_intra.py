"""Intra-window and continuous collection logic for PolymarketCollector.

Extracted from collector.py: collect_intra_window, _collect_intra_with_15m, start loop.
"""
from __future__ import annotations

import logging
import time

from polymarket.collector_sampling import sample_15m_once
from polymarket.collector_signals import binary_call_fair_value

logger = logging.getLogger(__name__)


def collect_intra_window(collector) -> None:
    """Run 30-second sampling within a single 5-minute window."""
    from polymarket.collector import _WINDOW_SEC, _INTRA_INTERVAL
    window_ts, strike, sigma, rsi_at_open, rsi_signal, btc_ret_3bar = (
        collector._init_5m_window()
    )

    while collector._running:
        now = time.time()
        elapsed = int(now - window_ts)
        if elapsed >= _WINDOW_SEC:
            break

        price = collector._get_binance_price()
        if price > 0:
            sample, up_mid, up_bid, up_ask, up_spread, fair_up, move_bps, pricing_delay = (
                collector._build_intra_sample(
                    window_ts, elapsed, price, strike, sigma,
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

        next_sample = window_ts + ((elapsed // _INTRA_INTERVAL) + 1) * _INTRA_INTERVAL
        sleep_sec = max(0, next_sample - time.time())
        end_time = time.time() + sleep_sec
        while collector._running and time.time() < end_time:
            time.sleep(min(1, max(0, end_time - time.time())))


def collect_intra_with_15m(collector, ts_15m, strike_15m, rsi_15m, rsi_sig_15m, ret3_15m):
    """Run 5m intra-window sampling, also sampling 15m market at each tick."""
    from polymarket.collector import _WINDOW_SEC, _INTRA_INTERVAL
    window_ts, strike, sigma, rsi_at_open, rsi_signal, btc_ret_3bar = (
        collector._init_5m_window()
    )

    while collector._running:
        now = time.time()
        elapsed = int(now - window_ts)
        if elapsed >= _WINDOW_SEC:
            break

        price = collector._get_binance_price()
        if price <= 0:
            time.sleep(_INTRA_INTERVAL)
            continue

        # --- 5m sample ---
        sample, up_mid, up_bid, up_ask, up_spread, fair_up, move_bps, pricing_delay = (
            collector._build_intra_sample(
                window_ts, elapsed, price, strike, sigma,
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

        # --- 15m sample (piggyback on same tick) ---
        try:
            sample_15m_once(
                collector._db_path, ts_15m, strike_15m, sigma,
                rsi_15m, rsi_sig_15m, ret3_15m,
                collector._get_binance_price, collector._get_clob_orderbook,
                collector._token_cache_15m, binary_call_fair_value,
            )
        except Exception:
            logger.debug("15m sample failed", exc_info=True)

        collector._vol_tracker.update(price)

        next_sample = window_ts + ((elapsed // _INTRA_INTERVAL) + 1) * _INTRA_INTERVAL
        sleep_sec = max(0, next_sample - time.time())
        end_time = time.time() + sleep_sec
        while collector._running and time.time() < end_time:
            time.sleep(min(1, max(0, end_time - time.time())))


def run_continuous(collector, mode: str = "basic"):
    """Run collector continuously, aligned to 5-minute boundaries."""
    from polymarket.collector import _SETTLE_OFFSET
    collector._running = True
    logger.info("Polymarket collector starting (db=%s, mode=%s)", collector._db_path, mode)

    # 15m window state (persists across 5m cycles)
    cur_15m_ts: int = 0
    strike_15m: float = 0.0
    rsi_15m_val: float = 50.0
    rsi_15m_sig: str = "neutral"
    ret_3bar_15m: float | None = None

    while collector._running:
        try:
            if mode == "intra":
                next_boundary = collector._next_window_ts()
                wait_sec = max(0, next_boundary - time.time() + _SETTLE_OFFSET)
                end_wait = time.time() + wait_sec
                while collector._running and time.time() < end_wait:
                    time.sleep(1)
                if not collector._running:
                    break

                # Check if a new 15m window just started
                new_15m_ts = collector._current_window_ts_15m()
                if new_15m_ts != cur_15m_ts:
                    cur_15m_ts = new_15m_ts
                    strike_15m = collector._get_binance_price()
                    if strike_15m > 0:
                        collector._rsi_tracker_15m.update(strike_15m)
                    rsi_15m_val = collector._rsi_tracker_15m.value
                    rsi_15m_sig = collector._rsi_tracker_15m.signal
                    closes_15m = collector._rsi_tracker_15m._closes
                    ret_3bar_15m = None
                    if len(closes_15m) >= 4:
                        ret_3bar_15m = (closes_15m[-1] - closes_15m[-4]) / closes_15m[-4]
                    collector._token_cache_15m = {"window_ts": None}
                    logger.info(
                        "15m window start: ts=%d strike=$%.0f RSI=%.0f(%s) ret3=%s",
                        cur_15m_ts, strike_15m, rsi_15m_val, rsi_15m_sig,
                        f"{ret_3bar_15m*100:+.2f}%" if ret_3bar_15m is not None else "N/A",
                    )

                collect_intra_with_15m(
                    collector,
                    cur_15m_ts, strike_15m, rsi_15m_val, rsi_15m_sig, ret_3bar_15m,
                )
            else:
                collector.collect_one()
        except Exception:
            logger.exception("Collection cycle failed")

        if mode != "intra":
            next_boundary = collector._next_window_ts()
            sleep_sec = max(1, next_boundary - time.time() + _SETTLE_OFFSET)
            logger.debug("Sleeping %.0fs until next window", sleep_sec)
            end_time = time.time() + sleep_sec
            while collector._running and time.time() < end_time:
                time.sleep(1)
