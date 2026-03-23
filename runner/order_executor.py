"""OrderExecutor — order execution with limit entry and safety checks.

Extracted from AlphaRunner to reduce god-class size.
Handles: limit entry (PostOnly + reprice), market fallback,
margin pre-flight, notional cap, circuit breaker, fill tracking.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from execution.balance_utils import get_total_and_free_balance
from execution.order_utils import reliable_close_position
from runner.strategy_config import MAX_ORDER_NOTIONAL, get_max_order_notional
from core.exceptions import VenueError

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Executes trading orders with limit-first strategy and safety checks.

    Owns: limit fill tracking, tick sizes, limit entry timeout.
    """

    def __init__(
        self,
        adapter: Any,
        symbol: str,
        osm: Any,
        circuit_breaker: Any,
        use_limit_entry: bool = True,
        limit_entry_timeout: float = 30.0,
        limit_entry_poll_interval: float = 2.0,
    ):
        self._adapter = adapter
        self._symbol = symbol
        self._osm = osm
        self._circuit_breaker = circuit_breaker
        self._use_limit_entry = use_limit_entry
        self._limit_entry_timeout = limit_entry_timeout
        self._limit_entry_poll_interval = limit_entry_poll_interval

        # Tick sizes for price improvement
        self._tick_sizes: dict[str, float] = {"BTCUSDT": 0.10, "ETHUSDT": 0.01}
        # Fill rate tracking
        self._limit_fills: int = 0
        self._market_fallbacks: int = 0

        # Try to fetch actual tick size from exchange
        try:
            info = adapter._client.get("/v5/market/instruments-info", {
                "category": "linear", "symbol": symbol,
            })
            items = info.get("result", {}).get("list", [])
            if items:
                tick = float(items[0].get("priceFilter", {}).get("tickSize", 0))
                if tick > 0:
                    self._tick_sizes[symbol] = tick
        except Exception as exc:
            logger.debug("%s tick_size fetch failed, using default: %s", symbol, exc)

    @property
    def limit_fills(self) -> int:
        return self._limit_fills

    @property
    def market_fallbacks(self) -> int:
        return self._market_fallbacks

    def _log_fill_rate(self, symbol: str) -> None:
        """Log limit fill rate every 10 trades."""
        total = self._limit_fills + self._market_fallbacks
        if total > 0 and total % 10 == 0:
            rate = self._limit_fills / total * 100
            logger.info(
                "%s LIMIT FILL RATE: %.1f%% (%d limit / %d market / %d total)",
                symbol, rate, self._limit_fills, self._market_fallbacks, total,
            )

    def execute_limit_entry(self, symbol: str, side: str, qty: float,
                            price: float) -> dict:
        """Try to open via limit order for maker fee (0 bps).

        Places PostOnly limit order at best bid+1 tick (buy) or ask-1 tick (sell).
        Adaptive timeout based on spread width and historical fill rate.
        Falls back to market order if not filled.
        """
        ticker = self._adapter.get_ticker(symbol)
        if not ticker or not ticker.get("bid1Price") or not ticker.get("ask1Price"):
            logger.warning("%s LIMIT ENTRY: ticker unavailable, falling back to market", symbol)
            result = self._adapter.send_market_order(symbol, side, qty)
            result["entry_method"] = "market_fallback"
            self._market_fallbacks += 1
            self._log_fill_rate(symbol)
            return result

        bid = float(ticker["bid1Price"])
        ask = float(ticker["ask1Price"])
        spread_pct = (ask - bid) / bid * 100 if bid > 0 else 0

        # Adaptive timeout based on historical fill rate
        total_attempts = self._limit_fills + self._market_fallbacks
        if total_attempts >= 5:
            fill_rate = self._limit_fills / total_attempts
            base_timeout = 15.0 + 30.0 * fill_rate
        else:
            base_timeout = 25.0

        if spread_pct < 0.02:
            timeout = base_timeout
        elif spread_pct <= 0.05:
            timeout = base_timeout * 0.8
        else:
            timeout = 5.0

        # Price improvement
        tick = self._tick_sizes.get(symbol, 0.01)
        spread_ticks = round((ask - bid) / tick) if tick > 0 else 999
        if side.lower() == "buy":
            limit_price = bid if spread_ticks <= 1 else bid + tick
        else:
            limit_price = ask if spread_ticks <= 1 else ask - tick

        logger.info(
            "%s LIMIT ENTRY: %s %.4f @ $%.2f (bid=$%.2f ask=$%.2f spread=%.4f%% timeout=%.0fs)",
            symbol, side, qty, limit_price, bid, ask, spread_pct, timeout,
        )

        # Submit PostOnly limit order
        limit_result = self._adapter.send_limit_order(
            symbol, side, qty, limit_price, tif="PostOnly",
        )
        if limit_result.get("status") == "error":
            logger.warning(
                "%s LIMIT ENTRY: PostOnly rejected (%s), falling back to market",
                symbol, limit_result.get("retMsg", ""),
            )
            result = self._adapter.send_market_order(symbol, side, qty)
            result["entry_method"] = "market_fallback"
            self._market_fallbacks += 1
            self._log_fill_rate(symbol)
            return result

        order_id = limit_result.get("orderId", "")
        logger.info("%s LIMIT ENTRY: order submitted orderId=%s", symbol, order_id)

        # Poll for fill
        deadline = time.time() + timeout
        filled = False
        while time.time() < deadline:
            time.sleep(self._limit_entry_poll_interval)
            try:
                open_orders = self._adapter.get_open_orders(symbol=symbol)
                still_open = any(o.order_id == order_id for o in open_orders)
                if not still_open:
                    fills = self._adapter.get_recent_fills(symbol=symbol)
                    for f in fills:
                        if getattr(f, "order_id", "") == order_id:
                            filled = True
                            break
                    if not filled:
                        positions = self._adapter.get_positions(symbol=symbol)
                        if any(not p.is_flat for p in positions):
                            filled = True
                    break
            except Exception as exc:
                logger.warning("%s LIMIT ENTRY: poll error: %s", symbol, exc)

        if filled:
            logger.info("%s LIMIT ENTRY: FILLED as maker (0 bps) orderId=%s", symbol, order_id)
            self._limit_fills += 1
            self._log_fill_rate(symbol)
            return {"orderId": order_id, "status": "submitted", "entry_method": "limit"}

        # Not filled — cancel and try reprice
        logger.info("%s LIMIT ENTRY: not filled after %.0fs, cancelling", symbol, timeout)
        try:
            self._adapter.cancel_order(symbol, order_id)
        except Exception as exc:
            logger.warning("%s LIMIT ENTRY: cancel error: %s", symbol, exc)
            # Check if it filled during cancel
            fills = self._adapter.get_recent_fills(symbol=symbol)
            for f in fills:
                if getattr(f, "order_id", "") == order_id:
                    self._limit_fills += 1
                    self._log_fill_rate(symbol)
                    return {"orderId": order_id, "status": "submitted", "entry_method": "limit"}

        # Reprice once at current best bid/ask
        try:
            self._adapter.cancel_order(symbol, order_id)
            ticker2 = self._adapter.get_ticker(symbol)
            if ticker2 and ticker2.get("bid1Price"):
                bid2 = float(ticker2["bid1Price"])
                ask2 = float(ticker2["ask1Price"])
                if side.lower() == "buy":
                    reprice = bid2 if spread_ticks <= 1 else bid2 + tick
                else:
                    reprice = ask2 if spread_ticks <= 1 else ask2 - tick

                logger.info("%s LIMIT REPRICE: %s @ $%.2f (was $%.2f)",
                            symbol, side, reprice, limit_price)
                reprice_result = self._adapter.send_limit_order(
                    symbol, side, qty, reprice, post_only=True,
                )
                if reprice_result.get("orderId"):
                    reprice_id = reprice_result["orderId"]
                    time.sleep(15)
                    fills = self._adapter.get_recent_fills(symbol=symbol)
                    for f in fills:
                        if getattr(f, "order_id", "") == reprice_id:
                            logger.info("%s LIMIT REPRICE: FILLED as maker", symbol)
                            self._limit_fills += 1
                            self._log_fill_rate(symbol)
                            return {"orderId": reprice_id, "status": "submitted",
                                    "entry_method": "limit_reprice"}
                    self._adapter.cancel_order(symbol, reprice_id)
        except Exception as e:
            logger.warning("%s LIMIT REPRICE failed: %s", symbol, e)

        # Fall back to market order
        result = self._adapter.send_market_order(symbol, side, qty)
        result["entry_method"] = "market_fallback"
        self._market_fallbacks += 1
        self._log_fill_rate(symbol)
        logger.info("%s LIMIT ENTRY: market fallback result=%s", symbol, result)
        return result

    def execute_signal_change(
        self,
        prev: int,
        new: int,
        price: float,
        *,
        killed: bool,
        dry_run: bool,
        entry_price: float,
        entry_size: float,
        position_size: float,
        min_size: float,
        z_scale: float,
        pnl_tracker: Any,
        risk_evaluator: Any,
        kill_switch: Any,
        state_store: Any,
        record_fill_fn,
        round_to_step_fn,
        force_flat_fn,
        stop_loss_manager: Any,
    ) -> dict:
        """Execute a signal change: close old position, open new one.

        Returns trade info dict.
        """
        if killed and prev == 0 and new != 0:
            return {"action": "killed", "reason": "drawdown_breaker"}

        trade_info: dict = {}
        _pending_close_size = 0.0
        if prev != 0 and entry_price > 0:
            _pending_close_size = entry_size if entry_size > 0 else position_size

        # Drawdown check
        kill_result = stop_loss_manager.check_drawdown_kill(
            self._adapter, self._symbol, pnl_tracker,
            risk_evaluator, kill_switch, dry_run,
        )
        if kill_result is not None:
            return kill_result

        if dry_run:
            if prev != 0 and entry_price > 0:
                trade = pnl_tracker.record_close(
                    symbol=self._symbol, side=prev,
                    entry_price=entry_price, exit_price=price,
                    size=_pending_close_size, reason="signal_change",
                )
                trade_info["closed_pnl"] = round(trade["pnl_usd"], 4)
                trade_info["closed_pct"] = round(trade["pnl_pct"], 2)
            trade_info["action"] = "dry_run"
            trade_info["from"] = prev
            trade_info["to"] = new
            return trade_info

        # Circuit breaker
        if not self._circuit_breaker.allow_request():
            cb_state = self._circuit_breaker.snapshot()
            logger.warning("%s CIRCUIT BREAKER OPEN: %s", self._symbol, cb_state)
            return {"action": "circuit_open", "state": str(cb_state)}

        # Close existing position
        if prev != 0:
            try:
                positions = self._adapter.get_positions(symbol=self._symbol)
                has_real_pos = any(not p.is_flat for p in positions)
            except Exception:
                has_real_pos = True

            if not has_real_pos:
                logger.warning(
                    "%s PHANTOM CLOSE: signal was %d but exchange has no position",
                    self._symbol, prev,
                )
                return {"action": "phantom_close", "from": prev, "to": new}

            close_id = f"close_{self._symbol}_{int(time.time())}"
            self._osm.register(close_id, self._symbol,
                               "sell" if prev == 1 else "buy",
                               "market", str(position_size))
            close_result = reliable_close_position(self._adapter, self._symbol)
            if close_result["status"] == "failed":
                logger.error("%s CLOSE FAILED after retries", self._symbol)
                self._osm.transition(close_id, "rejected", reason="reliable_close_failed")
                self._circuit_breaker.record_failure()
                return {"action": "close_failed", "result": close_result}
            if not close_result.get("verified", True):
                logger.warning("%s CLOSE: position verification failed, proceeding", self._symbol)
            self._osm.transition(close_id, "filled", filled_qty=str(position_size),
                                 avg_price=str(price))
            self._circuit_breaker.record_success()

            # Record PnL after venue close confirmed
            if entry_price > 0:
                trade = pnl_tracker.record_close(
                    symbol=self._symbol, side=prev,
                    entry_price=entry_price, exit_price=price,
                    size=_pending_close_size, reason="signal_change",
                )
                trade_info["closed_pnl"] = round(trade["pnl_usd"], 4)
                trade_info["closed_pct"] = round(trade["pnl_pct"], 2)
                logger.info(
                    "%s CLOSE %s: pnl=$%.4f (%.2f%%) total=$%.4f wins=%d/%d",
                    self._symbol, "long" if prev == 1 else "short",
                    trade["pnl_usd"], trade["pnl_pct"], pnl_tracker.total_pnl,
                    pnl_tracker.win_count, pnl_tracker.trade_count,
                )
                close_side = "sell" if prev == 1 else "buy"
                record_fill_fn(close_side, _pending_close_size, price,
                               realized_pnl=trade["pnl_usd"])

        if killed:
            return {"action": "killed", "reason": "drawdown_breaker",
                    "from": prev, "to": 0}

        # Open new position
        if new != 0:
            side = "buy" if new == 1 else "sell"
            open_id = f"open_{self._symbol}_{int(time.time())}"
            order_type = "limit" if self._use_limit_entry else "market"
            self._osm.register(open_id, self._symbol, side, order_type,
                               str(position_size))

            # Dedup check
            active = self._osm.active_count()
            if active > 2:
                logger.warning("%s DEDUP: %d active orders, skipping", self._symbol, active)
                self._osm.transition(open_id, "rejected", reason="dedup_active_orders")
                return {"action": "dedup_blocked", "active": active}

            position_size = round_to_step_fn(position_size)

            # Dynamic notional cap
            try:
                _eq_for_cap, _ = get_total_and_free_balance(self._adapter.get_balances())
                _eq_for_cap = _eq_for_cap or 0.0
            except Exception:
                _eq_for_cap = 0.0
            dynamic_cap = get_max_order_notional(_eq_for_cap) if _eq_for_cap > 0 else MAX_ORDER_NOTIONAL
            notional = position_size * price
            if notional > dynamic_cap:
                logger.warning(
                    "%s NOTIONAL CLAMP: $%.0f exceeds limit $%.0f — reducing",
                    self._symbol, notional, dynamic_cap,
                )
                position_size = round_to_step_fn(dynamic_cap / price)
                notional = position_size * price
                if position_size < min_size:
                    self._osm.transition(open_id, "rejected", reason="below_min_after_clamp")
                    return {"action": "blocked", "reason": "below_min_after_clamp"}

            # Margin pre-flight
            try:
                _equity, avail = get_total_and_free_balance(self._adapter.get_balances())
                lev = getattr(self, '_current_exchange_lev', 1) or 1
                margin_needed = notional / lev
                if avail is None:
                    logger.warning("%s MARGIN PRECHECK skipped: free balance unavailable", self._symbol)
                elif margin_needed > avail * 0.95:
                    logger.warning(
                        "%s MARGIN SKIP: need $%.0f but only $%.0f available (lev=%dx)",
                        self._symbol, margin_needed, avail, lev,
                    )
                    self._osm.transition(open_id, "rejected", reason="insufficient_margin")
                    return {"action": "margin_skip", "need": margin_needed, "avail": avail}
            except VenueError as exc:
                logger.error("VENUE_ERROR symbol=%s type=%s context=margin_precheck",
                             self._symbol, type(exc).__name__)
            except Exception as exc:
                logger.warning("%s MARGIN PRECHECK failed: %s", self._symbol, exc)

            # Safety: reject NaN/zero
            if (np.isnan(position_size) or position_size <= 0
                    or np.isnan(price) or price <= 0):
                logger.error(
                    "%s ORDER BLOCKED: invalid size=%.6f or price=%.2f",
                    self._symbol, position_size, price,
                )
                self._osm.transition(open_id, "rejected", reason="invalid_size_or_price")
                return {"action": "blocked", "reason": "nan_or_zero_size"}

            # Execute order
            if self._use_limit_entry:
                result = self.execute_limit_entry(self._symbol, side, position_size, price)
            else:
                result = self._adapter.send_market_order(self._symbol, side, position_size)

            if result.get("status") == "error" or result.get("retCode", 0) != 0:
                ret_msg = str(result.get("retMsg", ""))
                if "ab not enough" in ret_msg.lower() or "insufficient" in ret_msg.lower():
                    logger.error("MARGIN_INSUFFICIENT symbol=%s msg=%s", self._symbol, ret_msg)
                else:
                    logger.error("ORDER_REJECTED symbol=%s reason=%s", self._symbol, ret_msg)
                self._osm.transition(open_id, "rejected", reason=ret_msg)
                self._circuit_breaker.record_failure()
                return {"action": "order_failed", "result": result}

            result.get("orderId", open_id)
            self._osm.transition(open_id, "filled", filled_qty=str(position_size),
                                 avg_price=str(price))
            self._circuit_breaker.record_success()

            # Get actual fill price
            actual_price = price
            if result.get("orderId"):
                try:
                    time.sleep(0.3)
                    fills = self._adapter.get_recent_fills(symbol=self._symbol)
                    if fills:
                        actual_price = float(fills[0].price)
                        bar_slip = (actual_price - price) / price * 100
                        entry_method = result.get("entry_method", "unknown")
                        logger.info(
                            "%s FILL: method=%s fill=$%.2f bar_close=$%.2f bar_slip=%.3f%%",
                            self._symbol, entry_method, actual_price, price, bar_slip,
                        )
                except Exception as e:
                    logger.warning("%s failed to get fill price: %s", self._symbol, e)

            record_fill_fn(side, position_size, actual_price)
            atr = stop_loss_manager.current_atr()
            stop = stop_loss_manager.compute_stop_price(actual_price, actual_price, new)
            logger.info(
                "Opened %s %.4f @ ~$%.1f stop=$%.2f (ATR=%.2f%%): %s",
                side, position_size, actual_price, stop, atr * 100, result,
            )
            trade_info.update({
                "side": side, "qty": position_size, "result": result,
                "stop": round(stop, 2), "atr_pct": round(atr * 100, 2),
                "actual_price": actual_price,
            })
        else:
            trade_info["action"] = "flat"

        return trade_info
