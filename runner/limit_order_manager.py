"""LimitOrderManager — pre-places limit orders when signals are forming.

When intra-bar z-score approaches the deadzone threshold (|z| > 0.85 * dz),
pre-places a limit order at a favorable price offset (default 10bps) rather
than waiting for bar close and paying market-order slippage (~192bps avg).

Safety:
- One pending limit order per symbol at a time
- Stale orders auto-cancelled on bar close or after TTL (300s)
- Limit order qty = 50% of normal market order qty (conservative)
- Any exception is swallowed to never disrupt the main trading loop
- Uses AdaptivePositionSizer quantities scaled by qty_scale
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Default TTL for limit orders (seconds) — auto-cancel if not filled
_DEFAULT_TTL_S = 300.0

# Default qty scale — limit orders use fraction of normal market order qty
_DEFAULT_QTY_SCALE = 0.5


class LimitOrderManager:
    """Pre-places limit orders at favorable prices when signals are forming."""

    def __init__(
        self,
        adapter: Any,
        offset_bps: float = 10.0,
        post_only: bool = True,
        ttl_seconds: float = _DEFAULT_TTL_S,
        qty_scale: float = _DEFAULT_QTY_SCALE,
    ) -> None:
        self._adapter = adapter
        self._offset_bps = offset_bps
        self._post_only = post_only
        self._ttl_seconds = ttl_seconds
        self._qty_scale = max(0.01, min(1.0, qty_scale))  # clamp [1%, 100%]
        # symbol -> {orderId, side, qty, price, placed_at}
        self._pending: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_place(
        self,
        symbol: str,
        side: str,
        qty: float,
        current_price: float,
        z_score: float,
        deadzone: float,
    ) -> bool:
        """Place limit order if z approaching deadzone and no pending order.

        Returns True if a new order was placed.
        """
        if symbol in self._pending:
            return False  # already have a pending order

        if qty <= 0 or current_price <= 0:
            return False

        # Scale qty down — limit orders are conservative (default 50%)
        scaled_qty = qty * self._qty_scale
        if scaled_qty <= 0:
            return False

        # Calculate limit price with offset for favorable fill
        offset_mult = self._offset_bps / 10_000
        if side == "buy":
            # Place below current price (buy on dip)
            limit_price = current_price * (1 - offset_mult)
        else:
            # Place above current price (sell on spike)
            limit_price = current_price * (1 + offset_mult)

        # Round price to 2 decimal places for most crypto pairs
        # BTC: $0.10 tick, ETH: $0.01 tick — 2 decimals safe for both
        limit_price = round(limit_price, 2)

        try:
            resp = self._adapter.send_limit_order(
                symbol=symbol,
                side=side.capitalize(),
                qty=scaled_qty,
                price=limit_price,
                tif="GTC",
                post_only=self._post_only,
            )
            order_id = resp.get("orderId")
            if order_id and resp.get("status") != "error":
                self._pending[symbol] = {
                    "orderId": order_id,
                    "side": side,
                    "qty": scaled_qty,
                    "price": limit_price,
                    "placed_at": time.time(),
                }
                logger.info(
                    "LIMIT PRE-PLACED %s %s %.4f @ $%.2f "
                    "(z=%+.2f, dz=%.1f, offset=%dbps, scale=%.0f%%)",
                    symbol, side, scaled_qty, limit_price,
                    z_score, deadzone, int(self._offset_bps),
                    self._qty_scale * 100,
                )
                return True
            else:
                logger.warning(
                    "LIMIT order failed %s %s: %s",
                    symbol, side, resp.get("retMsg", resp),
                )
        except Exception:
            logger.warning(
                "LIMIT order exception %s %s", symbol, side, exc_info=True,
            )
        return False

    def cancel_stale(self, symbol: str) -> None:
        """Cancel pending limit order if signal disappeared or bar closed."""
        if symbol not in self._pending:
            return
        info = self._pending[symbol]
        try:
            resp = self._adapter.cancel_order(symbol, info["orderId"])
            status = resp.get("status", "")
            logger.info(
                "LIMIT CANCELLED %s orderId=%s status=%s",
                symbol, info["orderId"], status,
            )
        except Exception:
            logger.debug("LIMIT cancel exception %s", symbol, exc_info=True)
        del self._pending[symbol]

    def cancel_all_pending(self) -> None:
        """Cancel all pending limit orders (e.g., on shutdown)."""
        for symbol in list(self._pending):
            self.cancel_stale(symbol)

    def check_fill(self, symbol: str) -> dict[str, Any] | None:
        """Check if pending order was filled (fully or partially).

        Returns dict with `qty` reflecting actual filled quantity (NOT
        the original intent), or None if not yet filled at all.

        Behavior change vs old version:
          - PARTIAL fills now return the partial qty (was: returned None,
            treating "still in open_orders" as "not filled at all")
          - The returned dict's `qty` is the actual filled coin amount,
            not the placement intent
          - Any remaining unfilled portion is cancelled before returning
            so we don't have a phantom limit hanging around
        """
        if symbol not in self._pending:
            return None
        info = self._pending[symbol]
        try:
            open_orders = self._adapter.get_open_orders(symbol=symbol)
            still_open = None
            for order in open_orders:
                if getattr(order, "order_id", None) == info["orderId"]:
                    still_open = order
                    break

            if still_open is not None:
                # Order still active: check filled_qty for partial fill
                filled = float(getattr(still_open, "filled_qty", 0) or 0)
                if filled > 0:
                    # Partial fill — cancel remaining, return what filled
                    try:
                        self._adapter.cancel_order(symbol, info["orderId"])
                    except Exception:
                        logger.debug("LIMIT partial-cancel exception %s", symbol, exc_info=True)
                    del self._pending[symbol]
                    logger.info(
                        "LIMIT PARTIAL %s %s %.4f / %.4f intent — cancelled remainder",
                        symbol, info["side"], filled, info["qty"],
                    )
                    return {**info, "qty": filled, "partial": True}
                return None  # truly still open, not filled

            # Order not in open orders: filled or cancelled
            fills = self._adapter.get_recent_fills(symbol=symbol)
            matched_qty = 0.0
            matched_price = 0.0
            for fill in fills:
                if (
                    getattr(fill, "side", "").lower() == info["side"]
                    and abs(float(getattr(fill, "price", 0)) - info["price"])
                    < info["price"] * 0.005
                ):
                    fq = float(getattr(fill, "qty", 0) or 0)
                    fp = float(getattr(fill, "price", 0) or 0)
                    matched_qty += fq
                    matched_price = (
                        (matched_price * (matched_qty - fq) + fp * fq) / max(matched_qty, 1e-9)
                    )

            if matched_qty > 0:
                del self._pending[symbol]
                logger.info(
                    "LIMIT FILLED %s %s %.4f @ $%.2f (limit was $%.2f, intent was %.4f)",
                    symbol, info["side"], matched_qty, matched_price,
                    info["price"], info["qty"],
                )
                return {**info, "qty": matched_qty, "price": matched_price}

            # Not in open orders and no matching fill: cancelled externally
            del self._pending[symbol]
            logger.info("LIMIT EXPIRED/CANCELLED %s orderId=%s", symbol, info["orderId"])
        except Exception:
            logger.debug("LIMIT check_fill exception %s", symbol, exc_info=True)
        return None

    def cancel_expired(self) -> int:
        """Cancel orders that exceeded TTL. Returns count of cancelled orders."""
        now = time.time()
        cancelled = 0
        for symbol in list(self._pending):
            info = self._pending[symbol]
            age = now - info["placed_at"]
            if age > self._ttl_seconds:
                logger.info(
                    "LIMIT TTL EXPIRED %s orderId=%s age=%.0fs > ttl=%.0fs",
                    symbol, info["orderId"], age, self._ttl_seconds,
                )
                self.cancel_stale(symbol)
                cancelled += 1
        return cancelled

    def check_signal_conflict(
        self, symbol: str, new_signal: int,
    ) -> bool:
        """Check if a filled limit order conflicts with the new bar signal.

        Returns True if there is a conflict (limit fill side != new signal
        direction), meaning the caller should close the position.
        """
        info = self._pending.get(symbol)
        if info is None:
            return False
        limit_side_int = 1 if info["side"] == "buy" else -1
        if new_signal != 0 and new_signal != limit_side_int:
            logger.warning(
                "LIMIT CONFLICT %s: limit_fill=%s but new_signal=%d — must close",
                symbol, info["side"], new_signal,
            )
            return True
        return False

    def has_pending(self, symbol: str) -> bool:
        """Check if there is a pending limit order for this symbol."""
        return symbol in self._pending

    def get_pending_info(self, symbol: str) -> dict[str, Any] | None:
        """Get pending order info for a symbol."""
        return self._pending.get(symbol)

    @property
    def pending_symbols(self) -> list[str]:
        """List all symbols with pending limit orders."""
        return list(self._pending)
