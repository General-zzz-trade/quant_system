"""Market maker engine — main event loop wiring all components.

Flow per depth tick (100ms):
  1. Update vol estimator (from trades)
  2. Compute quotes via PerpQuoter
  3. Apply inventory side-blocking
  4. Risk monitor check
  5. OrderManager update (cancel stale, submit new)
  6. Process fills → InventoryTracker
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .config import MarketMakerConfig
from .inventory_tracker import InventoryTracker
from .order_manager import OrderManager
from .perp_quoter import PerpQuoter
from .risk_monitor import RiskMonitor
from .vol_estimator import VolEstimator

log = logging.getLogger(__name__)


class MarketMakerEngine:
    """Orchestrates quoting, fills, risk, and order management."""

    def __init__(
        self,
        cfg: MarketMakerConfig,
        gateway=None,
        depth_client=None,
        user_stream=None,
        microstructure=None,
    ) -> None:
        self._cfg = cfg
        self._gateway = gateway
        self._depth_client = depth_client
        self._user_stream = user_stream

        # Core components
        self._quoter = PerpQuoter(cfg)
        self._inventory = InventoryTracker(
            max_notional=cfg.max_inventory_notional,
            daily_loss_limit=cfg.daily_loss_limit,
        )
        self._orders = OrderManager(cfg, gateway)
        self._risk = RiskMonitor(cfg)
        self._vol = VolEstimator(alpha=cfg.vol_ema_alpha)

        # Microstructure (Rust)
        self._micro = microstructure  # RustStreamingMicrostructure or None

        # State
        self._running = False
        self._last_quote_time: float = 0.0
        self._funding_rate: float = 0.0
        self._best_bid: float = 0.0
        self._best_ask: float = 0.0
        self._mid: float = 0.0
        self._vpin: float = 0.0
        self._tick_count: int = 0
        self._last_micro_error_log: dict[str, float] = {"depth": 0.0, "trade": 0.0}

    @property
    def inventory(self) -> InventoryTracker:
        return self._inventory

    @property
    def risk(self) -> RiskMonitor:
        return self._risk

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Event handlers ──────────────────────────────────────

    def on_depth(self, snapshot: Any) -> None:
        """Handle depth update from BinanceDepthStreamClient."""
        bb = float(snapshot.best_bid) if snapshot.best_bid else 0.0
        ba = float(snapshot.best_ask) if snapshot.best_ask else 0.0
        if bb <= 0 or ba <= 0:
            return

        self._best_bid = bb
        self._best_ask = ba
        self._mid = (bb + ba) / 2.0
        self._tick_count += 1

        # Update microstructure if available
        if self._micro is not None:
            try:
                bids = [(float(lv.price), float(lv.qty)) for lv in snapshot.bids[:5]]
                asks = [(float(lv.price), float(lv.qty)) for lv in snapshot.asks[:5]]
                result = self._micro.on_depth(bids, asks)
                self._vpin = result.get("vpin", 0.0)
            except Exception:
                self._log_microstructure_error("depth")

        self._maybe_update_quotes()

    def on_trade(self, price: float, qty: float, side: str) -> None:
        """Handle aggTrade for vol estimation and microstructure."""
        self._vol.on_trade(price)
        if self._micro is not None:
            try:
                result = self._micro.on_trade(price, qty, side)
                self._vpin = result.get("vpin", 0.0)
            except Exception:
                self._log_microstructure_error("trade")

    def on_user_event(self, event: dict) -> None:
        """Handle user data stream events (fills, order updates)."""
        event_type = event.get("e", "")

        if event_type == "ORDER_TRADE_UPDATE":
            o = event.get("o", {})
            coid = o.get("c", "")
            status = o.get("X", "")
            side = o.get("S", "").upper()
            fill_qty = float(o.get("l", 0))  # last filled qty
            fill_price = float(o.get("L", 0))  # last fill price

            # Update order manager
            self._orders.on_order_response({
                "clientOrderId": coid,
                "status": status,
            })

            # Process fill
            if fill_qty > 0 and fill_price > 0:
                self._orders.on_fill(coid, fill_qty)
                fill_side = "buy" if side == "BUY" else "sell"
                rpnl = self._inventory.on_fill(fill_side, fill_qty, fill_price)
                log.info(
                    "FILL %s %.4f @ %.2f rpnl=%.4f net=%.4f",
                    fill_side, fill_qty, fill_price, rpnl, self._inventory.net_qty,
                )

        elif event_type == "ACCOUNT_UPDATE":
            # Could extract balance updates here
            pass

    def set_funding_rate(self, rate: float) -> None:
        """Update current funding rate (from REST poll or WS)."""
        self._funding_rate = rate

    # ── Quote cycle ─────────────────────────────────────────

    def _maybe_update_quotes(self) -> None:
        """Run quote update if enough time has passed."""
        now = time.monotonic()
        if now - self._last_quote_time < self._cfg.quote_update_interval_s:
            return
        self._last_quote_time = now

        # Risk check
        state = self._risk.check(
            self._inventory.daily_pnl,
            self._inventory.consecutive_losses,
        )
        if state == "killed":
            self._shutdown_positions()
            return
        if state == "paused":
            self._orders.cancel_all()
            return

        # Need vol estimate
        vol = self._vol.volatility
        if not self._vol.ready or vol <= 0:
            return

        # Time remaining in horizon (rolling 5 min)
        T = self._cfg.time_horizon_s
        time_remaining = max(0.1, T - (now % T)) / T

        # Compute quotes
        quote = self._quoter.compute_quotes(
            mid=self._mid,
            inventory=self._inventory.net_qty,
            vol=vol,
            time_remaining=time_remaining,
            funding_rate=self._funding_rate,
            vpin=self._vpin,
        )
        if quote is None:
            return

        # Inventory side-blocking
        ref = self._mid
        target_bid = quote.bid if self._inventory.can_buy(ref) else None
        target_ask = quote.ask if self._inventory.can_sell(ref) else None

        # Update orders
        self._orders.update_quotes(
            target_bid=target_bid,
            target_ask=target_ask,
            bid_size=quote.bid_size,
            ask_size=quote.ask_size,
            best_bid=self._best_bid,
            best_ask=self._best_ask,
        )

        # Periodic logging
        if self._tick_count % 100 == 0:
            self._inventory.update_unrealised(self._mid)
            log.info(
                "tick=%d mid=%.2f vol=%.6f vpin=%.3f inv=%.4f pnl=%.4f spread=%.2f",
                self._tick_count,
                self._mid,
                vol,
                self._vpin,
                self._inventory.net_qty,
                self._inventory.total_pnl,
                quote.spread,
            )

        # Cleanup terminal orders
        self._orders.cleanup_done_orders()

    def _shutdown_positions(self) -> None:
        """Cancel all orders and flatten inventory."""
        cancelled = self._orders.cancel_all()
        if cancelled:
            log.warning("Shutdown: cancelled %d orders", cancelled)

        if abs(self._inventory.net_qty) > 1e-10 and self._gateway is not None:
            side = "SELL" if self._inventory.net_qty > 0 else "BUY"
            qty = abs(self._inventory.net_qty)
            if not self._cfg.dry_run:
                try:
                    self._gateway.submit_order(
                        symbol=self._cfg.symbol,
                        side=side,
                        order_type="MARKET",
                        quantity=str(qty),
                    )
                    log.warning("Flatten: %s %.4f at market", side, qty)
                except Exception:
                    log.exception("Failed to flatten position")
            else:
                log.info("DRY flatten: %s %.4f", side, qty)

        self._running = False

    def _log_microstructure_error(self, stream: str, *, throttle_s: float = 60.0) -> None:
        now = time.monotonic()
        last = self._last_micro_error_log.get(stream, 0.0)
        if now - last < throttle_s:
            return
        self._last_micro_error_log[stream] = now
        log.warning(
            "Microstructure %s update failed; keeping previous state",
            stream,
            exc_info=True,
        )

    # ── Lifecycle ───────────────────────────────────────────

    def start(self) -> None:
        """Mark engine as running. Caller drives the event loop."""
        self._cfg.validate()
        self._running = True
        log.info(
            "Market maker started: %s dry_run=%s capital=%.0f",
            self._cfg.symbol,
            self._cfg.dry_run,
            self._cfg.capital,
        )

    def stop(self) -> None:
        """Graceful shutdown: cancel all, flatten, stop."""
        log.info("Stopping market maker...")
        self._risk.force_kill("manual_stop")
        self._shutdown_positions()
        self._running = False
        log.info(
            "Stopped. fills=%d rpnl=%.4f",
            self._inventory.total_fills,
            self._inventory.realised_pnl,
        )

    def run_sync(self) -> None:
        """Blocking event loop using depth_client.step().

        This is the simplest sync driver — call depth_client.step()
        in a tight loop, process user events between ticks.
        """
        if self._depth_client is None:
            raise RuntimeError("No depth_client provided")

        self.start()
        try:
            while self._running:
                # Depth tick
                try:
                    snap = self._depth_client.step()
                    if snap is not None:
                        self.on_depth(snap)
                except Exception:
                    log.exception("Depth step error")
                    time.sleep(0.1)

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt received")
        finally:
            self.stop()
