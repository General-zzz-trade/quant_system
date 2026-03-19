"""Bybit demo market maker — runs A-S quoting on Bybit linear perpetuals.

Uses existing MM engine components with Bybit REST API for order management.
Depth feed via Bybit public WS. Orders via REST (no WS-API on Bybit).

Usage:
    python3 -m scripts.run_bybit_mm --symbol ETHUSDT --leverage 20
    python3 -m scripts.run_bybit_mm --symbol ETHUSDT --leverage 20 --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
import websocket

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.market_maker.config import MarketMakerConfig
from execution.market_maker.inventory_tracker import InventoryTracker
from execution.market_maker.perp_quoter import PerpQuoter
from execution.market_maker.risk_monitor import RiskMonitor
from execution.market_maker.vol_estimator import VolEstimator
from execution.market_maker.metrics import MetricsCollector

log = logging.getLogger("bybit_mm")


class BybitMMRunner:
    """Market maker on Bybit demo using REST orders + WS depth."""

    def __init__(self, symbol: str, cfg: MarketMakerConfig, adapter, leverage: int = 20):
        self._symbol = symbol
        self._cfg = cfg
        self._adapter = adapter
        self._leverage = leverage

        # Components
        self._quoter = PerpQuoter(cfg)
        self._inventory = InventoryTracker(
            max_notional=cfg.max_inventory_notional,
            daily_loss_limit=cfg.daily_loss_limit,
        )
        self._risk = RiskMonitor(cfg)
        self._vol = VolEstimator(alpha=cfg.vol_ema_alpha, min_trades=20)
        self._metrics = MetricsCollector(maker_rebate_bps=-1.0)

        # Microstructure (VPIN + imbalance from tick data)
        self._micro = None
        try:
            from _quant_hotpath import RustStreamingMicrostructure
            self._micro = RustStreamingMicrostructure(
                trade_buffer_size=200, vpin_bucket_volume=50.0, vpin_n_buckets=50,
            )
            log.info("RustStreamingMicrostructure enabled")
        except ImportError:
            log.warning("RustStreamingMicrostructure not available")

        # State
        self._running = False
        self._best_bid = 0.0
        self._best_ask = 0.0
        self._mid = 0.0
        self._vpin = 0.0
        self._ob_imbalance = 0.0
        self._bid_depth = 0.0
        self._ask_depth = 0.0
        self._funding_rate = 0.0
        self._tick_count = 0
        self._last_quote_time = 0.0
        self._quote_mode = "adaptive"  # "as", "bbo", "adaptive"
        self._started_at = 0.0
        self._last_ws_message_time = 0.0
        self._last_depth_time = 0.0

        # Live order IDs
        self._bid_order_id: str | None = None
        self._ask_order_id: str | None = None
        self._last_bid_price: float = 0.0
        self._last_ask_price: float = 0.0

        # Fill detection: track last known exchange position to avoid duplicates (B1 fix)
        self._last_exchange_qty: float = 0.0
        self._last_exchange_side: str = ""

    def start(self):
        """Start the market maker."""
        self._running = True
        self._cfg.validate()
        now = time.monotonic()
        self._started_at = now
        self._last_ws_message_time = now
        self._last_depth_time = 0.0

        # Set leverage on exchange
        try:
            self._adapter._client.post('/v5/position/set-leverage', body={
                'category': 'linear',
                'symbol': self._symbol,
                'buyLeverage': str(self._leverage),
                'sellLeverage': str(self._leverage),
            })
            log.info("Set leverage to %dx for %s", self._leverage, self._symbol)
        except Exception as e:
            log.warning("Leverage set: %s", e)

        # Start WS for depth + trades
        self._start_ws()

    def _start_ws(self):
        """Connect to Bybit public WS for orderbook + trades."""
        url = "wss://stream.bybit.com/v5/public/linear"

        def on_message(ws, msg):
            self._handle_ws_message(msg)

        def on_open(ws):
            subs = [
                f"orderbook.1.{self._symbol}",  # L1 BBO (most stable)
                f"publicTrade.{self._symbol}",
                f"tickers.{self._symbol}",
            ]
            ws.send(json.dumps({"op": "subscribe", "args": subs}))
            log.info("WS subscribed: %s", subs)

        def on_error(ws, error):
            log.error("WS error: %s", error)

        def on_close(ws, code, msg):
            log.warning("WS closed: %s %s", code, msg)
            if self._running:
                time.sleep(3)
                self._start_ws()

        self._ws = websocket.WebSocketApp(
            url,
            on_message=on_message,
            on_open=on_open,
            on_error=on_error,
            on_close=on_close,
        )
        t = threading.Thread(target=self._ws.run_forever, daemon=True, name="bybit-ws")
        t.start()

    def _handle_ws_message(self, msg: str) -> None:
        """Parse and route a single Bybit public WS message."""
        try:
            data = json.loads(msg)
            self._last_ws_message_time = time.monotonic()
            topic = data.get("topic", "")
            if "orderbook" in topic:
                self._on_depth(data.get("data", {}))
            elif "publicTrade" in topic:
                self._on_trade(data.get("data", []))
            elif "tickers" in topic:
                d = data.get("data", {})
                fr = d.get("fundingRate")
                if fr:
                    self._funding_rate = float(fr)
        except Exception:
            log.exception("WS message handling failed: %s", msg[:200])

    def _on_depth(self, data: dict):
        """Handle orderbook update with imbalance computation."""
        bids = data.get("b", [])
        asks = data.get("a", [])
        if not bids or not asks:
            return
        self._last_depth_time = time.monotonic()

        self._best_bid = float(bids[0][0])
        self._best_ask = float(asks[0][0])
        self._mid = (self._best_bid + self._best_ask) / 2.0
        self._tick_count += 1

        # Compute depth and imbalance from top 5 levels
        bid_depth = sum(float(b[1]) * float(b[0]) for b in bids[:5])
        ask_depth = sum(float(a[1]) * float(a[0]) for a in asks[:5])
        self._bid_depth = bid_depth
        self._ask_depth = ask_depth
        total = bid_depth + ask_depth
        if total > 0:
            self._ob_imbalance = (bid_depth - ask_depth) / total

        # Update Rust microstructure
        if self._micro is not None:
            try:
                pairs = [(float(b[0]), float(b[1])) for b in bids[:5]]
                apairs = [(float(a[0]), float(a[1])) for a in asks[:5]]
                result = self._micro.on_depth(pairs, apairs)
                self._vpin = result.get("vpin", self._vpin)
                self._ob_imbalance = result.get("ob_imbalance", self._ob_imbalance)
            except Exception:
                log.warning("Microstructure depth update failed", exc_info=True)

        self._maybe_update_quotes()

    def _on_trade(self, trades: list):
        """Handle public trades with VPIN update."""
        for t in trades:
            price = float(t.get("p", 0))
            if price > 0:
                self._vol.on_trade(price)
                # Update microstructure
                if self._micro is not None:
                    try:
                        side = t.get("S", "Buy")
                        s = "buy" if side == "Buy" else "sell"
                        qty = float(t.get("v", 0))
                        result = self._micro.on_trade(price, qty, s)
                        self._vpin = result.get("vpin", self._vpin)
                    except Exception:
                        log.warning("Microstructure trade update failed", exc_info=True)

    def _maybe_update_quotes(self):
        """Core quote update cycle."""
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
            cancelled = self._cancel_all()
            if not cancelled:
                log.critical("Kill path cancel-all failed; resting orders may still be live")
            flattened = self._flatten()
            if not flattened and abs(self._inventory.net_qty) >= 1e-10:
                log.critical("Kill path flatten failed; position may remain open")
            self._running = False
            return
        if state == "paused":
            if not self._cancel_all():
                log.warning("Pause path cancel-all failed; keeping quotes frozen")
            return

        if not self._vol.ready or self._mid <= 0:
            return

        vol = self._vol.volatility
        if vol <= 0:
            return

        T = self._cfg.time_horizon_s
        time_remaining = max(0.1, T - (now % T)) / T

        quote = self._quoter.compute_quotes(
            mid=self._mid,
            inventory=self._inventory.net_qty,
            vol=vol,
            time_remaining=time_remaining,
            funding_rate=self._funding_rate,
            vpin=self._vpin,
            best_bid=self._best_bid,
            best_ask=self._best_ask,
            ob_imbalance=self._ob_imbalance,
            mode=self._quote_mode,
        )
        if quote is None:
            return

        # Side blocking
        bid_price = quote.bid if self._inventory.can_buy(self._mid) else None
        ask_price = quote.ask if self._inventory.can_sell(self._mid) else None

        # B3 fix: ensure PostOnly safety — bid must be < best_ask, ask > best_bid
        tick = self._cfg.tick_size
        if bid_price is not None and self._best_ask > 0:
            bid_price = min(bid_price, self._best_ask - tick)
        if ask_price is not None and self._best_bid > 0:
            ask_price = max(ask_price, self._best_bid + tick)

        # B5 fix: only cancel+resubmit if prices changed by >= 1 tick
        bid_changed = bid_price is None or abs((bid_price or 0) - self._last_bid_price) >= tick
        ask_changed = ask_price is None or abs((ask_price or 0) - self._last_ask_price) >= tick

        if not bid_changed and not ask_changed:
            return  # prices same, keep existing orders

        if not self._cancel_all():
            log.warning("Skipping quote refresh because cancel-all failed")
            return

        # Primary quotes
        if bid_price is not None:
            self._place_limit("Buy", bid_price, self._cfg.order_size_eth)
            self._last_bid_price = bid_price
        if ask_price is not None:
            self._place_limit("Sell", ask_price, self._cfg.order_size_eth)
            self._last_ask_price = ask_price

        # C6 fix: second layer at half size, only when VPIN is low (calm market)
        if self._vpin < 0.5:
            half_size = round(self._cfg.order_size_eth * 0.5, 3)
            if half_size >= 0.001 and bid_price is not None:
                self._place_limit("Buy", bid_price - tick * 3, half_size)
            if half_size >= 0.001 and ask_price is not None:
                self._place_limit("Sell", ask_price + tick * 3, half_size)

        self._metrics.record_quote()
        self._metrics.record_spread(quote.spread / self._mid * 10000)

        # Periodic logging
        if self._tick_count % 50 == 0:
            self._inventory.update_unrealised(self._mid)
            self._metrics.log_summary(self._inventory.net_qty, self._inventory.unrealised_pnl)
            log.info(
                "tick=%d mid=%.2f vol=%.6f vpin=%.3f imb=%+.2f inv=%.4f pnl=%.4f bid=%s ask=%s mode=%s",
                self._tick_count, self._mid, vol, self._vpin, self._ob_imbalance,
                self._inventory.net_qty, self._inventory.total_pnl,
                f"{bid_price:.2f}" if bid_price else "—",
                f"{ask_price:.2f}" if ask_price else "—",
                self._quote_mode,
            )

    def _place_limit(self, side: str, price: float, qty: float) -> str | None:
        """Place a limit order on Bybit."""
        if self._cfg.dry_run:
            log.info("DRY %s %.4f @ %.2f", side, qty, price)
            return None

        try:
            result = self._adapter._client.post('/v5/order/create', body={
                'category': 'linear',
                'symbol': self._symbol,
                'side': side,
                'orderType': 'Limit',
                'qty': str(qty),
                'price': str(price),
                'timeInForce': 'PostOnly',  # maker only
            })
            ret_code = result.get('retCode', -1)
            if ret_code == 0:
                oid = result.get('result', {}).get('orderId', '')
                if side == "Buy":
                    self._bid_order_id = oid
                else:
                    self._ask_order_id = oid
                log.debug("PLACED %s %.4f @ %.2f oid=%s", side, qty, price, oid)
                return oid
            else:
                log.debug("Order rejected: %s", result.get('retMsg', '?'))
        except Exception:
            log.exception("Place order failed")
        return None

    def _cancel_all(self) -> bool:
        """Cancel all open orders.

        Returns ``True`` only when the exchange acknowledged the cancel-all request.
        Local order ids are preserved on failure so the runner does not pretend
        the exchange is flat when it is not.
        """
        try:
            result = self._adapter._client.post('/v5/order/cancel-all', body={
                'category': 'linear',
                'symbol': self._symbol,
            })
        except Exception:
            log.exception("Cancel-all failed")
            return False
        ret_code = result.get('retCode', -1)
        if ret_code != 0:
            log.error("Cancel-all rejected: code=%s msg=%s", ret_code, result.get('retMsg', '?'))
            return False
        self._bid_order_id = None
        self._ask_order_id = None
        self._metrics.record_cancel()
        return True

    def _flatten(self) -> bool:
        """Close any open position at market.

        Returns ``True`` when the account is already flat or the exchange accepts
        the reduce-only flatten order.
        """
        if abs(self._inventory.net_qty) < 1e-10:
            return True
        side = "Sell" if self._inventory.net_qty > 0 else "Buy"
        qty = abs(self._inventory.net_qty)
        if self._cfg.dry_run:
            log.info("DRY FLATTEN %s %.4f", side, qty)
            return True
        try:
            result = self._adapter._client.post('/v5/order/create', body={
                'category': 'linear',
                'symbol': self._symbol,
                'side': side,
                'orderType': 'Market',
                'qty': str(qty),
                'reduceOnly': True,
            })
            ret_code = result.get('retCode', -1)
            if ret_code != 0:
                log.error("Flatten rejected: code=%s msg=%s", ret_code, result.get('retMsg', '?'))
                return False
            log.warning("FLATTEN %s %.4f", side, qty)
            return True
        except Exception:
            log.exception("Flatten failed")
            return False
    def _check_fills(self):
        """Poll exchange position and detect fills by comparing with last known state.

        B1 fix: Compare with last_exchange_qty (not inventory) to avoid
        detecting the same position as a new fill repeatedly.
        B4 fix: Process incremental qty changes, not jumps.
        """
        try:
            resp = self._adapter._client.get('/v5/position/list', params={
                'category': 'linear',
                'symbol': self._symbol,
            })
            pos_list = resp.get('result', {}).get('list', [])

            # Determine current exchange position
            exchange_qty = 0.0
            exchange_side = ""
            exchange_price = self._mid
            for p in pos_list:
                size = float(p.get('size', 0))
                if size > 0:
                    exchange_side = p.get('side', '')
                    exchange_qty = size if exchange_side == 'Buy' else -size
                    exchange_price = float(p.get('avgPrice', self._mid))
                    break

            # Compare with last known exchange state (not internal inventory)
            prev_qty = self._last_exchange_qty
            if abs(exchange_qty - prev_qty) < 1e-6:
                return  # no change

            # Record fill: only the incremental change
            delta = exchange_qty - prev_qty
            fill_side = "buy" if delta > 0 else "sell"
            fill_qty = abs(delta)

            # Update internal state
            self._last_exchange_qty = exchange_qty
            self._last_exchange_side = exchange_side

            # C4 fix: use mid price for fill (avgPrice is position average, not fill price)
            # This gives more accurate per-fill PnL tracking
            fill_price = self._mid if self._mid > 0 else exchange_price

            rpnl = self._inventory.on_fill(fill_side, fill_qty, fill_price)
            self._metrics.record_fill(fill_side, fill_qty, fill_price, rpnl)
            log.info("FILL %s %.4f @ %.2f rpnl=%.4f inv=%.4f exch=%.4f",
                     fill_side, fill_qty, fill_price, rpnl,
                     self._inventory.net_qty, exchange_qty)
        except Exception:
            log.exception("Fill polling failed")

    def _check_market_data_watchdog(self) -> None:
        """Fail fast when market data stops advancing.

        The runner should not stay `systemd active` while blind to the book.
        Let systemd restart the process instead of silently drifting.
        """
        now = time.monotonic()
        ws_idle = now - self._last_ws_message_time
        if ws_idle > self._cfg.market_data_stale_s:
            log.error("Market data stale: no WS message for %.1fs", ws_idle)
            raise RuntimeError(
                f"market data stale: no WS message for {ws_idle:.1f}s"
            )
        if self._last_depth_time == 0.0:
            startup_idle = now - self._started_at
            if startup_idle > self._cfg.market_data_stale_s:
                log.error("Market data stale: no orderbook depth after %.1fs", startup_idle)
                raise RuntimeError(
                    "market data stale: no orderbook depth received after startup"
                )
            return
        depth_idle = now - self._last_depth_time
        if depth_idle > self._cfg.market_data_stale_s:
            log.error("Market data stale: no orderbook depth for %.1fs", depth_idle)
            raise RuntimeError(
                f"market data stale: no orderbook depth for {depth_idle:.1f}s"
            )

    def run(self):
        """Main blocking loop."""
        self.start()
        log.info("Market maker running: %s leverage=%dx dry_run=%s",
                 self._symbol, self._leverage, self._cfg.dry_run)
        try:
            while self._running:
                self._check_market_data_watchdog()
                self._check_fills()
                time.sleep(1.0)
        except KeyboardInterrupt:
            log.info("Keyboard interrupt")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        cancelled = self._cancel_all()
        if not cancelled:
            log.warning("Stop path cancel-all failed; resting orders may remain live")
        flattened = self._flatten()
        if not flattened and abs(self._inventory.net_qty) >= 1e-10:
            log.warning("Stop path flatten failed; position may remain open")
        if hasattr(self, '_ws'):
            self._ws.close()
        self._metrics.log_summary(self._inventory.net_qty, self._inventory.unrealised_pnl)
        log.info("Market maker stopped. Fills=%d PnL=%.4f",
                 self._inventory.total_fills, self._inventory.realised_pnl)


def main():
    parser = argparse.ArgumentParser(description="Bybit demo market maker")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--leverage", type=int, default=20)
    parser.add_argument("--order-size", type=float, default=0.01, help="Order size in base")
    parser.add_argument("--max-inventory", type=float, default=100.0, help="Max inventory notional $")
    parser.add_argument("--loss-limit", type=float, default=20.0, help="Daily loss limit $")
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--min-spread", type=float, default=1.0, help="Min spread bps")
    parser.add_argument("--max-spread", type=float, default=20.0, help="Max spread bps")
    parser.add_argument("--mode", default="adaptive", choices=["as", "bbo", "adaptive"],
                        help="Quote mode: as(wide), bbo(tight), adaptive(auto)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-5s %(name)s  %(message)s"
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(level=level, format=fmt, handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/bybit_mm.log"),
    ])

    # Load Bybit adapter
    from execution.adapters.bybit.config import BybitConfig
    from execution.adapters.bybit.adapter import BybitAdapter

    bybit_cfg = BybitConfig(
        api_key=os.environ['BYBIT_API_KEY'],
        api_secret=os.environ['BYBIT_API_SECRET'],
        base_url=os.environ.get('BYBIT_BASE_URL', 'https://api-demo.bybit.com'),
    )
    adapter = BybitAdapter(bybit_cfg)

    mm_cfg = MarketMakerConfig(
        symbol=args.symbol,
        order_size_eth=args.order_size,
        max_inventory_notional=args.max_inventory,
        daily_loss_limit=args.loss_limit,
        gamma=args.gamma,
        min_spread_bps=args.min_spread,
        max_spread_bps=args.max_spread,
        dry_run=args.dry_run,
        quote_update_interval_s=0.5,  # 500ms on Bybit (REST rate limit)
        stale_order_s=3.0,
        # B2 fix: MM needs higher tolerance for small losses (normal spread friction)
        circuit_breaker_losses=15,     # was 3 — too sensitive for MM
        circuit_breaker_pause_s=30.0,  # was 120s — shorter pause
    )

    runner = BybitMMRunner(args.symbol, mm_cfg, adapter, leverage=args.leverage)
    runner._quote_mode = args.mode

    def _signal_handler(signum, frame):
        log.info("Signal %d, shutting down...", signum)
        runner.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    runner.run()


if __name__ == "__main__":
    main()
