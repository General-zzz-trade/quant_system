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
from execution.market_maker.metrics import MetricsCollector
from scripts.ops.runtime_ownership import claim_bybit_symbol_lease
from scripts.ops.runtime_kill_latch import (
    STARTUP_GUARD_EXIT_CODE,
    build_bybit_kill_latch,
    require_kill_latch_clear,
)

log = logging.getLogger("bybit_mm")

# Per-symbol exchange specs
_SYMBOL_SPECS = {
    'ETHUSDT':  {'tick_size': 0.01,   'qty_step': 0.01},
    'BTCUSDT':  {'tick_size': 0.10,   'qty_step': 0.001},
    'SOLUSDT':  {'tick_size': 0.01,   'qty_step': 0.1},
    'ORDIUSDT': {'tick_size': 0.001,  'qty_step': 0.01},
    'TIAUSDT':  {'tick_size': 0.0001, 'qty_step': 0.1},
    'SUIUSDT':  {'tick_size': 0.0001, 'qty_step': 0.1},
    'AXSUSDT':  {'tick_size': 0.001,  'qty_step': 0.1},
    'INJUSDT':  {'tick_size': 0.001,  'qty_step': 0.1},
    'ADAUSDT':  {'tick_size': 0.0001, 'qty_step': 1.0},
}


def _claim_runtime_symbol(adapter, symbol: str, *, dry_run: bool):
    if dry_run:
        return None
    return claim_bybit_symbol_lease(
        adapter=adapter,
        service_name="bybit-mm.service",
        symbols=(str(symbol).upper(),),
    )


def _build_runtime_kill_latch(adapter, symbol: str, *, dry_run: bool):
    if dry_run:
        return None
    return build_bybit_kill_latch(
        adapter=adapter,
        service_name="bybit-mm.service",
        scope_name=str(symbol).upper(),
    )


class BybitMMRunner:
    """Market maker on Bybit demo using REST orders + WS depth."""

    def __init__(self, symbol: str, cfg: MarketMakerConfig, adapter, leverage: int = 20, kill_latch=None):
        self._symbol = symbol
        self._cfg = cfg
        self._adapter = adapter
        self._leverage = leverage
        self._kill_latch = kill_latch

        # ── v5: Rust MM engine replaces Python quoter+inventory+vol+alpha ──
        from _quant_hotpath import RustMarketMaker, RustMMConfig, RustStreamingMicrostructure
        rust_cfg = RustMMConfig(
            tick_size=cfg.tick_size,
            qty_step=cfg.qty_step,
            order_size=cfg.order_size_eth,
            max_inventory_notional=cfg.max_inventory_notional,
            gamma=cfg.gamma,
            kappa=cfg.kappa,
            time_horizon_s=cfg.time_horizon_s,
            min_spread_bps=cfg.min_spread_bps,
            max_spread_bps=cfg.max_spread_bps,
            vpin_pause_threshold=0.8,
            vpin_widen_threshold=0.6,
            vpin_spread_mult=3.0,
            trend_pause_pct=0.005,
            inv_timeout_s=120.0,
            daily_loss_limit=cfg.daily_loss_limit,
        )
        self._engine = RustMarketMaker(rust_cfg)
        self._micro = RustStreamingMicrostructure(
            trade_buffer_size=200, vpin_bucket_volume=50.0, vpin_n_buckets=50,
        )
        self._metrics = MetricsCollector(maker_rebate_bps=-1.0)
        log.info("Rust MM engine + microstructure initialized")

        # State (minimal — most is in Rust engine now)
        self._running = False
        self._started_at = 0.0
        self._last_ws_message_time = 0.0
        self._last_depth_time = 0.0
        self._last_quote_time = 0.0
        self._last_bid_price: float = 0.0
        self._last_ask_price: float = 0.0
        self._bid_order_id: str | None = None
        self._ask_order_id: str | None = None
        self._hedge_order_id: str | None = None  # separate from quote orders

        # Fill detection
        self._last_exchange_qty: float = 0.0
        self._last_exchange_side: str = ""

        # Inventory hedge rate-limit
        self._last_taker_hedge_time: float = 0.0
        self._last_hedge_log: float = 0.0

        # ML signal polling
        self._ml_signal_ts: float = 0.0
        # Fill detection cooldown after hedge placement
        self._hedge_place_ts: float = 0.0
        self._fill_cooldown_s: float = 3.0  # skip fill detection for 3s after hedge

    def _arm_kill_latch(self) -> None:
        # C7 fix: use Rust engine state, not deleted Python objects
        if self._kill_latch is None or self._kill_latch.is_armed():
            return
        self._kill_latch.arm(
            reason=f"{self._symbol} risk kill",
            payload={
                "service": "bybit-mm.service",
                "symbol": self._symbol,
                "daily_pnl": round(self._engine.daily_pnl, 6),
            },
        )

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
                    self._engine.set_funding_rate(float(fr))
        except Exception:
            log.exception("WS message handling failed: %s", msg[:200])

    def _on_depth(self, data: dict):
        """Handle orderbook update → feed Rust engine + microstructure."""
        bids = data.get("b", [])
        asks = data.get("a", [])
        if not bids or not asks:
            return
        self._last_depth_time = time.monotonic()

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])

        # Compute imbalance from top levels
        bid_depth = sum(float(b[1]) * float(b[0]) for b in bids[:5])
        ask_depth = sum(float(a[1]) * float(a[0]) for a in asks[:5])
        total = bid_depth + ask_depth
        ob_imbalance = (bid_depth - ask_depth) / total if total > 0 else 0.0

        # Update microstructure for VPIN
        vpin = 0.0
        try:
            pairs = [(float(b[0]), float(b[1])) for b in bids[:5]]
            apairs = [(float(a[0]), float(a[1])) for a in asks[:5]]
            result = self._micro.on_depth(pairs, apairs)
            vpin = result.get("vpin", 0.0)
            ob_imbalance = result.get("ob_imbalance", ob_imbalance)
        except Exception:
            log.debug("Microstructure on_depth failed", exc_info=True)

        # Feed Rust engine
        self._engine.on_depth(best_bid, best_ask, ob_imbalance, vpin)
        if self._engine.tick_count <= 5:
            log.info("DEPTH_RECV #%d bid=%.4f ask=%.4f mid=%.4f",
                     self._engine.tick_count, best_bid, best_ask, self._engine.mid)
        self._maybe_update_quotes()

    def _on_trade(self, trades: list):
        """Handle public trades → feed Rust engine + microstructure."""
        for t in trades:
            price = float(t.get("p", 0))
            if price > 0:
                side_str = t.get("S", "Buy")
                is_buy = side_str == "Buy"
                qty = float(t.get("v", 0))
                # Feed Rust engine (vol estimation + momentum internally)
                self._engine.on_trade(price, qty, is_buy)
                # Feed microstructure for VPIN
                try:
                    s = "buy" if is_buy else "sell"
                    self._micro.on_trade(price, qty, s)
                except Exception:
                    log.debug("Microstructure on_trade failed", exc_info=True)

    def _maybe_update_quotes(self):
        """v10: Simple BBO join + amend. No pingpong — just dual-sided maker quotes."""
        now = time.monotonic()
        if now - self._last_quote_time < self._cfg.quote_update_interval_s:
            return
        self._last_quote_time = now

        # Diagnostic: log state every 100 ticks
        tc = self._engine.tick_count
        if tc > 0 and tc % 100 == 0:
            log.info("DIAG tick=%d mid=%.4f vol=%.8f ranging=%s trades=%d",
                     tc, self._engine.mid, self._engine.vol,
                     self._engine.is_ranging_regime(), self._engine.total_fills)

        # Regime filter (disabled for altcoins — they're always "volatile")
        if False and not self._engine.is_ranging_regime():
            if self._engine.net_qty != 0.0:
                self._force_hedge(self._engine.net_qty, target_inv=0.0, now=now)
            self._cancel_all()
            if self._engine.tick_count % 200 == 0:
                log.info("REGIME_PAUSE trending → paused")
            return

        # Rust quote
        T = self._cfg.time_horizon_s
        time_frac = max(0.1, T - (now % T)) / T
        q = self._engine.compute_quote(time_frac)

        if q.paused:
            if q.mode == "killed":
                try:
                    self._arm_kill_latch()
                except Exception as e:
                    log.error("Failed to arm kill latch: %s", e)
                self._cancel_all()
                self._flatten()
                self._running = False
                log.error("KILL: %s", q.pause_reason)
                return
            self._cancel_all()
            return

        # BBO quotes with amend
        tick = self._cfg.tick_size
        size = self._cfg.order_size_eth
        bid_price = q.bid_price if q.bid_price > 0 else 0.0
        ask_price = q.ask_price if q.ask_price > 0 else 0.0

        bid_changed = bid_price <= 0 or abs(bid_price - self._last_bid_price) >= tick
        ask_changed = ask_price <= 0 or abs(ask_price - self._last_ask_price) >= tick

        if not bid_changed and not ask_changed:
            return

        if bid_changed and bid_price > 0:
            self._place_limit("Buy", bid_price, size)
            self._last_bid_price = bid_price
        elif bid_price <= 0 and self._bid_order_id:
            self._cancel_side("Buy")

        if ask_changed and ask_price > 0:
            self._place_limit("Sell", ask_price, size)
            self._last_ask_price = ask_price
        elif ask_price <= 0 and self._ask_order_id:
            self._cancel_side("Sell")

        self._metrics.record_quote()

        # ── Logging ──────────────────────────────────────────
        inv = self._engine.net_qty
        mid = self._engine.mid
        if self._engine.tick_count % 50 == 0:
            self._metrics.log_summary(inv, 0.0)
            log.info(
                "tick=%d mid=%.2f inv=%.2f pnl=%.2f RT=%d TO=%d "
                "bid=%s ask=%s dd=%.2f maxInv=%.1f negRT=%d [v8]",
                self._engine.tick_count, mid, inv, self._engine.daily_pnl,
                self._engine.round_trips, self._engine.hedge_timeouts,
                f"{bid_price:.2f}" if bid_price > 0 else "—",
                f"{ask_price:.2f}" if ask_price > 0 else "—",
                self._engine.session_max_dd,
                self._engine.max_inventory_seen,
                self._engine.max_consecutive_neg,
            )

    def _place_limit(self, side: str, price: float, qty: float) -> str | None:
        """Place a limit order on Bybit. Uses amend if order already exists."""
        if self._cfg.dry_run:
            log.info("DRY %s %.4f @ %.2f", side, qty, price)
            return None

        # v8: Try amend existing order first (1 API call vs cancel+new = 2)
        existing_oid = self._bid_order_id if side == "Buy" else self._ask_order_id
        if existing_oid:
            try:
                result = self._adapter._client.post('/v5/order/amend', body={
                    'category': 'linear',
                    'symbol': self._symbol,
                    'orderId': existing_oid,
                    'qty': str(qty),
                    'price': str(price),
                })
                ret_code = result.get('retCode', -1)
                if ret_code == 0:
                    log.debug("AMEND %s %.4f @ %.2f oid=%s", side, qty, price, existing_oid)
                    return existing_oid
                # Amend failed (order filled/cancelled/expired) → fall through to new order
            except Exception:
                log.debug("Amend %s failed, placing new order", side, exc_info=True)

        # Place new order
        try:
            result = self._adapter._client.post('/v5/order/create', body={
                'category': 'linear',
                'symbol': self._symbol,
                'side': side,
                'orderType': 'Limit',
                'qty': str(qty),
                'price': str(price),
                'timeInForce': 'PostOnly',
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

    def _place_new_order(self, side: str, price: float, qty: float) -> str | None:
        """Place a NEW GTC limit order for hedge at a safe price.
        Price is adjusted 1 tick inside BBO to ensure it rests in the book."""
        if self._cfg.dry_run:
            return "dry"
        # Ensure price doesn't cross BBO (would be taker)
        ob = None
        try:
            ob = self._adapter._client.get('/v5/market/orderbook', params={
                'category': 'linear', 'symbol': self._symbol, 'limit': '1',
            })
        except Exception:
            log.debug("Orderbook fetch for hedge price failed", exc_info=True)
        if ob:
            r = ob.get('result', {})
            bids = r.get('b', [])
            asks = r.get('a', [])
            if bids and asks:
                bb = float(bids[0][0])
                ba = float(asks[0][0])
                if side == "Buy":
                    price = min(price, bb)  # don't cross above best bid
                else:
                    price = max(price, ba)  # don't cross below best ask
        try:
            result = self._adapter._client.post('/v5/order/create', body={
                'category': 'linear',
                'symbol': self._symbol,
                'side': side,
                'orderType': 'Limit',
                'qty': str(qty),
                'price': str(price),
                'timeInForce': 'GTC',
            })
            ret_code = result.get('retCode', -1)
            if ret_code == 0:
                oid = result.get('result', {}).get('orderId', '')
                self._hedge_order_id = oid  # separate slot from bid/ask
                return oid
            else:
                log.warning("Hedge place rejected: %s", result.get('retMsg', '?'))
        except Exception:
            log.exception("Hedge place failed")
        return None

    def _cancel_order_by_id(self, oid: str) -> None:
        """Cancel a specific order by ID."""
        try:
            self._adapter._client.post('/v5/order/cancel', body={
                'category': 'linear', 'symbol': self._symbol, 'orderId': oid,
            })
        except Exception as e:
            log.warning("Cancel order %s failed: %s", oid, e)

    def _cancel_side(self, side: str) -> None:
        """Cancel one side's order by orderId."""
        oid = self._bid_order_id if side == "Buy" else self._ask_order_id
        if not oid:
            return
        try:
            self._adapter._client.post('/v5/order/cancel', body={
                'category': 'linear',
                'symbol': self._symbol,
                'orderId': oid,
            })
        except Exception as e:
            log.warning("Cancel %s order %s failed: %s", side, oid, e)
        if side == "Buy":
            self._bid_order_id = None
        else:
            self._ask_order_id = None

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

    def _force_hedge(self, inv: float, target_inv: float, now: float) -> None:
        """Taker-hedge inventory towards target level.

        FIX 1: Forces inventory reduction when it stays too high too long.
        Rate-limited to max 1 hedge per 10 seconds.
        """
        if now - self._last_taker_hedge_time < 10.0:
            return  # rate limit
        if self._cfg.dry_run:
            return

        abs_inv = abs(inv)
        step = self._cfg.qty_step
        hedge_qty = int((abs_inv - target_inv) / step) * step
        hedge_qty = round(hedge_qty, 3)
        if hedge_qty < step:
            return

        side = "Sell" if inv > 0 else "Buy"
        try:
            result = self._adapter._client.post('/v5/order/create', body={
                'category': 'linear',
                'symbol': self._symbol,
                'side': side,
                'orderType': 'Market',
                'qty': str(hedge_qty),
            })
            ret_code = result.get('retCode', -1)
            if ret_code == 0:
                self._last_taker_hedge_time = now
                if now - self._last_hedge_log > 30:
                    log.warning(
                        "INV_HEDGE %s %.3f inv=%.1f→%.1f target=%.1f",
                        side, hedge_qty, abs_inv,
                        abs_inv - hedge_qty, target_inv,
                    )
                    self._last_hedge_log = now
            else:
                log.debug("Hedge rejected: %s", result.get('retMsg', '?'))
        except Exception:
            log.exception("Hedge order failed")

    def _flatten(self) -> bool:
        """Close any open position at market.

        E2 fix: query exchange for real position instead of trusting inventory tracker.
        Returns ``True`` when the account is already flat or the exchange accepts
        the reduce-only flatten order.
        """
        # E2 fix: get actual exchange position
        try:
            resp = self._adapter._client.get('/v5/position/list', params={
                'category': 'linear', 'symbol': self._symbol,
            })
            pos_list = resp.get('result', {}).get('list', [])
            exch_qty = 0.0
            exch_side = ""
            for p in pos_list:
                size = float(p.get('size', 0))
                if size > 0:
                    exch_side = p.get('side', '')
                    exch_qty = size
                    break
            if exch_qty < 1e-10:
                log.info("Flatten: exchange position already flat")
                return True
            side = "Sell" if exch_side == "Buy" else "Buy"
            qty = exch_qty
        except Exception:
            # Fallback to inventory
            if abs(self._engine.net_qty) < 1e-10:
                return True
            side = "Sell" if self._engine.net_qty > 0 else "Buy"
            qty = abs(self._engine.net_qty)
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
        """v10: Simple fill detection. Just track position changes, no pingpong."""
        try:
            resp = self._adapter._client.get('/v5/position/list', params={
                'category': 'linear', 'symbol': self._symbol,
            })
            pos_list = resp.get('result', {}).get('list', [])

            exch_qty = 0.0
            exch_price = self._engine.mid
            for p in pos_list:
                size = float(p.get('size', 0))
                if size > 0:
                    es = p.get('side', '')
                    exch_qty = size if es == 'Buy' else -size
                    exch_price = float(p.get('avgPrice', self._engine.mid))
                    break

            prev = self._last_exchange_qty
            if abs(exch_qty - prev) < 1e-6:
                return

            delta = exch_qty - prev
            fill_side = "buy" if delta > 0 else "sell"
            fill_qty = abs(delta)
            self._last_exchange_qty = exch_qty

            fill_price = self._engine.mid if self._engine.mid > 0 else exch_price
            is_buy = fill_side == "buy"
            rpnl = self._engine.on_fill(is_buy, fill_qty, fill_price)
            self._metrics.record_fill(fill_side, fill_qty, fill_price, rpnl)

            log.info("FILL %s %.2f @ %.2f rpnl=%.4f inv=%.2f exch=%.2f [v10]",
                     fill_side, fill_qty, fill_price, rpnl,
                     self._engine.net_qty, exch_qty)
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

    def _poll_ml_signal(self):
        """Fetch ML alpha signal from latest kline prediction.

        Uses a simple momentum proxy from recent klines as ML signal.
        In production, this would read from the alpha runner's signal store.
        """
        now = time.monotonic()
        if now - self._ml_signal_ts < 60:  # poll every 60s
            return

        try:
            # Fetch recent klines for simple trend signal
            resp = self._adapter._client.get('/v5/market/kline', params={
                'category': 'linear', 'symbol': self._symbol,
                'interval': '60', 'limit': '25',
            })
            klines = resp.get('result', {}).get('list', [])
            if len(klines) >= 20:
                # Klines are newest-first on Bybit
                closes = [float(k[4]) for k in reversed(klines)]
                # MA(5) vs MA(20) trend
                ma5 = sum(closes[-5:]) / 5
                ma20 = sum(closes[-20:]) / 20
                ret_12h = (closes[-1] - closes[-12]) / closes[-12] if closes[-12] > 0 else 0

                old_signal = self._ml_signal
                if ma5 > ma20 * 1.002 and ret_12h > 0.005:
                    self._ml_signal = 1
                elif ma5 < ma20 * 0.998 and ret_12h < -0.005:
                    self._ml_signal = -1
                else:
                    self._ml_signal = 0

                if self._ml_signal != old_signal:
                    log.info("ML_SIGNAL changed: %+d → %+d (ma5=%.1f ma20=%.1f ret12h=%.3f%%)",
                             old_signal, self._ml_signal, ma5, ma20, ret_12h * 100)

            self._ml_signal_ts = now
        except Exception as e:
            log.debug("ML signal poll failed (keeping previous): %s", e)

    def run(self):
        """Main blocking loop."""
        self.start()
        log.info("Market maker v4 running: %s leverage=%dx size=%.1f dry_run=%s",
                 self._symbol, self._leverage, self._cfg.order_size_eth, self._cfg.dry_run)
        try:
            while self._running:
                self._check_market_data_watchdog()
                self._check_fills()
                self._poll_ml_signal()
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
        if not flattened and abs(self._engine.net_qty) >= 1e-10:
            log.warning("Stop path flatten failed; position may remain open")
        if hasattr(self, '_ws'):
            self._ws.close()
        self._metrics.log_summary(self._engine.net_qty, 0.0)
        log.info("Market maker stopped. Fills=%d PnL=%.4f",
                 self._engine.total_fills, self._engine.daily_pnl)


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
    try:
        kill_latch = _build_runtime_kill_latch(adapter, args.symbol, dry_run=args.dry_run)
        if kill_latch is not None:
            require_kill_latch_clear(kill_latch, runtime_name="bybit-mm.service")
    except RuntimeError as exc:
        log.critical("Persistent runtime kill latch armed: %s", exc)
        raise SystemExit(STARTUP_GUARD_EXIT_CODE) from exc
    try:
        lease = _claim_runtime_symbol(adapter, args.symbol, dry_run=args.dry_run)
    except RuntimeError as exc:
        log.critical("Runtime symbol ownership conflict: %s", exc)
        raise SystemExit(STARTUP_GUARD_EXIT_CODE) from exc

    mm_cfg = MarketMakerConfig(
        symbol=args.symbol,
        order_size_eth=args.order_size,
        max_inventory_notional=args.max_inventory,
        daily_loss_limit=args.loss_limit,
        gamma=args.gamma,
        min_spread_bps=args.min_spread,
        max_spread_bps=args.max_spread,
        dry_run=args.dry_run,
        qty_step=_SYMBOL_SPECS.get(args.symbol, {}).get('qty_step', 0.01),
        tick_size=_SYMBOL_SPECS.get(args.symbol, {}).get('tick_size', 0.01),
        quote_update_interval_s=0.5,
        stale_order_s=3.0,
        circuit_breaker_losses=15,
        circuit_breaker_pause_s=30.0,
    )

    runner = BybitMMRunner(
        args.symbol,
        mm_cfg,
        adapter,
        leverage=args.leverage,
        kill_latch=kill_latch,
    )
    runner._quote_mode = args.mode

    def _signal_handler(signum, frame):
        log.info("Signal %d, shutting down...", signum)
        runner.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        runner.run()
    finally:
        if lease is not None:
            lease.release()


if __name__ == "__main__":
    main()
