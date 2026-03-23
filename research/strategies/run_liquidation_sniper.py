"""Liquidation Sniper — detect cascade events and trade the momentum.

Monitors Binance forceOrder WS stream for large liquidation clusters.
When total liquidations exceed threshold in a rolling window:
  → Open position in cascade direction (taker, fast)
  → Hold 5-60 seconds
  → Close with trailing stop or timeout

Usage:
    python3 -m scripts.run_liquidation_sniper --symbol BTCUSDT --leverage 50
    python3 -m scripts.run_liquidation_sniper --symbol ETHUSDT --leverage 30 --dry-run
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

log = logging.getLogger("liq_sniper")


class LiquidationSniper:
    """Detect liquidation cascades and trade the momentum."""

    def __init__(
        self,
        symbol: str,
        adapter,
        leverage: int = 50,
        # Cascade detection
        cascade_window_s: float = 10.0,     # rolling window to detect cascade
        cascade_threshold_usd: float = 500_000,  # $500K in window = cascade
        # Position management
        position_size_usd: float = 5000,    # notional per trade
        hold_timeout_s: float = 30.0,       # max hold time
        trail_stop_pct: float = 0.003,      # 0.3% trailing stop
        hard_stop_pct: float = 0.005,       # 0.5% hard stop
        # Risk
        max_trades_per_hour: int = 20,
        daily_loss_limit: float = 100.0,    # $100 daily loss limit
        dry_run: bool = False,
    ):
        self._symbol = symbol
        self._adapter = adapter
        self._leverage = leverage
        self._cascade_window = cascade_window_s
        self._cascade_threshold = cascade_threshold_usd
        self._position_size_usd = position_size_usd
        self._hold_timeout = hold_timeout_s
        self._trail_stop_pct = trail_stop_pct
        self._hard_stop_pct = hard_stop_pct
        self._max_trades_hour = max_trades_per_hour
        self._daily_loss_limit = daily_loss_limit
        self._dry_run = dry_run

        # State
        self._running = False
        self._recent_liqs: list[dict] = []  # (ts, side, notional)
        self._position_side: int = 0  # 0=flat, 1=long, -1=short
        self._entry_price: float = 0.0
        self._peak_price: float = 0.0
        self._entry_time: float = 0.0
        self._current_price: float = 0.0
        self._daily_pnl: float = 0.0
        self._trades_this_hour: int = 0
        self._hour_start: float = 0.0
        self._total_trades: int = 0
        self._total_pnl: float = 0.0
        self._wins: int = 0
        self._losses: int = 0

    def start(self):
        self._running = True
        self._hour_start = time.time()

        # Set leverage
        if not self._dry_run:
            try:
                result = self._adapter._client.post('/v5/position/set-leverage', body={
                    'category': 'linear', 'symbol': self._symbol,
                    'buyLeverage': str(self._leverage),
                    'sellLeverage': str(self._leverage),
                })
                if isinstance(result, dict) and result.get("retCode", 0) != 0:
                    log.warning("set_leverage API failed: %s", result.get("retMsg", "unknown"))
            except Exception as e:
                log.warning("set_leverage exception: %s", e)

        # Start forceOrder WS
        self._start_liq_ws()
        # Start price WS
        self._start_price_ws()

        log.info("Sniper started: %s %dx dry=%s threshold=$%d",
                 self._symbol, self._leverage, self._dry_run, self._cascade_threshold)

    def _start_liq_ws(self):
        def on_message(ws, msg):
            try:
                data = json.loads(msg)
                if 'e' in data and data['e'] == 'forceOrder':
                    self._on_liquidation(data['o'])
                elif 'data' in data:
                    # Array format
                    for item in (data['data'] if isinstance(data['data'], list) else [data['data']]):
                        if 'o' in item:
                            self._on_liquidation(item['o'])
            except Exception:
                pass

        def on_open(ws):
            log.info("ForceOrder WS connected")

        self._liq_ws = websocket.WebSocketApp(
            "wss://fstream.binance.com/ws/!forceOrder@arr",
            on_message=on_message,
            on_open=on_open,
            on_error=lambda ws, e: log.error("Liq WS error: %s", e),
            on_close=lambda ws, c, m: self._reconnect_liq() if self._running else None,
        )
        threading.Thread(target=self._liq_ws.run_forever, daemon=True, name="liq-ws").start()

    def _reconnect_liq(self):
        if self._running:
            time.sleep(3)
            self._start_liq_ws()

    def _reconnect_price(self):
        if self._running:
            time.sleep(3)
            self._start_price_ws()

    def _start_price_ws(self):
        sym = self._symbol.lower()
        def on_message(ws, msg):
            try:
                data = json.loads(msg)
                d = data.get('data', data)
                if 'p' in d:
                    self._current_price = float(d['p'])
                elif isinstance(d, list):
                    for t in d:
                        if 'p' in t:
                            self._current_price = float(t['p'])
            except Exception:
                pass

        self._price_ws = websocket.WebSocketApp(
            f"wss://fstream.binance.com/ws/{sym}@aggTrade",
            on_message=on_message,
            on_error=lambda ws, e: log.warning("Price WS error: %s", e),
            on_close=lambda ws, c, m: self._reconnect_price() if self._running else None,
        )
        threading.Thread(target=self._price_ws.run_forever, daemon=True, name="price-ws").start()

    def _on_liquidation(self, order: dict):
        """Process a single liquidation event."""
        symbol = order.get('s', '')
        if symbol != self._symbol:
            return

        side = order.get('S', '')  # BUY = short liquidated (bullish), SELL = long liquidated (bearish)
        price = float(order.get('p', 0))
        qty = float(order.get('q', 0))
        notional = price * qty

        now = time.time()
        self._recent_liqs.append({
            'ts': now,
            'side': side,
            'notional': notional,
        })

        # Prune old liquidations outside window
        cutoff = now - self._cascade_window
        self._recent_liqs = [lq for lq in self._recent_liqs if lq["ts"] > cutoff]

        # Check for cascade
        total_notional = sum(lq["notional"] for lq in self._recent_liqs)
        buy_notional = sum(lq["notional"] for lq in self._recent_liqs if lq["side"] == 'BUY')
        sell_notional = sum(lq["notional"] for lq in self._recent_liqs if lq["side"] == 'SELL')

        if total_notional >= self._cascade_threshold and self._position_side == 0:
            # Determine cascade direction
            if sell_notional > buy_notional * 1.5:
                # Longs being liquidated → price dropping → short
                self._enter_position(-1, price, total_notional, sell_notional)
            elif buy_notional > sell_notional * 1.5:
                # Shorts being liquidated → price rising → long
                self._enter_position(1, price, total_notional, buy_notional)

    def _enter_position(self, direction: int, price: float, total_liq: float, directional_liq: float):
        """Enter a position following the cascade."""
        now = time.time()

        # Rate limit
        if now - self._hour_start > 3600:
            self._trades_this_hour = 0
            self._hour_start = now
        if self._trades_this_hour >= self._max_trades_hour:
            return

        # Daily loss limit
        if self._daily_pnl <= -self._daily_loss_limit:
            return

        qty = self._position_size_usd / max(price, 1)
        # Round to exchange step
        qty = round(qty, 3)
        if qty < 0.001:
            return

        side = "Buy" if direction == 1 else "Sell"

        if self._dry_run:
            log.info("DRY ENTER %s %.3f @ $%.0f (cascade $%.0fK, dir_liq $%.0fK)",
                     side, qty, price, total_liq/1000, directional_liq/1000)
        else:
            try:
                result = self._adapter._client.post('/v5/order/create', body={
                    'category': 'linear', 'symbol': self._symbol,
                    'side': side, 'orderType': 'Market', 'qty': str(qty),
                })
                if result.get('retCode', -1) != 0:
                    log.warning("Entry failed: %s", result.get('retMsg', '?'))
                    return
            except Exception:
                log.exception("Entry failed")
                return

        self._position_side = direction
        self._entry_price = price
        self._peak_price = price
        self._entry_time = now
        self._trades_this_hour += 1
        self._total_trades += 1

        log.info("ENTER %s %.3f @ $%.0f (cascade $%.0fK in %.0fs)",
                 side, qty, price, total_liq/1000, self._cascade_window)

    def _check_exit(self):
        """Check trailing stop, hard stop, and timeout."""
        if self._position_side == 0:
            return

        price = self._current_price
        if price <= 0:
            return

        now = time.time()
        elapsed = now - self._entry_time
        direction = self._position_side

        # Update peak
        if direction == 1:
            self._peak_price = max(self._peak_price, price)
        else:
            self._peak_price = min(self._peak_price, price)

        # PnL
        if direction == 1:
            pnl_pct = (price - self._entry_price) / self._entry_price
        else:
            pnl_pct = (self._entry_price - price) / self._entry_price

        # Hard stop
        if pnl_pct <= -self._hard_stop_pct:
            self._exit_position("hard_stop", price, pnl_pct)
            return

        # Trailing stop (only after some profit)
        if pnl_pct > self._trail_stop_pct:
            if direction == 1:
                trail_from_peak = (self._peak_price - price) / self._peak_price
            else:
                trail_from_peak = (price - self._peak_price) / self._peak_price
            if trail_from_peak > self._trail_stop_pct:
                self._exit_position("trail_stop", price, pnl_pct)
                return

        # Timeout
        if elapsed > self._hold_timeout:
            self._exit_position("timeout", price, pnl_pct)

    def _exit_position(self, reason: str, price: float, pnl_pct: float):
        """Close the position."""
        direction = self._position_side
        side = "Sell" if direction == 1 else "Buy"
        qty = self._position_size_usd / max(self._entry_price, 1)
        qty = round(qty, 3)

        pnl_usd = pnl_pct * self._position_size_usd

        if not self._dry_run:
            try:
                result = self._adapter._client.post('/v5/order/create', body={
                    'category': 'linear', 'symbol': self._symbol,
                    'side': side, 'orderType': 'Market',
                    'qty': str(qty), 'reduceOnly': True,
                })
                if isinstance(result, dict) and result.get("retCode", 0) != 0:
                    log.warning("Exit order API failed: %s", result.get("retMsg", "unknown"))
            except Exception:
                log.exception("Exit failed")

        self._daily_pnl += pnl_usd
        self._total_pnl += pnl_usd
        if pnl_usd > 0:
            self._wins += 1
        else:
            self._losses += 1

        elapsed = time.time() - self._entry_time

        log.info("EXIT %s %.3f @ $%.0f pnl=$%.2f (%.2f%%) reason=%s hold=%.1fs | total=$%.2f W=%d L=%d",
                 side, qty, price, pnl_usd, pnl_pct*100, reason, elapsed,
                 self._total_pnl, self._wins, self._losses)

        self._position_side = 0
        self._entry_price = 0
        self._peak_price = 0

    def run(self):
        self.start()
        try:
            while self._running:
                self._check_exit()
                time.sleep(0.1)  # 100ms check interval
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self._running = False
        if self._position_side != 0:
            self._exit_position("shutdown", self._current_price,
                                 (self._current_price - self._entry_price) / max(self._entry_price, 1)
                                 * self._position_side)
        if hasattr(self, '_liq_ws'):
            self._liq_ws.close()
        if hasattr(self, '_price_ws'):
            self._price_ws.close()
        log.info("Sniper stopped. Trades=%d PnL=$%.2f W=%d L=%d",
                 self._total_trades, self._total_pnl, self._wins, self._losses)


def main():
    parser = argparse.ArgumentParser(description="Liquidation Sniper")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--leverage", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=500000, help="Cascade threshold USD")
    parser.add_argument("--window", type=float, default=10, help="Cascade detection window seconds")
    parser.add_argument("--hold-timeout", type=float, default=30, help="Max hold seconds")
    parser.add_argument("--position-size", type=float, default=5000, help="Position notional USD")
    parser.add_argument("--loss-limit", type=float, default=100, help="Daily loss limit USD")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(level=level,
                        format="%(asctime)s %(levelname)-5s %(name)s  %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler("logs/liq_sniper.log")])

    # Load Bybit adapter (or Binance for execution)
    from execution.adapters.bybit.config import BybitConfig
    from execution.adapters.bybit.adapter import BybitAdapter
    bybit_cfg = BybitConfig(
        api_key=os.environ['BYBIT_API_KEY'],
        api_secret=os.environ['BYBIT_API_SECRET'],
        base_url=os.environ.get('BYBIT_BASE_URL', 'https://api-demo.bybit.com'),
    )
    adapter = BybitAdapter(bybit_cfg)

    sniper = LiquidationSniper(
        symbol=args.symbol,
        adapter=adapter,
        leverage=args.leverage,
        cascade_threshold_usd=args.threshold,
        cascade_window_s=args.window,
        hold_timeout_s=args.hold_timeout,
        position_size_usd=args.position_size,
        daily_loss_limit=args.loss_limit,
        dry_run=args.dry_run,
    )

    def _sig(signum, frame):
        sniper.stop()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    sniper.run()


if __name__ == "__main__":
    main()
