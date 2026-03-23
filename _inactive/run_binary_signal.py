"""Binary 5m Up/Down signal trader on Bybit demo.

Replicates the Polymarket RSI strategy using BTC perpetual futures:
- Every 5 minutes: compute RSI(5) from recent 1m closes
- RSI < 30 → Buy (expect bounce), RSI > 70 → Sell (expect drop)
- Hold exactly 5 minutes, then close
- Tracks win/loss as binary outcome (Up or Down)

Validated: 27/27 months positive, 53-55% win rate, $70/day@$10/bet

Usage:
    python3 -m scripts.run_binary_signal --symbol BTCUSDT --leverage 20 --bet-size 5000
    python3 -m scripts.run_binary_signal --symbol BTCUSDT --dry-run
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
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger("binary_signal")


class BinaryEngine:
    """5-minute binary signal engine using RSI(5)."""

    def __init__(self, symbol: str, rsi_period: int = 5,
                 rsi_low: float = 30, rsi_high: float = 70):
        self.symbol = symbol
        self.lock = threading.Lock()
        self.rsi_period = rsi_period
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high

        # Price state
        self.mid = 0.0
        self.best_bid = 0.0
        self.best_ask = 0.0

        # 1m bar accumulator
        self.bar_open = 0.0
        self.bar_start = 0.0
        self.closes_1m: deque[float] = deque(maxlen=120)  # 2h of 1m bars

        # RSI state (EMA-based Wilder's RSI)
        self.avg_gain = 0.0
        self.avg_loss = 0.0
        self.rsi: float = 50.0
        self.rsi_ready = False
        self.rsi_count = 0

        # Position
        self.pos_side = 0       # 0=flat, +1=long, -1=short
        self.pos_qty = 0.0
        self.entry_price = 0.0
        self.entry_time = 0.0

        # Stats
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.trades = 0
        self.streak = 0

    def push_trade(self, price: float, qty: float, is_buy: bool):
        """Process trade tick: accumulate into 1m bars."""
        with self.lock:
            now = time.time()
            self.mid = price
            if is_buy:
                self.best_ask = price
            else:
                self.best_bid = price

            if self.bar_start == 0:
                self.bar_start = now
                self.bar_open = price

            # Close 1m bar
            if now - self.bar_start >= 60:
                self._close_1m_bar(price, now)

    def push_depth(self, bid: float, ask: float):
        """Update BBO."""
        with self.lock:
            self.best_bid = bid
            self.best_ask = ask
            self.mid = (bid + ask) / 2

    def _close_1m_bar(self, close: float, now: float):
        """Close a 1-minute bar and update RSI."""
        self.closes_1m.append(close)

        if len(self.closes_1m) >= 2:
            change = self.closes_1m[-1] - self.closes_1m[-2]
            gain = max(change, 0)
            loss = max(-change, 0)

            self.rsi_count += 1
            if self.rsi_count <= self.rsi_period:
                # Initial SMA period
                self.avg_gain += gain / self.rsi_period
                self.avg_loss += loss / self.rsi_period
                if self.rsi_count == self.rsi_period:
                    self.rsi_ready = True
                    if self.avg_loss > 0:
                        rs = self.avg_gain / self.avg_loss
                        self.rsi = 100 - 100 / (1 + rs)
                    else:
                        self.rsi = 100 if self.avg_gain > 0 else 50
            else:
                # EMA smoothing (Wilder's method)
                p = self.rsi_period
                self.avg_gain = (self.avg_gain * (p - 1) + gain) / p
                self.avg_loss = (self.avg_loss * (p - 1) + loss) / p
                if self.avg_loss > 0:
                    rs = self.avg_gain / self.avg_loss
                    self.rsi = 100 - 100 / (1 + rs)
                else:
                    self.rsi = 100 if self.avg_gain > 0 else 50

        # Reset bar
        self.bar_open = close
        self.bar_start = now

    def get_signal(self) -> int:
        """Return signal based on RSI: +1=buy, -1=sell, 0=no trade."""
        if not self.rsi_ready:
            return 0
        if self.rsi < self.rsi_low:
            return 1   # oversold → buy
        if self.rsi > self.rsi_high:
            return -1  # overbought → sell
        return 0


class BinaryTrader:
    """Trades 5-minute binary signals on Bybit."""

    def __init__(self, symbol: str, adapter, leverage: int = 20,
                 bet_size_usd: float = 5000, dry_run: bool = False,
                 rsi_low: float = 30, rsi_high: float = 70,
                 hold_seconds: int = 300):
        self._symbol = symbol
        self._adapter = adapter
        self._leverage = leverage
        self._bet_size = bet_size_usd
        self._dry_run = dry_run
        self._hold_seconds = hold_seconds
        self._running = False
        self._engine = BinaryEngine(symbol, rsi_low=rsi_low, rsi_high=rsi_high)
        self._ws_thread: threading.Thread | None = None
        self._daily_pnl = 0.0
        self._day_start = 0.0
        self._last_check = 0.0

    def start(self):
        self._running = True
        self._day_start = time.time()

        # Set leverage
        try:
            result = self._adapter._client.post('/v5/position/set-leverage', body={
                'category': 'linear', 'symbol': self._symbol,
                'buyLeverage': str(self._leverage),
                'sellLeverage': str(self._leverage),
            })
            if isinstance(result, dict) and result.get("retCode", 0) != 0:
                log.warning("set_leverage API failed: %s", result.get("retMsg", "unknown"))
        except Exception as e:
            log.warning("set_leverage: %s", e)

        self._start_ws()
        log.info("Binary trader started: %s %dx bet=$%.0f hold=%ds dry=%s RSI(%d/%d)",
                 self._symbol, self._leverage, self._bet_size, self._hold_seconds,
                 self._dry_run, self._engine.rsi_low, self._engine.rsi_high)

    def _start_ws(self):
        """Start Bybit WS for depth + trades."""
        import websocket as ws_lib
        subs = [f"orderbook.1.{self._symbol}", f"publicTrade.{self._symbol}"]

        def on_msg(ws, msg):
            try:
                data = json.loads(msg)
                topic = data.get("topic", "")
                if not topic:
                    return
                d = data.get("data", {})

                if "orderbook" in topic:
                    bids = d.get("b", [])
                    asks = d.get("a", [])
                    if bids and asks:
                        self._engine.push_depth(float(bids[0][0]), float(asks[0][0]))

                elif "publicTrade" in topic:
                    for t in (d if isinstance(d, list) else [d]):
                        p = float(t.get("p", 0))
                        q = float(t.get("v", 0))
                        buy = t.get("S") == "Buy"
                        if p > 0:
                            self._engine.push_trade(p, q, buy)
            except Exception:
                pass

        def on_open(ws):
            ws.send(json.dumps({"op": "subscribe", "args": subs}))
            log.info("WS subscribed: %s", subs)

        def on_error(ws, err):
            log.warning("WS error: %s", err)

        def on_close(ws, code, msg):
            log.warning("WS closed: %s %s", code, msg)
            if self._running:
                time.sleep(5)
                self._start_ws()

        url = "wss://stream.bybit.com/v5/public/linear"
        ws = ws_lib.WebSocketApp(url, on_message=on_msg, on_open=on_open,
                                 on_error=on_error, on_close=on_close)
        t = threading.Thread(target=ws.run_forever, daemon=True)
        t.start()
        self._ws_thread = t

    def run(self):
        """Main loop: check signal every 5 minutes."""
        self.start()

        while self._running:
            try:
                time.sleep(1)
                self._poll()
            except KeyboardInterrupt:
                break
            except Exception:
                log.exception("Poll error")

        self.stop()

    def _poll(self):
        now = time.time()

        # Daily reset
        if now - self._day_start > 86400:
            log.info("DAILY_RESET pnl=$%.2f W=%d L=%d",
                     self._daily_pnl, self._engine.wins, self._engine.losses)
            self._daily_pnl = 0.0
            self._day_start = now

        # Check exit (hold timeout)
        eng = self._engine
        if eng.pos_side != 0:
            elapsed = now - eng.entry_time
            if elapsed >= self._hold_seconds:
                self._close_position("TIMEOUT")
            return  # Don't enter while holding

        # Check signal every 5 seconds (quick check)
        if now - self._last_check < 5:
            return
        self._last_check = now

        # Log status every 60 seconds
        if int(now) % 60 < 6:
            self._log_status()

        # Enter anytime signal is valid (RSI extreme)
        # No 5-minute boundary restriction — trade whenever RSI crosses threshold

        signal = eng.get_signal()
        if signal != 0:
            log.info("SIGNAL %s RSI=%.1f → %s", self._symbol, eng.rsi,
                     "BUY" if signal == 1 else "SELL")
        if signal == 0:
            return

        self._enter(signal)

    def _enter(self, direction: int):
        eng = self._engine
        if eng.pos_side != 0:
            log.debug("Skip entry: already in position %+d", eng.pos_side)
            return

        price = eng.best_bid if direction == 1 else eng.best_ask
        if price <= 0:
            price = eng.mid  # fallback to mid
        if price <= 0:
            log.warning("Skip entry: price=%.2f (bid=%.2f ask=%.2f mid=%.2f)",
                        price, eng.best_bid, eng.best_ask, eng.mid)
            return
        log.info("ENTER_ATTEMPT %s dir=%+d price=%.2f bid=%.2f ask=%.2f mid=%.2f",
                 eng.symbol, direction, price, eng.best_bid, eng.best_ask, eng.mid)

        qty = self._bet_size / price
        # Round to step (BTC=0.001, ETH=0.01, SOL=0.1)
        step = 0.001  # BTC step
        qty = round(qty / step) * step
        qty = round(qty, 3)  # Fix floating point noise
        if qty <= 0:
            return

        side = "Buy" if direction == 1 else "Sell"
        qty_str = f"{qty:.3f}"

        if not self._dry_run:
            try:
                result = self._adapter._client.post('/v5/order/create', body={
                    'category': 'linear', 'symbol': self._symbol,
                    'side': side, 'orderType': 'Market',
                    'qty': qty_str,
                })
                log.info("ORDER_RESULT %s: retCode=%s msg=%s qty=%s", self._symbol,
                         result.get('retCode'), result.get('retMsg', ''), qty_str)
                if result.get('retCode', -1) != 0:
                    log.warning("Entry rejected: %s (qty=%s)", result.get('retMsg'), qty_str)
                    self._last_check = time.time() + 55  # cooldown 60s after rejection
                    return
            except Exception:
                log.exception("Entry failed")
                return

        eng.pos_side = direction
        eng.pos_qty = qty
        eng.entry_price = price
        eng.entry_time = time.time()
        eng.trades += 1

        log.info("ENTER %s %s %.4f @ $%.2f RSI=%.1f (trade #%d)",
                 self._symbol, side, qty, price, eng.rsi, eng.trades)

    def _close_position(self, reason: str):
        eng = self._engine
        if eng.pos_side == 0:
            return

        price = eng.mid
        pnl = (price - eng.entry_price) * eng.pos_side * eng.pos_qty
        pnl_pct = ((price - eng.entry_price) / eng.entry_price) * eng.pos_side
        is_up = price > eng.entry_price

        side = "Sell" if eng.pos_side == 1 else "Buy"

        if not self._dry_run:
            try:
                result = self._adapter._client.post('/v5/order/create', body={
                    'category': 'linear', 'symbol': self._symbol,
                    'side': side, 'orderType': 'Market',
                    'qty': f"{eng.pos_qty:.3f}", 'reduceOnly': True,
                })
                if isinstance(result, dict) and result.get("retCode", 0) != 0:
                    log.warning("Close order API failed: %s", result.get("retMsg", "unknown"))
            except Exception:
                log.exception("Close failed")

        self._daily_pnl += pnl
        result = "UP" if is_up else "DOWN"
        predicted = "UP" if eng.pos_side == 1 else "DOWN"
        correct = result == predicted

        if correct:
            eng.wins += 1
            eng.streak = max(eng.streak + 1, 1)
        else:
            eng.losses += 1
            eng.streak = min(eng.streak - 1, -1)
        eng.total_pnl += pnl

        wr = eng.wins / eng.trades if eng.trades > 0 else 0

        log.info("EXIT %s %s $%.2f (%.2f%%) [%s] predict=%s actual=%s %s | "
                 "WR=%.1f%% W=%d L=%d streak=%d daily=$%.2f total=$%.2f",
                 self._symbol, side, pnl, pnl_pct * 100, reason,
                 predicted, result, "WIN" if correct else "LOSS",
                 wr * 100, eng.wins, eng.losses, eng.streak,
                 self._daily_pnl, eng.total_pnl)

        eng.pos_side = 0
        eng.pos_qty = 0

    def _log_status(self):
        eng = self._engine
        wr = eng.wins / eng.trades * 100 if eng.trades > 0 else 0
        log.info("STATUS %s mid=$%.2f RSI=%.1f ready=%s rsi_n=%d pos=%+d bars=%d WR=%.1f%% W=%d L=%d pnl=$%.2f",
                 self._symbol, eng.mid, eng.rsi, eng.rsi_ready, eng.rsi_count,
                 eng.pos_side, len(eng.closes_1m), wr, eng.wins, eng.losses, self._daily_pnl)

    def stop(self):
        self._running = False
        eng = self._engine
        if eng.pos_side != 0:
            self._close_position("SHUTDOWN")
        wr = eng.wins / eng.trades * 100 if eng.trades > 0 else 0
        log.info("Binary trader stopped. WR=%.1f%% W=%d L=%d total=$%.2f",
                 wr, eng.wins, eng.losses, eng.total_pnl)


def main():
    parser = argparse.ArgumentParser(description="Binary 5m Up/Down signal trader")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--leverage", type=int, default=20)
    parser.add_argument("--bet-size", type=float, default=5000,
                        help="Position size per trade in USD")
    parser.add_argument("--hold", type=int, default=300,
                        help="Hold duration in seconds (default: 300 = 5min)")
    parser.add_argument("--rsi-low", type=float, default=30)
    parser.add_argument("--rsi-high", type=float, default=70)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)-15s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    from execution.adapters.bybit.client import BybitRestClient
    from execution.adapters.bybit.config import BybitConfig

    cfg = BybitConfig(
        api_key=os.environ.get('BYBIT_API_KEY', ''),
        api_secret=os.environ.get('BYBIT_API_SECRET', ''),
        base_url=os.environ.get('BYBIT_BASE_URL', 'https://api-demo.bybit.com'),
    )
    client = BybitRestClient(cfg)

    class Adapter:
        _client = client

    trader = BinaryTrader(
        symbol=args.symbol,
        adapter=Adapter(),
        leverage=args.leverage,
        bet_size_usd=args.bet_size,
        dry_run=args.dry_run,
        rsi_low=args.rsi_low,
        rsi_high=args.rsi_high,
        hold_seconds=args.hold,
    )

    def handle_signal(sig, frame):
        trader._running = False

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    trader.run()


if __name__ == "__main__":
    main()
