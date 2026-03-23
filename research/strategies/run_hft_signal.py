"""Signal-driven HFT v2 — 3-layer limit trading on BTC/ETH/SOL.

Layer A: Funding direction (8h cycle, limit entry, earn funding+direction)
Layer B: 5min momentum/VPIN (mid-frequency, limit entry/exit)
Layer C: Liquidation prediction (event-driven, pre-positioned limits)

All limit execution. No taker fees.

Usage:
    python3 -m scripts.run_hft_signal --symbols BTCUSDT ETHUSDT SOLUSDT --leverage 20
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

log = logging.getLogger("hft_signal")

SPECS = {
    'BTCUSDT': {'tick': 0.10, 'step': 0.001, 'min_not': 100},
    'ETHUSDT': {'tick': 0.01, 'step': 0.01, 'min_not': 20},
    'SOLUSDT': {'tick': 0.01, 'step': 0.1, 'min_not': 5},
}


class SymbolEngine:
    """Per-symbol state + signal computation. Thread-safe via lock."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.lock = threading.Lock()  # C1 fix: protect all mutable state
        self.mid = 0.0
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.funding_rate = 0.0

        # Position (C2 fix: separate pending vs confirmed)
        self.pos_side = 0       # 0=flat, +1/-1 = CONFIRMED position
        self.pos_qty = 0.0
        self.entry_price = 0.0
        self.entry_time = 0.0
        self.entry_reason = ""
        self.order_id: str | None = None
        self.pending = False    # C2: True while limit order unconfirmed

        # 5min bar accumulator
        self.bar_open = 0.0
        self.bar_high = 0.0
        self.bar_low = 999999.0
        self.bar_volume = 0.0
        self.bar_buy_vol = 0.0
        self.bar_start = 0.0
        self.bars: deque[dict] = deque(maxlen=60)  # 5h of 5min bars

        # OB state
        self.bid_depth = 0.0
        self.ask_depth = 0.0
        self.ob_imbalance = 0.0

        # Liquidation accumulator
        self.liq_window: list[dict] = []

        # Improvement 1+2: on-chain + DVOL signals (fed externally)
        self.onchain_z: float = 0.0      # TxTfrCnt z-score
        self.dvol_z: float = 0.0         # DVOL z-score 168h
        # Improvement 4: trend filter
        self.ma50: float = 0.0           # 50-bar MA from bars
        self.close_history: deque[float] = deque(maxlen=60)

        # Mean reversion state
        self.mr_recent_trades: deque = deque(maxlen=100)  # (timestamp, price, qty, is_buy)
        self.mr_signal = 0          # +1/-1/0 mean reversion signal
        self.mr_signal_time = 0.0   # when signal was generated
        self.mr_impact_price = 0.0  # price at impact detection
        self.mr_avg_trade_size = 0.0  # EMA of trade sizes

        # BTC lead-lag signal (set by SignalHFT orchestrator, not by SymbolEngine)
        self.btc_lead = 0  # +1/-1/0: BTC just moved, ALTs should follow

        # Stats
        self.trades = 0
        self.pnl = 0.0
        self.wins = 0
        self.losses = 0

    def push_trade(self, price: float, qty: float, is_buy: bool):
        """Accumulate trades into 5min bars + mean reversion detection. Thread-safe."""
        with self.lock:  # C1 fix
            now = time.time()
            if self.bar_start == 0:
                self.bar_start = now
                self.bar_open = price

            self.bar_high = max(self.bar_high, price)
            self.bar_low = min(self.bar_low, price)
            self.bar_volume += qty * price
            if is_buy:
                self.bar_buy_vol += qty * price
            self.mid = price

            # ── Mean reversion: track trades + detect impact ──
            self.mr_recent_trades.append((now, price, qty, is_buy))

            # Update EMA of trade size (notional)
            notional = qty * price
            alpha = 0.02
            if self.mr_avg_trade_size < 1:
                self.mr_avg_trade_size = notional  # bootstrap
            else:
                self.mr_avg_trade_size = (1 - alpha) * self.mr_avg_trade_size + alpha * notional

            # Check for impact: 3+ consecutive same-direction in 2s window
            # with total notional > 5x average
            self._detect_mr_impact(now)

            # Close 5min bar
            if now - self.bar_start >= 300:
                bar = {
                    'open': self.bar_open, 'high': self.bar_high,
                    'low': self.bar_low, 'close': price,
                    'volume': self.bar_volume, 'buy_pct': self.bar_buy_vol / max(self.bar_volume, 1),
                    'ts': now,
                }
                self.bars.append(bar)
                # Improvement 4: update MA50 from bar closes
                self.close_history.append(price)
                if len(self.close_history) >= 50:
                    self.ma50 = sum(self.close_history) / len(self.close_history)
                self.bar_open = price
                self.bar_high = price
                self.bar_low = price
                self.bar_volume = 0
                self.bar_buy_vol = 0
                self.bar_start = now

    def _detect_mr_impact(self, now: float):
        """Check recent trades for aggressive impact event. Must hold self.lock."""
        # Filter to 2-second window
        window_trades = [(ts, p, q, buy) for ts, p, q, buy in self.mr_recent_trades
                         if now - ts <= 2.0]
        if len(window_trades) < 3:
            return

        # Find longest consecutive same-direction run ending at most recent trade
        # Scan backwards from end of window
        last_dir = window_trades[-1][3]  # is_buy of last trade
        run_count = 0
        run_notional = 0.0
        for i in range(len(window_trades) - 1, -1, -1):
            ts, p, q, buy = window_trades[i]
            if buy == last_dir:
                run_count += 1
                run_notional += q * p
            else:
                break

        # Need 3+ consecutive same-direction AND total > 5x avg trade size
        if run_count >= 3 and self.mr_avg_trade_size > 0 and run_notional > 5 * self.mr_avg_trade_size:
            # Impact detected: signal opposite direction (mean reversion)
            self.mr_signal = -1 if last_dir else 1  # buy impact → short reversion
            self.mr_signal_time = now
            self.mr_impact_price = window_trades[-1][1]

    def mean_reversion_signal(self) -> int:
        """Return mean reversion signal. Expires after 30s."""
        if self.mr_signal == 0:
            return 0
        if time.time() - self.mr_signal_time > 30:
            self.mr_signal = 0
            return 0
        return self.mr_signal

    def push_depth(self, best_bid: float, best_ask: float, bid_depth: float, ask_depth: float):
        with self.lock:  # C1 fix
            self.best_bid = best_bid
            self.best_ask = best_ask
            self.mid = (best_bid + best_ask) / 2
            self.bid_depth = bid_depth
            self.ask_depth = ask_depth
            total = bid_depth + ask_depth
            self.ob_imbalance = (bid_depth - ask_depth) / total if total > 0 else 0

    def push_liq(self, side: str, notional: float):
        with self.lock:  # C1 fix
            now = time.time()
            self.liq_window.append({'ts': now, 'side': side, 'notional': notional})
            self.liq_window = [x for x in self.liq_window if x['ts'] > now - 10]

    # ── Layer A: Funding signal ──────────────────────────

    def funding_signal(self) -> int:
        """Funding direction. Lowered thresholds for BTC/ETH coverage."""
        if self.funding_rate > 0.0001:   # P2 fix: was 0.0003, too high for BTC/ETH
            return -1
        if self.funding_rate < -0.00005:  # P2 fix: was -0.0001
            return 1
        return 0

    # ── Layer B: 5min momentum signal ────────────────────

    def momentum_signal(self) -> int:
        """5min bar momentum + VPIN."""
        if len(self.bars) < 3:
            return 0

        b1 = self.bars[-1]
        b3 = self.bars[-3]

        # 3-bar momentum
        ret3 = (b1['close'] - b3['open']) / b3['open'] if b3['open'] > 0 else 0

        # Buy pressure (VPIN proxy)
        buy_pct = b1.get('buy_pct', 0.5)

        # Volume confirmation
        avg_vol = sum(b['volume'] for b in self.bars) / len(self.bars)
        vol_ratio = b1['volume'] / max(avg_vol, 1)

        # Signal: strong momentum + volume + directional flow
        if ret3 > 0.003 and buy_pct > 0.6 and vol_ratio > 1.2:
            return 1
        if ret3 < -0.003 and buy_pct < 0.4 and vol_ratio > 1.2:
            return -1
        return 0

    # ── Layer C: Liquidation prediction ──────────────────

    def liq_signal(self) -> int:
        """Detect liquidation cluster direction."""
        if not self.liq_window:
            return 0
        total = sum(x['notional'] for x in self.liq_window)
        if total < 50000:  # $50K minimum
            return 0
        buy_n = sum(x['notional'] for x in self.liq_window if x['side'] == 'BUY')
        sell_n = sum(x['notional'] for x in self.liq_window if x['side'] == 'SELL')
        if sell_n > buy_n * 1.3:
            return -1  # longs liquidated → short
        if buy_n > sell_n * 1.3:
            return 1  # shorts liquidated → long
        return 0

    # ── Improvement 1: On-chain signal ──────────────────

    def onchain_signal(self) -> int:
        """TxTfrCnt z-score: >1 → bullish activity, <-1 → bearish."""
        if self.onchain_z > 1.0:
            return 1
        if self.onchain_z < -1.0:
            return -1
        return 0

    # ── Improvement 2: DVOL signal ───────────────────────

    def dvol_signal(self) -> int:
        """DVOL z-score: high vol → avoid, low vol → favorable.
        Returns 0 (neutral) or -1 (block high vol regime)."""
        if self.dvol_z > 2.0:
            return -99  # extreme vol → block all trades
        return 0

    # ── Improvement 4: Trend filter ──────────────────────

    def trend_filter(self, direction: int) -> bool:
        """Returns True if direction agrees with trend (or trend unknown)."""
        if self.ma50 <= 0 or self.mid <= 0:
            return True  # no trend data → allow
        above_ma = self.mid > self.ma50
        if direction == 1 and above_ma:
            return True   # long + uptrend = OK
        if direction == -1 and not above_ma:
            return True   # short + downtrend = OK
        return False  # counter-trend → block

    # ── Combined signal (improved) ───────────────────────

    def combined_signal(self) -> tuple[int, str, float]:
        """Vote across 8 layers. Returns (direction, reason, confidence).
        Confidence: 0.5 for 2 votes, 1.0 for 3+."""
        fa = self.funding_signal()
        mb = self.momentum_signal()
        lc = self.liq_signal()
        oc = self.onchain_signal()    # Improvement 1
        dv = self.dvol_signal()       # Improvement 2
        mr = self.mean_reversion_signal()  # Tick-level mean reversion
        bl = self.btc_lead            # BTC lead-lag (set by orchestrator)

        # Improvement 2: DVOL extreme → block all
        if dv == -99:
            return 0, "dvol_block", 0.0

        # OB pressure
        ob = 0
        if self.ob_imbalance > 0.3:
            ob = 1
        elif self.ob_imbalance < -0.3:
            ob = -1

        # Improvement 3: funding gets 0.5 weight (reduce short bias)
        # Mean reversion gets 1.5 weight (strong short-lived signal)
        # BTC lead gets 2.0 weight (strongest — reaction to confirmed BTC move)
        votes = fa * 0.5 + mb + lc + ob + oc + mr * 1.5 + bl * 2.0

        reasons = []
        if fa != 0:
            reasons.append(f"f={fa:+d}")
        if mb != 0:
            reasons.append(f"m={mb:+d}")
        if lc != 0:
            reasons.append(f"l={lc:+d}")
        if ob != 0:
            reasons.append(f"ob={ob:+d}")
        if oc != 0:
            reasons.append(f"oc={oc:+d}")
        if mr != 0:
            reasons.append(f"mr={mr:+d}")
        if bl != 0:
            reasons.append(f"bl={bl:+d}")

        direction = 0
        if votes >= 1.5:
            direction = 1
        elif votes <= -1.5:
            direction = -1

        if direction == 0:
            return 0, "", 0.0

        # Improvement 4: trend filter
        if not self.trend_filter(direction):
            return 0, "trend_block", 0.0

        # Improvement 6: confidence from vote count
        confidence = min(abs(votes) / 3.0, 1.0)  # 1.5→0.5, 3.0→1.0

        return direction, " ".join(reasons), confidence


class SignalHFT:
    """Multi-layer, multi-symbol signal-driven limit trading."""

    def __init__(self, symbols: list[str], adapter, leverage: int = 20,
                 position_size_usd: float = 2000, daily_loss_limit: float = 200,
                 dry_run: bool = False):
        self._symbols = symbols
        self._adapter = adapter
        self._leverage = leverage
        self._pos_size = position_size_usd
        self._daily_loss = daily_loss_limit
        self._dry_run = dry_run
        self._running = False
        self._engines = {s: SymbolEngine(s) for s in symbols}
        self._daily_pnl = 0.0
        self._last_signal_check = 0.0
        self._day_start = time.time()          # H1 fix: track day boundary
        self._ws_thread: threading.Thread | None = None  # H3 fix: track WS thread
        # Correlation guard: max 1 directional bet across correlated symbols
        self._max_correlated_positions = 1

        # BTC lead-lag state (cross-symbol: BTC price feeds into ETH/SOL signals)
        self._btc_tick_prices: deque = deque(maxlen=200)  # (timestamp, price)
        self._btc_lead_signal = 0      # +1/-1/0: BTC just moved, ALTs should follow
        self._btc_lead_signal_time = 0.0
        self._btc_lead_threshold = 0.0003  # 0.03% move in 500ms
        self._btc_lead_window = 0.5       # detection window (seconds)
        self._btc_lead_expiry = 3.0       # signal valid for 3s (ALT lag window)

    def _update_btc_lead(self, price: float):
        """Track BTC tick prices and detect fast moves."""
        now = time.time()
        self._btc_tick_prices.append((now, price))

        # Find price at start of detection window (500ms ago)
        cutoff = now - self._btc_lead_window
        old_price = None
        for ts, p in self._btc_tick_prices:
            if ts >= cutoff:
                old_price = p
                break

        if old_price is None or old_price <= 0:
            return

        pct_change = (price - old_price) / old_price

        if abs(pct_change) >= self._btc_lead_threshold:
            self._btc_lead_signal = 1 if pct_change > 0 else -1
            self._btc_lead_signal_time = now

    def _get_btc_lead_signal(self) -> int:
        """Get BTC lead signal for ALT coins. Expires after 3s."""
        if self._btc_lead_signal == 0:
            return 0
        if time.time() - self._btc_lead_signal_time > self._btc_lead_expiry:
            self._btc_lead_signal = 0
            return 0
        return self._btc_lead_signal

    def start(self):
        self._running = True
        for sym in self._symbols:
            try:
                result = self._adapter._client.post('/v5/position/set-leverage', body={
                    'category': 'linear', 'symbol': sym,
                    'buyLeverage': str(self._leverage),
                    'sellLeverage': str(self._leverage),
                })
                if isinstance(result, dict) and result.get("retCode", 0) != 0:
                    log.warning("set_leverage API failed for %s: %s", sym, result.get("retMsg", "unknown"))
            except Exception as e:
                log.warning("set_leverage failed for %s: %s", sym, e)

        self._start_bybit_ws()
        self._start_liq_ws()
        log.info("HFT v2 started: %s %dx dry=%s", self._symbols, self._leverage, self._dry_run)

    def _start_bybit_ws(self):
        """Single Bybit WS for depth + trades + tickers."""
        import websocket as ws_lib
        subs = []
        for s in self._symbols:
            subs.extend([f"orderbook.1.{s}", f"publicTrade.{s}", f"tickers.{s}"])

        def on_msg(ws, msg):
            try:
                data = json.loads(msg)
                topic = data.get("topic", "")
                if not topic:
                    return
                for sym in self._symbols:
                    if sym not in topic:
                        continue
                    if "orderbook" in topic:
                        self._handle_depth(sym, data.get("data", {}))
                    elif "publicTrade" in topic:
                        self._handle_trades(sym, data.get("data", []))
                    elif "tickers" in topic:
                        d = data.get("data", {})
                        fr = d.get("fundingRate")
                        if fr:
                            self._engines[sym].funding_rate = float(fr)
                    break
            except Exception:
                log.debug("WS message parse error", exc_info=True)

        def on_open(ws):
            ws.send(json.dumps({"op": "subscribe", "args": subs}))
            log.info("Bybit WS subscribed: %d topics", len(subs))

        def on_close(ws, code, msg):
            if self._running:
                # H3 fix: only reconnect if old thread is dead
                if self._ws_thread and self._ws_thread.is_alive():
                    return
                log.warning("Bybit WS closed, reconnecting...")
                time.sleep(3)
                self._start_bybit_ws()

        def on_ping(ws, data):
            ws.pong(data)

        self._ws = ws_lib.WebSocketApp(
            "wss://stream.bybit.com/v5/public/linear",
            on_message=on_msg, on_open=on_open,
            on_close=on_close, on_ping=on_ping,
            on_error=lambda ws, e: log.error("WS: %s", e),
        )
        self._ws_thread = threading.Thread(
            target=self._ws.run_forever,
            daemon=True, name="bybit-ws",
        )
        self._ws_thread.start()

    def _start_liq_ws(self):
        import websocket as ws_lib

        def on_msg(ws, msg):
            try:
                data = json.loads(msg)
                if data.get('e') == 'forceOrder':
                    o = data['o']
                    sym = o.get('s', '')
                    if sym in self._engines:
                        self._engines[sym].push_liq(
                            o.get('S', ''),
                            float(o.get('p', 0)) * float(o.get('q', 0)),
                        )
            except Exception:
                log.debug("Liq WS message parse error", exc_info=True)

        self._liq_ws = ws_lib.WebSocketApp(
            "wss://fstream.binance.com/ws/!forceOrder@arr",
            on_message=on_msg,
            on_open=lambda ws: log.info("Liq WS connected"),
            on_close=lambda ws, c, m: (time.sleep(3), self._start_liq_ws()) if self._running else None,
        )
        threading.Thread(target=self._liq_ws.run_forever, daemon=True, name="liq-ws").start()

    def _handle_depth(self, sym: str, data: dict):
        bids = data.get("b", [])
        asks = data.get("a", [])
        if not bids or not asks:
            return
        bb = float(bids[0][0])
        ba = float(asks[0][0])
        bd = sum(float(b[1]) * float(b[0]) for b in bids)
        ad = sum(float(a[1]) * float(a[0]) for a in asks)
        self._engines[sym].push_depth(bb, ba, bd, ad)

    def _handle_trades(self, sym: str, trades: list):
        for t in trades:
            p = float(t.get("p", 0))
            q = float(t.get("v", 0))
            is_buy = t.get("S", "Buy") == "Buy"
            if p > 0:
                self._engines[sym].push_trade(p, q, is_buy)
                # BTC lead-lag: track BTC tick prices for ALT signal
                if sym == "BTCUSDT":
                    self._update_btc_lead(p)

    def _signal_check(self):
        """Run every 5 seconds: check signals + manage positions."""
        now = time.time()
        if now - self._last_signal_check < 5:
            return
        self._last_signal_check = now

        # H1 fix: daily PnL reset at midnight UTC
        if now - self._day_start > 86400:
            log.info("DAILY_RESET pnl=$%.2f", self._daily_pnl)
            self._daily_pnl = 0.0
            self._day_start = now

        if self._daily_pnl <= -self._daily_loss:
            return

        # ── Correlation guard: BTC/ETH/SOL move together ──
        # Max 1 position across all symbols. Pick highest-confidence signal.
        active_syms = [s for s, e in self._engines.items()
                       if e.pos_side != 0 or e.pending]

        # Phase 1: manage existing positions
        for sym in self._symbols:
            eng = self._engines[sym]
            if eng.pos_side != 0:
                self._check_exit(eng)

        # Phase 2: only enter if no active position anywhere
        if not active_syms:
            # Gather all signals, pick best
            candidates = []
            for sym in self._symbols:
                eng = self._engines[sym]
                if eng.pos_side != 0 or eng.pending:
                    continue
                # Set BTC lead signal: only for ALTs, not BTC itself
                if sym != "BTCUSDT":
                    eng.btc_lead = self._get_btc_lead_signal()
                else:
                    eng.btc_lead = 0  # BTC doesn't lead itself
                direction, reason, confidence = eng.combined_signal()
                if direction != 0:
                    candidates.append((confidence, sym, direction, reason))
            # Enter the highest-confidence signal only
            if candidates:
                candidates.sort(reverse=True)  # highest confidence first
                confidence, sym, direction, reason = candidates[0]
                self._enter(self._engines[sym], direction, reason, confidence)

    def _enter(self, eng: SymbolEngine, direction: int, reason: str, confidence: float = 1.0):
        # P1 fix: don't re-enter if already positioned or pending same direction
        if eng.pos_side == direction:
            return
        if eng.pending:
            return

        spec = SPECS.get(eng.symbol, {'tick': 0.01, 'step': 0.01, 'min_not': 5})
        price = eng.best_bid if direction == 1 else eng.best_ask
        if price <= 0:
            return

        # Improvement 6: Kelly sizing — scale by confidence
        # confidence 0.5 → 50% size, 1.0 → 100% size
        size_mult = max(0.3, min(confidence, 1.0))
        qty = self._pos_size * size_mult / price
        step = spec['step']
        qty = round(int(qty / step) * step, 6)
        if qty * price < spec['min_not']:
            return

        side = "Buy" if direction == 1 else "Sell"

        if self._dry_run:
            log.info("DRY ENTER %s %s %.4f @ $%.2f [%s]", eng.symbol, side, qty, price, reason)
            eng.order_id = "dry"
        else:
            try:
                result = self._adapter._client.post('/v5/order/create', body={
                    'category': 'linear', 'symbol': eng.symbol, 'side': side,
                    'orderType': 'Limit', 'qty': str(qty), 'price': str(price),
                    'timeInForce': 'PostOnly',
                })
                if result.get('retCode', -1) != 0:
                    log.debug("Entry rejected %s: %s", eng.symbol, result.get('retMsg', ''))
                    return
                eng.order_id = result.get('result', {}).get('orderId', '')
            except Exception as e:
                log.warning("Entry order failed %s: %s", eng.symbol, e)
                return

        # C2 fix: mark as pending until fill confirmed
        eng.pos_side = direction
        eng.pos_qty = qty
        eng.entry_price = price
        eng.entry_time = time.time()
        eng.entry_reason = reason
        eng.pending = True  # Not confirmed yet
        eng.trades += 1
        log.info("ENTER %s %s %.4f @ $%.2f [%s] (pending)", eng.symbol, side, qty, price, reason)

    def _check_exit(self, eng: SymbolEngine):
        # C2 fix: if pending, check if limit filled before applying exit logic
        if eng.pending:
            elapsed = time.time() - eng.entry_time
            if elapsed > 10:
                # Cancel unfilled limit + check if it filled
                try:
                    result = self._adapter._client.post('/v5/order/cancel-all', body={
                        'category': 'linear', 'symbol': eng.symbol,
                    })
                    if isinstance(result, dict) and result.get("retCode", 0) != 0:
                        log.warning("Cancel-all API failed %s: %s", eng.symbol, result.get("retMsg", "unknown"))
                except Exception as e:
                    log.warning("Cancel-all failed %s during pending check: %s", eng.symbol, e)
                try:
                    resp = self._adapter._client.get('/v5/position/list', params={
                        'category': 'linear', 'symbol': eng.symbol,
                    })
                    has_pos = False
                    for p in resp.get('result', {}).get('list', []):
                        if float(p.get('size', 0)) > 0:
                            has_pos = True
                            eng.entry_price = float(p.get('avgPrice', eng.entry_price))
                            eng.pending = False
                            eng.order_id = None
                            log.info("CONFIRMED %s fill @ $%.2f", eng.symbol, eng.entry_price)
                            break
                    if not has_pos:
                        eng.pos_side = 0
                        eng.pending = False
                        eng.order_id = None
                except Exception as e:
                    log.warning("Position check failed %s, resetting to flat: %s", eng.symbol, e)
                    eng.pos_side = 0
                    eng.pending = False
                    eng.order_id = None
            return  # Don't apply exit logic while pending

        price = eng.mid
        if price <= 0 or eng.entry_price <= 0:
            return

        elapsed = time.time() - eng.entry_time
        pnl_pct = ((price - eng.entry_price) / eng.entry_price) * eng.pos_side

        # BTC lead-lag entries: ultra-tight TP/SL/timeout (fast in-and-out)
        if "bl=" in eng.entry_reason:
            if pnl_pct > 0.001:
                self._exit(eng, f"BL_TP {pnl_pct*100:.2f}%")
            elif pnl_pct < -0.0008:
                self._exit(eng, f"BL_SL {pnl_pct*100:.2f}%")
            elif elapsed > 30:
                self._exit(eng, f"BL_TIMEOUT {elapsed:.0f}s")
            return

        # Mean reversion entries: tighter TP/SL/timeout
        if "mr=" in eng.entry_reason:
            if pnl_pct > 0.0015:
                self._exit(eng, f"MR_TP {pnl_pct*100:.2f}%")
            elif pnl_pct < -0.001:
                self._exit(eng, f"MR_SL {pnl_pct*100:.2f}%")
            elif elapsed > 60:
                self._exit(eng, f"MR_TIMEOUT {elapsed:.0f}s")
            return

        # TP: > 0.3%
        if pnl_pct > 0.003:
            self._exit(eng, f"TP {pnl_pct*100:.2f}%")
        # SL: < -0.5%
        elif pnl_pct < -0.005:
            self._exit(eng, f"SL {pnl_pct*100:.2f}%")
        # Timeout: 5min for momentum, 30min for funding
        elif "fund" in eng.entry_reason and elapsed > 1800:
            self._exit(eng, f"TIMEOUT_FUND {elapsed:.0f}s")
        elif "fund" not in eng.entry_reason and elapsed > 300:
            self._exit(eng, f"TIMEOUT {elapsed:.0f}s")

    def _exit(self, eng: SymbolEngine, reason: str):
        side = "Sell" if eng.pos_side == 1 else "Buy"
        price = eng.mid
        pnl = ((price - eng.entry_price) / eng.entry_price) * eng.pos_side * eng.pos_qty * eng.entry_price

        if not self._dry_run:
            try:
                result = self._adapter._client.post('/v5/order/cancel-all', body={
                    'category': 'linear', 'symbol': eng.symbol,
                })
                if isinstance(result, dict) and result.get("retCode", 0) != 0:
                    log.warning("Exit cancel-all API failed %s: %s", eng.symbol, result.get("retMsg", "unknown"))
                result = self._adapter._client.post('/v5/order/create', body={
                    'category': 'linear', 'symbol': eng.symbol, 'side': side,
                    'orderType': 'Market', 'qty': str(eng.pos_qty), 'reduceOnly': True,
                })
                if isinstance(result, dict) and result.get("retCode", 0) != 0:
                    log.warning("Exit order API failed %s: %s", eng.symbol, result.get("retMsg", "unknown"))
            except Exception:
                log.exception("Exit failed %s", eng.symbol)

        self._daily_pnl += pnl
        if pnl > 0:
            eng.wins += 1
        else:
            eng.losses += 1
        eng.pnl += pnl

        log.info("EXIT %s %s $%.2f (%.2f%%) [%s] | total=$%.2f W=%d L=%d",
                 eng.symbol, side, pnl, pnl / max(eng.pos_qty * eng.entry_price, 1) * 100,
                 reason, self._daily_pnl, eng.wins, eng.losses)

        eng.pos_side = 0
        eng.pos_qty = 0
        eng.order_id = None

    def _poll_onchain_dvol(self):
        """Poll on-chain metrics + DVOL every 5min (free APIs)."""
        for sym in self._symbols:
            eng = self._engines[sym]
            # On-chain: use Coin Metrics (already downloaded daily)
            # For live: poll from onchain_poller (if running)
            # DVOL: use Deribit API
            try:
                import urllib.request
                currency = "BTC" if "BTC" in sym else "ETH" if "ETH" in sym else "SOL"
                if currency in ("BTC", "ETH"):
                    url = f"https://www.deribit.com/api/v2/public/get_historical_volatility?currency={currency}"
                    resp = json.loads(urllib.request.urlopen(url, timeout=5).read())
                    data = resp.get('result', [])
                    if data:
                        current_iv = data[-1][1]
                        # Simple z-score vs recent
                        if len(data) > 50:
                            vals = [d[1] for d in data[-50:]]
                            mu = sum(vals) / len(vals)
                            sd = (sum((v - mu)**2 for v in vals) / len(vals)) ** 0.5
                            if sd > 0.1:
                                eng.dvol_z = (current_iv - mu) / sd
            except Exception as e:
                log.debug("DVOL poll failed for %s: %s", sym, e)

    def _log_status(self):
        for sym in self._symbols:
            eng = self._engines[sym]
            if eng.mid > 0:
                bars = len(eng.bars)
                fund_sig = eng.funding_signal()
                mom_sig = eng.momentum_signal()
                liq_sig = eng.liq_signal()
                mr_sig = eng.mean_reversion_signal()
                bl_sig = self._get_btc_lead_signal() if sym != "BTCUSDT" else 0
                log.info("STATUS %s mid=$%.1f pos=%+d bars=%d fund=%+d mom=%+d liq=%+d "
                         "ob=%.2f oc=%+d dvol=%.1f mr=%+d bl=%+d ma50=%.0f pnl=$%.2f",
                         sym, eng.mid, eng.pos_side, bars, fund_sig, mom_sig, liq_sig,
                         eng.ob_imbalance, eng.onchain_signal(), eng.dvol_z, mr_sig,
                         bl_sig, eng.ma50, eng.pnl)

    def run(self):
        self.start()
        last_log = 0.0
        last_ping = 0.0
        last_poll = 0.0
        try:
            while self._running:
                self._signal_check()
                now = time.time()
                if now - last_log > 60:
                    self._log_status()
                    last_log = now
                # Improvement 1+2: poll on-chain + DVOL every 5min
                if now - last_poll > 300:
                    self._poll_onchain_dvol()
                    last_poll = now
                # Send WS ping to keep alive
                if now - last_ping > 20:
                    try:
                        self._ws.send(json.dumps({"op": "ping"}))
                    except Exception as e:
                        log.debug("WS ping failed: %s", e)
                    last_ping = now
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self._running = False
        for sym in self._symbols:
            eng = self._engines[sym]
            if eng.pos_side != 0:
                self._exit(eng, "SHUTDOWN")
        if hasattr(self, '_ws'):
            self._ws.close()
        if hasattr(self, '_liq_ws'):
            self._liq_ws.close()
        log.info("HFT v2 stopped. daily=$%.2f", self._daily_pnl)


def main():
    parser = argparse.ArgumentParser(description="Signal-driven HFT v2")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--leverage", type=int, default=20)
    parser.add_argument("--position-size", type=float, default=2000)
    parser.add_argument("--loss-limit", type=float, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(level=level,
                        format="%(asctime)s %(levelname)-5s %(name)s  %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler("logs/hft_signal.log")])

    from execution.adapters.bybit.config import BybitConfig
    from execution.adapters.bybit.adapter import BybitAdapter
    cfg = BybitConfig(
        api_key=os.environ['BYBIT_API_KEY'],
        api_secret=os.environ['BYBIT_API_SECRET'],
        base_url=os.environ.get('BYBIT_BASE_URL', 'https://api-demo.bybit.com'),
    )
    adapter = BybitAdapter(cfg)

    hft = SignalHFT(symbols=args.symbols, adapter=adapter, leverage=args.leverage,
                    position_size_usd=args.position_size, daily_loss_limit=args.loss_limit,
                    dry_run=args.dry_run)

    def _sig(signum, frame):
        hft.stop()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    hft.run()


if __name__ == "__main__":
    main()
