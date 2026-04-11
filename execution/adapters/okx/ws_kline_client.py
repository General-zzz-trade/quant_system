"""OKX WebSocket client — kline (business channel) + tickers (public channel).

OKX v5 splits data across two public WS URLs:
- Public  (`/ws/v5/public`):   `tickers`, `trades`, `books` — used for on_tick
- Business (`/ws/v5/business`): `candle1H`, `candle4H` — used for on_bar

We open two connections and unify them behind a single on_bar / on_tick
callback interface that matches BinanceWsClient / BybitWsClient.

Subscribe message:
    {"op":"subscribe","args":[{"channel":"candle1H","instId":"BTC-USDT-SWAP"}]}

Candle message (note the array format — arg + data[][9]):
    {"arg":{"channel":"candle1H","instId":"BTC-USDT-SWAP"},
     "data":[["1775880000000","72878.5","72909.2","72778.1","72890.0",
              "248711.33","2486.11","18123456.2","1"]]}

    Array positions: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    confirm: "0" (still forming) or "1" (closed bar)

Ticker message:
    {"arg":{"channel":"tickers","instId":"BTC-USDT-SWAP"},
     "data":[{"instId":"BTC-USDT-SWAP","last":"72890","bidPx":"...","askPx":"..."}]}

Heartbeat: client must send "ping" text every ~25s, expects "pong".
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable

import websocket

from execution.adapters.okx.symbol_map import from_okx_symbol, to_okx_symbol
from execution.adapters.okx.urls import (
    OKX_WS_BUSINESS,
    OKX_WS_BUSINESS_DEMO,
    OKX_WS_PUBLIC,
    OKX_WS_PUBLIC_DEMO,
)

logger = logging.getLogger(__name__)


# Interval mapping: internal format → OKX channel name
# OKX uses candle1H/candle4H (uppercase H/M/D) not 1h/4h
_INTERVAL_TO_CHANNEL = {
    "60":  "candle1H",
    "240": "candle4H",
    "15":  "candle15m",
    "5":   "candle5m",
    "1":   "candle1m",
}
_CHANNEL_TO_INTERVAL = {v: k for k, v in _INTERVAL_TO_CHANNEL.items()}


class OkxWsClient:
    """Two-connection OKX WS client (candle on business, tickers on public).

    Parallel to BinanceWsClient: exposes the same on_bar / on_tick / start /
    stop / get_last_price surface so alpha_builder can swap venues cleanly.

    `symbols` uses internal naming (BTCUSDT, ETHUSDT); we translate to OKX
    instId (BTC-USDT-SWAP) at the wire boundary.
    """

    PING_INTERVAL_S = 25.0

    def __init__(
        self,
        symbols: list[str],
        interval: str = "60",
        on_bar: Callable[[str, dict], None] | None = None,
        on_tick: Callable[[str, float], None] | None = None,
        demo: bool = False,
    ) -> None:
        self._internal_symbols = [s.upper() for s in symbols]
        self._inst_ids = [to_okx_symbol(s) for s in self._internal_symbols]
        self._interval = str(interval)
        self._channel = _INTERVAL_TO_CHANNEL.get(self._interval, "candle1H")
        self._on_bar = on_bar
        self._on_tick = on_tick
        self._last_prices: dict[str, float] = {}
        self._last_bar_ts: dict[str, int] = {}
        self._business_url = OKX_WS_BUSINESS_DEMO if demo else OKX_WS_BUSINESS
        self._public_url = OKX_WS_PUBLIC_DEMO if demo else OKX_WS_PUBLIC

        self._ws_business: Any = None
        self._ws_public: Any = None
        self._thread_business: threading.Thread | None = None
        self._thread_public: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread_business = threading.Thread(
            target=self._run_loop,
            args=(self._business_url, self._subscribe_candles, "business"),
            daemon=True,
            name=f"okx-ws-candles-{self._channel}",
        )
        self._thread_public = threading.Thread(
            target=self._run_loop,
            args=(self._public_url, self._subscribe_tickers, "public"),
            daemon=True,
            name="okx-ws-tickers",
        )
        self._thread_business.start()
        self._thread_public.start()
        logger.info(
            "OKX WS started: symbols=%s channel=%s business_url=%s",
            self._inst_ids, self._channel, self._business_url,
        )

    def stop(self) -> None:
        self._running = False
        for ws in (self._ws_business, self._ws_public):
            if ws:
                try:
                    ws.close()
                except Exception:
                    pass
        for th in (self._thread_business, self._thread_public):
            if th and th.is_alive():
                th.join(timeout=3)
        logger.info("OKX WS stopped")

    def get_last_price(self, symbol: str) -> float:
        """Get last known price for an internal symbol name (BTCUSDT)."""
        return self._last_prices.get(symbol.upper(), 0.0)

    def get_last_funding_rate(self, symbol: str) -> float:
        """Funding not emitted on WS path — poll REST if needed."""
        return float("nan")

    # ── Subscription builders ─────────────────────────────────────
    def _subscribe_candles(self, ws: Any) -> None:
        args = [{"channel": self._channel, "instId": inst} for inst in self._inst_ids]
        ws.send(json.dumps({"op": "subscribe", "args": args}))

    def _subscribe_tickers(self, ws: Any) -> None:
        args = [{"channel": "tickers", "instId": inst} for inst in self._inst_ids]
        ws.send(json.dumps({"op": "subscribe", "args": args}))

    # ── Main loop with reconnection ───────────────────────────────
    def _run_loop(
        self,
        url: str,
        subscribe_fn: Callable[[Any], None],
        label: str,
    ) -> None:
        backoff = 1.0
        max_backoff = 60.0
        reconnect_count = 0
        while self._running:
            try:
                self._connect_and_listen(url, subscribe_fn, label)
                backoff = 1.0  # reset on clean exit
            except Exception as e:
                if not self._running:
                    break
                reconnect_count += 1
                logger.warning(
                    "OKX WS %s error (reconnect #%d): %s",
                    label, reconnect_count, e,
                )
            if self._running:
                time.sleep(min(backoff, max_backoff))
                backoff = min(backoff * 2, max_backoff)

    def _connect_and_listen(
        self,
        url: str,
        subscribe_fn: Callable[[Any], None],
        label: str,
    ) -> None:
        ws = websocket.WebSocket()
        ws.connect(url, timeout=30)
        ws.settimeout(30)

        if label == "business":
            self._ws_business = ws
        else:
            self._ws_public = ws

        subscribe_fn(ws)
        logger.info("OKX WS %s connected, subscribed.", label)

        last_ping = time.monotonic()
        while self._running:
            # Send heartbeat
            now = time.monotonic()
            if now - last_ping >= self.PING_INTERVAL_S:
                try:
                    ws.send("ping")
                except Exception:
                    pass
                last_ping = now

            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except websocket.WebSocketConnectionClosedException:
                logger.warning("OKX WS %s closed", label)
                break

            if not raw:
                continue
            # OKX returns a literal "pong" string in response to "ping"
            if raw == "pong":
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # Response to subscribe: {"event":"subscribe","arg":{...}}
            if isinstance(msg, dict) and msg.get("event") == "error":
                logger.error("OKX WS %s error: %s", label, msg)
                continue
            if isinstance(msg, dict) and msg.get("event") == "subscribe":
                continue

            self._dispatch(msg)

    # ── Message dispatch ──────────────────────────────────────────
    def _dispatch(self, msg: dict) -> None:
        arg = msg.get("arg", {})
        channel = arg.get("channel", "")
        data = msg.get("data") or []
        if not data:
            return

        if channel.startswith("candle"):
            self._handle_candles(arg, data)
        elif channel == "tickers":
            self._handle_tickers(arg, data)

    def _handle_candles(self, arg: dict, data: list) -> None:
        inst_id = arg.get("instId", "")
        try:
            internal_symbol = from_okx_symbol(inst_id)
        except KeyError:
            return

        for row in data:
            if len(row) < 9:
                continue
            ts_ms = int(row[0])
            confirm = row[8]  # "0" forming, "1" closed
            if str(confirm) != "1":
                continue  # skip unconfirmed bars

            if ts_ms <= self._last_bar_ts.get(internal_symbol, 0):
                continue
            self._last_bar_ts[internal_symbol] = ts_ms

            bar = {
                "time": ts_ms // 1000,
                "open":  float(row[1]),
                "high":  float(row[2]),
                "low":   float(row[3]),
                "close": float(row[4]),
                "volume":   float(row[5]),      # base ccy vol (coin units)
                "turnover": float(row[7]),      # quote ccy vol (USDT)
                "confirm": True,
                "interval": _CHANNEL_TO_INTERVAL.get(
                    arg.get("channel", ""), self._interval
                ),
            }

            logger.info(
                "OKX WS bar: %s interval=%s close=$%.2f",
                internal_symbol, bar["interval"], bar["close"],
            )

            if self._on_bar:
                try:
                    self._on_bar(internal_symbol, bar)
                except Exception:
                    logger.exception("on_bar callback error for %s", internal_symbol)

    def _handle_tickers(self, arg: dict, data: list) -> None:
        for t in data:
            inst_id = t.get("instId", arg.get("instId", ""))
            try:
                internal_symbol = from_okx_symbol(inst_id)
            except KeyError:
                continue

            try:
                price = float(t.get("last", 0))
            except (TypeError, ValueError):
                continue
            if price <= 0:
                continue

            self._last_prices[internal_symbol] = price

            if self._on_tick:
                try:
                    self._on_tick(internal_symbol, price)
                except Exception:
                    pass
