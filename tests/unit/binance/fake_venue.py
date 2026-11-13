"""In-memory Binance USDⓈ-M venue speaking the REST shapes BinanceAdapter uses.

Supports klines (closed + the open bar), funding/positioning, one-way positions,
MARKET and GTX LIMIT orders, cancels, order/trade queries, algo STOP_MARKET
closePosition orders, and fault injection around order placement.
"""
from __future__ import annotations

import itertools
import json
from decimal import Decimal as D

from execution.adapters.binance.rest import BinanceNonRetryableError, BinanceRetryableError

HOUR = 3_600_000


class FakeBinance:
    def __init__(self, bars: list[dict] | None = None, *, clock=None, equity: str = "10000",
                 funding: list[tuple[int, float]] | None = None, positioning: list[dict] | None = None) -> None:
        self.bars = {int(b["start"]): b for b in (bars or [])}
        self.clock = clock or (lambda: 0.0)
        self.equity = D(equity)
        self.funding = funding or []
        self.positioning = positioning or []
        self.price: dict[str, D] = {}
        self.positions: dict[str, D] = {}
        self.entry: dict[str, D] = {}
        self.orders: dict[str, dict] = {}
        self.trades: list[dict] = []
        self.algo: dict[str, dict] = {}
        self.ids = itertools.count(1000)
        self.fault: str | None = None     # timeout_before | timeout_after | reject | error_5xx
        self.fault_code = -2019
        self.position_error = False
        self.hedge = False
        self.leverage: dict[str, int] = {}
        self.posts: list[dict] = []

    # -- helpers ---------------------------------------------------------
    def now_ms(self) -> int:
        return int(self.clock() * 1000)

    def mark(self, symbol: str) -> D:
        if symbol in self.price:
            return self.price[symbol]
        closed = [t for t in self.bars if t + HOUR <= self.now_ms()]
        return D(str(self.bars[max(closed)]["close"])) if closed else D("100")

    def _apply_fill(self, order: dict, qty: D, price: D) -> None:
        sign = D(1) if order["side"] == "BUY" else D(-1)
        old = self.positions.get(order["symbol"], D(0))
        new = old + sign * qty
        if old == 0 or old * sign > 0:
            self.entry[order["symbol"]] = (abs(old) * self.entry.get(order["symbol"], D(0)) + qty * price) / abs(new)
        elif new * old < 0:
            self.entry[order["symbol"]] = price
        self.positions[order["symbol"]] = new
        order["executedQty"] = str(D(order["executedQty"]) + qty)
        order["status"] = "FILLED" if D(order["executedQty"]) == D(order["origQty"]) else "PARTIALLY_FILLED"
        self.trades.append({"id": next(self.ids), "orderId": int(order["orderId"]), "symbol": order["symbol"],
                            "side": order["side"], "qty": str(qty), "price": str(price),
                            "commission": str((qty * price * D("0.0002")).quantize(D("0.00000001"))),
                            "commissionAsset": "USDT", "maker": order["type"] == "LIMIT",
                            "time": self.now_ms()})

    def fill_limit(self, order_id: str, qty: D | None = None) -> None:
        order = self.orders[order_id]
        remaining = D(order["origQty"]) - D(order["executedQty"])
        self._apply_fill(order, min(remaining, qty or remaining), D(order["price"]))

    def trigger_stop(self, client_algo_id: str, price: D) -> None:
        algo = self.algo[client_algo_id]
        qty = abs(self.positions.get(algo["symbol"], D(0)))
        order = self._new_order({"symbol": algo["symbol"], "side": algo["side"], "type": "MARKET",
                                 "quantity": str(qty), "newClientOrderId": f"autoclose-{client_algo_id}"})
        self._apply_fill(order, qty, price)
        algo.update(algoStatus="FINISHED", actualOrderId=order["orderId"])

    def _new_order(self, p: dict) -> dict:
        order_id = str(next(self.ids))
        order = {"orderId": int(order_id), "symbol": p["symbol"], "side": p["side"], "type": p["type"],
                 "origQty": p["quantity"], "executedQty": "0", "price": p.get("price", "0"),
                 "status": "NEW", "clientOrderId": p.get("newClientOrderId", f"auto{order_id}"),
                 "avgPrice": "0", "timeInForce": p.get("timeInForce", "GTC")}
        self.orders[order_id] = order
        return order

    @staticmethod
    def _error(code: int, msg: str, status: int = 400):
        raise BinanceNonRetryableError(f"HTTP {status}: " + json.dumps({"code": code, "msg": msg}),
                                       http_status=status, code=code)

    # -- client protocol ---------------------------------------------------
    def request_public(self, *, method: str, path: str, params=None):
        p = dict(params or {})
        if path == "/fapi/v1/exchangeInfo":
            def info(symbol):
                return {"symbol": symbol, "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                    {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                    {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                    {"filterType": "MIN_NOTIONAL", "notional": "5"}]}
            return {"symbols": [info("BTCUSDT"), info("ETHUSDT")]}
        if path == "/fapi/v1/klines":
            end = int(p.get("endTime", 10**15))
            rows = [b for t, b in sorted(self.bars.items()) if t <= end and t <= self.now_ms()]
            rows = rows[-int(p["limit"]):]
            return [[b["start"], str(b["open"]), str(b["high"]), str(b["low"]), str(b["close"]),
                     str(b["volume"]), b["start"] + HOUR - 1, str(b.get("quote_volume", 0)),
                     int(b.get("trades", 0)), str(b.get("taker_buy_volume", 0)),
                     str(b.get("taker_buy_quote_volume", 0)), "0"] for b in rows]
        if path == "/fapi/v1/fundingRate":
            rows = [f for f in self.funding if f[0] <= self.now_ms()]
            return [{"fundingTime": t + 7, "fundingRate": str(r)} for t, r in rows[-int(p.get("limit", 1)):]]
        if path.startswith("/futures/data/"):
            key = {"openInterestHist": ("sumOpenInterest", "open_interest"),
                   "globalLongShortAccountRatio": ("longShortRatio", "ls_ratio"),
                   "topLongShortPositionRatio": ("longShortRatio", "top_trader_ls_ratio")}[path.rsplit("/", 1)[1]]
            rows = [r for r in self.positioning if r["timestamp"] <= self.now_ms()][-int(p.get("limit", 30)):]
            return [{"timestamp": r["timestamp"], key[0]: str(r[key[1]])} for r in rows]
        if path == "/fapi/v1/ticker/bookTicker":
            m = self.mark(p["symbol"])
            return {"bidPrice": str(m - D("0.1")), "askPrice": str(m + D("0.1"))}
        if path == "/fapi/v1/ticker/price":
            return {"price": str(self.mark(p["symbol"]))}
        raise AssertionError(f"unexpected public path {path}")

    def request_signed(self, *, method: str, path: str, params=None):
        p = dict(params or {})
        if path == "/fapi/v1/positionSide/dual":
            return {"dualSidePosition": self.hedge}
        if path == "/fapi/v2/account":
            return {"totalMarginBalance": str(self.equity), "assets": []}
        if path == "/fapi/v1/leverage":
            self.leverage[p["symbol"]] = int(p["leverage"])
            return {"leverage": int(p["leverage"])}
        if path == "/fapi/v2/positionRisk":
            if self.position_error:
                raise BinanceRetryableError("HTTP 503: unavailable")
            return [{"symbol": s, "positionAmt": str(q), "entryPrice": str(self.entry.get(s, 0)),
                     "markPrice": str(self.mark(s)), "positionSide": "BOTH", "leverage": "3"}
                    for s, q in self.positions.items() if not p.get("symbol") or p["symbol"] == s]
        if path == "/fapi/v1/openOrders":
            return [o for o in self.orders.values() if o["status"] in ("NEW", "PARTIALLY_FILLED")
                    and (not p.get("symbol") or o["symbol"] == p["symbol"])]
        if path == "/fapi/v1/order" and method == "POST":
            self.posts.append(p)
            fault, self.fault = self.fault, None
            if fault == "timeout_before":
                raise BinanceRetryableError("Network error: timed out")
            if fault == "reject":
                self._error(self.fault_code, "rejected by venue")
            if fault == "error_5xx":
                raise BinanceRetryableError("HTTP 503: unknown execution status")
            if any(o["clientOrderId"] == p["newClientOrderId"] for o in self.orders.values()):
                self._error(-4116, "ClientOrderId is duplicated")
            symbol, qty = p["symbol"], D(p["quantity"])
            if p.get("reduceOnly") in (True, "true"):
                position = self.positions.get(symbol, D(0))
                if position == 0 or (position > 0) == (p["side"] == "BUY"):
                    self._error(-2022, "ReduceOnly Order is rejected")
                qty = min(qty, abs(position))
                p["quantity"] = str(qty)
            if p["type"] == "LIMIT" and p.get("timeInForce") == "GTX":
                price, mark = D(p["price"]), self.mark(symbol)
                if (p["side"] == "BUY" and price >= mark) or (p["side"] == "SELL" and price <= mark):
                    self._error(-5022, "Post Only order will be rejected")
            order = self._new_order(p)
            if p["type"] == "MARKET":
                self._apply_fill(order, qty, self.mark(symbol))
            if fault == "timeout_after":
                raise BinanceRetryableError("Network error: timed out after acceptance")
            return {"orderId": order["orderId"], "clientOrderId": order["clientOrderId"]}
        if path == "/fapi/v1/order" and method == "GET":
            for o in self.orders.values():
                if (p.get("orderId") and str(o["orderId"]) == str(p["orderId"])) or \
                        (p.get("origClientOrderId") and o["clientOrderId"] == p["origClientOrderId"]):
                    return dict(o)
            self._error(-2013, "Order does not exist.")
        if path == "/fapi/v1/order" and method == "DELETE":
            order = self.orders[str(p["orderId"])]
            if order["status"] in ("NEW", "PARTIALLY_FILLED"):
                order["status"] = "CANCELED"
            return dict(order)
        if path == "/fapi/v1/userTrades":
            return [t for t in self.trades if t["symbol"] == p["symbol"]
                    and (not p.get("orderId") or str(t["orderId"]) == str(p["orderId"]))]
        if path == "/fapi/v1/algoOrder" and method == "POST":
            symbol, side, trigger = p["symbol"], p["side"], D(p["triggerPrice"])
            mark = self.mark(symbol)
            if (side == "SELL" and trigger >= mark) or (side == "BUY" and trigger <= mark):
                self._error(-2021, "Order would immediately trigger.")
            self.algo[p["clientAlgoId"]] = {"algoId": next(self.ids), "clientAlgoId": p["clientAlgoId"],
                                            "symbol": symbol, "side": side, "algoStatus": "NEW",
                                            "triggerPrice": p["triggerPrice"], "actualOrderId": "",
                                            "closePosition": p.get("closePosition")}
            return dict(self.algo[p["clientAlgoId"]])
        if path == "/fapi/v1/algoOrder" and method == "GET":
            if p["clientAlgoId"] not in self.algo:
                self._error(-2013, "Order does not exist.")
            return dict(self.algo[p["clientAlgoId"]])
        if path == "/fapi/v1/algoOrder" and method == "DELETE":
            self.algo[p["clientAlgoId"]]["algoStatus"] = "CANCELED"
            return {"code": 200, "msg": "success"}
        raise AssertionError(f"unexpected signed path {method} {path}")


def make_bars(count: int, start_ms: int = 1_700_006_400_000, *, base: float = 100.0) -> list[dict]:
    """Deterministic hourly klines (oscillating) with microstructure fields."""
    import math
    start_ms = start_ms // (4 * HOUR) * (4 * HOUR)
    rows = []
    for i in range(count):
        o = base + 8 * math.sin(i / 37) + 3 * math.sin(i / 7)
        c = base + 8 * math.sin((i + 1) / 37) + 3 * math.sin((i + 1) / 7)
        rows.append({"start": start_ms + i * HOUR, "open": round(o, 2), "high": round(max(o, c) * 1.004, 2),
                     "low": round(min(o, c) * 0.996, 2), "close": round(c, 2), "volume": 5000.0,
                     "quote_volume": round(5000 * c, 2), "trades": 900, "taker_buy_volume": 2500.0,
                     "taker_buy_quote_volume": round(2500 * c, 2)})
    return rows
