"""Binance USDⓈ-M perpetual venue adapter (REST), built for confirmed execution.

Contract highlights (the execution layer relies on every one of them):

- Snapshots never fail open: ``get_positions``/``get_open_orders`` raise on any
  error instead of returning an empty tuple that would look flat.
- Every order carries a deterministic client id (``newClientOrderId``) so an
  ambiguous submission can be looked up instead of resent.
- ``send_market_order`` returns ``submitted``, ``rejected`` (the venue refused
  the request; nothing was placed) or ``error`` (outcome unknown: timeout/5xx).
- ``get_order`` returns ``None`` only for the venue's "order does not exist".
- Quantities/prices are floored to exchange filters; orders below the minimum
  quantity/notional are rejected locally.
- One-way position mode only. Hedge mode is reported as an execution blocker.
- Protective stops use the Algo service (``POST /fapi/v1/algoOrder``,
  ``algoType=CONDITIONAL``, ``STOP_MARKET`` + ``closePosition``); conditional
  types on ``/fapi/v1/order`` are rejected by Binance since late 2025.
- ``get_klines`` returns only CLOSED bars, oldest first, with ``start`` in ms.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from types import SimpleNamespace
from typing import Any, Optional, Tuple

from trading.adapters.binance.config import BinanceConfig
from trading.adapters.binance.rest import (
    BinanceNonRetryableError, BinanceRateLimitError, BinanceRestClient, BinanceRestConfig,
)
from trading.adapters.binance.rate_limit import make_rate_limit_policy
from trading.domain.venue import BalanceSnapshot, CanonicalBalance, CanonicalFill, VenuePosition

logger = logging.getLogger(__name__)

# Hard ceiling on the exchange LEVERAGE SETTING (margin per position, liquidation distance).
# Actual exposure is capped separately at 3x equity by shared.exposure_gate. The setting
# changes only through human config (BINANCE_EXCHANGE_LEVERAGE); the default stays 3.
MAX_LEVERAGE = 10
DEFAULT_LEVERAGE = 3
MARGIN_TYPES = {"ISOLATED": "isolated", "CROSSED": "cross"}   # request value -> positionRisk value
_NO_MARGIN_CHANGE_NEEDED = -4046

_INTERVAL_MAP = {
    "1": "1m", "5": "5m", "15": "15m", "60": "1h", "240": "4h", "D": "1d",
    "1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d",
}
_INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
                "4h": 14_400_000, "1d": 86_400_000}
_ORDER_STATES = {"NEW": "new", "PARTIALLY_FILLED": "partially_filled", "FILLED": "filled",
                 "CANCELED": "canceled", "EXPIRED": "expired", "EXPIRED_IN_MATCH": "expired",
                 "REJECTED": "rejected"}
_ORDER_DOES_NOT_EXIST = -2013
_ALGO_DOES_NOT_EXIST = (-2013, -2011)


@dataclass(frozen=True, slots=True)
class SymbolFilters:
    step_size: Decimal
    min_qty: Decimal
    tick_size: Decimal
    min_notional: Decimal

    def floor_qty(self, qty: Decimal) -> Decimal:
        return (qty / self.step_size).to_integral_value(rounding=ROUND_DOWN) * self.step_size

    def floor_price(self, price: Decimal) -> Decimal:
        return (price / self.tick_size).to_integral_value(rounding=ROUND_DOWN) * self.tick_size


def _fmt(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return text if text else "0"


class BinanceAdapter:
    """Binance USDⓈ-M futures REST adapter (one-way mode, USDT margin)."""

    venue: str = "binance"

    def __init__(self, config: BinanceConfig, *, client: Any = None) -> None:
        self._config = config
        rest_cfg = BinanceRestConfig(base_url=config.base_url, api_key=config.api_key,
                                     api_secret=config.api_secret)
        self._client = client or BinanceRestClient(rest_cfg, rate_policy=make_rate_limit_policy())
        self._connected = False
        self._hedge_mode = False
        self._filters: dict[str, SymbolFilters] = {}

    # ------------------------------------------------------------------
    # Connection and account configuration
    # ------------------------------------------------------------------
    def connect(self, *, symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
                configure_leverage: bool = False, leverage: int = DEFAULT_LEVERAGE,
                margin_type: str | None = None) -> bool:
        """Load filters, detect position mode, verify auth; optionally set and verify leverage/margin type.

        Any symbol whose exchange setting differs from the request afterwards becomes an
        execution blocker (for example, Binance refuses a margin-type change while a
        position or order is open)."""
        try:
            info = self._client.request_public(method="GET", path="/fapi/v1/exchangeInfo")
            by_symbol = {s["symbol"]: s for s in info.get("symbols", [])}
            for symbol in symbols:
                self._filters[symbol] = self._parse_filters(by_symbol[symbol])
            mode = self._client.request_signed(method="GET", path="/fapi/v1/positionSide/dual")
            self._hedge_mode = bool(mode.get("dualSidePosition", False))
            self._client.request_signed(method="GET", path="/fapi/v2/account")
            self._config_issues = []
            if configure_leverage:
                for symbol in symbols:
                    if margin_type is not None:
                        self.set_margin_type(symbol, margin_type)
                    self.set_leverage(symbol, leverage)
                self._config_issues = self._verify_settings(symbols, leverage, margin_type)
            self._connected = True
            logger.info("Binance connected: %s symbols=%s hedge_mode=%s",
                        self._config.base_url, list(symbols), self._hedge_mode)
            return True
        except Exception as exc:
            logger.error("Binance connection failed: %s", exc)
            return False

    @staticmethod
    def _parse_filters(symbol_info: dict[str, Any]) -> SymbolFilters:
        filters = {f["filterType"]: f for f in symbol_info.get("filters", [])}
        lot = filters.get("MARKET_LOT_SIZE") or filters["LOT_SIZE"]
        step = Decimal(str(lot["stepSize"]))
        if step <= 0:
            step = Decimal(str(filters["LOT_SIZE"]["stepSize"]))
        notional = filters.get("MIN_NOTIONAL", {})
        return SymbolFilters(
            step_size=step,
            min_qty=Decimal(str(lot["minQty"])),
            tick_size=Decimal(str(filters["PRICE_FILTER"]["tickSize"])),
            min_notional=Decimal(str(notional.get("notional", notional.get("minNotional", "0")))),
        )

    def execution_blockers(self) -> list[str]:
        issues = []
        if not self._connected:
            issues.append("adapter not connected")
        if self._hedge_mode:
            issues.append("hedge (dual-side) position mode is not supported; switch the account to one-way mode")
        issues.extend(getattr(self, "_config_issues", []))
        return issues

    def filters(self, symbol: str) -> SymbolFilters:
        try:
            return self._filters[symbol]
        except KeyError:
            raise RuntimeError(f"No exchange filters loaded for {symbol}") from None

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        if not 1 <= int(leverage) <= MAX_LEVERAGE:
            raise ValueError(f"leverage must be between 1 and {MAX_LEVERAGE}")
        return self._client.request_signed(method="POST", path="/fapi/v1/leverage",
                                           params={"symbol": symbol, "leverage": int(leverage)})

    def set_margin_type(self, symbol: str, margin_type: str) -> None:
        margin_type = margin_type.upper()
        if margin_type not in MARGIN_TYPES:
            raise ValueError(f"margin type must be one of {sorted(MARGIN_TYPES)}")
        try:
            self._client.request_signed(method="POST", path="/fapi/v1/marginType",
                                        params={"symbol": symbol, "marginType": margin_type})
        except BinanceNonRetryableError as exc:
            if exc.code != _NO_MARGIN_CHANGE_NEEDED:
                logger.warning("%s: margin type change to %s refused: %s", symbol, margin_type, exc)

    def _verify_settings(self, symbols: tuple[str, ...], leverage: int, margin_type: str | None) -> list[str]:
        issues = []
        for symbol in symbols:
            rows = self._client.request_signed(method="GET", path="/fapi/v2/positionRisk",
                                               params={"symbol": symbol})
            row = next((r for r in rows if r.get("symbol") == symbol), None) if isinstance(rows, list) else None
            if row is None:
                issues.append(f"{symbol}: leverage/margin settings could not be read")
                continue
            if int(row.get("leverage", 0)) != int(leverage):
                issues.append(f"{symbol}: exchange leverage {row.get('leverage')} != required {leverage}")
            if margin_type is not None and str(row.get("marginType", "")).lower() != MARGIN_TYPES[margin_type.upper()]:
                issues.append(f"{symbol}: margin type {row.get('marginType')} != required {margin_type.upper()}")
        return issues

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------
    def get_account_equity(self) -> float:
        """USDT-M account equity including unrealized PnL; fail on absent data."""
        data = self._client.request_signed(method="GET", path="/fapi/v2/account")
        return float(data["totalMarginBalance"])

    def get_balances(self) -> BalanceSnapshot:
        try:
            data = self._client.request_signed(method="GET", path="/fapi/v2/account")
            ts_ms = int(time.time() * 1000)
            balances = []
            for a in data.get("assets", []):
                wallet = Decimal(str(a.get("walletBalance", "0")))
                if wallet == 0:
                    continue
                available = Decimal(str(a.get("availableBalance", "0")))
                balances.append(CanonicalBalance.from_free_locked(
                    venue="binance", asset=a.get("asset", ""), free=available,
                    locked=max(wallet - available, Decimal("0")), ts_ms=ts_ms))
            return BalanceSnapshot(venue="binance", balances=tuple(balances), ts_ms=ts_ms)
        except Exception as exc:
            logger.error("get_balances failed: %s", exc)
            return BalanceSnapshot(venue="binance", balances=(), ts_ms=0)

    def get_positions(self, symbol: str = "", *, strict: bool = True) -> Tuple[VenuePosition, ...]:
        """Non-flat positions. Errors propagate: an unknown snapshot is never 'flat'."""
        params: dict[str, Any] = {"symbol": symbol} if symbol else {}
        data = self._client.request_signed(method="GET", path="/fapi/v2/positionRisk", params=params)
        if not isinstance(data, list):
            raise RuntimeError("Invalid Binance position snapshot")
        positions = []
        for p in data:
            amt = Decimal(str(p.get("positionAmt", "0")))
            if not amt.is_finite():
                raise RuntimeError("Invalid Binance position quantity")
            if amt == 0:
                continue
            if p.get("positionSide", "BOTH") != "BOTH":
                raise RuntimeError("Hedge-mode position present; one-way mode required")
            positions.append(VenuePosition(
                venue="binance", symbol=p.get("symbol", ""), qty=amt,
                entry_price=Decimal(str(p.get("entryPrice", "0"))),
                mark_price=Decimal(str(p.get("markPrice", "0"))),
                liquidation_price=Decimal(str(p.get("liquidationPrice", "0"))),
                unrealized_pnl=Decimal(str(p.get("unRealizedProfit", "0"))),
                leverage=int(p.get("leverage", "1")), margin_type=p.get("marginType", "").lower(),
                ts_ms=int(time.time() * 1000)))
        return tuple(positions)

    def get_open_orders(self, symbol: str = "", *, strict: bool = True) -> tuple[Any, ...]:
        params: dict[str, Any] = {"symbol": symbol} if symbol else {}
        data = self._client.request_signed(method="GET", path="/fapi/v1/openOrders", params=params)
        if not isinstance(data, list):
            raise RuntimeError("Invalid Binance open-order snapshot")
        return tuple(SimpleNamespace(order_id=str(o["orderId"]), client_order_id=str(o.get("clientOrderId", "")),
                                     symbol=o["symbol"], side=o["side"].lower(), type=o.get("type", ""))
                     for o in data)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def send_market_order(self, symbol: str, side: str, qty: float, *, reduce_only: bool = False,
                          client_order_id: str = "") -> dict[str, Any]:
        """Place a market order; see module docstring for the result contract."""
        side = side.lower()
        if side not in ("buy", "sell") or not client_order_id:
            return {"status": "rejected", "msg": "side and client_order_id are required"}
        filters = self.filters(symbol)
        quantity = filters.floor_qty(Decimal(str(qty)))
        if not quantity.is_finite() or quantity < filters.min_qty or quantity <= 0:
            return {"status": "rejected", "msg": f"quantity {qty} below minimum {filters.min_qty}"}
        if self._hedge_mode:
            return {"status": "rejected", "msg": "hedge mode is not supported"}
        params: dict[str, Any] = {"symbol": symbol, "side": side.upper(), "type": "MARKET",
                                  "quantity": _fmt(quantity), "newClientOrderId": client_order_id,
                                  "newOrderRespType": "ACK"}
        if reduce_only:
            params["reduceOnly"] = True
        try:
            data = self._client.request_signed(method="POST", path="/fapi/v1/order", params=params)
        except (BinanceNonRetryableError, BinanceRateLimitError) as exc:
            return {"status": "rejected", "code": getattr(exc, "code", None), "msg": str(exc)}
        except Exception as exc:  # timeout, 5xx, network: the order may exist
            logger.error("Binance order outcome unknown for %s: %s", client_order_id, exc)
            return {"status": "error", "msg": str(exc)}
        return {"status": "submitted", "orderId": str(data.get("orderId", "")),
                "clientOrderId": str(data.get("clientOrderId", ""))}

    def send_limit_order(self, symbol: str, side: str, qty: float, price: float, *, post_only: bool = True,
                         reduce_only: bool = False, client_order_id: str = "") -> dict[str, Any]:
        """Post-only (GTX) limit order; a crossing order is rejected by Binance (-5022)."""
        side = side.lower()
        if side not in ("buy", "sell") or not client_order_id:
            return {"status": "rejected", "msg": "side and client_order_id are required"}
        filters = self.filters(symbol)
        quantity = filters.floor_qty(Decimal(str(qty)))
        limit = filters.floor_price(Decimal(str(price)))
        if quantity < filters.min_qty or quantity <= 0 or limit <= 0:
            return {"status": "rejected", "msg": "quantity or price below exchange minimum"}
        if not reduce_only and quantity * limit < filters.min_notional:
            return {"status": "rejected", "msg": f"notional below exchange minimum {filters.min_notional}"}
        if self._hedge_mode:
            return {"status": "rejected", "msg": "hedge mode is not supported"}
        params: dict[str, Any] = {"symbol": symbol, "side": side.upper(), "type": "LIMIT",
                                  "timeInForce": "GTX" if post_only else "GTC", "quantity": _fmt(quantity),
                                  "price": _fmt(limit), "newClientOrderId": client_order_id,
                                  "newOrderRespType": "ACK"}
        if reduce_only:
            params["reduceOnly"] = True
        try:
            data = self._client.request_signed(method="POST", path="/fapi/v1/order", params=params)
        except (BinanceNonRetryableError, BinanceRateLimitError) as exc:
            return {"status": "rejected", "code": getattr(exc, "code", None), "msg": str(exc)}
        except Exception as exc:
            logger.error("Binance order outcome unknown for %s: %s", client_order_id, exc)
            return {"status": "error", "msg": str(exc)}
        return {"status": "submitted", "orderId": str(data.get("orderId", "")),
                "clientOrderId": str(data.get("clientOrderId", ""))}

    def get_order(self, symbol: str, *, order_id: str = "", client_order_id: str = "") -> Any:
        if not order_id and not client_order_id:
            raise ValueError("order_id or client_order_id required")
        params: dict[str, Any] = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        else:
            params["origClientOrderId"] = client_order_id
        try:
            row = self._client.request_signed(method="GET", path="/fapi/v1/order", params=params)
        except BinanceNonRetryableError as exc:
            if exc.code == _ORDER_DOES_NOT_EXIST:
                return None
            raise
        if row.get("symbol") != symbol:
            raise ValueError("order symbol mismatch")
        return SimpleNamespace(order_id=str(row["orderId"]), client_order_id=str(row.get("clientOrderId", "")),
                               symbol=row["symbol"], side=row["side"].lower(),
                               state=_ORDER_STATES.get(row["status"], row["status"].lower()),
                               filled_qty=Decimal(str(row.get("executedQty", "0"))),
                               avg_price=Decimal(str(row.get("avgPrice", "0"))))

    def cancel_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        self._client.request_signed(method="DELETE", path="/fapi/v1/order",
                                    params={"symbol": symbol, "orderId": order_id})
        return {"status": "cancel_requested"}

    def get_recent_fills(self, *, symbol: Optional[str] = None, order_id: str = "",
                         since_ms: int = 0, limit: int = 100) -> Tuple[CanonicalFill, ...]:
        """Trade history; errors propagate (missing history keeps orders unresolved)."""
        if not symbol:
            raise ValueError("symbol is required for Binance trade history")
        params: dict[str, Any] = {"symbol": symbol, "limit": min(int(limit), 1000)}
        if order_id:
            params["orderId"] = order_id
        if since_ms:
            params["startTime"] = since_ms
        data = self._client.request_signed(method="GET", path="/fapi/v1/userTrades", params=params)
        if not isinstance(data, list):
            raise RuntimeError("Invalid Binance trade history")
        return tuple(CanonicalFill(
            venue="binance", symbol=f.get("symbol", ""), order_id=str(f.get("orderId", "")),
            trade_id=str(f.get("id", "")), fill_id=str(f.get("id", "")),
            side="buy" if f.get("side", "").upper() == "BUY" else "sell",
            qty=Decimal(str(f.get("qty", "0"))), price=Decimal(str(f.get("price", "0"))),
            fee=Decimal(str(f.get("commission", "0"))), fee_asset=f.get("commissionAsset"),
            liquidity="maker" if f.get("maker") else "taker", ts_ms=int(f.get("time", 0)),
        ) for f in data)

    # ------------------------------------------------------------------
    # Protective stops (Algo service)
    # ------------------------------------------------------------------
    def place_stop_market(self, symbol: str, side: str, trigger_price: Decimal, *,
                          client_algo_id: str) -> dict[str, Any]:
        """Close-all STOP_MARKET on mark price; returns submitted/rejected/error like orders."""
        price = self.filters(symbol).floor_price(Decimal(str(trigger_price)))
        if price <= 0:
            return {"status": "rejected", "msg": "invalid trigger price"}
        params = {"algoType": "CONDITIONAL", "symbol": symbol, "side": side.upper(), "type": "STOP_MARKET",
                  "triggerPrice": _fmt(price), "closePosition": "true", "workingType": "MARK_PRICE",
                  "priceProtect": "true", "clientAlgoId": client_algo_id}
        try:
            data = self._client.request_signed(method="POST", path="/fapi/v1/algoOrder", params=params)
        except (BinanceNonRetryableError, BinanceRateLimitError) as exc:
            return {"status": "rejected", "code": getattr(exc, "code", None), "msg": str(exc)}
        except Exception as exc:
            return {"status": "error", "msg": str(exc)}
        return {"status": "submitted", "algoId": str(data.get("algoId", ""))}

    def get_algo_order(self, *, client_algo_id: str) -> Any:
        try:
            row = self._client.request_signed(method="GET", path="/fapi/v1/algoOrder",
                                              params={"clientAlgoId": client_algo_id})
        except BinanceNonRetryableError as exc:
            if exc.code in _ALGO_DOES_NOT_EXIST:
                return None
            raise
        return SimpleNamespace(algo_id=str(row.get("algoId", "")), client_algo_id=str(row.get("clientAlgoId", "")),
                               symbol=row.get("symbol", ""), side=str(row.get("side", "")).lower(),
                               status=str(row.get("algoStatus", "")).upper(),
                               trigger_price=Decimal(str(row.get("triggerPrice", "0") or "0")),
                               actual_order_id=str(row.get("actualOrderId", "") or ""))

    def cancel_algo_order(self, *, client_algo_id: str) -> dict[str, Any]:
        self._client.request_signed(method="DELETE", path="/fapi/v1/algoOrder",
                                    params={"clientAlgoId": client_algo_id})
        return {"status": "cancel_requested"}

    # ------------------------------------------------------------------
    # Market data (public)
    # ------------------------------------------------------------------
    def get_ticker(self, symbol: str) -> dict[str, Any]:
        book = self._client.request_public(method="GET", path="/fapi/v1/ticker/bookTicker",
                                           params={"symbol": symbol})
        price = self._client.request_public(method="GET", path="/fapi/v1/ticker/price", params={"symbol": symbol})
        return {"symbol": symbol, "lastPrice": float(price["price"]), "bid1Price": float(book["bidPrice"]),
                "ask1Price": float(book["askPrice"])}

    def get_klines(self, symbol: str, interval: str = "60", limit: int = 200,
                   *, now_ms: int | None = None) -> list[dict[str, Any]]:
        """CLOSED klines only, oldest first; ``start`` is the open time in ms."""
        bn_interval = _INTERVAL_MAP.get(interval, interval)
        step = _INTERVAL_MS[bn_interval]
        now_ms = int(time.time() * 1000) if now_ms is None else now_ms
        last_open = now_ms // step * step - step  # most recent fully closed bar
        rows: dict[int, list[Any]] = {}
        end = last_open + step - 1
        while len(rows) < limit:
            batch = min(1500, limit - len(rows))
            data = self._client.request_public(method="GET", path="/fapi/v1/klines", params={
                "symbol": symbol, "interval": bn_interval, "limit": batch, "endTime": end})
            if not isinstance(data, list) or not data:
                break
            for k in data:
                open_time = int(k[0])
                if open_time <= last_open and int(k[6]) < now_ms:
                    rows[open_time] = k
            oldest = int(data[0][0])
            if len(data) < batch or oldest - 1 >= end:
                break
            end = oldest - 1
        bars = [{"start": t, "time": t // 1000, "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                 "close": float(k[4]), "volume": float(k[5]), "quote_volume": float(k[7]), "trades": int(k[8]),
                 "taker_buy_volume": float(k[9]), "taker_buy_quote_volume": float(k[10]), "confirm": True}
                for t, k in sorted(rows.items())]
        return bars[-limit:]

    def get_last_funding(self, symbol: str) -> tuple[int, float]:
        """Most recent SETTLED funding event (time ms, rate)."""
        data = self._client.request_public(method="GET", path="/fapi/v1/fundingRate",
                                           params={"symbol": symbol, "limit": 1})
        if not isinstance(data, list) or not data:
            raise RuntimeError("funding history unavailable")
        return int(data[-1]["fundingTime"]), float(data[-1]["fundingRate"])

    def get_funding_history(self, symbol: str, *, limit: int = 1000) -> list[tuple[int, float]]:
        """Settled funding events (time ms, rate), oldest first, up to ``limit`` (paginated)."""
        rows: dict[int, float] = {}
        end = None
        while len(rows) < limit:
            params: dict[str, Any] = {"symbol": symbol, "limit": min(1000, limit - len(rows))}
            if end is not None:
                params["endTime"] = end
            data = self._client.request_public(method="GET", path="/fapi/v1/fundingRate", params=params)
            if not isinstance(data, list) or not data:
                break
            for row in data:
                rows[int(row["fundingTime"])] = float(row["fundingRate"])
            oldest = min(int(r["fundingTime"]) for r in data)
            if len(data) < params["limit"] or (end is not None and oldest - 1 >= end):
                break
            end = oldest - 1
        return sorted(rows.items())[-limit:]

    def get_positioning(self, symbol: str, *, limit: int = 3) -> list[dict[str, Any]]:
        """Recent 5-minute OI and long/short snapshots, aligned by timestamp."""
        def series(path: str, key: str) -> dict[int, float]:
            data = self._client.request_public(method="GET", path=path,
                                               params={"symbol": symbol, "period": "5m", "limit": limit})
            if not isinstance(data, list):
                raise RuntimeError(f"invalid response from {path}")
            return {int(row["timestamp"]): float(row[key]) for row in data}
        oi = series("/futures/data/openInterestHist", "sumOpenInterest")
        ls = series("/futures/data/globalLongShortAccountRatio", "longShortRatio")
        top = series("/futures/data/topLongShortPositionRatio", "longShortRatio")
        return [{"timestamp": t, "open_interest": oi[t], "ls_ratio": ls.get(t), "top_trader_ls_ratio": top.get(t)}
                for t in sorted(oi)]

    def get_account_info(self) -> dict[str, Any]:
        bal = self.get_balances()
        result: dict[str, Any] = {"venue": "binance", "is_demo": self._config.is_demo}
        for b in bal.balances:
            result[f"{b.asset}_free"] = float(b.free)
            result[f"{b.asset}_locked"] = float(b.locked)
            result[f"{b.asset}_total"] = float(b.total)
        return result
