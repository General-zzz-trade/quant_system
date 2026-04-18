"""OKX USDT SWAP adapter — REST wrapper implementing the shared venue API.

Target surface (parallel to BinanceAdapter / BybitAdapter):
- connect() → bool
- get_balances() → BalanceSnapshot
- get_positions(symbol="") → tuple[VenuePosition, ...]
- send_market_order(symbol, side, qty, reduce_only=False) → dict
- send_limit_order(symbol, side, qty, price, ...) → dict
- cancel_order(symbol, order_id)
- cancel_all(symbol)
- close_position(symbol)
- get_recent_fills(symbol="")
- get_kline(symbol, interval, limit)
- set_leverage(symbol, leverage)

Internal naming (BTCUSDT) is used at the API surface; the adapter
translates to/from OKX instId (BTC-USDT-SWAP) internally.  Quantities
are accepted in **coin units** and converted to contracts via
`symbol_map.coin_to_contracts` before hitting the wire.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Optional

from execution.adapters.okx.config import OkxConfig
from execution.adapters.okx.rest import (
    OkxNonRetryableError,
    OkxRestClient,
    OkxRestConfig,
    OkxRetryableError,
)
from execution.adapters.okx.symbol_map import (
    InstrumentMeta,
    coin_to_contracts,
    contracts_to_coin,
    from_okx_symbol,
    round_price_to_tick,
    to_okx_symbol,
)
from execution.models.balances import BalanceSnapshot, CanonicalBalance
from execution.models.fills import CanonicalFill
from execution.models.positions import VenuePosition

logger = logging.getLogger(__name__)


# Internal interval → OKX bar name
_INTERVAL_MAP = {
    "1": "1m", "5": "5m", "15": "15m",
    "60": "1H", "240": "4H", "D": "1D",
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1h": "1H", "4h": "4H", "1d": "1D",
}


class OkxAdapter:
    """OKX USDT SWAP venue adapter."""

    venue: str = "okx"

    def __init__(self, config: OkxConfig) -> None:
        self._config = config
        rest_cfg = OkxRestConfig(
            api_key=config.api_key,
            api_secret=config.api_secret,
            passphrase=config.passphrase,
            base_url=config.base_url,
            simulated=config.simulated,
            timeout_s=config.timeout_s,
        )
        self._client = OkxRestClient(rest_cfg)
        self._instruments: dict[str, InstrumentMeta] = {}  # keyed by internal symbol
        self._connected = False

    # ------------------------------------------------------------------
    # Connection + instrument bootstrap
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        """Probe the account + fetch instrument meta for mapped symbols.

        Returns True on success.  Fails fast on auth errors so systemd
        restart policy can escalate.
        """
        try:
            # 1. Ping public endpoint — verifies connectivity without auth
            self._client.request_public(method="GET", path="/api/v5/public/time")

            # 2. Fetch instrument metadata for every mapped symbol
            from execution.adapters.okx.symbol_map import _SYMBOL_TO_OKX
            for internal_sym, inst_id in _SYMBOL_TO_OKX.items():
                resp = self._client.request_public(
                    method="GET",
                    path="/api/v5/public/instruments",
                    params={"instType": "SWAP", "instId": inst_id},
                )
                data = resp.get("data") or []
                if not data:
                    logger.error("OKX: no instrument data for %s", inst_id)
                    return False
                self._instruments[internal_sym] = InstrumentMeta.from_api(data[0])
                logger.info(
                    "OKX instrument %s: ctVal=%s %s lotSz=%s tickSz=%s lever=%d",
                    inst_id,
                    self._instruments[internal_sym].ct_val,
                    self._instruments[internal_sym].ct_val_ccy,
                    self._instruments[internal_sym].lot_sz,
                    self._instruments[internal_sym].tick_sz,
                    self._instruments[internal_sym].max_lever,
                )

            # 3. Signed: account balance (verifies auth works)
            bal = self._client.request_signed(
                method="GET",
                path="/api/v5/account/balance",
                params={"ccy": "USDT"},
            )
            data = bal.get("data") or []
            if data:
                details = data[0].get("details", [])
                for d in details:
                    if d.get("ccy") == "USDT":
                        logger.info(
                            "OKX connected: USDT eq=%s avail=%s",
                            d.get("eq"), d.get("availBal"),
                        )
                        break

            # 4. Set leverage on each tradeable instrument to the value
            #    configured by strategy.config.OKX_LEVERAGE (default 10).
            #    Must happen BEFORE any orders; otherwise OKX uses whatever
            #    leverage was last set (possibly 3x from the UI) and the
            #    backtest sizing assumptions won't match exchange reality.
            #    Best-effort — logged and swallowed so a set_leverage
            #    failure doesn't block trading (account may already be
            #    at the target leverage).
            try:
                import os as _os
                target_lev = int(float(_os.environ.get("OKX_LEVERAGE", "10")))
                from execution.adapters.okx.symbol_map import _SYMBOL_TO_OKX
                for internal_sym in _SYMBOL_TO_OKX.keys():
                    try:
                        self.set_leverage(internal_sym, target_lev)
                        logger.info("OKX set_leverage %s → %dx", internal_sym, target_lev)
                    except Exception as e:
                        logger.warning(
                            "OKX set_leverage failed for %s: %s (using account default)",
                            internal_sym, e,
                        )
            except Exception:
                logger.debug("OKX leverage init skipped", exc_info=True)

            self._connected = True
            return True
        except OkxNonRetryableError as e:
            logger.error("OKX auth/business error: %s", e)
            return False
        except Exception as e:
            logger.error("OKX connection failed: %s", e)
            return False

    def is_connected(self) -> bool:
        return self._connected

    def _get_meta(self, internal_symbol: str) -> InstrumentMeta:
        key = internal_symbol.upper()
        # Strip timeframe suffix
        for suffix in ("_4H", "_4h", "_1H", "_1h", "_15M", "_15m"):
            if key.endswith(suffix.upper()):
                key = key[: -len(suffix)]
                break
        meta = self._instruments.get(key)
        if meta is None:
            raise RuntimeError(
                f"OKX: instrument metadata not loaded for {internal_symbol}"
            )
        return meta

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------
    def get_balances(self) -> BalanceSnapshot:
        try:
            resp = self._client.request_signed(
                method="GET",
                path="/api/v5/account/balance",
                params={"ccy": "USDT"},
            )
            balances: list[CanonicalBalance] = []
            data = resp.get("data") or []
            if data:
                for d in data[0].get("details", []):
                    ccy = d.get("ccy", "")
                    eq = Decimal(str(d.get("eq") or "0"))
                    avail = Decimal(str(d.get("availBal") or "0"))
                    if eq > 0 or avail > 0:
                        locked = eq - avail if eq > avail else Decimal("0")
                        balances.append(
                            CanonicalBalance.from_free_locked(
                                venue=self.venue,
                                asset=ccy,
                                free=avail,
                                locked=locked,
                            )
                        )
            return BalanceSnapshot(venue=self.venue, balances=tuple(balances))
        except Exception as e:
            logger.error("OKX get_balances failed: %s", e)
            return BalanceSnapshot(venue=self.venue, balances=())

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------
    def get_positions(self, symbol: str = "") -> tuple[VenuePosition, ...]:
        params: dict[str, Any] = {"instType": "SWAP"}
        if symbol:
            try:
                params["instId"] = to_okx_symbol(symbol)
            except KeyError:
                pass

        try:
            resp = self._client.request_signed(
                method="GET",
                path="/api/v5/account/positions",
                params=params,
            )
        except Exception as e:
            logger.error("OKX get_positions failed: %s", e)
            return ()

        out: list[VenuePosition] = []
        for p in resp.get("data") or []:
            inst_id = p.get("instId", "")
            try:
                internal_sym = from_okx_symbol(inst_id)
            except KeyError:
                continue

            pos_contracts = Decimal(str(p.get("pos") or "0"))
            if pos_contracts == 0:
                continue

            meta = self._instruments.get(internal_sym)
            # contracts → coin qty (signed)
            if meta is not None:
                coin_qty = contracts_to_coin(abs(pos_contracts), meta)
                coin_qty = coin_qty if pos_contracts >= 0 else -coin_qty
            else:
                coin_qty = pos_contracts  # fallback (should not happen)

            avg_px = Decimal(str(p.get("avgPx") or "0"))
            out.append(
                VenuePosition(
                    venue=self.venue,
                    symbol=internal_sym,
                    qty=coin_qty,
                    entry_price=avg_px,
                )
            )
        return tuple(out)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def send_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        *,
        reduce_only: bool = False,
    ) -> dict:
        """Market order (coin qty → contracts conversion happens here)."""
        try:
            meta = self._get_meta(symbol)
        except RuntimeError as e:
            return {"status": "error", "msg": str(e)}

        contracts = coin_to_contracts(float(qty), meta)
        if contracts == 0:
            return {
                "status": "error",
                "msg": f"qty {qty} rounds to 0 contracts (below minSz)",
            }

        # Safety cap: notional USD ceiling (pulled from config)
        # Use last mark/ticker to estimate notional
        try:
            tk = self._client.request_public(
                method="GET",
                path="/api/v5/market/ticker",
                params={"instId": meta.inst_id},
            )
            px = float((tk.get("data") or [{}])[0].get("last", 0))
        except Exception:
            px = 0.0
        coin_actual = float(contracts_to_coin(contracts, meta))
        notional = px * coin_actual
        if (
            not reduce_only
            and self._config.max_order_notional_usd > 0
            and notional > self._config.max_order_notional_usd
        ):
            logger.warning(
                "OKX order REJECTED (notional cap): %s %s qty=%s coin=%.6f "
                "notional=$%.2f > cap=$%.2f",
                symbol, side, qty, coin_actual, notional,
                self._config.max_order_notional_usd,
            )
            return {
                "status": "error",
                "msg": f"notional ${notional:.2f} > cap ${self._config.max_order_notional_usd}",
            }

        body = {
            "instId": meta.inst_id,
            "tdMode": "cross",         # cross-margin SWAP
            "side": side.lower(),      # "buy" / "sell"
            "ordType": "market",
            "sz": str(contracts),
        }
        if reduce_only:
            body["reduceOnly"] = "true"

        logger.info(
            "OKX send_market_order: %s %s coin=%.6f contracts=%s notional≈$%.2f",
            meta.inst_id, side, coin_actual, contracts, notional,
        )

        try:
            resp = self._client.request_signed(
                method="POST",
                path="/api/v5/trade/order",
                body=body,
            )
        except OkxNonRetryableError as e:
            logger.error("OKX order rejected: %s", e)
            return {"status": "error", "msg": str(e)}
        except OkxRetryableError as e:
            logger.warning("OKX order transient error: %s", e)
            return {"status": "error", "msg": str(e), "retryable": True}

        data = resp.get("data") or []
        if not data:
            return {"status": "error", "msg": "empty data in response"}
        row = data[0]
        if row.get("sCode") and row["sCode"] != "0":
            return {
                "status": "error",
                "code": row.get("sCode"),
                "msg": row.get("sMsg", ""),
            }
        return {
            "status": "submitted",
            "orderId": row.get("ordId", ""),
            "clientOrderId": row.get("clOrdId", ""),
        }

    def send_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        *,
        tif: str = "GTC",
        reduce_only: bool = False,
        post_only: bool = False,
    ) -> dict:
        try:
            meta = self._get_meta(symbol)
        except RuntimeError as e:
            return {"status": "error", "msg": str(e)}

        contracts = coin_to_contracts(float(qty), meta)
        if contracts == 0:
            return {"status": "error", "msg": "qty below minSz"}

        tick_price = round_price_to_tick(float(price), meta)

        ord_type = "post_only" if post_only else "limit"
        body = {
            "instId": meta.inst_id,
            "tdMode": "cross",
            "side": side.lower(),
            "ordType": ord_type,
            "sz": str(contracts),
            "px": str(tick_price),
        }
        if reduce_only:
            body["reduceOnly"] = "true"

        try:
            resp = self._client.request_signed(
                method="POST",
                path="/api/v5/trade/order",
                body=body,
            )
        except OkxNonRetryableError as e:
            return {"status": "error", "msg": str(e)}
        data = resp.get("data") or []
        if not data:
            return {"status": "error", "msg": "empty data"}
        row = data[0]
        if row.get("sCode") and row["sCode"] != "0":
            return {"status": "error", "code": row.get("sCode"), "msg": row.get("sMsg", "")}
        return {"status": "submitted", "orderId": row.get("ordId", "")}

    def get_open_orders(self, *, symbol: str = "") -> tuple:
        """Get pending orders for a symbol (or all if not specified).

        Returns tuple of objects with .order_id, .symbol, .side, .qty,
        .filled_qty, .price attrs — duck-typed shape that matches what
        runner.limit_order_manager.check_fill expects.

        Without this, LimitOrderManager.check_fill silently raises
        AttributeError on every call (the previous behavior — only Bybit
        implemented get_open_orders), causing pre-placed limit orders to
        never be detected as filled. The result was decide() opening
        market orders on top of already-filled limits, producing the
        oversize positions seen on 2026-04-13 (ETH 0.353→0.817) and
        2026-04-18 (ETH -1.59 → -9.7 contracts).
        """
        params: dict[str, Any] = {"instType": "SWAP"}
        if symbol:
            try:
                params["instId"] = to_okx_symbol(symbol)
            except KeyError:
                return ()
        try:
            resp = self._client.request_signed(
                method="GET",
                path="/api/v5/trade/orders-pending",
                params=params,
            )
        except Exception as e:
            logger.warning("OKX get_open_orders failed: %s", e)
            return ()

        out = []
        for o in resp.get("data") or []:
            inst_id = o.get("instId", "")
            try:
                internal_sym = from_okx_symbol(inst_id)
            except KeyError:
                continue
            # OKX sz/fillSz are in contracts; convert to coin via ctVal
            meta = self._instruments.get(internal_sym)
            sz = Decimal(str(o.get("sz") or "0"))
            fillSz = Decimal(str(o.get("accFillSz") or o.get("fillSz") or "0"))
            qty_coin = contracts_to_coin(sz, meta) if meta else sz
            filled_coin = contracts_to_coin(fillSz, meta) if meta else fillSz
            # Build a lightweight object so check_fill's getattr-based access works
            from types import SimpleNamespace
            out.append(SimpleNamespace(
                order_id=o.get("ordId", ""),
                client_order_id=o.get("clOrdId", ""),
                symbol=internal_sym,
                side=o.get("side", ""),
                qty=qty_coin,
                filled_qty=filled_coin,
                price=Decimal(str(o.get("px") or "0")),
                state=o.get("state", ""),
            ))
        return tuple(out)

    def cancel_order(self, symbol: str, order_id: str) -> dict:
        try:
            meta = self._get_meta(symbol)
        except RuntimeError as e:
            return {"status": "error", "msg": str(e)}
        body = {"instId": meta.inst_id, "ordId": order_id}
        try:
            resp = self._client.request_signed(
                method="POST",
                path="/api/v5/trade/cancel-order",
                body=body,
            )
        except Exception as e:
            return {"status": "error", "msg": str(e)}
        data = resp.get("data") or [{}]
        row = data[0]
        if row.get("sCode") and row["sCode"] != "0":
            return {"status": "error", "msg": row.get("sMsg", "")}
        return {"status": "canceled"}

    def cancel_all(self, symbol: str = "") -> dict:
        """OKX doesn't have a single cancel-all endpoint per symbol. Iterate."""
        if not symbol:
            return {"status": "error", "msg": "symbol required"}
        try:
            meta = self._get_meta(symbol)
        except RuntimeError as e:
            return {"status": "error", "msg": str(e)}
        try:
            resp = self._client.request_signed(
                method="GET",
                path="/api/v5/trade/orders-pending",
                params={"instId": meta.inst_id, "instType": "SWAP"},
            )
        except Exception as e:
            return {"status": "error", "msg": str(e)}
        orders = resp.get("data") or []
        cancelled = 0
        for o in orders:
            ord_id = o.get("ordId")
            if ord_id:
                r = self.cancel_order(symbol, ord_id)
                if r.get("status") == "canceled":
                    cancelled += 1
        return {"status": "canceled", "count": cancelled}

    def close_position(self, symbol: str) -> dict:
        """Flatten a SWAP position using reduce_only market order."""
        positions = self.get_positions(symbol=symbol)
        for pos in positions:
            if pos.symbol == symbol and not pos.is_flat:
                side = "sell" if pos.is_long else "buy"
                qty = float(pos.abs_qty)
                return self.send_market_order(
                    symbol, side, qty, reduce_only=True,
                )
        return {"status": "no_position"}

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------
    def get_recent_fills(
        self,
        symbol: str = "",
        limit: int = 10,
    ) -> tuple[CanonicalFill, ...]:
        params: dict[str, Any] = {"instType": "SWAP", "limit": str(limit)}
        if symbol:
            try:
                params["instId"] = to_okx_symbol(symbol)
            except KeyError:
                pass
        try:
            resp = self._client.request_signed(
                method="GET",
                path="/api/v5/trade/fills",
                params=params,
            )
        except Exception as e:
            logger.warning("OKX get_recent_fills failed: %s", e)
            return ()

        out: list[CanonicalFill] = []
        for f in resp.get("data") or []:
            inst_id = f.get("instId", "")
            try:
                internal_sym = from_okx_symbol(inst_id)
                meta = self._instruments.get(internal_sym)
            except KeyError:
                continue
            if meta is None:
                continue
            contracts = Decimal(str(f.get("fillSz") or "0"))
            coin_qty = contracts_to_coin(contracts, meta)
            side = f.get("side", "").lower()
            order_id = str(f.get("ordId") or "")
            trade_id = str(f.get("tradeId") or f.get("billId") or "")
            out.append(
                CanonicalFill(
                    venue=self.venue,
                    symbol=internal_sym,
                    order_id=order_id,
                    trade_id=trade_id,
                    fill_id=trade_id or order_id,  # OKX tradeId is globally unique
                    side=side,
                    qty=coin_qty,
                    price=Decimal(str(f.get("fillPx") or "0")),
                    fee=Decimal(str(f.get("fee") or "0")),
                    fee_asset=f.get("feeCcy"),
                    ts_ms=int(f.get("ts") or 0),
                )
            )
        return tuple(out)

    # ------------------------------------------------------------------
    # Kline (REST fallback)
    # ------------------------------------------------------------------
    def get_klines(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 300,
    ) -> list[dict]:
        """Historical klines via REST for warmup/backfill.

        Auto-paginates when `limit` > 300 (OKX per-request max). Uses
        `/api/v5/market/candles` for the recent window and
        `/api/v5/market/history-candles` for older bars. The method name
        matches Binance/Bybit adapters for cross-venue compatibility.

        Returns list of dicts in the shared format (oldest-first):
            {"time": int(sec), "open":..., "high":..., "low":..., "close":...,
             "volume":..., "turnover":..., "confirm": True}
        """
        try:
            meta = self._get_meta(symbol)
        except RuntimeError:
            return []
        okx_bar = _INTERVAL_MAP.get(str(interval), "1H")
        _PAGE_MAX = 300

        # OKX returns newest-first rows. We accumulate them, then reverse
        # once at the end so callers see oldest-first order.
        rows: list[list[str]] = []
        seen_ts: set[int] = set()
        remaining = limit
        after_ts: Optional[int] = None

        # First call hits /market/candles (recent ~1440 bars). If the caller
        # wants more than that, we switch to /market/history-candles for
        # older bars automatically.
        path = "/api/v5/market/candles"
        while remaining > 0:
            batch = min(remaining, _PAGE_MAX)
            params: dict[str, Any] = {
                "instId": meta.inst_id,
                "bar": okx_bar,
                "limit": str(batch),
            }
            if after_ts is not None:
                params["after"] = str(after_ts)
            try:
                resp = self._client.request_public(
                    method="GET",
                    path=path,
                    params=params,
                )
            except Exception as e:
                logger.warning("OKX get_klines(%s) failed: %s", path, e)
                break

            page = resp.get("data") or []
            if not page:
                break

            fresh = 0
            for row in page:
                if len(row) < 9:
                    continue
                ts = int(row[0])
                if ts in seen_ts:
                    continue
                seen_ts.add(ts)
                rows.append(row)
                fresh += 1

            if fresh == 0:
                break

            # OKX /market/candles is newest-first; the last row is oldest.
            oldest_ts = int(page[-1][0])
            after_ts = oldest_ts  # next page fetches bars strictly older
            remaining -= fresh

            # /market/candles only serves the recent ~1440 bars. For deeper
            # history we switch to /market/history-candles. Triggered once.
            if path == "/api/v5/market/candles" and remaining > 0 and fresh < batch:
                path = "/api/v5/market/history-candles"
                continue

        # Reverse to oldest-first so feature engine accumulates correctly
        out: list[dict] = []
        for row in reversed(rows):
            confirm = str(row[8]) == "1"
            out.append({
                "time":    int(row[0]) // 1000,
                "open":    float(row[1]),
                "high":    float(row[2]),
                "low":     float(row[3]),
                "close":   float(row[4]),
                "volume":  float(row[5]),
                "turnover": float(row[7]),
                "confirm": confirm,
            })
        return out

    # ------------------------------------------------------------------
    # Leverage
    # ------------------------------------------------------------------
    def set_leverage(self, symbol: str, leverage: int) -> dict:
        try:
            meta = self._get_meta(symbol)
        except RuntimeError as e:
            return {"status": "error", "msg": str(e)}
        body = {
            "instId": meta.inst_id,
            "lever": str(leverage),
            "mgnMode": "cross",
        }
        try:
            resp = self._client.request_signed(
                method="POST",
                path="/api/v5/account/set-leverage",
                body=body,
            )
            data = resp.get("data") or []
            if data and data[0].get("lever"):
                return {"status": "ok", "lever": data[0]["lever"]}
            return {"status": "error", "msg": str(resp)}
        except Exception as e:
            return {"status": "error", "msg": str(e)}
