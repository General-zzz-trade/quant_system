"""Bounded public GET market data for account-free shadow observations.

No account configuration, authentication, order methods or worker threads are
provided. ``fetch(Request, timeout) -> bytes`` is injectable for offline tests.
The normal transport preserves system proxy/CA settings and refuses redirects.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
import math
import re
import time
from typing import Any, Callable, Iterable
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener


BYBIT_HOST = "https://api.bybit.com"
BINANCE_HOST = "https://fapi.binance.com"
KLINE_PATH = "/v5/market/kline"
TICKER_PATH = "/v5/market/tickers"
OI_PATH = "/futures/data/openInterestHist"
LS_PATH = "/futures/data/globalLongShortAccountRatio"
MAX_RESPONSE_BYTES = 1_048_576
MAX_CLOCK_SKEW_MS = 30_000
_MIN_EPOCH_MS = 1_000_000_000_000
_INTERVALS = {"1", "3", "5", "15", "30", "60", "120", "240", "360", "720"}
_PARAMETERS = {
    (BYBIT_HOST, KLINE_PATH): {"category", "symbol", "interval", "limit", "end"},
    (BYBIT_HOST, TICKER_PATH): {"category", "symbol"},
    (BINANCE_HOST, OI_PATH): {"symbol", "period", "limit", "endTime"},
    (BINANCE_HOST, LS_PATH): {"symbol", "period", "limit", "endTime"},
}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _positive(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid public market " + name)
    return float(value)


def _milliseconds(value: Any, name: str) -> int:
    if isinstance(value, str) and re.fullmatch(r"[0-9]{13,16}", value):
        value = int(value)
    if type(value) is not int or not _MIN_EPOCH_MS <= value <= 9_999_999_999_999_999:
        raise ValueError("Expected epoch milliseconds for " + name)
    return value


def _decimal(value: Any, name: str, *, nonnegative: bool = False, signed: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("Invalid public market " + name)
    try:
        number = Decimal(str(value))
        finite = number.is_finite() and math.isfinite(float(number))
    except (ValueError, InvalidOperation, OverflowError) as exc:
        raise ValueError("Invalid public market " + name) from exc
    if not finite or (not signed and (number < 0 if nonnegative else number <= 0)):
        raise ValueError("Invalid public market " + name)
    return number


def _decode(raw: bytes) -> Any:
    if not isinstance(raw, bytes) or len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Public market response must be bounded bytes")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate public market JSON key")
            result[key] = value
        return result

    def floating(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Non-finite public market JSON number")
        return result

    def constant(_):
        raise ValueError("Non-finite public market JSON number")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_float=floating, parse_constant=constant)
    except UnicodeError as exc:
        raise ValueError("Public market response is not UTF-8 JSON") from exc


class BybitPublicMarketData:
    """Public linear-market observations; all timestamps returned are epoch-ms."""

    venue = "bybit"

    def __init__(self, symbols: Iterable[str], *, fetch: Callable[[Request, float], bytes] | None = None,
                 clock: Callable[[], float] = time.time, request_timeout: float = 5.0) -> None:
        if isinstance(symbols, (str, bytes)):
            raise ValueError("Provide an explicit collection of public symbols")
        selected = tuple(symbols)
        if (not selected or len(selected) > 32
                or any(not isinstance(s, str) or not re.fullmatch(r"[A-Z0-9]{2,30}", s) for s in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("Public symbols must be unique uppercase market symbols")
        self.symbols = frozenset(selected)
        self._clock = clock
        self._timeout = _positive(request_timeout, "request timeout")
        if self._timeout > 30:
            raise ValueError("Public market request timeout exceeds 30 seconds")
        self._fetch = fetch if fetch is not None else self._fetch_public

    def _now_ms(self) -> int:
        return _milliseconds(int(_positive(self._clock(), "clock") * 1000), "clock")

    def _symbol(self, symbol: str) -> str:
        if not isinstance(symbol, str) or symbol not in self.symbols:
            raise ValueError("Public request symbol is not in the selected run manifest")
        return symbol

    @staticmethod
    def _fetch_public(request: Request, timeout: float) -> bytes:
        with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            if response.status != 200 or response.geturl() != request.full_url:
                raise ValueError("Public market response changed its request destination")
            return response.read(MAX_RESPONSE_BYTES + 1)

    def _request(self, host: str, path: str, params: dict) -> Any:
        if (host, path) not in _PARAMETERS or set(params) != _PARAMETERS[(host, path)]:
            raise ValueError("Only declared public market GET endpoints and parameters are allowed")
        request = Request(host + path + "?" + urlencode(params), method="GET",
                          headers={"Accept": "application/json", "User-Agent": "quant-public-shadow/1"})
        return _decode(self._fetch(request, self._timeout))

    def _bybit(self, path: str, params: dict) -> tuple[dict, int]:
        started = self._now_ms()
        payload = self._request(BYBIT_HOST, path, params)
        received = self._now_ms()
        if received < started:
            raise ValueError("Public market clock moved backwards")
        if (not isinstance(payload, dict) or type(payload.get("retCode")) is not int
                or payload["retCode"] != 0 or not isinstance(payload.get("result"), dict)):
            raise ValueError("Public market response did not report success")
        server_ms = _milliseconds(payload.get("time"), "server time")
        if not started - MAX_CLOCK_SKEW_MS <= server_ms <= received + MAX_CLOCK_SKEW_MS:
            raise ValueError("Public market server time is stale or future")
        result = payload["result"]
        if result.get("category") != "linear":
            raise ValueError("Public market category differs from linear")
        return result, min(server_ms, received)

    def get_klines(self, symbol: str, *, interval: str = "60", limit: int = 200) -> list[dict]:
        symbol = self._symbol(symbol)
        if not isinstance(interval, str) or interval not in _INTERVALS:
            raise ValueError("Public klines require an explicit supported minute interval")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Public kline limit must be between 1 and 1000")
        duration = int(interval) * 60_000
        end = self._now_ms() // duration * duration - 1
        result, observed = self._bybit(KLINE_PATH, {
            "category": "linear", "symbol": symbol, "interval": interval, "limit": limit, "end": end,
        })
        rows = result.get("list")
        if (result.get("symbol") != symbol or not isinstance(rows, list) or not 1 <= len(rows) <= limit):
            raise ValueError("Public kline result has the wrong symbol or observation count")
        points = {}
        for row in rows:
            if not isinstance(row, list) or len(row) != 7:
                raise ValueError("Public kline observation has an invalid shape")
            timestamp = _milliseconds(row[0], "bar time")
            if timestamp % duration or timestamp + duration > min(end + 1, observed):
                raise ValueError("Public kline is not an aligned closed bar")
            opening, high, low, closing = [_decimal(value, "OHLC") for value in row[1:5]]
            volume = _decimal(row[5], "volume", nonnegative=True)
            _decimal(row[6], "turnover", nonnegative=True)
            if not low <= min(opening, closing) <= max(opening, closing) <= high:
                raise ValueError("Public kline OHLC values contradict each other")
            bar = {"time": timestamp, "start": timestamp, "interval": interval, "confirm": True,
                   "open": str(opening), "high": str(high), "low": str(low), "close": str(closing),
                   "volume": str(volume)}
            if timestamp in points and points[timestamp] != bar:
                raise ValueError("Conflicting public observations for the same closed bar")
            points[timestamp] = bar
        times = sorted(points)
        if any(right - left != duration for left, right in zip(times, times[1:])):
            raise ValueError("Public kline history contains a gap")
        return [points[timestamp] for timestamp in reversed(times)]

    def get_closed_bars(self, symbol: str, *, interval: str = "60", limit: int = 200,
                        after_ms: int | None = None) -> list[dict]:
        if after_ms is not None and (type(after_ms) is not int or after_ms < 0):
            raise ValueError("Closed-bar cursor must be a nonnegative integer timestamp")
        rows = self.get_klines(symbol, interval=interval, limit=limit)
        pending = [row for row in reversed(rows) if after_ms is None or row["time"] > after_ms]
        # A bounded request can be internally contiguous yet omit bars since
        # the caller's last observation. Never silently advance across that gap.
        if (after_ms not in (None, 0) and pending
                and pending[0]["time"] != after_ms + int(interval) * 60_000):
            raise ValueError("Public kline window contains a gap after the closed-bar cursor")
        return pending

    def get_ticker(self, symbol: str) -> dict:
        symbol = self._symbol(symbol)
        result, _ = self._bybit(TICKER_PATH, {"category": "linear", "symbol": symbol})
        rows = result.get("list")
        if (not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict)
                or rows[0].get("symbol") != symbol):
            raise ValueError("Public ticker has the wrong symbol or observation count")
        row = rows[0]
        ticker = {"symbol": symbol, "lastPrice": str(_decimal(row.get("lastPrice"), "last price"))}
        if "markPrice" in row:
            ticker["markPrice"] = str(_decimal(row["markPrice"], "mark price"))
        if "fundingRate" in row:
            ticker["fundingRate"] = str(_decimal(row["fundingRate"], "funding rate", signed=True))
        return ticker

    def fetch_oi(self, path: str, params: dict[str, Any]) -> list[dict]:
        """Strict transport for BinanceOICache's existing causal observation parser."""
        if path not in {OI_PATH, LS_PATH} or not isinstance(params, dict):
            raise ValueError("Only the public hourly OI and global account-ratio endpoints are allowed")
        if set(params) != {"symbol", "period", "limit", "endTime"}:
            raise ValueError("Unexpected public OI query parameter")
        symbol = self._symbol(params["symbol"])
        if params["period"] != "1h" or type(params["limit"]) is not int or params["limit"] != 8:
            raise ValueError("Public OI queries must preserve the eight-point hourly contract")
        end = _milliseconds(params["endTime"], "OI query time")
        if end > self._now_ms():
            raise ValueError("Public OI query time is in the future")
        payload = self._request(BINANCE_HOST, path, dict(params))
        if not isinstance(payload, list) or not 1 <= len(payload) <= 8:
            raise ValueError("Public OI response must be a bounded observation list")
        field = "sumOpenInterest" if path == OI_PATH else "longShortRatio"
        result = []
        for row in payload:
            if not isinstance(row, dict) or row.get("symbol") != symbol:
                raise ValueError("Public OI response has the wrong symbol")
            timestamp = _milliseconds(row.get("timestamp"), "OI observation")
            if timestamp > end or (result and timestamp <= result[-1]["timestamp"]):
                raise ValueError("Public OI observations are future, repeated or out of order")
            result.append({"symbol": symbol, "timestamp": timestamp,
                           field: str(_decimal(row.get(field), field))})
        return result
