"""Causal hourly Binance USD-M open-interest and global account-ratio cache.

Matches the existing historical downloaders: ``sumOpenInterest`` is the base
asset quantity (not its quote value), and ``longShortRatio`` is the global
account ratio (not top-trader positions). No credentials or exchange writes.
HTTP response fixtures validate the documented shape; online acceptance is a
separate check. Cache failures never turn retained observations into ready data.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://fapi.binance.com"
OI_PATH = "/futures/data/openInterestHist"
LS_PATH = "/futures/data/globalLongShortAccountRatio"
HISTORY_POINTS = 8
EVENT_TTL_MS = 3_900_000
RECEIPT_TTL_MS = 120_000


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _milliseconds(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("Observation timestamps must be positive integer epoch milliseconds")
    return value


class BinanceOICache:
    """Refresh a coherent OI/LS pair; expose readiness separately from values.

    ``fetch(path, params)`` returns decoded public API JSON and can be injected
    for deterministic tests. ``clock`` returns UTC epoch seconds. ``get`` never
    performs I/O and only exposes numeric inputs when the whole pair is ready.
    ``get(as_of_ms=...)`` selects observations at or before a bar's causal time;
    receipt freshness and the most recent observation are still checked at now.
    """

    def __init__(
        self, symbol: str, *, fetch: Callable[[str, dict[str, Any]], Any] | None = None,
        clock: Callable[[], float] = time.time, refresh_seconds: float = 55.0,
        request_timeout: float = 5.0, event_ttl_ms: int = EVENT_TTL_MS,
        receipt_ttl_ms: int = RECEIPT_TTL_MS,
    ) -> None:
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9]{2,30}", symbol):
            raise ValueError("Expected an explicit uppercase Binance symbol")
        self.symbol = symbol
        self._clock = clock
        self._interval = _positive(refresh_seconds, "refresh interval")
        self._timeout = _positive(request_timeout, "request timeout")
        self._event_ttl = _milliseconds(event_ttl_ms)
        self._receipt_ttl = _milliseconds(receipt_ttl_ms)
        self._fetch = fetch or self._fetch_public
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._points: dict[str, list[tuple[int, float]]] = {"open_interest": [], "ls_ratio": []}
        self._received: int | None = None
        self._status, self._reason = "error", "not_initialized"

    def _now_ms(self) -> int:
        return int(_positive(self._clock(), "clock") * 1000)

    def _fetch_public(self, path: str, params: dict[str, Any]) -> Any:
        request = Request(BASE_URL + path + "?" + urlencode(params), headers={"Accept": "application/json"})
        with urlopen(request, timeout=self._timeout) as response:
            # Public endpoints are bounded to eight observations; reject a wrong
            # upstream response rather than reading an unbounded error document.
            payload = response.read(1_048_577)
        if len(payload) > 1_048_576:
            raise ValueError("Public data response exceeded size limit")
        return json.loads(payload)

    def _parse(self, payload: Any, field: str, now: int) -> list[tuple[int, float]]:
        if not isinstance(payload, list) or not 1 <= len(payload) <= HISTORY_POINTS:
            raise ValueError("Expected a nonempty bounded historical observation array")
        points = []
        for row in payload:
            if not isinstance(row, dict) or row.get("symbol") != self.symbol:
                raise ValueError("Observation symbol does not match the requested symbol")
            timestamp = _milliseconds(row.get("timestamp"))
            if timestamp > now:
                raise ValueError("Observation timestamp is in the future")
            if points and timestamp <= points[-1][0]:
                raise ValueError("Observation timestamps must increase strictly")
            points.append((timestamp, _positive(row.get(field), field)))
        return points

    def refresh(self) -> bool:
        """Perform one bounded refresh; errors revoke readiness until recovery."""
        with self._refresh_lock:
            if self._stop.is_set():
                return False
            try:
                started = self._now_ms()
                params = {"symbol": self.symbol, "period": "1h", "limit": HISTORY_POINTS, "endTime": started}
                oi = self._parse(self._fetch(OI_PATH, dict(params)), "sumOpenInterest", started)
                ls = self._parse(self._fetch(LS_PATH, dict(params)), "longShortRatio", started)
                received = self._now_ms()
                if received < started:
                    raise ValueError("Clock moved backwards during refresh")
                proposed = {"open_interest": oi, "ls_ratio": ls}
                with self._lock:
                    for name, points in proposed.items():
                        previous = dict(self._points[name])
                        if previous and points[-1][0] < max(previous):
                            raise ValueError("Latest observation regressed")
                        if any(ts in previous and previous[ts] != value for ts, value in points):
                            raise ValueError("Previously observed timestamp has conflicting values")
                        previous.update(points)
                        proposed[name] = sorted(previous.items())[-HISTORY_POINTS:]
                    if self._stop.is_set():
                        return False
                    # Publish the pair atomically only after both responses pass.
                    self._points = proposed
                    self._received = received
                    self._status, self._reason = "ready", ""
                return self.get()["status"] == "ready"
            except Exception as exc:
                with self._lock:
                    self._status = "error"
                    # Avoid logging arbitrary response bodies or transport URLs.
                    self._reason = "refresh_failed:" + type(exc).__name__
                return False

    def get(self, as_of_ms: int | None = None) -> dict[str, Any]:
        """Return a fresh causal pair or NaN values plus an explicit reason."""
        with self._lock:
            status, reason = self._status, self._reason
            result: dict[str, Any] = {
                "status": status, "reason": reason, "symbol": self.symbol,
                "source": "binance-usdm", "period": "1h",
                "open_interest": math.nan, "ls_ratio": math.nan,
                "oi_event_ts_ms": None, "ls_event_ts_ms": None,
                "received_at_ms": self._received, "checked_at_ms": None,
                "as_of_ms": as_of_ms,
            }
            try:
                now = self._now_ms()
                at = now if as_of_ms is None else _milliseconds(as_of_ms)
                result.update(checked_at_ms=now, as_of_ms=at)
                if at > now:
                    raise ValueError("Causal query timestamp is in the future")
                if status != "ready":
                    return result
                if self._received is None or now < self._received:
                    raise ValueError("Clock precedes the latest receipt")
                if now - self._received > self._receipt_ttl:
                    result.update(status="stale", reason="receipt_expired")
                    return result
                selected = {}
                for name, points in self._points.items():
                    if not points or now - points[-1][0] > self._event_ttl:
                        result.update(status="stale", reason=name + "_latest_expired")
                        return result
                    candidates = [(ts, value) for ts, value in points if ts <= at]
                    if not candidates or at - candidates[-1][0] > self._event_ttl:
                        result.update(status="stale", reason=name + "_causal_observation_unavailable")
                        return result
                    selected[name] = candidates[-1]
                result.update(open_interest=selected["open_interest"][1], ls_ratio=selected["ls_ratio"][1],
                              oi_event_ts_ms=selected["open_interest"][0], ls_event_ts_ms=selected["ls_ratio"][0])
                return result
            except (ValueError, TypeError, OverflowError):
                result.update(status="error", reason="invalid_clock_or_query")
                return result

    def _run(self) -> None:
        while not self._stop.is_set():
            self.refresh()
            self._stop.wait(self._interval)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self._stop.is_set():
                    raise RuntimeError("Previous cache worker has not stopped")
                return
            self._stop.clear()
            self._status, self._reason = "error", "waiting_for_refresh"
            self._thread = threading.Thread(target=self._run, name=f"oi-ls-{self.symbol}", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._status, self._reason = "error", "stopped"
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2 * self._timeout + 1)
            if thread.is_alive():
                raise RuntimeError("Cache worker did not stop within its request timeout budget")
