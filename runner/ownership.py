"""Host-local runtime ownership guard for active Bybit services.

Prevents two live services on the same host from trading the same symbol on the
same Bybit account/environment at the same time.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import fcntl

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_INPROCESS_CLAIMS: set[Path] = set()


def _safe_token(value: str) -> str:
    token = _SANITIZE_RE.sub("_", str(value).strip())
    return token.strip("._") or "unknown"


def _account_scope(adapter: Any) -> str:
    cfg = getattr(adapter, "_config", None)
    if cfg is None:
        raise TypeError("adapter must expose a _config with Bybit connection details")
    base_url = getattr(cfg, "base_url", "")
    account_type = getattr(cfg, "account_type", "")
    category = getattr(cfg, "category", "")
    api_key = getattr(cfg, "api_key", "")
    key_hash = hashlib.sha256(str(api_key).encode("utf-8")).hexdigest()[:12]
    return "|".join((str(base_url), str(account_type), str(category), key_hash))


def _lock_path(*, venue: str, account_scope: str, symbol: str, lock_dir: str) -> Path:
    account_hash = hashlib.sha256(account_scope.encode("utf-8")).hexdigest()[:16]
    filename = f"{_safe_token(venue)}_{account_hash}_{_safe_token(symbol)}.lock"
    return Path(lock_dir) / filename


class RuntimeSymbolLease:
    """Exclusive claim on one or more symbols for a live runtime."""

    def __init__(
        self,
        *,
        service_name: str,
        venue: str,
        account_scope: str,
        symbols: tuple[str, ...],
        lock_dir: str = "data/runtime/locks",
    ) -> None:
        self._service_name = service_name
        self._venue = venue
        self._account_scope = account_scope
        self._symbols = tuple(dict.fromkeys(symbols))
        self._lock_dir = lock_dir
        self._claims: dict[str, tuple[Path, Any]] = {}

    def acquire(self) -> "RuntimeSymbolLease":
        Path(self._lock_dir).mkdir(parents=True, exist_ok=True)
        for symbol in self._symbols:
            path = _lock_path(
                venue=self._venue,
                account_scope=self._account_scope,
                symbol=symbol,
                lock_dir=self._lock_dir,
            )
            if path in _INPROCESS_CLAIMS:
                holder = path.read_text(encoding="utf-8", errors="replace").strip()
                self.release()
                raise RuntimeError(
                    f"runtime symbol already claimed on this host: service={self._service_name} "
                    f"symbol={symbol} holder={holder or 'in-process'}"
                )
            handle = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.seek(0)
                holder = handle.read().strip()
                handle.close()
                self.release()
                raise RuntimeError(
                    f"runtime symbol already claimed: service={self._service_name} "
                    f"symbol={symbol} holder={holder or 'unknown'}"
                ) from None
            metadata = {
                "service": self._service_name,
                "symbol": symbol,
                "pid": os.getpid(),
                "venue": self._venue,
            }
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps(metadata, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
            self._claims[symbol] = (path, handle)
            _INPROCESS_CLAIMS.add(path)
        return self

    def release(self) -> None:
        for symbol, (path, handle) in list(self._claims.items())[::-1]:
            try:
                handle.seek(0)
                handle.truncate()
                handle.flush()
            except Exception:
                pass
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                handle.close()
            except Exception:
                pass
            _INPROCESS_CLAIMS.discard(path)
            self._claims.pop(symbol, None)

    def __enter__(self) -> "RuntimeSymbolLease":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def claim_bybit_symbol_lease(
    *,
    adapter: Any,
    service_name: str,
    symbols: tuple[str, ...],
    lock_dir: str = "data/runtime/locks",
) -> RuntimeSymbolLease:
    lease = RuntimeSymbolLease(
        service_name=service_name,
        venue="bybit",
        account_scope=_account_scope(adapter),
        symbols=tuple(str(symbol).upper() for symbol in symbols if symbol),
        lock_dir=lock_dir,
    )
    return lease.acquire()
