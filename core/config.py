"""Layered configuration service with hot-reload support.

Configuration is resolved bottom-up through layers:

    Defaults < File < Environment < Runtime (hot-update)

Each higher layer overrides the one below.  The ``ConfigService``
provides type-safe reads and change-notification callbacks.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

T = TypeVar("T")


# ── Errors ───────────────────────────────────────────────

class ConfigKeyError(KeyError):
    """Raised when a config key is missing from all layers."""


class ConfigTypeError(TypeError):
    """Raised when a config value cannot be cast to the requested type."""


# ── Layer protocol ───────────────────────────────────────

class ConfigLayer:
    """Base class for a configuration layer."""

    def has(self, key: str) -> bool:
        raise NotImplementedError

    def get_raw(self, key: str) -> Any:
        raise NotImplementedError


# ── Concrete layers ──────────────────────────────────────

class DefaultsLayer(ConfigLayer):
    """Hard-coded defaults (lowest priority)."""

    def __init__(self, defaults: Optional[Dict[str, Any]] = None) -> None:
        self._data: Dict[str, Any] = dict(defaults or {})

    def has(self, key: str) -> bool:
        return key in self._data

    def get_raw(self, key: str) -> Any:
        return self._data[key]


class FileLayer(ConfigLayer):
    """JSON or YAML file (loaded once at startup)."""

    def __init__(self, path: str | Path) -> None:
        self._data: Dict[str, Any] = {}
        p = Path(path)
        if p.exists():
            text = p.read_text(encoding="utf-8")
            suffix = p.suffix.lower()
            if suffix in (".json",):
                self._data = json.loads(text)
            elif suffix in (".yml", ".yaml"):
                try:
                    import yaml  # type: ignore[import-untyped]
                    self._data = yaml.safe_load(text) or {}
                except ImportError:
                    pass

    def has(self, key: str) -> bool:
        return key in self._data

    def get_raw(self, key: str) -> Any:
        return self._data[key]


class EnvLayer(ConfigLayer):
    """Environment variables with optional prefix.

    Key ``risk.max_leverage`` with prefix ``QS_`` looks up
    ``QS_RISK__MAX_LEVERAGE`` (dots → double-underscore, uppercased).
    """

    def __init__(self, prefix: str = "QS_") -> None:
        self._prefix = prefix

    def _env_key(self, key: str) -> str:
        return self._prefix + key.replace(".", "__").upper()

    def has(self, key: str) -> bool:
        return self._env_key(key) in os.environ

    def get_raw(self, key: str) -> Any:
        return os.environ[self._env_key(key)]


class RuntimeLayer(ConfigLayer):
    """In-memory overrides — for hot-reload / live tuning."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def get_raw(self, key: str) -> Any:
        with self._lock:
            return self._data[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)


# ── Config service ───────────────────────────────────────

class ConfigService:
    """Layered, type-safe, hot-reloadable configuration.

    Usage::

        svc = ConfigService(
            defaults={"risk.max_leverage": 3.0},
            file_path="config.json",
        )
        max_lev = svc.get("risk.max_leverage", float)  # → 3.0
        svc.watch("risk.max_leverage", lambda v: print(f"new: {v}"))
        svc.hot_update("risk.max_leverage", 2.0)  # triggers watcher
    """

    def __init__(
        self,
        *,
        defaults: Optional[Dict[str, Any]] = None,
        file_path: Optional[str | Path] = None,
        env_prefix: str = "QS_",
    ) -> None:
        self._runtime = RuntimeLayer()
        self._layers: list[ConfigLayer] = [
            DefaultsLayer(defaults),
        ]
        if file_path is not None:
            self._layers.append(FileLayer(file_path))
        self._layers.append(EnvLayer(env_prefix))
        self._layers.append(self._runtime)

        self._watchers: Dict[str, List[Callable[[Any], None]]] = {}
        self._lock = threading.Lock()

    def get(self, key: str, type_: Type[T] = str) -> T:  # type: ignore[assignment]
        """Read a config value, resolved top-down through layers."""
        # Walk layers in reverse (highest priority first)
        for layer in reversed(self._layers):
            if layer.has(key):
                raw = layer.get_raw(key)
                return self._cast(key, raw, type_)
        raise ConfigKeyError(key)

    def get_or(self, key: str, default: T, type_: Type[T] = str) -> T:  # type: ignore[assignment]
        """Like ``get`` but returns *default* if missing."""
        try:
            return self.get(key, type_)
        except ConfigKeyError:
            return default

    def watch(self, key: str, callback: Callable[[Any], None]) -> None:
        """Register a callback for when *key* changes via ``hot_update``."""
        with self._lock:
            self._watchers.setdefault(key, []).append(callback)

    def hot_update(self, key: str, value: Any) -> None:
        """Update *key* at runtime and notify watchers."""
        self._runtime.set(key, value)
        with self._lock:
            callbacks = list(self._watchers.get(key, []))
        for cb in callbacks:
            cb(value)

    @staticmethod
    def _cast(key: str, raw: Any, type_: Type[T]) -> T:
        if isinstance(raw, type_):
            return raw
        try:
            return type_(raw)  # type: ignore[call-arg]
        except (TypeError, ValueError) as e:
            raise ConfigTypeError(
                f"Cannot cast config {key!r} value {raw!r} to {type_.__name__}"
            ) from e
