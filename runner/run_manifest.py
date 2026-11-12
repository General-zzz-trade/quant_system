"""One explicit, account-free runner selection for the shadow entry point.

This module validates configuration and local artifact structure only. It never
deserializes a model, reads an account, or certifies online data/model quality.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping


_MAX_JSON_BYTES = 1_048_576
# The coordinator converts float USDT to signed i64 units at scale 1e8.
# Leave headroom below the native maximum for floating-point conversion.
_MAX_SHADOW_BALANCE = Decimal("90000000000")
_FIELDS = {"version", "mode", "venue", "currency", "shadow_balance", "models_root", "data_root",
           "runners", "disabled_runners"}
_CREDENTIAL = re.compile(
    r"(?:^|_)(?:api_?(?:key|secret)|secret|token|password|passphrase|private_?key|credentials?)(?:$|_)", re.I,
)


@dataclass(frozen=True)
class RunnerSpec:
    key: str
    symbol: str
    interval: str
    interval_ms: int
    model_dir: str
    model_path: Path
    warmup: int

    def binding(self) -> dict[str, Any]:
        return {"key": self.key, "symbol": self.symbol, "interval": self.interval,
                "model_dir": self.model_dir, "warmup": self.warmup}


@dataclass(frozen=True)
class RunManifest:
    manifest_path: Path
    project_root: Path
    models_root: Path
    data_root: Path
    shadow_balance: Decimal
    runners: tuple[RunnerSpec, ...]
    disabled_runner_keys: tuple[str, ...]
    digest: str
    files_validated: bool
    mode: str = "shadow"
    venue: str = "bybit"
    currency: str = "USDT"
    version: int = 1

    @property
    def runner_keys(self) -> tuple[str, ...]:
        return tuple(runner.key for runner in self.runners)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(runner.symbol for runner in self.runners))


def _safe_path(path: Path) -> Path:
    path = Path(path).absolute()
    if ".." in path.parts or any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Manifest paths cannot contain traversal or symbolic links")
    return path


def _relative(value: Any, name: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or "\x00" in value
            or any(ord(char) < 32 for char in value)):
        raise ValueError("Manifest needs a canonical relative " + name)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or str(path) == ".":
        raise ValueError("Manifest needs a canonical relative " + name)
    return value


def _strict_json(raw: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate manifest/model JSON key")
            if _CREDENTIAL.search(key.replace("-", "_")):
                raise ValueError("Manifest/model JSON cannot contain credential fields")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("Manifest/model JSON cannot contain non-finite numbers")

    def floating(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Manifest/model JSON cannot contain non-finite numbers")
        return number

    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=floating)
    if not isinstance(result, dict):
        raise ValueError("Manifest/model JSON must be an object")
    return result


def _regular(path: Path) -> Path:
    path = _safe_path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
        raise ValueError("Required shadow input must be a nonempty regular file")
    return path


def _json_file(path: Path) -> dict:
    path = _regular(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_JSON_BYTES:
            raise ValueError("Manifest/model JSON must be a bounded regular file")
        raw = stream.read(_MAX_JSON_BYTES + 1)
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError("Manifest/model JSON is too large")
    return _strict_json(raw.decode("utf-8"))


def _runner_keys(value: Any, name: str, *, empty: bool) -> tuple[str, ...]:
    if (not isinstance(value, list) or (not value and not empty) or len(value) > 128
            or any(not isinstance(key, str) or not re.fullmatch(r"[A-Z0-9]+(?:_[1-9][0-9]*[mh])?", key)
                   for key in value)):
        raise ValueError("Manifest needs an explicit " + name + " runner list")
    if len(set(value)) != len(value):
        raise ValueError("Manifest runner lists cannot contain duplicates")
    return tuple(value)


def _balance(value: Any) -> Decimal:
    # A string keeps the declared synthetic starting amount independent of
    # binary floating-point and makes this distinct from an account query.
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,18}(?:\.[0-9]{1,8})?", value):
        raise ValueError("shadow_balance must be an explicit positive decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Invalid synthetic shadow balance") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError("Synthetic shadow balance must be positive and finite")
    if result > _MAX_SHADOW_BALANCE:
        raise ValueError("Synthetic shadow balance exceeds the native fixed-point state limit")
    return result


def _model_references(config: dict) -> set[str]:
    primary = config.get("horizon_models")
    legacy = config.get("models")
    if primary:
        if (not isinstance(primary, list) or any(not isinstance(row, dict) or not row.get("lgbm")
                                                 for row in primary)):
            raise ValueError("Model config requires declared primary horizon models")
    elif not isinstance(legacy, list) or not legacy:
        raise ValueError("Model config has no declared primary models")
    result: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"lgbm", "xgb", "ridge", "meta"}:
                    if item in (None, ""):
                        continue
                    result.add(_relative(item, "model reference"))
                elif key == "models":
                    if not isinstance(item, list) or not item:
                        raise ValueError("Declared model list cannot be empty or malformed")
                    for name in item:
                        result.add(_relative(name, "model reference"))
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(config)
    if not result:
        raise ValueError("Model config has no declared artifacts")
    return result


def validate_run_inputs(manifest: RunManifest) -> tuple[Path, ...]:
    """Check required artifact presence/structure without loading model code.

    Freshness and causal observation validation remain the runtime data gate's
    responsibility. This check must not turn local file existence into readiness.
    """
    for root in (manifest.models_root, manifest.data_root):
        if not _safe_path(root).is_dir():
            raise ValueError("Shadow model/data root is missing")
    files: list[Path] = []
    for directory in dict.fromkeys(runner.model_path for runner in manifest.runners):
        if not _safe_path(directory).is_dir():
            raise ValueError("Selected shadow model directory is missing")
        # The model loader also selects optional Rust JSON/signature sidecars.
        # Reject links/special files anywhere in a selected model directory.
        for path in directory.rglob("*"):
            path = _safe_path(path)
            if not path.is_dir() and not stat.S_ISREG(path.stat().st_mode):
                raise ValueError("Shadow model directory contains a special file")
        config_path = directory / "config.json"
        config = _json_file(config_path)
        files.append(config_path)
        for name in sorted(_model_references(config)):
            artifact = _regular(directory / name)
            files.append(artifact)
            for sidecar in (artifact.with_suffix(".json"), Path(str(artifact) + ".sig")):
                if sidecar.exists() and sidecar != artifact:
                    files.append(_regular(sidecar))
    specifications = (
        ("_1h.csv", {"open_time", "open", "high", "low", "close", "volume"}),
        ("_open_interest.csv", {"timestamp", "sum_open_interest"}),
        ("_ls_ratio.csv", {"timestamp", "long_short_ratio"}),
    )
    for symbol in manifest.symbols:
        for suffix, required in specifications:
            path = _regular(manifest.data_root / (symbol + suffix))
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if (reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames)
                        or not required <= set(reader.fieldnames) or next(reader, None) is None):
                    raise ValueError("Required shadow CSV needs its declared columns and observations")
            files.append(path)
    return tuple(dict.fromkeys(files))


def load_run_manifest(path: Path, *, project_root: Path | None = None, validate_files: bool = True,
                      symbol_config: Mapping[str, dict] | None = None) -> RunManifest:
    """Resolve the explicit active/disabled partition against strategy metadata.

    ``validate_files=False`` is only a configuration inspection; it cannot prove
    a runnable manifest and is surfaced as ``files_validated=False``.
    """
    root = _safe_path(Path.cwd() if project_root is None else project_root)
    if not root.is_dir():
        raise ValueError("Shadow project root is missing")
    manifest_path = _safe_path(path if Path(path).is_absolute() else root / path)
    config = _json_file(manifest_path)
    if set(config) != _FIELDS:
        raise ValueError("Shadow manifest contains unknown or missing fields")
    if (type(config["version"]) is not int or config["version"] != 1
            or config["mode"] != "shadow" or config["venue"] != "bybit" or config["currency"] != "USDT"):
        raise ValueError("Manifest must explicitly select version 1 Bybit USDT shadow mode")
    active = _runner_keys(config["runners"], "active", empty=False)
    disabled = _runner_keys(config["disabled_runners"], "disabled", empty=True)
    if symbol_config is None:
        from strategy.config import SYMBOL_CONFIG
        symbol_config = SYMBOL_CONFIG
    if set(active) & set(disabled) or set(active) | set(disabled) != set(symbol_config):
        raise ValueError("Active/disabled runners must exactly partition known strategy runner keys")
    balance = _balance(config["shadow_balance"])
    models = _safe_path(root / _relative(config["models_root"], "models root"))
    data = _safe_path(root / _relative(config["data_root"], "data root"))
    if models == data or models in data.parents or data in models.parents:
        raise ValueError("Shadow model and data roots must not overlap")
    runners = []
    for key in active:
        record = symbol_config[key]
        if not isinstance(record, Mapping):
            raise ValueError("Invalid strategy runner binding")
        symbol = record.get("symbol", key)
        interval = record.get("interval", "60")
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9]+", symbol):
            raise ValueError("Invalid bound strategy market symbol")
        if (not isinstance(interval, str)
                or interval not in {"1", "3", "5", "15", "30", "60", "120", "240", "360", "720"}):
            raise ValueError("Unsupported bound strategy minute interval")
        warmup = record.get("warmup", 300 if "4h" in key else 800)
        # Startup warms through the latest closed bar in one bounded request.
        if type(warmup) is not int or not 1 <= warmup <= 1000:
            raise ValueError("Invalid bound strategy warmup count")
        directory = _relative(record.get("model_dir"), "model directory")
        runners.append(RunnerSpec(key, symbol, interval, int(interval) * 60_000, directory,
                                  _safe_path(models / directory), warmup))
    payload = {"manifest": config, "bindings": [runner.binding() for runner in runners],
               "strategy_bindings": {key: dict(symbol_config[key]) for key in active}}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    result = RunManifest(manifest_path, root, models, data, balance, tuple(runners), disabled, digest,
                         files_validated=validate_files)
    if validate_files:
        validate_run_inputs(result)
    return result
