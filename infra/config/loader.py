"""Config file loader with validation support.

Loads JSON or YAML config files and optionally validates required keys.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a JSON or YAML config file.

    YAML support is optional. If PyYAML is not installed, YAML files will raise.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")

    if suffix in {".json"}:
        return json.loads(text)

    if suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML config requires PyYAML") from e
        return yaml.safe_load(text) or {}

    raise ValueError(f"unsupported config type: {suffix}")


def validate_config(
    config: Dict[str, Any],
    *,
    required_keys: Sequence[str] = (),
    type_checks: Dict[str, type] | None = None,
) -> list[str]:
    """Validate config dict. Returns list of error messages (empty = valid).

    Parameters
    ----------
    config : dict
        The configuration to validate.
    required_keys : sequence of str
        Keys that must exist (dot-notation supported: "risk.max_leverage").
    type_checks : dict, optional
        Mapping of key -> expected type for type validation.
    """
    errors: list[str] = []

    for key in required_keys:
        if _resolve_key(config, key) is _MISSING:
            errors.append(f"missing required key: {key}")

    if type_checks:
        for key, expected in type_checks.items():
            val = _resolve_key(config, key)
            if val is not _MISSING and not isinstance(val, expected):
                errors.append(
                    f"type mismatch for '{key}': expected {expected.__name__}, "
                    f"got {type(val).__name__}"
                )

    return errors


_MISSING = object()


def _resolve_key(config: Dict[str, Any], key: str) -> Any:
    """Resolve a dot-notation key in a nested dict."""
    parts = key.split(".")
    current: Any = config
    for part in parts:
        if not isinstance(current, dict):
            return _MISSING
        current = current.get(part, _MISSING)
        if current is _MISSING:
            return _MISSING
    return current
