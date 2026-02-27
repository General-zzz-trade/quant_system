"""Config file loader with validation and secrets management.

Loads JSON or YAML config files, validates required keys, and provides
secure credential resolution via environment variables.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Protocol, Sequence


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


# ---------------------------------------------------------------------------
# Secrets management
# ---------------------------------------------------------------------------

class SecurityError(Exception):
    """Raised when a config contains plaintext secrets."""
    pass


class SecretProvider(Protocol):
    """Protocol for resolving secrets from secure sources."""

    def get_secret(self, key: str) -> str:
        """Resolve a secret by key name. Raises KeyError if not found."""
        ...


class EnvSecretProvider:
    """Resolves secrets from environment variables."""

    def get_secret(self, key: str) -> str:
        val = os.environ.get(key)
        if val is None:
            raise KeyError(f"environment variable not set: {key}")
        return val


# Patterns that look like hardcoded API keys/secrets
_SECRET_PATTERNS = [
    re.compile(r"^[A-Za-z0-9]{32,}$"),          # Long alphanumeric strings (API keys)
    re.compile(r"^[A-Fa-f0-9]{64}$"),            # 64-char hex (secret keys)
    re.compile(r"^sk[-_][A-Za-z0-9]{20,}$"),     # sk-prefixed keys
]

# Config keys that should never contain plaintext secrets
_SENSITIVE_KEY_PATTERNS = {"api_key", "api_secret", "secret_key", "password", "token"}


def check_no_plaintext_secrets(config: Dict[str, Any], *, path: str = "") -> list[str]:
    """Scan config for values that look like hardcoded secrets.

    Returns a list of violations. Raises SecurityError if any are found.
    """
    violations: list[str] = []
    _scan_dict(config, path, violations)
    return violations


def load_config_secure(path: str | Path) -> Dict[str, Any]:
    """Load config and reject if it contains plaintext secrets."""
    config = load_config(path)
    violations = check_no_plaintext_secrets(config)
    if violations:
        raise SecurityError(
            f"Config contains {len(violations)} potential plaintext secret(s):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )
    return config


def resolve_credentials(
    config: Dict[str, Any],
    provider: SecretProvider | None = None,
) -> Dict[str, str]:
    """Resolve API credentials from config using a SecretProvider.

    Expects config to have 'credentials.api_key_env' and 'credentials.api_secret_env'
    pointing to environment variable names.
    """
    provider = provider or EnvSecretProvider()
    creds = config.get("credentials", {})
    if not isinstance(creds, dict):
        return {}

    result: Dict[str, str] = {}
    for field in ("api_key_env", "api_secret_env"):
        env_var = creds.get(field)
        if isinstance(env_var, str) and env_var:
            result[field.replace("_env", "")] = provider.get_secret(env_var)
    return result


def _scan_dict(d: Dict[str, Any], path: str, violations: list[str]) -> None:
    for k, v in d.items():
        full_key = f"{path}.{k}" if path else k
        key_lower = k.lower()

        if isinstance(v, dict):
            _scan_dict(v, full_key, violations)
        elif isinstance(v, str) and v.strip():
            # Check if this key is sensitive
            is_sensitive_key = any(pat in key_lower for pat in _SENSITIVE_KEY_PATTERNS)

            if is_sensitive_key:
                # Keys ending in _env are expected to contain env var names
                if key_lower.endswith("_env"):
                    continue
                # Non-empty string in a sensitive key field = violation
                if len(v.strip()) > 8:
                    violations.append(f"{full_key}: potential plaintext secret detected")
            else:
                # For non-sensitive keys, check if value matches secret patterns
                for pattern in _SECRET_PATTERNS:
                    if pattern.match(v.strip()):
                        violations.append(f"{full_key}: value looks like a secret ({len(v)} chars)")
                        break
