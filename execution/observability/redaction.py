# execution/observability/redaction.py
"""Redact sensitive information from execution logs and payloads."""
from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Set

# 需要脱敏的字段名集合
SENSITIVE_KEYS: Set[str] = {
    "api_key", "apiKey", "secret", "secretKey", "api_secret",
    "password", "passphrase", "token", "signature",
    "X-MBX-APIKEY", "x-mbx-apikey",
}

# API key 模式 — 掩码中间部分
_KEY_PATTERN = re.compile(r"^(.{4}).+(.{4})$")


def redact_value(key: str, value: Any) -> Any:
    """对敏感字段值进行脱敏。"""
    if key in SENSITIVE_KEYS:
        if isinstance(value, str) and len(value) > 8:
            m = _KEY_PATTERN.match(value)
            if m:
                return f"{m.group(1)}****{m.group(2)}"
        return "****"
    return value


def redact_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
    """对 dict 中的敏感字段进行脱敏（递归）。"""
    result: Dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            result[k] = redact_dict(v)
        elif isinstance(v, (list, tuple)):
            result[k] = [
                redact_dict(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            result[k] = redact_value(k, v)
    return result


def redact_url(url: str) -> str:
    """脱敏 URL 中的 API key 参数。"""
    for key in ("apiKey", "api_key", "signature", "timestamp"):
        pattern = re.compile(f"({key}=)[^&]+")
        url = pattern.sub(f"\\1****", url)
    return url
