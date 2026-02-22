# context/versioning.py
"""Context versioning — track schema and state versions."""
from __future__ import annotations

from dataclasses import dataclass


CONTEXT_SCHEMA_VERSION = "1.0.0"
ACCOUNT_STATE_VERSION = "1.0.0"
MARKET_STATE_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class VersionInfo:
    """版本信息。"""
    schema_version: str
    component: str
    notes: str = ""


def get_context_version() -> VersionInfo:
    return VersionInfo(
        schema_version=CONTEXT_SCHEMA_VERSION,
        component="context",
        notes="v1.0: market + account, no margin model",
    )


def check_compatibility(required: str, actual: str) -> bool:
    """检查版本兼容性（major 版本必须相同）。"""
    req_parts = required.split(".")
    act_parts = actual.split(".")
    return req_parts[0] == act_parts[0]
