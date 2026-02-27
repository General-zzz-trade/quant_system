# execution/adapters/bitget/error_map.py
"""Bitget error code mapping."""
from __future__ import annotations

from enum import Enum
from typing import Mapping


class BitgetErrorAction(Enum):
    RETRY = "retry"
    REJECT = "reject"
    HALT = "halt"
    IGNORE = "ignore"


# Bitget V2 API common error codes
# Success = "00000"; anything else is an error
BITGET_ERROR_MAP: Mapping[str, tuple[BitgetErrorAction, str]] = {
    # Rate limiting / transient
    "40014": (BitgetErrorAction.RETRY, "rate limit exceeded"),
    "40700": (BitgetErrorAction.RETRY, "system busy"),
    "40710": (BitgetErrorAction.RETRY, "request timeout"),
    "50001": (BitgetErrorAction.RETRY, "service unavailable"),
    "40012": (BitgetErrorAction.RETRY, "too many requests"),
    # Auth / signature
    "40001": (BitgetErrorAction.HALT, "access key invalid"),
    "40002": (BitgetErrorAction.HALT, "invalid timestamp"),
    "40003": (BitgetErrorAction.HALT, "invalid sign"),
    "40004": (BitgetErrorAction.HALT, "invalid passphrase"),
    "40005": (BitgetErrorAction.HALT, "invalid access key"),
    "40006": (BitgetErrorAction.HALT, "ip not in whitelist"),
    "40007": (BitgetErrorAction.HALT, "invalid scope"),
    # Parameter / logic
    "40013": (BitgetErrorAction.REJECT, "invalid parameter"),
    "40015": (BitgetErrorAction.REJECT, "invalid symbol"),
    "40016": (BitgetErrorAction.REJECT, "duplicate clientOid"),
    "43001": (BitgetErrorAction.REJECT, "order does not exist"),
    "43002": (BitgetErrorAction.REJECT, "order already cancelled"),
    "43004": (BitgetErrorAction.REJECT, "insufficient balance"),
    "43006": (BitgetErrorAction.REJECT, "order qty too small"),
    "43007": (BitgetErrorAction.REJECT, "order qty too large"),
    "43008": (BitgetErrorAction.REJECT, "price out of range"),
    "43010": (BitgetErrorAction.REJECT, "position does not exist"),
    "43011": (BitgetErrorAction.REJECT, "order would trigger immediately"),
    "43012": (BitgetErrorAction.REJECT, "reduce only rejected"),
    "45110": (BitgetErrorAction.REJECT, "insufficient margin"),
}


def classify_error(code: str) -> tuple[BitgetErrorAction, str]:
    """Return handling strategy for a Bitget error code."""
    return BITGET_ERROR_MAP.get(code, (BitgetErrorAction.REJECT, f"unknown code {code}"))
