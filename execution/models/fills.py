# execution/models/fills.py
from __future__ import annotations

from typing import Any
from trading.domain.venue import CanonicalFill  # noqa: F401  re-exported


# CanonicalFill: moved to trading.domain.venue (migration step 8)


def fill_to_record(fill: Any) -> dict[str, str]:
    """Convert any fill-like object to a standard record dict.

    Works with CanonicalFill, SimpleNamespace, or any duck-typed fill.
    Fields aligned with CanonicalFill.to_record() — same 12 fields.
    Use this to replace all ad-hoc fill dict construction.
    """
    if isinstance(fill, CanonicalFill):
        return fill.to_record()
    return {
        "venue": str(getattr(fill, "venue", "")),
        "symbol": str(getattr(fill, "symbol", "")),
        "order_id": str(getattr(fill, "order_id", "")),
        "trade_id": str(getattr(fill, "trade_id", "")),
        "fill_id": str(getattr(fill, "fill_id", "")),
        "side": str(getattr(fill, "side", "")),
        "qty": str(getattr(fill, "qty", "")),
        "price": str(getattr(fill, "price", "")),
        "fee": str(getattr(fill, "fee", "")),
        "fee_asset": str(getattr(fill, "fee_asset", "") or ""),
        "liquidity": str(getattr(fill, "liquidity", "") or ""),
        "ts_ms": str(getattr(fill, "ts_ms", "")),
        "payload_digest": str(getattr(fill, "payload_digest", "")),
    }
