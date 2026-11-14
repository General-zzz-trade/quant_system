"""Compatibility shell: trading.adapters.binance.adapter (migration step 8)."""
from __future__ import annotations

from trading.adapters.binance import adapter as _impl

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
