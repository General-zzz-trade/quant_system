"""Compatibility shell: trading.adapters.binance.rest (migration step 8).

Names (including private helpers) are copied from the new module. Patch the new module
(``trading.adapters.binance.rest``), not this one, to change what the adapter uses.
"""
from __future__ import annotations

from trading.adapters.binance import rest as _impl

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
