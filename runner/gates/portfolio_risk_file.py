"""Portfolio risk file reader — used by CompositeRiskGate.

Thin read-only client for ``data/runtime/portfolio_risk.json`` produced
by ``scripts/portfolio_risk_monitor.py``.  Caches the latest snapshot
in-memory and only re-reads from disk when the file mtime changes.

Provides a single synchronous check::

    reader = PortfolioRiskFileReader()
    ok, reason = reader.check_would_exceed(
        symbol="BTCUSDT",
        venue="okx",
        side="buy",
        proposed_notional_usd=50.0,
    )

Semantics:
- Returns ``(True, "")`` (allow) when file is missing, stale
  (>PORTFOLIO_RISK_MAX_STALE_SEC old), or the symbol has no cap.
  This is a **fail-open** design — a broken monitor script should not
  block live trading.  The watchdog is expected to alert on stale files.
- Returns ``(True, "")`` when proposed + existing ≤ cap.
- Returns ``(False, detail)`` only when proposed + existing > cap.
- The proposed order's venue contribution is ADDED ON TOP of whatever
  that venue currently holds — because the runner's existing position
  is already reflected in the snapshot, and we're asking "will the
  TOTAL after this new fill exceed the limit?"
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SNAPSHOT_PATH = Path("data/runtime/portfolio_risk.json")


class PortfolioRiskFileReader:
    """In-memory cached reader with mtime-based invalidation."""

    def __init__(
        self,
        path: Path | None = None,
        max_stale_seconds: float = 300.0,
    ) -> None:
        self._path = path or SNAPSHOT_PATH
        self._max_stale = float(max_stale_seconds)
        self._cache: dict[str, Any] | None = None
        self._cache_mtime: float = 0.0

    # ------------------------------------------------------------------
    def _load(self) -> dict[str, Any] | None:
        if not self._path.exists():
            self._cache = None
            return None
        try:
            mtime = self._path.stat().st_mtime
            if self._cache is not None and mtime == self._cache_mtime:
                return self._cache
            with self._path.open() as f:
                data = json.load(f)
            self._cache = data
            self._cache_mtime = mtime
            return data
        except Exception as e:
            logger.debug("portfolio risk load failed: %s", e)
            return None

    # ------------------------------------------------------------------
    def check_would_exceed(
        self,
        *,
        symbol: str,
        venue: str,
        side: str,
        proposed_notional_usd: float,
    ) -> tuple[bool, str]:
        """Return (allowed, reason). Fail-open on missing/stale file."""
        snap = self._load()
        if snap is None:
            return True, ""  # file missing → allow

        age = time.time() - snap.get("ts", 0)
        if age > self._max_stale:
            logger.warning(
                "portfolio_risk.json stale %.0fs — falling through (monitor dead?)",
                age,
            )
            return True, f"portfolio_risk_stale({age:.0f}s)"

        sym_data = (snap.get("symbols") or {}).get(symbol.upper())
        if not sym_data:
            return True, ""

        cap = float(sym_data.get("max_total_notional_usd") or 0.0)
        if cap <= 0:
            return True, ""

        side_norm = (side or "").lower()
        if side_norm not in ("buy", "sell"):
            return True, ""

        # Existing totals INCLUDE this venue's contribution already.
        # We subtract this venue's current figure and re-add the proposed
        # figure as if it's a full replacement.  But that over-blocks
        # simple "add-to-position" orders.  Simpler model: proposed
        # notional is ADDITIONAL to whatever the snapshot shows, then
        # compare the sum to the cap.
        if side_norm == "buy":
            current = float(sym_data.get("total_long_notional_usd") or 0.0)
        else:
            current = float(sym_data.get("total_short_notional_usd") or 0.0)

        post_total = current + float(proposed_notional_usd)

        if post_total > cap:
            return False, (
                f"portfolio_cap {symbol} {side_norm}: "
                f"current=${current:.0f} + proposed=${proposed_notional_usd:.0f} "
                f"= ${post_total:.0f} > cap ${cap:.0f}"
            )
        return True, ""

    # Convenience: expose the snapshot (for logging / dashboards)
    def snapshot(self) -> dict[str, Any] | None:
        return self._load()
