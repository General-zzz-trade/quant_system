"""Structured decision audit logger — every signal/position/exit is logged as JSON.

Writes to data/runtime/decision_audit_{venue}.jsonl by default (one JSON
object per line). The per-venue split prevents two parallel runners
(binance + okx) from clobbering each other's session log on startup.
Every record is also tagged with a `venue` field for downstream queries.

Pytest safety: when running under pytest (detected via sys.modules), the
logger redirects to /tmp/test_decision_audit_{venue}.jsonl so unit-test
AlphaDecisionModule instantiations never pollute the production file.

Designed for post-trade analysis and regulatory audit trail.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
logger = logging.getLogger(__name__)

AUDIT_DIR = Path("data/runtime")
# Legacy single-file path — kept for backward compatibility with tools that
# still read this path directly (watchdog pre-multi-venue, etc.)
AUDIT_PATH = AUDIT_DIR / "decision_audit.jsonl"


def _in_pytest() -> bool:
    """Detect pytest so test modules don't write to production audit logs."""
    return "pytest" in sys.modules or bool(os.environ.get("PYTEST_CURRENT_TEST"))


def audit_path_for(venue: str) -> Path:
    """Return the per-venue audit log path.

    In pytest, redirects to /tmp to keep production data clean.
    """
    venue = (venue or "unknown").lower()
    if _in_pytest():
        return Path("/tmp") / f"test_decision_audit_{venue}.jsonl"
    return AUDIT_DIR / f"decision_audit_{venue}.jsonl"


class DecisionAuditLogger:
    """Append-only structured logger for decision events.

    Each instance writes to its own per-venue file. Every emitted record
    is stamped with the `venue` field so consumers that merge venues can
    still distinguish them.
    """

    def __init__(self, path: Path | None = None, venue: str = "binance"):
        self._venue = (venue or "unknown").lower()
        if path is None:
            path = audit_path_for(self._venue)
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None

    def _ensure_open(self):
        if self._file is None or self._file.closed:
            self._file = open(self._path, "a")

    def log_signal(self, *, symbol: str, runner_key: str, z_score: float,
                   signal: int, confidence: float = 0.0,
                   features: dict | None = None, **extra) -> None:
        """Log a signal generation event."""
        self._write("signal", symbol=symbol, runner_key=runner_key,
                    z_score=z_score, signal=signal, confidence=confidence,
                    top_features=_top_features(features), **extra)

    def log_entry(self, *, symbol: str, side: str, qty: float, price: float,
                  reason: str, **extra) -> None:
        """Log a position entry."""
        self._write("entry", symbol=symbol, side=side, qty=qty, price=price,
                    reason=reason, **extra)

    def log_exit(self, *, symbol: str, side: str, qty: float, price: float,
                 reason: str, pnl: float = 0.0, **extra) -> None:
        """Log a position exit."""
        self._write("exit", symbol=symbol, side=side, qty=qty, price=price,
                    reason=reason, pnl=pnl, **extra)

    def log_sizing(self, *, symbol: str, target_qty: float, equity: float,
                   leverage: float, ic_scale: float, regime_scale: float,
                   **extra) -> None:
        """Log position sizing decision."""
        self._write("sizing", symbol=symbol, target_qty=target_qty,
                    equity=equity, leverage=leverage, ic_scale=ic_scale,
                    regime_scale=regime_scale, **extra)

    def log_gate(self, *, symbol: str, gate_name: str, allowed: bool,
                 scale: float = 1.0, reason: str = "", **extra) -> None:
        """Log a gate decision."""
        self._write("gate", symbol=symbol, gate_name=gate_name,
                    allowed=allowed, scale=scale, reason=reason, **extra)

    def _write(self, event_type: str, **data) -> None:
        try:
            record = {
                "ts": time.time(),
                "type": event_type,
                "venue": self._venue,
                **data,
            }
            self._ensure_open()
            self._file.write(json.dumps(record, default=str) + "\n")
            self._file.flush()
        except Exception:
            logger.debug("Audit log write failed", exc_info=True)

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()


def _top_features(features: dict | None, n: int = 5) -> dict | None:
    """Extract top N features by absolute value for audit (compact)."""
    if not features:
        return None
    sorted_f = sorted(
        features.items(),
        key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
        reverse=True,
    )
    return dict(sorted_f[:n])
