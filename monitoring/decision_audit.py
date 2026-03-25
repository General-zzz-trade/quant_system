"""Structured decision audit logger — every signal/position/exit is logged as JSON.

Writes to data/runtime/decision_audit.jsonl (one JSON object per line).
Designed for post-trade analysis and regulatory audit trail.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
logger = logging.getLogger(__name__)

AUDIT_PATH = Path("data/runtime/decision_audit.jsonl")


class DecisionAuditLogger:
    """Append-only structured logger for decision events."""

    def __init__(self, path: Path = AUDIT_PATH):
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
