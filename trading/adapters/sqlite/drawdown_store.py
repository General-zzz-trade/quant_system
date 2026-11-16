"""Durable UTC day / ISO week drawdown latch in SQLite (implements the DrawdownGuard port).

The arithmetic lives in trading.domain.risk.drawdown; this class only stores and reads it. Scope must
identify an account AND environment. A restart never resets equity; unknown state blocks entries and
reads as latched. Deposits and withdrawals are not reconciled here.
"""
from __future__ import annotations

import sqlite3
import math
import time
from contextlib import AbstractContextManager
from pathlib import Path

from trading.adapters.sqlite.connection import connect

from trading.application.ports import Clock
from trading.domain.risk.drawdown import advance, periods, validate_limits


class SqliteDrawdownGuard:
    TABLES = ("drawdown_periods", "equity_observations")

    def __init__(self, path: Path, scope: str, *, daily_pct: float = 3.0,
                 weekly_pct: float = 10.0, max_age_seconds: float = 120.0, clock: Clock = time.time) -> None:
        if not scope.strip():
            raise ValueError("risk scope is required")
        self.limits = validate_limits(daily_pct, weekly_pct)
        if not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        self.path, self.scope, self.clock = Path(path), scope, clock
        self.max_age = max_age_seconds
        self._error = "equity not refreshed in this process"

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        """The state database as ``state_dir init`` created it; never creates a file or a table (ADR-0009)."""
        return connect(self.path)

    def invalidate(self, reason: str) -> None:
        self._error = reason

    def observe(self, equity: float, *, now: float | None = None) -> tuple[bool, str]:
        now = self.clock() if now is None else now
        try:
            if not math.isfinite(equity) or equity <= 0 or not math.isfinite(now):
                raise ValueError("invalid equity or observation time")
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                previous = db.execute(
                    "SELECT observed_at FROM equity_observations WHERE scope=?", (self.scope,)
                ).fetchone()
                if previous and previous[0] > now:
                    raise ValueError("equity clock moved backwards")
                for kind, period in periods(now).items():
                    db.execute("INSERT OR IGNORE INTO drawdown_periods VALUES (?, ?, ?, ?, ?, 0)",
                               (self.scope, kind, period, equity, self.limits[kind]))
                    baseline, stored = db.execute(
                        "SELECT baseline, limit_pct FROM drawdown_periods WHERE scope=? AND kind=? AND period=?",
                        (self.scope, kind, period),
                    ).fetchone()
                    peak, limit, breached = advance(baseline, stored, self.limits[kind], equity)
                    db.execute("""UPDATE drawdown_periods SET baseline=?, limit_pct=?, tripped=MAX(tripped, ?)
                        WHERE scope=? AND kind=? AND period=?""",
                               (peak, limit, int(breached), self.scope, kind, period))
                db.execute("INSERT OR REPLACE INTO equity_observations VALUES (?, ?, ?)",
                           (self.scope, equity, now))
            self._error = ""
        except Exception as exc:
            self.invalidate(f"equity refresh failed: {type(exc).__name__}: {exc}")
        return self.check(now=now)

    def latch_current(self, *, now: float) -> None:
        """Latch the current UTC day and ISO week (a restore, ADR-0016 point 6): entries stay blocked until each
        period ends, then the usual rules apply. Nothing here ever unlatches. A period without a row gets the last
        observed equity as its baseline (1.0 when none was observed: it is only the start of a later high-water
        mark, and the latch holds whatever it is)."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT equity FROM equity_observations WHERE scope=?", (self.scope,)).fetchone()
            baseline = float(row[0]) if row and row[0] > 0 else 1.0
            for kind, period in periods(now).items():
                db.execute("INSERT OR IGNORE INTO drawdown_periods VALUES (?, ?, ?, ?, ?, 1)",
                           (self.scope, kind, period, baseline, self.limits[kind]))
                db.execute("UPDATE drawdown_periods SET tripped=1 WHERE scope=? AND kind=? AND period=?",
                           (self.scope, kind, period))

    def latched(self, *, now: float | None = None) -> bool:
        """True when a drawdown limit has tripped in the current day or week; unknown state is latched."""
        now = self.clock() if now is None else now
        try:
            with self._connect() as db:
                for kind, period in periods(now).items():
                    row = db.execute("SELECT tripped FROM drawdown_periods WHERE scope=? AND kind=? AND period=?",
                                     (self.scope, kind, period)).fetchone()
                    if row is not None and row[0]:
                        return True
            return False
        except Exception:
            return True

    def check(self, *, now: float | None = None) -> tuple[bool, str]:
        if self._error:
            return False, self._error
        now = self.clock() if now is None else now
        try:
            with self._connect() as db:
                row = db.execute("SELECT observed_at FROM equity_observations WHERE scope=?",
                                 (self.scope,)).fetchone()
                if not row or not 0 <= now - row[0] <= self.max_age:
                    return False, "equity observation missing, stale or in the future"
                for kind, period in periods(now).items():
                    row = db.execute("""SELECT tripped FROM drawdown_periods
                        WHERE scope=? AND kind=? AND period=?""", (self.scope, kind, period)).fetchone()
                    if row is None:
                        return False, f"{kind} baseline not initialized"
                    if row[0]:
                        return False, f"{kind} drawdown latched for {period}"
            return True, ""
        except Exception as exc:
            return False, f"drawdown state unavailable: {type(exc).__name__}"
