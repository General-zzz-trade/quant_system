"""Durable UTC day/week drawdown gate, usable without Rust or exchange access."""
from __future__ import annotations

import math
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class DrawdownGuard:
    """Latch a breach until its UTC period ends; a restart never resets equity.

    Scope must identify an account AND environment (for example okx-demo-main).
    The first observed equity in a period is its baseline, not an invented
    midnight equity. Deposits/withdrawals require separate reconciliation.
    """

    def __init__(self, path: Path, scope: str, *, daily_pct: float = 3.0,
                 weekly_pct: float = 10.0, max_age_seconds: float = 120.0) -> None:
        if not scope.strip():
            raise ValueError("risk scope is required")
        if any(not math.isfinite(x) or not 0 < x < 100 for x in (daily_pct, weekly_pct)):
            raise ValueError("drawdown limits must be finite percentages between 0 and 100")
        if not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        self.path, self.scope = Path(path), scope
        self.limits = {"day": daily_pct, "week": weekly_pct}
        self.max_age = max_age_seconds
        self._error = "equity not refreshed in this process"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS drawdown_periods (
                scope TEXT, kind TEXT, period TEXT, baseline REAL NOT NULL,
                limit_pct REAL NOT NULL, tripped INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(scope, kind, period))""")
            db.execute("""CREATE TABLE IF NOT EXISTS equity_observations (
                scope TEXT PRIMARY KEY, equity REAL NOT NULL, observed_at REAL NOT NULL)""")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _periods(now: float) -> dict[str, str]:
        dt = datetime.fromtimestamp(now, timezone.utc)
        year, week, _ = dt.isocalendar()
        return {"day": dt.date().isoformat(), "week": f"{year}-W{week:02d}"}

    def invalidate(self, reason: str) -> None:
        self._error = reason

    def observe(self, equity: float, *, now: float | None = None) -> tuple[bool, str]:
        now = time.time() if now is None else now
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
                for kind, period in self._periods(now).items():
                    db.execute("INSERT OR IGNORE INTO drawdown_periods VALUES (?, ?, ?, ?, ?, 0)",
                               (self.scope, kind, period, equity, self.limits[kind]))
                    baseline, limit = db.execute(
                        "SELECT baseline, limit_pct FROM drawdown_periods WHERE scope=? AND kind=? AND period=?",
                        (self.scope, kind, period),
                    ).fetchone()
                    if (not math.isfinite(baseline) or baseline <= 0
                            or not math.isfinite(limit) or not 0 < limit < 100):
                        raise ValueError("invalid persisted equity baseline")
                    # Restarting with a looser config cannot un-trip or weaken this period.
                    limit = min(limit, self.limits[kind])
                    breached = equity <= baseline * (1 - limit / 100)
                    db.execute("""UPDATE drawdown_periods SET limit_pct=?, tripped=MAX(tripped, ?)
                        WHERE scope=? AND kind=? AND period=?""",
                               (limit, int(breached), self.scope, kind, period))
                db.execute("INSERT OR REPLACE INTO equity_observations VALUES (?, ?, ?)",
                           (self.scope, equity, now))
            self._error = ""
        except Exception as exc:
            self.invalidate(f"equity refresh failed: {type(exc).__name__}: {exc}")
        return self.check(now=now)

    def check(self, *, now: float | None = None) -> tuple[bool, str]:
        if self._error:
            return False, self._error
        now = time.time() if now is None else now
        try:
            with self._connect() as db:
                row = db.execute("SELECT observed_at FROM equity_observations WHERE scope=?",
                                 (self.scope,)).fetchone()
                if not row or not 0 <= now - row[0] <= self.max_age:
                    return False, "equity observation missing, stale or in the future"
                for kind, period in self._periods(now).items():
                    row = db.execute("""SELECT tripped FROM drawdown_periods
                        WHERE scope=? AND kind=? AND period=?""", (self.scope, kind, period)).fetchone()
                    if row is None:
                        return False, f"{kind} baseline not initialized"
                    if row[0]:
                        return False, f"{kind} drawdown latched for {period}"
            return True, ""
        except Exception as exc:
            return False, f"drawdown state unavailable: {type(exc).__name__}"
