# execution/adapters/binance/protective_stops.py
"""Exchange-side catastrophic stop for each confirmed strategy position.

The strategy's own exits run on hourly closes. This manager keeps one
Binance-side ``STOP_MARKET closePosition`` algo order per open position so a
large intrabar move, a crashed process or a lost connection still closes the
position at roughly ``stop_pct`` from the confirmed entry price.

Each ``sync()`` (called every few seconds from the runtime's single thread):

- flat position          -> cancel any active stop;
- position, no stop      -> place one (deterministic ``clientAlgoId``);
- position changed       -> replace the stop for the new position revision;
- stop triggered         -> adopt the resulting exchange order into the
                            durable journal so its fills reach the strategy
                            ledger exactly once;
- stop canceled/expired  -> place a new one (next attempt id);
- "would immediately trigger" -> report ``stop_breached`` so the runtime
                            closes the position with a market order.

State lives in the order journal database (table ``protective_stops``).
"""
from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)

ACTIVE = frozenset({"NEW", "TRIGGERING"})
TRIGGERED = frozenset({"TRIGGERED", "FINISHED"})
MIN_STOP_PCT, MAX_STOP_PCT = Decimal("0.005"), Decimal("0.15")
_IMMEDIATE_TRIGGER = -2021


class ProtectiveStopManager:
    def __init__(self, adapter: Any, executor: Any, symbol: str, *, stop_pct: Decimal,
                 clock: Any = time.time) -> None:
        stop_pct = Decimal(str(stop_pct))
        if not MIN_STOP_PCT <= stop_pct <= MAX_STOP_PCT:
            raise ValueError(f"stop_pct must be between {MIN_STOP_PCT} and {MAX_STOP_PCT}")
        self.adapter, self.executor, self.symbol = adapter, executor, symbol
        self.stop_pct, self.clock = stop_pct, clock
        self.journal = executor.journal
        with self.journal.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS protective_stops (
                scope TEXT NOT NULL, owner TEXT NOT NULL, client_algo_id TEXT NOT NULL,
                symbol TEXT NOT NULL, revision TEXT NOT NULL, attempt INTEGER NOT NULL,
                side TEXT NOT NULL, trigger_price TEXT NOT NULL, status TEXT NOT NULL,
                created_at REAL NOT NULL, PRIMARY KEY(scope, client_algo_id))""")

    # ------------------------------------------------------------------
    def _client_id(self, revision: str, attempt: int) -> str:
        digest = hashlib.sha256(f"{self.executor.scope}:{self.executor.owner}:{revision}:{attempt}".encode())
        return "qsstop" + digest.hexdigest()[:28]  # 34 chars, Binance limit is 36

    def _rows(self, *, live_only: bool = True) -> list[dict]:
        query = "SELECT * FROM protective_stops WHERE scope=? AND owner=? AND symbol=?"
        if live_only:
            query += " AND status IN ('active','unknown','canceling')"
        with self.journal.connect() as db:
            return [dict(r) for r in db.execute(query + " ORDER BY created_at", (
                self.executor.scope, self.executor.owner, self.symbol))]

    def _set(self, client_algo_id: str, status: str) -> None:
        with self.journal.connect() as db:
            db.execute("UPDATE protective_stops SET status=? WHERE scope=? AND client_algo_id=?",
                       (status, self.executor.scope, client_algo_id))

    def _position(self) -> tuple[Decimal, Decimal, str]:
        checkpoint = self.journal.checkpoint(self.executor.owner)
        if not checkpoint:
            return Decimal(0), Decimal(0), ""
        position = checkpoint["position"]
        return Decimal(position["qty"]), Decimal(position["entry_price"]), str(checkpoint.get("revision") or "")

    # ------------------------------------------------------------------
    def sync(self) -> list[str]:
        """Bring the exchange stop in line with the confirmed position. Returns issues."""
        issues: list[str] = []
        qty, entry, revision = self._position()
        current = None
        for row in self._rows():
            state = self.adapter.get_algo_order(client_algo_id=row["client_algo_id"])
            status = state.status if state is not None else None
            if status in TRIGGERED:
                if state.actual_order_id:
                    self._adopt(row, state)
                self._set(row["client_algo_id"], "triggered")
                issues.append(f"{self.symbol}: protective stop triggered")
                continue
            if status is None and row["status"] == "unknown" and self.clock() - row["created_at"] < 90:
                current = row  # ambiguous placement; keep looking it up before replacing
                continue
            if status not in ACTIVE:
                self._set(row["client_algo_id"], "inactive")
                continue
            if row["status"] == "canceling":
                # A stale stop must never survive to close a later position.
                self.adapter.cancel_algo_order(client_algo_id=row["client_algo_id"])
                continue
            if qty == 0 or row["revision"] != revision or current is not None:
                self.adapter.cancel_algo_order(client_algo_id=row["client_algo_id"])
                self._set(row["client_algo_id"], "canceling")
                continue
            if row["status"] != "active":
                self._set(row["client_algo_id"], "active")
            current = row
        if qty == 0 or current is not None:
            return issues
        if entry <= 0:
            return issues + [f"{self.symbol}: confirmed position has no entry price; stop not placed"]
        side = "sell" if qty > 0 else "buy"
        trigger = entry * (1 - self.stop_pct) if qty > 0 else entry * (1 + self.stop_pct)
        attempts = [r["attempt"] for r in self._rows(live_only=False) if r["revision"] == revision]
        attempt = (max(attempts) + 1) if attempts else 0
        client_algo_id = self._client_id(revision, attempt)
        with self.journal.connect() as db:
            db.execute("INSERT INTO protective_stops VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (self.executor.scope, self.executor.owner, client_algo_id, self.symbol, revision,
                        attempt, side, str(trigger), "unknown", float(self.clock())))
        result = self.adapter.place_stop_market(self.symbol, side, trigger, client_algo_id=client_algo_id)
        if result.get("status") == "submitted":
            self._set(client_algo_id, "active")
            logger.info("%s protective stop %s %s @ %s", self.symbol, side, client_algo_id, trigger)
        elif result.get("status") == "rejected":
            self._set(client_algo_id, "rejected")
            if result.get("code") == _IMMEDIATE_TRIGGER:
                issues.append(f"{self.symbol}: stop_breached (price already beyond {trigger})")
            else:
                issues.append(f"{self.symbol}: protective stop rejected: {result.get('msg')}")
        else:
            issues.append(f"{self.symbol}: protective stop placement outcome unknown; will query")
        return issues

    def _adopt(self, row: dict, state: Any) -> None:
        order = self.adapter.get_order(self.symbol, order_id=state.actual_order_id)
        if order is None:
            return
        self.executor.adopt_exchange_order(
            intent_id=f"stop-{row['client_algo_id']}", symbol=self.symbol, side=order.side,
            order_id=order.order_id, client_order_id=order.client_order_id, reason="protective_stop")

    def has_active_stop(self) -> bool:
        return any(r["status"] == "active" for r in self._rows())
