"""Immutable wallet baseline and idempotent exchange cash-flow reconciliation.

The strategy fill projection and this account ledger answer different questions.
This ledger uses Bybit transaction ``change`` once, including funding and fees;
adding fill fees or summing individual strategy balances here would double count.
Initialization is explicit and requires an existing identified order journal.
"""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation, localcontext
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

_TYPES = frozenset({"TRADE", "SETTLEMENT", "TRANSFER_IN", "TRANSFER_OUT", "FEE_REFUND"})
_NUMBERS = ("change", "cashFlow", "funding", "fee", "cashBalance", "bonusChange")
_IDENTITY = ("scope", "state_id", "currency", "baseline_balance", "baseline_start_ms", "source",
             "max_age_seconds", "version")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        raise ValueError(f"Invalid wallet {name}")
    return value


def _decimal(value: Any, name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Invalid wallet {name}")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid wallet {name}") from exc
    if (not number.is_finite() or abs(number.as_tuple().exponent) > 100
            or abs(number.adjusted()) > 100):
        raise ValueError(f"Invalid wallet {name}")
    return number


def _number_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    result = format(value, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def _ms(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not re.fullmatch(r"[0-9]+", str(value)):
        raise ValueError(f"Invalid wallet {name}")
    result = int(value)
    if result <= 0 or result > 2**63 - 1:
        raise ValueError(f"Invalid wallet {name}")
    return result


def _seconds(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Invalid wallet {name}")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"Invalid wallet {name}")
    return result


def _path(path: Path) -> Path:
    result = Path(path).absolute()
    if ".." in result.parts or any(p.is_symlink() for p in (result, *result.parents)):
        raise ValueError("Wallet journal paths cannot contain traversal or symbolic links")
    if not result.is_file():
        raise ValueError("Wallet requires an existing order journal; refusing an empty replacement")
    if any(Path(str(result) + suffix).is_symlink() for suffix in ("-wal", "-shm", "-journal")):
        raise ValueError("Wallet journal sidecars cannot be symbolic links")
    return result


def _runtime_identity(db: sqlite3.Connection, scope: str, state_id: str) -> None:
    rows = [tuple(row) for row in db.execute("SELECT singleton,state_id,scope FROM runtime_identity")]
    if rows != [(1, state_id, scope)]:
        raise ValueError("Wallet runtime identity mismatch")
    db.execute("SELECT scope,intent_id FROM submissions LIMIT 0")


def _json_object(value: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError("Wallet transaction JSON contains duplicate keys")
            result[key] = item
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"Invalid wallet JSON constant {value}")

    try:
        result = json.loads(value, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Invalid durable wallet transaction JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("Durable wallet transaction must be a JSON object")
    return result


def _transaction(raw: Mapping[str, Any], currency: str, start_ms: int, end_ms: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("Malformed wallet transaction")
    result = {"id": _text(raw.get("id"), "transaction ID"),
              "currency": _text(raw.get("currency"), "currency"),
              "type": _text(raw.get("type"), "transaction type"),
              "transactionTime": _ms(raw.get("transactionTime"), "transaction time")}
    if result["currency"] != currency or result["type"] not in _TYPES:
        raise ValueError("Unsupported wallet currency or transaction type")
    if not start_ms <= result["transactionTime"] <= end_ms:
        raise ValueError("Wallet transaction lies outside the collected window")
    category = raw.get("category")
    if category not in ("", "linear") or (result["type"] in {"TRADE", "SETTLEMENT"} and category != "linear"):
        raise ValueError("Unsupported wallet transaction category")
    result["category"] = category
    for field in ("transSubType", "extraFees"):
        if raw.get(field, "") != "":
            raise ValueError("Unsupported wallet transaction subtype or extra fees")
    for field in ("borrowAmount", "interest", "loanAmount"):
        if _decimal(raw.get(field, "0"), field) != 0:
            raise ValueError("Wallet borrowing is not supported")
    numbers = {name: _decimal(raw.get(name, "0") if name == "bonusChange" else raw.get(name), name)
               for name in _NUMBERS}
    if numbers["bonusChange"] != 0:
        raise ValueError("Wallet bonus transactions are not supported")
    with localcontext() as context:
        context.prec = 512
        if numbers["change"] != numbers["cashFlow"] + numbers["funding"] - numbers["fee"]:
            raise ValueError("Wallet change contradicts cash flow, funding and fee")
    result.update({name: _number_text(value) for name, value in numbers.items()})
    # Preserve the identities whose conflicts could change trade attribution.
    for name in ("symbol", "tradeId", "orderId", "orderLinkId", "side"):
        value = raw.get(name, "")
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError(f"Invalid wallet transaction {name}")
        result[name] = value
    return result


class AccountLedger:
    """A scope-level cash ledger in the existing execution SQLite database."""

    def __init__(self, path: Path, *, scope: str, state_id: str,
                 clock: Callable[[], float] = time.time) -> None:
        self.path, self.scope = _path(path), _text(scope, "scope")
        self.state_id, self.clock = _text(state_id, "state identity"), clock
        self._failure_latch: str | None = None
        if not re.fullmatch(r"[0-9a-f]{32}", self.state_id):
            raise ValueError("Invalid wallet state identity")
        try:
            with self._connect(check_metadata=False) as db:
                rows = db.execute("SELECT * FROM wallet_metadata").fetchall()
                if len(rows) != 1:
                    raise ValueError("Wallet requires exactly one immutable account baseline")
                metadata = dict(rows[0])
                if (metadata["scope"] != self.scope or metadata["state_id"] != self.state_id
                        or metadata["version"] != 1 or metadata["currency"] != "USDT"):
                    raise ValueError("Wallet baseline identity or version mismatch")
                self.currency = metadata["currency"]
                self.baseline_balance = _decimal(metadata["baseline_balance"], "baseline balance")
                self.baseline_start_ms = _ms(metadata["baseline_start_ms"], "baseline time")
                self.max_age_seconds = _seconds(metadata["max_age_seconds"], "maximum age")
                _text(metadata["source"], "baseline source")
                self._identity = {name: metadata[name] for name in _IDENTITY}
                db.execute("SELECT scope,transaction_id,transaction_ms,payload FROM wallet_transactions LIMIT 0")
                state = db.execute("SELECT * FROM wallet_reconciliation").fetchall()
                if len(state) != 1 or state[0]["scope"] != self.scope:
                    raise ValueError("Wallet reconciliation state missing or foreign")
                if db.execute("SELECT 1 FROM wallet_transactions WHERE scope!=? LIMIT 1", (self.scope,)).fetchone():
                    raise ValueError("Wallet contains foreign account transactions")
                self._validate_state(state[0])
                expected, _ = self._stored_facts(db)
                if expected != _decimal(state[0]["expected_balance"], "expected balance"):
                    raise ValueError("Wallet reconciliation does not match durable transactions")
        except sqlite3.Error as exc:
            raise ValueError("Wallet ledger schema is missing or invalid") from exc

    @classmethod
    def initialize(cls, path: Path, *, scope: str, state_id: str, balance: Decimal,
                   start_ms: int, source: str, currency: str = "USDT", max_age_seconds: float = 120,
                   clock: Callable[[], float] = time.time) -> AccountLedger:
        """Record an explicitly established balance as of start_ms, never replace it.

        Transactions at or before that timestamp belong to the baseline. The
        operator must establish this cutoff from a complete account statement;
        a present wallet response cannot reconstruct a historical baseline.
        """
        path, scope, state_id = _path(path), _text(scope, "scope"), _text(state_id, "state identity")
        if not re.fullmatch(r"[0-9a-f]{32}", state_id) or currency != "USDT":
            raise ValueError("Unsupported wallet identity or baseline currency")
        balance, start_ms = _decimal(balance, "baseline balance"), _ms(start_ms, "baseline time")
        source = _text(source, "baseline source")
        max_age_seconds = _seconds(max_age_seconds, "maximum age")
        now = _seconds(clock(), "clock")
        if start_ms > int(now * 1000):
            raise ValueError("Wallet baseline cannot be in the future")
        db = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=10)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                db.execute("BEGIN IMMEDIATE")
                _runtime_identity(db, scope, state_id)
                existing = db.execute("SELECT name FROM sqlite_master WHERE name LIKE 'wallet_%'").fetchall()
                if existing:
                    raise ValueError("Wallet state already exists; baseline reset is forbidden")
                db.execute("CREATE TABLE wallet_metadata (scope TEXT PRIMARY KEY,state_id TEXT NOT NULL,"
                           "currency TEXT NOT NULL,baseline_balance TEXT NOT NULL,baseline_start_ms INTEGER NOT NULL,"
                           "source TEXT NOT NULL,max_age_seconds REAL NOT NULL,version INTEGER NOT NULL)")
                db.execute("CREATE TABLE wallet_transactions (scope TEXT NOT NULL,transaction_id TEXT NOT NULL,"
                           "transaction_ms INTEGER NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(scope,transaction_id))")
                db.execute("CREATE TABLE wallet_reconciliation (scope TEXT PRIMARY KEY,"
                           "verified_through_ms INTEGER NOT NULL,"
                           "checked_at REAL,observed_through_ms INTEGER,ready INTEGER NOT NULL,reason TEXT NOT NULL,"
                           "expected_balance TEXT NOT NULL,actual_balance TEXT)")
                db.execute("INSERT INTO wallet_metadata VALUES (?,?,?,?,?,?,?,1)",
                           (scope, state_id, currency, _number_text(balance), start_ms, source, max_age_seconds))
                db.execute("INSERT INTO wallet_reconciliation VALUES (?,?,NULL,NULL,0,?,?,NULL)",
                           (scope, start_ms, "baseline_unreconciled", _number_text(balance)))
        finally:
            db.close()
        return cls(path, scope=scope, state_id=state_id, clock=clock)

    @contextmanager
    def _connect(self, *, check_metadata: bool = True) -> Iterator[sqlite3.Connection]:
        _path(self.path)
        db = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                _runtime_identity(db, self.scope, self.state_id)
                if check_metadata:
                    rows = db.execute("SELECT * FROM wallet_metadata").fetchall()
                    if len(rows) != 1 or any(rows[0][name] != value for name, value in self._identity.items()):
                        raise ValueError("Immutable wallet baseline changed; recovery required")
                yield db
        finally:
            db.close()

    def invalidate(self, reason: str) -> None:
        reason = _text(reason, "blocked reason")
        # Set memory first. If SQLite cannot persist the failure (disk full,
        # I/O error, lock failure), this running gate must still fail closed.
        self._failure_latch = reason
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE wallet_reconciliation SET ready=0,reason=? WHERE scope=?", (reason, self.scope))

    def _validate_state(self, row: sqlite3.Row) -> None:
        verified = _ms(row["verified_through_ms"], "verified watermark")
        if verified < self.baseline_start_ms or row["ready"] not in (0, 1):
            raise ValueError("Invalid wallet reconciliation state")
        expected = _decimal(row["expected_balance"], "expected balance")
        observed = row["observed_through_ms"]
        checked = row["checked_at"]
        actual = row["actual_balance"]
        reason = row["reason"]
        if not isinstance(reason, str) or "\x00" in reason:
            raise ValueError("Invalid wallet reconciliation reason")
        if observed is not None:
            observed = _ms(observed, "observed watermark")
            if observed < verified:
                raise ValueError("Wallet observed watermark precedes verified state")
        if checked is not None:
            _seconds(checked, "check time")
        if actual is not None:
            _decimal(actual, "actual balance")
        if row["ready"]:
            if reason or checked is None or actual is None or observed != verified:
                raise ValueError("Ready wallet reconciliation state is incomplete")
            if _decimal(actual, "actual balance") != expected:
                raise ValueError("Ready wallet reconciliation balances differ")
        elif not reason:
            raise ValueError("Blocked wallet reconciliation requires a reason")

    def _stored_facts(self, db: sqlite3.Connection) -> tuple[Decimal, int]:
        expected = self.baseline_balance
        count = 0
        with localcontext() as context:
            context.prec = 512
            for row in db.execute(
                "SELECT transaction_id,transaction_ms,payload FROM wallet_transactions "
                "WHERE scope=? ORDER BY transaction_id", (self.scope,),
            ):
                raw = _json_object(row["payload"])
                stamp = _ms(raw.get("transactionTime"), "stored transaction time")
                item = _transaction(raw, self.currency, stamp, stamp)
                canonical = json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
                if (row["transaction_id"] != item["id"] or row["transaction_ms"] != stamp
                        or row["payload"] != canonical or stamp <= self.baseline_start_ms):
                    raise ValueError("Durable wallet transaction identity or payload changed")
                expected += _decimal(item["change"], "stored change")
                count += 1
        return expected, count

    def ingest_and_reconcile(self, transactions: Sequence[Mapping[str, Any]], *, start_ms: int,
                             end_ms: int, wallet_balance: Decimal, currency: str = "USDT",
                             observed_at: float | None = None) -> dict[str, Any]:
        """Commit complete collected facts and compare the single wallet once.

        The caller must finish every REST page and obtain a stable wallet sample
        before calling. A mismatch retains valid facts but does not advance the
        verified watermark. Malformed/conflicting batches commit no new facts.
        """
        failure = None
        self._failure_latch = "wallet reconciliation in progress"
        try:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                try:
                    now = _seconds(self.clock(), "clock")
                    observed = now if observed_at is None else _seconds(observed_at, "observation time")
                    start, end = _ms(start_ms, "window start"), _ms(end_ms, "window end")
                    actual = _decimal(wallet_balance, "balance")
                    if currency != self.currency or not isinstance(transactions, (tuple, list)):
                        raise ValueError("Unsupported wallet currency or incomplete transaction batch")
                    previous = db.execute(
                        "SELECT verified_through_ms FROM wallet_reconciliation WHERE scope=?", (self.scope,),
                    ).fetchone()[0]
                    if start > previous or end < previous or start > end:
                        raise ValueError("Wallet collection window leaves a gap or moves backwards")
                    if (not 0 <= now - observed <= self.max_age_seconds
                            or not 0 <= now - end / 1000 <= self.max_age_seconds):
                        raise ValueError("Wallet observation or collected window is stale or in the future")
                    normalized: dict[str, tuple[int, str]] = {}
                    for raw in transactions:
                        item = _transaction(raw, self.currency, start, end)
                        if item["transactionTime"] <= self.baseline_start_ms:
                            continue
                        payload = json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
                        identity = item["id"]
                        value = (item["transactionTime"], payload)
                        if identity in normalized and normalized[identity] != value:
                            raise ValueError("Conflicting wallet transaction identity within batch")
                        normalized[identity] = value
                        old = db.execute(
                            "SELECT transaction_ms,payload FROM wallet_transactions "
                            "WHERE scope=? AND transaction_id=?", (self.scope, identity),
                        ).fetchone()
                        if old is not None and tuple(old) != value:
                            raise ValueError("Conflicting durable wallet transaction identity")
                    db.executemany(
                        "INSERT OR IGNORE INTO wallet_transactions VALUES (?,?,?,?)",
                        [(self.scope, identity, stamp, payload)
                         for identity, (stamp, payload) in normalized.items()],
                    )
                    expected, _ = self._stored_facts(db)
                    ready = expected == actual
                    db.execute(
                        "UPDATE wallet_reconciliation SET verified_through_ms=?,checked_at=?,"
                        "observed_through_ms=?,ready=?,reason=?,expected_balance=?,actual_balance=? WHERE scope=?",
                        (end if ready else previous, observed, end, int(ready),
                         "" if ready else "wallet_balance_mismatch",
                         _number_text(expected), _number_text(actual), self.scope),
                    )
                except Exception as exc:
                    # Do not let a rejected batch leave a valid prior status or any
                    # partially inserted facts. Invalidation is committed before unlock.
                    db.rollback()
                    db.execute("BEGIN IMMEDIATE")
                    reason = f"wallet_reconciliation_failed:{type(exc).__name__}"
                    self._failure_latch = reason
                    db.execute("UPDATE wallet_reconciliation SET ready=0,reason=? WHERE scope=?",
                               (reason, self.scope))
                    failure = exc
        except BaseException as exc:
            self._failure_latch = f"wallet_reconciliation_failed:{type(exc).__name__}"
            raise
        if failure is not None:
            raise failure
        self._failure_latch = None
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM wallet_reconciliation WHERE scope=?", (self.scope,)).fetchone()
            if row is None:
                raise ValueError("Wallet reconciliation state missing")
            self._validate_state(row)
            result = dict(row)
            expected, count = self._stored_facts(db)
            if expected != _decimal(result["expected_balance"], "expected balance"):
                raise ValueError("Wallet reconciliation does not match durable transactions")
        now = _seconds(self.clock(), "clock")
        ready, reason = bool(result["ready"]), result["reason"]
        if self._failure_latch is not None:
            ready, reason = False, self._failure_latch
        if ready:
            observed, through = result["checked_at"], result["verified_through_ms"]
            if (observed is None or not 0 <= now - observed <= self.max_age_seconds
                    or not 0 <= now - through / 1000 <= self.max_age_seconds):
                ready, reason = False, "wallet_reconciliation_stale"
        result.update(ready=ready, reason=reason, state_id=self.state_id, currency=self.currency,
                      baseline_start_ms=self.baseline_start_ms, baseline_balance=_number_text(self.baseline_balance),
                      transaction_count=count)
        return result

    def check(self) -> tuple[bool, str]:
        try:
            result = self.snapshot()
            return result["ready"], result["reason"]
        except Exception as exc:
            return False, f"wallet_state_unavailable:{type(exc).__name__}"
