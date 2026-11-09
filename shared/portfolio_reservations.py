"""Durable, cross-process admission budgets backed by verified exposure epochs.

An epoch freezes reconciled account exposure. Every subsequently admitted entry
adds a reservation; fills, cancellation, timeouts and restarts never release it.
Only a complete, quiescent reconciliation replaces the baseline and covers the
old reservations in one write transaction. This deliberately favors a blocked
entry over guessing whether a cross-database crash submitted an order.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Callable, Iterator, Sequence

_TERMINAL = frozenset({"filled", "canceled", "mmp_canceled", "rejected", "expired"})


@dataclass(frozen=True)
class ScopeRegistration:
    scope: str
    venue: str
    environment: str
    state_id: str


@dataclass(frozen=True)
class PositionEvidence:
    symbol: str
    side: str
    quantity: Decimal
    notional: Decimal


@dataclass(frozen=True)
class SubmissionEvidence:
    intent_id: str
    symbol: str
    side: str
    quantity: Decimal
    settled: bool
    status: str


@dataclass(frozen=True)
class ConsumerPosition:
    owner: str
    symbol: str
    quantity: Decimal  # signed, as stored by the durable consumer


@dataclass(frozen=True)
class AccountEvidence:
    scope: str
    venue: str
    environment: str
    state_id: str
    observed_at: float
    positions: tuple[PositionEvidence, ...]
    submissions: tuple[SubmissionEvidence, ...]
    consumer_positions: tuple[ConsumerPosition, ...]
    open_order_ids: tuple[str, ...] = ()
    unconsumed_fill_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReservationResult:
    allowed: bool
    reason: str = ""
    epoch: int | None = None


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value or any(c.isspace() for c in value):
        raise ValueError(f"invalid {name}")
    return value


def _decimal(value: object, name: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"invalid {name}")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if (not number.is_finite() or (positive and number <= 0)
            or abs(number.as_tuple().exponent) > 100 or abs(number.adjusted()) > 100):
        raise ValueError(f"invalid {name}")
    return number


def _add(left: Decimal, right: Decimal) -> Decimal:
    # Do not inherit the process's default 28-digit context: rounding a tiny
    # positive reservation away at the cap is still an admission error.
    with localcontext() as context:
        context.prec = 512
        return left + right


def _timestamp(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("invalid observation time")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("invalid observation time")
    return result


def _side(value: str) -> str:
    if value not in ("buy", "sell"):
        raise ValueError("invalid side")
    return value


def _path(path: Path) -> Path:
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("portfolio state cannot use symbolic links")
    for suffix in ("-wal", "-shm", "-journal"):
        if Path(str(path) + suffix).is_symlink():
            raise ValueError("portfolio sidecars cannot use symbolic links")
    return path


class PortfolioReservationStore:
    """Existing-state-only runtime handle; initialization is an explicit action."""

    def __init__(self, path: Path, *, group: str, clock: Callable[[], float] = time.time) -> None:
        self.path, self.group, self.clock = _path(path), _text(group, "group"), clock
        with self._connect(check_identity=False) as db:
            metadata = dict(db.execute("SELECT key,value FROM portfolio_metadata"))
            if metadata.get("version") != "1" or metadata.get("group") != self.group:
                raise ValueError("portfolio database identity mismatch")
            registrations = json.loads(metadata["registrations"])
            self.registrations = tuple(ScopeRegistration(**r) for r in registrations)
            self.caps = {k: _decimal(v, "cap", positive=True)
                         for k, v in json.loads(metadata["caps"]).items()}
            self.max_age_seconds = _timestamp(metadata["max_age_seconds"])
            self._identity = {k: metadata[k] for k in
                              ("version", "group", "registrations", "caps", "max_age_seconds")}
            self._validate_configuration(self.registrations, self.caps)
            # Missing or incompatible tables must fail at construction, not on
            # the first order after connecting to an account.
            db.execute("SELECT id,observed_at,baseline FROM portfolio_epochs LIMIT 0")
            db.execute("SELECT scope,intent_id,epoch,symbol,side,notional,quantity,covered_epoch "
                       "FROM portfolio_reservations LIMIT 0")

    @staticmethod
    def _validate_configuration(registrations: Sequence[ScopeRegistration], caps: dict) -> None:
        if not registrations or not caps:
            raise ValueError("explicit scopes and caps are required")
        seen = set()
        for registration in registrations:
            if type(registration) is not ScopeRegistration:
                raise ValueError("invalid scope registration")
            for name, value in asdict(registration).items():
                _text(value, name)
            if registration.scope in seen:
                raise ValueError("duplicate registered scope")
            seen.add(registration.scope)
        for symbol, cap in caps.items():
            if _text(symbol, "symbol") != symbol.upper():
                raise ValueError("symbols must use canonical uppercase names")
            _decimal(cap, "cap", positive=True)

    @classmethod
    def initialize(cls, path: Path, *, group: str, registrations: Sequence[ScopeRegistration],
                   caps: dict[str, Decimal], max_age_seconds: float = 300,
                   clock: Callable[[], float] = time.time) -> PortfolioReservationStore:
        """Create a new empty budget. No entry is eligible until reconciliation."""
        path, group = _path(path), _text(group, "group")
        cls._validate_configuration(registrations, caps)
        max_age_seconds = _timestamp(max_age_seconds)
        metadata = {
            "version": "1", "group": group,
            "registrations": json.dumps([asdict(r) for r in registrations], sort_keys=True),
            "caps": json.dumps({k: str(_decimal(v, "cap", positive=True)) for k, v in caps.items()},
                               sort_keys=True),
            "max_age_seconds": str(max_age_seconds), "blocked_reason": "baseline_uninitialized",
        }
        if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
            raise ValueError("existing portfolio sidecars require recovery, not initialization")
        # Exclusive creation prevents an operator typo from resetting budgets.
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        db = sqlite3.connect(path)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            with db:
                db.execute("CREATE TABLE portfolio_metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
                db.execute("CREATE TABLE portfolio_epochs (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                           "observed_at REAL NOT NULL,baseline TEXT NOT NULL)")
                db.execute("CREATE TABLE portfolio_reservations (scope TEXT NOT NULL,intent_id TEXT NOT NULL,"
                           "epoch INTEGER NOT NULL,symbol TEXT NOT NULL,side TEXT NOT NULL,"
                           "notional TEXT NOT NULL,quantity TEXT NOT NULL,covered_epoch INTEGER,"
                           "PRIMARY KEY(scope,intent_id))")
                db.executemany("INSERT INTO portfolio_metadata VALUES (?,?)", metadata.items())
        finally:
            db.close()
        return cls(path, group=group, clock=clock)

    @contextmanager
    def _connect(self, *, check_identity: bool = True) -> Iterator[sqlite3.Connection]:
        _path(self.path)
        if not self.path.is_file():
            raise ValueError("portfolio state missing; refusing an empty replacement")
        db = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                if check_identity:
                    metadata = dict(db.execute("SELECT key,value FROM portfolio_metadata"))
                    if any(metadata.get(k) != value for k, value in self._identity.items()):
                        raise ValueError("portfolio database configuration changed")
                yield db
        finally:
            db.close()

    def _registration(self, scope: str, venue: str, environment: str, state_id: str) -> None:
        registration = ScopeRegistration(scope, venue, environment, state_id)
        if registration not in self.registrations:
            raise ValueError("undeclared scope, venue, environment or runtime state")

    def _current(self, db: sqlite3.Connection) -> tuple[int, float, dict]:
        row = db.execute("SELECT * FROM portfolio_epochs ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            raise ValueError("baseline_uninitialized")
        age = _timestamp(self.clock()) - _timestamp(row["observed_at"])
        if age < 0 or age > self.max_age_seconds:
            raise ValueError("portfolio_baseline_stale")
        baseline = json.loads(row["baseline"])
        if not isinstance(baseline, dict) or set(baseline) != set(self.caps):
            raise ValueError("portfolio_baseline_invalid")
        for sides in baseline.values():
            if not isinstance(sides, dict) or set(sides) != {"buy", "sell"}:
                raise ValueError("portfolio_baseline_invalid")
            if any(_decimal(value, "baseline") < 0 for value in sides.values()):
                raise ValueError("portfolio_baseline_invalid")
        return row["id"], age, baseline

    def reserve(self, *, scope: str, venue: str, environment: str, state_id: str,
                intent_id: str, symbol: str, side: str, notional: Decimal,
                quantity: Decimal) -> ReservationResult:
        """Commit admission before the caller writes its submission journal."""
        try:
            self._registration(scope, venue, environment, state_id)
            _text(intent_id, "intent id")
            if symbol not in self.caps:
                raise ValueError("portfolio_symbol_unregistered")
            side = _side(side)
            notional = _decimal(notional, "notional", positive=True)
            quantity = _decimal(quantity, "quantity", positive=True)
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                blocked = db.execute("SELECT value FROM portfolio_metadata WHERE key='blocked_reason'").fetchone()
                if blocked is None or blocked[0]:
                    raise ValueError(blocked[0] if blocked else "portfolio_state_invalid")
                epoch, _, baseline = self._current(db)
                previous = db.execute("SELECT * FROM portfolio_reservations WHERE scope=? AND intent_id=?",
                                      (scope, intent_id)).fetchone()
                if previous is not None:
                    if (previous["symbol"] != symbol or previous["side"] != side
                            or Decimal(previous["notional"]) != notional
                            or Decimal(previous["quantity"]) != quantity):
                        raise ValueError("reservation_intent_conflict")
                    # Preserve identity even after a later epoch covers this intent.
                    return ReservationResult(True, "already_reserved", previous["epoch"])
                used = _decimal(baseline[symbol][side], "baseline")
                for row in db.execute(
                    "SELECT notional FROM portfolio_reservations WHERE symbol=? AND side=? "
                    "AND covered_epoch IS NULL", (symbol, side)):
                    used = _add(used, _decimal(row[0], "reservation", positive=True))
                if _add(used, notional) > self.caps[symbol]:
                    return ReservationResult(False, f"portfolio_cap {symbol} {side}: "
                                             f"{used}+{notional}>{self.caps[symbol]}", epoch)
                db.execute("INSERT INTO portfolio_reservations VALUES (?,?,?,?,?,?,?,NULL)",
                           (scope, intent_id, epoch, symbol, side, str(notional), str(quantity)))
                return ReservationResult(True, "", epoch)
        except (ValueError, TypeError, KeyError, sqlite3.Error, OSError, InvalidOperation) as exc:
            return ReservationResult(False, str(exc))

    def invalidate(self, reason: str) -> None:
        """Persist a failed reconciliation so a restart cannot forget it."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("invalidation reason is required")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE portfolio_metadata SET value=? WHERE key='blocked_reason'", (reason,))

    def reconcile(self, collect: Callable[[], Sequence[AccountEvidence]]) -> int:
        """Collect strict account/journal evidence while new admissions are locked.

        The callback must query all registered accounts and read each matching
        runtime journal. A cached monitor JSON or a boolean 'complete' flag is
        not accepted. Open orders, unconsumed fills, unresolved submissions and
        orphan reservations prevent epoch replacement. A failed observation
        also invalidates the old baseline, without deleting any reservation.
        """
        failure: Exception | None = None
        epoch = 0
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            started = _timestamp(self.clock())
            db.execute("SAVEPOINT reconcile_epoch")
            try:
                evidence = collect()
                finished = _timestamp(self.clock())
                baseline, observed_at = self._validate_evidence(db, evidence, started, finished)
                epoch = db.execute("INSERT INTO portfolio_epochs (observed_at,baseline) VALUES (?,?)",
                                   (observed_at, json.dumps(baseline, sort_keys=True))).lastrowid
                db.execute("UPDATE portfolio_reservations SET covered_epoch=? WHERE covered_epoch IS NULL", (epoch,))
                db.execute("UPDATE portfolio_metadata SET value='' WHERE key='blocked_reason'")
            except Exception as exc:
                # Commit the failure marker under the same admission lock, then
                # raise. No interval exists where stale success can admit an order.
                failure = exc
                db.execute("ROLLBACK TO reconcile_epoch")
                db.execute("UPDATE portfolio_metadata SET value=? WHERE key='blocked_reason'",
                           (f"reconciliation_failed:{type(exc).__name__}",))
            db.execute("RELEASE reconcile_epoch")
        if failure is not None:
            raise ValueError(f"portfolio reconciliation failed: {failure}") from failure
        return epoch

    def _validate_evidence(self, db: sqlite3.Connection, evidence: Sequence[AccountEvidence],
                           started: float, finished: float) -> tuple[dict, float]:
        if not isinstance(evidence, (tuple, list)) or len(evidence) != len(self.registrations):
            raise ValueError("account coverage incomplete")
        baseline = {symbol: {"buy": Decimal(0), "sell": Decimal(0)} for symbol in self.caps}
        scopes, submissions, observed = set(), {}, []
        for account in evidence:
            if type(account) is not AccountEvidence:
                raise ValueError("typed account evidence required")
            self._registration(account.scope, account.venue, account.environment, account.state_id)
            if account.scope in scopes:
                raise ValueError("duplicate account evidence")
            scopes.add(account.scope)
            timestamp = _timestamp(account.observed_at)
            if timestamp < started or timestamp > finished or finished - timestamp > self.max_age_seconds:
                raise ValueError("account observation is stale or predates reconciliation")
            observed.append(timestamp)
            collections = (account.positions, account.submissions, account.consumer_positions,
                           account.open_order_ids, account.unconsumed_fill_ids)
            if any(not isinstance(values, (tuple, list)) for values in collections):
                raise ValueError("malformed account evidence collections")
            if account.open_order_ids or account.unconsumed_fill_ids:
                raise ValueError("account has open orders or unconsumed fills")
            actual, expected, position_keys, consumer_keys = {}, {}, set(), set()
            for position in account.positions:
                if type(position) is not PositionEvidence or position.symbol not in self.caps:
                    raise ValueError("unregistered or invalid position")
                side = _side(position.side)
                key = (position.symbol, side)
                if key in position_keys:
                    raise ValueError("duplicate position evidence")
                position_keys.add(key)
                qty = _decimal(position.quantity, "position quantity", positive=True)
                amount = _decimal(position.notional, "position notional", positive=True)
                baseline[position.symbol][side] = _add(baseline[position.symbol][side], amount)
                actual[position.symbol] = _add(actual.get(position.symbol, Decimal(0)),
                                               qty if side == "buy" else qty.copy_negate())
            for consumer in account.consumer_positions:
                if type(consumer) is not ConsumerPosition or consumer.symbol not in self.caps:
                    raise ValueError("unregistered or invalid consumer position")
                key = (_text(consumer.owner, "consumer owner"), consumer.symbol)
                if key in consumer_keys:
                    raise ValueError("duplicate consumer position")
                consumer_keys.add(key)
                expected[consumer.symbol] = _add(expected.get(consumer.symbol, Decimal(0)), _decimal(
                    consumer.quantity, "consumer quantity"))
            for symbol in set(actual) | set(expected):
                if actual.get(symbol, 0) != expected.get(symbol, 0):
                    raise ValueError("venue and durable consumer quantities differ")
            for submission in account.submissions:
                if type(submission) is not SubmissionEvidence:
                    raise ValueError("invalid submission evidence")
                key = (account.scope, _text(submission.intent_id, "submission intent"))
                if key in submissions or submission.symbol not in self.caps:
                    raise ValueError("duplicate or unregistered submission")
                _side(submission.side)
                _decimal(submission.quantity, "submission quantity", positive=True)
                if submission.settled is not True or submission.status not in _TERMINAL:
                    raise ValueError("unresolved or unknown submission")
                submissions[key] = submission
        for reservation in db.execute("SELECT * FROM portfolio_reservations WHERE covered_epoch IS NULL"):
            submission = submissions.get((reservation["scope"], reservation["intent_id"]))
            if submission is None:
                raise ValueError("orphan reservation lacks terminal journal evidence")
            if (submission.symbol != reservation["symbol"] or submission.side != reservation["side"]
                    or _decimal(submission.quantity, "quantity") != Decimal(reservation["quantity"])):
                raise ValueError("reservation and submission identity differ")
        return ({symbol: {side: str(amount) for side, amount in sides.items()}
                 for symbol, sides in baseline.items()}, min(observed))

    def snapshot(self) -> dict:
        """Read-only JSON-safe operational status; reading never releases budget."""
        with self._connect() as db:
            db.execute("BEGIN")
            blocked = db.execute("SELECT value FROM portfolio_metadata WHERE key='blocked_reason'").fetchone()
            reason = blocked[0] if blocked else "portfolio_state_invalid"
            epoch, age, baseline = None, None, {}
            try:
                epoch, age, baseline = self._current(db)
            except ValueError as exc:
                reason = reason or str(exc)
            reservations = [dict(r) for r in db.execute(
                "SELECT * FROM portfolio_reservations ORDER BY scope,intent_id")]
            used = {symbol: dict(sides) for symbol, sides in baseline.items()}
            for reservation in reservations:
                if reservation["covered_epoch"] is None and reservation["symbol"] in used:
                    symbol, side = reservation["symbol"], reservation["side"]
                    used[symbol][side] = str(_add(Decimal(used[symbol][side]), Decimal(reservation["notional"])))
            return {"group": self.group, "ready": not reason, "reason": reason, "epoch": epoch,
                    "age_seconds": age, "baseline": baseline, "used": used,
                    "reservations": reservations, "registrations": [asdict(r) for r in self.registrations],
                    "caps": {symbol: str(cap) for symbol, cap in self.caps.items()}}
