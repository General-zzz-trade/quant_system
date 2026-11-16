"""The parts only a paper run plugs into the shared execution path (ADR-0013 plan B, S10 part 3).

None of them decides anything about trading. They give the path its venue (the simulated exchange over the research
brokers), the entry gates' clock on bar time, each symbol's own simulated equity for sizing, the paper directory's
state database with the paper state saved atomically in it, and the ``paper_divergence`` events.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from trading.adapters.files.state_files import JsonPaperStore
from trading.adapters.simulated.bar_broker import SYMBOL_RULES, BarBroker, broker_config
from trading.adapters.simulated.exchange import BarStopModel, SimFilters, SimulatedExchange
from trading.adapters.sqlite import schema
from trading.adapters.sqlite.paper_state import SqlitePaperState
from trading.application.ports import EntryPauses, EventLog
from trading.domain.events import EventKind
from trading.domain.issues import IssueKind
from trading.domain.orders import DUPLICATE
from trading.domain.strategy.overlay import StrategySpec
from trading.infrastructure.state_dir import MANIFEST, is_managed

logger = logging.getLogger(__name__)

PAPER_STATE = "paper_state.json"
PAPER_STATE_DIR = Path("data/runtime/paper")          # paper's default state directory (execution's is its own)
# A paper record exists when any of these is in the directory; its state database must then exist too.
PAPER_RECORD = (PAPER_STATE, "paper_state.sqlite")
STOP_KINDS = frozenset({IssueKind.STOP_TRIGGERED.value, IssueKind.STOP_BREACHED.value})


def paper_exchange(spec: StrategySpec, symbols: Sequence[str]) -> SimulatedExchange:
    """One research broker per symbol, configured exactly as research configures it (``broker_config`` and the
    research lot constants), behind the exchange ports; stops by the bar stop model (decision L)."""
    brokers = {s: BarBroker(broker_config(spec, SYMBOL_RULES[s])) for s in symbols}
    filters = {s: SimFilters(SYMBOL_RULES[s].step, SYMBOL_RULES[s].min_qty, SYMBOL_RULES[s].min_notional)
               for s in symbols}
    return SimulatedExchange(brokers, filters=filters, stop_model=BarStopModel())


class BarTimeClock:
    """``ModePolicy.limiter_clock == "bar_time"``: the entry gates' clock in paper (the ``GateClock`` port, and a
    ``Clock`` in seconds for the order limiter). The market side sets it to a bar's close and moves it on by one poll
    for the retry of deferred entries. It never goes back, so the per-second window stays monotonic when symbols
    catch up at different times."""

    def __init__(self) -> None:
        self._ms = 0

    def at(self, ms: int) -> None:
        self._ms = max(self._ms, int(ms))

    def advance(self, seconds: float) -> None:
        self._ms += int(seconds * 1000)

    def ms(self) -> int:
        return self._ms

    def __call__(self) -> float:
        return self._ms / 1000


class SymbolVenueEquity:
    """``ModePolicy.sizing_equity == "per_symbol_venue"``: a symbol sizes from its own simulated account, as research
    does (audit H1-3). Read at every decision; zero, never a guess, when the account is not positive."""

    def __init__(self, venue: SimulatedExchange, symbol: str) -> None:
        self.venue, self.symbol = venue, symbol

    def current(self) -> Decimal:
        equity = self.venue.get_symbol_equity(self.symbol)
        return equity if equity > 0 else Decimal(0)


class PaperDivergenceLog:
    """The event log of a paper run, and the only writer of ``paper_divergence`` events (ADR-0013 "intended
    divergences"): one follows each event that means paper has left research on purpose.

    - an entry the gates or the order journal refused for good (``blocked`` without ``retry``): ``entry_refused``.
      Not a reduction (its intent said ``reduce_only``) and not a repeated intent (the journal's ``duplicate``
      refusal, e.g. a bar decided again after a crash): research would not have placed either anew;
    - a protective stop that went off or could not be placed because price was beyond it (the ``issue`` of that
      kind, written once per stop): ``stop_triggered`` or ``stop_breached``;
    - a new entry pause (reported by ``DivergencePauses``): ``pause``;
    - a risk exit the exit side starts (reported by it once per symbol and cause): ``flatten`` for a kill switch or
      a latched drawdown, ``cancel_entries`` for a kill switch without flatten or an issue that cancels resting
      entries, each with its ``reason``.
    """

    def __init__(self, inner: EventLog) -> None:
        self.inner = inner
        self._reductions: set[str] = set()       # intents that said reduce_only, by order id

    def write(self, kind: EventKind, **fields: Any) -> None:
        self.inner.write(kind, **fields)
        if kind is EventKind.INTENT and fields.get("reduce_only"):
            self._reductions.add(str(fields.get("order_id")))
        if kind is EventKind.BLOCKED and not fields.get("retry") and fields.get("refusal") != DUPLICATE \
                and str(fields.get("order_id")) not in self._reductions:
            self._divergence("entry_refused", symbol=fields.get("symbol"), order_id=fields.get("order_id"),
                             reason=fields.get("reason"))
        elif kind is EventKind.ISSUE and fields.get("issue_kind") in STOP_KINDS:
            self._divergence(str(fields["issue_kind"]), symbol=fields.get("symbol"), issue=fields.get("issue"))

    def paused(self, symbol: str, cause: str) -> None:
        self._divergence("pause", symbol=symbol, reason=cause)

    def risk_exit(self, symbol: str, action: str, reason: str) -> None:
        self._divergence(action, symbol=symbol, reason=reason)

    def _divergence(self, cause: str, **fields: Any) -> None:
        self.inner.write(EventKind.PAPER_DIVERGENCE, cause=cause, **fields)


class DivergencePauses:
    """The entry pauses of a paper run: a pause that is new is reported to the divergence log."""

    def __init__(self, inner: EntryPauses, divergences: PaperDivergenceLog) -> None:
        self.inner, self.divergences = inner, divergences

    def pause(self, symbol: str, cause: str, triggers: Iterable[str] = ()) -> bool:
        new = self.inner.pause(symbol, cause, triggers=triggers)
        if new:
            self.divergences.paused(symbol, cause)
        return new

    def reason(self, symbol: str) -> str | None:
        return self.inner.reason(symbol)

    def covered(self, symbol: str) -> set[str]:
        return self.inner.covered(symbol)


class PaperStateStore:
    """The ``PaperStore`` of a paper run (S10 decision 4): the document in the state database, written atomically with
    the order journal (``SqlitePaperState`` and ``PaperOrderJournal``), and after every save a JSON copy beside it
    (``paper_state.json``; migration plan step 10: until the audit after step 15 says it may stop). The copy is
    never read back: the database is the one store. So a copy that cannot be written is logged and never fails the
    journal operation whose transaction has already committed."""

    def __init__(self, state: SqlitePaperState, copy: JsonPaperStore) -> None:
        self.state, self.copy = state, copy
        state.saved = self._copy

    def _copy(self, document: dict[str, Any]) -> None:
        try:
            self.copy.save(document)
        except Exception:
            logger.warning("Paper state copy %s not written; the state database holds the state", self.copy.path,
                           exc_info=True)

    def load(self) -> dict[str, Any] | None:
        return self.state.load()

    def save(self, document: dict[str, Any]) -> None:
        self.state.save(document)

    def archive(self) -> str | None:
        """Only the copy: a new paper record sets the whole database aside (``paper_state_db``)."""
        return self.copy.archive()


def refuse_execution_directory(state_dir: Path) -> None:
    """Paper never runs in, resets or creates anything in an execution state directory (one with a manifest)."""
    if is_managed(state_dir):
        raise RuntimeError(f"{state_dir} is an execution state directory (it has {MANIFEST}); paper uses a "
                           f"directory of its own, such as the default {PAPER_STATE_DIR}")


def paper_state_db(state_dir: Path, *, reset: bool, stamp_ms: int) -> tuple[Path, str | None]:
    """The paper directory's state database (ADR-0009: the order journal, stops, limits, pauses and drawdown, and
    the paper state).

    Paper creates it at the first start of a paper record; a reset sets the old one aside first, under a name with
    ``stamp_ms`` (returned). A missing database while a paper record exists is refused: its orders and positions
    would be lost, and a new paper record is started only by a person (``--reset-paper-state``).

    An execution state directory (one with a manifest) is refused before anything is touched, a reset included:
    its database holds an account's orders, pauses, drawdown latch and counts (tech lead 2026-11-16, S10 audit H1).
    """
    state_dir = Path(state_dir)
    refuse_execution_directory(state_dir)
    path = state_dir / schema.STATE_DB
    archived = None
    if reset and path.exists():
        n, aside = 0, state_dir / f"state.{stamp_ms}.sqlite"
        while aside.exists():
            n += 1
            aside = state_dir / f"state.{stamp_ms}.{n}.sqlite"
        for suffix in ("-wal", "-shm", ""):             # the database file last: its absence means "set aside"
            old = Path(str(path) + suffix)
            if old.exists():
                os.replace(old, Path(str(aside) + suffix))
        archived = aside.name
    if not path.exists():
        if not reset and (record := [name for name in PAPER_RECORD if (state_dir / name).exists()]):
            raise RuntimeError(f"paper state database {path} is missing but a paper record exists "
                               f"({', '.join(record)}); start a new paper record with --reset-paper-state")
        state_dir.mkdir(parents=True, exist_ok=True)
        schema.create(path, paper=True)
    return path, archived


__all__ = ["BarTimeClock", "DivergencePauses", "PAPER_STATE", "PAPER_STATE_DIR", "PaperDivergenceLog",
           "PaperStateStore", "SymbolVenueEquity", "paper_exchange", "paper_state_db", "refuse_execution_directory"]
