"""What a problem found during a cycle does to trading (ADR-0014): one fixed table from kind to effects.

The kind decides; the text of a problem is for people only. The table is pinned row by row by
tests/unit/trading/test_issues.py, and tests/architecture/test_issue_rules.py keeps text searches out of the
application layer.
"""
from __future__ import annotations

from enum import Enum


class IssueKind(Enum):
    STOP_BREACHED = "stop_breached"          # the stop could not be placed: the price is already beyond it
    STOP_TRIGGERED = "stop_triggered"        # the exchange stop went off
    STOP_MISSING = "stop_missing"            # a confirmed position without an active exchange stop
    STOP_UNCERTAIN = "stop_uncertain"        # a stop's outcome is not known yet
    LEDGER_MISMATCH = "ledger_mismatch"      # the venue position differs from the ledger
    FOREIGN_HOLDING = "foreign_holding"      # an open order the journal does not know
    UNRESOLVED_ORDER = "unresolved_order"    # a journal order whose outcome is not settled
    UNDELIVERED_FILL = "undelivered_fill"    # a confirmed fill the ledger has not consumed
    RECONCILE_FAILED = "reconcile_failed"    # reconciliation itself failed
    VENUE_BLOCKED = "venue_blocked"          # the venue adapter refuses execution (not connected, hedge mode)
    PAUSE_NOT_SAVED = "pause_not_saved"      # an entry pause could not be written
    MARKET_REFUSED = "market_refused"        # the exchange calls a market-data error final
    MARKET_DATA = "market_data"              # market data unavailable for now (retried with backoff)
    FLATTEN_FAILED = "flatten_failed"        # a flatten raised; it is tried again while its cause lasts
    CANCEL_FAILED = "cancel_failed"          # cancelling resting entries raised; tried again likewise
    DECISION_FAILED = "decision_failed"      # one symbol's bar could not be processed; the next bar tries again
    CYCLE_FAILED = "cycle_failed"            # a whole block of the cycle (market, exit or status side) raised
    EVENT_LOG_FAILED = "event_log_failed"    # the event log cannot be written; exits go on (S3 audit finding 3)


class Effect(Enum):
    BLOCK_SYMBOL_ENTRIES = "block_symbol_entries"
    BLOCK_ALL_ENTRIES = "block_all_entries"
    FLATTEN_SYMBOL = "flatten_symbol"
    PAUSE_SYMBOL = "pause_symbol"            # written from the durable stop rows, so a restart cannot lose it
    ALERT_ONLY = "alert_only"                # no trading effect; a lasting issue is sent to a person


_BLOCK = frozenset({Effect.BLOCK_SYMBOL_ENTRIES})

EFFECTS: dict[IssueKind, frozenset[Effect]] = {
    IssueKind.STOP_BREACHED: frozenset({Effect.FLATTEN_SYMBOL, Effect.PAUSE_SYMBOL, Effect.BLOCK_SYMBOL_ENTRIES}),
    IssueKind.STOP_TRIGGERED: frozenset({Effect.PAUSE_SYMBOL, Effect.BLOCK_SYMBOL_ENTRIES}),
    IssueKind.STOP_MISSING: _BLOCK,
    IssueKind.STOP_UNCERTAIN: _BLOCK,
    IssueKind.LEDGER_MISMATCH: _BLOCK,
    IssueKind.FOREIGN_HOLDING: _BLOCK,
    IssueKind.UNRESOLVED_ORDER: _BLOCK,
    IssueKind.UNDELIVERED_FILL: _BLOCK,
    IssueKind.RECONCILE_FAILED: _BLOCK,
    IssueKind.VENUE_BLOCKED: _BLOCK,
    IssueKind.PAUSE_NOT_SAVED: _BLOCK,
    IssueKind.MARKET_REFUSED: frozenset({Effect.BLOCK_ALL_ENTRIES}),
    IssueKind.MARKET_DATA: frozenset({Effect.ALERT_ONLY}),
    IssueKind.FLATTEN_FAILED: frozenset({Effect.ALERT_ONLY}),
    IssueKind.CANCEL_FAILED: frozenset({Effect.ALERT_ONLY}),
    IssueKind.DECISION_FAILED: _BLOCK,
    IssueKind.CYCLE_FAILED: frozenset({Effect.BLOCK_ALL_ENTRIES}),
    IssueKind.EVENT_LOG_FAILED: frozenset({Effect.BLOCK_ALL_ENTRIES}),
}


def effects(kind: IssueKind) -> frozenset[Effect]:
    """The effects of ``kind``. A kind missing from the table blocks every entry (ADR-0014: fail closed)."""
    return EFFECTS.get(kind, frozenset({Effect.BLOCK_ALL_ENTRIES}))


__all__ = ["EFFECTS", "Effect", "IssueKind", "effects"]
