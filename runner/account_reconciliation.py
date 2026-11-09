"""Strict, read-only Bybit cash-flow collection for the durable wallet ledger."""
from __future__ import annotations

from decimal import Decimal
import os
from pathlib import Path
import time
from typing import Any, Mapping

from shared.account_ledger import AccountLedger
from shared.runtime_state import MANIFEST_NAME, _json, validate_runtime_state

_MAX_WINDOW_MS = 7 * 24 * 60 * 60 * 1000


def account_ledger_preflight(environ: Mapping[str, str] | None = None) -> AccountLedger:
    """Open an explicitly initialized ledger before any account connection."""
    env = os.environ if environ is None else environ
    scope = env.get("RISK_SCOPE", "").strip()
    state_text = env.get("RUNTIME_STATE_DIR", "").strip()
    if not scope or not state_text:
        raise ValueError("Execution requires identified runtime state for wallet reconciliation")
    state_dir = Path(state_text)
    journal = validate_runtime_state(state_dir, scope, require_writable=False)
    manifest = _json((state_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    return AccountLedger(journal, scope=scope, state_id=manifest["state_id"])


def _wallet(snapshot: Mapping[str, Any], currency: str) -> tuple[float, Decimal]:
    if not isinstance(snapshot, Mapping) or set(snapshot) != {"observed_at", "balances"}:
        raise ValueError("Malformed wallet snapshot")
    observed = float(snapshot["observed_at"])
    balances = snapshot["balances"]
    if not observed or not isinstance(balances, Mapping) or set(balances) != {currency}:
        raise ValueError("Wallet snapshot does not cover the ledger currency exactly")
    balance = Decimal(str(balances[currency]))
    if not balance.is_finite():
        raise ValueError("Wallet snapshot balance is invalid")
    return observed, balance


def reconcile_bybit_wallet(
    ledger: AccountLedger, adapter: Any, *, clock=time.time, full: bool = False,
) -> dict[str, Any]:
    """Collect complete bounded windows, require a stable wallet, then commit.

    A seven-day overlap is used during normal operation because Bybit documents
    delayed transaction delivery. A mismatch remains blocked and the next call
    performs a full replay from the immutable baseline.
    """
    try:
        if getattr(adapter, "venue", None) != "bybit":
            raise ValueError("Wallet collector currently supports Bybit only")
        status = ledger.snapshot()
        # Persistently block admission before the first account request. A
        # crash, timeout or failed invalidation write cannot leave old success
        # usable by this process or a restart.
        ledger.invalidate("wallet reconciliation in progress")
        now_ms = int(float(clock()) * 1000)
        before_observed, before_balance = _wallet(adapter.get_wallet_snapshot(), ledger.currency)
        cutoff = min(now_ms, int(before_observed * 1000))
        if cutoff <= ledger.baseline_start_ms:
            raise ValueError("Wallet collection cutoff does not follow its baseline")
        if full or not status["ready"]:
            start = ledger.baseline_start_ms
        else:
            start = max(ledger.baseline_start_ms, int(status["verified_through_ms"]) - _MAX_WINDOW_MS)
        transactions: list[Mapping[str, Any]] = []
        cursor = start
        while cursor <= cutoff:
            end = min(cutoff, cursor + _MAX_WINDOW_MS - 1)
            transactions.extend(adapter.get_transaction_log(start_ms=cursor, end_ms=end))
            if end == cutoff:
                break
            # One-millisecond overlap prevents an undocumented inclusive/exclusive
            # boundary from losing a transaction; durable IDs make it harmless.
            cursor = end
        after_observed, after_balance = _wallet(adapter.get_wallet_snapshot(), ledger.currency)
        if after_observed < before_observed or after_balance != before_balance:
            raise ValueError("Wallet changed during transaction collection")
        return ledger.ingest_and_reconcile(
            transactions, start_ms=start, end_ms=cutoff, wallet_balance=before_balance,
            currency=ledger.currency, observed_at=after_observed,
        )
    except Exception:
        ledger.invalidate("wallet collection or reconciliation failed")
        raise
