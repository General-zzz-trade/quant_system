"""Initialize, inspect or reconcile an identified Bybit wallet cash ledger.

Initialization records an operator-supplied historical statement cutoff. It
does not infer a baseline from current equity. Reconciliation uses account GET
endpoints only and never places, cancels or modifies an order.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path

from runner.account_reconciliation import reconcile_bybit_wallet
from shared.account_ledger import AccountLedger
from shared.runtime_state import MANIFEST_NAME, RuntimeStateLease, _json, validate_runtime_state


def _identity(state_dir: Path, scope: str) -> tuple[Path, str]:
    journal = validate_runtime_state(state_dir, scope, require_writable=False)
    manifest = _json((state_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    return journal, manifest["state_id"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "inspect", "reconcile"))
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--balance", help="immutable USDT baseline from a complete statement")
    parser.add_argument("--start-ms", type=int, help="inclusive statement cutoff in epoch milliseconds")
    parser.add_argument("--source", help="non-secret statement/evidence reference")
    parser.add_argument("--full", action="store_true", help="replay from the immutable baseline")
    args = parser.parse_args(argv)
    journal, state_id = _identity(args.state_dir, args.scope)
    if args.action == "init":
        if args.balance is None or args.start_ms is None or not args.source:
            parser.error("init requires --balance, --start-ms and --source")
        # Initialization changes the account journal and therefore requires the
        # execution writer to be stopped for the duration of this short action.
        with RuntimeStateLease(args.state_dir, args.scope, journal_path=journal):
            ledger = AccountLedger.initialize(
                journal, scope=args.scope, state_id=state_id, balance=Decimal(args.balance),
                start_ms=args.start_ms, source=args.source,
            )
    else:
        ledger = AccountLedger(journal, scope=args.scope, state_id=state_id)
    if args.action == "reconcile":
        from execution.adapters.bybit.adapter import BybitAdapter
        from execution.adapters.bybit.config import BybitConfig

        key, secret = os.environ.get("BYBIT_API_KEY"), os.environ.get("BYBIT_API_SECRET")
        base_url = os.environ.get("BYBIT_BASE_URL", "https://api-demo.bybit.com")
        if not key or not secret:
            raise ValueError("Reconciliation requires Bybit read credentials")
        if base_url not in {
            "https://api-demo.bybit.com", "https://api-testnet.bybit.com", "https://api.bybit.com",
        }:
            raise ValueError("Reconciliation requires an exact recognized Bybit endpoint")
        adapter = BybitAdapter(BybitConfig(api_key=key, api_secret=secret, base_url=base_url))
        reconcile_bybit_wallet(ledger, adapter, full=args.full)
    print(json.dumps(ledger.snapshot(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
