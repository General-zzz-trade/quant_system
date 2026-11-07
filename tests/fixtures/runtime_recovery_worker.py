"""Synthetic exchange recovery worker for disposable, network-disabled containers."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

from shared.confirmed_execution import ConfirmedExecutor
from shared.runtime_state import RuntimeStateLease, initialize_runtime_state

SCOPE = "offline-container-recovery"
OWNER = "alpha.recovery-test"


def no_network(*args, **kwargs):
    raise AssertionError("Recovery fixture cannot access the network")


class FakeExchange:
    def __init__(self, *, recovering=False):
        self.recovering = recovering
        self.placements = []
        self.queries = []
        self.client_id = ""

    def send_market_order(self, symbol, side, qty, **kwargs):
        self.placements.append((symbol, side, qty, kwargs))
        if self.recovering:
            raise AssertionError("Recovery must not resubmit an existing order")
        self.client_id = kwargs["client_order_id"]
        if symbol == "BTCUSDT":
            raise TimeoutError("Synthetic accepted request with lost response")
        return {"status": "submitted", "orderId": "synthetic-eth-order"}

    def get_order(self, symbol, **kwargs):
        self.queries.append((symbol, kwargs))
        if symbol == "BTCUSDT":
            return None
        return SimpleNamespace(
            symbol=symbol, side="buy", order_id="synthetic-eth-order",
            client_order_id=self.client_id, status="filled", filled_qty=Decimal("1"),
        )

    def get_recent_fills(self, **kwargs):
        return [SimpleNamespace(
            symbol="ETHUSDT", side="buy", order_id="synthetic-eth-order", trade_id="synthetic-trade",
            qty=Decimal("1"), price=Decimal("100"), fee=Decimal("0.1"), fee_asset="USDT",
        )]


def executor(exchange, journal):
    return ConfirmedExecutor(exchange, journal, SCOPE, owner=OWNER, wait_seconds=0,
                             entry_order_type="market")


def identity(directory):
    return json.loads((directory / "runtime-state.json").read_text())["state_id"]


def prepare(directory):
    initialize_runtime_state(directory, SCOPE)
    with RuntimeStateLease(directory, SCOPE) as lease:
        exchange = FakeExchange()
        execution = executor(exchange, lease.journal_path)
        with execution.journal.connect() as db:
            db.execute("INSERT INTO execution_consumers VALUES (?, ?, ?)",
                       (SCOPE, OWNER, json.dumps({"revision": "before", "applied": 0})))
        assert execution.submit(intent_id="unknown-intent", symbol="BTCUSDT", side="buy",
                                qty=Decimal("1"), parent={"owner": OWNER}) == ()
        fills = execution.submit(intent_id="filled-intent", symbol="ETHUSDT", side="buy",
                                 qty=Decimal("1"), parent={"owner": OWNER})
        assert len(fills) == 1
        assert len(exchange.placements) == 2
        assert len(execution.journal.fills(OWNER)) == 1
        assert execution.journal.checkpoint(OWNER) == {"revision": "before", "applied": 0}
        assert [r["intent_id"] for r in execution.unresolved()] == ["unknown-intent"]
        unknown_id = execution.pending["unknown-intent"].client_id
    return {"phase": "prepared", "scope": SCOPE, "state_id": identity(directory), "unknown_client_id": unknown_id,
            "unresolved": 1, "unconsumed": 1, "applied": 0}


def contender(directory):
    try:
        with RuntimeStateLease(directory, SCOPE):
            pass
    except RuntimeError as exc:
        if "already has an execution process" not in str(exc):
            raise
        print(json.dumps({"phase": "second-writer-blocked"}))
        return 23
    raise AssertionError("A second execution writer acquired the same account lease")


def recover(directory, expected_state_id, expected_client_id):
    # This must be the first operation: no automatic initialization on recovery.
    with RuntimeStateLease(directory, SCOPE) as lease:
        assert identity(directory) == expected_state_id, "State identity changed across containers"
        child = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "contender", "--state-dir", str(directory)],
            capture_output=True, text=True, timeout=15,
        )
        assert child.returncode == 23, "Second writer was not rejected: " + child.stdout + child.stderr
        assert json.loads(child.stdout)["phase"] == "second-writer-blocked"
        exchange = FakeExchange(recovering=True)
        execution = executor(exchange, lease.journal_path)
        assert execution.pending["unknown-intent"].client_id == expected_client_id
        assert execution.poll() == ()
        assert exchange.queries and exchange.queries[0][1]["client_order_id"] == expected_client_id
        assert execution.submit(intent_id="unknown-intent", symbol="BTCUSDT", side="buy",
                                qty=Decimal("1"), parent={"owner": OWNER}) == ()
        assert execution.submit(intent_id="another-intent", symbol="BTCUSDT", side="buy",
                                qty=Decimal("1"), parent={"owner": OWNER}) == ()
        assert exchange.placements == [], "Recovered or unresolved order was submitted again"
        assert [r["intent_id"] for r in execution.unresolved()] == ["unknown-intent"]
        checkpoint = execution.journal.checkpoint(OWNER)
        assert checkpoint == {"revision": "before", "applied": 0}
        facts = execution.journal.fills(OWNER)
        assert len(facts) == 1
        assert facts[0]["qty"] == "1" and facts[0]["price"] == "100"
        applied = []

        def apply():
            applied.append(facts[0]["fill_id"])
            return {"revision": facts[0]["fill_id"], "applied": len(applied), "quantity": "1"}

        def rollback():
            applied.clear()

        assert execution.journal.consume(OWNER, facts[0]["fill_id"], apply, rollback,
                                         expected_revision="before")
        assert not execution.journal.consume(OWNER, facts[0]["fill_id"], apply, rollback,
                                             expected_revision=facts[0]["fill_id"])
        assert len(applied) == 1
        assert execution.journal.fills(OWNER) == []
        assert execution.journal.checkpoint(OWNER)["applied"] == 1
    return {"phase": "recovered", "state_id": expected_state_id, "new_placements": 0,
            "unresolved": 1, "applied": 1, "duplicate_applied": 0, "second_writer_blocked": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "recover", "contender"))
    parser.add_argument("--state-dir", type=Path, default=Path("/state"))
    parser.add_argument("--state-id", default="")
    parser.add_argument("--client-id", default="")
    args = parser.parse_args()
    socket.socket.connect = no_network
    socket.create_connection = no_network
    if args.phase == "contender":
        return contender(args.state_dir)
    try:
        result = (prepare(args.state_dir) if args.phase == "prepare" else
                  recover(args.state_dir, args.state_id, args.client_id))
    except (ValueError, OSError, RuntimeError) as exc:
        print(json.dumps({"phase": "blocked", "reason": str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
