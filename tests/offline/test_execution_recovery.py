"""Restart, outbox, isolation and crash-window tests against production code."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace as NS

from shared.confirmed_execution import ConfirmedExecutor
from shared.execution_journal import ExecutionJournal
from shared.strategy_position import StrategyPosition
from test_execution_safety import Exchange, fill


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "orders.sqlite"
        self.exchange = Exchange()
        self.executor = self.restart()

    def restart(self, owner="alpha.BTCUSDT"):
        return ConfirmedExecutor(self.exchange, self.path, "demo", wait_seconds=0, owner=owner)

    def submit(self):
        return self.executor.submit(intent_id="i1", symbol="BTCUSDT", side="buy", qty=D(1),
                                    parent={"source": "alpha.BTCUSDT", "order_id": "i1"})

    def test_partial_fill_restart_delayed_remainder_and_durable_dedup(self):
        self.exchange.cancel_ack_only = True
        self.exchange.filled_qty = D("0.4")
        self.exchange.fills = [fill()]
        self.submit()
        self.executor = self.restart()
        pending = self.executor.pending["i1"]
        self.assertEqual(pending.confirmed_qty, D("0.4"))
        self.assertEqual(pending.seen, {"trade-1"})
        self.exchange.state = "filled"
        self.exchange.filled_qty = D(1)
        self.exchange.fills += [fill("t2", qty="0.6")]
        observed = self.executor.poll()
        self.assertEqual([f.qty for _, f in observed], [D("0.6")])
        self.assertEqual(self.executor.unresolved(), [])
        self.assertEqual(len(self.executor.journal.fills(self.executor.owner)), 2)
        self.assertEqual(len([c for c in self.exchange.calls if c[0] == "limit"]), 1)

    def test_restart_recovers_parent_and_timeout_by_client_id(self):
        self.exchange.submit_timeout = True
        self.exchange.lookup_fails = True
        self.submit()
        original_id = self.exchange.client_id
        self.executor = self.restart()
        self.assertEqual(self.executor.pending["i1"].client_id, original_id)
        self.assertEqual(json.loads(self.executor.journal.orders()[0]["parent"])["order_id"], "i1")
        self.exchange.lookup_fails = False
        self.exchange.state = "filled"
        self.exchange.filled_qty = D(1)
        self.exchange.fills = [fill(qty="1")]
        self.assertEqual(len(self.executor.poll()), 1)
        self.assertEqual(len([c for c in self.exchange.calls if c[0] == "limit"]), 1)

    def test_owner_and_account_isolation(self):
        self.exchange.lookup_fails = True
        self.submit()
        self.assertFalse(self.restart("alpha.BTCUSDT_4h").pending)
        other = ConfirmedExecutor(self.exchange, self.path, "live", wait_seconds=0, owner=self.executor.owner)
        self.assertFalse(other.pending)
        self.assertFalse(other.unresolved())

    def test_legacy_schema_is_migrated_without_unlocking_or_guessing_side(self):
        path = Path(self.temp.name) / "legacy.sqlite"
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE submissions(scope TEXT,intent_id TEXT,symbol TEXT,client_id TEXT,"
                       "order_id TEXT,settled INTEGER,PRIMARY KEY(scope,intent_id))")
            db.execute("INSERT INTO submissions VALUES('demo','old','BTCUSDT','cl','o',0)")
        executor = ConfirmedExecutor(self.exchange, path, "demo", wait_seconds=0)
        self.assertEqual(len(executor.unresolved()), 1)
        self.assertFalse(executor.pending)
        self.assertEqual(executor.submit(intent_id="new", symbol="BTCUSDT", side="buy", qty=D(1)), ())
        self.assertFalse(self.exchange.calls)

    def test_undelivered_terminal_fill_blocks_next_entry(self):
        self.exchange.state = "filled"
        self.exchange.filled_qty = D(1)
        self.exchange.fills = [fill(qty="1")]
        self.submit()
        self.assertEqual(self.executor.unresolved(), [])
        self.assertEqual(self.executor.submit(intent_id="i2", symbol="BTCUSDT", side="buy", qty=D(1)), ())
        self.assertEqual(len([c for c in self.exchange.calls if c[0] == "limit"]), 1)

    def seed_fill(self):
        self.exchange.filled_qty = D("0.4")
        self.exchange.fills = [fill()]
        self.submit()
        return self.executor.journal.fills(self.executor.owner)[0]

    def test_transaction_failure_rolls_back_state_and_leaves_outbox(self):
        fact = self.seed_fill()
        memory = []
        def apply():
            memory.append(fact)
            raise RuntimeError("crash window")
        with self.assertRaises(RuntimeError):
            self.executor.journal.consume(self.executor.owner, fact["fill_id"], apply, memory.clear)
        self.assertFalse(memory)
        self.assertEqual(len(self.executor.journal.fills(self.executor.owner)), 1)
        self.assertIsNone(self.executor.journal.checkpoint(self.executor.owner))

    def test_process_death_before_and_after_commit(self):
        fact = self.seed_fill()
        code = '''
import os, sys
from pathlib import Path
from shared.execution_journal import ExecutionJournal
from shared.strategy_position import StrategyPosition
from types import SimpleNamespace
j=ExecutionJournal(Path(sys.argv[1]), 'demo')
owner='alpha.BTCUSDT'
f=j.fills(owner)[0]
def apply():
    p=StrategyPosition()
    p.apply(SimpleNamespace(**f))
    if sys.argv[2] == 'before': os._exit(86)
    return {'position':p.to_dict(), 'revision':f['fill_id']}
j.consume(owner, f['fill_id'], apply, lambda:None)
os._exit(87)
'''
        for phase, exit_code in (("before", 86), ("after", 87)):
            result = subprocess.run([sys.executable, "-c", code, str(self.path), phase],
                                    env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, exit_code, result.stderr)
            journal = ExecutionJournal(self.path, "demo")
            if phase == "before":
                self.assertIsNone(journal.checkpoint(self.executor.owner))
                self.assertEqual(len(journal.fills(self.executor.owner)), 1)
            else:
                checkpoint = journal.checkpoint(self.executor.owner)
                self.assertEqual(D(checkpoint["position"]["qty"]), D("0.4"))
                self.assertEqual(journal.fills(self.executor.owner), [])
                self.assertFalse(journal.consume(self.executor.owner, fact["fill_id"],
                                                lambda: self.fail("duplicate applied"), lambda:None))

    def test_stale_consumer_cannot_overwrite_committed_checkpoint(self):
        fact = self.seed_fill()
        self.executor.journal.consume(self.executor.owner, fact["fill_id"],
                                      lambda: {"revision": fact["fill_id"]}, lambda:None)
        with self.executor.journal.connect() as db:
            new = {**fact, "fill_id": "new"}
            db.execute("INSERT INTO execution_fills VALUES(?,?,?,?,?,0)",
                       ("demo", "new", "i1", self.executor.owner, json.dumps(new)))
        with self.assertRaises(RuntimeError):
            self.executor.journal.consume(self.executor.owner, "new", lambda: {}, lambda:None)


class PositionTests(unittest.TestCase):
    def test_partial_close_and_realized_pnl_fees(self):
        position = StrategyPosition()
        position.apply(NS(qty="0.4", price="100", side="buy", fee="-0.02", fee_asset="USDT"))
        position.apply(NS(qty="0.6", price="110", side="buy", fee="-0.03", fee_asset="USDT"))
        self.assertEqual(position.qty, D(1))
        self.assertEqual(position.entry_price, D(106))
        position.apply(NS(qty="0.3", price="120", side="sell"))
        self.assertEqual(position.qty, D("0.7"))
        self.assertEqual(position.entry_price, D(106))
        self.assertEqual(position.realized_pnl, D("4.2"))
        self.assertEqual(position.fees["USDT"], D("-0.05"))
        restored = StrategyPosition.from_dict(position.to_dict())
        self.assertEqual(restored, position)


class LeaseTests(unittest.TestCase):
    def test_two_processes_cannot_execute_same_scope(self):
        from shared.execution_lease import ExecutionLease
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "orders.sqlite"
            code = '''
import sys
from pathlib import Path
from shared.execution_lease import ExecutionLease
try:
    lease=ExecutionLease(Path(sys.argv[1]), sys.argv[2])
except RuntimeError:
    sys.exit(42)
lease.close()
'''
            def attempt(scope):
                return subprocess.run([sys.executable, "-c", code, str(path), scope],
                                      capture_output=True, timeout=20).returncode
            with ExecutionLease(path, "demo"):
                self.assertEqual(attempt("demo"), 42)
                self.assertEqual(attempt("other"), 0)
            self.assertEqual(attempt("demo"), 0)
