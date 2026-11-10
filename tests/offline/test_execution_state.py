"""Execution preflight uses temporary state and rendered fixtures, never accounts."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from runner.alpha_cli import parse_args
from scripts.check_execution_runtime import check_execution_runtime
from shared.account_ledger import AccountLedger
from shared.portfolio_reservations import PortfolioReservationStore, ScopeRegistration
from shared.runtime_backup import backup_state
from shared.runtime_state import (
    MANIFEST_NAME, initialize_runtime_state, validate_execution_state, validate_runtime_state,
)
from strategy.config import PORTFOLIO_NOTIONAL_CAPS_USD


class ExecutionStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / "account"
        self.scope = "bybit-demo-test"
        self.journal = initialize_runtime_state(self.state, self.scope)
        self.identity = json.loads((self.state / MANIFEST_NAME).read_text())["state_id"]

    def wallet(self):
        return AccountLedger.initialize(self.journal, scope=self.scope, state_id=self.identity,
                                        balance=Decimal(100), start_ms=1_000_000,
                                        source="offline statement", clock=lambda: 2000)

    def test_runtime_without_wallet_is_preserved_but_execution_blocked(self):
        self.assertEqual(validate_runtime_state(self.state, self.scope), self.journal)
        with self.assertRaisesRegex(ValueError, "wallet baseline"):
            validate_execution_state(self.state, self.scope)
        self.assertEqual(validate_runtime_state(self.state, self.scope), self.journal)

    def test_unreconciled_and_mismatched_wallet_are_valid_state_but_not_ready(self):
        wallet = self.wallet()
        self.assertEqual(validate_execution_state(self.state, self.scope), self.journal)
        self.assertFalse(wallet.check()[0])
        wallet.ingest_and_reconcile([], start_ms=1_000_000, end_ms=2_000_000,
                                    wallet_balance=Decimal(101), observed_at=2000)
        self.assertEqual(validate_execution_state(self.state, self.scope), self.journal)
        self.assertFalse(wallet.check()[0])

    def test_partial_wallet_schema_blocks_backup_and_execution(self):
        self.wallet()
        with sqlite3.connect(self.journal) as db:
            db.execute("DROP TABLE wallet_transactions")
        for validator in (validate_runtime_state, validate_execution_state):
            with self.assertRaisesRegex(ValueError, "partial"):
                validator(self.state, self.scope)
        target = self.root / "backup"
        with self.assertRaisesRegex(ValueError, "partial"):
            backup_state(self.state, target, self.scope)
        self.assertFalse(target.exists())

    def test_wallet_foreign_identity_and_scope_are_rejected(self):
        self.wallet()
        with sqlite3.connect(self.journal) as db:
            db.execute("UPDATE wallet_metadata SET state_id=?", ("0" * 32,))
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_runtime_state(self.state, self.scope)
        with sqlite3.connect(self.journal) as db:
            db.execute("UPDATE wallet_metadata SET state_id=?,scope='foreign'", (self.identity,))
        with self.assertRaisesRegex(ValueError, "different account"):
            validate_execution_state(self.state, self.scope)

    def test_canonical_wallet_payload_is_validated_for_backups(self):
        self.wallet()
        with sqlite3.connect(self.journal) as db:
            db.execute("INSERT INTO wallet_transactions VALUES (?,?,?,?)", (self.scope, "bad", 1_000_001, "{}"))
        with self.assertRaises(ValueError):
            validate_runtime_state(self.state, self.scope)

    def test_cli_rejects_unmanaged_execution_and_keeps_paper_available(self):
        with patch.dict("os.environ", {"RISK_SCOPE": self.scope, "RUNTIME_STATE_DIR": ""}):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                parse_args(["--enable-execution"])
            self.assertEqual(error.exception.code, 2)
            self.assertTrue(parse_args(["--dry-run"]).dry_run)
        with patch.dict("os.environ", {"RISK_SCOPE": self.scope, "RUNTIME_STATE_DIR": str(self.state)}):
            self.assertFalse(parse_args(["--enable-execution"]).dry_run)


class ExecutionMountTests(unittest.TestCase):
    def setUp(self):
        ExecutionStateTests.setUp(self)
        ExecutionStateTests.wallet(self)
        self.project = self.root / "checkout"
        self.project.mkdir()
        self.risk = self.root / "risk"
        self.risk.mkdir()
        self.store = PortfolioReservationStore.initialize(
            self.risk / "risk.sqlite", group="risk-group",
            registrations=[ScopeRegistration(self.scope, "bybit", "demo", self.identity)],
            caps={name: Decimal(str(cap)) for name, cap in PORTFOLIO_NOTIONAL_CAPS_USD.items()},
        )
        self.config = {"services": {"candidate": {
            "command": ["python3", "-m", "runner.alpha_main", "--enable-execution", "--venue", "bybit"],
            "environment": {"RISK_SCOPE": self.scope, "RUNTIME_STATE_DIR": "/app/data/runtime",
                "ORDER_JOURNAL_PATH": "/app/data/runtime/orders.sqlite",
                "DRAWDOWN_STATE_PATH": "/app/data/runtime/drawdown.sqlite",
                "PORTFOLIO_RISK_PATH": "/app/data/portfolio/risk.sqlite", "PORTFOLIO_RISK_GROUP": "risk-group",
                "BYBIT_BASE_URL": "https://api-demo.bybit.com"},
            "volumes": [
                {"type": "bind", "source": str(self.state), "target": "/app/data/runtime",
                 "read_only": False, "bind": {"create_host_path": False}},
                {"type": "bind", "source": str(self.risk), "target": "/app/data/portfolio",
                 "read_only": False, "bind": {"create_host_path": False}},
            ],
        }}}

    def check(self, config=None):
        return check_execution_runtime(config or self.config, self.project, "candidate")

    def test_explicit_persistent_bindings_validate_without_claiming_eligible(self):
        result = self.check()
        self.assertTrue(result["state_validated"])
        self.assertFalse(result["execution_eligible"])
        self.assertEqual(result["reason"], "fresh_account_reconciliation_required")
        self.assertFalse(self.store.snapshot()["ready"])

    def test_missing_readonly_auto_created_and_checkout_mounts_are_rejected(self):
        for defect in ("missing", "readonly", "create", "checkout", "nested", "risk_file"):
            with self.subTest(defect=defect):
                config = deepcopy(self.config)
                volumes = config["services"]["candidate"]["volumes"]
                if defect == "missing":
                    volumes.pop()
                elif defect == "readonly":
                    volumes[0]["read_only"] = True
                elif defect == "create":
                    volumes[0]["bind"]["create_host_path"] = True
                elif defect == "checkout":
                    volumes[0]["source"] = str(self.project)
                elif defect == "nested":
                    volumes.append({"target": "/app/data/runtime/orders.sqlite", "type": "bind"})
                else:
                    volumes[1]["target"] = "/app/data/portfolio/risk.sqlite"
                with self.assertRaises(ValueError):
                    self.check(config)

    def test_environment_scope_state_and_group_must_match_registry(self):
        for name, value in (("RISK_SCOPE", "foreign"), ("PORTFOLIO_RISK_GROUP", "foreign"),
                            ("BYBIT_BASE_URL", "https://api.bybit.com"),
                            ("ORDER_JOURNAL_PATH", "/tmp/orders.sqlite")):
            with self.subTest(name=name):
                config = deepcopy(self.config)
                config["services"]["candidate"]["environment"][name] = value
                with self.assertRaises(ValueError):
                    self.check(config)

    def test_dry_run_or_noncanonical_entrypoint_cannot_claim_execution_readiness(self):
        for command in (["python3", "-m", "scripts.run_bybit_alpha", "--ws"],
                        ["python3", "-m", "runner.alpha_main", "--dry-run"],
                        ["sh", "-c", "python3 -m runner.alpha_main --enable-execution"]):
            config = deepcopy(self.config)
            config["services"]["candidate"]["command"] = command
            with self.assertRaises(ValueError):
                self.check(config)

    def test_shared_risk_database_is_not_recreated_when_missing(self):
        (self.risk / "risk.sqlite").unlink()
        with self.assertRaises(ValueError):
            self.check()
        self.assertFalse((self.risk / "risk.sqlite").exists())


if __name__ == "__main__":
    unittest.main()
