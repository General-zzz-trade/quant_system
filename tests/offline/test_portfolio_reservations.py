"""Portfolio admission concurrency and recovery using temporary SQLite only."""
from __future__ import annotations

import multiprocessing
import os
import queue
import sqlite3
import tempfile
import time
import unittest
from dataclasses import replace
from decimal import Decimal as D
from pathlib import Path

from shared.portfolio_reservations import (
    AccountEvidence,
    ConsumerPosition,
    PortfolioReservationStore,
    PositionEvidence,
    ScopeRegistration,
    SubmissionEvidence,
)

REGISTRY = (ScopeRegistration("account-a", "bybit", "paper", "state-a"),
            ScopeRegistration("account-b", "bybit", "paper", "state-b"))


def flat(registration, now):
    return AccountEvidence(registration.scope, registration.venue, registration.environment,
                           registration.state_id, now, (), (), ())


def admit(store, scope="account-a", intent="entry-a", amount="60", **overrides):
    registration = next(r for r in REGISTRY if r.scope == scope)
    values = dict(scope=scope, venue=registration.venue, environment=registration.environment,
                  state_id=registration.state_id, intent_id=intent, symbol="BTCUSDT", side="buy",
                  notional=D(amount), quantity=D("1"))
    values.update(overrides)
    return store.reserve(**values)


def race_worker(path, scope, start, results):
    store = PortfolioReservationStore(Path(path), group="portfolio-test")
    start.wait(10)
    result = admit(store, scope, scope)
    results.put((result.allowed, result.reason))


def crash_worker(path):
    store = PortfolioReservationStore(Path(path), group="portfolio-test")
    if not admit(store).allowed:
        os._exit(2)
    # Simulate process death between risk reservation and journal submission.
    os._exit(17)


def blocked_worker(path, start, result):
    store = PortfolioReservationStore(Path(path), group="portfolio-test")
    start.set()
    result.put(admit(store, "account-b", "while-reconciling", amount="60").allowed)


class PortfolioReservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "portfolio.sqlite"
        self.now = time.time()
        self.store = PortfolioReservationStore.initialize(
            self.path, group="portfolio-test", registrations=REGISTRY,
            caps={"BTCUSDT": D("100")}, clock=lambda: self.now,
        )

    def evidence(self):
        return [flat(r, self.now) for r in REGISTRY]

    def baseline(self):
        return self.store.reconcile(self.evidence)

    def terminal(self, *, intent="entry-a", status="filled", quantity="1"):
        return SubmissionEvidence(intent, "BTCUSDT", "buy", D(quantity), True, status)

    def filled_evidence(self, amount="60"):
        accounts = self.evidence()
        accounts[0] = replace(accounts[0],
                              positions=(PositionEvidence("BTCUSDT", "buy", D(1), D(amount)),),
                              submissions=(self.terminal(),),
                              consumer_positions=(ConsumerPosition("alpha.BTCUSDT", "BTCUSDT", D(1)),))
        return accounts

    def test_explicit_initialization_requires_reconciliation(self):
        self.assertFalse(admit(self.store).allowed)
        self.assertEqual(self.store.snapshot()["reason"], "baseline_uninitialized")
        self.assertEqual(self.baseline(), 1)
        self.assertTrue(admit(self.store).allowed)

    def test_missing_or_changed_identity_never_creates_empty_state(self):
        missing = self.path.with_name("missing.sqlite")
        with self.assertRaisesRegex(ValueError, "refusing an empty replacement"):
            PortfolioReservationStore(missing, group="portfolio-test")
        self.assertFalse(missing.exists())
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            PortfolioReservationStore(self.path, group="another-group")
        with self.assertRaises(FileExistsError):
            PortfolioReservationStore.initialize(self.path, group="portfolio-test",
                                                 registrations=REGISTRY, caps={"BTCUSDT": D(999)})
        self.baseline()
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE portfolio_metadata SET value='{}' WHERE key='caps'")
        self.assertFalse(admit(self.store).allowed)

    def test_two_processes_share_one_cap(self):
        # Child clocks use current wall time, so their admission age remains valid.
        self.baseline()
        context = multiprocessing.get_context("spawn")
        start, results = context.Event(), context.Queue()
        workers = [context.Process(target=race_worker,
                                   args=(str(self.path), r.scope, start, results)) for r in REGISTRY]
        for worker in workers:
            worker.start()
        start.set()
        outcomes = [results.get(timeout=15) for _ in workers]
        for worker in workers:
            worker.join(timeout=15)
            self.assertEqual(worker.exitcode, 0)
        self.assertEqual(sum(allowed for allowed, _ in outcomes), 1, outcomes)
        self.assertEqual(self.store.snapshot()["used"]["BTCUSDT"]["buy"], "60")

    def test_process_crash_before_submission_keeps_orphan_budget(self):
        self.baseline()
        context = multiprocessing.get_context("spawn")
        worker = context.Process(target=crash_worker, args=(str(self.path),))
        worker.start()
        worker.join(timeout=15)
        self.assertEqual(worker.exitcode, 17)
        restarted = PortfolioReservationStore(self.path, group="portfolio-test", clock=lambda: self.now)
        self.assertFalse(admit(restarted, "account-b", "second").allowed)
        with self.assertRaisesRegex(ValueError, "orphan reservation"):
            restarted.reconcile(self.evidence)
        self.assertEqual(len(restarted.snapshot()["reservations"]), 1)
        self.assertFalse(restarted.snapshot()["ready"])

    def test_same_intent_is_idempotent_and_conflicting_reuse_is_denied(self):
        self.baseline()
        self.assertTrue(admit(self.store).allowed)
        self.assertTrue(admit(self.store).allowed)
        self.assertEqual(len(self.store.snapshot()["reservations"]), 1)
        for changes in ({"notional": D(59)}, {"quantity": D(2)}, {"side": "sell"}):
            with self.subTest(changes=changes):
                result = admit(self.store, **changes)
                self.assertFalse(result.allowed)
                self.assertIn("conflict", result.reason)

    def test_filled_order_is_counted_once_only_after_verified_epoch(self):
        self.baseline()
        self.assertTrue(admit(self.store).allowed)
        self.assertEqual(self.store.snapshot()["used"]["BTCUSDT"]["buy"], "60")
        self.assertEqual(self.store.reconcile(self.filled_evidence), 2)
        status = self.store.snapshot()
        self.assertEqual(status["baseline"]["BTCUSDT"]["buy"], "60")
        self.assertEqual(status["used"]["BTCUSDT"]["buy"], "60")
        self.assertEqual(status["reservations"][0]["covered_epoch"], 2)
        self.assertTrue(admit(self.store, "account-b", "remaining", amount="40").allowed)
        self.assertFalse(admit(self.store, "account-b", "excess", amount="0.01").allowed)
        # Historical IDs stay bound; covering does not permit reuse as a new order.
        self.assertFalse(admit(self.store, amount="70").allowed)

    def test_canceled_order_releases_only_after_complete_reconciliation(self):
        self.baseline()
        admit(self.store)
        self.assertFalse(admit(self.store, "account-b", "second").allowed)
        accounts = self.evidence()
        accounts[0] = replace(accounts[0], submissions=(self.terminal(status="canceled"),))
        self.store.reconcile(lambda: accounts)
        self.assertTrue(admit(self.store, "account-b", "second").allowed)

    def test_unknown_order_and_timeout_never_release_reservation(self):
        self.baseline()
        admit(self.store)
        accounts = self.evidence()
        accounts[0] = replace(accounts[0], submissions=(replace(self.terminal(), settled=False, status="unknown"),))
        with self.assertRaisesRegex(ValueError, "unknown submission"):
            self.store.reconcile(lambda: accounts)
        self.now += 10_000
        self.assertFalse(admit(self.store, "account-b", "second").allowed)
        self.assertIsNone(self.store.snapshot()["reservations"][0]["covered_epoch"])

    def test_stale_or_future_baseline_denies_entry(self):
        self.baseline()
        original = self.now
        for delta in (301, -1):
            with self.subTest(delta=delta):
                self.now = original + delta
                self.assertFalse(admit(self.store).allowed)
        self.now = original
        self.assertTrue(admit(self.store).allowed)

    def test_failed_refresh_invalidates_fresh_baseline_persistently(self):
        self.baseline()

        def broken():
            raise TimeoutError("fake account timeout")

        with self.assertRaisesRegex(ValueError, "fake account timeout"):
            self.store.reconcile(broken)
        self.assertFalse(admit(self.store).allowed)
        restarted = PortfolioReservationStore(self.path, group="portfolio-test", clock=lambda: self.now)
        self.assertFalse(admit(restarted).allowed)
        restarted.reconcile(self.evidence)
        self.assertTrue(admit(restarted).allowed)

    def test_missing_duplicate_or_boolean_account_proof_is_rejected(self):
        for evidence in (self.evidence()[:1], [self.evidence()[0]] * 2,
                         {"complete": True, "positions": []}, [True, True]):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                self.store.reconcile(lambda: evidence)
        self.assertFalse(self.store.snapshot()["ready"])

    def test_registry_cannot_be_bypassed_by_scope_venue_environment_or_state(self):
        self.baseline()
        for field in ("scope", "venue", "environment", "state_id"):
            with self.subTest(field=field):
                arguments = dict(scope="account-a", venue="bybit", environment="paper", state_id="state-a",
                                 intent_id="bad", symbol="BTCUSDT", side="buy", notional=D(1), quantity=D(1))
                arguments[field] = "other"
                self.assertFalse(self.store.reserve(**arguments).allowed)
                accounts = self.evidence()
                accounts[0] = replace(accounts[0], **{field: "other"})
                with self.assertRaises(ValueError):
                    self.store.reconcile(lambda: accounts)

    def test_malformed_account_or_incomplete_journal_blocks(self):
        changes = (
            {"observed_at": self.now - 1}, {"observed_at": self.now + 1},
            {"open_order_ids": ("unowned-order",)}, {"unconsumed_fill_ids": ("pending-fill",)},
            {"positions": None},
            {"positions": (PositionEvidence("BTCUSDT", "buy", D(1), D(10)),)},
            {"positions": (PositionEvidence("UNKNOWN", "buy", D(1), D(10)),)},
            {"consumer_positions": (ConsumerPosition("owner", "BTCUSDT", D("NaN")),)},
            {"submissions": (replace(self.terminal(), settled=1),)},
        )
        for change in changes:
            with self.subTest(change=change):
                accounts = self.evidence()
                accounts[0] = replace(accounts[0], **change)
                with self.assertRaises(ValueError):
                    self.store.reconcile(lambda: accounts)
                self.assertFalse(self.store.snapshot()["ready"])

    def test_submission_must_match_reserved_symbol_side_and_quantity(self):
        self.baseline()
        admit(self.store)
        for change in ({"side": "sell"}, {"quantity": D(2)}, {"intent_id": "different"}):
            with self.subTest(change=change):
                accounts = self.evidence()
                accounts[0] = replace(accounts[0], submissions=(replace(self.terminal(), **change),))
                with self.assertRaises(ValueError):
                    self.store.reconcile(lambda: accounts)
                self.assertIsNone(self.store.snapshot()["reservations"][0]["covered_epoch"])

    def test_reconciliation_write_lock_blocks_concurrent_admission(self):
        self.baseline()
        context = multiprocessing.get_context("spawn")
        started, result = context.Event(), context.Queue()
        worker = context.Process(target=blocked_worker, args=(str(self.path), started, result))

        def collect():
            worker.start()
            self.assertTrue(started.wait(10))
            # A writer cannot complete while the observation callback owns the
            # same SQLite write lock. Return a baseline consuming 60 of the cap.
            with self.assertRaises(queue.Empty):
                result.get(timeout=0.2)
            accounts = self.evidence()
            accounts[0] = replace(accounts[0],
                                  positions=(PositionEvidence("BTCUSDT", "buy", D(1), D(60)),),
                                  consumer_positions=(ConsumerPosition("owner", "BTCUSDT", D(1)),))
            return accounts

        self.store.reconcile(collect)
        self.assertFalse(result.get(timeout=15))
        worker.join(timeout=15)
        self.assertEqual(worker.exitcode, 0)

    def test_negative_nonfinite_zero_and_unregistered_requests_fail(self):
        self.baseline()
        for changes in ({"notional": D(0)}, {"notional": D(-1)}, {"notional": D("NaN")},
                        {"quantity": D("Infinity")}, {"quantity": D(0)},
                        {"symbol": "UNKNOWN"}, {"side": "BUY"}, {"intent_id": ""}):
            with self.subTest(changes=changes):
                self.assertFalse(admit(self.store, **changes).allowed)
        self.assertEqual(self.store.snapshot()["reservations"], [])

    def test_buy_and_sell_budgets_preserve_existing_directional_cap_semantics(self):
        self.baseline()
        self.assertTrue(admit(self.store, amount="100").allowed)
        self.assertTrue(admit(self.store, "account-b", "short", amount="100", side="sell").allowed)
        self.assertFalse(admit(self.store, "account-b", "more", amount="1", side="sell").allowed)

    def test_decimal_precision_cannot_round_away_an_excess(self):
        self.baseline()
        self.assertTrue(admit(self.store, amount="100").allowed)
        self.assertFalse(admit(self.store, "account-b", "dust", amount="0.000000000000000000000000000001").allowed)

    def test_epoch_update_failure_rolls_back_baseline_and_covers_nothing(self):
        self.baseline()
        admit(self.store)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TRIGGER fail_cover BEFORE UPDATE OF covered_epoch ON portfolio_reservations "
                       "BEGIN SELECT RAISE(ABORT, 'injected epoch failure'); END")
        with self.assertRaisesRegex(ValueError, "injected epoch failure"):
            self.store.reconcile(self.filled_evidence)
        status = self.store.snapshot()
        self.assertFalse(status["ready"])
        self.assertEqual(status["epoch"], 1)
        self.assertEqual(status["baseline"]["BTCUSDT"]["buy"], "0")
        self.assertIsNone(status["reservations"][0]["covered_epoch"])

    def test_symlink_state_is_rejected(self):
        alias = self.path.with_name("alias.sqlite")
        alias.symlink_to(self.path)
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            PortfolioReservationStore(alias, group="portfolio-test")


if __name__ == "__main__":
    unittest.main()
