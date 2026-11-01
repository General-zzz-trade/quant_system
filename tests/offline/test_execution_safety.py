"""Fault injection against the production lifecycle; no network or Rust."""
from __future__ import annotations

import logging
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace as NS

from shared.confirmed_execution import ConfirmedExecutor


def fill(trade_id="trade-1", *, order_id="exchange-1", qty="0.4", price="100", side="buy"):
    return NS(symbol="BTCUSDT", order_id=order_id, trade_id=trade_id, side=side, qty=D(qty), price=D(price))


class Exchange:
    def __init__(self):
        self.calls = []
        self.client_id = ""
        self.state = "live"
        self.filled_qty = D(0)
        self.fills = []
        self.cancel_fails = False
        self.cancel_ack_only = False
        self.submit_timeout = False
        self.lookup_fails = False
        self.hide_order = False
        self.quote = 100

    def get_ticker(self, symbol):
        return {"bid1Price": self.quote, "ask1Price": self.quote}

    def send_limit_order(self, symbol, side, qty, price, **kwargs):
        self.calls.append(("limit", qty, kwargs))
        self.client_id = kwargs["client_order_id"]
        if self.submit_timeout:
            raise TimeoutError("accepted by exchange, response lost")
        return {"status": "submitted", "orderId": "exchange-1"}

    def send_market_order(self, symbol, side, qty, **kwargs):
        self.calls.append(("market", qty, kwargs))
        self.client_id = kwargs["client_order_id"]
        return {"status": "submitted", "orderId": "exchange-1"}

    def get_order(self, symbol, **kwargs):
        if self.lookup_fails:
            raise ConnectionError("exchange unavailable")
        if self.hide_order:
            return None
        return NS(order_id="exchange-1", client_order_id=self.client_id,
                  state=self.state, filled_qty=self.filled_qty)

    def cancel_order(self, symbol, order_id):
        self.calls.append(("cancel", order_id))
        if self.cancel_fails:
            raise TimeoutError("cancel timeout")
        if not self.cancel_ack_only:
            self.state = "canceled"
        return {"status": "canceled"}  # even this misleading ACK is not terminal proof

    def get_recent_fills(self, **kwargs):
        return tuple(self.fills)


class ExecutionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "orders.sqlite"
        self.exchange = Exchange()
        self.executor = ConfirmedExecutor(self.exchange, self.path, "okx-demo-test", wait_seconds=0)
        # Fault logs are expected; unittest failures still include assertions.
        self._log_level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, self._log_level)

    def submit(self, intent_id="intent-1", **kwargs):
        return self.executor.submit(intent_id=intent_id, symbol="BTCUSDT", side="buy", qty=D(1), **kwargs)

    def placements(self):
        return [call for call in self.exchange.calls if call[0] in ("market", "limit")]

    def test_ack_without_fills_emits_nothing(self):
        self.assertEqual(self.submit(), ())
        self.assertEqual(len(self.placements()), 1)
        self.assertEqual(self.executor.unresolved(), [])

    def test_partial_fill_uses_actual_quantity_and_never_market_fallback(self):
        self.exchange.fills = [fill()]
        self.exchange.filled_qty = D("0.4")
        facts = self.submit()
        self.assertEqual([f.qty for f in facts], [D("0.4")])
        self.assertEqual([c[0] for c in self.placements()], ["limit"])
        self.assertEqual(self.executor.unresolved(), [])

    def test_same_side_unrelated_order_is_not_our_fill(self):
        self.exchange.fills = [fill(order_id="someone-else")]
        self.assertEqual(self.submit(), ())

    def test_fill_wrong_side_or_invalid_facts_are_not_emitted(self):
        self.exchange.fills = [fill(side="sell"), fill("", qty="1"), fill(price="0"), fill(qty="NaN")]
        self.assertEqual(self.submit(), ())

    def test_duplicate_trade_records_count_once(self):
        self.exchange.fills = [fill(), fill()]
        self.exchange.filled_qty = D("0.4")
        self.assertEqual(len(self.submit()), 1)

    def test_delayed_fill_emitted_on_poll_once(self):
        self.exchange.state = "filled"
        self.exchange.filled_qty = D(1)
        self.assertEqual(self.submit(), ())
        self.assertEqual(len(self.executor.unresolved()), 1)
        self.exchange.fills = [fill(qty="1")]
        facts = self.executor.poll()
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0][0], "intent-1")
        self.assertEqual(self.executor.poll(), ())

    def test_cancel_timeout_does_not_send_second_order(self):
        self.exchange.cancel_fails = True
        self.submit()
        self.submit("intent-2")
        self.assertEqual(len(self.placements()), 1)
        self.assertEqual(len(self.executor.unresolved()), 1)

    def test_cancel_ack_without_terminal_state_keeps_entry_lock(self):
        self.exchange.cancel_ack_only = True
        self.submit()
        self.submit("intent-2")
        self.assertEqual(len(self.placements()), 1)
        self.assertEqual(len(self.executor.unresolved()), 1)

    def test_timeout_after_acceptance_queries_client_id_without_resend(self):
        self.exchange.submit_timeout = True
        self.exchange.state = "filled"
        self.exchange.filled_qty = D(1)
        self.exchange.fills = [fill(qty="1")]
        self.assertEqual(len(self.submit()), 1)
        self.assertEqual(len(self.placements()), 1)
        self.assertEqual(len(self.exchange.client_id), 32)

    def test_order_not_found_is_not_proof_of_rejection(self):
        self.exchange.hide_order = True
        self.submit()
        self.submit("intent-2")
        self.assertEqual(len(self.placements()), 1)

    def test_partial_history_does_not_unlock_terminal_order(self):
        self.exchange.state = "filled"
        self.exchange.filled_qty = D(1)
        self.exchange.fills = [fill()]
        self.submit()
        self.submit("intent-2")
        self.assertEqual(len(self.placements()), 1)
        self.exchange.fills.append(fill("trade-2", qty="0.6"))
        self.assertEqual(len(self.executor.poll()), 1)
        self.assertEqual(self.executor.unresolved(), [])

    def test_restart_preserves_unresolved_order_lock(self):
        self.exchange.lookup_fails = True
        self.submit()
        self.executor = ConfirmedExecutor(self.exchange, self.path, "okx-demo-test", wait_seconds=0)
        self.submit("intent-2")
        self.assertEqual(len(self.placements()), 1)

    def test_completed_intent_is_not_resubmitted_after_restart(self):
        self.submit()
        self.executor = ConfirmedExecutor(self.exchange, self.path, "okx-demo-test", wait_seconds=0)
        self.submit()
        self.assertEqual(len(self.placements()), 1)

    def test_reduce_only_still_available_with_unresolved_entry(self):
        self.exchange.lookup_fails = True
        self.submit()
        self.submit("exit-1", reduce_only=True)
        self.assertEqual(len(self.placements()), 2)
        self.assertTrue(self.placements()[1][2]["reduce_only"])

    def test_parallel_coordinators_cannot_reserve_two_entries(self):
        other = ConfirmedExecutor(self.exchange, self.path, "okx-demo-test", wait_seconds=0)
        self.exchange.lookup_fails = True
        with ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(ex.submit, intent_id=f"intent-{i}", symbol="BTCUSDT", side="buy", qty=D(1))
                    for i, ex in enumerate((self.executor, other))]
            for job in jobs:
                job.result()
        self.assertEqual(len(self.placements()), 1)

    def test_unavailable_journal_blocks_before_exchange_write(self):
        self.path.write_text("corrupt SQLite")
        with self.assertRaises(Exception):
            self.submit()
        self.assertEqual(self.placements(), [])

    def test_invalid_quote_blocks_without_submission(self):
        for price in (0, float("nan"), float("inf")):
            self.exchange.quote = price
            with self.assertRaises(ValueError):
                self.submit()
        self.assertEqual(self.placements(), [])
