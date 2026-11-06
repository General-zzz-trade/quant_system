"""Market submission crash windows against the production journal, without Rust."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal as D
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

from shared.confirmed_execution import ConfirmedExecutor


class MarketExchange:
    def __init__(self):
        self.placements = []
        self.status = "new"
        self.client_id = ""
        self.hidden = False
        self.timeout = False
        self.fills = []
        self.filled_qty = D(0)

    def send_market_order(self, symbol, side, qty, **kwargs):
        self.placements.append((symbol, side, qty, kwargs))
        self.client_id = kwargs["client_order_id"]
        if self.timeout:
            raise TimeoutError("accepted, response lost")
        return {"status": "submitted", "orderId": "venue-1"}

    def send_limit_order(self, *args, **kwargs):
        raise AssertionError("market path placed a limit order")

    def get_ticker(self, *args, **kwargs):
        raise AssertionError("market path requested a limit quote")

    def cancel_order(self, *args, **kwargs):
        raise AssertionError("market recovery attempted automatic cancellation")

    def get_order(self, symbol, **kwargs):
        if self.hidden:
            return None
        return NS(symbol=symbol, side="buy", order_id="venue-1", client_order_id=self.client_id,
                  status=self.status, filled_qty=self.filled_qty)

    def get_recent_fills(self, **kwargs):
        return self.fills


def trade(trade_id="trade-1", qty="0.4", fee="0"):
    return NS(symbol="BTCUSDT", side="buy", order_id="venue-1", trade_id=trade_id,
              qty=D(qty), price=D(100), fee=D(fee), fee_asset="USDT")


class MarketExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "orders.sqlite"
        self.exchange = MarketExchange()
        self.executor = self.restart()

    def restart(self):
        return ConfirmedExecutor(self.exchange, self.path, "bybit-test", owner="alpha.BTCUSDT",
                                 wait_seconds=0, entry_order_type="market")

    def submit(self, executor=None, intent="intent-1"):
        return (executor or self.executor).submit(
            intent_id=intent, symbol="BTCUSDT", side="buy", qty=D(1),
            parent={"source": "alpha.BTCUSDT"},
        )

    def test_ack_is_not_a_fill_and_market_type_survives_restart(self):
        self.assertEqual(self.submit(), ())
        self.assertEqual(self.executor.journal.orders()[0]["order_type"], "market")
        self.executor = self.restart()
        self.assertFalse(self.executor.pending["intent-1"].is_limit)
        self.assertEqual(self.executor.poll(), ())
        self.assertEqual(len(self.exchange.placements), 1)
        self.assertEqual(self.executor.journal.fills("alpha.BTCUSDT"), [])

    def test_timeout_and_absent_lookup_never_resubmit(self):
        self.exchange.timeout = self.exchange.hidden = True
        self.assertEqual(self.submit(), ())
        original = self.exchange.client_id
        self.executor = self.restart()
        self.assertEqual(self.executor.pending["intent-1"].client_id, original)
        self.assertEqual(self.submit(), ())
        self.assertEqual(self.submit(intent="intent-2"), ())
        self.assertEqual(len(self.exchange.placements), 1)
        self.assertEqual(self.executor.journal.orders()[0]["status"], "unknown")

    def test_terminal_history_lag_stays_locked_until_all_trades_arrive(self):
        self.exchange.status, self.exchange.filled_qty = "filled", D(1)
        self.exchange.fills = [trade()]
        self.assertEqual([f.qty for f in self.submit()], [D("0.4")])
        self.assertEqual(len(self.executor.unresolved()), 1)
        self.exchange.fills += [trade("trade-2", "0.6")]
        self.assertEqual([f.qty for _, f in self.executor.poll()], [D("0.6")])
        self.assertEqual(self.executor.unresolved(), [])
        self.assertEqual(self.executor.poll(), ())
        self.assertEqual(len(self.exchange.placements), 1)

    def test_invalid_fee_never_becomes_a_durable_fact(self):
        self.exchange.status, self.exchange.filled_qty = "filled", D("0.4")
        self.exchange.fills = [trade(fee="NaN")]
        self.assertEqual(self.submit(), ())
        self.assertEqual(len(self.executor.unresolved()), 1)
        self.assertEqual(self.executor.journal.fills("alpha.BTCUSDT"), [])

    def test_two_coordinators_cannot_place_the_same_intent(self):
        self.exchange.hidden = True
        other = self.restart()
        with ThreadPoolExecutor(2) as pool:
            results = [pool.submit(self.submit, ex) for ex in (self.executor, other)]
            for result in results:
                result.result()
        self.assertEqual(len(self.exchange.placements), 1)

    def test_conflicting_duplicate_trade_cannot_unlock_order(self):
        self.exchange.filled_qty = D("0.4")
        self.exchange.fills = [trade()]
        self.submit()
        self.exchange.status, self.exchange.filled_qty = "filled", D("0.4")
        self.exchange.fills = [trade(qty="0.8")]
        self.assertEqual(self.executor.poll(), ())
        self.assertEqual(len(self.executor.unresolved()), 1)
        facts = self.executor.journal.fills("alpha.BTCUSDT")
        self.assertEqual([D(f["qty"]) for f in facts], [D("0.4")])

    def test_wrong_side_trade_cannot_confirm_an_empty_terminal_order(self):
        self.exchange.status = "filled"
        wrong = trade()
        wrong.side = "sell"
        self.exchange.fills = [wrong]
        self.assertEqual(self.submit(), ())
        self.assertEqual(len(self.executor.unresolved()), 1)

    def test_unconsumed_terminal_trade_blocks_another_owner_reduction(self):
        self.exchange.status, self.exchange.filled_qty = "filled", D(1)
        self.exchange.fills = [trade(qty="1")]
        self.submit()
        self.assertEqual(self.executor.unresolved(), [])
        self.assertEqual(self.executor.submit(intent_id="close-1", symbol="BTCUSDT", side="sell", qty=D(1),
                                              reduce_only=True, parent={}), ())
        self.assertEqual(len(self.exchange.placements), 1)

    def test_reduction_checkpoint_revision_is_checked_in_reservation_transaction(self):
        with self.executor.journal.connect() as db:
            db.execute("INSERT INTO execution_consumers VALUES(?,?,?)",
                       ("bybit-test", "alpha.BTCUSDT", json.dumps({"revision": "new-fill"})))
        self.assertEqual(self.executor.submit(intent_id="close-1", symbol="BTCUSDT", side="sell", qty=D(1),
                                              reduce_only=True, parent={"consumer_revision": "old-fill"}), ())
        self.assertEqual(self.exchange.placements, [])

    def test_process_death_before_or_after_placement_preserves_unknown_reservation(self):
        source = '''
import json, os, sys
from decimal import Decimal
from pathlib import Path
from shared.confirmed_execution import ConfirmedExecutor
class Exchange:
    def send_market_order(self, symbol, side, qty, **kwargs):
        Path(sys.argv[2]).write_text(json.dumps(kwargs))
        os._exit(86)
ex=ConfirmedExecutor(Exchange(), Path(sys.argv[1]), 'bybit-test', owner='alpha.BTCUSDT',
                     wait_seconds=0, entry_order_type='market')
reserve=ex._reserve
def interrupted(*args, **kwargs):
    result=reserve(*args, **kwargs)
    os._exit(85)
if sys.argv[3]=='before': ex._reserve=interrupted
ex.submit(intent_id='intent-1', symbol='BTCUSDT', side='buy', qty=Decimal(1), parent={'owner':'alpha.BTCUSDT'})
'''
        for phase, expected in (("before", 85), ("after", 86)):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as scratch:
                self.path = Path(scratch) / "orders.sqlite"
                call_file = Path(scratch) / "accepted.json"
                result = subprocess.run(
                    [sys.executable, "-B", "-c", source, str(self.path), str(call_file), phase],
                    env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
                    capture_output=True, text=True, timeout=20,
                )
                self.assertEqual(result.returncode, expected, result.stderr)
                self.exchange = MarketExchange()
                self.exchange.hidden = phase == "before"
                self.executor = self.restart()
                self.assertFalse(self.executor.pending["intent-1"].is_limit)
                if phase == "after":
                    self.exchange.client_id = json.loads(call_file.read_text())["client_order_id"]
                    self.exchange.status, self.exchange.filled_qty = "filled", D(1)
                    self.exchange.fills = [trade(qty="1")]
                    self.assertEqual(len(self.executor.poll()), 1)
                else:
                    self.assertEqual(self.executor.poll(), ())
                self.assertEqual(self.submit(), ())
                self.assertEqual(self.exchange.placements, [])


if __name__ == "__main__":
    unittest.main()
