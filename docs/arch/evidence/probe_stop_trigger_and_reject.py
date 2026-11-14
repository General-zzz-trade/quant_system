# ruff: noqa
# Read-only diagnosis probe (no network): runs the real runtime classes against the in-memory fake venue used by the unit tests.
# Usage: python3 -B <this file> <empty work dir> [scenario]. See docs/arch/current-state.md.
"""Offline probe #2 (fake venue only). Position is built through the real entry->fill->consume flow."""
import sys, tempfile, logging
from decimal import Decimal as D
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
sys.dont_write_bytecode = True
logging.disable(logging.CRITICAL)
from execution.adapters.binance import BinanceConfig
from execution.adapters.binance.adapter import BinanceAdapter
from execution.adapters.binance.execution_adapter import BinanceExecutionAdapter
from execution.adapters.binance.protective_stops import ProtectiveStopManager
from execution.adapters.binance.rest import BinanceNonRetryableError
from lab.orders import make_order
from runner.binance_target import TargetFillConsumer
from tests.unit.binance.fake_venue import FakeBinance, make_bars

OWNER = "target.BTCUSDT"

def env_with_position():
    tmp = Path(tempfile.mkdtemp())
    clock = {"now": 1_800_000_000.0}
    fake = FakeBinance(make_bars(10), clock=lambda: clock["now"])
    fake.price["BTCUSDT"] = D("100")
    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
    assert adapter.connect(symbols=("BTCUSDT",))
    ex = BinanceExecutionAdapter(adapter, scope="binance-testnet-probe", owner=OWNER, entry_mode="market",
                                 journal_path=tmp / "orders.sqlite", wait_seconds=0, clock=lambda: clock["now"])
    ex._ready = True
    consumer = TargetFillConsumer(ex)
    entry = make_order(order_id="BTCUSDT:1:rebalance:0", source=OWNER, symbol="BTCUSDT", side="buy",
                       qty=D("0.1"), price=D("100"), reduce_only=False, reason="rebalance")
    for ev in ex.send_order(entry):
        consumer.consume(ev)
    stops = ProtectiveStopManager(adapter, ex, "BTCUSDT", stop_pct=D("0.04"), clock=lambda: clock["now"])
    return fake, adapter, ex, clock, consumer, stops

print("=== B. stop reported TRIGGERED before actualOrderId is known")
fake, adapter, ex, clock, consumer, stops = env_with_position()
print("setup: venue", fake.positions["BTCUSDT"], "ledger", consumer.position.qty, "reconcile:", ex.reconcile_account("BTCUSDT"))
stops.sync()
(cid,) = [k for k, v in fake.algo.items() if v["algoStatus"] == "NEW"]
fake.algo[cid]["algoStatus"] = "TRIGGERED"          # status flips first; actualOrderId still ""
print("sync #1 issues:", stops.sync())
fake.trigger_stop(cid, D("95.9"))                   # venue then closes the position (FINISHED + actualOrderId)
print("venue position after stop fill:", fake.positions["BTCUSDT"])
print("sync #2 issues:", stops.sync())
print("stop rows (status, attempt):", [(r["status"], r["attempt"]) for r in stops._rows(live_only=False)])
print("adopted stop intents in journal:", [r["intent_id"] for r in ex.journal.orders(OWNER) if r["intent_id"].startswith("stop-")])
print("fills delivered to strategy:", [(f.side, f.qty) for f in ex.reconcile_pending()])
print("ledger qty:", ex.journal.checkpoint(OWNER)["position"]["qty"], "| venue qty:", fake.positions["BTCUSDT"])
print("reconcile_account:", ex.reconcile_account("BTCUSDT"), "| _ready:", ex._ready)
print("algo orders on venue:", {k[:12]: v["algoStatus"] for k, v in fake.algo.items()})
flat = make_order(order_id="BTCUSDT:2:kill_switch:0", source=OWNER, symbol="BTCUSDT", side="sell",
                  qty=D("0.1"), price=D("100"), reduce_only=True, reason="kill_switch")
n = len(fake.posts); ex.send_order(flat)
print("kill flatten placed an order:", len(fake.posts) > n)

print("=== C. stop placement rejected for a reason other than -2021")
fake, adapter, ex, clock, consumer, stops = env_with_position()
orig = fake.request_signed
def refusing(*, method, path, params=None):
    if path == "/fapi/v1/algoOrder" and method == "POST":
        raise BinanceNonRetryableError('HTTP 400: {"code":-4120,"msg":"not supported"}', http_status=400, code=-4120)
    return orig(method=method, path=path, params=params)
fake.request_signed = refusing
for i in range(3):
    issues = ex.reconcile_account("BTCUSDT") + stops.sync()     # same expression as runner/binance_target.py:304
    print(f"cycle #{i+1}: issues={issues} | entries _ready={ex._ready} | has_active_stop={stops.has_active_stop()}")
print("stop rows (status, attempt):", sorted((r["status"], r["attempt"]) for r in stops._rows(live_only=False)))
print("venue position still open:", fake.positions["BTCUSDT"])
