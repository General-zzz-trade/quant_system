# ruff: noqa
# Read-only diagnosis probe (no network): runs the real runtime classes against the in-memory fake venue used by the unit tests.
# Usage: python3 -B <this file> <empty work dir> [scenario]. See docs/arch/current-state.md.
"""Read-only experiment (no network): protective stop observed as TRIGGERED before actualOrderId is available;
and the execution lease with two state directories."""
import os, sys, json, logging
from decimal import Decimal as D
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
logging.disable(logging.CRITICAL)
from execution.adapters.binance import BinanceConfig
from execution.adapters.binance.adapter import BinanceAdapter
from execution.adapters.binance.execution_adapter import BinanceExecutionAdapter
from execution.adapters.binance.protective_stops import ProtectiveStopManager
from shared.execution_lease import ExecutionLease
from tests.unit.binance.fake_venue import FakeBinance, make_bars

OWNER = "alpha.BTCUSDT"
tmp = Path(sys.argv[1]); tmp.mkdir(parents=True, exist_ok=True)
clock = {"now": 1_800_000_000.0}
fake = FakeBinance(make_bars(10), clock=lambda: clock["now"]); fake.price["BTCUSDT"] = D("100")
adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
assert adapter.connect(symbols=("BTCUSDT",))
ex = BinanceExecutionAdapter(adapter, scope="binance-testnet-ut", owner=OWNER, journal_path=tmp / "orders.sqlite",
                             wait_seconds=0, clock=lambda: clock["now"], entry_mode="market")
stops = ProtectiveStopManager(adapter, ex, "BTCUSDT", stop_pct=D("0.04"), clock=lambda: clock["now"])
ex2 = None
from decision.modules.alpha_orders import make_open_order
from runner.binance_target import TargetFillConsumer
ex._ready = True
consumer = TargetFillConsumer(ex)
ex.send_order(make_open_order("BTCUSDT", "BTCUSDT", 100.0, 1, D("0.1"))[0])   # real market entry through the journal
for ev in ex.reconcile_pending():
    consumer.consume(ev)
print("S- entry: ledger", consumer.position.qty, "venue", fake.positions["BTCUSDT"], "unsettled:", [r["intent_id"] for r in ex.journal.orders(OWNER, unresolved=True)])
print("S0 sync:", stops.sync())
(cid,) = [k for k, v in fake.algo.items() if v["algoStatus"] == "NEW"]
# venue: stop fires, market close order fills, but the algo row shows TRIGGERED with no actualOrderId yet
algo = fake.algo[cid]
order = fake._new_order({"symbol": "BTCUSDT", "side": "SELL", "type": "MARKET", "quantity": "0.1", "newClientOrderId": "autoclose"})
fake._apply_fill(order, D("0.1"), D("95.9"))
algo.update(algoStatus="TRIGGERED", actualOrderId="")
print("S1 sync (TRIGGERED, actualOrderId empty):", stops.sync())
algo.update(algoStatus="FINISHED", actualOrderId=order["orderId"])   # now the venue reports the order id
clock["now"] += 5
print("S2 sync (FINISHED, actualOrderId set):", stops.sync())
print("S2 adopted stop rows in submissions:", [r["intent_id"] for r in ex.journal.orders(OWNER) if r["intent_id"].startswith("stop-")])
print("S2 fills delivered to strategy:", ex.reconcile_pending())
print("S2 ledger qty:", ex.journal.checkpoint(OWNER)["position"]["qty"], "venue qty:", fake.positions["BTCUSDT"])
print("S2 reconcile_account:", ex.reconcile_account("BTCUSDT"))
with ex.journal.connect() as db:
    print("S2 protective_stops rows:", [(r["attempt"], r["status"]) for r in db.execute("SELECT attempt,status FROM protective_stops ORDER BY created_at")])
print("S2 live venue stops:", [(v["side"], v["triggerPrice"], v["algoStatus"]) for v in fake.algo.values() if v["algoStatus"] == "NEW"])

# lease: same scope, two state directories
a = ExecutionLease(tmp / "binance" / "orders.sqlite", "binance-testnet-acct1")
try:
    b = ExecutionLease(tmp / "binance_target" / "orders.sqlite", "binance-testnet-acct1")
    print("L1 second lease with SAME scope in another state dir: acquired"); b.close()
except RuntimeError as exc:
    print("L1 refused:", exc)
try:
    ExecutionLease(tmp / "binance" / "orders.sqlite", "binance-testnet-acct1")
    print("L2 second lease same dir: acquired")
except RuntimeError as exc:
    print("L2 same dir refused:", exc)
a.close()
