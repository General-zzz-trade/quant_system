# ruff: noqa
# Read-only diagnosis probe (no network): runs the real runtime classes against the in-memory fake venue used by the unit tests.
# Usage: python3 -B <this file> <empty work dir> [scenario]. See docs/arch/current-state.md.
"""Read-only experiment (no network): one order filled in 150 trades while the executor asks userTrades with limit=100."""
import sys, logging
from decimal import Decimal as D
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
logging.disable(logging.CRITICAL)
from decision.modules.alpha_orders import make_close_order, make_open_order
from execution.adapters.binance import BinanceConfig
from execution.adapters.binance.adapter import BinanceAdapter
from execution.adapters.binance.execution_adapter import BinanceExecutionAdapter
from runner.binance_target import TargetFillConsumer
from tests.unit.binance.fake_venue import FakeBinance, make_bars

class Venue(FakeBinance):
    limits = []
    def _apply_fill(self, order, qty, price):          # every market order is matched in 0.001 slices
        if order["type"] == "MARKET" and qty > D("0.001"):
            for _ in range(int(qty / D("0.001"))):
                super()._apply_fill(order, D("0.001"), price)
            return
        super()._apply_fill(order, qty, price)
    def request_signed(self, *, method, path, params=None):
        rows = super().request_signed(method=method, path=path, params=params)
        if path == "/fapi/v1/userTrades":                # honour the request's limit like the real endpoint
            self.limits.append(params.get("limit"))
            return rows[:int(params["limit"])]
        return rows

OWNER = "alpha.BTCUSDT"
tmp = Path(sys.argv[1]); tmp.mkdir(parents=True, exist_ok=True)
clock = {"now": 1_800_000_000.0}
fake = Venue(make_bars(10), clock=lambda: clock["now"]); fake.price["BTCUSDT"] = D("100")
adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
assert adapter.connect(symbols=("BTCUSDT",))
ex = BinanceExecutionAdapter(adapter, scope="binance-testnet-ut", owner=OWNER, journal_path=tmp / "orders.sqlite",
                             wait_seconds=0, clock=lambda: clock["now"], entry_mode="market")
ex._ready = True
consumer = TargetFillConsumer(ex)
def drain():
    for ev in ex.reconcile_pending():
        consumer.consume(ev)
def rows():
    return [(r["status"], r["settled"], r["reducing"], r["qty"]) for r in ex.journal.orders(OWNER)]

ex.send_order(make_open_order("BTCUSDT", "BTCUSDT", 100.0, 1, D("0.05"))[0]); drain()      # 50 trades: settles
print("T0 entry 0.05 (50 trades):", rows(), "ledger", consumer.position.qty, "venue", fake.positions["BTCUSDT"])
ex.reconcile_account("BTCUSDT")
ex.send_order(make_open_order("BTCUSDT", "BTCUSDT", 100.0, 1, D("0.15"))[0]); drain()      # 150 trades
for _ in range(3):
    clock["now"] += 3600; drain()
print("T1 entry 0.15 (150 trades), 3h later:", rows()[-1], "ledger", consumer.position.qty, "venue", fake.positions["BTCUSDT"])
print("T1 userTrades limits requested:", sorted(set(fake.limits)))
print("T1 reconcile_account:", ex.reconcile_account("BTCUSDT"))
posts = len(fake.posts)
ex.send_order(make_close_order("BTCUSDT", "BTCUSDT", 100.0, 1, "kill_switch", D("0"), D("0.001"))[0]); drain()
for _ in range(3):
    clock["now"] += 3600; drain()
print("T2 close-all (qty=0) sent:", len(fake.posts) - posts, "POST; row:", rows()[-1], "ledger", consumer.position.qty, "venue", fake.positions["BTCUSDT"])
posts = len(fake.posts)
ex.send_order(make_close_order("BTCUSDT", "BTCUSDT", 100.0, 1, "kill_switch_retry", D("0"), D("0.001"))[0]); drain()
print("T3 another close attempt:", len(fake.posts) - posts, "POST; unresolved rows:", [(r["status"], r["reducing"]) for r in ex.journal.orders(OWNER, unresolved=True)],
      "ledger", consumer.position.qty, "venue", fake.positions["BTCUSDT"])
