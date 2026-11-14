# ruff: noqa
# Read-only diagnosis probe (no network): runs the real runtime classes against the in-memory fake venue used by the unit tests.
# Usage: python3 -B <this file> <empty work dir> [scenario]. See docs/arch/current-state.md.
"""Read-only experiment (no network): TargetRuntime execution mode against the repo's in-memory FakeBinance.
A) kill-switch flatten rejected once -> is it retried within the same bar?
E) maker entry resting + kill(flatten) -> is the entry cancelled / the position flattened?
"""
import os, sys, sqlite3, logging
from decimal import Decimal as D
from pathlib import Path
for name in ("QUANT_MODEL_SIGN_KEY", "BYBIT_BASE_URL", "BINANCE_TESTNET", "OKX_API_KEY"):
    os.environ.pop(name, None)
os.environ["QUANT_ALLOW_UNSIGNED_MODELS"] = "1"
os.environ["RISK_SCOPE"] = "binance-testnet-ut"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
logging.disable(logging.CRITICAL)
from lab.strategy import StrategySpec
from runner.binance_target import TargetRuntime
from tests.unit.lab.test_target_runtime import _world, _advance, LIVE

def rows(state_dir, like):
    db = sqlite3.connect(state_dir / "orders.sqlite")
    out = db.execute("SELECT intent_id,status,settled,reducing FROM submissions WHERE intent_id LIKE ? ORDER BY rowid", (like,)).fetchall()
    db.close()
    return out

def scenario_a(tmp):
    spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")
    world = _world(tmp, spec)
    btc = world["fakes"]["BTCUSDT"]
    state = tmp / "exec"
    rt = TargetRuntime(spec_path=world["spec_path"], state_dir=state, mode="execution", market=world["adapter"],
                       account=world["adapter"], clock=lambda: world["clock"]["now"], testnet=True)
    rt.warmup()
    i = world["warm"]
    while btc.positions.get("BTCUSDT", D(0)) == 0 and i < world["warm"] + LIVE - 10:
        _advance(world, rt, i); i += 1
    _advance(world, rt, i); i += 1
    s = rt.sessions["BTCUSDT"]
    print("A0 position ledger/venue:", s.position_qty(), btc.positions["BTCUSDT"], "pending:", dict(s.pending))
    posts0 = len(btc.posts)
    rt.kill.write(active=True, reason="operator", flatten=True)
    btc.fault, btc.fault_code = "reject", -1008          # one venue rejection of the reduce-only market order
    for n in range(6):                                    # 6 cycles, 6 s apart, SAME hourly bar
        world["clock"]["now"] += 6
        rt.cycle()
        print(f"A{n+1} cycle: BTC order POSTs since kill={len(btc.posts)-posts0} venue_pos={btc.positions['BTCUSDT']} "
              f"ledger={s.position_qty()} pending={dict(s.pending)}")
    print("A journal rows (kill_switch):", rows(state, "BTCUSDT:%:kill_switch:%"))
    ev = [l for l in (state / "target_events.jsonl").read_text().splitlines() if '"kill_switch"' in l and '"intent"' in l and "BTCUSDT" in l]
    print("A 'intent' lines written to target_events.jsonl for BTC kill_switch:", len(ev))
    _advance(world, rt, i)                                 # next hourly bar closes
    print(f"A next bar: BTC order POSTs since kill={len(btc.posts)-posts0} venue_pos={btc.positions['BTCUSDT']} ledger={s.position_qty()}")
    print("A journal rows (kill_switch):", rows(state, "BTCUSDT:%:kill_switch:%"))

def scenario_e(tmp):
    spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="maker", ttl_bars=3)
    world = _world(tmp, spec)
    btc = world["fakes"]["BTCUSDT"]
    state = tmp / "exec"
    rt = TargetRuntime(spec_path=world["spec_path"], state_dir=state, mode="execution", market=world["adapter"],
                       account=world["adapter"], clock=lambda: world["clock"]["now"], testnet=True)
    rt.warmup()
    i = world["warm"]
    s = rt.sessions["BTCUSDT"]
    while not btc.orders and i < world["warm"] + LIVE - 10:
        _advance(world, rt, i); i += 1
    oid = next(iter(btc.orders))
    o = btc.orders[oid]
    print("E0 resting entry:", o["type"], o["timeInForce"], o["side"], o["origQty"], o["status"], "pending:", dict(s.pending))
    btc.fill_limit(oid, D("0.001"))                       # small partial fill
    world["clock"]["now"] += 6; rt.cycle()
    print("E1 after partial fill: ledger=", s.position_qty(), "venue=", btc.positions["BTCUSDT"], "order status=", btc.orders[oid]["status"])
    rt.kill.write(active=True, reason="operator", flatten=True)
    posts0 = len(btc.posts)
    for n in range(3):
        world["clock"]["now"] += 6; rt.cycle()
        print(f"E{n+2} kill+flatten active: POSTs since kill={len(btc.posts)-posts0} entry status={btc.orders[oid]['status']} "
              f"venue_pos={btc.positions['BTCUSDT']} pending={dict(s.pending)}")
    btc.fill_limit(oid, D("0.002"))                       # the resting entry keeps filling under an active kill switch
    world["clock"]["now"] += 6; rt.cycle()
    print(f"E5 entry filled more under kill: venue_pos={btc.positions['BTCUSDT']} ledger={s.position_qty()} entry status={btc.orders[oid]['status']} POSTs since kill={len(btc.posts)-posts0}")

if __name__ == "__main__":
    base = Path(sys.argv[1])
    which = sys.argv[2]
    (base / which).mkdir(parents=True, exist_ok=True)
    {"a": scenario_a, "e": scenario_e}[which](base / which)
