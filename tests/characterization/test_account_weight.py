"""R5 of the runtime extension (docs/arch/runtime-lab-strategies-plan.md): with 10 symbols, the requests the runtime
makes stay well inside the client's request-weight budget.

Binance allows 2400 request weight per minute per IP (``GET /fapi/v1/exchangeInfo`` ``rateLimits``, read 2026-11-16);
the client's own budget is half that, 1200 a minute, shared with anything else on the host's IP
(``trading.adapters.binance.rate_limit``: a pool of 1200 refilled 20 a second, synced from the used-weight header; a
request beyond it is refused locally). The weights are the client's table
(``trading.adapters.binance.rate_limit.endpoint_weight``, ADR-0011 point 7: a position read 5, one symbol's open
orders 1, every symbol's 40). Not in the measure: a strategy capital adds ``/fapi/v1/income`` reads (30 per symbol
every 10 minutes, about 30 a minute at 10 symbols) and the loss stop's position read of each held symbol once a minute
(5 each, about 50 a minute at 10 symbols): about 520 in all, still under the bound. The world: 10 symbols on fake
venues, the overlay trading every one of them, then ten minutes of 5-second cycles inside one hour with the positions
held - the steady state of the exit side, which makes almost all of the requests.
"""
from __future__ import annotations

import collections
import json

from lab.strategy import StrategySpec
from runner.binance_target import TargetRuntime
from tests.support.state_db import state_dir_db
from tests.unit.binance.fake_venue import HOUR, FakeBinance, make_bars
from trading.adapters.binance.adapter import BinanceAdapter, BinanceConfig
from trading.adapters.binance.rate_limit import endpoint_weight

SPEC = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT",
           "LTCUSDT")
LIVE = 30
CLIENT_BUDGET_PER_MINUTE = 1200          # R5 audit S2: the limit that binds first
ACCOUNT_WIDE = ("/fapi/v2/positionRisk", "/fapi/v1/openOrders", "/fapi/v1/openAlgoOrders")


def _world(tmp_path):
    warm = SPEC.window_hours() + 48
    rows = {s: make_bars(warm + LIVE + 1, base=100.0 * (k + 1)) for k, s in enumerate(SYMBOLS)}
    start = rows["BTCUSDT"][0]["start"]
    funding = [(t, 0.0001) for t in range(start, start + (warm + LIVE + 9) * HOUR, 8 * HOUR)]
    clock = {"now": (rows["BTCUSDT"][warm]["start"] + 5_000) / 1000}
    fakes = {s: FakeBinance(r, clock=lambda: clock["now"], funding=funding) for s, r in rows.items()}
    for fake in fakes.values():
        fake.listed, fake.funding_hours = SYMBOLS, {s: 8 for s in SYMBOLS}
    weights: collections.Counter = collections.Counter()
    calls: collections.Counter = collections.Counter()

    def count(path, params):
        key = (path, "symbol" if (params or {}).get("symbol") else "account")
        calls[key] += 1
        weights[key] += endpoint_weight(path, params)

    class Router:      # one fake venue per symbol; an account-wide read asks each of them, and is counted once
        def request_public(self, *, method, path, params=None):
            count(path, params)
            return fakes[(params or {}).get("symbol", "BTCUSDT")].request_public(method=method, path=path,
                                                                                  params=params)

        def request_signed(self, *, method, path, params=None):
            count(path, params)
            p = params or {}
            if p.get("symbol"):
                return fakes[p["symbol"]].request_signed(method=method, path=path, params=params)
            if p.get("clientAlgoId"):
                owner = next((f for f in fakes.values() if p["clientAlgoId"] in f.algo), fakes["BTCUSDT"])
                return owner.request_signed(method=method, path=path, params=params)
            if path in ACCOUNT_WIDE:        # each venue answers for its own symbol and one no venue owns (plan D2)
                return [row for s, f in fakes.items()
                        for row in f.request_signed(method=method, path=path, params=p)
                        if row["symbol"] == s or row["symbol"] not in fakes]
            return fakes["BTCUSDT"].request_signed(method=method, path=path, params=params)

    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=Router())
    assert adapter.connect(symbols=SYMBOLS)
    spec_path = tmp_path / "final_spec.json"
    spec_path.write_text(json.dumps({"schema": 2, "kind": "target_strategy",
                                     "strategy": {"name": "overlay", "params": SPEC.record()},
                                     "execution": {"band": SPEC.band, "execution": SPEC.execution,
                                                   "ttl_bars": SPEC.ttl_bars},
                                     "symbols": list(SYMBOLS), "allow_short": True, "passed": False}))
    return dict(rows=rows, clock=clock, fakes=fakes, adapter=adapter, spec_path=spec_path, warm=warm,
                weights=weights, calls=calls)


def _steady_minute(tmp_path):
    world = _world(tmp_path)
    state_dir_db(tmp_path / "exec")
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                            market=world["adapter"], account=world["adapter"],
                            clock=lambda: world["clock"]["now"], testnet=True)
    runtime.sessions["BTCUSDT"].gate.inner._limiter._monotonic = lambda: world["clock"]["now"]
    runtime.warmup()
    for i in range(world["warm"], world["warm"] + LIVE):
        world["clock"]["now"] = (world["rows"]["BTCUSDT"][i]["start"] + HOUR + 5_000) / 1000
        for _ in range(3):                           # a few cycles per close, so entries the per-second limit
            runtime.cycle()                          # deferred are sent, as the next 5-second cycles send them
            world["clock"]["now"] += 5
    holding = [s for s, f in world["fakes"].items() if f.positions.get(s)]
    world["weights"].clear()
    world["calls"].clear()
    for _ in range(12 * 10):
        world["clock"]["now"] += 5
        runtime.cycle()
    per_minute = {key: n / 10 for key, n in world["weights"].items()}
    calls = {key: n / 10 for key, n in world["calls"].items()}
    return per_minute, calls, holding


def test_ten_symbols_use_a_small_part_of_the_request_weight(tmp_path):
    per_minute, calls, holding = _steady_minute(tmp_path)
    total = sum(per_minute.values())
    print("weight per minute", total, sorted(per_minute.items()), "calls", sorted(calls.items()))
    assert len(holding) >= 8                         # most symbols hold a position, each with a stop to keep
    assert total <= CLIENT_BUDGET_PER_MINUTE / 2, (total, per_minute)
