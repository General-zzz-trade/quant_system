"""Target runtime: paper == lab simulation; execution lifecycle with restart."""
from __future__ import annotations

import json
from decimal import Decimal as D

import pytest

from execution.adapters.binance import BinanceConfig
from execution.adapters.binance.adapter import BinanceAdapter
from lab.data import frame_from_bars, funding_series
from lab.simulate import simulate
from lab.strategy import StrategySpec, compute_targets
from runner.binance_main import bar_from_kline
from runner.binance_target import TargetRuntime
from tests.unit.binance.fake_venue import HOUR, FakeBinance, make_bars

LIVE = 360


@pytest.fixture(autouse=True)
def _dev_signing(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BYBIT_BASE_URL", "BINANCE_TESTNET", "OKX_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")


def _world(tmp_path, spec: StrategySpec, *, passed=False):
    warm = spec.window_hours() + 48
    rows = {s: make_bars(warm + LIVE, base=b) for s, b in (("BTCUSDT", 30000.0), ("ETHUSDT", 2000.0))}
    start = rows["BTCUSDT"][0]["start"]
    end = start + (warm + LIVE + 9) * HOUR  # settlements published through the last processed close
    funding = [(t, 0.0001 if (t // HOUR) % 3 else -0.0002) for t in range(start, end, 8 * HOUR)]
    clock = {"now": (rows["BTCUSDT"][warm]["start"] + 5_000) / 1000}
    fakes = {s: FakeBinance(r, clock=lambda: clock["now"], funding=funding) for s, r in rows.items()}

    class Router:  # one venue per symbol, BinanceAdapter-compatible client
        def request_public(self, *, method, path, params=None):
            sym = (params or {}).get("symbol", "BTCUSDT")
            return fakes[sym].request_public(method=method, path=path, params=params)

        def request_signed(self, *, method, path, params=None):
            p = params or {}
            if p.get("symbol"):
                return fakes[p["symbol"]].request_signed(method=method, path=path, params=params)
            if p.get("clientAlgoId"):  # algo-order lookups carry no symbol: the venue holding the id answers
                owner = next((f for f in fakes.values() if p["clientAlgoId"] in f.algo), fakes["BTCUSDT"])
                return owner.request_signed(method=method, path=path, params=params)
            if path in ("/fapi/v2/positionRisk", "/fapi/v1/openOrders"):  # account-wide reads span both
                return [row for f in fakes.values() for row in f.request_signed(method=method, path=path, params=p)]
            return fakes["BTCUSDT"].request_signed(method=method, path=path, params=params)

    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=Router())
    assert adapter.connect(symbols=("BTCUSDT", "ETHUSDT"))
    spec_path = tmp_path / "final_spec.json"
    spec_path.write_text(json.dumps({"schema": 1, "kind": "btc_eth_multitimeframe_target",
                                     "symbols": ["BTCUSDT", "ETHUSDT"], "spec": spec.record(), "passed": passed}))
    return {"rows": rows, "funding": funding, "clock": clock, "fakes": fakes, "adapter": adapter,
            "spec_path": spec_path, "warm": warm}


def _advance(world, runtime, i):
    world["clock"]["now"] = (world["rows"]["BTCUSDT"][i]["start"] + HOUR + 5_000) / 1000
    runtime.cycle()


@pytest.mark.parametrize("execution,ttl", [("market", 1), ("maker", 1), ("maker", 4)])
def test_paper_runtime_reproduces_lab_simulation(tmp_path, execution, ttl):
    spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05,
                        execution=execution, ttl_bars=ttl)
    world = _world(tmp_path, spec)
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "paper", mode="paper",
                            market=world["adapter"], clock=lambda: world["clock"]["now"])
    runtime.warmup()
    for i in range(world["warm"], world["warm"] + LIVE):
        _advance(world, runtime, i)
    traded = 0
    for symbol in ("BTCUSDT", "ETHUSDT"):
        bars = [bar_from_kline(r) for r in world["rows"][symbol]]
        funding = {t: (D(str(r)), D("1")) for t, r in world["funding"]}
        targets = compute_targets(frame_from_bars(bars), funding_series(funding), spec)
        lab = simulate(symbol, bars, targets, funding, spec, start_ms=bars[world["warm"]].open_time)
        broker = runtime.sessions[symbol].sleeve.broker
        key = lambda f: (f["at"], f["side"], f["qty"], f["price"], f["fee"])  # noqa: E731
        assert [key(f) for f in broker.fills] == [key(f) for f in lab.fills]
        assert runtime.sessions[symbol].cursor == bars[-1].open_time
        assert float(broker.equity) == pytest.approx(lab.curve["equity"].iloc[-1], abs=1e-9)
        traded += len(lab.fills)
    assert traded > 0


def test_execution_lifecycle_restart_and_flatten(tmp_path, monkeypatch):
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")
    spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")
    world = _world(tmp_path, spec)
    btc = world["fakes"]["BTCUSDT"]

    def build():
        return TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                             market=world["adapter"], account=world["adapter"], clock=lambda: world["clock"]["now"],
                             testnet=True)

    runtime = build()
    runtime.warmup()
    i = world["warm"]
    while btc.positions.get("BTCUSDT", D(0)) == 0 and i < world["warm"] + LIVE - 10:
        _advance(world, runtime, i)
        i += 1
    position = btc.positions.get("BTCUSDT", D(0))
    assert position != 0, "a market entry should have filled"
    for fake in world["fakes"].values():  # one exchange-side stop per venue at most, never a pile
        assert len([a for a in fake.algo.values() if a["algoStatus"] == "NEW"]) <= 1
    _advance(world, runtime, i)
    i += 1
    session = runtime.sessions["BTCUSDT"]
    assert session.position_qty() == btc.positions["BTCUSDT"]
    assert any(a["algoStatus"] == "NEW" for a in btc.algo.values()), "exchange-side stop expected"

    posts = len(btc.posts)
    runtime = build()          # restart: position restored from the durable journal, nothing resent
    runtime.warmup()
    assert runtime.sessions["BTCUSDT"].position_qty() == btc.positions["BTCUSDT"]
    assert len(btc.posts) == posts

    runtime.kill.write(active=True, reason="test", flatten=True)
    _advance(world, runtime, i)
    _advance(world, runtime, i + 1)
    assert btc.positions["BTCUSDT"] == 0
    assert all(f.positions.get(s, D(0)) == 0 for s, f in world["fakes"].items())
    assert not [a for f in world["fakes"].values() for a in f.algo.values() if a["algoStatus"] == "NEW"]
    assert runtime.sessions["BTCUSDT"].position_qty() == 0
    assert not any(a["algoStatus"] == "NEW" for a in btc.algo.values())


def test_mainnet_refused_without_passed_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("RISK_SCOPE", "binance-mainnet-ut")
    spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10)
    world = _world(tmp_path, spec, passed=False)
    with pytest.raises(RuntimeError, match="gate not passed"):
        TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "m", mode="execution",
                      market=world["adapter"], account=world["adapter"], testnet=False)
