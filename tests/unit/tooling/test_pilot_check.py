"""The pilot's start check (tech lead 2026-11-17: "试运行前一键检查"): ``account_check --pilot``.

One read-only command lists what a mainnet pilot start still lacks: the settings (secrets only as set or not), the
pilot's own limits, the alert file's folder, the signed spec, the state directory and the account (no position or
open order at all, each spec symbol's leverage and margin type as the settings say, one-way mode, the wallet against
the strategy capital). GET requests only, on the allowlist; no secret in the output. Fake transport only.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json

import pytest

from scripts import account_check as ac
from tests.unit.tooling.test_account_check import CANNED, KEY, SECRET, Venue
from trading.infrastructure import state_dir as managed

SIGN = "sign-key-not-printed"
SCOPE = "binance-mainnet-pilot"


def _spec(tmp_path, *, passed=True, signed=True, symbols=("BTCUSDT", "ETHUSDT")):
    from lab.strategy import StrategySpec
    spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")
    path = tmp_path / "final_spec.json"
    path.write_text(json.dumps({"schema": 1, "kind": "btc_eth_multitimeframe_target", "symbols": list(symbols),
                                "spec": spec.record(), "passed": passed}))
    if signed:
        (tmp_path / "final_spec.json.sig").write_text(hmac.new(SIGN.encode(), path.read_bytes(),
                                                              hashlib.sha256).hexdigest())
    return path


def _state(tmp_path, monkeypatch):
    env = {"RISK_SCOPE": SCOPE, "BINANCE_TESTNET": "0", "BINANCE_API_KEY": KEY, "BINANCE_API_SECRET": SECRET}
    state = tmp_path / "state"
    assert managed.main(["--state-dir", str(state), "init"], stdout=io.StringIO(), env=env) == 0
    return state


def _settings(tmp_path, **over):
    alerts = tmp_path / "alerts"
    alerts.mkdir(exist_ok=True)
    values = {"BINANCE_API_KEY": KEY, "BINANCE_API_SECRET": SECRET, "BINANCE_TESTNET": "0", "RISK_SCOPE": SCOPE,
              "STRATEGY_CAPITAL_USDT": "150", "STRATEGY_LEVERAGE": "2", "STOP_PCT": "0.045",
              "BINANCE_EXCHANGE_LEVERAGE": "10", "BINANCE_MARGIN_TYPE": "ISOLATED",
              "ALERT_FILE": str(alerts / "mainnet.jsonl"), "HEARTBEAT_URL": "https://hc.example.invalid/ping",
              "QUANT_MODEL_SIGN_KEY": SIGN, "MAX_DAILY_DRAWDOWN_PCT": "3"}
    values.update(over)
    path = tmp_path / "credentials"
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items() if v is not None))
    return path


READY_POSITIONS = [
    {"symbol": "BTCUSDT", "positionSide": "BOTH", "positionAmt": "0", "leverage": "10", "marginType": "isolated"},
    {"symbol": "ETHUSDT", "positionSide": "BOTH", "positionAmt": "0", "leverage": "10", "marginType": "isolated"}]


def _run(tmp_path, monkeypatch, *, venue=None, spec=None, state=None, **settings):
    venue = venue or Venue(answers={"/fapi/v2/positionRisk": lambda q: READY_POSITIONS})
    out = io.StringIO()
    code = ac.main(["--env-file", str(_settings(tmp_path, **settings)), "--pilot", "--spec",
                    str(spec or _spec(tmp_path)), "--state-dir", str(state or _state(tmp_path, monkeypatch))],
                   fetch=venue, stdout=out)
    return code, out.getvalue(), venue


def test_a_ready_account_passes_every_item(tmp_path, monkeypatch):
    code, out, venue = _run(tmp_path, monkeypatch)
    assert code == 0, out
    assert "可以启动" in out and "✗" not in out
    assert KEY not in out and SECRET not in out and SIGN not in out          # no secret, ever
    assert {path for _, path, _, _ in venue.calls} <= {p for _, p, _ in ac.ALLOWED}
    assert all(not any(w in q for w in ("leverage=", "marginType=")) for _, _, q, _ in venue.calls)


@pytest.mark.parametrize("change, item", [
    (dict(MAX_DAILY_DRAWDOWN_PCT="5"), "日亏"),
    (dict(MAX_DAILY_DRAWDOWN_PCT=None), "日亏"),
    (dict(BINANCE_TESTNET="1"), "主网"),
    (dict(QUANT_MODEL_SIGN_KEY=None), "签名"),
    (dict(HEARTBEAT_URL=None), "配置检查"),
])
def test_a_setting_the_pilot_needs_is_named(tmp_path, monkeypatch, change, item):
    code, out, _ = _run(tmp_path, monkeypatch, **change)
    assert code == 1 and "还缺" in out
    assert any(line.startswith("✗") and item in line for line in out.splitlines()), out


def test_a_position_or_an_open_order_anywhere_on_the_account_is_named(tmp_path, monkeypatch):
    positions = READY_POSITIONS + [CANNED["/fapi/v2/positionRisk"][0]]          # a manual SOXL short
    venue = Venue(answers={"/fapi/v2/positionRisk": lambda q: positions,
                           "/fapi/v1/openOrders": lambda q: [{"symbol": "ETHUSDT", "side": "BUY", "type": "LIMIT"}]})
    code, out, _ = _run(tmp_path, monkeypatch, venue=venue)
    assert code == 1
    assert any(line.startswith("✗") and "仓位" in line and "SOXLUSDT" in line for line in out.splitlines())
    assert any(line.startswith("✗") and "挂单" in line for line in out.splitlines())


def test_a_spec_symbol_with_other_leverage_or_margin_is_named(tmp_path, monkeypatch):
    rows = [dict(READY_POSITIONS[0]), {**READY_POSITIONS[1], "leverage": "11", "marginType": "cross"}]
    code, out, _ = _run(tmp_path, monkeypatch, venue=Venue(answers={"/fapi/v2/positionRisk": lambda q: rows}))
    assert code == 1
    assert any(line.startswith("✗") and "ETHUSDT" in line and "11" in line for line in out.splitlines())
    assert any(line.startswith("✓") and "BTCUSDT" in line for line in out.splitlines())


def test_an_unsigned_unpassed_or_tampered_spec_is_named(tmp_path, monkeypatch):
    for name, kwargs in (("unsigned", dict(signed=False)), ("unpassed", dict(passed=False))):
        folder = tmp_path / name
        folder.mkdir()
        code, out, _ = _run(folder, monkeypatch, spec=_spec(folder, **kwargs))
        assert code == 1 and any(line.startswith("✗") and "规格" in line for line in out.splitlines()), out
    folder = tmp_path / "tampered"
    folder.mkdir()
    spec = _spec(folder)
    spec.write_text(spec.read_text().replace('"passed": true', '"passed": true '))   # changed after signing
    code, out, _ = _run(folder, monkeypatch, spec=spec)
    assert code == 1 and any(line.startswith("✗") and "签名" in line for line in out.splitlines())


def test_a_spec_with_a_symbol_other_than_btc_and_eth_is_named(tmp_path, monkeypatch):
    """Mainnet trades BTCUSDT and ETHUSDT only (owner 2026-11-17); the runtime refuses any other, so does the check."""
    rows = READY_POSITIONS + [{**READY_POSITIONS[0], "symbol": "SOLUSDT"}]
    venue = Venue(answers={"/fapi/v2/positionRisk": lambda q: rows})
    code, out, _ = _run(tmp_path, monkeypatch, venue=venue, spec=_spec(tmp_path, symbols=("SOLUSDT", "BTCUSDT")))
    assert code == 1
    assert any(line.startswith("✗") and "BTCUSDT 和 ETHUSDT" in line and "SOLUSDT" in line for line in out.splitlines())
    # a schema-2 spec too (audit of PR #153, M1)
    from lab.strategy import StrategySpec
    params = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")
    folder = tmp_path / "v2"
    folder.mkdir()
    path = folder / "final_spec.json"
    path.write_text(json.dumps({"schema": 2, "kind": "target_strategy",
                                "strategy": {"name": "overlay", "params": params.record()},
                                "execution": {"band": 0.05, "execution": "market", "ttl_bars": params.ttl_bars},
                                "symbols": ["SOLUSDT", "BTCUSDT"], "allow_short": False, "passed": True}))
    (folder / "final_spec.json.sig").write_text(hmac.new(SIGN.encode(), path.read_bytes(), hashlib.sha256).hexdigest())
    code, out, _ = _run(folder, monkeypatch, venue=venue, spec=path)
    assert code == 1
    assert any(line.startswith("✗") and "BTCUSDT 和 ETHUSDT" in line and "SOLUSDT" in line for line in out.splitlines())


def test_a_state_directory_that_fails_its_check_or_belongs_to_another_scope_is_named(tmp_path, monkeypatch):
    code, out, _ = _run(tmp_path, monkeypatch, state=tmp_path / "never-initialized")
    assert code == 1 and any(line.startswith("✗") and "状态目录" in line for line in out.splitlines())


def test_an_alert_folder_that_cannot_be_written_is_named(tmp_path, monkeypatch):
    code, out, _ = _run(tmp_path, monkeypatch, ALERT_FILE=str(tmp_path / "missing" / "a.jsonl"))
    assert code == 1 and any(line.startswith("✗") and "告警" in line for line in out.splitlines())


def test_a_wallet_below_the_strategy_capital_is_named(tmp_path, monkeypatch):
    code, out, _ = _run(tmp_path, monkeypatch, STRATEGY_CAPITAL_USDT="300")         # the fake wallet holds 167.5
    assert code == 1 and any(line.startswith("✗") and "余额" in line for line in out.splitlines())


def test_hedge_mode_is_named(tmp_path, monkeypatch):
    venue = Venue(answers={"/fapi/v2/positionRisk": lambda q: READY_POSITIONS,
                           "/fapi/v1/positionSide/dual": lambda q: {"dualSidePosition": True}})
    code, out, _ = _run(tmp_path, monkeypatch, venue=venue)
    assert code == 1 and any(line.startswith("✗") and "单向" in line for line in out.splitlines())


# -- plan D1 (audit L7): the same contract rule the runtime's start applies --------------------------------------
def test_each_spec_symbols_contract_and_margin_tier_pass_on_a_ready_account(tmp_path, monkeypatch):
    code, out, venue = _run(tmp_path, monkeypatch)
    assert code == 0
    assert "✓ BTCUSDT：合约和保证金档位（正在交易、没安排下架、按 300 USDT 的档位允许 10 倍和止损）" in out
    paths = {(path, "X-MBX-APIKEY" in headers) for _, path, _, headers in venue.calls}
    assert ("/fapi/v1/exchangeInfo", False) in paths and ("/fapi/v1/leverageBracket", True) in paths


def test_a_delisting_or_a_tier_that_puts_the_stop_beyond_half_the_distance_is_named(tmp_path, monkeypatch):
    info = {"symbols": [{"symbol": "BTCUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT",
                         "deliveryDate": 4133404800000},
                        {"symbol": "ETHUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT",
                         "deliveryDate": 1_795_500_000_000}]}
    tiers = [{"symbol": "BTCUSDT", "brackets": [{"bracket": 1, "initialLeverage": 20, "notionalCap": 5000,
                                                 "notionalFloor": 0, "maintMarginRatio": 0.02, "cum": 0}]},
             {"symbol": "ETHUSDT", "brackets": [{"bracket": 1, "initialLeverage": 125, "notionalCap": 50000,
                                                 "notionalFloor": 0, "maintMarginRatio": 0.004, "cum": 0}]}]
    venue = Venue(answers={"/fapi/v2/positionRisk": lambda q: READY_POSITIONS, "/fapi/v1/exchangeInfo": lambda q: info,
                           "/fapi/v1/leverageBracket": lambda q: tiers})
    code, out, _ = _run(tmp_path, monkeypatch, venue=venue)
    assert code != 0
    assert "✗ BTCUSDT：合约和保证金档位（BTCUSDT: protective stop 4.50% is not inside 50%" in out
    assert "✗ ETHUSDT：合约和保证金档位（ETHUSDT: delisting scheduled at 2026-11-24" in out
