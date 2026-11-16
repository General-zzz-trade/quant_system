"""The pilot's two settings (owner, 2026-11-16 evening): positions of 2 to 3 times the strategy capital on 10x
isolated margin, with a 4.5 % exchange stop.

- ``STRATEGY_LEVERAGE``: how many times the strategy capital a full position is. 1 unless set; above 1 only with a
  strategy capital; at most 3 (``MAX_TOTAL_LEVERAGE``, the account's exposure cap) and never above the exchange
  leverage setting the account must have. A person writes it; nothing in the program changes it.
- ``STOP_PCT``: the exchange stop's distance when the command line gives none (the mainnet unit gives none). 8 %
  unless set; within the stop manager's bounds. The command line's ``--stop-pct`` still wins.
"""
from __future__ import annotations

import json
from decimal import Decimal as D

import pytest

from trading.infrastructure.config import load_config

ENV = {"RISK_SCOPE": "binance-ut", "BINANCE_TESTNET": "0", "HEARTBEAT_URL": "https://hc.example.invalid/x",
       "ALERT_FILE": "/var/lib/quant/alerts/mainnet.jsonl", "STRATEGY_CAPITAL_USDT": "150",
       "BINANCE_EXCHANGE_LEVERAGE": "10", "BINANCE_MARGIN_TYPE": "ISOLATED"}


def _load(stop_pct=None, **over):
    env = {k: v for k, v in dict(ENV, **over).items() if v is not None}
    return load_config(env, execution=True, stop_pct=stop_pct)[0]


@pytest.mark.parametrize("raw,expected", [(None, "1"), ("", "1"), ("1", "1"), ("2", "2"), ("2.5", "2.5"), ("3", "3")])
def test_the_strategy_leverage_is_read_once_and_recorded(raw, expected):
    config = _load(STRATEGY_LEVERAGE=raw)
    assert config.strategy_leverage == D(expected)
    assert json.loads(json.dumps(config.record()))["strategy_leverage"] == expected


@pytest.mark.parametrize("raw", ["3.01", "4", "10", "0.99", "0", "-2", "nan", "inf", "two"])
def test_a_leverage_outside_1_to_3_is_refused_and_named(raw):
    with pytest.raises(RuntimeError, match="STRATEGY_LEVERAGE"):
        _load(STRATEGY_LEVERAGE=raw)


def test_the_leverage_may_not_exceed_the_exchange_leverage_setting():
    with pytest.raises(RuntimeError, match="BINANCE_EXCHANGE_LEVERAGE"):
        _load(STRATEGY_LEVERAGE="3", BINANCE_EXCHANGE_LEVERAGE="2")
    assert _load(STRATEGY_LEVERAGE="2", BINANCE_EXCHANGE_LEVERAGE="2").strategy_leverage == D(2)


def test_a_leverage_above_1_needs_a_strategy_capital():
    testnet = dict(BINANCE_TESTNET="1", RISK_SCOPE="binance-testnet-ut", STRATEGY_CAPITAL_USDT=None)
    assert _load(**testnet).strategy_leverage == D(1)
    with pytest.raises(RuntimeError, match="STRATEGY_CAPITAL_USDT"):
        _load(**testnet, STRATEGY_LEVERAGE="2")


def test_paper_reads_neither_setting():
    config = load_config({"STRATEGY_LEVERAGE": "3", "STOP_PCT": "0.045"}, execution=False, stop_pct=None)[0]
    assert config.strategy_leverage == D(1) and config.stop_pct == D("0.08")


def test_the_stop_distance_comes_from_the_command_line_then_stop_pct_then_8_per_cent():
    assert _load().stop_pct == D("0.08")
    assert _load(STOP_PCT="0.045").stop_pct == D("0.045")
    assert _load(D("0.03"), STOP_PCT="0.045").stop_pct == D("0.03")             # the command line wins
    assert json.loads(json.dumps(_load(STOP_PCT="0.045").record()))["stop_pct"] == "0.045"


@pytest.mark.parametrize("raw", ["0.004", "0.16", "0", "-0.05", "nan", "4.5", "five"])
def test_a_stop_distance_outside_its_bounds_is_refused_and_named(raw):
    with pytest.raises(RuntimeError, match="STOP_PCT"):
        _load(STOP_PCT=raw)


def test_the_pilot_s_combination_passes_the_liquidation_check_and_8_per_cent_does_not():
    from trading.domain.risk.liquidation import check_stop_inside_liquidation
    pilot = _load(STOP_PCT="0.045", STRATEGY_LEVERAGE="2")
    check_stop_inside_liquidation(float(pilot.stop_pct), pilot.leverage, pilot.margin)
    default = _load()
    with pytest.raises(RuntimeError, match="stop"):
        check_stop_inside_liquidation(float(default.stop_pct), default.leverage, default.margin)
