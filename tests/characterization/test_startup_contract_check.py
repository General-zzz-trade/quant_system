"""Plan D1 (docs/arch/dynamic-universe-plan.md): execution start-up refuses a contract that is not a trading USDT
perpetual, has a delisting scheduled, does not allow the configured exchange leverage, or - with isolated margin - whose
own maintenance margin tier puts the protective stop beyond half the liquidation distance. Read-only, before the
runtime is built; driven through the entry point with the fake venue (``test_startup_account_check``'s world)."""
from __future__ import annotations

import re

import pytest

from tests.characterization.test_startup_account_check import _refused, _started, world  # noqa: F401 (a fixture)

TIER = {"bracket": 1, "notionalFloor": 0, "cum": 0.0}


def _isolated(world, monkeypatch, stop="0.045"):  # noqa: F811
    monkeypatch.setenv("BINANCE_MARGIN_TYPE", "ISOLATED")
    monkeypatch.setenv("BINANCE_EXCHANGE_LEVERAGE", "10")
    for symbol in ("BTCUSDT", "ETHUSDT"):
        world["fake"].leverage[symbol] = 10
        world["fake"].margin_type[symbol] = "isolated"
    world["argv"] = world["argv"] + ["--stop-pct", stop]


def test_the_margin_tiers_are_read_at_start_up_and_a_normal_contract_starts(world):  # noqa: F811
    _started(world)
    assert ("GET", "/fapi/v1/leverageBracket") in world["fake"].requests


@pytest.mark.parametrize("fields, words", [
    ({"status": "SETTLING"}, "ETHUSDT: not trading (settling)"),
    ({"status": "TRADING_HALT"}, "ETHUSDT: not trading (halted)"),
    ({"status": "PENDING_TRADING"}, "ETHUSDT: not trading (pending)"),
    ({"contractType": "CURRENT_QUARTER"}, "ETHUSDT: not a USDT perpetual"),
    ({"deliveryDate": 1_795_000_000_000}, "ETHUSDT: delisting scheduled at 2026-11-18"),
])
def test_a_contract_that_is_not_a_trading_usdt_perpetual_refuses_the_start(world, fields, words):  # noqa: F811
    world["fake"].contract["ETHUSDT"] = fields
    _refused(world, re.escape(words))


def test_a_contract_that_does_not_allow_the_configured_leverage_refuses_the_start(world):  # noqa: F811
    world["fake"].brackets["BTCUSDT"] = [{**TIER, "initialLeverage": 2, "notionalCap": 5000, "maintMarginRatio": 0.01}]
    _refused(world, re.escape("BTCUSDT: the exchange allows at most 2x, the configuration asks 3x"))


def test_unreadable_margin_tiers_refuse_the_start(world):  # noqa: F811
    world["fake"].bracket_error = True
    _refused(world, "margin tiers could not be read")


def test_isolated_a_stop_beyond_half_the_liquidation_distance_at_the_contracts_own_tier_refuses(world,  # noqa: F811
                                                                                              monkeypatch):
    _isolated(world, monkeypatch)
    world["fake"].brackets["ETHUSDT"] = [{**TIER, "initialLeverage": 20, "notionalCap": 5000,
                                          "maintMarginRatio": 0.02}]
    _refused(world, re.escape("ETHUSDT: protective stop 4.50% is not inside 50% of the 10x isolated liquidation "
                              "distance at its maintenance margin rate 2.00%"))


def test_isolated_a_stop_inside_half_the_distance_at_each_contracts_tier_starts(world, monkeypatch):  # noqa: F811
    _isolated(world, monkeypatch)
    _started(world)


@pytest.mark.parametrize("capital, multiple, starts", [("999", "1", True), ("1000", "1", False), ("500", "1", True),
                                                       ("500", "2", False)])
def test_the_tier_is_read_for_twice_each_symbols_share_of_the_strategy_capital(world, monkeypatch,  # noqa: F811
                                                                               capital, multiple, starts):
    """1,000 USDT over two symbols at 1x is 500 a symbol; twice that is 1,000, the first notional of the 2 % tier.
    At a strategy leverage of 2, 500 USDT reaches it too."""
    _isolated(world, monkeypatch)
    monkeypatch.setenv("STRATEGY_CAPITAL_USDT", capital)
    monkeypatch.setenv("STRATEGY_LEVERAGE", multiple)
    world["fake"].brackets["ETHUSDT"] = [
        {**TIER, "initialLeverage": 50, "notionalCap": 1000, "maintMarginRatio": 0.004},
        {**TIER, "bracket": 2, "initialLeverage": 20, "notionalFloor": 1000, "notionalCap": 5000,
         "maintMarginRatio": 0.02}]
    if starts:
        _started(world)
    else:
        _refused(world, re.escape("ETHUSDT: protective stop 4.50%"))
