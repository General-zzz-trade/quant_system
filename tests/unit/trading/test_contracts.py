"""Plan D1 (docs/arch/dynamic-universe-plan.md): the start-up rule for one contract - a trading USDT perpetual (crypto
or TradFi), no delisting scheduled, the configured exchange leverage allowed, and with isolated margin the stop inside
half the liquidation distance at the contract's own maintenance margin tier."""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from trading.domain.contracts import Bracket, Contract, MarginTiers, contract_problems

OPEN = Contract("ETHUSDT", "crypto", "open", "USDT", None)


def tiers(*rows, symbol="ETHUSDT"):
    """rows: (floor, cap, maintenance rate, max leverage)"""
    return MarginTiers(symbol, tuple(Bracket(D(a), D(b), D(m), x) for a, b, m, x in rows))


LOW = tiers(("0", "50000", "0.004", 125), ("50000", "250000", "0.005", 100))


def check(contract=OPEN, margin_tiers=LOW, *, stop="0.045", leverage=10, margin="ISOLATED", notional="300"):
    return contract_problems("ETHUSDT", contract, margin_tiers, stop_pct=D(stop), leverage=leverage, margin=margin,
                             notional=D(notional))


def test_a_trading_perpetual_with_room_for_its_stop_has_no_problem():
    assert check() == []
    assert check(Contract("XAUUSDT", "tradfi", "open", "USDT", None)) == []          # TradFi too (owner 2026-11-19)


def test_the_tier_is_the_one_of_the_position_size():
    t = tiers(("0", "5000", "0.01", 50), ("5000", "25000", "0.025", 20))
    assert t.at(D("0")).maintenance_rate == D("0.01") and t.at(D("4999.99")).maintenance_rate == D("0.01")
    assert t.at(D("5000")).maintenance_rate == D("0.025") and t.at(D("25000")) is None
    assert t.max_leverage() == 50 and MarginTiers("X", ()).max_leverage() == 0


@pytest.mark.parametrize("contract, words", [
    (None, "not listed"),
    (Contract("ETHUSDT", "other", "open", "USDT", None), "not a USDT perpetual"),
    (Contract("ETHUSDT", "crypto", "open", "USDC", None), "not a USDT perpetual"),
    (Contract("ETHUSDT", "crypto", "settling", "USDT", None), "not trading (settling)"),
    (Contract("ETHUSDT", "tradfi", "halted", "USDT", None), "not trading (halted)"),
    (Contract("ETHUSDT", "crypto", "pending", "USDT", None), "not trading (pending)"),
    (Contract("ETHUSDT", "crypto", "open", "USDT", 1_795_000_000_000), "delisting scheduled at 2026-11-18"),
])
def test_a_contract_that_is_not_a_trading_usdt_perpetual_is_refused(contract, words):
    problems = check(contract)
    assert len(problems) == 1 and words in problems[0] and problems[0].startswith("ETHUSDT: ")


def test_unknown_margin_tiers_are_refused():
    assert check(margin_tiers=None) == ["ETHUSDT: margin tiers unknown"]
    assert check(margin_tiers=MarginTiers("ETHUSDT", ())) == ["ETHUSDT: margin tiers unknown"]


def test_an_exchange_leverage_above_the_contracts_maximum_is_refused_with_any_margin_mode():
    capped = tiers(("0", "5000", "0.01", 5))
    assert check(margin_tiers=capped, leverage=10, margin=None) == [
        "ETHUSDT: the exchange allows at most 5x, the configuration asks 10x"]
    assert check(margin_tiers=capped, leverage=5, margin=None, stop="0.5") == []


def test_isolated_the_stop_must_sit_inside_half_the_liquidation_distance_at_the_contracts_own_rate():
    # 10x isolated: distance 1/10 - rate. At 0.4 % half of it is 4.8 %; at 2 % it is 4 %, so a 4.5 % stop fails.
    assert check(margin_tiers=tiers(("0", "5000", "0.004", 50)), stop="0.045") == []
    problems = check(margin_tiers=tiers(("0", "5000", "0.02", 50)), stop="0.045")
    assert len(problems) == 1 and "protective stop 4.50%" in problems[0] and "2.00%" in problems[0]
    assert "at most 0.0400" in problems[0]
    assert check(margin_tiers=tiers(("0", "5000", "0.02", 50)), stop="0.04") == []


def test_the_rate_is_read_at_the_planned_position_size():
    two = tiers(("0", "1000", "0.004", 50), ("1000", "5000", "0.02", 20))
    assert check(margin_tiers=two, notional="999") == []
    assert "2.00%" in check(margin_tiers=two, notional="1000")[0]
    assert check(margin_tiers=two, notional="5000") == ["ETHUSDT: no margin tier for a position of 5000 USDT"]


def test_cross_margin_has_no_liquidation_rule_here():
    assert check(margin_tiers=tiers(("0", "5000", "0.05", 50)), margin=None, stop="0.2") == []


# -- audit of D1 (2026-11-19) -----------------------------------------------------------------------------------
def test_the_leverage_rule_holds_with_isolated_margin_too():
    problems = check(margin_tiers=tiers(("0", "5000", "0.004", 5)), leverage=10, stop="0.01")
    assert problems == ["ETHUSDT: the exchange allows at most 5x, the configuration asks 10x"]


def test_a_delisting_time_of_zero_is_a_scheduled_delisting():
    assert check(Contract("ETHUSDT", "crypto", "open", "USDT", 0)) == [
        "ETHUSDT: delisting scheduled at 1970-01-01T00:00+00:00"]


def test_the_leverage_of_the_planned_positions_tier_must_allow_the_configured_leverage():
    two = tiers(("0", "500", "0.004", 10), ("500", "5000", "0.004", 5))
    assert check(margin_tiers=two, notional="499") == []
    assert check(margin_tiers=two, notional="600") == [
        "ETHUSDT: the exchange allows at most 5x for a position of 600 USDT, the configuration asks 10x"]
    assert check(margin_tiers=two, notional="600", margin=None) == [
        "ETHUSDT: the exchange allows at most 5x for a position of 600 USDT, the configuration asks 10x"]


@pytest.mark.parametrize("rate", ["NaN", "Infinity", "-0.01", "0.1"])
def test_an_unusable_maintenance_rate_is_refused(rate):
    problems = check(margin_tiers=tiers(("0", "5000", rate, 50)))
    assert problems == [f"ETHUSDT: maintenance margin rate {rate} is not usable"]
