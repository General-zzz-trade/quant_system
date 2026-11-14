"""Confirmed-fill accounting in StrategyPosition: short, flat and flip (migration plan 0.6, step 9)."""
from __future__ import annotations

from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from trading.domain.position import StrategyPosition


def _fill(side, qty, price, fee="0", fee_asset="USDT"):
    return SimpleNamespace(side=side, qty=qty, price=price, fee=fee, fee_asset=fee_asset)


def _state(position):
    return position.qty, position.entry_price, position.realized_pnl


def test_a_short_averages_its_entry_and_gains_when_covered_lower():
    position = StrategyPosition()
    position.apply(_fill("sell", "2", "100"))
    position.apply(_fill("sell", "2", "110"))
    assert _state(position) == (D("-4"), D("105"), D("0"))
    position.apply(_fill("buy", "1", "95"))                   # partial cover keeps the entry price
    assert _state(position) == (D("-3"), D("105"), D("10"))


def test_closing_to_flat_books_the_rest_and_clears_the_entry_price():
    position = StrategyPosition(D("-3"), D("105"), D("10"))
    position.apply(_fill("buy", "3", "100"))
    assert _state(position) == (D("0"), D("0"), D("25"))
    position.apply(_fill("buy", "1", "200"))                  # a new long starts from its own price
    assert _state(position) == (D("1"), D("200"), D("25"))


def test_a_flip_books_only_the_closed_part_and_restarts_at_the_flip_price():
    position = StrategyPosition()
    position.apply(_fill("buy", "1", "100"))
    position.apply(_fill("sell", "3", "90"))                  # long 1 -> short 2
    assert _state(position) == (D("-2"), D("90"), D("-10"))
    position.apply(_fill("buy", "2", "80"))
    assert _state(position) == (D("0"), D("0"), D("10"))


def test_a_flip_from_short_to_long_restarts_at_the_flip_price():
    position = StrategyPosition(D("-1"), D("50"))
    position.apply(_fill("buy", "4", "40"))                   # short 1 -> long 3
    assert _state(position) == (D("3"), D("40"), D("10"))


def test_fees_keep_their_sign_and_currency():
    position = StrategyPosition()
    position.apply(_fill("buy", "1", "100", fee="0.04", fee_asset="USDT"))
    position.apply(_fill("sell", "1", "100", fee="-0.01", fee_asset="USDT"))   # maker rebate
    position.apply(_fill("buy", "1", "100", fee="0.0001", fee_asset="BNB"))
    position.apply(_fill("sell", "1", "100", fee="0.02", fee_asset=None))
    assert position.fees == {"USDT": D("0.03"), "BNB": D("0.0001"), "unknown": D("0.02")}


@pytest.mark.parametrize("fact", [
    _fill("BUY", "1", "100"),
    _fill("hold", "1", "100"),
    _fill("buy", "0", "100"),
    _fill("buy", "-1", "100"),
    _fill("buy", "NaN", "100"),
    _fill("buy", "1", "0"),
    _fill("buy", "1", "Infinity"),
    _fill("buy", "1", "100", fee="NaN"),
])
def test_an_invalid_fill_changes_nothing(fact):
    position = StrategyPosition(D("1"), D("100"), D("5"), {"USDT": D("1")})
    with pytest.raises(ValueError):
        position.apply(fact)
    assert _state(position) == (D("1"), D("100"), D("5")) and position.fees == {"USDT": D("1")}


def test_the_stored_form_round_trips_exactly():
    position = StrategyPosition(D("-0.003"), D("65000.1"), D("-12.5"), {"USDT": D("0.07"), "BNB": D("0.0001")})
    restored = StrategyPosition.from_dict(position.to_dict())
    assert restored == position
    assert StrategyPosition.from_dict({"qty": "1", "entry_price": "2", "realized_pnl": "3"}).fees == {}
