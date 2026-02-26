"""Property-based tests for state reducers."""
from __future__ import annotations

from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from state.account import AccountState
from state.market import MarketState
from state.position import PositionState
from state.reducers.account import AccountReducer
from state.reducers.market import MarketReducer
from state.reducers.position import PositionReducer

from tests_unit.properties.strategies import fill_events, market_events


# ── PositionReducer invariants ───────────────────────────


@given(events=st.lists(fill_events(), min_size=1, max_size=50))
@settings(max_examples=200)
def test_position_reducer_never_returns_none(events):
    """PositionReducer always returns a valid state, never None."""
    state = PositionState.empty("BTCUSDT")
    reducer = PositionReducer()
    for ev in events:
        result = reducer.reduce(state, ev)
        assert result.state is not None
        state = result.state


@given(events=st.lists(fill_events(), min_size=1, max_size=50))
@settings(max_examples=200)
def test_position_flat_implies_no_avg_price(events):
    """When position qty is zero, avg_price should be None."""
    state = PositionState.empty("BTCUSDT")
    reducer = PositionReducer()
    for ev in events:
        result = reducer.reduce(state, ev)
        state = result.state
    if state.qty == Decimal("0"):
        assert state.avg_price is None


@given(events=st.lists(fill_events(), min_size=1, max_size=50))
@settings(max_examples=200)
def test_position_avg_price_non_negative(events):
    """avg_price is always None (flat) or positive."""
    state = PositionState.empty("BTCUSDT")
    reducer = PositionReducer()
    for ev in events:
        result = reducer.reduce(state, ev)
        state = result.state
    if state.avg_price is not None:
        assert state.avg_price > 0


# ── AccountReducer invariants ────────────────────────────


@given(events=st.lists(fill_events(), min_size=1, max_size=50))
@settings(max_examples=200)
def test_account_reducer_never_returns_none(events):
    """AccountReducer always returns a valid state."""
    state = AccountState.initial(currency="USDT", balance=Decimal("10000"))
    reducer = AccountReducer()
    for ev in events:
        result = reducer.reduce(state, ev)
        assert result.state is not None
        state = result.state


@given(events=st.lists(fill_events(), min_size=1, max_size=50))
@settings(max_examples=200)
def test_account_fees_always_non_negative(events):
    """Accumulated fees_paid should never be negative."""
    state = AccountState.initial(currency="USDT", balance=Decimal("10000"))
    reducer = AccountReducer()
    for ev in events:
        # Only use non-negative fees
        if ev.fee is not None and ev.fee < 0:
            ev.fee = Decimal("0")
        result = reducer.reduce(state, ev)
        state = result.state
    assert state.fees_paid >= 0


# ── MarketReducer invariants ─────────────────────────────


@given(events=st.lists(market_events(), min_size=1, max_size=30))
@settings(max_examples=200)
def test_market_reducer_close_equals_last_price(events):
    """After any market bar, close equals last_price."""
    state = MarketState.empty("BTCUSDT")
    reducer = MarketReducer()
    for ev in events:
        result = reducer.reduce(state, ev)
        state = result.state
    # After at least one bar, last_price must be set
    assert state.last_price is not None
    assert state.close == state.last_price


@given(events=st.lists(market_events(), min_size=1, max_size=30))
@settings(max_examples=200)
def test_market_reducer_never_returns_none(events):
    """MarketReducer always returns a valid state."""
    state = MarketState.empty("BTCUSDT")
    reducer = MarketReducer()
    for ev in events:
        result = reducer.reduce(state, ev)
        assert result.state is not None
        state = result.state


@given(events=st.lists(market_events(), min_size=1, max_size=30))
@settings(max_examples=200)
def test_market_reducer_high_ge_low(events):
    """After any bar, high >= low (when both are set)."""
    state = MarketState.empty("BTCUSDT")
    reducer = MarketReducer()
    for ev in events:
        result = reducer.reduce(state, ev)
        state = result.state
    if state.high is not None and state.low is not None:
        assert state.high >= state.low
