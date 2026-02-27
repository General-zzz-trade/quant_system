"""Property-based tests for StatePipeline."""
from __future__ import annotations

from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from state.account import AccountState
from state.market import MarketState
from state.position import PositionState
from engine.pipeline import PipelineConfig, PipelineInput, StatePipeline

from tests_unit.properties.strategies import fill_events, market_events


def _make_pipeline() -> StatePipeline:
    return StatePipeline(config=PipelineConfig(
        build_snapshot_on_change_only=False,
        fail_on_missing_symbol=False,
    ))


def _run_pipeline(pipeline: StatePipeline, events: list) -> tuple:
    """Run a sequence of events through the pipeline, return final state."""
    markets = {"BTCUSDT": MarketState.empty("BTCUSDT")}
    account = AccountState.initial(currency="USDT", balance=Decimal("10000"))
    positions: dict[str, PositionState] = {}
    idx = 0
    indices = []

    for ev in events:
        inp = PipelineInput(
            event=ev,
            event_index=idx,
            symbol_default="BTCUSDT",
            markets=markets,
            account=account,
            positions=positions,
        )
        out = pipeline.apply(inp)
        markets = dict(out.markets)
        account = out.account
        positions = dict(out.positions)
        idx = out.event_index
        if out.advanced:
            indices.append(idx)

    return markets, account, positions, idx, indices


@given(events=st.lists(
    st.one_of(fill_events(), market_events()),
    min_size=1,
    max_size=30,
))
@settings(max_examples=200)
def test_deterministic_replay(events):
    """Same event sequence produces identical state."""
    pipeline = _make_pipeline()
    m1, a1, p1, idx1, _ = _run_pipeline(pipeline, events)
    m2, a2, p2, idx2, _ = _run_pipeline(pipeline, events)

    assert m1 == m2
    assert a1 == a2
    assert p1 == p2
    assert idx1 == idx2


@given(events=st.lists(
    st.one_of(fill_events(), market_events()),
    min_size=1,
    max_size=30,
))
@settings(max_examples=200)
def test_event_index_monotonic(events):
    """event_index is strictly monotonically increasing for fact events."""
    pipeline = _make_pipeline()
    _, _, _, _, indices = _run_pipeline(pipeline, events)

    for i in range(len(indices) - 1):
        assert indices[i] < indices[i + 1]


@given(events=st.lists(
    st.one_of(fill_events(), market_events()),
    min_size=1,
    max_size=30,
))
@settings(max_examples=200)
def test_pipeline_always_returns_valid_output(events):
    """Pipeline output is never None for any valid event."""
    pipeline = _make_pipeline()
    markets = {"BTCUSDT": MarketState.empty("BTCUSDT")}
    account = AccountState.initial(currency="USDT", balance=Decimal("10000"))
    positions: dict[str, PositionState] = {}
    idx = 0

    for ev in events:
        inp = PipelineInput(
            event=ev,
            event_index=idx,
            symbol_default="BTCUSDT",
            markets=markets,
            account=account,
            positions=positions,
        )
        out = pipeline.apply(inp)
        assert out is not None
        assert out.market is not None
        assert out.account is not None
        markets = dict(out.markets)
        account = out.account
        positions = dict(out.positions)
        idx = out.event_index


@given(events=st.lists(fill_events(), min_size=1, max_size=20))
@settings(max_examples=200)
def test_pipeline_fill_events_always_advance(events):
    """Fill events are fact events and always advance event_index."""
    pipeline = _make_pipeline()
    markets = {"BTCUSDT": MarketState.empty("BTCUSDT")}
    account = AccountState.initial(currency="USDT", balance=Decimal("10000"))
    positions: dict[str, PositionState] = {}
    idx = 0
    advanced_count = 0

    for ev in events:
        inp = PipelineInput(
            event=ev,
            event_index=idx,
            symbol_default="BTCUSDT",
            markets=markets,
            account=account,
            positions=positions,
        )
        out = pipeline.apply(inp)
        if out.advanced:
            advanced_count += 1
        markets = dict(out.markets)
        account = out.account
        positions = dict(out.positions)
        idx = out.event_index

    assert advanced_count == len(events)
