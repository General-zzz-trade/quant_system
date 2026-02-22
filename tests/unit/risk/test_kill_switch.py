# tests/unit/risk/test_kill_switch.py
"""KillSwitch unit tests — covers scoping, modes, TTL, and order gating."""
from __future__ import annotations

import pytest
from risk.kill_switch import KillSwitch, KillScope, KillMode, KillRecord, KillSwitchError


@pytest.fixture
def ks() -> KillSwitch:
    return KillSwitch()


# ---------------------------------------------------------------------------
# trigger / clear basics
# ---------------------------------------------------------------------------

class TestTriggerClear:
    def test_trigger_returns_record(self, ks: KillSwitch) -> None:
        rec = ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT", reason="test")
        assert isinstance(rec, KillRecord)
        assert rec.scope == KillScope.SYMBOL
        assert rec.key == "BTCUSDT"
        assert rec.mode == KillMode.HARD_KILL  # default
        assert rec.reason == "test"

    def test_global_key_forced_to_star(self, ks: KillSwitch) -> None:
        rec = ks.trigger(scope=KillScope.GLOBAL, key="anything")
        assert rec.key == "*"

    def test_empty_key_raises(self, ks: KillSwitch) -> None:
        with pytest.raises(KillSwitchError, match="key"):
            ks.trigger(scope=KillScope.SYMBOL, key="")

    def test_clear_existing(self, ks: KillSwitch) -> None:
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT")
        assert ks.clear(scope=KillScope.SYMBOL, key="BTCUSDT") is True

    def test_clear_nonexistent(self, ks: KillSwitch) -> None:
        assert ks.clear(scope=KillScope.SYMBOL, key="BTCUSDT") is False

    def test_clear_all(self, ks: KillSwitch) -> None:
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT")
        ks.trigger(scope=KillScope.STRATEGY, key="strat_1")
        ks.clear_all()
        assert ks.active_records() == ()


# ---------------------------------------------------------------------------
# TTL / expiration
# ---------------------------------------------------------------------------

class TestTTL:
    def test_no_ttl_never_expires(self, ks: KillSwitch) -> None:
        rec = ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT", ttl_seconds=None, now_ts=1000.0)
        assert rec.is_expired(now_ts=999999.0) is False

    def test_ttl_expires(self, ks: KillSwitch) -> None:
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT", ttl_seconds=60, now_ts=1000.0)
        # Before expiry
        rec = ks.is_killed(symbol="BTCUSDT", now_ts=1050.0)
        assert rec is not None
        # After expiry
        rec = ks.is_killed(symbol="BTCUSDT", now_ts=1061.0)
        assert rec is None

    def test_expired_records_cleaned(self, ks: KillSwitch) -> None:
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT", ttl_seconds=10, now_ts=1000.0)
        active = ks.active_records(now_ts=1020.0)
        assert len(active) == 0


# ---------------------------------------------------------------------------
# Scope priority: GLOBAL > STRATEGY > SYMBOL
# ---------------------------------------------------------------------------

class TestScopePriority:
    def test_global_kills_all(self, ks: KillSwitch) -> None:
        ks.trigger(scope=KillScope.GLOBAL, key="*")
        assert ks.is_killed(symbol="BTCUSDT") is not None
        assert ks.is_killed(symbol="ETHUSDT", strategy_id="s1") is not None

    def test_strategy_kill_specific(self, ks: KillSwitch) -> None:
        ks.trigger(scope=KillScope.STRATEGY, key="strat_a")
        assert ks.is_killed(strategy_id="strat_a") is not None
        assert ks.is_killed(strategy_id="strat_b") is None

    def test_symbol_kill_specific(self, ks: KillSwitch) -> None:
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT")
        assert ks.is_killed(symbol="BTCUSDT") is not None
        assert ks.is_killed(symbol="ETHUSDT") is None

    def test_global_overrides_symbol(self, ks: KillSwitch) -> None:
        ks.trigger(scope=KillScope.GLOBAL, key="*", mode=KillMode.HARD_KILL)
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT", mode=KillMode.REDUCE_ONLY)
        rec = ks.is_killed(symbol="BTCUSDT")
        assert rec is not None
        assert rec.scope == KillScope.GLOBAL  # global wins

    def test_no_kill_returns_none(self, ks: KillSwitch) -> None:
        assert ks.is_killed(symbol="BTCUSDT") is None


# ---------------------------------------------------------------------------
# allow_order gate
# ---------------------------------------------------------------------------

class TestAllowOrder:
    def test_no_kill_allows(self, ks: KillSwitch) -> None:
        allowed, rec = ks.allow_order(symbol="BTCUSDT", strategy_id=None, reduce_only=False)
        assert allowed is True
        assert rec is None

    def test_hard_kill_blocks_all(self, ks: KillSwitch) -> None:
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT", mode=KillMode.HARD_KILL)
        allowed, rec = ks.allow_order(symbol="BTCUSDT", strategy_id=None, reduce_only=False)
        assert allowed is False
        # Even reduce_only is blocked by HARD_KILL
        allowed2, _ = ks.allow_order(symbol="BTCUSDT", strategy_id=None, reduce_only=True)
        assert allowed2 is False

    def test_reduce_only_allows_reduce(self, ks: KillSwitch) -> None:
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT", mode=KillMode.REDUCE_ONLY)
        allowed, rec = ks.allow_order(symbol="BTCUSDT", strategy_id=None, reduce_only=True)
        assert allowed is True
        assert rec is not None  # kill record is still returned for audit

    def test_reduce_only_blocks_new_orders(self, ks: KillSwitch) -> None:
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT", mode=KillMode.REDUCE_ONLY)
        allowed, rec = ks.allow_order(symbol="BTCUSDT", strategy_id=None, reduce_only=False)
        assert allowed is False

    def test_strategy_kill_blocks_strategy_orders(self, ks: KillSwitch) -> None:
        ks.trigger(scope=KillScope.STRATEGY, key="my_strat")
        allowed, _ = ks.allow_order(symbol="BTCUSDT", strategy_id="my_strat", reduce_only=False)
        assert allowed is False
        # Different strategy is fine
        allowed2, _ = ks.allow_order(symbol="BTCUSDT", strategy_id="other", reduce_only=False)
        assert allowed2 is True


# ---------------------------------------------------------------------------
# Overwrite behavior
# ---------------------------------------------------------------------------

class TestOverwrite:
    def test_trigger_overwrites_previous(self, ks: KillSwitch) -> None:
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT", mode=KillMode.HARD_KILL, reason="first")
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT", mode=KillMode.REDUCE_ONLY, reason="second")
        rec = ks.is_killed(symbol="BTCUSDT")
        assert rec is not None
        assert rec.mode == KillMode.REDUCE_ONLY
        assert rec.reason == "second"
