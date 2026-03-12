"""Tests for ExitManager — trailing stop, z-cap, time filter, signal exits."""
from __future__ import annotations

import pytest
from alpha.v11_config import ExitConfig, TimeFilterConfig
from decision.exit_manager import ExitManager


@pytest.fixture
def default_exit_mgr():
    """ExitManager with v10 defaults (no trailing stop, no z-cap)."""
    return ExitManager(config=ExitConfig(), min_hold=12, max_hold=96)


@pytest.fixture
def trailing_exit_mgr():
    """ExitManager with trailing stop enabled."""
    cfg = ExitConfig(trailing_stop_pct=0.02)
    return ExitManager(config=cfg, min_hold=12, max_hold=96)


@pytest.fixture
def zcap_exit_mgr():
    """ExitManager with z-score cap enabled."""
    cfg = ExitConfig(zscore_cap=4.0)
    return ExitManager(config=cfg, min_hold=12, max_hold=96)


@pytest.fixture
def time_filter_exit_mgr():
    """ExitManager with time filter enabled."""
    tf = TimeFilterConfig(enabled=True, skip_hours_utc=[0, 1, 2, 3])
    cfg = ExitConfig(time_filter=tf)
    return ExitManager(config=cfg, min_hold=12, max_hold=96)


class TestTrailingStop:
    def test_no_exit_within_threshold(self, trailing_exit_mgr):
        mgr = trailing_exit_mgr
        mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=1.0)
        # Price goes up then drops 1% — within 2% threshold
        mgr.update_price("ETHUSDT", 2100.0)
        should_exit, reason = mgr.check_exit("ETHUSDT", 2080.0, bar=20, z_score=0.5, position=1.0)
        assert not should_exit

    def test_exit_on_drawdown(self, trailing_exit_mgr):
        mgr = trailing_exit_mgr
        mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=1.0)
        mgr.update_price("ETHUSDT", 2100.0)  # peak
        # Drop > 2% from peak
        should_exit, reason = mgr.check_exit("ETHUSDT", 2050.0, bar=20, z_score=0.5, position=1.0)
        assert should_exit
        assert "trailing_stop" in reason

    def test_short_trailing_stop(self, trailing_exit_mgr):
        mgr = trailing_exit_mgr
        mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=-1.0)
        mgr.update_price("ETHUSDT", 1900.0)  # peak for short (lower = better)
        # Price rises > 2% from trough
        should_exit, reason = mgr.check_exit("ETHUSDT", 1940.0, bar=20, z_score=-0.5, position=-1.0)
        assert should_exit
        assert "trailing_stop" in reason

    def test_no_trailing_when_disabled(self, default_exit_mgr):
        mgr = default_exit_mgr
        mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=1.0)
        mgr.update_price("ETHUSDT", 2100.0)
        # Even big drawdown doesn't trigger trailing stop when disabled
        should_exit, _ = mgr.check_exit("ETHUSDT", 1900.0, bar=20, z_score=0.5, position=1.0)
        assert not should_exit

    def test_min_hold_respected(self, trailing_exit_mgr):
        mgr = trailing_exit_mgr
        mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=1.0)
        mgr.update_price("ETHUSDT", 2100.0)
        # Big drawdown but within min_hold
        should_exit, _ = mgr.check_exit("ETHUSDT", 1800.0, bar=10, z_score=0.5, position=1.0)
        assert not should_exit


class TestMaxHold:
    def test_max_hold_forces_exit(self, default_exit_mgr):
        mgr = default_exit_mgr
        mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=1.0)
        should_exit, reason = mgr.check_exit("ETHUSDT", 2000.0, bar=97, z_score=1.0, position=1.0)
        assert should_exit
        assert "max_hold" in reason


class TestSignalExits:
    def test_reversal_exit(self, default_exit_mgr):
        mgr = default_exit_mgr
        mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=1.0)
        # Long position with negative z-score (reversal)
        should_exit, reason = mgr.check_exit("ETHUSDT", 2000.0, bar=20, z_score=-0.5, position=1.0)
        assert should_exit
        assert "reversal" in reason

    def test_deadzone_fade_exit(self, default_exit_mgr):
        mgr = default_exit_mgr
        mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=1.0)
        # Z-score too weak to hold
        should_exit, reason = mgr.check_exit("ETHUSDT", 2000.0, bar=20, z_score=0.1, position=1.0)
        assert should_exit
        assert "deadzone_fade" in reason

    def test_no_exit_on_strong_signal(self, default_exit_mgr):
        mgr = default_exit_mgr
        mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=1.0)
        should_exit, _ = mgr.check_exit("ETHUSDT", 2000.0, bar=20, z_score=0.8, position=1.0)
        assert not should_exit


class TestZScoreCap:
    def test_zcap_blocks_entry(self, zcap_exit_mgr):
        mgr = zcap_exit_mgr
        assert not mgr.allow_entry(z_score=5.0)

    def test_zcap_allows_normal_entry(self, zcap_exit_mgr):
        mgr = zcap_exit_mgr
        assert mgr.allow_entry(z_score=2.0)

    def test_zcap_disabled_allows_all(self, default_exit_mgr):
        mgr = default_exit_mgr
        assert mgr.allow_entry(z_score=100.0)


class TestTimeFilter:
    def test_blocks_skip_hours(self, time_filter_exit_mgr):
        mgr = time_filter_exit_mgr
        assert not mgr.allow_entry(z_score=1.0, hour_utc=0)
        assert not mgr.allow_entry(z_score=1.0, hour_utc=3)

    def test_allows_active_hours(self, time_filter_exit_mgr):
        mgr = time_filter_exit_mgr
        assert mgr.allow_entry(z_score=1.0, hour_utc=4)
        assert mgr.allow_entry(z_score=1.0, hour_utc=12)

    def test_none_hour_allowed(self, time_filter_exit_mgr):
        mgr = time_filter_exit_mgr
        assert mgr.allow_entry(z_score=1.0, hour_utc=None)


class TestOnExitClearsState:
    def test_on_exit_clears(self, trailing_exit_mgr):
        mgr = trailing_exit_mgr
        mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=1.0)
        mgr.on_exit("ETHUSDT")
        # After exit, check_exit returns False (no position tracked)
        should_exit, _ = mgr.check_exit("ETHUSDT", 1800.0, bar=100, z_score=-2.0, position=0.0)
        assert not should_exit
