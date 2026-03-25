"""Parity tests: Rust RustExitManager vs Python ExitManager.

Runs identical scenarios through both implementations and asserts matching outputs
for check_exit (should_exit + reason prefix) and allow_entry.
"""
from __future__ import annotations

import pytest
from alpha.v11_config import ExitConfig, TimeFilterConfig
from decision.exit_manager import ExitManager

try:
    from _quant_hotpath import RustExitManager
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust _quant_hotpath not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pair(
    trailing_stop_pct=0.0,
    reversal_threshold=-0.3,
    deadzone_fade=0.2,
    zscore_cap=0.0,
    time_filter_enabled=False,
    skip_hours=None,
    min_hold=12,
    max_hold=96,
):
    """Create matched Python ExitManager + Rust RustExitManager."""
    skip = skip_hours or []
    if time_filter_enabled:
        tf = TimeFilterConfig(enabled=True, skip_hours_utc=skip)
    else:
        tf = TimeFilterConfig()
    cfg = ExitConfig(
        trailing_stop_pct=trailing_stop_pct,
        reversal_threshold=reversal_threshold,
        deadzone_fade=deadzone_fade,
        zscore_cap=zscore_cap,
        time_filter=tf,
    )
    py = ExitManager(config=cfg, min_hold=min_hold, max_hold=max_hold)
    rs = RustExitManager(
        trailing_stop_pct=trailing_stop_pct,
        reversal_threshold=reversal_threshold,
        deadzone_fade=deadzone_fade,
        zscore_cap=zscore_cap,
        time_filter_enabled=time_filter_enabled,
        skip_hours=skip,
        min_hold=min_hold,
        max_hold=max_hold,
    )
    return py, rs


def _assert_check_exit_match(py_result, rs_result):
    """Assert check_exit outputs match (bool + reason prefix)."""
    py_exit, py_reason = py_result
    rs_exit, rs_reason = rs_result
    assert py_exit == rs_exit, f"exit mismatch: py={py_exit} rs={rs_exit}"
    # Compare reason prefix (before '=' sign) for matching exit type
    if py_exit:
        py_prefix = py_reason.split("=")[0] if "=" in py_reason else py_reason
        rs_prefix = rs_reason.split("=")[0] if "=" in rs_reason else rs_reason
        assert py_prefix == rs_prefix, (
            f"reason prefix mismatch: py='{py_reason}' rs='{rs_reason}'"
        )


# ---------------------------------------------------------------------------
# Tests mirroring test_exit_manager.py
# ---------------------------------------------------------------------------

class TestTrailingStopParity:
    def test_no_exit_within_threshold(self):
        py, rs = _make_pair(trailing_stop_pct=0.02)
        for mgr in (py, rs):
            mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=1.0)
            mgr.update_price("ETHUSDT", 2100.0)
        py_r = py.check_exit("ETHUSDT", 2080.0, bar=20, z_score=0.5, position=1.0)
        rs_r = rs.check_exit("ETHUSDT", 2080.0, bar=20, z_score=0.5, position=1.0)
        _assert_check_exit_match(py_r, rs_r)
        assert not py_r[0]

    def test_exit_on_drawdown(self):
        py, rs = _make_pair(trailing_stop_pct=0.02)
        for mgr in (py, rs):
            mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=1.0)
            mgr.update_price("ETHUSDT", 2100.0)
        py_r = py.check_exit("ETHUSDT", 2050.0, bar=20, z_score=0.5, position=1.0)
        rs_r = rs.check_exit("ETHUSDT", 2050.0, bar=20, z_score=0.5, position=1.0)
        _assert_check_exit_match(py_r, rs_r)
        assert py_r[0]
        assert "trailing_stop" in py_r[1]

    def test_short_trailing_stop(self):
        py, rs = _make_pair(trailing_stop_pct=0.02)
        for mgr in (py, rs):
            mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=-1.0)
            mgr.update_price("ETHUSDT", 1900.0)
        py_r = py.check_exit("ETHUSDT", 1940.0, bar=20, z_score=-0.5, position=-1.0)
        rs_r = rs.check_exit("ETHUSDT", 1940.0, bar=20, z_score=-0.5, position=-1.0)
        _assert_check_exit_match(py_r, rs_r)
        assert py_r[0]
        assert "trailing_stop" in py_r[1]

    def test_no_trailing_when_disabled(self):
        py, rs = _make_pair()  # trailing_stop_pct=0.0 by default
        for mgr in (py, rs):
            mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=1.0)
            mgr.update_price("ETHUSDT", 2100.0)
        py_r = py.check_exit("ETHUSDT", 1900.0, bar=20, z_score=0.5, position=1.0)
        rs_r = rs.check_exit("ETHUSDT", 1900.0, bar=20, z_score=0.5, position=1.0)
        _assert_check_exit_match(py_r, rs_r)
        # No trailing stop but might still exit due to other rules
        assert py_r[0] == rs_r[0]

    def test_min_hold_respected(self):
        py, rs = _make_pair(trailing_stop_pct=0.02)
        for mgr in (py, rs):
            mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=1.0)
            mgr.update_price("ETHUSDT", 2100.0)
        py_r = py.check_exit("ETHUSDT", 1800.0, bar=10, z_score=0.5, position=1.0)
        rs_r = rs.check_exit("ETHUSDT", 1800.0, bar=10, z_score=0.5, position=1.0)
        _assert_check_exit_match(py_r, rs_r)
        assert not py_r[0]


class TestMaxHoldParity:
    def test_max_hold_forces_exit(self):
        py, rs = _make_pair()
        for mgr in (py, rs):
            mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=1.0)
        py_r = py.check_exit("ETHUSDT", 2000.0, bar=97, z_score=1.0, position=1.0)
        rs_r = rs.check_exit("ETHUSDT", 2000.0, bar=97, z_score=1.0, position=1.0)
        _assert_check_exit_match(py_r, rs_r)
        assert py_r[0]
        assert "max_hold" in py_r[1]


class TestSignalExitsParity:
    def test_reversal_exit(self):
        py, rs = _make_pair()
        for mgr in (py, rs):
            mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=1.0)
        # Long position with z=-0.5: position*z = -0.5 < -0.3
        py_r = py.check_exit("ETHUSDT", 2000.0, bar=20, z_score=-0.5, position=1.0)
        rs_r = rs.check_exit("ETHUSDT", 2000.0, bar=20, z_score=-0.5, position=1.0)
        _assert_check_exit_match(py_r, rs_r)
        assert py_r[0]
        assert "reversal" in py_r[1]

    def test_deadzone_fade_exit(self):
        py, rs = _make_pair()
        for mgr in (py, rs):
            mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=1.0)
        # |z_score| = 0.1 < 0.2 deadzone_fade
        py_r = py.check_exit("ETHUSDT", 2000.0, bar=20, z_score=0.1, position=1.0)
        rs_r = rs.check_exit("ETHUSDT", 2000.0, bar=20, z_score=0.1, position=1.0)
        _assert_check_exit_match(py_r, rs_r)
        assert py_r[0]
        assert "deadzone_fade" in py_r[1]

    def test_no_exit_on_strong_signal(self):
        py, rs = _make_pair()
        for mgr in (py, rs):
            mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=1.0)
        py_r = py.check_exit("ETHUSDT", 2000.0, bar=20, z_score=0.8, position=1.0)
        rs_r = rs.check_exit("ETHUSDT", 2000.0, bar=20, z_score=0.8, position=1.0)
        _assert_check_exit_match(py_r, rs_r)
        assert not py_r[0]


class TestZScoreCapParity:
    def test_zcap_blocks_entry(self):
        py, rs = _make_pair(zscore_cap=4.0)
        assert py.allow_entry(z_score=5.0) == rs.allow_entry(z_score=5.0, hour_utc=None)
        assert not py.allow_entry(z_score=5.0)

    def test_zcap_allows_normal_entry(self):
        py, rs = _make_pair(zscore_cap=4.0)
        assert py.allow_entry(z_score=2.0) == rs.allow_entry(z_score=2.0, hour_utc=None)
        assert py.allow_entry(z_score=2.0)

    def test_zcap_disabled_allows_all(self):
        py, rs = _make_pair()  # zscore_cap=0.0
        assert py.allow_entry(z_score=100.0) == rs.allow_entry(z_score=100.0, hour_utc=None)
        assert py.allow_entry(z_score=100.0)


class TestTimeFilterParity:
    def test_blocks_skip_hours(self):
        py, rs = _make_pair(time_filter_enabled=True, skip_hours=[0, 1, 2, 3])
        for hour in [0, 1, 2, 3]:
            assert py.allow_entry(z_score=1.0, hour_utc=hour) == rs.allow_entry(z_score=1.0, hour_utc=hour)
            assert not py.allow_entry(z_score=1.0, hour_utc=hour)

    def test_allows_active_hours(self):
        py, rs = _make_pair(time_filter_enabled=True, skip_hours=[0, 1, 2, 3])
        for hour in [4, 12, 23]:
            assert py.allow_entry(z_score=1.0, hour_utc=hour) == rs.allow_entry(z_score=1.0, hour_utc=hour)
            assert py.allow_entry(z_score=1.0, hour_utc=hour)

    def test_none_hour_allowed(self):
        py, rs = _make_pair(time_filter_enabled=True, skip_hours=[0, 1, 2, 3])
        assert py.allow_entry(z_score=1.0, hour_utc=None) == rs.allow_entry(z_score=1.0, hour_utc=None)
        assert py.allow_entry(z_score=1.0, hour_utc=None)


class TestOnExitClearsParity:
    def test_on_exit_clears(self):
        py, rs = _make_pair(trailing_stop_pct=0.02)
        for mgr in (py, rs):
            mgr.on_entry("ETHUSDT", 2000.0, bar=1, direction=1.0)
            mgr.on_exit("ETHUSDT")
        py_r = py.check_exit("ETHUSDT", 1800.0, bar=100, z_score=-2.0, position=0.0)
        rs_r = rs.check_exit("ETHUSDT", 1800.0, bar=100, z_score=-2.0, position=0.0)
        _assert_check_exit_match(py_r, rs_r)
        assert not py_r[0]


class TestCheckpointParity:
    def test_checkpoint_roundtrip(self):
        """Python checkpoint -> Rust restore and vice versa."""
        py, rs = _make_pair(trailing_stop_pct=0.02)
        for mgr in (py, rs):
            mgr.on_entry("BTCUSDT", price=40000.0, bar=10, direction=1.0)
            mgr.update_price("BTCUSDT", 41000.0)
            mgr.on_entry("ETHUSDT", price=3000.0, bar=15, direction=-1.0)
            mgr.update_price("ETHUSDT", 2900.0)

        py_data = py.checkpoint()
        rs_data = rs.checkpoint()

        # Both checkpoints should have same keys and values
        assert set(py_data.keys()) == set(rs_data.keys())
        for sym in py_data:
            for key in ("entry_price", "peak_price", "entry_bar", "direction"):
                assert abs(py_data[sym][key] - rs_data[sym][key]) < 1e-9, (
                    f"{sym}.{key}: py={py_data[sym][key]} rs={rs_data[sym][key]}"
                )

    def test_python_checkpoint_to_rust_restore(self):
        """Checkpoint from Python, restore into Rust, verify exit checks match."""
        py1, _ = _make_pair(trailing_stop_pct=0.02)
        py1.on_entry("BTCUSDT", price=40000.0, bar=10, direction=1.0)
        py1.update_price("BTCUSDT", 41000.0)
        data = py1.checkpoint()

        # Restore into fresh Rust manager
        _, rs2 = _make_pair(trailing_stop_pct=0.02)
        rs2.restore(data)

        # Continue tracking — should produce same results
        py1.update_price("BTCUSDT", 42000.0)
        rs2.update_price("BTCUSDT", 42000.0)

        py_r = py1.check_exit("BTCUSDT", 41000.0, bar=30, z_score=0.5, position=1.0)
        rs_r = rs2.check_exit("BTCUSDT", 41000.0, bar=30, z_score=0.5, position=1.0)
        _assert_check_exit_match(py_r, rs_r)

    def test_rust_checkpoint_to_python_restore(self):
        """Checkpoint from Rust, restore into Python, verify exit checks match."""
        _, rs1 = _make_pair(trailing_stop_pct=0.02)
        rs1.on_entry("ETHUSDT", price=3000.0, bar=5, direction=-1.0)
        rs1.update_price("ETHUSDT", 2900.0)
        data = rs1.checkpoint()

        # Restore into fresh Python manager
        py2, _ = _make_pair(trailing_stop_pct=0.02)
        py2.restore(data)

        py_r = py2.check_exit("ETHUSDT", 2960.0, bar=20, z_score=-0.3, position=-1.0)
        rs_r = rs1.check_exit("ETHUSDT", 2960.0, bar=20, z_score=-0.3, position=-1.0)
        _assert_check_exit_match(py_r, rs_r)

    def test_checkpoint_empty(self):
        py, rs = _make_pair()
        assert py.checkpoint() == {}
        assert rs.checkpoint() == {}

    def test_restore_empty(self):
        py, rs = _make_pair()
        for mgr in (py, rs):
            mgr.on_entry("BTCUSDT", 40000.0, bar=1, direction=1.0)
            mgr.restore({})
        py_r = py.check_exit("BTCUSDT", 40000.0, bar=100, z_score=-2.0, position=1.0)
        rs_r = rs.check_exit("BTCUSDT", 40000.0, bar=100, z_score=-2.0, position=1.0)
        _assert_check_exit_match(py_r, rs_r)
        assert not py_r[0]  # position cleared by restore({})


class TestExitReasonFormatParity:
    """Verify that the reason string format matches exactly between implementations."""

    def test_trailing_stop_format(self):
        py, rs = _make_pair(trailing_stop_pct=0.02)
        for mgr in (py, rs):
            mgr.on_entry("ETH", 2000.0, bar=1, direction=1.0)
            mgr.update_price("ETH", 2100.0)
        py_r = py.check_exit("ETH", 2050.0, bar=20, z_score=0.5, position=1.0)
        rs_r = rs.check_exit("ETH", 2050.0, bar=20, z_score=0.5, position=1.0)
        # Both should have trailing_stop=0.0238 format
        assert py_r[1] == rs_r[1], f"py='{py_r[1]}' rs='{rs_r[1]}'"

    def test_max_hold_format(self):
        py, rs = _make_pair()
        for mgr in (py, rs):
            mgr.on_entry("ETH", 2000.0, bar=1, direction=1.0)
        py_r = py.check_exit("ETH", 2000.0, bar=97, z_score=1.0, position=1.0)
        rs_r = rs.check_exit("ETH", 2000.0, bar=97, z_score=1.0, position=1.0)
        assert py_r[1] == rs_r[1], f"py='{py_r[1]}' rs='{rs_r[1]}'"

    def test_reversal_format(self):
        py, rs = _make_pair()
        for mgr in (py, rs):
            mgr.on_entry("ETH", 2000.0, bar=1, direction=1.0)
        py_r = py.check_exit("ETH", 2000.0, bar=20, z_score=-0.5, position=1.0)
        rs_r = rs.check_exit("ETH", 2000.0, bar=20, z_score=-0.5, position=1.0)
        assert py_r[1] == rs_r[1], f"py='{py_r[1]}' rs='{rs_r[1]}'"

    def test_deadzone_fade_format(self):
        py, rs = _make_pair()
        for mgr in (py, rs):
            mgr.on_entry("ETH", 2000.0, bar=1, direction=1.0)
        py_r = py.check_exit("ETH", 2000.0, bar=20, z_score=0.1, position=1.0)
        rs_r = rs.check_exit("ETH", 2000.0, bar=20, z_score=0.1, position=1.0)
        assert py_r[1] == rs_r[1], f"py='{py_r[1]}' rs='{rs_r[1]}'"


class TestEdgeCases:
    def test_unknown_symbol_check_exit(self):
        py, rs = _make_pair()
        py_r = py.check_exit("UNKNOWN", 1000.0, bar=50, z_score=1.0, position=1.0)
        rs_r = rs.check_exit("UNKNOWN", 1000.0, bar=50, z_score=1.0, position=1.0)
        _assert_check_exit_match(py_r, rs_r)
        assert not py_r[0]

    def test_on_exit_nonexistent_symbol(self):
        """on_exit for unknown symbol should not raise."""
        py, rs = _make_pair()
        py.on_exit("UNKNOWN")
        rs.on_exit("UNKNOWN")

    def test_update_price_nonexistent_symbol(self):
        """update_price for unknown symbol should not raise."""
        py, rs = _make_pair()
        py.update_price("UNKNOWN", 1000.0)
        rs.update_price("UNKNOWN", 1000.0)

    def test_multiple_symbols(self):
        """Verify independent tracking of multiple symbols."""
        py, rs = _make_pair(trailing_stop_pct=0.02)
        for mgr in (py, rs):
            mgr.on_entry("BTCUSDT", 40000.0, bar=1, direction=1.0)
            mgr.on_entry("ETHUSDT", 3000.0, bar=1, direction=-1.0)
            mgr.update_price("BTCUSDT", 42000.0)
            mgr.update_price("ETHUSDT", 2800.0)

        # Check BTC exit
        py_r1 = py.check_exit("BTCUSDT", 41000.0, bar=20, z_score=0.5, position=1.0)
        rs_r1 = rs.check_exit("BTCUSDT", 41000.0, bar=20, z_score=0.5, position=1.0)
        _assert_check_exit_match(py_r1, rs_r1)

        # Check ETH exit
        py_r2 = py.check_exit("ETHUSDT", 2870.0, bar=20, z_score=-0.5, position=-1.0)
        rs_r2 = rs.check_exit("ETHUSDT", 2870.0, bar=20, z_score=-0.5, position=-1.0)
        _assert_check_exit_match(py_r2, rs_r2)

    def test_exact_min_hold_boundary(self):
        """At exactly min_hold, exits should be checked."""
        py, rs = _make_pair()
        for mgr in (py, rs):
            mgr.on_entry("ETH", 2000.0, bar=0, direction=1.0)
        # bar=12, entry_bar=0, held=12 == min_hold → should check exits
        py_r = py.check_exit("ETH", 2000.0, bar=12, z_score=0.1, position=1.0)
        rs_r = rs.check_exit("ETH", 2000.0, bar=12, z_score=0.1, position=1.0)
        _assert_check_exit_match(py_r, rs_r)

    def test_exact_max_hold_boundary(self):
        """At exactly max_hold, should force exit."""
        py, rs = _make_pair(max_hold=50)
        for mgr in (py, rs):
            mgr.on_entry("ETH", 2000.0, bar=0, direction=1.0)
        # bar=50, entry_bar=0, held=50 >= max_hold=50 → exit
        py_r = py.check_exit("ETH", 2000.0, bar=50, z_score=1.0, position=1.0)
        rs_r = rs.check_exit("ETH", 2000.0, bar=50, z_score=1.0, position=1.0)
        _assert_check_exit_match(py_r, rs_r)
        assert py_r[0]
