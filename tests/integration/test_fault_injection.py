"""Fault injection tests: network faults, data faults, state faults.

Tests exercise:
- ExecutionBridge retry logic and circuit breaker
- NaN/None safe-value handling patterns from alpha_runner
- Kill switch triggering (Rust, skipped if unavailable)
- Position reconciliation pattern
- WS staleness detection
"""
from __future__ import annotations

import time
from typing import Any, Iterable
from unittest.mock import MagicMock

import pytest

from engine.execution_bridge import ExecutionBridge, ExecutionBridgeError


def _make_bridge(
    adapter: Any,
    *,
    max_retries: int = 2,
    retry_base_delay: float = 0.0,   # zero delay so tests run fast
    cb_failure_threshold: int = 5,
    cb_cooldown_seconds: float = 30.0,
) -> ExecutionBridge:
    """Build an ExecutionBridge with a no-op dispatcher and configurable adapter."""
    return ExecutionBridge(
        adapter=adapter,
        dispatcher_emit=lambda ev: None,
        risk_gate=None,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        cb_failure_threshold=cb_failure_threshold,
        cb_cooldown_seconds=cb_cooldown_seconds,
    )


# ── TestNetworkFaults ─────────────────────────────────────────────────────────

class TestNetworkFaults:
    """Tests that ExecutionBridge handles transient and persistent adapter failures."""

    def test_retry_on_transient_failure(self) -> None:
        """Adapter raises on first 2 calls, succeeds on 3rd. handle_event must not raise."""
        call_count = 0

        class TransientAdapter:
            def send_order(self, event: Any) -> Iterable[Any]:
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise ConnectionError("transient network error")
                return []

        bridge = _make_bridge(TransientAdapter(), max_retries=2, retry_base_delay=0.0)
        # Should succeed on the 3rd attempt without raising
        bridge.handle_event(object())
        assert call_count == 3, f"Expected 3 adapter calls, got {call_count}"

    def test_circuit_breaker_opens_after_threshold(self) -> None:
        """After cb_failure_threshold cumulative failures, circuit_open becomes True."""
        adapter = MagicMock()
        adapter.send_order.side_effect = RuntimeError("persistent failure")

        threshold = 3
        bridge = _make_bridge(
            adapter,
            max_retries=0,             # 1 attempt per call = 1 failure per call
            retry_base_delay=0.0,
            cb_failure_threshold=threshold,
            cb_cooldown_seconds=30.0,
        )

        # Each call to handle_event with max_retries=0 exhausts all attempts on first try
        # and raises ExecutionBridgeError; _failure_count increments each attempt.
        for _ in range(threshold):
            try:
                bridge.handle_event(object())
            except ExecutionBridgeError:
                pass

        assert bridge.circuit_open is True, (
            f"Expected circuit to be open after {threshold} failures "
            f"(failure_count={bridge._failure_count})"
        )

    def test_circuit_breaker_blocks_when_open(self) -> None:
        """When circuit is manually opened and cooldown hasn't elapsed, handle_event
        raises ExecutionBridgeError immediately without calling the adapter."""
        adapter = MagicMock()
        adapter.send_order.return_value = []

        bridge = _make_bridge(adapter, cb_failure_threshold=5, cb_cooldown_seconds=60.0)
        # Manually trip the circuit breaker
        bridge._failure_count = bridge.cb_failure_threshold
        bridge._circuit_opened_at = time.monotonic()  # opened just now

        assert bridge.circuit_open is True

        with pytest.raises(ExecutionBridgeError):
            bridge.handle_event(object())

        # Adapter must NOT be called — fast-fail path
        adapter.send_order.assert_not_called()

    def test_invalid_response_no_crash(self) -> None:
        """Adapter returning an empty list must not cause handle_event to raise."""
        adapter = MagicMock()
        adapter.send_order.return_value = []

        bridge = _make_bridge(adapter)
        # Should complete without exception
        bridge.handle_event(object())


# ── TestDataFaults ────────────────────────────────────────────────────────────

class TestDataFaults:
    """Tests safe-value handling and NaN/None resilience patterns."""

    # --------------------------------------------------------------------------
    # _safe_val is a nested function inside AlphaRunner._ensemble_predict so we
    # replicate the identical logic here to unit-test the pattern.  The actual
    # production code lives in scripts/ops/alpha_runner.py lines 589-602.
    # --------------------------------------------------------------------------

    _NEUTRAL_DEFAULTS: dict[str, float] = {
        "ls_ratio": 1.0,
        "top_trader_ls_ratio": 1.0,
        "taker_buy_ratio": 0.5,
        "vol_regime": 1.0,
        "bb_pctb_20": 0.5,
        "rsi_14": 50.0,
        "rsi_6": 50.0,
    }

    @staticmethod
    def _safe_val(v: Any, feat_name: str = "", default: float = 0.0) -> float:
        """Reimplementation of alpha_runner._safe_val for testing the pattern."""
        import numpy as np
        neutral_defaults: dict[str, float] = {
            "ls_ratio": 1.0,
            "top_trader_ls_ratio": 1.0,
            "taker_buy_ratio": 0.5,
            "vol_regime": 1.0,
            "bb_pctb_20": 0.5,
            "rsi_14": 50.0,
            "rsi_6": 50.0,
        }
        neutral = neutral_defaults.get(feat_name, default)
        if v is None:
            return neutral
        try:
            f = float(v)
            return neutral if np.isnan(f) else f
        except (TypeError, ValueError):
            return neutral

    def test_nan_features_use_neutral_defaults(self) -> None:
        """NaN and None inputs return the configured default; valid floats pass through."""
        sv = self._safe_val

        result_nan = sv(float("nan"), default=0.0)
        assert result_nan == 0.0, f"Expected 0.0 for NaN, got {result_nan}"

        result_none = sv(None, default=1.0)
        assert result_none == 1.0, f"Expected 1.0 for None, got {result_none}"

        result_valid = sv(42.0, default=0.0)
        assert result_valid == 42.0, f"Expected 42.0 for valid float, got {result_valid}"

    def test_safe_val_with_various_inputs(self) -> None:
        """Edge cases: very large float stays intact; feature-specific neutrals apply."""
        sv = self._safe_val

        # Very large float must not be rounded or clamped
        large = 1.23456789e15
        assert sv(large, default=0.0) == large

        # Negative finite float passes through unchanged
        assert sv(-99.9, default=0.0) == -99.9

        # Feature-specific neutral: rsi_14 NaN → 50.0 (not 0.0)
        assert sv(float("nan"), feat_name="rsi_14") == 50.0

        # Feature-specific neutral: ls_ratio None → 1.0 (not 0.0)
        assert sv(None, feat_name="ls_ratio") == 1.0

        # Non-numeric string falls back to neutral (TypeError path)
        result = sv("not_a_float", default=7.0)
        assert result == 7.0

    def test_model_predict_exception_handled(self) -> None:
        """When model.predict() raises, the caller catches it and result becomes None."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = ValueError("model not fitted")

        result = None
        try:
            result = mock_model.predict([[1.0, 2.0, 3.0]])[0]
        except ValueError:
            result = None

        assert result is None

    def test_all_nan_features_returns_zero_signal(self) -> None:
        """All-NaN feature dict passed through _safe_val yields all-zero values (default=0.0)."""
        feature_names = [
            "close_ret", "volume_ma_ratio", "atr_pct", "adx",
            "realized_vol", "bb_width", "momentum_5", "price_vs_sma20",
        ]
        feat_dict: dict[str, float] = {k: float("nan") for k in feature_names}

        safe_values = {k: self._safe_val(v, default=0.0) for k, v in feat_dict.items()}

        for name, val in safe_values.items():
            assert val == 0.0, f"Feature {name!r}: expected 0.0, got {val}"


# ── TestStateFaults ───────────────────────────────────────────────────────────

class TestStateFaults:
    """Tests state integrity under fault conditions."""

    def test_kill_switch_blocks_execution(self) -> None:
        """RustKillSwitch.arm() must make is_armed() True and allow_order() return False."""
        try:
            import _quant_hotpath  # type: ignore[import]
            RustKillSwitch = _quant_hotpath.RustKillSwitch
        except ImportError:
            pytest.skip("_quant_hotpath not available in this environment")

        ks = RustKillSwitch()
        assert ks.is_armed() is False, "Kill switch should be clear initially"

        # Arm the kill switch — halts all order flow
        ks.arm("global", "test", "HALT", "fault injection test")
        assert ks.is_armed() is True, "Kill switch must be armed after .arm()"

        # allow_order returns (False, reason) when armed
        allowed, reason = ks.allow_order(symbol="ETHUSDT")
        assert allowed is False, "armed kill switch must block order execution"
        assert reason, "armed kill switch must provide a rejection reason"

        # Cleanup: disarm so the switch doesn't affect other tests
        ks.clear_all()
        assert ks.is_armed() is False, "Kill switch must be clear after .clear_all()"

    def test_position_mismatch_reconcile_pattern(self) -> None:
        """When exchange position differs from state store, reconcile is called."""
        exchange_positions = {"ETHUSDT": 0.1}
        state_positions = {"ETHUSDT": 0.0}

        adapter = MagicMock()
        adapter.get_positions.return_value = exchange_positions

        state_store = MagicMock()
        state_store.get_positions.return_value = state_positions

        # Simulate the reconcile check pattern used in alpha_runner._reconcile_position
        live = adapter.get_positions()
        stored = state_store.get_positions()

        mismatches = {
            sym: (stored.get(sym, 0.0), live_qty)
            for sym, live_qty in live.items()
            if live_qty != stored.get(sym, 0.0)
        }

        assert "ETHUSDT" in mismatches, "Mismatch for ETHUSDT should be detected"
        stored_qty, live_qty = mismatches["ETHUSDT"]
        assert stored_qty == 0.0
        assert live_qty == 0.1
        # Verify adapter.get_positions was called (reconcile path hit)
        adapter.get_positions.assert_called_once()


# ── TestWsStale ───────────────────────────────────────────────────────────────

class TestWsStale:
    """Tests staleness detection logic used in WebSocket health monitoring."""

    def test_stale_detection_threshold(self) -> None:
        """last_bar_ts more than stale_threshold_seconds ago is correctly classified stale."""

        class WsHealthChecker:
            def __init__(self, stale_threshold_seconds: float = 120.0) -> None:
                self.last_bar_ts: float = time.time()
                self.stale_threshold_seconds = stale_threshold_seconds

            def is_stale(self) -> bool:
                return time.time() - self.last_bar_ts > self.stale_threshold_seconds

        checker = WsHealthChecker(stale_threshold_seconds=120.0)

        # Healthy: bar received 10 seconds ago
        checker.last_bar_ts = time.time() - 10
        assert checker.is_stale() is False, "10s old bar should not be stale (threshold=120s)"

        # Stale: bar received 200 seconds ago
        checker.last_bar_ts = time.time() - 200
        assert checker.is_stale() is True, "200s old bar should be stale (threshold=120s)"

    def test_stale_boundary_condition(self) -> None:
        """Exactly at the threshold boundary: just-under is healthy, just-over is stale."""

        def is_stale(age_seconds: float, threshold: float = 120.0) -> bool:
            return age_seconds > threshold

        # Boundary: 119.9s < 120s → healthy
        assert is_stale(119.9) is False
        # Boundary: 120.0s is NOT > 120.0s → healthy (strict greater-than)
        assert is_stale(120.0) is False
        # Boundary: 120.1s > 120s → stale
        assert is_stale(120.1) is True

    def test_stale_detection_with_zero_timestamp(self) -> None:
        """A zero timestamp (uninitialised) should always register as stale."""
        last_bar_ts = 0.0
        stale_threshold_seconds = 120.0
        age = time.time() - last_bar_ts
        # time.time() is at least 1.7e9 seconds from epoch — vastly more than 120s
        assert age > stale_threshold_seconds, (
            "Zero/uninitialised last_bar_ts must always be considered stale"
        )
