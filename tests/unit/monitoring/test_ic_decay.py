"""Tests for monitoring.ic_decay_monitor module."""
from __future__ import annotations

import json

import pandas as pd


# ---------------------------------------------------------------------------
# _classify_decay — IC threshold classification
# ---------------------------------------------------------------------------

class TestClassifyDecay:
    """Tests for _classify_decay: GREEN/YELLOW/RED classification."""

    def test_green_when_ic_above_50pct(self):
        """Rolling IC >= 50% of training IC should be GREEN."""
        from monitoring.ic_decay_monitor import _classify_decay

        assert _classify_decay(0.06, 0.10) == "GREEN"
        assert _classify_decay(0.10, 0.10) == "GREEN"

    def test_yellow_when_ic_25_to_50pct(self):
        """Rolling IC between 25-50% of training IC should be YELLOW."""
        from monitoring.ic_decay_monitor import _classify_decay

        assert _classify_decay(0.03, 0.10) == "YELLOW"
        assert _classify_decay(0.04, 0.10) == "YELLOW"

    def test_red_when_ic_below_25pct(self):
        """Rolling IC < 25% of training IC should be RED."""
        from monitoring.ic_decay_monitor import _classify_decay

        assert _classify_decay(0.02, 0.10) == "RED"
        assert _classify_decay(0.01, 0.10) == "RED"

    def test_red_when_negative_ic(self):
        """Negative rolling IC should always be RED."""
        from monitoring.ic_decay_monitor import _classify_decay

        assert _classify_decay(-0.05, 0.10) == "RED"

    def test_red_when_nan(self):
        """NaN IC should be RED."""
        from monitoring.ic_decay_monitor import _classify_decay

        assert _classify_decay(float("nan"), 0.10) == "RED"
        assert _classify_decay(0.05, float("nan")) == "RED"

    def test_red_when_training_ic_zero_or_negative(self):
        """Zero or negative training IC should be RED."""
        from monitoring.ic_decay_monitor import _classify_decay

        assert _classify_decay(0.05, 0.0) == "RED"
        assert _classify_decay(0.05, -0.01) == "RED"


# ---------------------------------------------------------------------------
# _resample_1h_to_4h
# ---------------------------------------------------------------------------

class TestResample1hTo4h:
    """Tests for _resample_1h_to_4h resampling."""

    def test_produces_correct_4h_bars(self):
        """8 aligned 1h bars should produce 2 x 4h bars."""
        from monitoring.ic_decay_monitor import _resample_1h_to_4h

        # Use a 4h-aligned epoch ms (2023-11-15 00:00 UTC)
        base_ms = 1700006400000
        rows = []
        for i in range(8):
            rows.append({
                "open_time": base_ms + i * 3600_000,
                "open": 100.0 + i,
                "high": 105.0 + i,
                "low": 95.0 + i,
                "close": 102.0 + i,
                "volume": 1000.0,
            })
        df = pd.DataFrame(rows)

        result = _resample_1h_to_4h(df)

        assert len(result) == 2
        # First 4h bar: open from bar 0, close from bar 3
        assert result.iloc[0]["open"] == 100.0
        assert result.iloc[0]["close"] == 105.0  # 102 + 3
        # High is max of first 4 bars
        assert result.iloc[0]["high"] == max(105.0, 106.0, 107.0, 108.0)
        # Volume is sum
        assert result.iloc[0]["volume"] == 4000.0

    def test_handles_extra_columns(self):
        """Should aggregate extra columns (quote_volume, trades) by sum."""
        from monitoring.ic_decay_monitor import _resample_1h_to_4h

        # Use 4h-aligned base (2023-11-15 00:00 UTC)
        base_ms = 1700006400000
        rows = []
        for i in range(4):
            rows.append({
                "open_time": base_ms + i * 3600_000,
                "open": 100.0, "high": 110.0, "low": 90.0,
                "close": 100.0, "volume": 500.0,
                "trades": 100,
            })
        df = pd.DataFrame(rows)

        result = _resample_1h_to_4h(df)
        assert len(result) >= 1
        assert result.iloc[0]["trades"] == 400


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    """Tests for save_results JSON output structure."""

    def test_save_results_creates_valid_json(self, tmp_path):
        """save_results should produce a valid JSON file with expected keys."""
        from monitoring.ic_decay_monitor import save_results
        import monitoring.ic_decay_monitor as mod

        orig_path = mod.OUTPUT_PATH
        try:
            mod.OUTPUT_PATH = tmp_path / "ic_health.json"

            results = [
                {
                    "model": "BTCUSDT_gate_v2",
                    "symbol": "BTCUSDT",
                    "interval": "1h",
                    "overall_status": "GREEN",
                    "horizons": [],
                }
            ]
            save_results(results)

            data = json.loads(mod.OUTPUT_PATH.read_text())
            assert "timestamp" in data
            assert "models" in data
            assert len(data["models"]) == 1
            assert data["models"][0]["overall_status"] == "GREEN"
        finally:
            mod.OUTPUT_PATH = orig_path


# ---------------------------------------------------------------------------
# Consecutive RED tracking and auto-retrain trigger
# ---------------------------------------------------------------------------

class TestRedHistory:
    """Tests for _update_red_history and _models_needing_retrain."""

    def _make_result(self, model: str, status: str) -> dict:
        return {"model": model, "overall_status": status, "symbol": "BTCUSDT"}

    def test_red_model_accumulates_history(self, tmp_path):
        """RED model should accumulate dates in history."""
        import monitoring.ic_decay_monitor as mod

        orig = mod.RED_HISTORY_PATH
        try:
            mod.RED_HISTORY_PATH = tmp_path / "red_hist.json"
            results = [self._make_result("BTCUSDT_gate_v2", "RED")]
            history = mod._update_red_history(results)
            assert "BTCUSDT_gate_v2" in history
            assert len(history["BTCUSDT_gate_v2"]) == 1
        finally:
            mod.RED_HISTORY_PATH = orig

    def test_green_clears_history(self, tmp_path):
        """GREEN result should clear that model's RED streak."""
        import monitoring.ic_decay_monitor as mod

        orig = mod.RED_HISTORY_PATH
        try:
            mod.RED_HISTORY_PATH = tmp_path / "red_hist.json"
            # Seed with RED history
            mod._save_red_history({"BTCUSDT_gate_v2": ["2026-03-01", "2026-03-02"]})

            results = [self._make_result("BTCUSDT_gate_v2", "GREEN")]
            history = mod._update_red_history(results)
            assert "BTCUSDT_gate_v2" not in history
        finally:
            mod.RED_HISTORY_PATH = orig

    def test_no_retrain_before_threshold(self, tmp_path):
        """Should not trigger retrain with fewer than 3 consecutive RED days."""
        import monitoring.ic_decay_monitor as mod

        orig = mod.RED_HISTORY_PATH
        try:
            mod.RED_HISTORY_PATH = tmp_path / "red_hist.json"
            # Only 2 consecutive days
            history = {"BTCUSDT_gate_v2": ["2026-03-24", "2026-03-25"]}
            results = [self._make_result("BTCUSDT_gate_v2", "RED")]
            candidates = mod._models_needing_retrain(results, history)
            assert candidates == []
        finally:
            mod.RED_HISTORY_PATH = orig

    def test_retrain_triggered_at_threshold(self, tmp_path):
        """Should trigger retrain after 3 consecutive RED days."""
        import monitoring.ic_decay_monitor as mod

        orig = mod.RED_HISTORY_PATH
        try:
            mod.RED_HISTORY_PATH = tmp_path / "red_hist.json"
            history = {"BTCUSDT_gate_v2": ["2026-03-24", "2026-03-25", "2026-03-26"]}
            results = [self._make_result("BTCUSDT_gate_v2", "RED")]
            candidates = mod._models_needing_retrain(results, history)
            assert candidates == ["BTCUSDT_gate_v2"]
        finally:
            mod.RED_HISTORY_PATH = orig

    def test_non_consecutive_dates_rejected(self, tmp_path):
        """Non-consecutive dates (gap > 1 day) should not trigger retrain."""
        import monitoring.ic_decay_monitor as mod

        orig = mod.RED_HISTORY_PATH
        try:
            mod.RED_HISTORY_PATH = tmp_path / "red_hist.json"
            # Gap between day 2 and day 4
            history = {"BTCUSDT_gate_v2": ["2026-03-22", "2026-03-24", "2026-03-26"]}
            results = [self._make_result("BTCUSDT_gate_v2", "RED")]
            candidates = mod._models_needing_retrain(results, history)
            assert candidates == []
        finally:
            mod.RED_HISTORY_PATH = orig

    def test_green_model_not_retrained(self, tmp_path):
        """GREEN model should never appear in retrain candidates."""
        import monitoring.ic_decay_monitor as mod

        orig = mod.RED_HISTORY_PATH
        try:
            mod.RED_HISTORY_PATH = tmp_path / "red_hist.json"
            history = {"BTCUSDT_gate_v2": ["2026-03-24", "2026-03-25", "2026-03-26"]}
            results = [self._make_result("BTCUSDT_gate_v2", "GREEN")]
            candidates = mod._models_needing_retrain(results, history)
            assert candidates == []
        finally:
            mod.RED_HISTORY_PATH = orig

    def test_cooldown_prevents_retrain(self, tmp_path):
        """48h cooldown should prevent re-triggering."""
        import monitoring.ic_decay_monitor as mod
        from datetime import datetime, timezone

        orig_hist = mod.RED_HISTORY_PATH
        orig_last = mod.LAST_IC_RETRAIN_PATH
        try:
            mod.RED_HISTORY_PATH = tmp_path / "red_hist.json"
            mod.LAST_IC_RETRAIN_PATH = tmp_path / "last_retrain.txt"

            # Write a recent timestamp (1 hour ago)
            recent_ts = datetime.now(timezone.utc).timestamp() - 3600
            mod.LAST_IC_RETRAIN_PATH.write_text(str(recent_ts))

            assert mod._should_retrain() is False
        finally:
            mod.RED_HISTORY_PATH = orig_hist
            mod.LAST_IC_RETRAIN_PATH = orig_last

    def test_cooldown_expired_allows_retrain(self, tmp_path):
        """Expired cooldown (>48h) should allow retrain."""
        import monitoring.ic_decay_monitor as mod
        from datetime import datetime, timezone

        orig_last = mod.LAST_IC_RETRAIN_PATH
        try:
            mod.LAST_IC_RETRAIN_PATH = tmp_path / "last_retrain.txt"

            # Write an old timestamp (72 hours ago)
            old_ts = datetime.now(timezone.utc).timestamp() - 72 * 3600
            mod.LAST_IC_RETRAIN_PATH.write_text(str(old_ts))

            assert mod._should_retrain() is True
        finally:
            mod.LAST_IC_RETRAIN_PATH = orig_last

    def test_model_to_symbol_mapping(self):
        """_model_to_symbol should map known model dirs to symbols."""
        from monitoring.ic_decay_monitor import _model_to_symbol

        assert _model_to_symbol("BTCUSDT_gate_v2") == "BTCUSDT"
        assert _model_to_symbol("ETHUSDT_gate_v2") == "ETHUSDT"
        assert _model_to_symbol("BTCUSDT_4h") == "BTCUSDT"
        assert _model_to_symbol("unknown_model") is None
