"""Tests for monitoring.decision_audit module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# DecisionAuditLogger — file writing
# ---------------------------------------------------------------------------

class TestDecisionAuditLogger:
    """Tests for DecisionAuditLogger JSONL writing."""

    def test_writes_jsonl_to_file(self, tmp_path: Path):
        """Logger should write valid JSONL lines to the file."""
        from monitoring.decision_audit import DecisionAuditLogger

        audit_path = tmp_path / "audit.jsonl"
        logger = DecisionAuditLogger(path=audit_path)

        logger.log_signal(
            symbol="BTCUSDT", runner_key="BTCUSDT_1h",
            z_score=1.5, signal=1, confidence=0.8,
        )
        logger.close()

        lines = audit_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["type"] == "signal"
        assert record["symbol"] == "BTCUSDT"
        assert record["z_score"] == 1.5
        assert record["signal"] == 1

    def test_log_entry_produces_valid_json(self, tmp_path: Path):
        """log_entry should produce a valid JSON line with expected fields."""
        from monitoring.decision_audit import DecisionAuditLogger

        audit_path = tmp_path / "audit.jsonl"
        logger = DecisionAuditLogger(path=audit_path)

        logger.log_entry(
            symbol="ETHUSDT", side="Buy", qty=0.5, price=3000.0,
            reason="signal_long",
        )
        logger.close()

        record = json.loads(audit_path.read_text().strip())
        assert record["type"] == "entry"
        assert record["side"] == "Buy"
        assert record["qty"] == 0.5
        assert record["price"] == 3000.0
        assert "ts" in record

    def test_log_exit_produces_valid_json(self, tmp_path: Path):
        """log_exit should produce a valid JSON line with pnl field."""
        from monitoring.decision_audit import DecisionAuditLogger

        audit_path = tmp_path / "audit.jsonl"
        logger = DecisionAuditLogger(path=audit_path)

        logger.log_exit(
            symbol="BTCUSDT", side="Sell", qty=0.01, price=65000.0,
            reason="atr_stop", pnl=-50.0,
        )
        logger.close()

        record = json.loads(audit_path.read_text().strip())
        assert record["type"] == "exit"
        assert record["reason"] == "atr_stop"
        assert record["pnl"] == -50.0

    def test_multiple_events_appended(self, tmp_path: Path):
        """Multiple log calls should append lines, not overwrite."""
        from monitoring.decision_audit import DecisionAuditLogger

        audit_path = tmp_path / "audit.jsonl"
        logger = DecisionAuditLogger(path=audit_path)

        logger.log_signal(symbol="BTCUSDT", runner_key="k", z_score=0.5, signal=0)
        logger.log_entry(symbol="BTCUSDT", side="Buy", qty=0.01, price=60000, reason="test")
        logger.log_exit(symbol="BTCUSDT", side="Sell", qty=0.01, price=61000, reason="tp", pnl=10)
        logger.close()

        lines = audit_path.read_text().strip().split("\n")
        assert len(lines) == 3
        types = [json.loads(ln)["type"] for ln in lines]
        assert types == ["signal", "entry", "exit"]

    def test_never_raises_on_write_failure(self, tmp_path: Path):
        """Logger should silently handle write failures in _write."""
        from monitoring.decision_audit import DecisionAuditLogger

        audit_path = tmp_path / "audit.jsonl"
        logger = DecisionAuditLogger(path=audit_path)

        # Write one record successfully
        logger.log_signal(symbol="X", runner_key="k", z_score=0, signal=0)

        # Now close the underlying file and make it unwritable
        logger.close()
        audit_path.chmod(0o000)

        # _write should swallow the error, not raise
        logger.log_signal(symbol="Y", runner_key="k", z_score=1, signal=1)

        # Restore permissions for cleanup
        audit_path.chmod(0o644)


# ---------------------------------------------------------------------------
# _top_features
# ---------------------------------------------------------------------------

class TestTopFeatures:
    """Tests for _top_features helper."""

    def test_extracts_top_n_by_abs_value(self):
        """Should return top N features sorted by absolute value."""
        from monitoring.decision_audit import _top_features

        features = {
            "rsi_14": 0.3,
            "macd": -0.8,
            "volume_ratio": 0.1,
            "atr": 0.5,
            "spread": -0.9,
            "oi_change": 0.2,
        }

        result = _top_features(features, n=3)
        assert result is not None
        assert len(result) == 3
        keys = list(result.keys())
        assert keys[0] == "spread"   # abs(-0.9) = 0.9
        assert keys[1] == "macd"     # abs(-0.8) = 0.8
        assert keys[2] == "atr"      # abs(0.5) = 0.5

    def test_returns_none_for_empty_features(self):
        """Should return None when features is None or empty."""
        from monitoring.decision_audit import _top_features

        assert _top_features(None) is None
        assert _top_features({}) is None

    def test_handles_non_numeric_values(self):
        """Should handle features with non-numeric values gracefully."""
        from monitoring.decision_audit import _top_features

        features = {"feat_a": 0.5, "feat_b": "N/A", "feat_c": -0.3}
        result = _top_features(features, n=5)
        assert result is not None
        # feat_a (0.5) should sort before feat_c (0.3) and "N/A" (0)
        keys = list(result.keys())
        assert keys[0] == "feat_a"
