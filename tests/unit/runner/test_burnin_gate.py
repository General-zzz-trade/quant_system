"""Tests for BurninGate pre-flight check."""
import json
import os
import tempfile

from runner.preflight import BurninGate


class TestBurninGate:
    def test_testnet_skips_gate(self):
        gate = BurninGate(report_path="/nonexistent")
        result = gate.check(testnet=True)
        assert result.passed is True

    def test_no_report_fails(self):
        gate = BurninGate(report_path="/nonexistent/burnin_report.json")
        result = gate.check(testnet=False)
        assert result.passed is False
        assert "No burn-in report" in result.message

    def test_all_phases_passed(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([
                {"phase": "A", "passed": True},
                {"phase": "B", "passed": True},
                {"phase": "C", "passed": True},
            ], f)
            path = f.name
        try:
            gate = BurninGate(report_path=path)
            result = gate.check(testnet=False)
            assert result.passed is True
        finally:
            os.unlink(path)

    def test_missing_phase_fails(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([
                {"phase": "A", "passed": True},
                {"phase": "B", "passed": True},
                # Phase C missing
            ], f)
            path = f.name
        try:
            gate = BurninGate(report_path=path)
            result = gate.check(testnet=False)
            assert result.passed is False
            assert "C" in result.message
        finally:
            os.unlink(path)

    def test_failed_phase_fails(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([
                {"phase": "A", "passed": True},
                {"phase": "B", "passed": False},
                {"phase": "C", "passed": True},
            ], f)
            path = f.name
        try:
            gate = BurninGate(report_path=path)
            result = gate.check(testnet=False)
            assert result.passed is False
        finally:
            os.unlink(path)
