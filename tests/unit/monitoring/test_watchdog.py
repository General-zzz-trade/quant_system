"""Tests for monitoring.watchdog module."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# check_service
# ---------------------------------------------------------------------------

class TestCheckService:
    """Tests for check_service (systemd service status)."""

    @patch("monitoring.watchdog.subprocess")
    def test_active_service_healthy(self, mock_sp):
        """Active service with recent logs should be healthy."""
        from monitoring.watchdog import check_service

        # systemctl is-active returns "active"
        mock_run = MagicMock()
        mock_run.stdout = "active\n"
        mock_run.returncode = 0

        # journalctl returns recent lines with unix timestamp
        mock_journal = MagicMock()
        ts = str(time.time())
        mock_journal.stdout = f"{ts} some log line\n"
        mock_journal.returncode = 0

        # Error grep returns empty
        mock_err = MagicMock()
        mock_err.stdout = ""

        mock_sp.run.side_effect = [mock_run, mock_journal, mock_err]

        result = check_service("test-svc", {
            "unit": "test.service",
            "journal_match": "test",
            "max_silent_s": 120,
        })

        assert result["state"] == "active"
        assert result["status"] == "healthy"

    @patch("monitoring.watchdog.subprocess")
    def test_inactive_service(self, mock_sp):
        """Inactive service should report inactive status."""
        from monitoring.watchdog import check_service

        mock_run = MagicMock()
        mock_run.stdout = "inactive\n"
        mock_sp.run.return_value = mock_run

        result = check_service("stopped-svc", {
            "unit": "stopped.service",
            "journal_match": "stopped",
            "max_silent_s": 120,
        })

        assert result["state"] == "inactive"
        assert result["status"] == "inactive"

    @patch("monitoring.watchdog.subprocess")
    def test_stale_service_no_logs(self, mock_sp):
        """Active service with no recent logs should be stale."""
        from monitoring.watchdog import check_service

        mock_active = MagicMock()
        mock_active.stdout = "active\n"

        mock_journal = MagicMock()
        mock_journal.stdout = ""  # no recent lines

        mock_err = MagicMock()
        mock_err.stdout = ""

        mock_sp.run.side_effect = [mock_active, mock_journal, mock_err]

        result = check_service("quiet-svc", {
            "unit": "quiet.service",
            "journal_match": "quiet",
            "max_silent_s": 120,
        })

        assert result["status"] == "stale"
        assert "no_recent_logs" in result["problems"]


# ---------------------------------------------------------------------------
# check_data_freshness
# ---------------------------------------------------------------------------

class TestCheckDataFreshness:
    """Tests for check_data_freshness (file timestamp checks)."""

    @patch("monitoring.watchdog.DATA_FILES", {
        "test_data": ("nonexistent_file.csv", 48 * 3600),
    })
    def test_missing_file_detected(self):
        """Missing data file should report 'missing' status."""
        from monitoring.watchdog import check_data_freshness

        results = check_data_freshness()
        assert len(results) == 1
        assert results[0]["status"] == "missing"
        assert "file_not_found" in results[0]["problems"]

    @patch("monitoring.watchdog.DATA_FILES")
    def test_fresh_file_detected(self, mock_files, tmp_path: Path):
        """Recently modified file should report 'fresh' status."""
        from monitoring.watchdog import check_data_freshness

        # Create a fresh file
        fpath = tmp_path / "fresh_data.csv"
        fpath.write_text("open_time,close\n1,2\n")

        mock_files.__iter__ = lambda self: iter({"fresh": (str(fpath), 48 * 3600)}.items())
        mock_files.items = lambda: {"fresh": (str(fpath), 48 * 3600)}.items()

        results = check_data_freshness()
        assert len(results) == 1
        assert results[0]["status"] == "fresh"

    @patch("monitoring.watchdog.DATA_FILES")
    def test_stale_file_detected(self, mock_files, tmp_path: Path):
        """Old file should report 'stale' status."""
        from monitoring.watchdog import check_data_freshness

        fpath = tmp_path / "stale_data.csv"
        fpath.write_text("open_time,close\n1,2\n")
        # Set modification time to 72 hours ago
        old_time = time.time() - 72 * 3600
        os.utime(fpath, (old_time, old_time))

        mock_files.items = lambda: {"stale": (str(fpath), 48 * 3600)}.items()

        results = check_data_freshness()
        assert len(results) == 1
        assert results[0]["status"] == "stale"


# ---------------------------------------------------------------------------
# Alert triggering
# ---------------------------------------------------------------------------

class TestAlertTriggering:
    """Tests for alert logic in run_watchdog."""

    def test_status_changed_returns_true_on_first_run(self):
        """First run (no status file) should trigger alert."""
        from monitoring.watchdog import _status_changed

        result = _status_changed("warning")
        assert result is True

    @patch("monitoring.watchdog.STATUS_FILE", "/tmp/nonexistent_status_xyzzy.json")
    def test_status_changed_returns_true_when_no_file(self):
        """Should return True when status file doesn't exist."""
        from monitoring.watchdog import _status_changed

        result = _status_changed("critical")
        assert result is True

    def test_status_changed_detects_change(self, tmp_path: Path):
        """Should return True when status differs from previous run."""
        from monitoring import watchdog

        status_file = tmp_path / "health_status.json"
        status_file.write_text(json.dumps({"overall": "healthy"}))

        orig = watchdog.STATUS_FILE
        try:
            watchdog.STATUS_FILE = str(status_file)
            assert watchdog._status_changed("warning") is True
            assert watchdog._status_changed("healthy") is False
        finally:
            watchdog.STATUS_FILE = orig
