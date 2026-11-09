"""Native alpha data/signal acceptance with explicit synthetic dependencies."""
import json
import socket
import subprocess
import sys
from unittest.mock import patch

import pytest

from runner.paper_check import check_paper_contract


def test_real_builder_and_native_data_signal_contract():
    result = check_paper_contract()
    assert result["status"] == "passed"
    assert result["historical_bars"] == 8
    assert result["live_bars"] == 1
    assert result["model_evaluations"] >= 9
    assert result["signal_evaluations"] >= 1
    assert result["stale_data_blocked"] and result["failed_refresh_blocked"]
    assert result["venue_order_calls"] == 0
    assert not result["online_feed_verified"]
    assert not result["trained_models_verified"]
    assert not result["account_verified"]
    assert not result["exchange_settlement_verified"]


def test_stale_fixture_cannot_report_paper_contract_success():
    with pytest.raises(RuntimeError, match="live fixture data is not ready"):
        check_paper_contract(inject_stale_data=True)


def test_synthetic_model_cannot_escape_network_guard():
    from runner.paper_check import _SyntheticModel

    def accidental_network(self, rows):
        socket.create_connection(("example.invalid", 443))

    with patch.object(_SyntheticModel, "predict", accidental_network):
        with pytest.raises(Exception, match="Decision module failed|network is forbidden"):
            check_paper_contract()


def test_canonical_entry_exposes_separate_offline_paper_contract():
    result = subprocess.run(
        [sys.executable, "-B", "-m", "runner.alpha_main", "--check-paper-contract"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["scope"] == "offline-paper-data-signal-contract"
    assert data["venue_order_calls"] == 0
