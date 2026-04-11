"""Unit tests for PortfolioRiskFileReader."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from runner.gates.portfolio_risk_file import PortfolioRiskFileReader


def _write_snapshot(path: Path, symbols: dict, ts: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "ts": ts if ts is not None else time.time(),
        "symbols": symbols,
    }))


@pytest.fixture
def tmp_snapshot(tmp_path: Path) -> Path:
    return tmp_path / "portfolio_risk.json"


# ── Fail-open paths ───────────────────────────────────────────────
class TestFailOpen:
    def test_missing_file_allows(self, tmp_snapshot: Path):
        reader = PortfolioRiskFileReader(path=tmp_snapshot)
        ok, reason = reader.check_would_exceed(
            symbol="BTCUSDT", venue="binance", side="buy",
            proposed_notional_usd=100.0,
        )
        assert ok
        assert reason == ""

    def test_stale_file_allows_with_warning(self, tmp_snapshot: Path):
        _write_snapshot(tmp_snapshot, {"BTCUSDT": {
            "total_long_notional_usd": 10_000,
            "total_short_notional_usd": 0,
            "max_total_notional_usd": 100,  # tiny cap, would block fresh
        }}, ts=time.time() - 1000.0)  # 1000s old
        reader = PortfolioRiskFileReader(path=tmp_snapshot, max_stale_seconds=60)
        ok, reason = reader.check_would_exceed(
            symbol="BTCUSDT", venue="binance", side="buy",
            proposed_notional_usd=100.0,
        )
        assert ok
        assert "stale" in reason

    def test_unknown_symbol_allows(self, tmp_snapshot: Path):
        _write_snapshot(tmp_snapshot, {"BTCUSDT": {
            "total_long_notional_usd": 0,
            "total_short_notional_usd": 0,
            "max_total_notional_usd": 1000,
        }})
        reader = PortfolioRiskFileReader(path=tmp_snapshot)
        ok, _ = reader.check_would_exceed(
            symbol="DOGEUSDT", venue="binance", side="buy",
            proposed_notional_usd=100.0,
        )
        assert ok


# ── Cap enforcement ───────────────────────────────────────────────
class TestCapEnforcement:
    def test_within_long_cap_allows(self, tmp_snapshot: Path):
        _write_snapshot(tmp_snapshot, {"BTCUSDT": {
            "total_long_notional_usd": 500,
            "total_short_notional_usd": 0,
            "max_total_notional_usd": 1000,
        }})
        reader = PortfolioRiskFileReader(path=tmp_snapshot)
        ok, _ = reader.check_would_exceed(
            symbol="BTCUSDT", venue="okx", side="buy",
            proposed_notional_usd=400.0,  # 500 + 400 = 900 < 1000
        )
        assert ok

    def test_exceeds_long_cap_blocks(self, tmp_snapshot: Path):
        _write_snapshot(tmp_snapshot, {"BTCUSDT": {
            "total_long_notional_usd": 500,
            "total_short_notional_usd": 0,
            "max_total_notional_usd": 1000,
        }})
        reader = PortfolioRiskFileReader(path=tmp_snapshot)
        ok, reason = reader.check_would_exceed(
            symbol="BTCUSDT", venue="okx", side="buy",
            proposed_notional_usd=600.0,  # 500 + 600 = 1100 > 1000
        )
        assert not ok
        assert "portfolio_cap" in reason
        assert "1100" in reason
        assert "1000" in reason

    def test_short_side_isolated_from_long(self, tmp_snapshot: Path):
        """A fully-used long budget must NOT block a short order."""
        _write_snapshot(tmp_snapshot, {"BTCUSDT": {
            "total_long_notional_usd": 1000,   # fully used long
            "total_short_notional_usd": 0,
            "max_total_notional_usd": 1000,
        }})
        reader = PortfolioRiskFileReader(path=tmp_snapshot)
        ok, _ = reader.check_would_exceed(
            symbol="BTCUSDT", venue="okx", side="sell",
            proposed_notional_usd=500.0,  # short leg, independent
        )
        assert ok  # short budget is clean

    def test_short_cap_enforced(self, tmp_snapshot: Path):
        _write_snapshot(tmp_snapshot, {"BTCUSDT": {
            "total_long_notional_usd": 0,
            "total_short_notional_usd": 800,
            "max_total_notional_usd": 1000,
        }})
        reader = PortfolioRiskFileReader(path=tmp_snapshot)
        ok, reason = reader.check_would_exceed(
            symbol="BTCUSDT", venue="okx", side="sell",
            proposed_notional_usd=300.0,  # 800 + 300 = 1100 > 1000
        )
        assert not ok
        assert "sell" in reason


# ── Caching + invalidation ────────────────────────────────────────
class TestCache:
    def test_mtime_invalidation(self, tmp_snapshot: Path):
        _write_snapshot(tmp_snapshot, {"BTCUSDT": {
            "total_long_notional_usd": 500,
            "total_short_notional_usd": 0,
            "max_total_notional_usd": 1000,
        }})
        reader = PortfolioRiskFileReader(path=tmp_snapshot)
        ok1, _ = reader.check_would_exceed(
            symbol="BTCUSDT", venue="okx", side="buy",
            proposed_notional_usd=400.0,
        )
        assert ok1

        # Rewrite with new values (different mtime)
        time.sleep(0.01)
        _write_snapshot(tmp_snapshot, {"BTCUSDT": {
            "total_long_notional_usd": 900,  # almost full
            "total_short_notional_usd": 0,
            "max_total_notional_usd": 1000,
        }})
        ok2, reason2 = reader.check_would_exceed(
            symbol="BTCUSDT", venue="okx", side="buy",
            proposed_notional_usd=200.0,  # 900 + 200 = 1100 > 1000
        )
        assert not ok2
        assert "portfolio_cap" in reason2
