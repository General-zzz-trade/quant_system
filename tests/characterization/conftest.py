"""Characterization tests: they pin today's behaviour of the money path before it is moved.

A test marked ``known_defect("Rn")`` pins a defect listed in docs/arch/current-state.md section 5.
Its assertions change only in the migration step that implements the approved fix, and that step's
log entry names every assertion it changed.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BYBIT_BASE_URL", "BINANCE_TESTNET", "OKX_API_KEY",
                 "BINANCE_API_KEY", "BINANCE_API_SECRET", "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")
