"""The hard leverage cap is 10 (owner 2026-11-17: "改为资金的10倍，我要上调", then "放宽到 10 倍" after the 2-10x study,
research/results/overlay_pilot_leverage_2026-11-17/). Only the ceiling moves: what applies is still what a person
writes, and every default stays where it was.

- ``MAX_TOTAL_LEVERAGE`` (trading/domain/risk/exposure.py): the most ``STRATEGY_LEVERAGE`` and the account exposure
  cap (``BINANCE_MAX_LEVERAGE``) may be; it was 3.
- Unset, the account cap is still 3 (``DEFAULT_ACCOUNT_LEVERAGE``) and the strategy leverage 1; a gate built without a
  cap gets 3, not the ceiling.
- ``STRATEGY_LEVERAGE`` still may not exceed the exchange leverage setting, whose own ceiling stays 10, and the
  strategy loss stop's ceiling stays 15 % (owner, same day).
"""
from __future__ import annotations

from decimal import Decimal as D

import pandas as pd
import pytest

from lab import um_lab as lab
from tests.unit.lab.test_um_lab import snapshot  # noqa: F401  (the synthetic snapshot fixture)
from tests.unit.trading.test_pilot_settings import _load
from tests.characterization.test_runtime_decisions import _start
from tests.unit.trading.test_runtime_wiring_limits import _offline_env  # noqa: F401  (autouse fixture)
from trading.adapters.binance.adapter import MAX_LEVERAGE
from trading.application.risk.exposure_gate import AccountExposureGate
from trading.domain.risk.exposure import DEFAULT_ACCOUNT_LEVERAGE, MAX_TOTAL_LEVERAGE, validated_leverage
from trading.infrastructure.config import STRATEGY_LOSS_STOP_MAX_PCT, risk_limits


class _Account:
    def get_account_equity(self):
        return "150"

    def get_positions(self, strict=False):
        return []

    def get_ticker(self, symbol):
        return {"lastPrice": "100"}


def test_the_ceiling_is_ten_and_the_defaults_stay():
    assert MAX_TOTAL_LEVERAGE == D(10) and DEFAULT_ACCOUNT_LEVERAGE == D(3)
    assert risk_limits({}).max_leverage == D(3)
    assert AccountExposureGate(_Account()).max_leverage == D(3)
    assert MAX_LEVERAGE == 10 and STRATEGY_LOSS_STOP_MAX_PCT == D(15)         # not raised (owner 2026-11-17)


@pytest.mark.parametrize("value, ok", [("10", True), ("9.5", True), ("10.01", False), ("11", False), ("0", False)])
def test_an_account_cap_up_to_ten_is_taken(value, ok):
    if ok:
        assert validated_leverage(value) == D(value)
    else:
        with pytest.raises(ValueError, match="max_leverage"):
            validated_leverage(value)


def test_a_gate_at_ten_lets_an_entry_reach_ten_times_the_equity():
    gate = AccountExposureGate(_Account(), max_leverage=D(10))     # (allowed, reason)
    assert gate.check_would_exceed(symbol="BTCUSDT", venue="binance", side="buy",
                                   proposed_notional_usd=1500.0) == (True, "")
    allowed, reason = gate.check_would_exceed(symbol="BTCUSDT", venue="binance", side="buy",
                                              proposed_notional_usd=1501.0)
    assert not allowed and "10x equity" in reason


@pytest.mark.parametrize("raw", ["4", "7.5", "10"])
def test_a_strategy_leverage_up_to_the_exchange_setting_is_taken(raw):
    assert _load(STRATEGY_LEVERAGE=raw).strategy_leverage == D(raw)


def test_the_strategy_leverage_still_stays_within_the_exchange_setting():
    with pytest.raises(RuntimeError, match="BINANCE_EXCHANGE_LEVERAGE"):
        _load(STRATEGY_LEVERAGE="7", BINANCE_EXCHANGE_LEVERAGE="5")


@pytest.mark.parametrize("setting, applied", [(None, D(3)), ("10", D(10)), ("20", D(10))])
def test_the_runtime_s_gate_gets_the_person_s_cap_within_the_ceiling(tmp_path, monkeypatch, setting, applied):
    if setting is not None:
        monkeypatch.setenv("BINANCE_MAX_LEVERAGE", setting)
    _, runtime = _start(tmp_path)
    assert runtime.sessions["BTCUSDT"].gate.inner._portfolio_reader.max_leverage == applied


def test_the_lab_replays_up_to_the_ceiling_and_within_the_stop_s_exchange_leverage(snapshot):  # noqa: F811
    df = lab.frame("BTCUSDT", snapshot)
    weights = pd.Series(0.5, index=df.index)
    out = lab.replay("BTCUSDT", weights, snapshot=snapshot, starting_balance="1000", stop_pct="0.045", leverage="10")
    assert len(out["fills"]) > 0
    with pytest.raises(ValueError, match="at most the exchange leverage the stop allows"):
        lab.replay("BTCUSDT", weights, snapshot=snapshot, starting_balance="1000", stop_pct="0.08", leverage="7")
    with pytest.raises(ValueError, match=r"needs a leverage in \[1, 10\]"):
        lab.replay("BTCUSDT", weights, snapshot=snapshot, starting_balance="1000", stop_pct="0.045", leverage="11")
