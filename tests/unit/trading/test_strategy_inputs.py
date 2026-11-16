"""R7a of the runtime extension (docs/arch/runtime-lab-strategies-plan.md): what a strategy can read.

- A window's hourly frame carries the kline fields the research lab's ideas use - ``quote_volume``, ``trades``,
  ``taker_buy_volume``, ``taker_buy_quote_volume``, named as in the lab's data snapshots - when the bars have them;
  a bar without them gives NaN (unknown), never 0. The overlay's targets do not change.
- A strategy names the symbols it reads besides each traded one (``references``, e.g. BTCUSDT for the ideas gated on
  BTC's trend). The runtime has no data-only symbols: a spec whose strategy reads a symbol it does not list is refused
  by name, so such a strategy runs only with that symbol in its list.
"""
from __future__ import annotations

import json
from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

from trading.domain.bars import HOUR, Bar
from trading.domain.market import frame_from_bars
from trading.domain.strategy import spec as spec_module
from trading.domain.strategy.base import STRATEGIES, OverlayStrategy, PerSymbol, Window
from trading.domain.strategy.overlay import StrategySpec, compute_targets
from trading.domain.strategy.spec import parse_spec

T0 = 1_700_006_400_000 // HOUR * HOUR
EXTRAS = ("quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume")
OVERLAY = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")


def _bars(n, *, extras=True):
    out = []
    for k in range(n):
        c = D(100) + D(k % 7) - D(k % 3)
        more = {"quote_volume": D(1000 + k), "trades": D(10 + k), "taker_buy_volume": D(4 + k),
                "taker_buy_quote_volume": D(400 + k)} if extras else None
        out.append(Bar(T0 + k * HOUR, c, c + 1, c - 1, c, D(5), extras=more))
    return out


def test_a_window_frame_carries_the_kline_fields_named_as_in_the_lab_snapshots():
    frame = frame_from_bars(_bars(3), extras=True)
    assert list(frame["taker_buy_quote_volume"]) == [400.0, 401.0, 402.0]
    assert list(frame["quote_volume"]) == [1000.0, 1001.0, 1002.0] and list(frame["trades"]) == [10.0, 11.0, 12.0]
    assert set(EXTRAS) <= set(frame.columns)
    plain = frame_from_bars(_bars(3))
    assert not set(EXTRAS) & set(plain.columns)                     # unchanged for the existing callers


def test_a_bar_without_the_fields_gives_unknown_not_zero():
    bars = _bars(2) + _bars(1, extras=False)
    frame = frame_from_bars(bars, extras=True)
    assert frame["quote_volume"].iloc[:2].tolist() == [1000.0, 1001.0]
    assert all(np.isnan(frame[c].iloc[2]) for c in EXTRAS)


def test_the_overlay_targets_do_not_change_with_the_extra_columns():
    bars = _bars(24 * 60)
    funding = pd.Series(1e-4, index=[T0 + k * 8 * HOUR for k in range(180)])
    pd.testing.assert_frame_equal(compute_targets(frame_from_bars(bars, extras=True), funding, OVERLAY),
                                  compute_targets(frame_from_bars(bars), funding, OVERLAY))


def test_the_session_window_has_the_fields():
    from trading.application.session import TargetSession

    class Parts:
        executor = consumer = stops = gate = flattener = equity = None
    session = TargetSession("BTCUSDT", OVERLAY, journal=None, rules=None, make_order=None, execution=Parts())
    session.bars.extend(_bars(30))
    assert session.window().hourly["taker_buy_quote_volume"].iloc[-1] == 429.0


def test_a_strategy_reads_no_other_symbol_unless_it_says_so():
    assert OverlayStrategy(OVERLAY).references() == () and PerSymbol().references() == ()


class Gated(PerSymbol):
    """Reads BTCUSDT's window whatever symbol it trades."""
    name = "gated"

    def __init__(self, params):
        self.params = params

    def references(self):
        return ("BTCUSDT",)


def _v2(symbols):
    return {"schema": 2, "kind": "target_strategy", "strategy": {"name": "gated", "params": {}},
            "execution": {"band": 0.05, "execution": "market", "ttl_bars": 1}, "symbols": symbols,
            "allow_short": False, "passed": False}


def test_a_spec_whose_strategy_reads_an_unlisted_symbol_is_refused(monkeypatch):
    monkeypatch.setitem(STRATEGIES, "gated", Gated)
    monkeypatch.setitem(spec_module.PARAMS, "gated", lambda p: dict(p))
    with pytest.raises(ValueError, match="BTCUSDT"):
        parse_spec(_v2(["ETHUSDT", "SOLUSDT"]))
    spec = parse_spec(_v2(["BTCUSDT", "ETHUSDT"]))
    assert spec.strategy().references() == ("BTCUSDT",)
    assert json.loads(json.dumps(spec.record()))["symbols"] == ["BTCUSDT", "ETHUSDT"]


def test_a_window_is_still_two_parts():
    window = Window(frame_from_bars(_bars(2), extras=True), pd.Series(dtype=float))
    assert len(window.hourly) == 2 and window.funding.empty


def test_the_running_runtime_s_windows_carry_the_exchange_s_fields(tmp_path, monkeypatch):
    from runner.binance_target import TargetRuntime
    from tests.unit.lab.test_target_runtime import _world
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.delenv("QUANT_MODEL_SIGN_KEY", raising=False)
    world = _world(tmp_path, OVERLAY)
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "paper", mode="paper",
                            market=world["adapter"], clock=lambda: world["clock"]["now"])
    runtime.warmup()
    hourly = runtime.sessions["ETHUSDT"].window().hourly
    rows = {r["start"]: r for r in world["rows"]["ETHUSDT"]}
    last = rows[int(hourly["open_time"].iloc[-1])]
    assert hourly["taker_buy_quote_volume"].iloc[-1] == pytest.approx(last["taker_buy_quote_volume"])
    assert hourly["quote_volume"].iloc[-1] == pytest.approx(last["quote_volume"])
    assert not hourly[list(EXTRAS)].isna().any().any()
