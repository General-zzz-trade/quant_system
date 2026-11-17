"""The research broker models positions up to 10 times the equity on isolated margin.

Owner 2026-11-17: "改为资金的10倍", then "10 倍，止损线一起重定": research first shows what 5x to 10x would do.
Never above the isolated (exchange) leverage. The runtime's own caps (trading/domain/risk/exposure.py) are not
touched here, and paper never runs isolated mode. Synthetic bars only."""
from __future__ import annotations

from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

from data.quality.causal_bars import HOUR, Bar
from lab.simulate import SymbolRules, simulate
from lab.strategy import StrategySpec
from trading.adapters.simulated.bar_broker import RESEARCH_MAX_EXPOSURE, ReplayConfig

RULES = SymbolRules(D("0.001"), D("0.001"), D("5"))


def _isolated(**kwargs):
    base = {"starting_balance": "1000", "isolated_leverage": "10", "max_exposure": "10", "stop_pct": "0.045",
            "entry_mode": "market", "qty_step": "0.001", "min_notional": "5"}
    return ReplayConfig(**{**base, **kwargs})


def test_research_takes_up_to_ten_times_the_equity():
    assert RESEARCH_MAX_EXPOSURE == D(10)
    assert _isolated().max_exposure == "10"
    assert _isolated(isolated_leverage="20", max_exposure="7", stop_pct="0.02").max_exposure == "7"


@pytest.mark.parametrize("kwargs", [{"isolated_leverage": "20", "max_exposure": "10.1", "stop_pct": "0.02"},
                                    {"isolated_leverage": "5", "max_exposure": "6", "stop_pct": "0.08"}])
def test_never_above_ten_or_the_isolated_leverage(kwargs):
    with pytest.raises(ValueError, match="max_exposure"):
        _isolated(**kwargs)


def test_a_sleeve_holds_nine_times_its_equity_at_ten_times():
    bars = [Bar(i * HOUR, D(100), D("100.2"), D("99.8"), D(100), D(10_000_000)) for i in range(10)]
    targets = pd.DataFrame({"available_at": np.array([b.available_at for b in bars], dtype=np.int64),
                            "target": 9.0, "ready": True})               # 0.9 of a position at 10x
    result = simulate("ALTUSDT", bars, targets, {}, StrategySpec(execution="market"), start_ms=0, rules=RULES,
                      starting_balance="1000", stop_pct="0.045", leverage="10", exchange_leverage=10)
    held = result.curve.iloc[-1]
    assert held["qty"] * held["mark"] / held["equity"] == pytest.approx(9.0, rel=0.01)
