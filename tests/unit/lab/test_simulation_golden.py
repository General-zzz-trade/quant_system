"""The research simulation's exact output on fixed synthetic data, pinned before the research broker stopped
using the Rust event types (ADR-0006, 2026-11-15). Paper and research share the broker, so a parity test
between them cannot see a change to it; this fingerprint can. A change here needs a recorded reason."""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal as D

import pandas as pd
import pytest

from lab.simulate import SYMBOL_RULES, simulate
from lab.strategy import StrategySpec, compute_targets
from tests.unit.lab.test_lab_engine import _bars, _funding
from trading.domain.market import frame_from_bars

PINNED = {          # computed on master a462529, before the research broker stopped using the Rust types
    "market": "ed9f1262fd7c77c696fcae73e3e4e340580ab01f7e7d5bf6311ad92a8c24c863",
    "maker": "f5342f065a7598bfbe78310ce23ba085e673640734b664d2a37b2810bb9fe5b7",
}


def _fingerprint(execution: str) -> str:
    bars = _bars(24 * 300)
    funding = {int(t): (D(str(round(r, 8))), D("100")) for t, r in _funding(len(bars)).items()}
    spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, execution=execution)
    targets = compute_targets(frame_from_bars(bars), pd.Series({t: float(r) for t, (r, _) in funding.items()}), spec)
    result = simulate("ETHUSDT", bars, targets, funding, spec, start_ms=bars[24 * 60].open_time,
                      rules=SYMBOL_RULES["ETHUSDT"])
    record = {"curve": [[repr(v) for v in row] for row in result.curve.itertuples(index=False)],
              "fills": result.fills, "orders": result.orders}
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()


@pytest.mark.parametrize("execution", sorted(PINNED))
def test_the_research_simulation_output_is_unchanged(execution):
    assert _fingerprint(execution) == PINNED[execution]
