"""Shared fixture: one development run of the forward-test candidates on a synthetic snapshot."""
from __future__ import annotations

import pytest

from tests.unit.lab.test_pair_study import write_snapshot

DAY = 24 * 3_600_000
T0 = 1_609_459_200_000          # 2021-01-01
CANDIDATE_WINDOW = {"b_start": T0 + 300 * DAY, "c_start": T0 + 200 * DAY, "limit": T0 + 410 * DAY}


@pytest.fixture(scope="session")
def study(tmp_path_factory):
    """(root, market dir, premium dir, registry, summary); specs are in root / "out"."""
    from lab import candidates_study
    root = tmp_path_factory.mktemp("candidates")
    market, premium = write_snapshot(root / "data", 420)
    registry = root / "trials.jsonl"
    summary = candidates_study.run_study(market, premium, root / "out", registry=registry, **CANDIDATE_WINDOW)
    return root, market, premium, registry, summary
