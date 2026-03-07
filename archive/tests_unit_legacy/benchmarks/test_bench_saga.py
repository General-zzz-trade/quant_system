"""Performance benchmark: SagaManager transitions/sec."""
from __future__ import annotations

import time

import pytest

from engine.saga import SagaManager, SagaState


@pytest.mark.benchmark
def test_saga_transition_throughput():
    """Saga should process >10K operations/sec (create + transitions)."""
    n_sagas = 5_000
    mgr = SagaManager(max_completed=n_sagas)

    start = time.perf_counter()
    for i in range(n_sagas):
        oid = f"order-{i}"
        mgr.create(oid, f"intent-{i}", symbol="BTCUSDT", side="buy", qty=1.0)
        mgr.transition(oid, SagaState.SUBMITTED)
        mgr.transition(oid, SagaState.ACKED)
        mgr.transition(oid, SagaState.FILLED)
    elapsed = time.perf_counter() - start

    total_ops = n_sagas * 4  # 1 create + 3 transitions
    rate = total_ops / elapsed

    print(f"\nSaga: {rate:,.0f} ops/sec ({elapsed:.3f}s for {total_ops:,} operations)")
    assert rate > 10_000, f"Saga too slow: {rate:.0f} ops/sec"
