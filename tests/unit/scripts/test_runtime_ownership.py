from __future__ import annotations

from types import SimpleNamespace

import pytest

from execution.adapters.bybit.config import BybitConfig
from scripts.ops.runtime_ownership import claim_bybit_symbol_lease


def _adapter(*, api_key: str = "key-a", base_url: str = "https://api-demo.bybit.com"):
    return SimpleNamespace(
        _config=BybitConfig(
            api_key=api_key,
            api_secret="secret",
            base_url=base_url,
            account_type="UNIFIED",
            category="linear",
        )
    )


def test_claim_same_symbol_same_account_conflicts(tmp_path):
    lease = claim_bybit_symbol_lease(
        adapter=_adapter(api_key="same-account"),
        service_name="svc-a",
        symbols=("ETHUSDT",),
        lock_dir=str(tmp_path),
    )
    try:
        with pytest.raises(RuntimeError, match="runtime symbol already claimed"):
            claim_bybit_symbol_lease(
                adapter=_adapter(api_key="same-account"),
                service_name="svc-b",
                symbols=("ETHUSDT",),
                lock_dir=str(tmp_path),
            )
    finally:
        lease.release()


def test_claim_same_symbol_different_account_succeeds(tmp_path):
    lease_a = claim_bybit_symbol_lease(
        adapter=_adapter(api_key="account-a"),
        service_name="svc-a",
        symbols=("ETHUSDT",),
        lock_dir=str(tmp_path),
    )
    try:
        lease_b = claim_bybit_symbol_lease(
            adapter=_adapter(api_key="account-b"),
            service_name="svc-b",
            symbols=("ETHUSDT",),
            lock_dir=str(tmp_path),
        )
        lease_b.release()
    finally:
        lease_a.release()


def test_claim_different_symbols_same_account_succeeds(tmp_path):
    lease_a = claim_bybit_symbol_lease(
        adapter=_adapter(api_key="same-account"),
        service_name="svc-a",
        symbols=("BTCUSDT",),
        lock_dir=str(tmp_path),
    )
    try:
        lease_b = claim_bybit_symbol_lease(
            adapter=_adapter(api_key="same-account"),
            service_name="svc-b",
            symbols=("ETHUSDT",),
            lock_dir=str(tmp_path),
        )
        lease_b.release()
    finally:
        lease_a.release()
