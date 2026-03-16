"""Tests for state.shared_position.SharedPositionStore."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

from state.shared_position import SharedPositionStore


class TestSharedPositionStore:
    def test_get_position_default_zero(self):
        store = SharedPositionStore()
        assert store.get_position("ETHUSDT") == Decimal("0")

    def test_update_position(self):
        store = SharedPositionStore()
        store.update_position("ETHUSDT", Decimal("5.0"))
        assert store.get_position("ETHUSDT") == Decimal("5.0")

    def test_add_fill_buy(self):
        store = SharedPositionStore()
        result = store.add_fill("ETHUSDT", Decimal("2.0"), "buy")
        assert result == Decimal("2.0")
        assert store.get_position("ETHUSDT") == Decimal("2.0")

    def test_add_fill_sell(self):
        store = SharedPositionStore()
        store.update_position("ETHUSDT", Decimal("5.0"))
        result = store.add_fill("ETHUSDT", Decimal("3.0"), "sell")
        assert result == Decimal("2.0")
        assert store.get_position("ETHUSDT") == Decimal("2.0")

    def test_add_fill_accumulation(self):
        store = SharedPositionStore()
        store.add_fill("ETHUSDT", Decimal("1.0"), "buy")
        store.add_fill("ETHUSDT", Decimal("2.0"), "buy")
        store.add_fill("ETHUSDT", Decimal("0.5"), "sell")
        assert store.get_position("ETHUSDT") == Decimal("2.5")

    def test_all_positions_returns_copy(self):
        store = SharedPositionStore()
        store.update_position("ETHUSDT", Decimal("1.0"))
        store.update_position("BTCUSDT", Decimal("0.5"))
        positions = store.all_positions()
        assert positions == {
            "ETHUSDT": Decimal("1.0"),
            "BTCUSDT": Decimal("0.5"),
        }
        # Mutating the copy should not affect the store
        positions["ETHUSDT"] = Decimal("999")
        assert store.get_position("ETHUSDT") == Decimal("1.0")

    def test_multi_symbol_isolation(self):
        store = SharedPositionStore()
        store.update_position("ETHUSDT", Decimal("10"))
        store.update_position("BTCUSDT", Decimal("0.1"))
        assert store.get_position("ETHUSDT") == Decimal("10")
        assert store.get_position("BTCUSDT") == Decimal("0.1")
        assert store.get_position("SOLUSDT") == Decimal("0")

    def test_thread_safety_concurrent_fills(self):
        store = SharedPositionStore()
        n = 100

        def buy_one():
            store.add_fill("ETHUSDT", Decimal("1"), "buy")

        def sell_one():
            store.add_fill("ETHUSDT", Decimal("1"), "sell")

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            for _ in range(n):
                futures.append(pool.submit(buy_one))
            for _ in range(n):
                futures.append(pool.submit(sell_one))
            for f in futures:
                f.result()

        # n buys and n sells should net to zero
        assert store.get_position("ETHUSDT") == Decimal("0")
