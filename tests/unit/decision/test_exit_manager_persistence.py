"""Tests for ExitManager checkpoint/restore persistence."""
from __future__ import annotations


from decision.exit_manager import ExitManager


def _make_exit_manager(**kwargs) -> ExitManager:
    from alpha.v11_config import ExitConfig
    return ExitManager(config=ExitConfig(), **kwargs)


class TestExitManagerPersistence:
    def test_checkpoint_restore_roundtrip(self):
        em = _make_exit_manager()
        em.on_entry("BTCUSDT", price=40000.0, bar=10, direction=1.0)
        em.update_price("BTCUSDT", 41000.0)
        em.on_entry("ETHUSDT", price=3000.0, bar=15, direction=-1.0)
        em.update_price("ETHUSDT", 2900.0)

        data = em.checkpoint()
        assert "BTCUSDT" in data
        assert "ETHUSDT" in data
        assert data["BTCUSDT"]["peak_price"] == 41000.0
        assert data["ETHUSDT"]["direction"] == -1.0

        # Restore into fresh manager
        em2 = _make_exit_manager()
        assert em2._positions == {}
        em2.restore(data)

        assert "BTCUSDT" in em2._positions
        assert "ETHUSDT" in em2._positions
        assert em2._positions["BTCUSDT"].entry_price == 40000.0
        assert em2._positions["BTCUSDT"].peak_price == 41000.0
        assert em2._positions["BTCUSDT"].entry_bar == 10
        assert em2._positions["BTCUSDT"].direction == 1.0
        assert em2._positions["ETHUSDT"].entry_price == 3000.0
        assert em2._positions["ETHUSDT"].peak_price == 2900.0
        assert em2._positions["ETHUSDT"].direction == -1.0

    def test_restore_empty_graceful(self):
        em = _make_exit_manager()
        em.restore({})
        assert em._positions == {}

    def test_restore_partial(self):
        em = _make_exit_manager()
        em.on_entry("BTCUSDT", price=40000.0, bar=5, direction=1.0)
        em.on_entry("ETHUSDT", price=3000.0, bar=5, direction=1.0)

        # Checkpoint only BTC
        data = {"BTCUSDT": {"entry_price": 40000.0, "peak_price": 40500.0,
                            "entry_bar": 5, "direction": 1.0}}

        em2 = _make_exit_manager()
        em2.restore(data)
        assert "BTCUSDT" in em2._positions
        assert "ETHUSDT" not in em2._positions

    def test_checkpoint_empty(self):
        em = _make_exit_manager()
        data = em.checkpoint()
        assert data == {}

    def test_restored_manager_continues_tracking(self):
        """After restore, price tracking and exit checks should work normally."""
        em = _make_exit_manager()
        em.on_entry("BTCUSDT", price=40000.0, bar=10, direction=1.0)
        em.update_price("BTCUSDT", 42000.0)

        data = em.checkpoint()
        em2 = _make_exit_manager()
        em2.restore(data)

        # Continue price tracking on restored manager
        em2.update_price("BTCUSDT", 43000.0)
        assert em2._positions["BTCUSDT"].peak_price == 43000.0

        # Exit should work
        em2.on_exit("BTCUSDT")
        assert "BTCUSDT" not in em2._positions
