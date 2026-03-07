"""Tests for HFT risk checker."""
from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from engine.tick_engine import HFTOrder
from state.shared_position import SharedPositionStore
from strategies.hft.risk import HFTRiskChecker, HFTRiskConfig


def _order(
    side: str = "buy",
    qty: float = 0.01,
    symbol: str = "BTCUSDT",
    reduce_only: bool = False,
) -> HFTOrder:
    return HFTOrder(
        symbol=symbol,
        side=side,
        qty=qty,
        strategy_id="test",
        reduce_only=reduce_only,
    )


class TestHFTRiskChecker:
    def test_basic_allow(self):
        checker = HFTRiskChecker()
        allowed, reason = checker.check(_order())
        assert allowed is True
        assert reason == "ok"

    def test_kill_switch(self):
        checker = HFTRiskChecker()
        checker.kill()
        allowed, reason = checker.check(_order())
        assert allowed is False
        assert "kill_switch" in reason

    def test_unkill(self):
        checker = HFTRiskChecker()
        checker.kill()
        checker.unkill()
        allowed, _ = checker.check(_order())
        assert allowed is True

    def test_daily_loss_limit(self):
        cfg = HFTRiskConfig(max_daily_loss=100.0)
        checker = HFTRiskChecker(cfg=cfg)
        checker.update_pnl(-150.0)
        allowed, reason = checker.check(_order())
        assert allowed is False
        assert "daily_loss" in reason

    def test_daily_loss_reset(self):
        cfg = HFTRiskConfig(max_daily_loss=100.0)
        checker = HFTRiskChecker(cfg=cfg)
        checker.update_pnl(-150.0)
        checker.reset_daily()
        allowed, _ = checker.check(_order())
        assert allowed is True

    def test_rate_limit(self):
        cfg = HFTRiskConfig(max_orders_per_second=3)
        checker = HFTRiskChecker(cfg=cfg)

        for _ in range(3):
            allowed, _ = checker.check(_order())
            assert allowed is True

        allowed, reason = checker.check(_order())
        assert allowed is False
        assert "rate_limit" in reason

    def test_position_notional_limit(self):
        store = SharedPositionStore()
        store.update_position("BTCUSDT", Decimal("0.5"))

        cfg = HFTRiskConfig(max_position_notional=10_000.0)
        checker = HFTRiskChecker(
            cfg=cfg,
            position_store=store,
            reference_prices={"BTCUSDT": 50000.0},
        )

        # Current notional: 0.5 * 50000 = 25000 > 10000
        # Adding more should be blocked
        allowed, reason = checker.check(_order(side="buy", qty=0.01))
        assert allowed is False
        assert "notional" in reason

    def test_reduce_only_allowed_over_limit(self):
        store = SharedPositionStore()
        store.update_position("BTCUSDT", Decimal("0.5"))

        cfg = HFTRiskConfig(max_position_notional=10_000.0)
        checker = HFTRiskChecker(
            cfg=cfg,
            position_store=store,
            reference_prices={"BTCUSDT": 50000.0},
        )

        # reduce_only should be allowed even over limit
        allowed, reason = checker.check(
            _order(side="sell", qty=0.1, reduce_only=True)
        )
        assert allowed is True

    def test_loss_cooldown(self):
        cfg = HFTRiskConfig(
            max_daily_loss=100.0,
            cooldown_after_loss_s=60.0,
        )
        checker = HFTRiskChecker(cfg=cfg)
        # Loss > 50% of max_daily_loss
        checker.update_pnl(-60.0)
        allowed, reason = checker.check(_order())
        assert allowed is False
        assert "cooldown" in reason

    def test_no_position_store_allows(self):
        checker = HFTRiskChecker()
        allowed, _ = checker.check(_order())
        assert allowed is True
