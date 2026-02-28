# tests_unit/test_live_bar_validator.py
"""Tests for LiveBarValidator — live data quality gate."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest

from data.quality.live_validator import LiveBarQualityConfig, LiveBarValidator


@dataclass(frozen=True)
class FakeBar:
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


def _bar(o=100, h=105, lo=95, c=102, v=1000):
    return FakeBar(
        open=Decimal(str(o)),
        high=Decimal(str(h)),
        low=Decimal(str(lo)),
        close=Decimal(str(c)),
        volume=Decimal(str(v)),
    )


class TestLiveBarValidator:
    def test_valid_bar_passes(self):
        v = LiveBarValidator()
        assert v.validate(_bar()) is True
        assert v.stats() == {"passed": 1, "rejected": 0}

    def test_negative_volume_rejected(self):
        v = LiveBarValidator()
        assert v.validate(_bar(v=-1)) is False
        assert v.stats() == {"passed": 0, "rejected": 1}

    def test_ohlc_violation_high_below_close(self):
        v = LiveBarValidator()
        assert v.validate(_bar(h=90, c=102)) is False
        assert v.stats()["rejected"] == 1

    def test_ohlc_violation_high_below_low(self):
        v = LiveBarValidator()
        assert v.validate(_bar(h=80, lo=95)) is False
        assert v.stats()["rejected"] == 1

    def test_ohlc_violation_low_above_open(self):
        v = LiveBarValidator()
        assert v.validate(_bar(o=100, lo=110, h=120, c=115)) is False
        assert v.stats()["rejected"] == 1

    def test_excessive_price_change_rejected(self):
        v = LiveBarValidator()
        v.validate(_bar(c=100))
        assert v.validate(_bar(c=200)) is False
        assert v.stats() == {"passed": 1, "rejected": 1}

    def test_price_change_within_threshold_passes(self):
        v = LiveBarValidator()
        v.validate(_bar(c=100))
        assert v.validate(_bar(o=100, h=130, lo=95, c=130)) is True
        assert v.stats() == {"passed": 2, "rejected": 0}

    def test_first_bar_skips_price_change_check(self):
        v = LiveBarValidator(config=LiveBarQualityConfig(max_price_change_pct=1.0))
        assert v.validate(_bar(o=50000, h=51000, lo=49000, c=50000)) is True

    def test_stats_accumulate(self):
        v = LiveBarValidator()
        v.validate(_bar())
        v.validate(_bar())
        v.validate(_bar(v=-1))
        v.validate(_bar())
        assert v.stats() == {"passed": 3, "rejected": 1}

    def test_disabled_volume_check(self):
        cfg = LiveBarQualityConfig(reject_negative_volume=False)
        v = LiveBarValidator(config=cfg)
        assert v.validate(_bar(v=-5)) is True

    def test_disabled_ohlc_check(self):
        cfg = LiveBarQualityConfig(reject_ohlc_violation=False)
        v = LiveBarValidator(config=cfg)
        assert v.validate(_bar(h=80, lo=95)) is True

    def test_disabled_price_change_check(self):
        cfg = LiveBarQualityConfig(max_price_change_pct=0.0)
        v = LiveBarValidator(config=cfg)
        v.validate(_bar(c=100))
        assert v.validate(_bar(o=1000, h=1050, lo=950, c=1000)) is True

    def test_all_checks_disabled(self):
        cfg = LiveBarQualityConfig(
            reject_negative_volume=False,
            reject_ohlc_violation=False,
            max_price_change_pct=0.0,
        )
        v = LiveBarValidator(config=cfg)
        assert v.validate(_bar(o=200, h=50, lo=300, c=10, v=-99)) is True
        assert v.stats() == {"passed": 1, "rejected": 0}
