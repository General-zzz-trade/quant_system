"""Tests for alpha/models — MACrossAlpha, LGBMAlphaModel, XGBAlphaModel."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from alpha.base import Signal
from alpha.models.lgbm_alpha import LGBMAlphaModel, _Signal as LGBMSignal
from alpha.models.xgb_alpha import XGBAlphaModel, _Signal as XGBSignal
from alpha.models.ma_cross import MACrossAlpha


# ── MACrossAlpha ───────────────────────────────────────────


class TestMACrossAlpha:
    def test_no_signal_during_warmup(self):
        model = MACrossAlpha(fast=3, slow=5)
        # Only 5 closes — need slow+2=7 to produce signals
        closes = [100.0, 101.0, 102.0, 103.0, 104.0]
        result = model.predict(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={"close": closes},
        )
        assert result is None

    def test_bullish_cross(self):
        model = MACrossAlpha(fast=2, slow=4)
        # Build closes where fast MA crosses above slow MA
        # We need at least slow+2=6 values
        # Previous bar: fast_ma <= slow_ma; current bar: fast_ma > slow_ma
        closes = [100.0, 98.0, 96.0, 94.0, 100.0, 108.0]
        result = model.predict(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={"close": closes},
        )
        assert result is not None
        assert result.side == "long"
        assert result.strength == 1.0

    def test_bearish_cross(self):
        model = MACrossAlpha(fast=2, slow=4)
        # Build closes where fast MA crosses below slow MA
        closes = [100.0, 105.0, 110.0, 115.0, 105.0, 90.0]
        result = model.predict(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={"close": closes},
        )
        assert result is not None
        assert result.side == "short"
        assert result.strength == 1.0

    def test_no_cross_flat(self):
        model = MACrossAlpha(fast=2, slow=4)
        # All same price => no cross
        closes = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        result = model.predict(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={"close": closes},
        )
        assert result is not None
        assert result.side == "flat"

    def test_empty_closes(self):
        model = MACrossAlpha()
        result = model.predict(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={"close": []},
        )
        assert result is None

    def test_missing_close_key(self):
        model = MACrossAlpha()
        result = model.predict(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={"sma": 100},
        )
        assert result is None

    def test_meta_contains_params(self):
        model = MACrossAlpha(fast=5, slow=10)
        closes = [float(100 + i) for i in range(15)]
        result = model.predict(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={"close": closes},
        )
        if result is not None:
            assert result.meta["fast"] == 5
            assert result.meta["slow"] == 10


# ── LGBMAlphaModel ─────────────────────────────────────────


class TestLGBMAlphaModel:
    def test_predict_none_without_model(self):
        model = LGBMAlphaModel(name="test", feature_names=("f1", "f2"))
        result = model.predict(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={"f1": 1.0, "f2": 2.0},
        )
        assert result is None

    def test_predict_long_with_mock_model(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]

        model = LGBMAlphaModel(name="test", feature_names=("f1", "f2"), threshold=0.1)
        model._model = mock_model

        result = model.predict(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={"f1": 1.0, "f2": 2.0},
        )
        assert result is not None
        assert result.side == "long"
        assert result.strength == 0.5

    def test_predict_short_with_mock_model(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [-0.3]

        model = LGBMAlphaModel(name="test", feature_names=("f1",), threshold=0.1)
        model._model = mock_model

        result = model.predict(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={"f1": 1.0},
        )
        assert result is not None
        assert result.side == "short"
        assert result.strength == 0.3

    def test_predict_flat_within_threshold(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.05]

        model = LGBMAlphaModel(name="test", feature_names=("f1",), threshold=0.1)
        model._model = mock_model

        result = model.predict(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={"f1": 1.0},
        )
        assert result is not None
        assert result.side == "flat"
        assert result.strength == 0.0

    def test_strength_capped_at_1(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [5.0]

        model = LGBMAlphaModel(name="test", feature_names=("f1",))
        model._model = mock_model

        result = model.predict(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={"f1": 1.0},
        )
        assert result.strength == 1.0

    def test_missing_feature_defaults_to_nan(self):
        """Missing features should be NaN (LGBM handles natively), not 0."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]

        model = LGBMAlphaModel(name="test", feature_names=("f1", "f2"))
        model._model = mock_model

        model.predict(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={"f1": 1.0},  # f2 missing
        )
        call_args = mock_model.predict.call_args[0][0]
        import math
        assert call_args[0][0] == 1.0
        assert math.isnan(call_args[0][1])


# ── XGBAlphaModel ──────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:.*joblib will operate in serial mode.*:UserWarning")
class TestXGBAlphaModel:
    def test_predict_none_without_model(self):
        model = XGBAlphaModel(name="test", feature_names=("f1",))
        result = model.predict(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={"f1": 1.0},
        )
        assert result is None

    def test_predict_long_with_mock(self):
        import numpy as np

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8])

        model = XGBAlphaModel(name="xgb_test", feature_names=("f1", "f2"), threshold=0.1)
        model._model = mock_model

        result = model.predict(
            symbol="ETHUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={"f1": 1.0, "f2": 2.0},
        )
        assert result is not None
        assert result.side == "long"
        assert result.strength == 0.8

    def test_predict_short_with_mock(self):
        import numpy as np

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([-0.6])

        model = XGBAlphaModel(name="xgb_test", feature_names=("f1",), threshold=0.1)
        model._model = mock_model

        result = model.predict(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={"f1": 1.0},
        )
        assert result.side == "short"
        assert result.strength == 0.6

    def test_predict_flat_with_mock(self):
        import numpy as np

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.0])

        model = XGBAlphaModel(name="xgb_test", feature_names=("f1",), threshold=0.1)
        model._model = mock_model

        result = model.predict(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={"f1": 1.0},
        )
        assert result.side == "flat"
        assert result.strength == 0.0

    def test_signal_fields(self):
        import numpy as np

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5])

        model = XGBAlphaModel(name="xgb_sig", feature_names=("f1",))
        model._model = mock_model

        result = model.predict(
            symbol="BTCUSDT",
            ts=datetime(2024, 6, 15, tzinfo=timezone.utc),
            features={"f1": 42.0},
        )
        assert result.symbol == "BTCUSDT"
        assert result.ts == datetime(2024, 6, 15, tzinfo=timezone.utc)
