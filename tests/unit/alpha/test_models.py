"""Tests for alpha/models — LGBMAlphaModel, XGBAlphaModel."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from alpha.models.lgbm_alpha import LGBMAlphaModel
from alpha.models.xgb_alpha import XGBAlphaModel


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


@pytest.mark.slow
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
