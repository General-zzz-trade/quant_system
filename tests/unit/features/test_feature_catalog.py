from features.feature_catalog import PRODUCTION_FEATURES, validate_model_features


def test_production_features_not_empty():
    assert len(PRODUCTION_FEATURES) >= 100  # ~105 expected


def test_validate_known_features():
    msgs = validate_model_features(["ret_1", "ret_3", "atr_norm_14"])
    assert msgs == []


def test_validate_unknown_features():
    msgs = validate_model_features(["ret_1", "fake_feature_xyz"])
    assert len(msgs) == 1
    assert "fake_feature_xyz" in msgs[0]


def test_validate_strict_raises():
    import pytest
    with pytest.raises(ValueError, match="fake_feature"):
        validate_model_features(["fake_feature"], strict=True)


def test_validate_empty_passes():
    assert validate_model_features([]) == []


def test_validate_model_name_in_message():
    msgs = validate_model_features(["nonexistent"], model_name="test_model")
    assert len(msgs) == 1
    assert "Model test_model" in msgs[0]
