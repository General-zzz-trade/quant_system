from __future__ import annotations

import pytest

from infra.model_signing import verify_file


def test_verify_file_rejects_when_key_missing_and_unsigned_not_allowed(tmp_path, monkeypatch: pytest.MonkeyPatch):
    model = tmp_path / "model.pkl"
    model.write_bytes(b"dummy")
    monkeypatch.delenv("QUANT_MODEL_SIGN_KEY", raising=False)
    monkeypatch.delenv("QUANT_ALLOW_UNSIGNED_MODELS", raising=False)

    with pytest.raises(ValueError, match="QUANT_MODEL_SIGN_KEY"):
        verify_file(model)


def test_verify_file_allows_unsigned_in_dev_mode(tmp_path, monkeypatch: pytest.MonkeyPatch):
    model = tmp_path / "model.pkl"
    model.write_bytes(b"dummy")
    monkeypatch.delenv("QUANT_MODEL_SIGN_KEY", raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")

    assert verify_file(model) is True
