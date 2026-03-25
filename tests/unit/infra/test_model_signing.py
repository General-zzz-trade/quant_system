from __future__ import annotations

import pytest

from infra.model_signing import verify_file, sign_file, sign_model_dir, allow_unsigned_models


def test_verify_file_rejects_when_key_missing_and_unsigned_not_allowed(tmp_path, monkeypatch: pytest.MonkeyPatch):
    model = tmp_path / "model.pkl"
    model.write_bytes(b"dummy")
    monkeypatch.delenv("QUANT_MODEL_SIGN_KEY", raising=False)
    monkeypatch.delenv("QUANT_ALLOW_UNSIGNED_MODELS", raising=False)
    monkeypatch.delenv("BYBIT_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="QUANT_MODEL_SIGN_KEY"):
        verify_file(model)


def test_verify_file_allows_unsigned_in_dev_mode(tmp_path, monkeypatch: pytest.MonkeyPatch):
    model = tmp_path / "model.pkl"
    model.write_bytes(b"dummy")
    monkeypatch.delenv("QUANT_MODEL_SIGN_KEY", raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.delenv("BYBIT_BASE_URL", raising=False)

    assert verify_file(model) is True


def test_verify_file_rejects_unsigned_in_live_mode(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """In live mode, unsigned models must ALWAYS be rejected."""
    model = tmp_path / "model.pkl"
    model.write_bytes(b"dummy")
    monkeypatch.delenv("QUANT_MODEL_SIGN_KEY", raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("BYBIT_BASE_URL", "https://api.bybit.com")

    with pytest.raises(ValueError, match="LIVE MODE"):
        verify_file(model)


def test_allow_unsigned_blocked_in_live_mode(monkeypatch: pytest.MonkeyPatch):
    """allow_unsigned_models() must return False in live mode."""
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("BYBIT_BASE_URL", "https://api.bybit.com")
    assert allow_unsigned_models() is False


def test_allow_unsigned_works_in_demo_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("BYBIT_BASE_URL", "https://api-demo.bybit.com")
    assert allow_unsigned_models() is True


def test_sign_and_verify_roundtrip(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Sign a file, then verify it succeeds."""
    monkeypatch.setenv("QUANT_MODEL_SIGN_KEY", "test-secret-key-1234")
    monkeypatch.delenv("BYBIT_BASE_URL", raising=False)

    model = tmp_path / "model.pkl"
    model.write_bytes(b"model-content-here")

    sign_file(model)
    assert verify_file(model) is True


def test_verify_rejects_tampered_file(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Tampered file must fail verification."""
    monkeypatch.setenv("QUANT_MODEL_SIGN_KEY", "test-secret-key-1234")
    monkeypatch.delenv("BYBIT_BASE_URL", raising=False)

    model = tmp_path / "model.pkl"
    model.write_bytes(b"original-content")
    sign_file(model)

    # Tamper with the file
    model.write_bytes(b"tampered-content")
    with pytest.raises(ValueError, match="Signature mismatch"):
        verify_file(model)


def test_verify_rejects_missing_sig_in_live(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Live mode: missing .sig file must raise with LIVE MODE message."""
    monkeypatch.setenv("QUANT_MODEL_SIGN_KEY", "test-secret-key-1234")
    monkeypatch.setenv("BYBIT_BASE_URL", "https://api.bybit.com")

    model = tmp_path / "model.pkl"
    model.write_bytes(b"content")
    # No .sig file created

    with pytest.raises(ValueError, match="LIVE MODE"):
        verify_file(model)


def test_sign_model_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """sign_model_dir signs all .pkl and .json files."""
    monkeypatch.setenv("QUANT_MODEL_SIGN_KEY", "test-secret-key-1234")

    (tmp_path / "model.pkl").write_bytes(b"pkl-data")
    (tmp_path / "config.json").write_text('{"key": "value"}')
    (tmp_path / "readme.txt").write_text("not signed")

    count = sign_model_dir(tmp_path)
    assert count == 2
    assert (tmp_path / "model.pkl.sig").exists()
    assert (tmp_path / "config.json.sig").exists()
    assert not (tmp_path / "readme.txt.sig").exists()


def test_sign_model_dir_skips_without_key(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("QUANT_MODEL_SIGN_KEY", raising=False)
    (tmp_path / "model.pkl").write_bytes(b"data")
    count = sign_model_dir(tmp_path)
    assert count == 0
