# tests/unit/infra/test_config_security.py
"""Tests for config security: no plaintext secrets, env-based resolution."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from infra.config.loader import (
    SecurityError,
    EnvSecretProvider,
    check_no_plaintext_secrets,
    load_config_secure,
    resolve_credentials,
)


class TestCheckNoPlaintextSecrets:
    def test_clean_config_passes(self) -> None:
        config = {
            "trading": {"symbol": "BTCUSDT", "exchange": "binance"},
            "credentials": {"api_key_env": "BINANCE_API_KEY", "api_secret_env": "BINANCE_SECRET"},
        }
        violations = check_no_plaintext_secrets(config)
        assert violations == []

    def test_detects_hardcoded_api_key(self) -> None:
        config = {
            "credentials": {"api_key": "aB1cD2eF3gH4iJ5kL6mN7oP8qR9sT0u"}
        }
        violations = check_no_plaintext_secrets(config)
        assert len(violations) == 1
        assert "api_key" in violations[0]

    def test_detects_hardcoded_api_secret(self) -> None:
        config = {
            "credentials": {"api_secret": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"}
        }
        violations = check_no_plaintext_secrets(config)
        assert len(violations) >= 1

    def test_env_suffix_keys_are_allowed(self) -> None:
        config = {
            "credentials": {
                "api_key_env": "MY_API_KEY_VAR",
                "api_secret_env": "MY_SECRET_VAR",
            }
        }
        violations = check_no_plaintext_secrets(config)
        assert violations == []

    def test_nested_secrets_detected(self) -> None:
        config = {
            "exchange": {
                "binance": {
                    "api_key": "aBcDeFgHiJkLmNoPqRsTuVwXyZ123456"
                }
            }
        }
        violations = check_no_plaintext_secrets(config)
        assert len(violations) >= 1

    def test_short_values_ignored(self) -> None:
        """Short strings (< 9 chars) in sensitive keys should not trigger."""
        config = {"credentials": {"api_key": "test"}}
        violations = check_no_plaintext_secrets(config)
        assert violations == []


class TestLoadConfigSecure:
    def test_clean_json_loads(self, tmp_path: Path) -> None:
        cfg = {"trading": {"symbol": "BTCUSDT"}}
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg))
        result = load_config_secure(p)
        assert result["trading"]["symbol"] == "BTCUSDT"

    def test_rejects_secret_in_json(self, tmp_path: Path) -> None:
        cfg = {"credentials": {"api_key": "aBcDeFgHiJkLmNoPqRsTuVwXyZ123456"}}
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg))
        with pytest.raises(SecurityError, match="plaintext secret"):
            load_config_secure(p)


class TestResolveCredentials:
    def test_resolves_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_API_KEY", "my-key-123")
        monkeypatch.setenv("TEST_API_SECRET", "my-secret-456")

        config = {
            "credentials": {
                "api_key_env": "TEST_API_KEY",
                "api_secret_env": "TEST_API_SECRET",
            }
        }
        creds = resolve_credentials(config)
        assert creds["api_key"] == "my-key-123"
        assert creds["api_secret"] == "my-secret-456"

    def test_missing_env_raises(self) -> None:
        config = {
            "credentials": {"api_key_env": "NONEXISTENT_VAR_12345"}
        }
        with pytest.raises(KeyError, match="not set"):
            resolve_credentials(config)

    def test_empty_credentials_returns_empty(self) -> None:
        config = {"trading": {"symbol": "BTCUSDT"}}
        creds = resolve_credentials(config)
        assert creds == {}

    def test_custom_provider(self) -> None:
        class MockProvider:
            def get_secret(self, key: str) -> str:
                return f"resolved-{key}"

        config = {"credentials": {"api_key_env": "MY_KEY"}}
        creds = resolve_credentials(config, provider=MockProvider())
        assert creds["api_key"] == "resolved-MY_KEY"
