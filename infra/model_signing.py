"""Model artifact signing — HMAC-SHA256 verification for pickle files.

Prevents arbitrary code execution from tampered model artifacts.
The signing key is read from QUANT_MODEL_SIGN_KEY env var.
If the env var is not set, signing is skipped with a warning (dev mode).
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_ENV_KEY = "QUANT_MODEL_SIGN_KEY"
_SIG_SUFFIX = ".sig"


def _get_key() -> bytes | None:
    key = os.environ.get(_ENV_KEY)
    if not key:
        return None
    return key.encode("utf-8")


def sign_file(path: Path) -> None:
    """Write an HMAC-SHA256 signature alongside the artifact."""
    key = _get_key()
    if key is None:
        logger.warning("Model signing skipped: %s not set", _ENV_KEY)
        return
    data = path.read_bytes()
    sig = hmac.new(key, data, hashlib.sha256).hexdigest()
    sig_path = path.with_suffix(path.suffix + _SIG_SUFFIX)
    sig_path.write_text(sig)


def verify_file(path: Path) -> bool:
    """Verify the HMAC-SHA256 signature of a model artifact.

    Returns True if:
      - Signature matches, OR
      - Signing key is not configured (dev mode — logs warning).
    Raises ValueError if signature is missing or invalid.
    """
    key = _get_key()
    if key is None:
        logger.warning("Model signature verification skipped: %s not set", _ENV_KEY)
        return True

    sig_path = path.with_suffix(path.suffix + _SIG_SUFFIX)
    if not sig_path.exists():
        raise ValueError(
            f"Missing signature file {sig_path}. "
            f"Re-save the model with signing enabled."
        )

    expected_sig = sig_path.read_text().strip()
    data = path.read_bytes()
    actual_sig = hmac.new(key, data, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError(
            f"Signature mismatch for {path}. Artifact may be tampered."
        )
    return True
