"""Model artifact signing — HMAC-SHA256 verification for model files.

Prevents loading of tampered model artifacts via HMAC-SHA256 signatures.
The signing key is read from QUANT_MODEL_SIGN_KEY env var.
If the env var is not set, signing is skipped with a warning (dev/demo mode only).

Any configuration that can reach a real-money account (Bybit mainnet URL,
Binance with testnet disabled, OKX credentials without simulated trading)
counts as live: unsigned models are then ALWAYS rejected regardless of
QUANT_ALLOW_UNSIGNED_MODELS.

Every artifact is read exactly once: the verified bytes are the bytes that get
deserialized, so a file swapped between check and load cannot be used.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import pickle  # noqa: S403 — HMAC-verified before deserialize
from pathlib import Path

logger = logging.getLogger(__name__)

_ENV_KEY = "QUANT_MODEL_SIGN_KEY"
_ALLOW_UNSIGNED_ENV = "QUANT_ALLOW_UNSIGNED_MODELS"
_SIG_SUFFIX = ".sig"
_SIGNABLE_SUFFIXES = {".pkl", ".json"}


_FALSE = {"0", "false", "no", "off"}
_TRUE = {"1", "true", "yes", "on"}


def _is_live() -> bool:
    """Return True if this process may be configured against a real-money venue."""
    env = os.environ
    if env.get("BYBIT_BASE_URL", "").startswith("https://api.bybit.com"):
        return True
    if env.get("BINANCE_TESTNET", "1").strip().lower() in _FALSE:
        return True
    if env.get("OKX_API_KEY", "").strip() and env.get("OKX_SIMULATED", "0").strip().lower() not in _TRUE:
        return True
    return False


def _get_key() -> bytes | None:
    key = os.environ.get(_ENV_KEY)
    if not key:
        return None
    return key.encode("utf-8")


def allow_unsigned_models() -> bool:
    """Whether unsigned model loading is explicitly allowed (demo/dev only).

    In live mode (see ``_is_live``), this ALWAYS returns False — unsigned
    models are never allowed in production.
    """
    if _is_live():
        return False
    raw = os.environ.get(_ALLOW_UNSIGNED_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def is_verification_enforced() -> bool:
    """Whether model signature verification must be enforced."""
    return _get_key() is not None or not allow_unsigned_models()


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


def verify_file(path: Path, data: bytes | None = None) -> bool:
    """Verify the HMAC-SHA256 signature of a model artifact.

    ``data`` is the exact content that the caller will deserialize; when it is
    omitted the file is read here.

    Returns True if:
      - Signature matches, OR
      - Signing key is not configured AND not in live mode AND bypass enabled.
    Raises ValueError if signature is missing, invalid, or bypass not allowed.
    """
    key = _get_key()
    if key is None:
        if _is_live():
            raise ValueError(
                f"LIVE MODE: {_ENV_KEY} must be set. "
                f"Unsigned models are never allowed in live trading."
            )
        if allow_unsigned_models():
            logger.warning(
                "Model signature verification BYPASSED (demo/dev mode): "
                "%s not set and %s enabled",
                _ENV_KEY,
                _ALLOW_UNSIGNED_ENV,
            )
            return True
        raise ValueError(
            f"Model signature verification requires {_ENV_KEY}. "
            f"For development only, set {_ALLOW_UNSIGNED_ENV}=1 to allow unsigned models."
        )

    sig_path = path.with_suffix(path.suffix + _SIG_SUFFIX)
    if not sig_path.exists():
        if _is_live():
            raise ValueError(
                f"LIVE MODE: Missing signature file {sig_path}. "
                f"All models must be signed for live trading."
            )
        raise ValueError(
            f"Missing signature file {sig_path}. "
            f"Re-save the model with signing enabled."
        )

    expected_sig = sig_path.read_text().strip()
    if data is None:
        data = path.read_bytes()
    actual_sig = hmac.new(key, data, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError(
            f"Signature mismatch for {path}. Artifact may be tampered."
        )
    return True


def resolve_artifact(model_dir: str | Path, name: str) -> Path:
    """Resolve a config-supplied artifact name strictly inside ``model_dir``."""
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError(f"Invalid model artifact name: {name!r}")
    base = Path(model_dir).resolve()
    path = (base / name).resolve()
    if path != base and base not in path.parents:
        raise ValueError(f"Model artifact escapes its model directory: {name!r}")
    return path


def read_verified_bytes(path: str | Path) -> bytes:
    """Read an artifact once and verify exactly those bytes."""
    p = Path(path)
    data = p.read_bytes()
    verify_file(p, data=data)
    return data


def load_verified_pickle(path: str | Path):  # noqa: S301
    """Verify an artifact's HMAC signature, then load it via pickle."""
    return pickle.loads(read_verified_bytes(path))  # noqa: S301 — HMAC-verified bytes


def load_verified_json(path: str | Path):
    """Verify a JSON artifact (config or model export), then parse it."""
    return json.loads(read_verified_bytes(path).decode("utf-8"))


def sign_model_dir(model_dir: Path) -> int:
    """Sign all signable artifacts (.pkl, .json) in a model directory.

    Returns the number of files signed. Skips if QUANT_MODEL_SIGN_KEY is not set.
    """
    key = _get_key()
    if key is None:
        logger.warning("sign_model_dir skipped: %s not set", _ENV_KEY)
        return 0

    if not model_dir.is_dir():
        logger.warning("sign_model_dir: %s is not a directory", model_dir)
        return 0

    signed = 0
    for child in sorted(model_dir.iterdir()):
        if child.suffix in _SIGNABLE_SUFFIXES and not child.name.endswith(_SIG_SUFFIX):
            sign_file(child)
            signed += 1
            logger.info("Signed %s", child)
    logger.info("sign_model_dir: signed %d files in %s", signed, model_dir)
    return signed
