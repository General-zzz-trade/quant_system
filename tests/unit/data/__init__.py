"""Test package shim that preserves access to the real top-level ``data`` package."""

from pathlib import Path

_REAL_DATA_PACKAGE = Path(__file__).resolve().parents[3] / "data"
if str(_REAL_DATA_PACKAGE) not in __path__:
    __path__.append(str(_REAL_DATA_PACKAGE))
