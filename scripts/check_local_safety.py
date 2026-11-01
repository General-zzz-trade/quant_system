"""Run deterministic production safety tests, with network access disabled."""
from __future__ import annotations

import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    cache = root / ".cache" / "local-safety"
    cache.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(dir=cache) as scratch,
        patch.object(tempfile, "tempdir", scratch),
        patch.object(socket.socket, "connect", side_effect=AssertionError("offline checks cannot use network")),
        patch.object(socket, "create_connection", side_effect=AssertionError("offline checks cannot use network")),
    ):
        suite = unittest.defaultTestLoader.discover(str(root / "tests" / "offline"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        print("\nLocal safety checks passed. This is NOT a strategy-profitability or live-deployment approval.")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
