"""Compose deployment must not claim to deploy host systemd changes."""
from __future__ import annotations

import argparse
import sys
from pathlib import PurePosixPath


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-changed-files-stdin", action="store_true", required=True)
    parser.parse_args()
    blocked = []
    for line in sys.stdin:
        name = line.rstrip("\n")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            blocked.append(name)
        elif name.startswith(("infra/systemd/", "deploy/systemd/")) or name.endswith(".service"):
            blocked.append(name)
    if blocked:
        print("Compose deployment blocked: host service changes require a separate deployment:", file=sys.stderr)
        print("\n".join(blocked), file=sys.stderr)
        return 1
    print("Compose deployment scope passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
