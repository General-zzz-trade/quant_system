#!/usr/bin/env python3
"""Split the whole test suite into shards that CI runs in parallel.

    python3 scripts/ci_shard.py <index> <count>      # prints the test files of one shard

Every ``test_*.py`` under ``tests/`` except ``tests/performance`` belongs to exactly one shard, so the
shards together are the whole suite (``tests/architecture`` checks this). Files are dealt out heaviest
first to the lightest shard. A file weighs its measured seconds in ``scripts/ci_durations.json`` (the tests
over 1 s in a CI run) plus an estimate for its other tests from its size; without that file, its size alone.
The split depends only on the files in the checkout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDED = ("performance",)
DURATIONS = "scripts/ci_durations.json"


def weights(files: list[Path], root: Path = ROOT) -> dict[Path, float]:
    path = root / DURATIONS
    if not path.is_file():
        return {f: float(f.stat().st_size) for f in files}
    data = json.loads(path.read_text(encoding="utf-8"))
    per_byte, measured = float(data["seconds_per_kilobyte"]) / 1000, data["seconds"]
    return {f: float(measured.get(str(f.relative_to(root)), 0.0)) + f.stat().st_size * per_byte for f in files}


def test_files(root: Path = ROOT) -> list[Path]:
    tests = root / "tests"
    return sorted(p for p in tests.rglob("test_*.py") if p.relative_to(tests).parts[0] not in EXCLUDED)


def shards(count: int, root: Path = ROOT) -> list[list[str]]:
    if count < 1:
        raise ValueError("shard count must be positive")
    out: list[list[str]] = [[] for _ in range(count)]
    load = [0.0] * count
    files = test_files(root)
    weight = weights(files, root)
    for path in sorted(files, key=lambda p: (-weight[p], str(p))):
        k = load.index(min(load))
        out[k].append(str(path.relative_to(root)))
        load[k] += weight[path]
    return [sorted(files) for files in out]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    index, count = int(argv[0]), int(argv[1])
    if not 0 <= index < count:
        print("shard index out of range", file=sys.stderr)
        return 2
    print(" ".join(shards(count)[index]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
