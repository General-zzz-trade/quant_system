"""Mutation self-check: does a test catch a small defect injected into the code?

    python3 scripts/dev/mutate.py "<default pytest paths>" <mutants.json> [jobs]

Each mutant is ``[label, file, old, new]`` or ``[label, file, old, new, "<pytest paths>"]``; the optional
fifth item picks the tests for that mutant (an empty string means the default paths). ``old`` must occur
exactly once in ``file``.

Every worker gets its own git worktree of HEAD with the uncommitted changes and untracked files copied
in, so the real working tree is never modified. The selected tests must pass on the unmutated copy first
("BASELINE FAILS" otherwise): a failing baseline would make every mutant look killed.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())


def make_copy(k: int) -> pathlib.Path:
    path = pathlib.Path(tempfile.mkdtemp(prefix=f"mut{k}-")) / "wt"
    subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "-q", "--detach", str(path), "HEAD"], check=True)
    patch = subprocess.run(["git", "-C", str(ROOT), "diff", "HEAD", "--binary"], capture_output=True,
                           check=True).stdout
    if patch.strip():
        subprocess.run(["git", "-C", str(path), "apply", "--whitespace=nowarn"], input=patch, check=True)
    others = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "--others", "--exclude-standard"],
                                     text=True)
    for name in others.splitlines():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    return path


def _pytest(copy: pathlib.Path, selected: list[str]) -> int:
    return subprocess.run(["nice", "-n", "10", sys.executable, "-m", "pytest", *selected, "-q", "-p",
                           "no:cacheprovider", "-x"], capture_output=True, text=True, cwd=copy).returncode


class Worker:
    def __init__(self, copy: pathlib.Path, default_tests: list[str]):
        self.copy, self.default_tests, self.baseline = copy, default_tests, {}

    def run(self, mutant: list[str]) -> str:
        label, name, old, new = mutant[:4]
        selected = (mutant[4].split() if len(mutant) > 4 else []) or self.default_tests
        key = tuple(selected)
        if key not in self.baseline:
            self.baseline[key] = _pytest(self.copy, selected) == 0
        if not self.baseline[key]:
            return f"{label}: BASELINE FAILS (the selected tests fail without the mutant)"
        path = self.copy / name
        original = path.read_text()
        if original.count(old) != 1:
            return f"{label}: PATTERN COUNT {original.count(old)}"
        path.write_text(original.replace(old, new, 1))
        try:
            code = _pytest(self.copy, selected)
        finally:
            path.write_text(original)
        return f"{label}: {'killed' if code else 'SURVIVED'}"


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    default_tests, spec = argv[1].split(), json.loads(pathlib.Path(argv[2]).read_text())
    jobs = int(argv[3]) if len(argv) > 3 else 2
    copies = [make_copy(k) for k in range(jobs)]
    try:
        workers = [Worker(copy, default_tests) for copy in copies]
        queues = [spec[k::jobs] for k in range(jobs)]
        with ThreadPoolExecutor(jobs) as pool:
            chunks = list(pool.map(lambda kq: [workers[kq[0]].run(m) for m in kq[1]], enumerate(queues)))
        order = {m[0]: i for i, m in enumerate(spec)}
        lines = sorted((line for chunk in chunks for line in chunk), key=lambda s: order.get(s.split(":")[0], 0))
        print("\n".join(lines))
        return 1 if any("SURVIVED" in line or "BASELINE" in line or "PATTERN" in line for line in lines) else 0
    finally:
        for copy in copies:
            subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(copy)], check=False)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
