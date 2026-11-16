"""ADR-0017 / ADR-0016 (S8): the host install script, checked statically.

There is no container runtime on the development host, so the "run it once in a clean container" acceptance is
a manual step before M5 (recorded in the execution log). What can be checked here: it is valid bash that stops at
the first failing step, installs only hash-checked packages, reads no secret, and never enables, starts or
restarts a service.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "infra" / "install_host.sh"


def _code() -> list[str]:
    """The script's commands: comments and the closing here-document of instructions removed."""
    text = SCRIPT.read_text()
    text = text.split("cat <<EOF", 1)[0]
    return [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def test_it_is_valid_bash_that_stops_at_the_first_failure():
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True).returncode == 0
    assert "set -euo pipefail" in _code()


def test_packages_are_installed_only_against_their_hashes():
    pip = [line for line in _code() if "pip install" in line]
    assert len(pip) == 1 and "--require-hashes" in pip[0] and "--no-deps" in pip[0]


def test_it_never_enables_starts_or_restarts_a_service():
    for line in _code():
        assert not re.search(r"systemctl\s+(enable|start|restart|reload-or-restart|try-restart)", line), line


def test_it_reads_no_secret():
    for line in _code():
        assert ".env" not in line, line


def test_it_refuses_to_run_as_anyone_but_root():
    assert any('"$(id -u)" -ne 0' in line for line in _code())
