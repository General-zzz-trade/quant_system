"""infra/deploy_logrotate.sh, step 4 of the host install: a long validation output must not end the install
(2026-11-17: `logrotate -d | head -5` under pipefail died of SIGPIPE, exit 141), and a rejected config still fails.
A fake `logrotate` on PATH; the source and target files are in a temporary directory."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "infra" / "deploy_logrotate.sh"


def _run(tmp_path, *, lines, code):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "logrotate"
    fake.write_text(f"#!/usr/bin/env bash\nfor i in $(seq 1 {lines}); do echo \"debug line $i\"; done\nexit {code}\n")
    fake.chmod(0o755)
    src = tmp_path / "quant-system"
    src.write_text("/tmp/x.log { daily }\n")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "QUANT_LOGROTATE_SRC": str(src),
           "QUANT_LOGROTATE_DST": str(tmp_path / "installed")}
    return subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=env)


def test_a_long_validation_output_does_not_end_the_install(tmp_path):
    result = _run(tmp_path, lines=5000, code=0)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[:5] == [f"debug line {i}" for i in range(1, 6)]
    assert "debug line 6" not in result.stdout
    assert (tmp_path / "installed").read_text() == "/tmp/x.log { daily }\n"


def test_a_rejected_config_still_fails(tmp_path):
    result = _run(tmp_path, lines=3, code=1)
    assert result.returncode == 1 and "logrotate rejected" in result.stdout
