# tests/unit/test_security_basics.py
"""Basic security invariant tests."""
from __future__ import annotations

import os
import re
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


class TestGitignore:
    def test_env_in_gitignore(self):
        gitignore = _ROOT / ".gitignore"
        assert gitignore.exists(), ".gitignore missing"
        text = gitignore.read_text()
        assert ".env" in text, ".env not in .gitignore"

    def test_key_files_in_gitignore(self):
        gitignore = _ROOT / ".gitignore"
        text = gitignore.read_text()
        for pattern in ["*.key", "*.pem"]:
            assert pattern in text, f"{pattern} not in .gitignore"


class TestMaxOrderNotional:
    def test_max_order_notional_exists(self):
        config = _ROOT / "strategy" / "config.py"
        assert config.exists()
        text = config.read_text()
        assert "MAX_ORDER_NOTIONAL" in text

    def test_max_order_notional_reasonable(self):
        from strategy.config import MAX_ORDER_NOTIONAL_PCT, MAX_ORDER_NOTIONAL_CEILING
        assert 0 < MAX_ORDER_NOTIONAL_PCT <= 5.0, f"PCT={MAX_ORDER_NOTIONAL_PCT} out of range"
        assert MAX_ORDER_NOTIONAL_CEILING <= 200_000, f"Ceiling={MAX_ORDER_NOTIONAL_CEILING} too high"


class TestNoHardcodedSecrets:
    _SECRET_RE = re.compile(
        r"""(?:api_key|api_secret|password)\s*=\s*['"][A-Za-z0-9+/=]{20,}['"]""",
        re.I,
    )

    def test_no_secrets_in_python_files(self):
        skip_dirs = {".git", "__pycache__", ".mypy_cache", "node_modules"}
        skip_files = {".env.example", "security_scan.py", "test_security_basics.py"}
        issues = []
        for root, dirs, files in os.walk(_ROOT):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for fname in files:
                if not fname.endswith(".py") or fname in skip_files:
                    continue
                fpath = Path(root) / fname
                try:
                    text = fpath.read_text(errors="replace")
                except OSError:
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if line.lstrip().startswith("#"):
                        continue
                    if self._SECRET_RE.search(line):
                        rel = fpath.relative_to(_ROOT)
                        issues.append(f"{rel}:{i}")
        assert not issues, f"Potential hardcoded secrets found: {issues}"


class TestDockerfile:
    def test_dockerfile_exists(self):
        assert (_ROOT / "Dockerfile").exists(), "Dockerfile missing (required by CI)"

    def test_dockerfile_has_ci_stage(self):
        text = (_ROOT / "Dockerfile").read_text()
        assert "AS ci" in text, "Dockerfile missing 'ci' stage"

    def test_dockerfile_has_paper_stage(self):
        text = (_ROOT / "Dockerfile").read_text()
        assert "AS paper" in text, "Dockerfile missing 'paper' stage"

    def test_docker_compose_exists(self):
        assert (_ROOT / "docker-compose.yml").exists(), "docker-compose.yml missing"
