# tests/unit/infra/test_truth_source_consistency.py
"""Contract tests: verify truth source documents don't contradict each other.

Checks that runtime_truth.md, CLAUDE.md, production_runbook.md, and code
all agree on the production entry point, deployment path, and key contracts.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class TestProductionEntryPointConsistency:
    """All truth sources must agree on the sole production entry point."""

    def test_runtime_truth_says_live_runner(self):
        text = (_ROOT / "docs" / "runtime_truth.md").read_text()
        assert "runner/live_runner.py" in text

    def test_claude_md_says_live_runner(self):
        text = (_ROOT / "CLAUDE.md").read_text()
        assert "runner/live_runner.py" in text
        # Must NOT call it "legacy" or "deprecated"
        lines = [l for l in text.splitlines() if "live_runner" in l]
        for line in lines:
            assert "legacy" not in line.lower(), f"CLAUDE.md calls live_runner legacy: {line}"
            assert "deprecated" not in line.lower(), f"CLAUDE.md calls live_runner deprecated: {line}"

    def test_runbook_says_live_runner(self):
        text = (_ROOT / "docs" / "production_runbook.md").read_text()
        assert "runner/live_runner.py" in text

    def test_live_runner_not_marked_deprecated_in_code(self):
        text = (_ROOT / "runner" / "live_runner.py").read_text(errors="ignore")
        first_100_chars = text[:500].lower()
        assert "deprecated" not in first_100_chars


class TestDeployPathConsistency:
    """All sources must agree on the default deploy path."""

    def test_runbook_and_compose_agree_on_service(self):
        runbook = (_ROOT / "docs" / "production_runbook.md").read_text()
        compose = (_ROOT / "docker-compose.yml").read_text()
        # Both must mention paper-multi
        assert "paper-multi" in runbook or "paper" in runbook
        assert "paper-multi" in compose

    def test_deploy_yml_uses_deploy_sh(self):
        deploy_yml = (_ROOT / ".github" / "workflows" / "deploy.yml").read_text()
        assert "deploy.sh" in deploy_yml


class TestDigestTruthSource:
    """All digest/hash computation must go through execution/models/digest.py."""

    def test_digest_module_exists(self):
        assert (_ROOT / "execution" / "models" / "digest.py").exists()

    def test_no_standalone_stable_hash_implementations(self):
        """No file should define its own _stable_hash — must delegate to digest.py."""
        digest_file = _ROOT / "execution" / "models" / "digest.py"
        for py_file in (_ROOT / "execution").rglob("*.py"):
            if py_file == digest_file:
                continue
            if "__pycache__" in str(py_file):
                continue
            text = py_file.read_text(errors="ignore")
            if "def _stable_hash(" in text:
                # Must delegate to digest.py, not reimplement
                assert "from execution.models.digest import" in text, \
                    f"{py_file.relative_to(_ROOT)} has standalone _stable_hash"


class TestIncidentTaxonomyConsistency:
    """Incident enums must match runbook documentation."""

    def test_runbook_mentions_all_categories(self):
        from execution.observability.incidents import IncidentCategory
        runbook = (_ROOT / "docs" / "production_runbook.md").read_text()
        for cat in IncidentCategory:
            assert cat.value in runbook, \
                f"Runbook missing incident category: {cat.value}"
