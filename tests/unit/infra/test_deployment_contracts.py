# tests/unit/infra/test_deployment_contracts.py
"""Contract tests: verify deployment artifacts are consistent.

These tests catch drift between Dockerfile, docker-compose.yml,
deploy.sh, CI workflow, and production_runbook.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class TestDockerfileConsistency:
    def test_dockerfile_has_paper_target(self):
        """Compose references 'paper' build target — Dockerfile must define it."""
        dockerfile = (_ROOT / "Dockerfile").read_text()
        assert "AS paper" in dockerfile or "as paper" in dockerfile

    def test_dockerfile_copies_all_production_modules(self):
        """Dockerfile must COPY all production Python packages."""
        dockerfile = (_ROOT / "Dockerfile").read_text()
        required = [
            "alpha/", "engine/", "execution/", "features/",
            "runner/", "risk/", "state/", "decision/",
            "monitoring/", "event/",
        ]
        for mod in required:
            assert f"COPY {mod}" in dockerfile, f"Dockerfile missing COPY {mod}"

    def test_dockerfile_does_not_copy_tests(self):
        """Tests should NOT be in production image."""
        dockerfile = (_ROOT / "Dockerfile").read_text()
        # Only the CI stage should have tests
        paper_section = dockerfile.split("AS paper")[1] if "AS paper" in dockerfile else ""
        assert "COPY tests/" not in paper_section


class TestComposeConsistency:
    def test_compose_has_paper_multi_service(self):
        """Default deploy target must exist."""
        compose = (_ROOT / "docker-compose.yml").read_text()
        assert "paper-multi" in compose

    def test_compose_healthcheck_exists(self):
        """Service must have healthcheck for deploy gating."""
        compose = (_ROOT / "docker-compose.yml").read_text()
        assert "healthcheck" in compose

    def test_compose_restart_policy(self):
        """Production service must restart on failure."""
        compose = (_ROOT / "docker-compose.yml").read_text()
        assert "unless-stopped" in compose or "always" in compose


class TestDeployScriptConsistency:
    def test_deploy_script_targets_paper_multi(self):
        """deploy.sh must target the same service as compose."""
        deploy = (_ROOT / "scripts" / "deploy.sh").read_text()
        assert "paper-multi" in deploy

    def test_deploy_script_has_healthcheck_wait(self):
        """deploy.sh must wait for healthcheck before declaring success."""
        deploy = (_ROOT / "scripts" / "deploy.sh").read_text()
        assert "healthy" in deploy.lower() or "health" in deploy.lower()


class TestCIConsistency:
    def test_ci_runs_pytest(self):
        """CI must run Python tests."""
        ci = (_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "pytest" in ci

    def test_ci_runs_execution_tests(self):
        """CI must run execution subsystem tests."""
        ci = (_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "execution/tests" in ci

    def test_ci_runs_lint(self):
        """CI must run linter."""
        ci = (_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "ruff" in ci

    def test_ci_validates_compose(self):
        """CI must validate docker-compose config."""
        ci = (_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "compose" in ci.lower()


class TestRunbookConsistency:
    def test_runbook_references_live_runner(self):
        """Runbook must reference the sole production entry point."""
        runbook = (_ROOT / "docs" / "production_runbook.md").read_text()
        assert "runner/live_runner.py" in runbook

    def test_runbook_has_incident_taxonomy(self):
        """Runbook must document incident categories."""
        runbook = (_ROOT / "docs" / "production_runbook.md").read_text()
        assert "execution_timeout" in runbook
        assert "execution_reconcile" in runbook
        assert "IncidentCategory" in runbook or "incident_state" in runbook


class TestGitignoreSafety:
    def test_env_files_ignored(self):
        """All .env files must be gitignored."""
        gitignore = (_ROOT / ".gitignore").read_text()
        assert ".env" in gitignore

    def test_env_polymarket_ignored(self):
        gitignore = (_ROOT / ".gitignore").read_text()
        assert ".env.polymarket" in gitignore
