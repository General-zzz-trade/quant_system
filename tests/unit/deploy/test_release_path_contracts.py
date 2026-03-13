from __future__ import annotations

import re
from pathlib import Path

import pytest


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_ci_workflow_covers_default_release_gate() -> None:
    text = _read(".github/workflows/ci.yml")
    for snippet in (
        "docker build --target paper -t quant-ci:paper .",
        "docker compose config >/dev/null",
        "bash scripts/ci_default_release_smoke.sh",
        "python -m pytest execution/tests/ -x -q --tb=short",
        "cargo test --manifest-path /app/ext/rust/Cargo.toml --locked",
    ):
        assert snippet in text


def test_default_release_path_uses_same_service_name_across_compose_deploy_and_workflow() -> None:
    compose = _read("docker-compose.yml")
    assert "\n  paper-multi:\n" in compose
    assert "--config /app/infra/config/examples/testnet_multi_gate_v2.yaml" in compose

    deploy_script = _read("scripts/deploy.sh")
    match = re.search(r"^SERVICES=\(([^)]*)\)", deploy_script, flags=re.MULTILINE)
    assert match is not None
    assert match.group(1).split() == ["paper-multi"]

    deploy_workflow = _read(".github/workflows/deploy.yml")
    assert 'workflows: ["CI"]' in deploy_workflow
    assert "bash scripts/deploy.sh" in deploy_workflow
    assert "paper-multi" in deploy_workflow


def test_deploy_readme_names_the_only_default_truth_sources() -> None:
    text = _read("deploy/README.md")
    for snippet in (
        "repo-root `docker-compose.yml`",
        "`.github/workflows/ci.yml`",
        "`.github/workflows/deploy.yml`",
        "`scripts/deploy.sh`",
        "`runner/live_runner.py`",
        "`deploy/systemd/logrotate-quant.conf`",
        "optional host-tuning helpers",
    ):
        assert snippet in text


@pytest.mark.parametrize(
    "path, snippets",
    [
        (
            "deploy/docker/docker-compose.yml",
            (
                "# Status: non-default deployment example.",
                "# Default release truth lives at repo-root docker-compose.yml + GitHub Actions.",
            ),
        ),
        (
            "deploy/systemd/quant-trader.service",
            (
                "# Status: candidate-production systemd unit.",
                "# Default release truth lives at repo-root docker-compose.yml + GitHub Actions.",
            ),
        ),
        (
            "deploy/k8s/deployment.yaml",
            (
                "# Status: candidate-production Kubernetes manifest.",
                "# This is not the current default release path.",
            ),
        ),
        (
            "deploy/argocd/application.yaml",
            (
                "# Status: experimental/candidate GitOps manifest.",
                "# This is not the current default release path.",
            ),
        ),
        (
            "deploy/argocd/rollback-config.yaml",
            (
                "# Status: experimental/candidate GitOps manifest.",
                "# This is not the current default release path.",
            ),
        ),
        (
            "deploy/argocd/analysis-template.yaml",
            (
                "# Status: experimental/candidate GitOps manifest.",
                "# This is not the current default release path.",
            ),
        ),
        (
            "deploy/argocd/project.yaml",
            (
                "# Status: experimental/candidate GitOps manifest.",
                "# This is not the current default release path.",
            ),
        ),
        (
            "deploy/argocd/rollout.yaml",
            (
                "# Status: experimental/candidate GitOps manifest.",
                "# This is not the current default release path.",
            ),
        ),
        (
            "deploy/k8s/external-secret.yaml",
            (
                "# Status: candidate-production Kubernetes manifest.",
                "# This is not the current default release path.",
            ),
        ),
        (
            "deploy/k8s/hpa.yaml",
            (
                "# Status: candidate-production Kubernetes manifest.",
                "# This is not the current default release path.",
            ),
        ),
        (
            "deploy/k8s/leader-election.yaml",
            (
                "# Status: candidate-production Kubernetes manifest.",
                "# This is not the current default release path.",
            ),
        ),
        (
            "deploy/k8s/namespace.yaml",
            (
                "# Status: candidate-production Kubernetes manifest.",
                "# This is not the current default release path.",
            ),
        ),
        (
            "deploy/k8s/network-policy.yaml",
            (
                "# Status: candidate-production Kubernetes manifest.",
                "# This is not the current default release path.",
            ),
        ),
        (
            "deploy/k8s/pdb.yaml",
            (
                "# Status: candidate-production Kubernetes manifest.",
                "# This is not the current default release path.",
            ),
        ),
        (
            "deploy/k8s/priority-class.yaml",
            (
                "# Status: candidate-production Kubernetes manifest.",
                "# This is not the current default release path.",
            ),
        ),
        (
            "deploy/k8s/pvc.yaml",
            (
                "# Status: candidate-production Kubernetes manifest.",
                "# This is not the current default release path.",
            ),
        ),
        (
            "deploy/k8s/secret-rotation-cronjob.yaml",
            (
                "# Status: candidate-production Kubernetes manifest.",
                "# This is not the current default release path.",
            ),
        ),
        (
            "deploy/systemd/logrotate-quant.conf",
            (
                "# Status: candidate-production systemd support file.",
                "# This is not the current default release path.",
            ),
        ),
    ],
)
def test_candidate_deploy_artifacts_have_explicit_non_default_banners(
    path: str,
    snippets: tuple[str, str],
) -> None:
    header = "\n".join(_read(path).splitlines()[:4])
    for snippet in snippets:
        assert snippet in header


def test_deprecated_deploy_dockerfile_fails_fast_with_repo_root_guidance() -> None:
    text = _read("deploy/docker/Dockerfile")
    assert "# Deprecated: use the repo-root Dockerfile target `paper`." in text
    assert "deploy/docker/Dockerfile is deprecated. Use the repo-root Dockerfile target paper." in text


def test_ci_smoke_script_uses_default_deploy_and_rollback_paths() -> None:
    text = _read("scripts/ci_default_release_smoke.sh")
    for snippet in (
        'SERVICE="${SMOKE_SERVICE:-paper-multi}"',
        'PROJECT_NAME="${COMPOSE_PROJECT_NAME:-quant-ci-smoke}"',
        "bash scripts/deploy.sh",
        'docker compose up -d --no-deps --force-recreate "$SERVICE"',
    ):
        assert snippet in text


@pytest.mark.parametrize(
    "path, snippets",
    [
        (
            "docs/api.md",
            (
                "runtime_direction.md",
                "repo-root `docker-compose.yml`",
                "`scripts/deploy.sh`",
                "health_auth_token_env",
            ),
        ),
        (
            "docs/production_runbook.md",
            (
                "repo-root `docker-compose.yml`",
                "`.github/workflows/deploy.yml`",
                "`paper-multi`",
                "health_auth_token_env",
            ),
        ),
        (
            "docs/operations.md",
            (
                "repo-root `docker-compose.yml`",
                "`.github/workflows/ci.yml`",
                "`scripts/deploy.sh`",
                "health_auth_token_env",
            ),
        ),
    ],
)
def test_default_docs_reference_the_same_release_path(
    path: str,
    snippets: tuple[str, ...],
) -> None:
    text = _read(path)
    for snippet in snippets:
        assert snippet in text
