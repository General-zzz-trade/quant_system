from __future__ import annotations

from pathlib import Path

import yaml

from scripts.check_deploy_scope import (
    COMPOSE_DEPLOY_SERVICES,
    host_managed_changes,
    unknown_compose_services,
)

_ROOT = Path("/quant_system")


def test_compose_services_have_healthchecks_and_images():
    compose = yaml.safe_load((_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    for name in ("quant-paper", "quant-live", "quant-framework"):
        assert name in services
        assert services[name].get("image")
        assert services[name].get("healthcheck"), f"{name} missing healthcheck"


def test_deploy_script_defaults_to_real_service_name():
    deploy_sh = (_ROOT / "scripts/deploy.sh").read_text(encoding="utf-8")
    assert "SERVICES=(quant-paper)" in deploy_sh
    assert "alpha-runner" not in deploy_sh
    assert "paper-multi" not in deploy_sh
    assert "python3 -m scripts.check_deploy_scope --validate-services" in deploy_sh


def test_deploy_workflow_and_smoke_use_real_service_names():
    deploy_yml = (_ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
    smoke_sh = (_ROOT / "scripts/ci_default_release_smoke.sh").read_text(encoding="utf-8")
    assert "quant-paper" in deploy_yml
    assert "paper-multi" not in deploy_yml
    assert 'SERVICE="${SMOKE_SERVICE:-quant-paper}"' in smoke_sh
    assert "DEPLOY_SERVICES: quant-paper" in deploy_yml
    assert "python3 -m scripts.check_deploy_scope --guard-changed-files-stdin" in deploy_yml
    assert "steps.rolling_deploy.conclusion == 'failure'" in deploy_yml


def test_deploy_scope_only_allows_compose_services():
    assert tuple(COMPOSE_DEPLOY_SERVICES) == ("quant-paper", "quant-live", "quant-framework")
    assert unknown_compose_services(["quant-paper"]) == []
    assert unknown_compose_services(["bybit-alpha.service"]) == ["bybit-alpha.service"]


def test_host_managed_runtime_changes_are_blocked():
    changed = [
        "docs/deploy_truth.md",
        "scripts/run_bybit_mm.py",
        "execution/market_maker/engine.py",
    ]
    assert host_managed_changes(changed) == [
        "scripts/run_bybit_mm.py",
        "execution/market_maker/engine.py",
    ]
