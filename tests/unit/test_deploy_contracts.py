from __future__ import annotations

from pathlib import Path

import yaml


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


def test_deploy_workflow_and_smoke_use_real_service_names():
    deploy_yml = (_ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
    smoke_sh = (_ROOT / "scripts/ci_default_release_smoke.sh").read_text(encoding="utf-8")
    assert "quant-paper" in deploy_yml
    assert "paper-multi" not in deploy_yml
    assert 'SERVICE="${SMOKE_SERVICE:-quant-paper}"' in smoke_sh
