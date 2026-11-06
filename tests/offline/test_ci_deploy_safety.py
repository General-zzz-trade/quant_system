"""CI and paper deployment regression checks; Docker is always a fake executable."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def workflow_step(path: Path, title: str) -> str:
    """Extract a literal run block, so tests execute the actual workflow command."""
    lines = path.read_text().splitlines()
    start = lines.index(f"      - name: {title}")
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("      - "):
            break
        if lines[i] == "        run: |":
            result = []
            for line in lines[i + 1:]:
                if line and not line.startswith("          "):
                    break
                result.append(line[10:])
            return "\n".join(result)
        if lines[i].startswith("        run: "):
            return lines[i][13:]
    raise AssertionError(f"Missing run block: {title}")


class SecurityAndDeployTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)

    def command(self, args, *, text=None):
        return subprocess.run(args, cwd=self.root, env=self.env, input=text,
                              capture_output=True, text=True, timeout=15)

    def track(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        subprocess.run(["git", "add", "-f", "--", name], cwd=self.root, check=True)

    def scan(self):
        # Execute exactly the security job's run command against the fixture repo.
        target = self.root / "scripts"
        target.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / "scripts/check_ci_security.py", target / "check_ci_security.py")
        return self.command(["bash", "-eo", "pipefail", "-c", workflow_step(
            ROOT / ".github/workflows/ci.yml",
            "Check tracked credential assignments and private-key file names",
        )])

    def test_safe_environment_lookups_and_empty_literals_pass(self):
        self.track("safe.py", 'import os\nBYBIT_API_KEY = os.getenv("BYBIT_API_KEY")\nAPI_SECRET = ""\n')
        # Local secrets are intentionally not read by the tracked-source check.
        (self.root / ".env").write_text("must never be printed")
        self.assertEqual(self.scan().returncode, 0)

    def test_fake_credentials_fail_without_printing_values(self):
        cases = [
            'BYBIT_API_KEY = "fake-key-not-valid" # example os.environ .env\n',
            'BYBIT_API_SECRET: str = "fake-key-not-valid"\n',
            'OKX_API_PASSPHRASE = "fake-key-not-valid"\n',
            'class Config:\n API_KEY = "fake-key-not-valid"\n',
            'BYBIT_API_KEY, BYBIT_API_SECRET = "fake-key-not-valid", "fake-key-not-valid"\n',
        ]
        for source in cases:
            with self.subTest(source=source):
                self.track("example.py", source)
                result = self.scan()
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("example.py:", result.stderr)
                self.assertNotIn("fake-key-not-valid", result.stdout + result.stderr)

    def test_private_key_filename_fails(self):
        self.track("nested/test.pem", "not a real private key\n")
        self.assertEqual(self.scan().returncode, 1)

    def test_scan_failure_is_not_success(self):
        self.track("broken.py", "this is invalid Python !\n")
        self.assertEqual(self.scan().returncode, 1)
        shutil.rmtree(self.root / ".git")
        self.assertEqual(self.scan().returncode, 1)

    def scope(self, names):
        return self.command([sys.executable, str(ROOT / "scripts/check_deploy_scope.py"),
                             "--guard-changed-files-stdin"], text=names)

    def test_scope_allows_compose_sources_and_blocks_host_services(self):
        self.assertEqual(self.scope("Dockerfile\nrunner/alpha_main.py\nCargo.toml\n").returncode, 0)
        for name in ["infra/systemd/bybit-alpha.service", "deploy/systemd/test.timer", "../escape"]:
            with self.subTest(name=name):
                self.assertEqual(self.scope(name + "\n").returncode, 1)

    def paper_fixture(self):
        scripts = self.root / "scripts"
        scripts.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / "scripts/deploy.sh", scripts / "deploy.sh")
        (scripts / "run_bybit_alpha.py").write_text("# Never executed by deployment tests\n")
        shutil.copyfile(ROOT / "docker-compose.yml", self.root / "docker-compose.yml")
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        docker = bin_dir / "docker"
        docker.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
with Path(os.environ["MOCK_DOCKER_LOG"]).open("a") as out:
    out.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1:3] == ["compose", "config"]:
    raise SystemExit(int(os.environ.get("MOCK_CONFIG_EXIT", "0")))
if sys.argv[1:3] == ["compose", "up"]:
    raise SystemExit(int(os.environ.get("MOCK_UP_EXIT", "0")))
if sys.argv[1:2] == ["tag"]:
    raise SystemExit(int(os.environ.get("MOCK_TAG_EXIT", "0")))
raise SystemExit("unexpected Docker operation in test")
''')
        docker.chmod(0o755)
        self.env.update(PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
                        MOCK_DOCKER_LOG=str(self.root / "docker.jsonl"), DEPLOY_SERVICES="quant-paper")

    def docker_calls(self):
        path = self.root / "docker.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def deploy(self, *args):
        return self.command(["bash", "scripts/deploy.sh", *args])

    def test_preflight_never_restarts_and_success_waits_for_health(self):
        self.paper_fixture()
        self.assertEqual(self.deploy("--check", "quant-paper").returncode, 0)
        self.assertEqual(self.docker_calls(), [["compose", "config", "--quiet"]])
        self.assertEqual(self.deploy("quant-paper").returncode, 0)
        self.assertEqual(self.docker_calls()[-1], ["compose", "up", "-d", "--no-deps", "--no-build",
                                                  "--force-recreate", "--wait", "--wait-timeout", "120", "quant-paper"])

    def test_missing_entry_blocks_before_docker(self):
        self.paper_fixture()
        entry = self.root / "scripts/run_bybit_alpha.py"
        entry.unlink()
        self.assertNotEqual(self.deploy("--check", "quant-paper").returncode, 0)
        self.assertEqual(self.docker_calls(), [])
        external = self.root / "unversioned-entry.py"
        external.write_text("# external code must not satisfy preflight\n")
        entry.symlink_to(external)
        self.assertNotEqual(self.deploy("--check", "quant-paper").returncode, 0)
        self.assertEqual(self.docker_calls(), [])

    def test_live_and_extra_services_are_rejected(self):
        self.paper_fixture()
        for args in [("quant-live",), ("quant-paper", "quant-live"), ()]:
            with self.subTest(args=args):
                self.assertEqual(self.deploy(*args).returncode, 2)
        self.assertEqual(self.docker_calls(), [])

    def test_config_and_health_failures_propagate(self):
        self.paper_fixture()
        self.env["MOCK_CONFIG_EXIT"] = "17"
        self.assertEqual(self.deploy("quant-paper").returncode, 17)
        self.assertFalse(any(call[:2] == ["compose", "up"] for call in self.docker_calls()))
        self.env["MOCK_CONFIG_EXIT"] = "0"
        self.env["MOCK_UP_EXIT"] = "23"
        self.assertEqual(self.deploy("quant-paper").returncode, 23)

    def rollback(self, available="true"):
        block = workflow_step(ROOT / ".github/workflows/deploy.yml", "Rollback on failure")
        block = block.replace("${{ steps.changes.outputs.prev_commit }}", "mock-previous-sha")
        block = block.replace("${{ steps.backup.outputs.available }}", available)
        self.env.update(TELEGRAM_BOT_TOKEN="", TELEGRAM_CHAT_ID="")
        return self.command(["bash", "-eo", "pipefail", "-c", block])

    def test_rollback_success_and_failure_are_verified(self):
        self.paper_fixture()
        self.assertEqual(self.rollback().returncode, 0)
        self.env["MOCK_UP_EXIT"] = "23"
        self.assertEqual(self.rollback().returncode, 23)
        self.env["MOCK_TAG_EXIT"] = "19"
        self.assertEqual(self.rollback().returncode, 19)
        count = len(self.docker_calls())
        self.assertEqual(self.rollback(available="").returncode, 1)
        self.assertEqual(len(self.docker_calls()), count)

    def test_workflow_references_and_paper_only_build(self):
        deploy = ROOT / ".github/workflows/deploy.yml"
        for name in ["scripts/deploy.sh", "scripts/check_deploy_scope.py", "scripts/check_ci_security.py"]:
            self.assertTrue((ROOT / name).is_file())
            self.assertFalse((ROOT / name).is_symlink())
        self.assertEqual(workflow_step(deploy, "Docker build"), 'docker compose build --no-cache "$DEPLOY_SERVICES"')
        self.assertIn("DEPLOY_SERVICES: quant-paper", deploy.read_text())
        self.assertNotIn("needs_rebuild", deploy.read_text())
        self.assertNotIn("git diff --name-only \"$PREV\" HEAD || true", deploy.read_text())
        self.assertIn("inputs.confirm_deploy && github.ref == 'refs/heads/master'", deploy.read_text())

    def test_invalid_previous_revision_stops_before_diff(self):
        home = self.root / "home"
        state = home / ".local/state/quant-system/quant-paper-last-good-deploy"
        state.parent.mkdir(parents=True)
        state.write_text("invalid-revision\n")
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        git = bin_dir / "git"
        git.write_text('#!/bin/sh\nexit 17\n')
        git.chmod(0o755)
        self.env.update(HOME=str(home), GITHUB_OUTPUT=str(self.root / "output"),
                        PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
        step = workflow_step(ROOT / ".github/workflows/deploy.yml", "Resolve previous deployed revision")
        self.assertEqual(self.command(["bash", "-eo", "pipefail", "-c", step]).returncode, 17)
        self.assertFalse((self.root / "output").exists())
        step = workflow_step(ROOT / ".github/workflows/deploy.yml", "Guard deploy scope")
        step = step.replace("${{ steps.changes.outputs.prev_commit }}", "invalid-revision")
        self.assertEqual(self.command(["bash", "-eo", "pipefail", "-c", step]).returncode, 17)


if __name__ == "__main__":
    unittest.main()
