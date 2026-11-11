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

from shared.runtime_state import initialize_runtime_state


ROOT = Path(__file__).resolve().parents[2]
REVISION = "a" * 40
PREVIOUS_REVISION = "b" * 40
IMAGE_ID = "sha256:" + "c" * 64
INPUT_DIGEST = "d" * 64


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
        for key in list(self.env):
            if key.startswith("QUANT_RELEASE_"):
                self.env.pop(key)
        self.env["HOME"] = str(self.root / "home")
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
        shutil.copyfile(ROOT / "scripts/check_paper_runtime.py", scripts / "check_paper_runtime.py")
        # The acceptance algorithm has separate tests. Here a deterministic
        # executable verifies shell orchestration and every failure boundary.
        (scripts / "check_release_acceptance.py").write_text('''import argparse, json, os, sys
from pathlib import Path
parser = argparse.ArgumentParser()
for name in ("revision", "repository", "image", "output"):
    parser.add_argument("--" + name, required=True)
parser.add_argument("--preflight")
parser.add_argument("--runtime-status")
args = parser.parse_args()
config = json.load(sys.stdin)
assert args.image == config["services"]["quant-paper"]["image"]
stage = "postflight" if args.preflight else "preflight"
with Path("release-calls.jsonl").open("a") as out:
    out.write(json.dumps({"stage": stage, "image": args.image, "revision": args.revision}) + "\\n")
failure = int(os.environ.get("MOCK_RELEASE_" + stage.upper() + "_EXIT", "0"))
if failure:
    raise SystemExit(failure)
release = {"revision": args.revision, "image_id": "sha256:" + "c" * 64, "input_digest": "d" * 64}
if args.preflight:
    assert json.loads(Path(args.preflight).read_text())["release"] == release
    assert json.loads(Path(args.runtime_status).read_text())["synthetic"] is True
    assert args.image == release["image_id"]
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"release": release, "paper_accepted": stage == "postflight", "stage": stage}))
''')
        (scripts / "check_runtime_health.py").write_text("# Invoked only through fake Docker\n")
        shared = self.root / "shared"
        shared.mkdir(exist_ok=True)
        for name in ("runtime_state.py", "execution_lease.py", "execution_journal.py", "drawdown_guard.py"):
            shutil.copyfile(ROOT / "shared" / name, shared / name)
        (scripts / "run_bybit_alpha.py").write_text("# Never executed by deployment tests\n")
        shutil.copyfile(ROOT / "docker-compose.yml", self.root / "docker-compose.yml")
        state = tempfile.TemporaryDirectory()
        self.addCleanup(state.cleanup)
        self.state = Path(state.name)
        initialize_runtime_state(self.state, "paper-test")
        self.compose = {"services": {"quant-paper": {
            "image": "quant-paper:latest",
            "command": ["python3", "-m", "scripts.run_bybit_alpha", "--dry-run"],
            "environment": {"RUNTIME_STATE_DIR": "/app/data/runtime", "RISK_SCOPE": "paper-test",
                            "ORDER_JOURNAL_PATH": "/app/data/runtime/orders.sqlite",
                            "DRAWDOWN_STATE_PATH": "/app/data/runtime/drawdown.sqlite"},
            "volumes": [{"type": "bind", "source": str(self.state), "target": "/app/data/runtime",
                         "bind": {"create_host_path": False}}],
        }}}
        self.compose_path = self.root / "compose.json"
        self.compose_path.write_text(json.dumps(self.compose))
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        docker = bin_dir / "docker"
        docker.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
with Path(os.environ["MOCK_DOCKER_LOG"]).open("a") as out:
    out.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1:3] == ["compose", "config"]:
    if sys.argv[3:] == ["--format", "json"]:
        config = json.loads(Path("compose.json").read_text())
        config["services"]["quant-paper"]["image"] = os.environ.get("QUANT_RELEASE_IMAGE", "quant-paper:latest")
        print(json.dumps(config))
    raise SystemExit(int(os.environ.get("MOCK_CONFIG_EXIT", "0")))
if sys.argv[1:3] == ["compose", "up"]:
    Path("up-environment.json").write_text(json.dumps({key: os.environ.get(key) for key in (
        "QUANT_RELEASE_IMAGE", "QUANT_RELEASE_REVISION", "QUANT_RELEASE_IMAGE_ID", "QUANT_RELEASE_INPUT_DIGEST"
    )}))
    raise SystemExit(int(os.environ.get("MOCK_UP_EXIT", "0")))
if sys.argv[1:5] == ["compose", "exec", "-T", "quant-paper"]:
    if sys.argv[5:] == ["cat", "/app/data/runtime/runtime-status.json"]:
        print(json.dumps({"synthetic": True}))
        raise SystemExit(int(os.environ.get("MOCK_STATUS_EXIT", "0")))
    if sys.argv[5:8] == ["python3", "-m", "scripts.check_runtime_health"]:
        raise SystemExit(int(os.environ.get("MOCK_HEALTH_EXIT", "0")))
if sys.argv[1:2] == ["tag"]:
    raise SystemExit(int(os.environ.get("MOCK_TAG_EXIT", "0")))
raise SystemExit("unexpected Docker operation in test")
''')
        docker.chmod(0o755)
        self.env.update(PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
                        MOCK_DOCKER_LOG=str(self.root / "docker.jsonl"), DEPLOY_SERVICES="quant-paper",
                        GITHUB_OUTPUT=str(self.root / "github-output"),
                        QUANT_RELEASE_REVISION=REVISION, QUANT_RELEASE_REPOSITORY="fixture/repository")
        self.previous_report = Path(self.env["HOME"]) / (
            ".local/state/quant-system/quant-paper-last-good-deploy.release.json"
        )
        self.previous_report.parent.mkdir(parents=True)
        self.previous_report.write_text(json.dumps({"paper_accepted": True, "release": {
            "revision": PREVIOUS_REVISION, "image_id": IMAGE_ID, "input_digest": INPUT_DIGEST,
        }}))

    def docker_calls(self):
        path = self.root / "docker.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def deploy(self, *args):
        return self.command(["bash", "scripts/deploy.sh", *args])

    def test_preflight_never_restarts_and_success_waits_for_health(self):
        self.paper_fixture()
        self.assertEqual(self.deploy("--check", "quant-paper").returncode, 0)
        self.assertEqual(self.docker_calls(), [["compose", "config", "--quiet"],
                                               ["compose", "config", "--format", "json"]])
        self.assertEqual(self.deploy("quant-paper").returncode, 0)
        calls = self.docker_calls()
        self.assertIn(["compose", "up", "-d", "--no-deps", "--no-build",
                       "--force-recreate", "--wait", "--wait-timeout", "120", "quant-paper"], calls)
        self.assertIn(["compose", "exec", "-T", "quant-paper", "python3", "-m", "scripts.check_runtime_health",
                       "--require", "ready", "--expected-revision", REVISION, "--expected-image-id", IMAGE_ID,
                       "--expected-input-digest", INPUT_DIGEST], calls)
        self.assertEqual(json.loads((self.root / "up-environment.json").read_text()), {
            "QUANT_RELEASE_IMAGE": IMAGE_ID, "QUANT_RELEASE_IMAGE_ID": IMAGE_ID,
            "QUANT_RELEASE_REVISION": REVISION, "QUANT_RELEASE_INPUT_DIGEST": INPUT_DIGEST,
        })
        report = json.loads((self.root / ".cache/release/postflight.json").read_text())
        self.assertTrue(report["paper_accepted"])
        releases = [json.loads(line) for line in (self.root / "release-calls.jsonl").read_text().splitlines()]
        self.assertEqual([row["stage"] for row in releases], ["preflight", "postflight"])
        self.assertEqual(releases[-1]["image"], IMAGE_ID)

    def test_release_preflight_failure_blocks_before_restart(self):
        self.paper_fixture()
        self.env["MOCK_RELEASE_PREFLIGHT_EXIT"] = "29"
        self.assertEqual(self.deploy("quant-paper").returncode, 29)
        self.assertFalse(any(call[:2] == ["compose", "up"] for call in self.docker_calls()))
        self.assertFalse((self.root / ".cache/release/postflight.json").exists())
        self.assertFalse(Path(self.env["GITHUB_OUTPUT"]).exists())

    def test_post_health_and_acceptance_failures_propagate(self):
        self.paper_fixture()
        for variable, code in (("MOCK_HEALTH_EXIT", 31), ("MOCK_STATUS_EXIT", 37),
                               ("MOCK_RELEASE_POSTFLIGHT_EXIT", 41)):
            with self.subTest(variable=variable):
                self.env[variable] = str(code)
                self.assertEqual(self.deploy("quant-paper").returncode, code)
                self.assertTrue(any(call[:2] == ["compose", "up"] for call in self.docker_calls()))
                self.assertFalse((self.root / ".cache/release/postflight.json").exists())
                self.env.pop(variable)

    def test_missing_release_revision_blocks_before_restart(self):
        self.paper_fixture()
        self.env.pop("QUANT_RELEASE_REVISION")
        self.assertNotEqual(self.deploy("quant-paper").returncode, 0)
        self.assertFalse(any(call[:2] == ["compose", "up"] for call in self.docker_calls()))

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

    def test_missing_state_blocks_before_service_change(self):
        self.paper_fixture()
        (self.state / "orders.sqlite").unlink()
        result = self.deploy("quant-paper")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(any(call[:2] == ["compose", "up"] for call in self.docker_calls()))

    def test_missing_mount_and_wrong_identity_block_service_change(self):
        self.paper_fixture()
        service = self.compose["services"]["quant-paper"]
        original = service["volumes"]
        service["volumes"] = []
        self.compose_path.write_text(json.dumps(self.compose))
        self.assertEqual(self.deploy("quant-paper").returncode, 1)
        service["volumes"] = original
        service["environment"]["RISK_SCOPE"] = "wrong-scope"
        self.compose_path.write_text(json.dumps(self.compose))
        self.assertEqual(self.deploy("quant-paper").returncode, 1)
        self.assertFalse(any(call[:2] == ["compose", "up"] for call in self.docker_calls()))

    def test_unsafe_state_mounts_block_service_change(self):
        self.paper_fixture()
        original = json.dumps(self.compose)
        for defect in ("read_only", "auto_create", "nested", "checkout", "journal_outside", "empty_scope"):
            with self.subTest(defect=defect):
                config = json.loads(original)
                service = config["services"]["quant-paper"]
                mount = service["volumes"][0]
                if defect == "read_only":
                    mount["read_only"] = True
                elif defect == "auto_create":
                    mount["bind"]["create_host_path"] = True
                elif defect == "nested":
                    service["volumes"].append({"target": "/app/data/runtime/orders.sqlite"})
                elif defect == "checkout":
                    mount["source"] = str(self.root / "data/runtime")
                elif defect == "journal_outside":
                    service["environment"]["ORDER_JOURNAL_PATH"] = "/tmp/orders.sqlite"
                else:
                    service["environment"]["RISK_SCOPE"] = ""
                self.compose_path.write_text(json.dumps(config))
                self.assertEqual(self.deploy("quant-paper").returncode, 1)
        self.assertFalse(any(call[:2] == ["compose", "up"] for call in self.docker_calls()))

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
        self.assertEqual(Path(self.env["GITHUB_OUTPUT"]).read_text(), "service_change_started=true\n")

    def rollback(self, available="true"):
        block = workflow_step(ROOT / ".github/workflows/deploy.yml", "Rollback on failure")
        block = block.replace("${{ steps.changes.outputs.prev_commit }}", PREVIOUS_REVISION)
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

    def test_rollback_requires_accepted_matching_release_evidence(self):
        self.paper_fixture()
        original = self.previous_report.read_text()
        self.previous_report.unlink()
        self.assertEqual(self.rollback().returncode, 1)
        self.assertEqual(self.docker_calls(), [])
        for defect in ("not_accepted", "input_changed", "image_changed", "revision_changed"):
            with self.subTest(defect=defect):
                report = json.loads(original)
                if defect == "not_accepted":
                    report["paper_accepted"] = False
                else:
                    field = {"input_changed": "input_digest", "image_changed": "image_id",
                             "revision_changed": "revision"}[defect]
                    report["release"][field] = "changed"
                self.previous_report.write_text(json.dumps(report))
                self.assertEqual(self.rollback().returncode, 1)
                self.assertFalse(any(call[:2] == ["compose", "up"] for call in self.docker_calls()))

    def test_workflow_references_and_paper_only_build(self):
        deploy = ROOT / ".github/workflows/deploy.yml"
        for name in ["scripts/deploy.sh", "scripts/check_deploy_scope.py", "scripts/check_ci_security.py",
                     "scripts/check_runtime_health.py", "scripts/check_release_acceptance.py"]:
            self.assertTrue((ROOT / name).is_file())
            self.assertFalse((ROOT / name).is_symlink())
        self.assertEqual(workflow_step(deploy, "Docker build"), 'docker compose build --no-cache "$DEPLOY_SERVICES"')
        self.assertIn("DEPLOY_SERVICES: quant-paper", deploy.read_text())
        self.assertNotIn("needs_rebuild", deploy.read_text())
        self.assertNotIn("git diff --name-only \"$PREV\" HEAD || true", deploy.read_text())
        self.assertIn("inputs.confirm_deploy && github.ref == 'refs/heads/master'", deploy.read_text())
        self.assertIn("steps.rolling_deploy.outputs.service_change_started == 'true'", deploy.read_text())

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
