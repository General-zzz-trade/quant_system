"""Exact release acceptance rejects stale, substituted and incomplete evidence."""
from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shared.release_acceptance import fetch_ci_runs, release_input_manifest, validate_release
from shared.runtime_state import initialize_runtime_state


class ReleaseAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "checkout"
        self.project.mkdir()
        self.models = self.project / "models_v8"
        self.model = self.models / "BTC_model"
        self.model.mkdir(parents=True)
        self.data = self.project / "data_files"
        self.data.mkdir()
        self.model_config = {"horizon_models": [{"lgbm": "model.pkl", "ridge": "ridge.pkl"}],
                             "regime_models": {"bear": {"1": {"lgbm": "bear.pkl"}}}}
        (self.model / "config.json").write_text(json.dumps(self.model_config))
        for name in ("model.pkl", "ridge.pkl", "bear.pkl", "model.json", "model.pkl.sig"):
            (self.model / name).write_bytes(b"offline fixture artifact; never deserialized")
        for suffix in ("_1h.csv", "_open_interest.csv", "_ls_ratio.csv"):
            (self.data / ("BTCUSDT" + suffix)).write_text("timestamp,value\n1000000,1\n")
        (self.data / "auxiliary.csv").write_text("timestamp,value\n1000000,2\n")
        self.state = self.root / "state"
        self.scope = "paper-fixture"
        initialize_runtime_state(self.state, self.scope)
        self.identity = json.loads((self.state / "runtime-state.json").read_text())
        self.revision = "a" * 40
        self.repository = "fixture/repo"
        self.run = {"id": 42, "head_sha": self.revision, "head_branch": "master", "event": "push",
                    "path": ".github/workflows/ci.yml", "status": "completed", "conclusion": "success",
                    "repository": {"full_name": self.repository}}
        self.image = {"Id": "sha256:" + "b" * 64,
                      "Config": {"Labels": {"org.opencontainers.image.revision": self.revision}}}
        self.symbol_config = {"BTCUSDT": {"model_dir": "BTC_model"}}
        self.config = {"services": {"quant-paper": {
            "image": "quant-paper:latest",
            "command": ["python3", "-m", "scripts.run_bybit_alpha", "--symbols", "BTCUSDT", "--ws", "--dry-run"],
            "environment": {"RUNTIME_STATE_DIR": "/app/data/runtime", "RISK_SCOPE": self.scope,
                            "ORDER_JOURNAL_PATH": "/app/data/runtime/orders.sqlite",
                            "DRAWDOWN_STATE_PATH": "/app/data/runtime/drawdown.sqlite"},
            "volumes": [{"type": "bind", "source": str(self.state), "target": "/app/data/runtime"},
                        {"type": "bind", "source": str(self.models), "target": "/app/models_v8", "read_only": True},
                        {"type": "bind", "source": str(self.data), "target": "/app/data_files", "read_only": True}],
        }}}

    def validate(self, **kwargs):
        arguments = dict(repository=self.repository, revision=self.revision, config=self.config,
                         project_root=self.project, image_inspect=self.image, ci_runs=[self.run],
                         symbol_config=self.symbol_config, now=10000)
        arguments.update(kwargs)
        return validate_release(**arguments)

    def health(self, receipt):
        return {"version": 1, "scope": self.scope, "state_id": self.identity["state_id"],
                "boot_id": "c" * 32, "pid": os.getpid(), "mode": "paper", "phase": "running",
                "process_start_ticks": 123, "system_boot_id": "e" * 32, "pid_namespace": "pid:[1]",
                "runner_keys": ["BTCUSDT"], "execution_runner_keys": [], "started_at": 10000,
                "heartbeat_at": 10000, "release": receipt["release"], "execution": None,
                "data": {"BTCUSDT": {"ready": True, "mode": "live", "status": "ready",
                                     "required_warmup_bars": 100, "valid_warmup_bars": 100,
                                     "interval_ms": 3_600_000, "bar_ts_ms": 7_200_000,
                                     "received_at_ms": 10_000_000, "open_interest": 100, "ls_ratio": 1,
                                     "oi_event_ts_ms": 7_200_000, "ls_event_ts_ms": 7_200_000,
                                     "external_conditions": {}}}}

    def test_preflight_binds_ci_image_state_and_complete_input_manifest(self):
        result = self.validate()
        self.assertEqual(result["stage"], "preflight")
        self.assertFalse(result["paper_accepted"])
        self.assertFalse(result["execution_eligible"])
        self.assertFalse(result["live_acceptance"])
        self.assertEqual(result["ci"]["id"], 42)
        self.assertEqual(result["runtime"]["state_id"], self.identity["state_id"])
        names = {row["path"] for row in result["inputs"]["files"]}
        self.assertIn("models/BTC_model/bear.pkl", names)
        self.assertIn("models/BTC_model/model.json", names)
        self.assertIn("models/BTC_model/model.pkl.sig", names)
        self.assertIn("data/auxiliary.csv", names)

    def test_exact_ci_revision_branch_workflow_event_and_repository_are_required(self):
        for key, value in (("head_sha", "d" * 40), ("head_branch", "feature"),
                           ("path", ".github/workflows/other.yml"), ("event", "pull_request"),
                           ("status", "in_progress"), ("conclusion", "failure"),
                           ("repository", {"full_name": "other/repo"}), ("repository", None)):
            with self.subTest(field=key):
                run = dict(self.run, **{key: value})
                with self.assertRaises(ValueError):
                    self.validate(ci_runs=[run])

    def test_later_failed_or_running_ci_does_not_reuse_older_success(self):
        for status in ("in_progress", "completed"):
            with self.assertRaisesRegex(ValueError, "Latest"):
                self.validate(ci_runs=[self.run, dict(self.run, id=43, status=status, conclusion="failure")])

    def test_full_sha_and_immutable_image_revision_required(self):
        with self.assertRaises(ValueError):
            self.validate(revision="a" * 7)
        for image in ({"Id": "quant-paper:latest", "Config": self.image["Config"]},
                      {"Id": self.image["Id"], "Config": {"Labels": {}}},
                      {"Id": self.image["Id"], "Config": {"Labels": None}},
                      {"Id": self.image["Id"], "Config": None}):
            with self.assertRaises(ValueError):
                self.validate(image_inspect=image)

    def test_missing_declared_optional_model_and_required_csv_are_blocked(self):
        for path in (self.model / "ridge.pkl", self.model / "bear.pkl", self.data / "BTCUSDT_ls_ratio.csv"):
            content = path.read_bytes()
            path.unlink()
            with self.assertRaises((ValueError, OSError)):
                self.validate()
            path.write_bytes(content)

    def test_empty_and_symlinked_artifacts_are_blocked(self):
        artifact = self.model / "ridge.pkl"
        artifact.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.validate()
        artifact.unlink()
        artifact.symlink_to(self.model / "model.pkl")
        with self.assertRaisesRegex(ValueError, "symbolic"):
            self.validate()

    def test_directory_symlink_and_traversing_model_reference_are_blocked(self):
        link = self.root / "models-link"
        link.symlink_to(self.models, target_is_directory=True)
        self.config["services"]["quant-paper"]["volumes"][1]["source"] = str(link)
        with self.assertRaises(ValueError):
            self.validate()
        self.config["services"]["quant-paper"]["volumes"][1]["source"] = str(self.models)
        self.model_config["horizon_models"][0]["lgbm"] = "../outside.pkl"
        (self.model / "config.json").write_text(json.dumps(self.model_config))
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.validate()

    def test_unknown_runner_and_execution_command_are_blocked(self):
        command = self.config["services"]["quant-paper"]["command"]
        command[4] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.validate()
        command[4] = "BTCUSDT"
        command.append("--enable-execution")
        with self.assertRaises(ValueError):
            self.validate()

    def test_writable_and_shadowed_artifact_mounts_are_blocked(self):
        mounts = self.config["services"]["quant-paper"]["volumes"]
        mounts[1]["read_only"] = False
        with self.assertRaises(ValueError):
            self.validate()
        mounts[1]["read_only"] = True
        mounts.append({"target": "/app/data_files/BTCUSDT_1h.csv", "source": str(self.root)})
        with self.assertRaisesRegex(ValueError, "shadowed"):
            self.validate()

    def test_manifest_is_content_based_and_includes_auxiliary_input_changes(self):
        first = release_input_manifest(self.config, symbol_config=self.symbol_config)
        os.utime(self.data / "auxiliary.csv", (100, 100))
        self.assertEqual(first, release_input_manifest(self.config, symbol_config=self.symbol_config))
        (self.data / "auxiliary.csv").write_text("timestamp,value\n1000000,3\n")
        self.assertNotEqual(first["sha256"], release_input_manifest(
            self.config, symbol_config=self.symbol_config)["sha256"])

    def test_required_input_cannot_disappear_between_check_and_manifest(self):
        from shared.release_acceptance import _tree
        for path, selected_prefix in ((self.model / "ridge.pkl", "models/BTC_model"),
                                      (self.data / "BTCUSDT_ls_ratio.csv", "data")):
            with self.subTest(path=path.name):
                content = path.read_bytes()

                def removed(directory, prefix):
                    if prefix == selected_prefix:
                        path.unlink()
                    return _tree(directory, prefix)

                try:
                    with patch("shared.release_acceptance._tree", side_effect=removed):
                        with self.assertRaisesRegex(ValueError, "changed during"):
                            self.validate()
                finally:
                    path.write_bytes(content)

    def test_source_overrides_cannot_escape_the_image_revision(self):
        original = copy.deepcopy(self.config)
        for field, value in (("entrypoint", ["sh", "-c"]), ("working_dir", "/other"),
                             ("tmpfs", ["/app"]), ("volumes_from", ["other-container"]),
                             ("configs", [{"source": "override", "target": "/app/runner/alpha_main.py"}])):
            with self.subTest(field=field):
                config = copy.deepcopy(original)
                config["services"]["quant-paper"][field] = value
                with self.assertRaises(ValueError):
                    self.validate(config=config)
        for target in ("/app", "/app/runner", "/app/shared", "/app/runner/alpha_main.py"):
            config = copy.deepcopy(original)
            config["services"]["quant-paper"]["volumes"].append(
                {"type": "bind", "source": str(self.project), "target": target})
            with self.assertRaisesRegex(ValueError, "replace image"):
                self.validate(config=config)

    def test_postflight_accepts_fresh_matching_paper_without_claiming_execution(self):
        before = self.validate()
        result = self.validate(previous_report=before, health=self.health(before))
        self.assertTrue(result["paper_accepted"])
        self.assertEqual(result["stage"], "postflight")
        self.assertFalse(result["live_acceptance"])
        self.assertFalse(result["execution_eligible"])

    def test_preflight_cannot_substitute_for_postflight_health(self):
        before = self.validate()
        with self.assertRaisesRegex(ValueError, "both"):
            self.validate(previous_report=before)
        with self.assertRaisesRegex(ValueError, "both"):
            self.validate(health=self.health(before))

    def test_artifact_or_image_change_after_preflight_fails(self):
        before = self.validate()
        image = copy.deepcopy(self.image)
        image["Id"] = "sha256:" + "d" * 64
        with self.assertRaisesRegex(ValueError, "changed"):
            self.validate(previous_report=before, health=self.health(before), image_inspect=image)
        (self.data / "auxiliary.csv").write_text("changed bytes")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.validate(previous_report=before, health=self.health(before))

    def test_stale_or_forged_ready_health_fails(self):
        before = self.validate()
        for mutation in (lambda value: value.update(heartbeat_at=9900, ready=True),
                         lambda value: value.update(phase="stopped", ready=True),
                         lambda value: value["data"]["BTCUSDT"].update(ready=False),
                         lambda value: value["release"].update(revision="d" * 40),
                         lambda value: value.update(scope="different-account"),
                         lambda value: value.update(state_id="d" * 32)):
            status = copy.deepcopy(self.health(before))
            mutation(status)
            with self.assertRaises(ValueError):
                self.validate(previous_report=before, health=status)

    def test_partial_healthy_runner_set_is_rejected(self):
        before = self.validate()
        status = self.health(before)
        status["runner_keys"] = ["ETHUSDT"]
        status["data"] = {"ETHUSDT": status["data"]["BTCUSDT"]}
        with self.assertRaises(ValueError):
            self.validate(previous_report=before, health=status)

    def test_previous_process_cannot_satisfy_postflight(self):
        before = self.validate()
        status = self.health(before)
        status["started_at"] = 9999
        with self.assertRaisesRegex(ValueError, "after this release"):
            self.validate(previous_report=before, health=status)

    def test_invalid_preflight_time_cannot_admit_previous_process(self):
        for timestamp in (float("nan"), float("inf"), -1, 10001, True, "10000"):
            with self.subTest(timestamp=timestamp):
                before = self.validate()
                before["checked_at"] = timestamp
                with self.assertRaises(ValueError):
                    self.validate(previous_report=before, health=self.health(before))
        for timestamp in (float("nan"), float("inf"), -1, True):
            with self.assertRaises(ValueError):
                self.validate(now=timestamp)

    def test_malformed_mount_and_receipt_are_rejected(self):
        before = self.validate()
        with self.assertRaises(ValueError):
            self.validate(previous_report=[], health=self.health(before))
        for mounts in (None, ["not a mount object"]):
            config = copy.deepcopy(self.config)
            config["services"]["quant-paper"]["volumes"] = mounts
            with self.assertRaises(ValueError):
                release_input_manifest(config, symbol_config=self.symbol_config)

    def test_execution_acceptance_explicitly_fails_even_for_valid_paper(self):
        with self.assertRaisesRegex(ValueError, "paper only"):
            self.validate(require_execution=True)

    def test_cli_fetches_ci_and_inspects_image_without_docker_mutation(self):
        from scripts.check_release_acceptance import main
        output = self.root / "receipt.json"
        with (patch("scripts.check_release_acceptance.fetch_ci_runs", return_value=[self.run]) as fetch,
              patch("scripts.check_release_acceptance.subprocess.run") as run,
              patch("strategy.config.SYMBOL_CONFIG", self.symbol_config),
              patch("sys.stdin", io.StringIO(json.dumps(self.config))),
              patch("sys.stdout", io.StringIO())):
            run.return_value.stdout = json.dumps(self.image)
            code = main(["--revision", self.revision, "--repository", self.repository,
                         "--image", "quant-paper:latest", "--project-root", str(self.project),
                         "--output", str(output)])
        self.assertEqual(code, 0)
        self.assertEqual(fetch.call_args.args, (self.repository, self.revision))
        self.assertEqual(run.call_args.args[0][:3], ["docker", "image", "inspect"])
        self.assertFalse(json.loads(output.read_text())["paper_accepted"])

    def test_cli_failure_does_not_publish_acceptance_or_leak_rendered_environment(self):
        from scripts.check_release_acceptance import main
        self.config["services"]["quant-paper"]["environment"]["DUMMY_PRIVATE_VALUE"] = "do-not-log-fixture"
        output = self.root / "receipt.json"
        error = io.StringIO()
        with (patch("scripts.check_release_acceptance.fetch_ci_runs", return_value=[]),
              patch("scripts.check_release_acceptance.subprocess.run") as run,
              patch("sys.stdin", io.StringIO(json.dumps(self.config))), patch("sys.stderr", error)):
            run.return_value.stdout = json.dumps(self.image)
            code = main(["--revision", self.revision, "--repository", self.repository,
                         "--image", "quant-paper:latest", "--output", str(output)])
        self.assertEqual(code, 1)
        self.assertFalse(output.exists())
        self.assertNotIn("do-not-log-fixture", error.getvalue())

    def test_ci_fetch_uses_fixed_github_origin_and_rejects_malformed_identity(self):
        payload = io.BytesIO(json.dumps({"workflow_runs": [self.run]}).encode())
        with patch("urllib.request.urlopen", return_value=payload) as request:
            self.assertEqual(fetch_ci_runs(self.repository, self.revision), [self.run])
        self.assertTrue(request.call_args.args[0].full_url.startswith("https://api.github.com/repos/fixture/repo/"))
        with patch("urllib.request.urlopen") as request:
            with self.assertRaises(ValueError):
                fetch_ci_runs("bad/repo/../../host", self.revision)
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
