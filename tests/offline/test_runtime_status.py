"""Readiness must depend on fresh runtime evidence, not successful imports."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.check_runtime_health import check_runtime_health
from shared.runtime_status import RuntimeStatusWriter, STATUS_NAME, evaluate_status, read_json, strict_json


class RuntimeStatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / STATUS_NAME
        self.now = 1_800_000_000.0
        self.state_id = "a" * 32
        self.release = {"revision": "b" * 40, "image_id": "sha256:" + "c" * 64, "input_digest": "d" * 64}
        (self.root / "runtime-state.json").write_text(json.dumps({
            "version": 1, "scope": "paper", "state_id": self.state_id,
        }))

    def data(self):
        now_ms = int(self.now * 1000)
        return {"BTCUSDT": {
            "ready": True, "status": "ready", "mode": "live", "interval_ms": 3_600_000,
            "bar_ts_ms": now_ms - 3_600_000, "received_at_ms": now_ms - 1_000,
            "valid_warmup_bars": 120, "required_warmup_bars": 120,
            "open_interest": 10.5, "ls_ratio": 1.2,
            "oi_event_ts_ms": now_ms - 3_630_000, "ls_event_ts_ms": now_ms - 3_630_000,
            "external_conditions": {"batch_inputs": {"ready": True, "reason": "sensitive account detail"}},
            "reason": "sensitive exchange response", "account_id": "must-not-be-published",
        }}

    def execution(self):
        return {"modules": {"BTCUSDT": True},
                "reconciliations": {"BTCUSDT": {"ready": True, "checked_at": self.now - 2}},
                "risk": {"ready": True, "age_seconds": 10, "captured_at": self.now, "max_age_seconds": 300,
                         "reservations": [{"intent_id": "secret-id"}]},
                "wallet": {"ready": True, "checked_at": self.now - 1,
                           "verified_through_ms": int((self.now - 1) * 1000),
                           "max_age_seconds": 120, "actual_balance": "1234"},
                "journal": {"unresolved": 0, "unknown": 0, "unconsumed": 0}}

    def writer(self, *, mode="paper", runner_keys=None, **kwargs):
        return RuntimeStatusWriter(self.path, scope="paper", state_id=self.state_id, mode=mode,
                                   runner_keys=runner_keys or ["BTCUSDT"], release=self.release,
                                   clock=lambda: self.now, **kwargs)

    def write(self, *, mode="paper"):
        self.writer(mode=mode).publish("running", data=self.data(),
                                       execution=self.execution() if mode == "execution" else None)
        return read_json(self.path)

    def test_fresh_paper_is_ready_but_never_execution_eligible(self):
        status = self.write()
        result = check_runtime_health(self.root, scope="paper", expected_release=self.release, now=self.now)
        self.assertTrue(result["live"])
        self.assertTrue(result["ready"])
        self.assertFalse(result["execution_eligible"])
        self.assertEqual(status["execution_runner_keys"], [])

    def test_starting_and_stopped_cannot_pass_readiness(self):
        writer = self.writer()
        result = writer.publish("starting")
        self.assertTrue(result["live"])
        self.assertFalse(result["ready"])
        for phase in ("stopping", "stopped"):
            result = writer.publish(phase, data=self.data())
            self.assertFalse(result["live"])
            self.assertFalse(result["ready"])

    def test_no_heartbeat_worker_keeps_a_hung_loop_healthy(self):
        status = self.write()
        self.assertTrue(evaluate_status(status, now=self.now + 30)["live"])
        result = evaluate_status(status, now=self.now + 31)
        self.assertFalse(result["live"])
        self.assertFalse(result["ready"])
        self.assertIn("heartbeat_stale_or_future", result["reasons"])

    def test_future_or_backwards_heartbeat_is_not_healthy(self):
        status = self.write()
        status["heartbeat_at"] += 1
        self.assertFalse(evaluate_status(status, now=self.now)["ready"])
        status["heartbeat_at"] = status["started_at"] - 1
        with self.assertRaises(ValueError):
            evaluate_status(status, now=self.now)

    def test_missing_or_extra_runner_evidence_blocks_readiness(self):
        status = self.write()
        status["data"] = {}
        self.assertFalse(evaluate_status(status, now=self.now)["ready"])
        status["data"] = self.data()
        status["runner_keys"] += ["ETHUSDT"]
        self.assertFalse(evaluate_status(status, now=self.now)["ready"])

    def test_advertised_ready_flag_cannot_hide_bad_data(self):
        original = self.write()
        mutations = [
            {"mode": "warmup"}, {"status": "error"}, {"ready": False},
            {"valid_warmup_bars": 119}, {"bar_ts_ms": int(self.now * 1000) - 3_900_001},
            {"bar_ts_ms": int(self.now * 1000) + 1},
            {"received_at_ms": int(self.now * 1000) - 120_001},
            {"received_at_ms": int(self.now * 1000) + 1},
            {"oi_event_ts_ms": int(self.now * 1000)},
            {"ls_event_ts_ms": int(self.now * 1000) - 7_500_001},
            {"external_conditions": {"batch_inputs": {"ready": False}}},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                status = deepcopy(original)
                status.update(ready=True, execution_eligible=True)
                status["data"]["BTCUSDT"].update(mutation)
                result = evaluate_status(status, now=self.now)
                self.assertFalse(result["ready"])
                self.assertFalse(result["execution_eligible"])

    def test_nonfinite_nonpositive_and_malformed_observations_are_rejected(self):
        original = self.write()
        for field, value in (("open_interest", float("nan")), ("ls_ratio", 0),
                             ("ready", "true"), ("interval_ms", True), ("received_at_ms", "123")):
            with self.subTest(field=field):
                status = deepcopy(original)
                status["data"]["BTCUSDT"][field] = value
                with self.assertRaises(ValueError):
                    evaluate_status(status, now=self.now)

    def test_execution_requires_every_fresh_source(self):
        original = self.write(mode="execution")
        self.assertTrue(evaluate_status(original, now=self.now)["execution_eligible"])
        mutations = [
            ("modules", {"BTCUSDT": False}),
            ("modules", {}),
            ("reconciliations", {"BTCUSDT": {"ready": False, "checked_at": self.now}}),
            ("reconciliations", {"BTCUSDT": {"ready": True, "checked_at": self.now - 121}}),
            ("reconciliations", {"BTCUSDT": {"ready": True, "checked_at": self.now + 1}}),
            ("risk", {**self.execution()["risk"], "ready": False, "age_seconds": 0}),
            ("risk", {**self.execution()["risk"], "ready": True, "age_seconds": 301}),
            ("wallet", {**self.execution()["wallet"], "ready": False}),
            ("wallet", {**self.execution()["wallet"], "ready": True, "checked_at": self.now - 121,
                        "verified_through_ms": int(self.now * 1000)}),
            ("wallet", {**self.execution()["wallet"], "ready": True, "checked_at": self.now,
                        "verified_through_ms": int((self.now - 121) * 1000)}),
        ]
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                status = deepcopy(original)
                status["execution"][field] = value
                result = evaluate_status(status, now=self.now)
                self.assertTrue(result["live"])
                self.assertFalse(result["ready"])
                self.assertFalse(result["execution_eligible"])

    def test_unresolved_unknown_and_unconsumed_each_block_execution(self):
        original = self.write(mode="execution")
        for key in ("unresolved", "unknown", "unconsumed"):
            status = deepcopy(original)
            status["execution"]["journal"][key] = 1
            self.assertFalse(evaluate_status(status, now=self.now)["execution_eligible"])

    def test_evidence_ages_independently_while_heartbeat_is_fresh(self):
        status = self.write(mode="execution")
        status["execution"]["risk"]["age_seconds"] = 290
        self.assertFalse(evaluate_status(status, now=self.now + 20)["execution_eligible"])

    def test_stricter_wallet_and_risk_expiry_are_preserved(self):
        original = self.write(mode="execution")
        for key, limit in (("wallet", 5), ("risk", 15)):
            status = deepcopy(original)
            status["execution"][key]["max_age_seconds"] = limit
            self.assertTrue(evaluate_status(status, now=self.now)["execution_eligible"])
            self.assertFalse(evaluate_status(status, now=self.now + 6)["execution_eligible"])

    def test_slow_snapshot_collection_does_not_reset_risk_age(self):
        status = self.write(mode="execution")
        status["execution"]["risk"].update(age_seconds=290, captured_at=self.now - 11)
        self.assertFalse(evaluate_status(status, now=self.now)["execution_eligible"])
        status = self.write(mode="execution")
        status["execution"]["wallet"]["checked_at"] = self.now - 110
        self.assertFalse(evaluate_status(status, now=self.now + 20)["execution_eligible"])

    def test_signal_only_runner_does_not_need_an_execution_adapter(self):
        writer = self.writer(mode="execution", runner_keys=["BTCUSDT", "BTCUSDT_4h"],
                             execution_runner_keys=["BTCUSDT"])
        data = self.data()
        data["BTCUSDT_4h"] = deepcopy(data["BTCUSDT"])
        result = writer.publish("running", data=data, execution=self.execution())
        self.assertTrue(result["execution_eligible"])
        status = read_json(self.path)
        status["execution_runner_keys"] = []
        with self.assertRaises(ValueError):
            evaluate_status(status, now=self.now)

    def test_execution_startup_without_account_evidence_is_unready(self):
        result = self.writer(mode="execution").publish("starting")
        self.assertTrue(result["live"])
        self.assertFalse(result["ready"])
        self.assertFalse(result["execution_eligible"])

    def test_paper_cannot_assert_execution_evidence(self):
        with self.assertRaises(ValueError):
            self.writer().publish("running", data=self.data(), execution=self.execution())

    def test_status_strips_account_values_order_identifiers_and_free_text(self):
        self.write(mode="execution")
        raw = self.path.read_text()
        for text in ("sensitive", "account_id", "actual_balance", "secret-id", "intent_id", "reservations"):
            self.assertNotIn(text, raw)

    def test_wrong_scope_state_or_release_is_rejected(self):
        status = self.write()
        for kwargs in ({"expected_scope": "other"}, {"expected_state_id": "f" * 32},
                       {"expected_release": {**self.release, "revision": "f" * 40}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate_status(status, now=self.now, **kwargs)
        for field in self.release:
            malformed = deepcopy(status)
            del malformed["release"][field]
            with self.assertRaises(ValueError):
                evaluate_status(malformed, now=self.now)

    def test_missing_manifest_identity_cannot_disable_identity_check(self):
        self.write()
        (self.root / "runtime-state.json").write_text('{"version":1,"scope":"paper"}')
        with self.assertRaises(ValueError):
            check_runtime_health(self.root, scope="paper", now=self.now)

    def test_missing_and_duplicate_json_nonfinite_and_overflow_fail_closed(self):
        with self.assertRaises(FileNotFoundError):
            read_json(self.path)
        for raw in ('{"ready":true,"ready":false}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '[]'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                strict_json(raw)

    def test_symlinks_special_and_oversized_files_are_rejected(self):
        self.path.symlink_to(self.root / "runtime-state.json")
        with self.assertRaises(ValueError):
            read_json(self.path)
        self.path.unlink()
        os.mkfifo(self.path)
        with self.assertRaises(ValueError):
            read_json(self.path)
        self.path.unlink()
        self.path.write_bytes(b" " * 262_145)
        with self.assertRaises(ValueError):
            read_json(self.path)

    def test_failed_atomic_replace_retains_previous_observation_without_refresh(self):
        writer = self.writer()
        writer.publish("running", data=self.data())
        original = self.path.read_bytes()
        self.now += 31
        with patch("shared.runtime_status.os.replace", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                writer.publish("running", data=self.data())
        self.assertEqual(self.path.read_bytes(), original)
        self.assertFalse(evaluate_status(read_json(self.path), now=self.now)["ready"])
        self.assertEqual(list(self.root.glob(".runtime-status-*")), [])

    def test_invalid_publication_does_not_replace_valid_json(self):
        writer = self.writer()
        writer.publish("running", data=self.data())
        original = self.path.read_bytes()
        invalid = self.data()
        invalid["BTCUSDT"]["ls_ratio"] = float("nan")
        with self.assertRaises(ValueError):
            writer.publish("running", data=invalid)
        self.assertEqual(self.path.read_bytes(), original)

    def test_dead_process_cannot_pass_health(self):
        status = self.write()
        with patch("shared.runtime_status.os.kill", side_effect=ProcessLookupError):
            result = evaluate_status(status, now=self.now, check_pid=True)
        self.assertFalse(result["live"])
        self.assertFalse(result["ready"])

    def test_reused_pid_cannot_reuse_previous_process_health(self):
        original = self.write()
        for key, value in (("process_start_ticks", original["process_start_ticks"] + 1),
                           ("system_boot_id", "f" * 32), ("pid_namespace", "pid:[1]")):
            with self.subTest(key=key):
                status = deepcopy(original)
                status[key] = value
                result = evaluate_status(status, now=self.now, check_pid=True)
                self.assertFalse(result["live"])
                self.assertFalse(result["ready"])

    def test_cli_exit_codes_for_good_missing_stale_and_execution_required(self):
        self.now = time.time()
        self.write()
        command = [sys.executable, "-m", "scripts.check_runtime_health", "--state-dir", str(self.root),
                   "--scope", "paper"]
        good = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(good.returncode, 0, good.stderr + good.stdout)
        execution = subprocess.run(command + ["--require", "execution"], capture_output=True, text=True, check=False)
        self.assertEqual(execution.returncode, 1)
        self.now -= 31
        self.write()
        stale = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(stale.returncode, 1)
        self.path.unlink()
        missing = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(missing.returncode, 1)
        self.assertFalse(json.loads(missing.stdout)["ready"])

    def test_cli_binds_release_identity_from_compose_environment(self):
        self.now = time.time()
        self.write()
        command = [sys.executable, "-m", "scripts.check_runtime_health", "--state-dir", str(self.root),
                   "--scope", "paper"]
        env = {**os.environ, "QUANT_RELEASE_REVISION": self.release["revision"],
               "QUANT_RELEASE_IMAGE_ID": self.release["image_id"],
               "QUANT_RELEASE_INPUT_DIGEST": self.release["input_digest"]}
        self.assertEqual(subprocess.run(command, env=env, capture_output=True).returncode, 0)
        env["QUANT_RELEASE_REVISION"] = "f" * 40
        self.assertEqual(subprocess.run(command, env=env, capture_output=True).returncode, 1)


if __name__ == "__main__":
    unittest.main()
