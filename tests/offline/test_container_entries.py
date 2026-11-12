"""Exercise packaged operational entry points, including inside the CI image."""
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ContainerEntryTests(unittest.TestCase):
    def test_paper_image_default_uses_the_same_shadow_manifest_entry(self):
        from scripts.check_paper_runtime import SHADOW_COMMAND
        paper = (ROOT / "Dockerfile").read_text().split("FROM base AS paper\n", 1)[1]
        paper = paper.split("FROM paper AS live\n", 1)[0]
        commands = [json.loads(line[4:]) for line in paper.splitlines() if line.startswith("CMD ")]
        self.assertEqual(commands, [SHADOW_COMMAND])
        self.assertTrue((ROOT / "config/shadow.json").is_file())

    def test_shadow_help_requires_no_model_account_or_native_extension(self):
        self.assertTrue((ROOT / "runner/shadow_main.py").is_file())
        result = subprocess.run([sys.executable, "-m", "runner.shadow_main", "--help"], cwd=ROOT,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--manifest", result.stdout)
        self.assertIn("--dry-run", result.stdout)

    def test_operational_clis_are_shipped_and_help_needs_no_account(self):
        for name in (
            "check_runtime_health", "check_release_acceptance", "check_execution_runtime",
            "migrate_runtime_state", "risk_group_backup", "account_ledger_state", "portfolio_risk_state",
        ):
            with self.subTest(entry=name):
                path = ROOT / "scripts" / (name + ".py")
                self.assertTrue(path.is_file(), "Operational entry is absent from the build context")
                self.assertFalse(path.is_symlink())
                result = subprocess.run(
                    [sys.executable, "-m", "scripts." + name, "--help"], cwd=ROOT,
                    capture_output=True, text=True, timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
