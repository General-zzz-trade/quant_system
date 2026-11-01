import contextlib
import io
import unittest
from unittest.mock import patch

from runner.alpha_cli import parse_args


class CliSafetyTests(unittest.TestCase):
    def test_default_and_explicit_dry_run_never_enable_execution(self):
        with patch.dict("os.environ", {"RISK_SCOPE": "okx-live-a"}):
            self.assertTrue(parse_args([]).dry_run)
            self.assertTrue(parse_args(["--dry-run"]).dry_run)

    def test_execution_requires_explicit_switch_and_scope(self):
        with patch.dict("os.environ", {"RISK_SCOPE": ""}), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                parse_args(["--enable-execution"])
            self.assertEqual(ctx.exception.code, 2)
        with patch.dict("os.environ", {"RISK_SCOPE": "okx-demo-a"}):
            self.assertFalse(parse_args(["--enable-execution"]).dry_run)

    def test_conflicting_modes_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--dry-run", "--enable-execution"])
