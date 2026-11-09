"""Batch model inputs cannot be vouched for by an unrelated live cache."""
from pathlib import Path
import tempfile
import unittest

from runner.data_contract import inspect_batch_inputs


class BatchDataContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.at = 1_800_000_000_000 // 3_600_000 * 3_600_000
        self.now = self.at + 3_600_005
        self.write()

    def write(self, *, oi_age=0, value="100", future_only=False):
        (self.root / "BTCUSDT_1h.csv").write_text(f"open_time,close\n{self.at},100\n{self.at+3600000},101\n")
        observation = self.at + 1 if future_only else self.at - oi_age
        (self.root / "BTCUSDT_open_interest.csv").write_text(
            f"timestamp,sum_open_interest\n{observation},{value}\n")
        (self.root / "BTCUSDT_ls_ratio.csv").write_text(f"timestamp,long_short_ratio\n{self.at},1.2\n")

    def check(self, **kwargs):
        return inspect_batch_inputs("BTCUSDT", self.root, as_of_ms=self.at, now_ms=self.now, **kwargs)

    def test_fresh_causal_inputs_pass_and_forming_candle_is_not_selected(self):
        self.assertTrue(self.check()["ready"])
        self.assertEqual(self.check()["as_of_ms"], self.at)

    def test_stale_and_future_only_observations_block(self):
        self.write(oi_age=3_900_001)
        self.assertFalse(self.check()["ready"])
        self.write(future_only=True)
        self.assertFalse(self.check()["ready"])

    def test_missing_or_unclosed_market_candle_blocks(self):
        (self.root / "BTCUSDT_1h.csv").write_text(f"open_time,close\n{self.at-3600000},100\n")
        self.assertFalse(self.check()["ready"])
        self.write()
        self.now = self.at + 1000
        self.assertFalse(self.check()["ready"])

    def test_nonfinite_values_and_conflicting_observations_block(self):
        for value in ("nan", "inf", "0", "-1"):
            with self.subTest(value=value):
                self.write(value=value)
                self.assertFalse(self.check()["ready"])
        self.write()
        with (self.root / "BTCUSDT_ls_ratio.csv").open("a") as out:
            out.write(f"{self.at},1.8\n")
        self.assertFalse(self.check()["ready"])

    def test_missing_files_and_expired_market_bar_block(self):
        (self.root / "BTCUSDT_ls_ratio.csv").unlink()
        self.assertFalse(self.check()["ready"])
        self.write()
        self.now += 300_000
        self.assertFalse(self.check()["ready"])

    def test_file_change_changes_prediction_cache_identity(self):
        before = self.check()["signature"]
        self.write(value="12345")
        self.assertNotEqual(self.check()["signature"], before)


if __name__ == "__main__":
    unittest.main()
