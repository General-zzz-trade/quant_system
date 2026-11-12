"""Shadow selection and input contracts do not require accounts or model loads."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from runner.run_manifest import load_run_manifest, validate_run_inputs


ROOT = Path(__file__).resolve().parents[2]


class RunManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "shadow.json"
        self.registry = {
            "BTCUSDT": {"model_dir": "BTC_model"},
            "BTCUSDT_4h": {"symbol": "BTCUSDT", "interval": "240", "model_dir": "BTC_4h", "warmup": 300},
            "BTCUSDT_15m": {"symbol": "BTCUSDT", "interval": "15", "model_dir": "BTC_15m"},
        }
        self.config = {"version": 1, "mode": "shadow", "venue": "bybit", "currency": "USDT",
                       "shadow_balance": "10000.00", "models_root": "models", "data_root": "data",
                       "runners": ["BTCUSDT", "BTCUSDT_4h"], "disabled_runners": ["BTCUSDT_15m"]}
        self.model_config = {"horizon_models": [{"lgbm": "model.pkl", "ridge": "ridge.pkl", "xgb": "xgb.json"}],
                             "regime_models": {"bear": {"lgbm": "bear.pkl"}}}
        for name in ("BTC_model", "BTC_4h"):
            directory = self.root / "models" / name
            directory.mkdir(parents=True)
            (directory / "config.json").write_text(json.dumps(self.model_config))
            for filename in ("model.pkl", "ridge.pkl", "xgb.json", "bear.pkl"):
                (directory / filename).write_text("fixture bytes are never deserialized")
        data = self.root / "data"
        data.mkdir()
        (data / "BTCUSDT_1h.csv").write_text("open_time,open,high,low,close,volume\n1000,1,2,1,2,3\n")
        (data / "BTCUSDT_open_interest.csv").write_text("timestamp,sum_open_interest\n1000,2\n")
        (data / "BTCUSDT_ls_ratio.csv").write_text("timestamp,long_short_ratio\n1000,1\n")
        self.write()

    def write(self):
        self.path.write_text(json.dumps(self.config))

    def load(self, **kwargs):
        return load_run_manifest(self.path, project_root=self.root, symbol_config=self.registry, **kwargs)

    def test_checked_in_manifest_partitions_actual_registry_without_loading_models(self):
        result = load_run_manifest(ROOT / "config/shadow.json", project_root=ROOT, validate_files=False)
        self.assertEqual(result.runner_keys, ("BTCUSDT", "ETHUSDT", "BTCUSDT_4h", "ETHUSDT_4h"))
        self.assertEqual(result.disabled_runner_keys, ("BTCUSDT_15m", "ETHUSDT_15m"))
        self.assertFalse(result.files_validated)
        self.assertEqual(result.shadow_balance, Decimal("10000"))

    def test_valid_manifest_resolves_actual_binding_and_checks_artifacts_without_deserialization(self):
        result = self.load()
        self.assertTrue(result.files_validated)
        self.assertEqual(result.symbols, ("BTCUSDT",))
        self.assertEqual(result.runners[0].warmup, 800)
        self.assertEqual(result.runners[1].interval_ms, 14_400_000)
        self.assertEqual(result.runners[1].model_path, self.root / "models/BTC_4h")
        self.assertIn(self.root / "models/BTC_model/bear.pkl", validate_run_inputs(result))
        self.assertEqual(result.shadow_balance, Decimal("10000.00"))

    def test_digest_is_deterministic_and_binds_selected_strategy_metadata(self):
        first = self.load()
        self.path.write_text(json.dumps(self.config, indent=4))
        self.assertEqual(first.digest, self.load().digest)
        self.registry["BTCUSDT_4h"]["warmup"] = 301
        self.assertNotEqual(first.digest, self.load().digest)
        before_size_change = self.load().digest
        self.registry["BTCUSDT"]["size"] = 0.001
        self.assertNotEqual(before_size_change, self.load().digest)
        self.config["shadow_balance"] = "20000"
        self.write()
        self.assertNotEqual(first.digest, self.load().digest)

    def test_exact_active_disabled_partition_prevents_unknown_omitted_and_conflicting_runners(self):
        original = deepcopy(self.config)
        mutations = [(["BTCUSDT", "SUIUSDT"], ["BTCUSDT_15m"]),
                     (["BTCUSDT", "BTCUSDT"], ["BTCUSDT_4h", "BTCUSDT_15m"]),
                     (["BTCUSDT"], ["BTCUSDT_15m"]),
                     (["BTCUSDT"], ["BTCUSDT", "BTCUSDT_4h", "BTCUSDT_15m"]),
                     ([], list(self.registry)),
                     (["BTCUSDT", "BTCUSDT_4h"], ["BTCUSDT_15m", "BTCUSDT_15m"])]
        for active, disabled in mutations:
            with self.subTest(active=active, disabled=disabled):
                self.config = dict(original, runners=active, disabled_runners=disabled)
                self.write()
                with self.assertRaises(ValueError):
                    self.load(validate_files=False)

    def test_unsupported_modes_currencies_and_unknown_configuration_are_rejected(self):
        original = deepcopy(self.config)
        for key, value in (("mode", "execution"), ("venue", "okx"), ("currency", "BTC"),
                           ("version", True), ("version", 2), ("notify", True), ("api_key", "synthetic")):
            with self.subTest(key=key):
                self.config = dict(original, **{key: value})
                self.write()
                with self.assertRaises(ValueError):
                    self.load(validate_files=False)

    def test_missing_and_malformed_synthetic_balance_never_falls_back_to_account(self):
        for value in (None, "0", "-1", "NaN", "Infinity", "1e3", 1000, True, " 1", "1.000000001"):
            with self.subTest(value=value):
                self.config["shadow_balance"] = value
                self.write()
                with self.assertRaises(ValueError):
                    self.load(validate_files=False)

    def test_synthetic_balance_fits_native_fixed_point_state(self):
        for value in ("0.00000001", "90000000000"):
            with self.subTest(value=value):
                self.config["shadow_balance"] = value
                self.write()
                balance = self.load(validate_files=False).shadow_balance
                self.assertLessEqual(int(float(balance) * 100_000_000), 2**63 - 1)
                self.assertGreater(int(float(balance) * 100_000_000), 0)
        for value in ("90000000000.00000001", "92233720368.54775808", "999999999999999999"):
            with self.subTest(value=value):
                self.config["shadow_balance"] = value
                self.write()
                with self.assertRaisesRegex(ValueError, "fixed-point"):
                    self.load(validate_files=False)

    def test_warmup_fits_public_api_response_limit(self):
        self.registry["BTCUSDT"]["warmup"] = 1000
        self.assertEqual(self.load(validate_files=False).runners[0].warmup, 1000)
        for warmup in (1001, 100_000):
            with self.subTest(warmup=warmup):
                self.registry["BTCUSDT"]["warmup"] = warmup
                with self.assertRaisesRegex(ValueError, "warmup"):
                    self.load(validate_files=False)

    def test_duplicate_and_nonfinite_json_are_rejected(self):
        raw = json.dumps(self.config)
        for value in (raw[:-1] + ',"mode":"shadow"}',
                      raw.replace('"shadow_balance": "10000.00"', '"shadow_balance": NaN'),
                      raw.replace('"shadow_balance": "10000.00"', '"shadow_balance": 1e9999')):
            self.path.write_text(value)
            with self.assertRaises(ValueError):
                self.load(validate_files=False)

    def test_root_paths_must_be_distinct_canonical_project_relative_directories(self):
        original = deepcopy(self.config)
        for path in ("../outside", "/absolute", "models/../data", ".", "./models", "models//nested", "models\\nested"):
            with self.subTest(path=path):
                self.config = dict(original, models_root=path)
                self.write()
                with self.assertRaises(ValueError):
                    self.load(validate_files=False)
        for path in ("data", "data/models"):
            self.config = dict(original, models_root=path)
            self.write()
            with self.assertRaises(ValueError):
                self.load(validate_files=False)

    def test_missing_empty_and_symbolic_artifacts_block_default_validation(self):
        path = self.root / "models/BTC_model/ridge.pkl"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()
        self.assertFalse(self.load(validate_files=False).files_validated)
        path.write_bytes(b"")
        with self.assertRaises(ValueError):
            self.load()
        path.unlink()
        path.symlink_to(self.root / "models/BTC_model/model.pkl")
        with self.assertRaises(ValueError):
            self.load()
        path.unlink()
        path.write_bytes(original)

    def test_manifest_and_root_directory_symlinks_are_rejected(self):
        alias = self.root / "alias.json"
        alias.symlink_to(self.path)
        with self.assertRaises(ValueError):
            load_run_manifest(alias, project_root=self.root, symbol_config=self.registry)
        alias.unlink()
        alias.symlink_to(self.root / "models", target_is_directory=True)
        self.config["models_root"] = "alias.json"
        self.write()
        with self.assertRaises(ValueError):
            self.load(validate_files=False)

    def test_model_reference_cannot_escape_directory_and_optional_sidecars_are_checked(self):
        directory = self.root / "models/BTC_model"
        self.model_config["horizon_models"][0]["ridge"] = "../outside.pkl"
        (directory / "config.json").write_text(json.dumps(self.model_config))
        with self.assertRaises(ValueError):
            self.load()
        self.model_config["horizon_models"][0]["ridge"] = "ridge.pkl"
        (directory / "config.json").write_text(json.dumps(self.model_config))
        (directory / "model.json").write_bytes(b"")
        with self.assertRaises(ValueError):
            self.load()

    def test_legacy_flat_model_format_is_supported_without_inventing_missing_artifacts(self):
        path = self.root / "models/BTC_model/config.json"
        path.write_text(json.dumps({"models": ["model.pkl", "xgb.json"]}))
        self.assertTrue(self.load().files_validated)
        path.write_text(json.dumps({"models": []}))
        with self.assertRaises(ValueError):
            self.load()

    def test_model_config_credential_and_nonfinite_fields_are_rejected(self):
        path = self.root / "models/BTC_model/config.json"
        for field in ({"private-key": "synthetic"}, {"token": "synthetic"}, {"bad": float("nan")},
                      {"BYBIT_API_KEY": "synthetic"}, {"metadata": {"telegram_bot_token": "synthetic"}}):
            path.write_text(json.dumps({**self.model_config, **field}))
            with self.assertRaises(ValueError):
                self.load()

    def test_required_csv_needs_expected_headers_and_at_least_one_observation(self):
        path = self.root / "data/BTCUSDT_open_interest.csv"
        for text in ("timestamp,other\n1000,1\n", "timestamp,sum_open_interest\n",
                     "timestamp,sum_open_interest,sum_open_interest\n1000,1,1\n"):
            path.write_text(text)
            with self.assertRaises(ValueError):
                self.load()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_registry_binding_validates_market_interval_model_directory_and_warmup(self):
        original = dict(self.registry["BTCUSDT"])
        for fields in ({"symbol": "../BTC"}, {"interval": "240.0"}, {"interval": 60},
                       {"model_dir": "../../outside"}, {"warmup": True}, {"warmup": 0}):
            with self.subTest(fields=fields):
                self.registry["BTCUSDT"] = {**original, **fields}
                with self.assertRaises(ValueError):
                    self.load(validate_files=False)


if __name__ == "__main__":
    unittest.main()
