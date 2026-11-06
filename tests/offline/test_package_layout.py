"""Source wheel layout contracts, without importing trading or build packages."""
from __future__ import annotations

from pathlib import Path
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[2]


class PackageLayoutTests(unittest.TestCase):
    def setUp(self):
        self.project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.wheel = self.project["tool"]["hatch"]["build"]["targets"]["wheel"]

    def test_every_runtime_package_is_explicitly_included(self):
        actual = {path.name for path in ROOT.iterdir()
                  if path.is_dir() and (path / "__init__.py").is_file() and path.name != "scripts"}
        selected = {pattern.split("/")[1] for pattern in self.wheel["include"]}
        self.assertEqual(selected, actual)

    def test_local_state_and_legacy_script_links_cannot_enter_wheel(self):
        # ignore-vcs is needed for data/ code, so selection must still be source-only.
        self.assertTrue(self.wheel["ignore-vcs"])
        self.assertNotIn("packages", self.wheel)
        self.assertNotIn("only-include", self.wheel)
        self.assertTrue(all(pattern.endswith(("/**/*.py", "/**/py.typed"))
                            for pattern in self.wheel["include"]))
        self.assertIn("**/tests/**", self.wheel["exclude"])
        for package in {pattern.split("/")[1] for pattern in self.wheel["include"]}:
            with self.subTest(package=package):
                self.assertTrue((ROOT / package / "__init__.py").is_file())
                self.assertFalse(any(path.is_symlink() for path in (ROOT / package).rglob("*.py")))

    def test_python_backend_remains_separate_from_native_extension(self):
        self.assertEqual(self.project["build-system"]["build-backend"], "hatchling.build")
        cargo = tomllib.loads((ROOT / "Cargo.toml").read_text())
        self.assertEqual(cargo["lib"]["name"], "_quant_hotpath")
        self.assertIn("cdylib", cargo["lib"]["crate-type"])
        self.assertFalse(any(pattern.startswith("/_quant_hotpath/") for pattern in self.wheel["include"]))

    @unittest.skipUnless(importlib.util.find_spec("hatchling"), "install hatchling to validate built wheel contents")
    def test_built_wheel_contains_code_only_and_all_runtime_packages(self):
        with tempfile.TemporaryDirectory() as scratch:
            fixture = Path(scratch)
            shutil.copyfile(ROOT / "pyproject.toml", fixture / "pyproject.toml")
            for pattern in self.wheel["include"]:
                package = pattern.split("/")[1]
                if not (fixture / package).exists():
                    shutil.copytree(ROOT / package, fixture / package,
                                    ignore=shutil.ignore_patterns("__pycache__", ".cache"))
            # Non-code state under an included package must never enter a wheel.
            runtime = fixture / "data/runtime"
            runtime.mkdir(parents=True, exist_ok=True)
            (runtime / "orders.json").write_text('{"must_not_ship": true}')
            (fixture / "scripts").mkdir()
            (fixture / "scripts/external.py").symlink_to(fixture / "nonexistent-external.py")
            result = subprocess.run(
                [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "--directory", str(fixture / "dist")],
                cwd=fixture, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            wheel, = (fixture / "dist").glob("*.whl")
            with zipfile.ZipFile(wheel) as archive:
                names = archive.namelist()
            code = [name for name in names if ".dist-info/" not in name]
            self.assertTrue(code)
            self.assertTrue(all(name.endswith((".py", "/py.typed")) for name in code))
            self.assertFalse(any(name.startswith("scripts/") or "/tests/" in name for name in code))
            for package in {pattern.split("/")[1] for pattern in self.wheel["include"]}:
                self.assertIn(f"{package}/__init__.py", names)
            self.assertIn("data/quality/causal_bars.py", names)


if __name__ == "__main__":
    unittest.main()
