#!/usr/bin/env python3
"""Compatibility entrypoint for ``scripts.data.download_spot_klines``."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("scripts.data.download_spot_klines", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("scripts.data.download_spot_klines")
