#!/usr/bin/env python3
"""Compatibility entrypoint for ``scripts.research.ic_analysis_onchain``."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("scripts.research.ic_analysis_onchain", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("scripts.research.ic_analysis_onchain")
