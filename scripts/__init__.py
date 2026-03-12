"""Utility entrypoints for research, training, validation, data, and ops.

The `scripts/` package is intentionally separate from the production runtime.
It contains operator-facing tools that sit around the core engine:

- `train`: model training and retraining
- `validate`: backtests, walk-forward validation, parity checks
- `research`: feature studies and diagnostic analysis
- `data`: historical data download and refresh
- `ops`: paper/testnet helpers, smoke tests, operational utilities
- `shared`: reusable post-processing helpers for research scripts

See `scripts/catalog.py` and `scripts/README.md` for the maintained index.
"""
