"""Backward-compat: redirects to alpha.retrain.retrain_15m."""
from alpha.retrain.retrain_15m import *  # noqa: F401, F403
if __name__ == "__main__":
    from alpha.retrain.retrain_15m import main
    main()
