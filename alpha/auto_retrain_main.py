"""Backward-compat: redirects to alpha.retrain.cli."""
from alpha.retrain.cli import *  # noqa: F401, F403
if __name__ == "__main__":
    from alpha.retrain.cli import main
    main()
