"""Compatibility re-export — moved to runner.main."""
from runner.main import *  # noqa: F401,F403
from runner.main import main  # noqa: F401

if __name__ == "__main__":
    main()
