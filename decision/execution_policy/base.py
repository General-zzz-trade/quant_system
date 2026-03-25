"""Backward-compat stub — lazy re-export from strategy/execution_policy/base.py"""


def __getattr__(name: str):  # noqa: ANN001
    import strategy.execution_policy.base as _mod
    return getattr(_mod, name)
