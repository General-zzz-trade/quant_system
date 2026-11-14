"""Binance target-position trading system, organised as ports and adapters.

Dependencies point inward only: infrastructure -> adapters -> application -> domain.
The rules are enforced by ``.importlinter`` and ``tests/architecture``; see docs/arch/target.md.
Nothing in this package imports the legacy top-level packages (runner, shared, execution, ...).
"""
