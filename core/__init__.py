"""Core foundation layer — types, clock, effects, errors.

This is the bottom of the dependency hierarchy.  Nothing in ``core/`` may
import from ``state/``, ``engine/``, ``event/``, ``risk/``, ``decision/``,
``execution/``, or ``infra/``.  All higher layers depend on ``core/``.
"""
