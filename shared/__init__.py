"""Shared utilities used by both training and live monitoring paths.

Code placed here must be importable without side effects and should have
zero heavy dependencies beyond numpy/scipy so it can be invoked from
Rust training subprocesses, pytest, and live runner hot paths.
"""
