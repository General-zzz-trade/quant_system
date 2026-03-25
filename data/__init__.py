"""Data layer — 5 logical domains.

Domain 1: Ingestion     downloads/  — batch data download scripts (klines, funding, OI, etc.)
Domain 2: Storage       store.py + backends/ + loaders/ — types, persistence, and loading
Domain 3: Quality       quality/  — bar validation, gap detection, live event filtering
Domain 4: Ops           scheduler/ — data freshness monitoring and job scheduling
Domain 5: Runtime       oi_cache.py — live OI/LS background cache

Data-only directories (no Python code):
  live/        — runtime SQLite DBs (ack_store, event_log, state)
  runtime/     — runtime artifacts (decision_audit, health, checkpoints)
  onchain/     — on-chain CSV datasets
  options/     — options SQLite DBs
  polymarket/  — Polymarket collector DBs and CSVs
  ticks/       — tick-level SQLite DBs
"""
from data.store import Bar, TimeSeriesStore

__all__ = [
    "Bar",
    "TimeSeriesStore",
]
