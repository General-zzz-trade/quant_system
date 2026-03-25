"""Ingestion — batch data download scripts.

CLI entry points (run as modules):
  python3 -m data.downloads.data_refresh        # Full incremental sync
  python3 -m data.downloads.data_refresh_cli     # CLI wrapper

Individual downloaders cover klines (Bybit/Binance, multiple timeframes),
funding rates, open interest, on-chain metrics, options (Deribit), macro
indicators (FRED), and sentiment (fear & greed).
"""
