"""OKX API base URLs.

Live and demo share the same host; demo is flagged via header
`x-simulated-trading: 1`. See `rest.OkxRestConfig.simulated`.
"""
from __future__ import annotations

OKX_REST_BASE = "https://www.okx.com"
OKX_REST_AWS = "https://aws.okx.com"  # AWS cluster mirror

OKX_WS_PUBLIC = "wss://ws.okx.com:8443/ws/v5/public"
OKX_WS_PRIVATE = "wss://ws.okx.com:8443/ws/v5/private"
OKX_WS_BUSINESS = "wss://ws.okx.com:8443/ws/v5/business"  # candle channels

# Demo environments
OKX_WS_PUBLIC_DEMO = "wss://wspap.okx.com:8443/ws/v5/public"
OKX_WS_PRIVATE_DEMO = "wss://wspap.okx.com:8443/ws/v5/private"
OKX_WS_BUSINESS_DEMO = "wss://wspap.okx.com:8443/ws/v5/business"
