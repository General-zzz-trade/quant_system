"""OKX exchange adapter (USDT perpetual SWAP).

OKX API v5: https://www.okx.com/docs-v5/
Auth: HMAC-SHA256(secret, timestamp + method + path + body), base64.
Symbol: BTCUSDT ↔ BTC-USDT-SWAP; qty is in **contracts** (1 ct = 0.01 BTC /
0.1 ETH) not coin units — conversion lives in `symbol_map.py`.
"""
