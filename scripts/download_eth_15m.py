"""Download ETHUSDT 15m kline data from Binance Futures API."""
import csv
import time
import urllib.request
import json

OUTPUT = "/quant_system/data_files/ETHUSDT_15m.csv"
SYMBOL = "ETHUSDT"
INTERVAL = "15m"
LIMIT = 1500
BASE_URL = "https://fapi.binance.com/fapi/v1/klines"

# 2024-01-01 00:00:00 UTC in ms
START_MS = 1704067200000
# 2026-03-12 23:59:59 UTC in ms
END_MS = 1741910399000

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore"
]

all_rows = []
current_start = START_MS
batch = 0

while current_start < END_MS:
    url = f"{BASE_URL}?symbol={SYMBOL}&interval={INTERVAL}&limit={LIMIT}&startTime={current_start}"
    if END_MS:
        url += f"&endTime={END_MS}"

    for attempt in range(5):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            break
        except Exception as e:
            if attempt < 4:
                wait = 2 ** attempt
                print(f"  Retry {attempt+1} after error: {e}, waiting {wait}s")
                time.sleep(wait)
            else:
                raise

    if not data:
        break

    batch += 1
    rows = []
    for candle in data:
        rows.append(candle[:12])

    all_rows.extend(rows)
    last_close_time = data[-1][6]
    current_start = last_close_time + 1

    if batch % 10 == 0:
        print(f"Batch {batch}: {len(all_rows)} rows so far, latest close_time={last_close_time}")

    # Rate limit: stay well under 1200 req/min
    time.sleep(0.15)

print(f"\nTotal batches: {batch}")
print(f"Total rows before dedup: {len(all_rows)}")

# Deduplicate by open_time
seen = set()
unique_rows = []
for row in all_rows:
    ot = row[0]
    if ot not in seen:
        seen.add(ot)
        unique_rows.append(row)

unique_rows.sort(key=lambda r: r[0])
print(f"Total rows after dedup: {len(unique_rows)}")

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(COLUMNS)
    for row in unique_rows:
        writer.writerow(row)

print(f"Saved to {OUTPUT}")
print(f"First row open_time: {unique_rows[0][0]}")
print(f"Last row open_time:  {unique_rows[-1][0]}")
print(f"Final row count: {len(unique_rows)}")
