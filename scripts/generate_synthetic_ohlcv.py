#!/usr/bin/env python3
"""
Generate a synthetic OHLCV CSV for testing the feature pipeline.
Writes sample_data/test_samples_ohlcv.csv with N rows (default 500).
"""
import csv, math, random, os
from datetime import datetime, timedelta

OUT = "sample_data/test_samples_ohlcv.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

N = 500
start_price = 110000.0
vol_base = 10.0
dt = datetime.utcnow()

rows = []
price = start_price
for i in range(N):
    # simple random walk with occasional volatility
    drift = random.gauss(0, 0.0005)  # small drift
    shock = random.gauss(0, 0.001) * (1 + 0.1 * math.sin(i / 50.0))
    price = max(1.0, price * (1.0 + drift + shock))
    high = price * (1.0 + abs(random.gauss(0, 0.0015)))
    low = price * (1.0 - abs(random.gauss(0, 0.0015)))
    open_p = price / (1.0 + random.gauss(0, 0.0005))
    close_p = price
    volume = max(0.1, vol_base + random.gauss(0, 2.0))
    ts = (dt + timedelta(minutes=5*i)).isoformat() + "Z"
    rows.append([ts, round(open_p, 6), round(high, 6), round(low, 6), round(close_p, 6), round(volume, 6)])

with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp","open","high","low","close","volume"])
    writer.writerows(rows)

print("Wrote", OUT, "rows=", N)
