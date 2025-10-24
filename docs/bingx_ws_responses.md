# BingX WebSocket Response Formats

This document describes the exact response formats from BingX WebSocket API based on real captured data.

## General Response Structure

All BingX WebSocket messages follow this pattern:
```json
{
  "code": 0,           // Response code: 0 = success, other = error
  "dataType": "...",   // Data type identifier
  "data": {...}        // Actual data (format varies by type)
}
```

## Subscription Confirmation

When subscribing to a channel, BingX sends a confirmation:

```json
{
  "id": "test_ticker_btc",
  "code": 0,
  "msg": "",
  "dataType": "",
  "data": null
}
```

**Fields:**
- `id` (string): The subscription ID you sent
- `code` (number): 0 = success, non-zero = error
- `msg` (string): Error message if code != 0
- `dataType` (string): Empty for subscription confirmations
- `data` (null): Always null for confirmations

## Ticker Data Format

**Data Type:** `{SYMBOL}@ticker` (e.g., `BTC-USDT@ticker`)

```json
{
  "code": 0,
  "dataType": "BTC-USDT@ticker",
  "data": {
    "e": "24hTicker",          // Event type
    "E": 1761327444754,        // Event time (timestamp in ms)
    "s": "BTC-USDT",           // Symbol
    "p": "-436.0",             // Price change
    "P": "-0.39",              // Price change percent
    "c": "110267.8",           // Close price (last price)
    "L": "0.0006",             // Last quantity
    "h": "112080.0",           // 24h high
    "l": "109283.7",           // 24h low
    "v": "15204.7267",         // 24h volume
    "q": "171854.69",          // 24h quote volume
    "o": "110703.8",           // Open price
    "O": 1761327293627,        // Open time
    "C": 1761327444478,        // Close time
    "A": "110267.8",           // Best ask price
    "a": "2.5786",             // Best ask quantity
    "B": "110267.6",           // Best bid price
    "b": "5.4305"              // Best bid quantity
  }
}
```

**Data Types:**
- All prices are **strings** (must be converted to float)
- Timestamps are **numbers** (milliseconds)
- Volumes are **strings** (must be converted to float)

## Kline/Candlestick Data Format

**Data Type:** `{SYMBOL}@kline_{INTERVAL}` (e.g., `BTC-USDT@kline_1m`)

**⚠️ CRITICAL: The `data` field is an ARRAY of kline objects!**

```json
{
  "code": 0,
  "dataType": "BTC-USDT@kline_1m",
  "s": "BTC-USDT",             // Symbol
  "data": [                     // ARRAY of kline objects
    {
      "c": "110267.6",         // Close price
      "o": "110298.6",         // Open price
      "h": "110298.6",         // High price
      "l": "110265.1",         // Low price
      "v": "2.0741",           // Volume
      "T": 1761327420000       // Timestamp (ms)
    }
  ]
}
```

**Important Notes:**
- `data` is always an **ARRAY**, even for single kline updates
- Usually contains 1 element (the current/latest kline)
- May contain multiple elements for batch updates
- All prices/volumes are **strings** (must be converted to float)
- Timestamp `T` is a **number** (milliseconds)

**Common Bug:**
The original code tried to do `data.get("data", {})` and then treat it as a dict, but it's actually a list!

**Correct Parsing:**
```python
kline_data = data.get("data", [])  # Get as list
if isinstance(kline_data, list):
    for kline_obj in kline_data:
        # kline_obj is a dict with c, o, h, l, v, T fields
        timestamp = kline_obj.get('T', 0)
        open_price = float(kline_obj.get('o', 0))
        high = float(kline_obj.get('h', 0))
        low = float(kline_obj.get('l', 0))
        close = float(kline_obj.get('c', 0))
        volume = float(kline_obj.get('v', 0))
```

## Depth/Orderbook Data Format

**Data Type:** `{SYMBOL}@depth{LEVELS}@{UPDATE_SPEED}` (e.g., `BTC-USDT@depth20@500ms`)

```json
{
  "code": 0,
  "dataType": "BTC-USDT@depth20@500ms",
  "ts": 1761327444754,        // Timestamp
  "data": {
    "bids": [                  // Array of [price, quantity]
      ["110267.6", "5.4305"],
      ["110267.5", "0.0434"],
      ["110267.2", "0.0818"]
      // ... up to 20 levels
    ],
    "asks": [                  // Array of [price, quantity]
      ["110275.2", "0.2302"],
      ["110274.1", "0.2171"],
      ["110273.0", "0.1786"]
      // ... up to 20 levels
    ]
  }
}
```

**Data Types:**
- `bids` and `asks` are arrays of 2-element arrays
- Each element is `[price_string, quantity_string]`
- Both price and quantity are **strings** (must be converted to float)

## Ping/Pong Messages

BingX sends periodic Ping messages:
- **Received:** String `"Ping"` (GZIP compressed)
- **Response:** Must send string `"Pong"` (not GZIP compressed)

## GZIP Compression

**All messages from BingX are GZIP compressed!**

**Decompression:**
```python
import gzip
import io

def decompress_message(message):
    if isinstance(message, bytes):
        compressed_data = gzip.GzipFile(fileobj=io.BytesIO(message), mode='rb')
        decompressed_data = compressed_data.read()
        return decompressed_data.decode('utf-8')
    return message
```

## Symbol Format

BingX uses hyphen-separated format:
- BingX format: `BTC-USDT`
- CCXT format: `BTC/USDT:USDT` (for futures)

## Error Responses

When an error occurs, the response contains:
```json
{
  "code": <non-zero>,
  "msg": "Error description",
  "dataType": "",
  "data": null
}
```

## Message Statistics (from 60-second test)

- **Ticker messages:** ~120 (2 per second per symbol)
- **Kline messages:** ~193 (updated when price changes)
- **Depth messages:** ~301 (high frequency, ~5 per second)
- **Subscription confirmations:** 5 (one per subscription)
- **Ping/Pong:** Periodic (every 30 seconds or so)

## Key Takeaways for Implementation

1. **Kline data is always an array** - Don't use `.get()` directly on it
2. **All numeric values are strings** - Convert to float before use
3. **All messages are GZIP compressed** - Use `gzip.GzipFile` with `io.BytesIO`
4. **Ping/Pong are plain strings** - Not JSON, just "Ping"/"Pong"
5. **Field names are abbreviated** - `c` for close, `o` for open, etc.
6. **Timestamps are in milliseconds** - Not seconds
