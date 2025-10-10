from typing import List, Tuple

def simple_spread_scan(tickers: dict, fee_pct: float) -> List[Tuple[str, float]]:
    res = []
    symbols = set.intersection(*[set(d.keys()) for d in tickers.values()]) if tickers else set()
    for sym in symbols:
        prices = [(ex, d[sym]) for ex, d in tickers.items() if sym in d]
        if not prices:
            continue
        mx = max(prices, key=lambda x: x[1])
        mn = min(prices, key=lambda x: x[1])
        spread = (mx[1] - mn[1]) / mn[1]
        if spread > 3*fee_pct:
            res.append((sym, spread))
    return sorted(res, key=lambda x: x[1], reverse=True)
