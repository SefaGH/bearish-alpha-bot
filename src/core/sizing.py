def position_size_usdt(entry: float, stop: float, risk_usd: float, side: str):
    dist = (stop - entry) if side == 'long' else (entry - stop)
    dist = max(dist, entry * 0.001)
    qty = risk_usd / dist
    return max(qty, 0.0)
