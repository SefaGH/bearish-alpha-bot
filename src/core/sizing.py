def position_size_usdt(entry: float, stop: float, risk_usd: float, side: str):
    # Calculate distance: for LONG, stop is below entry; for SHORT, stop is above entry
    dist = (entry - stop) if side == 'long' else (stop - entry)
    # Ensure minimum distance (0.1% of entry price) to avoid division by zero or unrealistic leverage
    dist = max(dist, entry * 0.001)
    qty = risk_usd / dist
    return max(qty, 0.0)
