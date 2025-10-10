def initial_stops(side: str, entry: float, atr: float, sl_atr_mult: float, tp_pct: float):
    if side == 'sell':
        sl = entry + sl_atr_mult * atr
        tp = entry * (1 - tp_pct)
    else:
        sl = entry * (1 - sl_atr_mult) if sl_atr_mult < 0.5 else entry - sl_atr_mult * atr
        tp = entry * (1 + tp_pct)
    return tp, sl

def trail_level(side: str, last_price: float, atr: float, trail_atr_mult: float):
    # returns new stop suggestion based on last price +/- k*ATR
    if side == 'sell':
        return last_price + trail_atr_mult * atr
    else:
        return last_price - trail_atr_mult * atr
