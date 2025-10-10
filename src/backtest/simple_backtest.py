import pandas as pd

def simulate(df: pd.DataFrame, side: str, tp_pct: float, sl_pct: float):
    entries, pnls = 0, []
    for i in range(len(df)-1):
        entry = df['close'].iloc[i]
        if side == 'buy':
            tp = entry * (1 + tp_pct)
            sl = entry * (1 - sl_pct)
            hi = df['high'].iloc[i+1]
            lo = df['low'].iloc[i+1]
            if lo <= sl: pnls.append(-sl_pct); entries += 1; continue
            if hi >= tp: pnls.append(tp_pct); entries += 1; continue
        else:
            tp = entry * (1 - tp_pct)
            sl = entry * (1 + sl_pct)
            hi = df['high'].iloc[i+1]
            lo = df['low'].iloc[i+1]
            if hi >= sl: pnls.append(-sl_pct); entries += 1; continue
            if lo <= tp: pnls.append(tp_pct); entries += 1; continue
    win = sum(1 for x in pnls if x>0)
    rr = (sum(x for x in pnls if x>0) / max(1, win)) / (abs(sum(x for x in pnls if x<0)) / max(1, len(pnls)-win)) if pnls else 0
    return {'trades': entries, 'win_rate': win/max(1, len(pnls)), 'avg_pnl': sum(pnls)/max(1, len(pnls)), 'rr': rr}
