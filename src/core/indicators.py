# indicators_rsi_fixed.py
# Drop-in RSI implementation (Wilder) to replace your current rsi() in src/core/indicators.py
# Usage: copy the rsi() function body into your indicators.py and ensure numpy is imported.

import numpy as np
import pandas as pd

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's RSI with stable handling of NaNs and modern pandas API.
    - Uses ewm(..., adjust=False, min_periods=period) for Wilder smoothing.
    - Avoids deprecated fillna(method='bfill') by using .bfill().
    - Returns a Series aligned to input index with early values backfilled then defaulted to 50.0.
    """
    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()

    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)

    alpha = 1.0 / period  # Wilder smoothing equivalent
    roll_up = up.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi.bfill().fillna(50.0)
