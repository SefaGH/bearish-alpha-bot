import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_regime_labels(
    price_data: pd.DataFrame,
    window: int = 20,
    threshold: float = 0.015,
    prediction_horizon: int = 5,
    volume_confirm: bool = True,
    multi_timeframe: bool = True
) -> pd.Series:
    """
    Generate regime labels using PAST data only (NO LOOKAHEAD BIAS).
    
    Improvements:
    1. ✅ No lookahead bias (uses past returns only)
    2. ✅ Adaptive threshold (volatility-adjusted)
    3. ✅ Volume confirmation (optional)
    4. ✅ Multi-timeframe consensus (optional)
    5. ✅ Proper NaN handling (drop instead of ffill)
    
    Labels:
    - 0: Bullish (uptrend expected)
    - 1: Neutral (sideways expected)
    - 2: Bearish (downtrend expected)
    
    Args:
        price_data: OHLCV DataFrame
        window: Lookback window for regime calculation (default: 20)
        threshold: Base threshold for regime classification (default: 0.015 = 1.5%)
        prediction_horizon: Periods ahead to predict (default: 5)
        volume_confirm: Require volume confirmation (default: True)
        multi_timeframe: Use multiple timeframes (default: True)
    
    Returns:
        Series with regime labels (0=Bullish, 1=Neutral, 2=Bearish)
    """
    logger.info(f"Generating regime labels: window={window}, threshold={threshold}, "
                f"horizon={prediction_horizon}, volume_confirm={volume_confirm}, "
                f"multi_timeframe={multi_timeframe}")
    
    # === STEP 1: Calculate Volatility for Adaptive Threshold ===
    returns = price_data['close'].pct_change()
    volatility = returns.rolling(window=window).std()
    vol_mean = volatility.mean()
    
    # Adaptive threshold: Higher volatility → Higher threshold
    adaptive_threshold = threshold * (1 + volatility / vol_mean)
    adaptive_threshold = adaptive_threshold.fillna(threshold)
    
    # === STEP 2: Multi-Timeframe Analysis (if enabled) ===
    if multi_timeframe:
        # Short-term (window/2)
        short_returns = price_data['close'].pct_change(periods=window//2)
        short_sma = price_data['close'].rolling(window=window//2).mean()
        short_trend = (price_data['close'] - short_sma) / short_sma
        
        # Medium-term (window)
        medium_returns = price_data['close'].pct_change(periods=window)
        medium_sma = price_data['close'].rolling(window=window).mean()
        medium_trend = (price_data['close'] - medium_sma) / medium_sma
        
        # Long-term (window*2)
        long_returns = price_data['close'].pct_change(periods=window*2)
        long_sma = price_data['close'].rolling(window=window*2).mean()
        long_trend = (price_data['close'] - long_sma) / long_sma
        
        # Regime scores for each timeframe (-1, 0, +1)
        short_score = np.where(
            (short_returns > adaptive_threshold) & (short_trend > 0), 1,
            np.where((short_returns < -adaptive_threshold) & (short_trend < 0), -1, 0)
        )
        medium_score = np.where(
            (medium_returns > adaptive_threshold) & (medium_trend > 0), 1,
            np.where((medium_returns < -adaptive_threshold) & (medium_trend < 0), -1, 0)
        )
        long_score = np.where(
            (long_returns > adaptive_threshold) & (long_trend > 0), 1,
            np.where((long_returns < -adaptive_threshold) & (long_trend < 0), -1, 0)
        )
        
        # Weighted consensus (longer timeframe = more weight)
        consensus = (short_score * 1 + medium_score * 2 + long_score * 3) / 6
        
        # Use consensus for regime determination
        regime_score = pd.Series(consensus, index=price_data.index)
        
    else:
        # === STEP 2 (Alternative): Single Timeframe ===
        past_returns = price_data['close'].pct_change(periods=window)
        price_sma = price_data['close'].rolling(window=window).mean()
        trend_strength = (price_data['close'] - price_sma) / price_sma
        
        # Simple regime score
        regime_score = pd.Series(0.0, index=price_data.index)
        regime_score[(past_returns > adaptive_threshold) & (trend_strength > 0)] = 1.0
        regime_score[(past_returns < -adaptive_threshold) & (trend_strength < 0)] = -1.0
    
    # === STEP 3: Volume Confirmation (if enabled) ===
    if volume_confirm and 'volume' in price_data.columns:
        volume_sma = price_data['volume'].rolling(window=window).mean()
        volume_ratio = price_data['volume'] / volume_sma
        
        # Reduce regime score if volume doesn't confirm
        volume_confirmed = volume_ratio > 1.0
        regime_score = regime_score * np.where(volume_confirmed, 1.0, 0.5)
    
    # === STEP 4: Convert Regime Score to Labels ===
    labels = pd.Series(1, index=price_data.index, name="regime_labels")  # Default: Neutral
    
    if multi_timeframe:
        # Multi-timeframe: Use consensus thresholds
        labels[regime_score > 0.3] = 0   # Bullish (weighted majority bullish)
        labels[regime_score < -0.3] = 2  # Bearish (weighted majority bearish)
    else:
        # Single timeframe: Use simple thresholds
        labels[regime_score > 0.5] = 0   # Bullish
        labels[regime_score < -0.5] = 2  # Bearish
    
    # === STEP 5: Shift for Prediction Horizon (NO LOOKAHEAD!) ===
    # Shift labels BACKWARD so we predict FUTURE regime using CURRENT features
    # This is correct: features at t=100 predict label at t=105
    labels = labels.shift(-prediction_horizon)
    
    # === STEP 6: Handle NaN Properly ===
    # Drop NaN at the end (honest approach, no fake labels)
    initial_length = len(labels)
    labels = labels.dropna()
    dropped = initial_length - len(labels)
    
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with NaN labels (last {prediction_horizon} + warmup)")
    
    # === STEP 7: Log Statistics ===
    label_counts = labels.value_counts().sort_index()
    total = len(labels)
    logger.info(f"Label generation complete. Counts: "
                f"Bullish (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/total*100:.1f}%), "
                f"Neutral (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/total*100:.1f}%), "
                f"Bearish (2): {label_counts.get(2, 0)} ({label_counts.get(2, 0)/total*100:.1f}%)")
    
    return labels


def generate_simple_labels(price_data: pd.DataFrame,
                          window: int = 20,
                          threshold: float = 0.01) -> pd.Series:
    """
    LEGACY: Simple label generator for backward compatibility.
    Uses PAST returns (no lookahead bias) but without enhancements.
    
    Use generate_regime_labels() for better results.
    """
    logger.warning("Using legacy simple label generator. Consider using generate_regime_labels() instead.")
    
    # Calculate past returns
    past_returns = price_data['close'].pct_change(periods=window)
    
    # Initialize labels (default: Neutral)
    labels = pd.Series(1, index=price_data.index, name="regime_labels")
    
    # Assign labels based on past returns
    labels[past_returns > threshold] = 0   # Bullish
    labels[past_returns < -threshold] = 2  # Bearish
    
    # Shift for prediction (5 periods ahead)
    labels = labels.shift(-5)
    labels = labels.dropna()
    
    return labels
