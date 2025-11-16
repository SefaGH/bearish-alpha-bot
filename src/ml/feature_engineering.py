"""
Feature Engineering Pipeline for ML Market Regime Prediction.

Advanced feature extraction from market data for regime prediction models.

This module provides a comprehensive feature engineering pipeline with ~87 features
including:
- Technical indicators (RSI, MACD, EMA, Bollinger Bands, ATR)
- Market microstructure (price range, volume patterns, returns)
- Volatility features (realized volatility, Parkinson volatility, regime classification)
- Momentum features (ROC, MA slopes, trend strength)
- Advanced momentum (momentum at multiple periods, acceleration, cumulative)
- Advanced volume (volume momentum, VWAP, OBV)
- Advanced volatility (ATR ratio, BB width, historical volatility)
- Advanced trend (ADX, directional indicators, MA distance ratios)
- Support/resistance (distance from highs/lows, range position)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

# pandas_ta'yı güvenli bir şekilde import et
try:
    # <<< KÖK NEDEN DÜZELTMESİ: Doğru kütüphane adı kullanıldı.
    import pandas_ta_classic as ta
except ImportError:
    ta = None

logger = logging.getLogger(__name__)


class TechnicalIndicatorFeatures:
    """Extract technical indicator features from price data."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize technical indicator feature extractor with config."""
        if not ta:
            raise ImportError("pandas_ta_classic kütüphanesi bulunamadı. Lütfen 'pip install pandas-ta-classic' ile kurun.")
        
        # Config varsa kullan, yoksa varsayılan değerleri kullan
        config = config or {}
        self.rsi_period = config.get('rsi_period', 14)
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2)
        self.atr_period = config.get('atr_period', 14)
        
    def compute(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicator features.
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicator features
        """
        features = pd.DataFrame(index=price_data.index)
        n = len(price_data) # Hata durumunda NaN serisi oluşturmak için
        
        try:
            # --- GÜNCELLENMİŞ BLOK (ANALİZ-O) ---
            # RSI (Guarded)
            try:
                rsi_series = ta.rsi(price_data['close'], length=self.rsi_period)
                features['rsi'] = rsi_series
                features['rsi_oversold'] = (features['rsi'] < 30).astype(float)
                features['rsi_overbought'] = (features['rsi'] > 70).astype(float)
            except Exception:
                features['rsi'] = pd.Series([np.nan] * n, index=price_data.index)
                features['rsi_oversold'] = pd.Series([np.nan] * n, index=price_data.index)
                features['rsi_overbought'] = pd.Series([np.nan] * n, index=price_data.index)

            # MACD (Guarded)
            try:
                macd = ta.macd(price_data['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
                if macd is None or not isinstance(macd, (pd.DataFrame, pd.Series)):
                    raise ValueError("macd returned None or unsupported type")
                # Protect against differing column names
                try:
                    features['macd'] = macd[f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
                    features['macd_signal'] = macd[f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
                    features['macd_histogram'] = macd[f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
                except Exception:
                    # best-effort: try to pick first three columns if named differently
                    if isinstance(macd, pd.DataFrame) and macd.shape[1] >= 3:
                        features['macd'] = macd.iloc[:, 0]
                        features['macd_signal'] = macd.iloc[:, 1]
                        features['macd_histogram'] = macd.iloc[:, 2]
                    else:
                        raise
                features['macd_cross'] = np.sign(features['macd_histogram'])
            except Exception:
                # fallback: fill NaNs
                features['macd'] = pd.Series([np.nan] * n, index=price_data.index)
                features['macd_signal'] = pd.Series([np.nan] * n, index=price_data.index)
                features['macd_histogram'] = pd.Series([np.nan] * n, index=price_data.index)
                features['macd_cross'] = pd.Series([0.0] * n, index=price_data.index)
            # --- GÜNCELLEME SONU ---
            
            # EMA (Bu bölümü de güvenli hale getirelim)
            try:
                features['ema_20'] = ta.ema(price_data['close'], length=20)
                features['ema_50'] = ta.ema(price_data['close'], length=50)
                features['ema_cross'] = (features['ema_20'] > features['ema_50']).astype(float)
            except Exception:
                features['ema_20'] = pd.Series([np.nan] * n, index=price_data.index)
                features['ema_50'] = pd.Series([np.nan] * n, index=price_data.index)
                features['ema_cross'] = pd.Series([np.nan] * n, index=price_data.index)

            # Bollinger Bands (Bu bölümü de güvenli hale getirelim)
            try:
                bbands = ta.bbands(price_data['close'], length=self.bb_period)
                features['bb_upper'] = bbands[f'BBU_{self.bb_period}_2.0']
                features['bb_lower'] = bbands[f'BBL_{self.bb_period}_2.0']
                bb_range = features['bb_upper'] - features['bb_lower']
                features['bb_width'] = bb_range / price_data['close']
                features['bb_position'] = (price_data['close'] - features['bb_lower']) / (bb_range + 1e-10)
            except Exception:
                features['bb_upper'] = pd.Series([np.nan] * n, index=price_data.index)
                features['bb_lower'] = pd.Series([np.nan] * n, index=price_data.index)
                features['bb_width'] = pd.Series([np.nan] * n, index=price_data.index)
                features['bb_position'] = pd.Series([np.nan] * n, index=price_data.index)

            # ATR (Bu bölümü de güvenli hale getirelim)
            try:
                features['atr'] = ta.atr(price_data['high'], price_data['low'], price_data['close'], length=self.atr_period)
                features['atr_pct'] = features['atr'] / price_data['close']
            except Exception:
                features['atr'] = pd.Series([np.nan] * n, index=price_data.index)
                features['atr_pct'] = pd.Series([np.nan] * n, index=price_data.index)
            
        except Exception as e:
            logger.error(f"Error computing technical indicators: {e}")
        
        return features


class MarketMicrostructureFeatures:
    """Extract market microstructure features."""
    
    def compute(self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None,
                orderbook_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute market microstructure features.
        
        Args:
            price_data: DataFrame with OHLCV data
            volume_data: Optional volume-specific data
            orderbook_data: Optional order book data
            
        Returns:
            DataFrame with microstructure features
        """
        features = pd.DataFrame(index=price_data.index)
        
        try:
            # Price-based microstructure
            features['price_range'] = (price_data['high'] - price_data['low']) / price_data['close']
            features['close_position'] = (price_data['close'] - price_data['low']) / (price_data['high'] - price_data['low'] + 1e-10)
            
            # Volume features
            if 'volume' in price_data.columns:
                features['volume'] = price_data['volume']
                features['volume_ma'] = price_data['volume'].rolling(window=20).mean()
                features['volume_ratio'] = price_data['volume'] / (features['volume_ma'] + 1e-10)
            
            # Price momentum
            features['returns_1'] = price_data['close'].pct_change(1)
            features['returns_5'] = price_data['close'].pct_change(5)
            features['returns_10'] = price_data['close'].pct_change(10)
            
        except Exception as e:
            logger.error(f"Error computing market microstructure features: {e}")
        
        return features


class VolatilityFeatures:
    """Extract volatility-related features."""

    def __init__(self, windows: List[int] = None):
        """Initialize with configurable windows."""
        self.windows = windows or [5, 10, 20, 50]
    
    def compute(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility features across multiple windows."""
        features = pd.DataFrame(index=price_data.index)
        
        try:
            returns = price_data['close'].pct_change()
            
            for window in self.windows:  # self.windows kullan
                # Realized volatility
                features[f'vol_{window}'] = returns.rolling(window=window).std()
                
                # Parkinson volatility (using high-low range)
                hl_ratio = np.log(price_data['high'] / price_data['low'])
                features[f'parkinson_vol_{window}'] = np.sqrt(
                    (hl_ratio ** 2).rolling(window=window).mean() / (4 * np.log(2))
                )
            
            # Volatility regime classification
            vol_mean = features['vol_20'].rolling(window=50).mean()
            vol_std = features['vol_20'].rolling(window=50).std()
            features['vol_regime'] = (features['vol_20'] - vol_mean) / (vol_std + 1e-10)
            
        except Exception as e:
            logger.error(f"Error computing volatility features: {e}")
        
        return features


class MomentumFeatures:
    """Extract momentum and trend features."""
    
    def __init__(self, windows: List[int] = None):
        """Initialize with configurable windows."""
        self.windows = windows or [5, 10, 20, 50]
    
    def compute(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum and trend features.
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=price_data.index)
        
        try:
            for window in self.windows:  # self.windows kullan (parametreden değil)
                # Rate of change
                features[f'roc_{window}'] = price_data['close'].pct_change(window)
                
                # Moving average slope
                ma = price_data['close'].rolling(window=window).mean()
                features[f'ma_slope_{window}'] = ma.pct_change(1)
            
            # Trend strength hesaplaması (mevcut kod korunuyor)
            ema20 = ta.ema(price_data['close'], length=20)
            ema50 = ta.ema(price_data['close'], length=50)
            
            if ema20 is not None and ema50 is not None:
                features['trend_strength'] = (ema20 - ema50) / price_data['close']
            else:
                features['trend_strength'] = np.nan
            
            # Momentum regime
            mom_mean = features['roc_20'].rolling(window=50).mean()
            mom_std = features['roc_20'].rolling(window=50).std()
            features['momentum_regime'] = (features['roc_20'] - mom_mean) / (mom_std + 1e-10)
            
        except Exception as e:
            logger.error(f"Error computing momentum features: {e}")
        
        return features
        

class CrossAssetFeatures:
    """Extract cross-asset correlation features."""
    
    def compute(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute cross-asset correlation features.
        
        Args:
            price_data: Dictionary of DataFrames with price data for different assets
            
        Returns:
            DataFrame with cross-asset features
        """
        # Placeholder for cross-asset features
        # In real implementation, would compute correlations with other markets
        features = pd.DataFrame()
        
        try:
            if len(price_data) > 1:
                # Compute correlation matrix
                returns_dict = {}
                for symbol, data in price_data.items():
                    returns_dict[symbol] = data['close'].pct_change()
                
                returns_df = pd.DataFrame(returns_dict)
                
                # Rolling correlation features
                for window in [20, 50]:
                    corr = returns_df.rolling(window=window).corr()
                    # Extract correlation features (simplified)
                    pass
        
        except Exception as e:
            logger.error(f"Error computing cross-asset features: {e}")
        
        return features


class AdvancedMomentumFeatures:
    """Extract advanced momentum features for improved prediction."""
    
    def compute(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute advanced momentum features.
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with advanced momentum features
        """
        features = pd.DataFrame(index=price_data.index)
        n = len(price_data)
        
        try:
            close = price_data['close']
            
            # Price momentum at multiple periods
            for period in [3, 5, 10, 14, 20, 30]:
                features[f'momentum_{period}'] = close.pct_change(period)
            
            # Momentum acceleration (rate of change of momentum)
            features['momentum_acceleration'] = features['momentum_10'].diff()
            
            # Cumulative momentum (rolling sum of returns)
            returns = close.pct_change()
            features['cumulative_momentum_10'] = returns.rolling(window=10).sum()
            features['cumulative_momentum_20'] = returns.rolling(window=20).sum()
            
        except Exception as e:
            logger.error(f"Error computing advanced momentum features: {e}")
            # Fill with NaN on error
            for col in features.columns:
                if col not in features or features[col].isna().all():
                    features[col] = pd.Series([np.nan] * n, index=price_data.index)
        
        return features


class AdvancedVolumeFeatures:
    """Extract advanced volume features for improved prediction."""
    
    def compute(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute advanced volume features.
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with advanced volume features
        """
        features = pd.DataFrame(index=price_data.index)
        n = len(price_data)
        
        try:
            if 'volume' not in price_data.columns:
                logger.warning("Volume data not available for advanced volume features")
                # Return empty features with NaN
                for col_name in ['volume_momentum_5', 'volume_momentum_10', 'volume_ma_ratio_5', 
                                 'volume_ma_ratio_20', 'vwap', 'distance_from_vwap', 'obv', 'obv_momentum']:
                    features[col_name] = pd.Series([np.nan] * n, index=price_data.index)
                return features
            
            volume = price_data['volume']
            close = price_data['close']
            
            # Volume momentum (rate of change of volume)
            features['volume_momentum_5'] = volume.pct_change(5)
            features['volume_momentum_10'] = volume.pct_change(10)
            
            # Volume MA ratios
            volume_ma_5 = volume.rolling(window=5).mean()
            volume_ma_20 = volume.rolling(window=20).mean()
            features['volume_ma_ratio_5'] = volume / (volume_ma_5 + 1e-10)
            features['volume_ma_ratio_20'] = volume / (volume_ma_20 + 1e-10)
            
            # VWAP (Volume Weighted Average Price)
            if 'high' in price_data.columns and 'low' in price_data.columns:
                typical_price = (price_data['high'] + price_data['low'] + close) / 3
                features['vwap'] = (typical_price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
                features['distance_from_vwap'] = (close - features['vwap']) / (features['vwap'] + 1e-10)
            else:
                features['vwap'] = pd.Series([np.nan] * n, index=price_data.index)
                features['distance_from_vwap'] = pd.Series([np.nan] * n, index=price_data.index)
            
            # OBV (On-Balance Volume)
            price_change = close.diff()
            obv = pd.Series(index=price_data.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(price_data)):
                if price_change.iloc[i] > 0:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif price_change.iloc[i] < 0:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            features['obv'] = obv
            features['obv_momentum'] = obv.pct_change(10)
            
        except Exception as e:
            logger.error(f"Error computing advanced volume features: {e}")
            # Fill with NaN on error
            for col in features.columns:
                if col not in features or features[col].isna().all():
                    features[col] = pd.Series([np.nan] * n, index=price_data.index)
        
        return features


class AdvancedVolatilityFeatures:
    """Extract advanced volatility features for improved prediction."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with config."""
        config = config or {}
        self.atr_period = config.get('atr_period', 14)
        self.bb_period = config.get('bb_period', 20)
    
    def compute(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute advanced volatility features.
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with advanced volatility features
        """
        features = pd.DataFrame(index=price_data.index)
        n = len(price_data)
        
        try:
            close = price_data['close']
            returns = close.pct_change()
            
            # ATR ratio (normalized volatility)
            if 'high' in price_data.columns and 'low' in price_data.columns:
                atr = ta.atr(price_data['high'], price_data['low'], close, length=self.atr_period)
                if atr is not None:
                    features['atr_ratio'] = atr / close
                    features['atr_momentum'] = atr.pct_change(5)
                else:
                    features['atr_ratio'] = pd.Series([np.nan] * n, index=price_data.index)
                    features['atr_momentum'] = pd.Series([np.nan] * n, index=price_data.index)
            else:
                features['atr_ratio'] = pd.Series([np.nan] * n, index=price_data.index)
                features['atr_momentum'] = pd.Series([np.nan] * n, index=price_data.index)
            
            # Bollinger Band width and momentum
            bbands = ta.bbands(close, length=self.bb_period)
            if bbands is not None and not bbands.empty:
                try:
                    bb_upper = bbands[f'BBU_{self.bb_period}_2.0']
                    bb_lower = bbands[f'BBL_{self.bb_period}_2.0']
                    bb_middle = bbands[f'BBM_{self.bb_period}_2.0']
                    
                    bb_width = (bb_upper - bb_lower) / (bb_middle + 1e-10)
                    features['bb_width_normalized'] = bb_width
                    features['bb_width_momentum'] = bb_width.pct_change(5)
                    
                    # BB position (where price is in BB channel)
                    features['bb_position_advanced'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
                except Exception:
                    features['bb_width_normalized'] = pd.Series([np.nan] * n, index=price_data.index)
                    features['bb_width_momentum'] = pd.Series([np.nan] * n, index=price_data.index)
                    features['bb_position_advanced'] = pd.Series([np.nan] * n, index=price_data.index)
            else:
                features['bb_width_normalized'] = pd.Series([np.nan] * n, index=price_data.index)
                features['bb_width_momentum'] = pd.Series([np.nan] * n, index=price_data.index)
                features['bb_position_advanced'] = pd.Series([np.nan] * n, index=price_data.index)
            
            # Historical volatility (rolling std of returns)
            features['hist_volatility_5'] = returns.rolling(window=5).std()
            features['hist_volatility_10'] = returns.rolling(window=10).std()
            features['hist_volatility_20'] = returns.rolling(window=20).std()
            
            # Volatility ratio (short-term / long-term)
            features['volatility_ratio'] = features['hist_volatility_5'] / (features['hist_volatility_20'] + 1e-10)
            
        except Exception as e:
            logger.error(f"Error computing advanced volatility features: {e}")
            # Fill with NaN on error
            for col in features.columns:
                if col not in features or features[col].isna().all():
                    features[col] = pd.Series([np.nan] * n, index=price_data.index)
        
        return features


class AdvancedTrendFeatures:
    """Extract advanced trend features for improved prediction."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with config."""
        config = config or {}
        self.adx_period = config.get('adx_period', 14)
    
    def compute(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute advanced trend features.
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with advanced trend features
        """
        features = pd.DataFrame(index=price_data.index)
        n = len(price_data)
        
        try:
            close = price_data['close']
            
            # ADX indicators
            if 'high' in price_data.columns and 'low' in price_data.columns:
                adx_result = ta.adx(price_data['high'], price_data['low'], close, length=self.adx_period)
                if adx_result is not None and not adx_result.empty:
                    try:
                        features['adx'] = adx_result[f'ADX_{self.adx_period}']
                        features['adx_strong_trend'] = (features['adx'] > 25).astype(float)
                        features['adx_momentum'] = features['adx'].pct_change(5)
                        
                        # Directional indicators
                        features['plus_di'] = adx_result[f'DMP_{self.adx_period}']
                        features['minus_di'] = adx_result[f'DMN_{self.adx_period}']
                        features['di_difference'] = features['plus_di'] - features['minus_di']
                        features['di_ratio'] = features['plus_di'] / (features['minus_di'] + 1e-10)
                    except Exception:
                        for col_name in ['adx', 'adx_strong_trend', 'adx_momentum', 'plus_di', 
                                        'minus_di', 'di_difference', 'di_ratio']:
                            features[col_name] = pd.Series([np.nan] * n, index=price_data.index)
                else:
                    for col_name in ['adx', 'adx_strong_trend', 'adx_momentum', 'plus_di', 
                                    'minus_di', 'di_difference', 'di_ratio']:
                        features[col_name] = pd.Series([np.nan] * n, index=price_data.index)
            else:
                for col_name in ['adx', 'adx_strong_trend', 'adx_momentum', 'plus_di', 
                                'minus_di', 'di_difference', 'di_ratio']:
                    features[col_name] = pd.Series([np.nan] * n, index=price_data.index)
            
            # Moving average features
            ema_10 = ta.ema(close, length=10)
            ema_20 = ta.ema(close, length=20)
            ema_50 = ta.ema(close, length=50)
            
            if ema_10 is not None and ema_20 is not None:
                features['ma_distance_ratio_10_20'] = (ema_10 - ema_20) / (ema_20 + 1e-10)
            else:
                features['ma_distance_ratio_10_20'] = pd.Series([np.nan] * n, index=price_data.index)
            
            if ema_20 is not None and ema_50 is not None:
                features['ma_distance_ratio_20_50'] = (ema_20 - ema_50) / (ema_50 + 1e-10)
            else:
                features['ma_distance_ratio_20_50'] = pd.Series([np.nan] * n, index=price_data.index)
            
            # Trend consistency (% time price above MA)
            if ema_20 is not None:
                price_above_ma = (close > ema_20).astype(float)
                features['trend_consistency'] = price_above_ma.rolling(window=20).mean()
            else:
                features['trend_consistency'] = pd.Series([np.nan] * n, index=price_data.index)
            
        except Exception as e:
            logger.error(f"Error computing advanced trend features: {e}")
            # Fill with NaN on error
            for col in features.columns:
                if col not in features or features[col].isna().all():
                    features[col] = pd.Series([np.nan] * n, index=price_data.index)
        
        return features


class SupportResistanceFeatures:
    """Extract support/resistance features for improved prediction."""
    
    def compute(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute support/resistance features.
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with support/resistance features
        """
        features = pd.DataFrame(index=price_data.index)
        n = len(price_data)
        
        try:
            close = price_data['close']
            high = price_data.get('high', close)
            low = price_data.get('low', close)
            
            # Distance from recent highs
            for period in [10, 20, 50]:
                recent_high = high.rolling(window=period).max()
                features[f'distance_from_high_{period}'] = (close - recent_high) / (recent_high + 1e-10)
            
            # Distance from recent lows
            for period in [10, 20, 50]:
                recent_low = low.rolling(window=period).min()
                features[f'distance_from_low_{period}'] = (close - recent_low) / (recent_low + 1e-10)
            
            # Range position (where price is in recent range)
            for period in [20, 50]:
                recent_high = high.rolling(window=period).max()
                recent_low = low.rolling(window=period).min()
                range_size = recent_high - recent_low
                features[f'range_position_{period}'] = (close - recent_low) / (range_size + 1e-10)
            
        except Exception as e:
            logger.error(f"Error computing support/resistance features: {e}")
            # Fill with NaN on error
            for col in features.columns:
                if col not in features or features[col].isna().all():
                    features[col] = pd.Series([np.nan] * n, index=price_data.index)
        
        return features


class FeatureEngineeringPipeline:
    """
    Advanced feature engineering pipeline for regime prediction.
    
    Combines multiple feature extraction methods to create a comprehensive
    feature set for machine learning models.
    """

    # ==================== KESİN ÇÖZÜM ADIM 1: SABİT ÖZELLİK LİSTESİ ====================
    # Bu liste, modelin eğitildiği özelliklerin tam ve sıralı listesidir.
    # Scaler'ın ve modelin beklediği "altın standart" budur.
    # ÖNEMLİ: Eğer train_all_models.py'de yeni özellikler eklenirse, bu liste güncellenmelidir!
    FEATURE_COLUMNS = [
        'technical_rsi', 'technical_rsi_oversold', 'technical_rsi_overbought', 'technical_macd',
        'technical_macd_signal', 'technical_macd_histogram', 'technical_macd_cross', 'technical_ema_20',
        'technical_ema_50', 'technical_ema_cross', 'technical_bb_upper', 'technical_bb_lower',
        'technical_bb_width', 'technical_bb_position', 'technical_atr', 'technical_atr_pct',
        'microstructure_price_range', 'microstructure_close_position', 'microstructure_volume',
        'microstructure_volume_ma', 'microstructure_volume_ratio', 'microstructure_returns_1',
        'microstructure_returns_5', 'microstructure_returns_10', 'volatility_vol_5',
        'volatility_parkinson_vol_5', 'volatility_vol_10', 'volatility_parkinson_vol_10',
        'volatility_vol_20', 'volatility_parkinson_vol_20', 'volatility_vol_50',
        'volatility_parkinson_vol_50', 'volatility_vol_regime', 'momentum_roc_5',
        'momentum_ma_slope_5', 'momentum_roc_10', 'momentum_ma_slope_10', 'momentum_roc_20',
        'momentum_ma_slope_20', 'momentum_roc_50', 'momentum_ma_slope_50', 'momentum_momentum_regime'
    ]
    # =================================================================================
    
    @staticmethod
    def _parse_window_list(window_str, default='5,10,20,50'):
        """
        Parse window list from various string formats.
        
        Handles multiple input formats:
        - Plain CSV: "5,10,20,50"
        - With brackets: "[5,10,20,50]"
        - With quotes: "['5','10','20','50']" or '["5","10","20","50"]'
        - Mixed: "[5, 10, 20, 50]" (with spaces)
        - Already a list: [5, 10, 20, 50]
        
        Args:
            window_str: Input string or list to parse
            default: Default CSV string to use if parsing fails
            
        Returns:
            List of integers representing window sizes
        """
        # If already a list, return as-is
        if not isinstance(window_str, str):
            return window_str if window_str else [int(x) for x in default.split(',')]
        
        # Handle empty strings by using default
        if not window_str.strip():
            return [int(x) for x in default.split(',')]
        
        try:
            # Remove all brackets, quotes, and extra spaces
            import re
            cleaned = re.sub(r'[\[\]"\'\s]', '', window_str)
            # Split by comma and convert to integers, filtering empty strings
            return [int(x) for x in cleaned.split(',') if x]
        except ValueError as e:
            logger.warning(f"Failed to parse window list '{window_str}': {e}. Using default: {default}")
            return [int(x) for x in default.split(',')]
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the feature engineering pipeline with config."""
        # Keep a reference to the raw config that was supplied by the caller.
        self.raw_config = config or {}

        # Normalize configuration so that `self.config` always points at the ML block.
        if isinstance(self.raw_config.get('ml'), dict):
            self.config = self.raw_config.get('ml') or {}
        else:
            self.config = self.raw_config

        # Feature-specific overrides live under ml.features; fall back to empty dict.
        self.features_config = self.config.get('features', {}) or {}

        # Model bundle configuration may be provided alongside ml or may require a reload.
        models_cfg = None
        if isinstance(self.raw_config.get('models'), dict):
            models_cfg = self.raw_config.get('models')
        elif isinstance(self.config.get('models'), dict):
            models_cfg = self.config.get('models')

        if not isinstance(models_cfg, dict) or not models_cfg:
            try:
                try:
                    from src.config.live_trading_config import LiveTradingConfiguration
                except ModuleNotFoundError:
                    from config.live_trading_config import LiveTradingConfiguration

                global_config = LiveTradingConfiguration.load(log_summary=False)
                models_cfg = global_config.get('models', {}) or {}
            except Exception as exc:
                logger.warning(
                    "Model bundle configuration not supplied; defaulting to legacy manifest (%s)",
                    exc
                )
                models_cfg = {}

        self.models_config = models_cfg

        logger.info(
            "FeatureEngineeringPipeline initialized with config keys: %s",
            list(self.config.keys())
        )
        
        # Initialize ManifestManager for dynamic feature configuration
        from .manifest_manager import ManifestManager
        self.manifest_mgr = ManifestManager()
        
        # Load manifest from active bundle
        bundle_path = self.models_config.get('active_bundle', 'artifacts/legacy')
        try:
            self.manifest = self.manifest_mgr.load_manifest(bundle_path)
            self.expected_feature_count = self.manifest['feature_count']
            self.manifest_mode = self.manifest.get('mode', 'legacy')
            
            # Get selected features for each mode
            self.price_features = self.manifest_mgr.get_selected_features('price')
            self.regime_features = self.manifest_mgr.get_selected_features('regime')
            
            logger.info(f"✅ FeatureEngineering initialized with manifest: {self.manifest.get('version')}")
            logger.info(f"   Manifest mode: {self.manifest_mode}")
            logger.info(f"   Expected feature count: {self.expected_feature_count}")
            logger.info(f"   Price features: {len(self.price_features)}")
            logger.info(f"   Regime features: {len(self.regime_features)}")
        except Exception as e:
            logger.warning(f"Failed to load manifest, using defaults: {e}")
            self.manifest = {'feature_count': 42, 'mode': 'legacy'}
            self.manifest_mode = 'legacy'
            self.expected_feature_count = 42
            self.price_features = list(range(42))
            self.regime_features = list(range(42))
        
        # GEMMA integration: auto-enable when GEMMA manifest is active unless explicitly disabled
        gemma_config = self.config.get('gemma', {})
        gemma_model_overrides = self.models_config.get('gemma', {})
        gemma_requested = gemma_config.get('enabled', gemma_model_overrides.get('use_manifest'))

        if self.manifest_mode != 'legacy':
            if gemma_requested is False:
                logger.info(
                    "GEMMA manifest detected but disabled via config; running legacy extractor despite GEMMA bundle."
                )
                self.gemma_enabled = False
            else:
                self.gemma_enabled = True
        else:
            if gemma_requested:
                logger.warning(
                    "GEMMA feature extraction requested but active manifest is 'legacy'. "
                    "Falling back to legacy 42-feature pipeline until GEMMA bundle is activated."
                )
            self.gemma_enabled = False
        
        # Determine if we should use advanced features (default: True for new training)
        self.use_advanced_features = self.config.get('use_advanced_features', True)
        # Determine if we should align to legacy feature set (default: False for new models)
        self.use_legacy_alignment = self.config.get('use_legacy_alignment', False)
        
        # Pass config to sub-components
        self.technical_indicators = TechnicalIndicatorFeatures(self.features_config)
        self.market_microstructure = MarketMicrostructureFeatures()
        
        # Parse volatility windows from config - handle multiple formats
        vol_windows = self._parse_window_list(
            self.features_config.get('volatility_windows', '5,10,20,50'),
            default='5,10,20,50'
        )
        self.volatility_features = VolatilityFeatures(windows=vol_windows)
        
        # Parse momentum windows from config - handle multiple formats
        mom_windows = self._parse_window_list(
            self.features_config.get('momentum_windows', '5,10,20,50'),
            default='5,10,20,50'
        )
        self.momentum_features = MomentumFeatures(windows=mom_windows)
        
        self.cross_asset_features = CrossAssetFeatures()
        
        # Initialize advanced feature extractors
        self.advanced_momentum = AdvancedMomentumFeatures()
        self.advanced_volume = AdvancedVolumeFeatures()
        self.advanced_volatility = AdvancedVolatilityFeatures(self.features_config)
        self.advanced_trend = AdvancedTrendFeatures(self.features_config)
        self.support_resistance = SupportResistanceFeatures()
        
    def extract_features(self, price_data: pd.DataFrame, 
                        volume_data: Optional[pd.DataFrame] = None,
                        orderbook_data: Optional[pd.DataFrame] = None,
                        mode: str = 'price') -> pd.DataFrame:
        """
        Main feature extraction method with manifest-based feature selection.
        
        Args:
            price_data: DataFrame with OHLCV data and indicators
            volume_data: Optional volume-specific data
            orderbook_data: Optional order book data
            mode: Feature selection mode - 'price', 'regime', or 'all' (default: 'price')
            
        Returns:
            DataFrame with extracted features based on manifest configuration
        """
        try:
            # 1. Extract full feature set based on mode
            if self.gemma_enabled:
                logger.debug("🧬 GEMMA mode enabled - extracting all 87 features")
                all_features = self.extract_gemma_features(price_data)
            else:
                logger.debug("📦 Legacy mode - extracting standard features")
                all_features = self.extract_legacy_features(price_data, volume_data, orderbook_data)
            
            # 2. Apply feature selection based on mode and manifest
            if mode == 'price':
                selected_columns = self.price_features
            elif mode == 'regime':
                selected_columns = self.regime_features
            else:
                # Return all features
                return all_features
            
            # 3. Select only required columns (by index or name)
            try:
                # If selected_columns contains indices, convert to column names
                if selected_columns and isinstance(selected_columns[0], int):
                    feature_cols = [all_features.columns[i] for i in selected_columns if i < len(all_features.columns)]
                else:
                    feature_cols = selected_columns
                
                selected_features = all_features[feature_cols]
                
                # 4. Validate output shape
                if len(selected_features.columns) != self.expected_feature_count:
                    logger.warning(
                        f"Feature count mismatch: {len(selected_features.columns)} != {self.expected_feature_count}"
                    )
                    # In lenient mode, continue; in strict mode this could raise
                    if self.models_config.get('deployment', {}).get('validation_mode') == 'strict':
                        raise ValueError(
                            f"Feature count mismatch: {len(selected_features.columns)} != {self.expected_feature_count}"
                        )
                
                logger.debug(f"✅ Extracted {len(selected_features.columns)} features for mode={mode}")
                return selected_features
                
            except (KeyError, IndexError) as e:
                logger.error(f"Feature selection failed: {e}")
                # Return all features as fallback
                logger.warning(f"Returning all {len(all_features.columns)} features as fallback")
                return all_features
                
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}", exc_info=True)
            return pd.DataFrame()

    def extract_advanced_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all advanced features from price data.
        
        This method calls all advanced feature extraction sub-methods and combines
        them into a single DataFrame with appropriate prefixes.
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all advanced features
        """
        advanced_features = pd.DataFrame(index=price_data.index)
        
        try:
            logger.info("Extracting advanced features...")
            
            # Extract advanced momentum features
            logger.debug("Computing advanced momentum features...")
            momentum_adv = self.advanced_momentum.compute(price_data)
            if not momentum_adv.empty:
                momentum_adv = momentum_adv.add_prefix('advanced_momentum_')
                advanced_features = pd.concat([advanced_features, momentum_adv], axis=1)
            
            # Extract advanced volume features
            logger.debug("Computing advanced volume features...")
            volume_adv = self.advanced_volume.compute(price_data)
            if not volume_adv.empty:
                volume_adv = volume_adv.add_prefix('advanced_volume_')
                advanced_features = pd.concat([advanced_features, volume_adv], axis=1)
            
            # Extract advanced volatility features
            logger.debug("Computing advanced volatility features...")
            volatility_adv = self.advanced_volatility.compute(price_data)
            if not volatility_adv.empty:
                volatility_adv = volatility_adv.add_prefix('advanced_volatility_')
                advanced_features = pd.concat([advanced_features, volatility_adv], axis=1)
            
            # Extract advanced trend features
            logger.debug("Computing advanced trend features...")
            trend_adv = self.advanced_trend.compute(price_data)
            if not trend_adv.empty:
                trend_adv = trend_adv.add_prefix('advanced_trend_')
                advanced_features = pd.concat([advanced_features, trend_adv], axis=1)
            
            # Extract support/resistance features
            logger.debug("Computing support/resistance features...")
            sr_features = self.support_resistance.compute(price_data)
            if not sr_features.empty:
                sr_features = sr_features.add_prefix('support_resistance_')
                advanced_features = pd.concat([advanced_features, sr_features], axis=1)
            
            logger.info(f"✅ Extracted {len(advanced_features.columns)} advanced features")
            
        except Exception as e:
            logger.error(f"Error extracting advanced features: {e}", exc_info=True)
        
        return advanced_features

    # ==================== KESİN ÇÖZÜM ADIM 2: YENİ HİZALAMA METODU ====================
    def align_and_finalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aligns the DataFrame to match the FEATURE_COLUMNS structure.
        
        This ensures that the feature set is always consistent for the scaler and model.
        It adds missing columns as NaN and reorders existing columns.
        
        Args:
            df: The DataFrame with dynamically generated features.
            
        Returns:
            A new DataFrame that is perfectly aligned with FEATURE_COLUMNS.
        """
        # Gelen DataFrame'i kopyalayarak orijinalini bozmayalım
        aligned_df = df.copy()

        # Eksik sütunları bul ve NaN olarak ekle
        missing_cols = set(self.FEATURE_COLUMNS) - set(aligned_df.columns)
        for col in missing_cols:
            aligned_df[col] = np.nan
            logger.debug(f"Added missing feature column '{col}' as NaN.")

        # Sadece FEATURE_COLUMNS'da olanları ve doğru sırada al
        # `reindex` metodu bu işi tek adımda ve güvenli bir şekilde yapar.
        final_df = aligned_df.reindex(columns=self.FEATURE_COLUMNS)
        
        return final_df
    
    # ==================== GEMMA FEATURE ENGINEERING METHODS ====================
    
    def calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def calculate_stochastic(self, df: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator."""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        stoch_d = stoch_k.rolling(window=3).mean()
        return stoch_k, stoch_d
    
    def calculate_williams_r(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Williams %R indicator."""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        return -100 * (high_max - df['close']) / (high_max - low_min + 1e-10)
    
    def calculate_obv(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv.rolling(window=period).mean()
    
    def calculate_mfi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))
        return mfi
    
    def calculate_vwap(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    
    def calculate_bollinger_bands(self, series: pd.Series, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (2 * std)
        lower = middle - (2 * std)
        return upper, middle, lower
    
    def calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def calculate_keltner_channels(self, df: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate Keltner Channels."""
        ema = df['close'].ewm(span=period).mean()
        atr = self.calculate_atr(df, period)
        upper = ema + (2 * atr)
        lower = ema - (2 * atr)
        return upper, lower
    
    def calculate_donchian(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Donchian Channel midpoint."""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        return (high_max + low_min) / 2
    
    def calculate_macd(self, series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = series.ewm(span=12).mean()
        ema_slow = series.ewm(span=26).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ADX indicator."""
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self.calculate_atr(df, 1)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (tr.rolling(window=period).mean() + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (tr.rolling(window=period).mean() + 1e-10))
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()
        return adx
    
    def calculate_plus_di(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate +DI indicator."""
        plus_dm = df['high'].diff()
        plus_dm[plus_dm < 0] = 0
        tr = self.calculate_atr(df, 1)
        return 100 * (plus_dm.rolling(window=period).mean() / (tr.rolling(window=period).mean() + 1e-10))
    
    def calculate_minus_di(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate -DI indicator."""
        minus_dm = -df['low'].diff()
        minus_dm[minus_dm < 0] = 0
        tr = self.calculate_atr(df, 1)
        return 100 * (minus_dm.rolling(window=period).mean() / (tr.rolling(window=period).mean() + 1e-10))
    
    def calculate_cci(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = (typical_price - sma).abs().rolling(window=period).mean()
        return (typical_price - sma) / (0.015 * mad + 1e-10)
    
    def calculate_roc(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Rate of Change."""
        return series.pct_change(period) * 100
    
    def calculate_momentum(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Momentum."""
        return series.diff(period)
    
    def calculate_trix(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate TRIX indicator."""
        ema1 = series.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        return ema3.pct_change() * 100
    
    def calculate_dpo(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Detrended Price Oscillator."""
        sma = series.rolling(window=period).mean()
        return series.shift(int(period/2) + 1) - sma
    
    def calculate_vortex(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Vortex Indicator (positive)."""
        vm_plus = np.abs(df['high'] - df['low'].shift(1))
        vm_minus = np.abs(df['low'] - df['high'].shift(1))
        tr = self.calculate_atr(df, 1)
        vi_plus = vm_plus.rolling(window=period).sum() / (tr.rolling(window=period).sum() + 1e-10)
        return vi_plus
    
    def calculate_support_resistance(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate support and resistance levels."""
        support = df['low'].rolling(window=20).min()
        resistance = df['high'].rolling(window=20).max()
        return support, resistance
    
    def calculate_pivot_points(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate pivot points."""
        pivot = (df['high'] + df['low'] + df['close']) / 3
        r1 = 2 * pivot - df['low']
        s1 = 2 * pivot - df['high']
        return {'pivot': pivot, 'r1': r1, 's1': s1}
    
    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Fibonacci retracement levels."""
        high_20 = df['high'].rolling(window=20).max()
        low_20 = df['low'].rolling(window=20).min()
        diff = high_20 - low_20
        return {
            '38.2': high_20 - (0.382 * diff),
            '50.0': high_20 - (0.5 * diff),
            '61.8': high_20 - (0.618 * diff)
        }
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength."""
        ema_20 = df['close'].ewm(span=20).mean()
        ema_50 = df['close'].ewm(span=50).mean()
        return (ema_20 - ema_50) / (ema_50 + 1e-10)
    
    def calculate_market_phase(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market phase (0=accumulation, 1=uptrend, 2=distribution, 3=downtrend)."""
        rsi = self.calculate_rsi(df['close'], 14)
        adx = self.calculate_adx(df, 14)
        
        # Simple phase detection based on RSI and ADX
        phase = pd.Series(0, index=df.index)
        phase[(rsi < 50) & (adx < 25)] = 0  # Accumulation
        phase[(rsi >= 50) & (adx >= 25)] = 1  # Uptrend
        phase[(rsi >= 50) & (adx < 25)] = 2  # Distribution
        phase[(rsi < 50) & (adx >= 25)] = 3  # Downtrend
        
        return phase

    def extract_gemma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract 87 features for GEMMA pipeline.
        Maintains compatibility with existing 42-feature system.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with 87 features for GEMMA
        """
        features = pd.DataFrame(index=df.index)

        # Price-based features (30)
        for period in [5, 10, 15, 20, 30]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            features[f'rsi_{period}'] = self.calculate_rsi(df['close'], period)
            stoch_k, stoch_d = self.calculate_stochastic(df, period)
            features[f'stoch_k_{period}'] = stoch_k
            features[f'stoch_d_{period}'] = stoch_d
            features[f'williams_r_{period}'] = self.calculate_williams_r(df, period)

        # Volume-based features (15)
        for period in [5, 10, 15]:
            features[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = df['volume'] / (df['volume'].rolling(period).mean() + 1e-10)
            features[f'obv_{period}'] = self.calculate_obv(df, period)
            features[f'mfi_{period}'] = self.calculate_mfi(df, period)
            features[f'vwap_{period}'] = self.calculate_vwap(df, period)

        # Volatility features (20)
        for period in [10, 20]:
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['close'], period)
            features[f'bb_upper_{period}'] = bb_upper
            features[f'bb_middle_{period}'] = bb_middle
            features[f'bb_lower_{period}'] = bb_lower
            features[f'bb_width_{period}'] = bb_upper - bb_lower
            features[f'bb_position_{period}'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
            features[f'atr_{period}'] = self.calculate_atr(df, period)
            features[f'volatility_{period}'] = df['close'].rolling(period).std()
            keltner_upper, keltner_lower = self.calculate_keltner_channels(df, period)
            features[f'keltner_upper_{period}'] = keltner_upper
            features[f'keltner_lower_{period}'] = keltner_lower
            features[f'donchian_{period}'] = self.calculate_donchian(df, period)

        # Trend features (12)
        macd, signal, histogram = self.calculate_macd(df['close'])
        features['macd_line'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = histogram
        features['adx_14'] = self.calculate_adx(df, 14)
        features['plus_di_14'] = self.calculate_plus_di(df, 14)
        features['minus_di_14'] = self.calculate_minus_di(df, 14)
        features['cci_20'] = self.calculate_cci(df, 20)
        features['roc_10'] = self.calculate_roc(df['close'], 10)
        features['momentum_10'] = self.calculate_momentum(df['close'], 10)
        features['trix_15'] = self.calculate_trix(df['close'], 15)
        features['dpo_20'] = self.calculate_dpo(df['close'], 20)
        features['vortex_pos_14'] = self.calculate_vortex(df, 14)

        # Market structure features (10)
        support, resistance = self.calculate_support_resistance(df)
        features['support_distance'] = (df['close'] - support) / (df['close'] + 1e-10)
        features['resistance_distance'] = (resistance - df['close']) / (df['close'] + 1e-10)
        pivot = self.calculate_pivot_points(df)
        features['pivot_point'] = pivot['pivot']
        features['r1_level'] = pivot['r1']
        features['s1_level'] = pivot['s1']
        fib_levels = self.calculate_fibonacci_levels(df)
        features['fib_38'] = fib_levels['38.2']
        features['fib_50'] = fib_levels['50.0']
        features['fib_62'] = fib_levels['61.8']
        features['trend_strength'] = self.calculate_trend_strength(df)
        features['market_phase'] = self.calculate_market_phase(df)

        # Fill NaN values with forward fill, then zero
        features = features.ffill().fillna(0)
        
        # Validate feature count (informational, not strict)
        if features.shape[1] != 87:
            logger.warning(f"GEMMA features: expected 87, got {features.shape[1]}")
        
        logger.debug(f"✅ Extracted {features.shape[1]} GEMMA features")
        return features
    
    def extract_legacy_features(self, price_data: pd.DataFrame, 
                               volume_data: Optional[pd.DataFrame] = None,
                               orderbook_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract legacy 42-feature set (existing implementation).
        This is the original extract_features method behavior.
        
        Args:
            price_data: DataFrame with OHLCV data and indicators
            volume_data: Optional volume-specific data
            orderbook_data: Optional order book data
            
        Returns:
            DataFrame with legacy features
        """
        features = {}
        
        try:
            # Technical indicator features
            features['technical'] = self.technical_indicators.compute(price_data)
            
            # Market microstructure features
            features['microstructure'] = self.market_microstructure.compute(
                price_data, volume_data, orderbook_data
            )
            
            # Volatility regime features
            features['volatility'] = self._compute_volatility_features(price_data)
            
            # Momentum and trend features
            features['momentum'] = self._compute_momentum_features(price_data)
            
            # Combine all features
            combined_features = self._combine_features(features)
            
            # Extract and merge advanced features if enabled
            if self.use_advanced_features:
                logger.info("Extracting advanced features...")
                advanced_features = self.extract_advanced_features(price_data)
                if not advanced_features.empty:
                    combined_features = pd.concat([combined_features, advanced_features], axis=1)
            
            # Apply alignment for legacy compatibility if needed
            if self.use_legacy_alignment:
                finalized_features = self.align_and_finalize_features(combined_features)
            else:
                finalized_features = combined_features

            finalized_features.replace([np.inf, -np.inf], np.nan, inplace=True)

            logger.info(f"Extracted {len(finalized_features.columns)} legacy features from price data")
            return finalized_features
            
        except Exception as e:
            logger.error(f"Error in legacy feature extraction: {e}", exc_info=True)
            return pd.DataFrame()
    
    # ==================== END OF GEMMA METHODS ====================
    
    def _compute_volatility_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced volatility feature extraction.
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility features
        """
        return self.volatility_features.compute(price_data)
    
    def _compute_momentum_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Momentum and trend feature extraction.
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum features
        """
        return self.momentum_features.compute(price_data)
    
    def _combine_features(self, features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine all feature sets into a single DataFrame.
        
        Args:
            features: Dictionary of feature DataFrames
            
        Returns:
            Combined DataFrame with all features
        """
        combined = pd.DataFrame()
        
        for feature_type, feature_df in features.items():
            if not feature_df.empty:
                # Add prefix to avoid column name conflicts
                feature_df = feature_df.add_prefix(f'{feature_type}_')
                if combined.empty:
                    combined = feature_df
                else:
                    combined = pd.concat([combined, feature_df], axis=1)
        
        return combined

    def prepare_for_training(self, features: pd.DataFrame, 
                           labels: pd.Series,
                           feature_selection_mode: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for model training with PROPER ALIGNMENT.
        
        CRITICAL: This method MUST preserve temporal alignment between features and labels.
        DO NOT trim features and labels separately - this breaks prediction relationships!
        
        Args:
            features: DataFrame with features (can be 42, 82, or 87 features)
            labels: Series with regime labels (same index as features)
            feature_selection_mode: 
                'auto' - Use features as-is from input (respects feature selection)
                'legacy' - Use hardcoded 42 FEATURE_COLUMNS (backward compatibility)
                'all' - Use all available features (for testing)
        
        Returns:
            Tuple of (X, y) as numpy arrays, properly aligned
        """
        try:
            logger.info("="*70)
            logger.info("🔧 PREPARING DATA FOR TRAINING (PROPER ALIGNMENT)")
            logger.info("="*70)
            
            # =================================================================
            # STEP 1: JOIN FEATURES AND LABELS (PRESERVES ALIGNMENT!)
            # =================================================================
            logger.info("Step 1: Joining features and labels on index (preserves alignment)...")
            
            labels_df = labels.to_frame(name='label')
            combined_df = features.join(labels_df, how='inner')
            
            if combined_df.empty:
                logger.error("❌ No data remains after inner join of features and labels!")
                logger.error("   Check: Do features and labels have matching indices?")
                return np.array([]), np.array([])
            
            logger.info(f"   Joined {len(combined_df)} rows with matching indices")
            
            # =================================================================
            # STEP 2: DROP ROWS WITH NaN LABELS (PREDICTION HORIZON)
            # =================================================================
            logger.info("Step 2: Dropping rows with NaN labels (prediction horizon)...")
            
            initial_rows = len(combined_df)
            combined_df.dropna(subset=['label'], inplace=True)
            dropped = initial_rows - len(combined_df)
            
            if dropped > 0:
                logger.info(f"   Dropped {dropped} rows with NaN labels")
            
            if combined_df.empty:
                logger.error("❌ No data remains after dropping NaN labels!")
                return np.array([]), np.array([])
            
            # =================================================================
            # STEP 3: DETERMINE FEATURE COLUMNS BASED ON MODE
            # =================================================================
            logger.info(f"Step 3: Selecting features (mode: {feature_selection_mode})...")
            
            if feature_selection_mode == 'auto':
                # Use features as-is (respects feature selection)
                feature_columns = [col for col in combined_df.columns if col != 'label']
                logger.info(f"   🎯 AUTO MODE: Using {len(feature_columns)} features from input")
                
            elif feature_selection_mode == 'legacy':
                # Use hardcoded 42 features (backward compatibility)
                # Check which are available
                available_features = [col for col in self.FEATURE_COLUMNS if col in combined_df.columns]
                if len(available_features) < len(self.FEATURE_COLUMNS):
                    missing = set(self.FEATURE_COLUMNS) - set(available_features)
                    logger.warning(f"   ⚠️ Missing {len(missing)} legacy features: {list(missing)[:5]}...")
                feature_columns = available_features
                logger.info(f"   📦 LEGACY MODE: Using {len(feature_columns)} hardcoded features")
                
            elif feature_selection_mode == 'all':
                # Use all features (for testing)
                feature_columns = [col for col in combined_df.columns if col != 'label']
                logger.info(f"   🔬 ALL MODE: Using {len(feature_columns)} features")
                
            else:
                raise ValueError(f"Invalid feature_selection_mode: {feature_selection_mode}")
            
            # =================================================================
            # STEP 4: DROP ROWS WITH NaN FEATURES (WARMUP PERIOD)
            # =================================================================
            logger.info("Step 4: Dropping rows with NaN features (warmup period)...")
            
            initial_rows = len(combined_df)
            combined_df.dropna(subset=feature_columns, inplace=True)
            dropped = initial_rows - len(combined_df)
            
            if dropped > 0:
                logger.info(f"   Dropped {dropped} rows with NaN features")
            
            if combined_df.empty:
                logger.error("❌ No data remains after dropping NaN features!")
                return np.array([]), np.array([])
            
            # =================================================================
            # STEP 5: CONVERT LABEL TO INTEGER
            # =================================================================
            combined_df['label'] = combined_df['label'].astype(int)
            
            # =================================================================
            # STEP 6: EXTRACT FINAL X AND Y (PERFECTLY ALIGNED!)
            # =================================================================
            X = combined_df[feature_columns].values
            y = combined_df['label'].values
            
            logger.info("="*70)
            logger.info("✅ DATA PREPARATION COMPLETE")
            logger.info("="*70)
            logger.info(f"   Final samples: {len(X)}")
            logger.info(f"   Features: {X.shape[1]}")
            logger.info(f"   Label distribution:")
            unique, counts = np.unique(y, return_counts=True)
            for label_val, count in zip(unique, counts):
                pct = count / len(y) * 100
                label_name = {0: 'Bullish', 1: 'Neutral', 2: 'Bearish'}.get(label_val, f'Class{label_val}')
                logger.info(f"      {label_name}: {count} ({pct:.1f}%)")
            logger.info("="*70)
            
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Error in prepare_for_training: {e}", exc_info=True)
            return np.array([]), np.array([])
