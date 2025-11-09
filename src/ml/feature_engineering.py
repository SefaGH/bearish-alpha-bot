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
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the feature engineering pipeline with config."""
        self.config = config or {}

        # Debug log ekle
        logger.info(f"FeatureEngineeringPipeline initialized with config: {list(self.config.keys())}")
        
        # Determine if we should use advanced features (default: True for new training)
        self.use_advanced_features = self.config.get('use_advanced_features', True)
        # Determine if we should align to legacy feature set (default: False for new models)
        self.use_legacy_alignment = self.config.get('use_legacy_alignment', False)
        
        # Pass config to sub-components
        self.technical_indicators = TechnicalIndicatorFeatures(self.config)
        self.market_microstructure = MarketMicrostructureFeatures()
        
        # Parse volatility windows from config
        vol_windows_str = self.config.get('volatility_windows', '5,10,20,50')
        if isinstance(vol_windows_str, str):
            vol_windows = [int(w.strip()) for w in vol_windows_str.split(',')]
        else:
            vol_windows = vol_windows_str
        self.volatility_features = VolatilityFeatures(windows=vol_windows)
        
        # Parse momentum windows from config
        mom_windows_str = self.config.get('momentum_windows', '5,10,20,50')
        if isinstance(mom_windows_str, str):
            mom_windows = [int(w.strip()) for w in mom_windows_str.split(',')]
        else:
            mom_windows = mom_windows_str
        self.momentum_features = MomentumFeatures(windows=mom_windows)
        
        self.cross_asset_features = CrossAssetFeatures()
        
        # Initialize advanced feature extractors
        self.advanced_momentum = AdvancedMomentumFeatures()
        self.advanced_volume = AdvancedVolumeFeatures()
        self.advanced_volatility = AdvancedVolatilityFeatures(self.config)
        self.advanced_trend = AdvancedTrendFeatures(self.config)
        self.support_resistance = SupportResistanceFeatures()
        
    def extract_features(self, price_data: pd.DataFrame, 
                        volume_data: Optional[pd.DataFrame] = None,
                        orderbook_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract comprehensive feature set for ML models.
        
        Args:
            price_data: DataFrame with OHLCV data and indicators
            volume_data: Optional volume-specific data
            orderbook_data: Optional order book data
            
        Returns:
            DataFrame with all extracted features
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
                # ==================== KESİN ÇÖZÜM ADIM 3: FİNAL HİZALAMA ====================
                # Özellikleri, scaler'ın beklediği kesin formata getir.
                finalized_features = self.align_and_finalize_features(combined_features)
                # ==========================================================================
            else:
                # Use all features without alignment (for new model training)
                finalized_features = combined_features

            finalized_features.replace([np.inf, -np.inf], np.nan, inplace=True)
            # NOT: Buradaki dropna(), tahmin sırasında en son satırı kaybedebileceği için
            # regime_predictor içinde yapılması daha güvenlidir. Bu yüzden buradan kaldırıyoruz.
            # finalized_features.dropna(inplace=True)

            logger.info(f"Extracted and aligned {len(finalized_features.columns)} features from price data")
            return finalized_features
            
        except Exception as e:
            logger.error(f"Error in feature extraction pipeline: {e}", exc_info=True)
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
