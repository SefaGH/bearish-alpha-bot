"""
Feature Engineering Pipeline for ML Market Regime Prediction.

Advanced feature extraction from market data for regime prediction models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicatorFeatures:
    """Extract technical indicator features from price data."""
    
    def __init__(self):
        """Initialize technical indicator feature extractor."""
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.atr_period = 14
    
    def compute(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicator features.
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicator features
        """
        features = pd.DataFrame(index=price_data.index)
        
        try:
            # RSI-based features
            if 'rsi' in price_data.columns:
                features['rsi'] = price_data['rsi']
                features['rsi_oversold'] = (price_data['rsi'] < 30).astype(float)
                features['rsi_overbought'] = (price_data['rsi'] > 70).astype(float)
            
            # MACD-based features
            if 'macd' in price_data.columns and 'macd_signal' in price_data.columns:
                features['macd'] = price_data['macd']
                features['macd_signal'] = price_data['macd_signal']
                features['macd_histogram'] = price_data['macd'] - price_data['macd_signal']
                features['macd_cross'] = np.sign(features['macd_histogram'])
            
            # EMA-based features
            if 'ema_20' in price_data.columns and 'ema_50' in price_data.columns:
                features['ema_20'] = price_data['ema_20']
                features['ema_50'] = price_data['ema_50']
                features['ema_cross'] = (price_data['ema_20'] > price_data['ema_50']).astype(float)
            
            # Bollinger Bands features
            if 'bb_upper' in price_data.columns and 'bb_lower' in price_data.columns:
                features['bb_upper'] = price_data['bb_upper']
                features['bb_lower'] = price_data['bb_lower']
                bb_range = price_data['bb_upper'] - price_data['bb_lower']
                features['bb_width'] = bb_range / price_data['close']
                features['bb_position'] = (price_data['close'] - price_data['bb_lower']) / bb_range
            
            # ATR features
            if 'atr' in price_data.columns:
                features['atr'] = price_data['atr']
                features['atr_pct'] = price_data['atr'] / price_data['close']
            
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
    
    def compute(self, price_data: pd.DataFrame, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Compute volatility features across multiple windows.
        
        Args:
            price_data: DataFrame with OHLCV data
            windows: List of window sizes for volatility calculation
            
        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=price_data.index)
        
        try:
            returns = price_data['close'].pct_change()
            
            for window in windows:
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
    
    def compute(self, price_data: pd.DataFrame, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Compute momentum and trend features.
        
        Args:
            price_data: DataFrame with OHLCV data
            windows: List of window sizes for momentum calculation
            
        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=price_data.index)
        
        try:
            for window in windows:
                # Rate of change
                features[f'roc_{window}'] = price_data['close'].pct_change(window)
                
                # Moving average slope
                ma = price_data['close'].rolling(window=window).mean()
                features[f'ma_slope_{window}'] = ma.pct_change(1)
            
            # Trend strength
            if 'ema_20' in price_data.columns and 'ema_50' in price_data.columns:
                features['trend_strength'] = (price_data['ema_20'] - price_data['ema_50']) / price_data['close']
            
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


class FeatureEngineeringPipeline:
    """
    Advanced feature engineering pipeline for regime prediction.
    
    Combines multiple feature extraction methods to create a comprehensive
    feature set for machine learning models.
    """
    
    def __init__(self):
        """Initialize the feature engineering pipeline."""
        self.technical_indicators = TechnicalIndicatorFeatures()
        self.market_microstructure = MarketMicrostructureFeatures()
        self.volatility_features = VolatilityFeatures()
        self.momentum_features = MomentumFeatures()
        self.cross_asset_features = CrossAssetFeatures()
        
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
            
            logger.info(f"Extracted {len(combined_features.columns)} features from price data")
            return combined_features
            
        except Exception as e:
            logger.error(f"Error in feature extraction pipeline: {e}")
            return pd.DataFrame()
    
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
                           labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for model training.
        
        Args:
            features: DataFrame with extracted features
            labels: Series with regime labels
            
        Returns:
            Tuple of (features_array, labels_array)
        """
        # Align features and labels on index
        common_idx = features.index.intersection(labels.index)
        features_aligned = features.loc[common_idx]
        labels_aligned = labels.loc[common_idx]
        
        # Remove NaN values
        valid_idx = ~(features_aligned.isna().any(axis=1) | labels_aligned.isna())
        features_clean = features_aligned[valid_idx]
        labels_clean = labels_aligned[valid_idx]
        
        # Convert to numpy arrays
        X = features_clean.values
        y = labels_clean.values
        
        logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features for training")
        
        return X, y
