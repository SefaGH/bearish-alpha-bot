"""
Feature Engineering Pipeline for ML Market Regime Prediction.

Advanced feature extraction from market data for regime prediction models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
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
        
        try:
            # RSI
            features['rsi'] = ta.rsi(price_data['close'], length=self.rsi_period)
            features['rsi_oversold'] = (features['rsi'] < 30).astype(float)
            features['rsi_overbought'] = (features['rsi'] > 70).astype(float)
            
            # MACD
            macd = ta.macd(price_data['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            features['macd'] = macd[f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            features['macd_signal'] = macd[f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            features['macd_histogram'] = macd[f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            features['macd_cross'] = np.sign(features['macd_histogram'])
            
            # EMA
            features['ema_20'] = ta.ema(price_data['close'], length=20)
            features['ema_50'] = ta.ema(price_data['close'], length=50)
            features['ema_cross'] = (features['ema_20'] > features['ema_50']).astype(float)
            
            # Bollinger Bands
            bbands = ta.bbands(price_data['close'], length=self.bb_period)
            features['bb_upper'] = bbands[f'BBU_{self.bb_period}_2.0']
            features['bb_lower'] = bbands[f'BBL_{self.bb_period}_2.0']
            bb_range = features['bb_upper'] - features['bb_lower']
            features['bb_width'] = bb_range / price_data['close']
            features['bb_position'] = (price_data['close'] - features['bb_lower']) / (bb_range + 1e-10)
            
            # ATR
            features['atr'] = ta.atr(price_data['high'], price_data['low'], price_data['close'], length=self.atr_period)
            features['atr_pct'] = features['atr'] / price_data['close']
            
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
            
            # --- 🔥🔥🔥 NİHAİ DÜZELTME: 'trend_strength' HESAPLAMASI ---
            # 'ema_20' ve 'ema_50' sütunlarına bağımlı olmak yerine,
            # bu değerleri doğrudan burada hesapla.
            ema20 = ta.ema(price_data['close'], length=20)
            ema50 = ta.ema(price_data['close'], length=50)
            
            # `ema20` veya `ema50` NaN değilse hesapla
            if ema20 is not None and ema50 is not None:
                features['trend_strength'] = (ema20 - ema50) / price_data['close']
            else:
                features['trend_strength'] = np.nan # Hesaplama başarısız olursa NaN ata
            # --- 🔥🔥🔥 DÜZELTME SONU ---

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
            
            # ==================== KESİN ÇÖZÜM ADIM 3: FİNAL HİZALAMA ====================
            # Özellikleri, scaler'ın beklediği kesin formata getir.
            finalized_features = self.align_and_finalize_features(combined_features)
            # ==========================================================================

            finalized_features.replace([np.inf, -np.inf], np.nan, inplace=True)
            # NOT: Buradaki dropna(), tahmin sırasında en son satırı kaybedebileceği için
            # regime_predictor içinde yapılması daha güvenlidir. Bu yüzden buradan kaldırıyoruz.
            # finalized_features.dropna(inplace=True)

            logger.info(f"Extracted and aligned {len(finalized_features.columns)} features from price data")
            return finalized_features
            
        except Exception as e:
            logger.error(f"Error in feature extraction pipeline: {e}", exc_info=True)
            return pd.DataFrame()

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
                           labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for model training by aligning, cleaning, and converting them.
        (Nihai ve en sağlamlaştırılmış versiyon)
        """
        try:
            # 1. Özellikleri, sabit listeye göre hizala ve sırala.
            features_aligned = self.align_and_finalize_features(features)
            
            # 2. Etiketleri bir DataFrame'e dönüştür ve adını 'label' yap.
            labels_df = labels.to_frame(name='label')
            
            # 3. Özellikler ve etiketleri birleştir.
            combined_df = pd.concat([features_aligned, labels_df], axis=1)
            
            # 4. SADECE etiketi olmayan (NaN) satırları sil.
            combined_df.dropna(subset=['label'], inplace=True)
            
            if combined_df.empty:
                logger.warning("No data remains after dropping rows with missing labels.")
                return np.array([]), np.array([])
            
            # 5. Etiketleri tamsayıya dönüştür.
            combined_df['label'] = combined_df['label'].astype(int)
            
            # --- 🔥🔥🔥 NİHAİ DÜZELTME: Veri Temizleme Mantığı ---
            # 6. Özellik (X) tarafındaki NaN değerleri doldur.
            # Önce ffill (ileri doldurma), sonra bfill (geri doldurma).
            # Bu, hem baştaki hem de ortadaki boşlukları doldurmayı garanti eder.
            feature_columns = self.FEATURE_COLUMNS
            combined_df[feature_columns] = combined_df[feature_columns].ffill().bfill()
            
            # 7. Bu adımdan sonra hala NaN kalıyorsa, bu satırlar gerçekten sorunludur.
            # Bu yüzden bu satırları atıyoruz. Bu işlem artık tüm tabloyu silmemeli.
            initial_shape = combined_df.shape[0]
            combined_df.dropna(inplace=True)
            final_shape = combined_df.shape[0]

            if initial_shape > final_shape:
                 logger.warning(f"Dropped {initial_shape - final_shape} rows that still contained NaNs after ffill/bfill.")
            # --- 🔥🔥🔥 DÜZELTME SONU ---

            # 8. Nihai X ve y'yi oluştur.
            if combined_df.empty:
                logger.warning("After all cleaning steps, no data remains for training.")
                return np.array([]), np.array([])
                
            X = combined_df[feature_columns].values
            y = combined_df['label'].values
            
            logger.info(f"✅ Prepared {len(X)} samples with {X.shape[1]} features for training")
            return X, y
            
        except Exception as e:
            logger.error(f"Veri hazırlama hatası: {e}", exc_info=True)
            return np.array([]), np.array([])
