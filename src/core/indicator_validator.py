"""
Indicator Warmup Validation Module
Ensures all technical indicators are properly calculated before trading begins.
(MİMARİ GÜNCELLEME: Artık doğrudan MarketDataPipeline üzerinden çalışır)
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime, timezone

# MarketDataPipeline'ı tip ipucu için import ediyoruz.
from .market_data_pipeline import MarketDataPipeline

logger = logging.getLogger(__name__)

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not found. Indicator validation will be limited.")


class IndicatorValidator:
    """
    Validates that all required technical indicators are properly warmed up
    and returning valid values.
    (MİMARİ GÜNCELLEME: Artık ana veri kaynağı olarak MarketDataPipeline kullanır)
    """
    
    REQUIRED_CANDLES = 250  # İndikatörler için gereken minimum mum sayısı
    
    def __init__(self, pipeline: MarketDataPipeline):
        """
        Initialize the validator with a reference to the MarketDataPipeline.

        Args:
            pipeline (MarketDataPipeline): The central data pipeline for the bot.
        """
        self.pipeline = pipeline

        if not self.pipeline:
            logger.error(
                "❌ CRITICAL: IndicatorValidator cannot function without a MarketDataPipeline instance."
            )
            raise ValueError("IndicatorValidator requires a valid MarketDataPipeline.")
        
    async def validate_all(
        self, 
        symbols: List[str], 
        timeframes: List[str]
    ) -> Dict[str, Dict]:
        """
        Validates all specified symbols and their most critical timeframe.
        """
        logger.info("="*80)
        logger.info("🔍 INDICATOR WARMUP VERIFICATION (POST-PREFETCH)")
        logger.info("="*80)
        
        if not TALIB_AVAILABLE:
            logger.error("❌ TA-Lib is not installed. Cannot perform indicator validation.")
            results = {s: {'status': 'FAIL', 'reason': 'TA-Lib not installed'} for s in symbols}
            self._log_validation_summary(results)
            return results

        tasks = [self.validate_symbol(symbol, timeframes) for symbol in symbols]
        validation_outputs = await asyncio.gather(*tasks)
        
        results = {symbol: output for symbol, output in zip(symbols, validation_outputs)}
            
        self._log_validation_summary(results)
        return results
    
    async def validate_symbol(
        self, 
        symbol: str, 
        timeframes: List[str]
    ) -> Dict:
        """
        Validates the indicators for a single symbol using the MarketDataPipeline.
        """
        logger.info(f"\n📊 Validating indicators for {symbol}...")
        results = {'status': 'FAIL', 'reason': 'Unknown failure'}
        
        validation_tf = '1m' if '1m' in timeframes else timeframes[0]
        
        # Adım 1: Veriyi MarketDataPipeline üzerinden talep et
        df = None
        try:
            # Pipeline, WebSocket veya REST API'den veriyi getirecektir.
            df = await self.pipeline.get_latest_ohlcv(
                symbol=symbol, 
                timeframe=validation_tf
            )
            
            if df is None or df.empty or len(df) < self.REQUIRED_CANDLES:
                live_data_count = len(df) if df is not None else 0
                reason = (f"Insufficient data from pipeline for '{validation_tf}': "
                          f"found {live_data_count}, required {self.REQUIRED_CANDLES}. "
                          "This could mean both WebSocket priming and REST fallback failed.")
                logger.error(f"❌ {symbol}: {reason}")
                results['reason'] = reason
                return results

            logger.info(f"✅ Data availability check passed: {len(df)} candles found via pipeline for validation.")
            
        except Exception as e:
            reason = f"Failed to retrieve data from MarketDataPipeline: {e}"
            logger.error(f"❌ {symbol}: {reason}", exc_info=True)
            results['reason'] = reason
            return results

        # Adım 2: İndikatörleri Doğrula
        try:
            # MarketDataPipeline zaten indikatörleri eklemiş olmalı.
            # Burada bu indikatörlerin varlığını ve geçerliliğini kontrol ediyoruz.
            required_indicators = ['RSI_14', 'ATR_14', 'EMA_200']
            missing_indicators = [ind for ind in required_indicators if ind not in df.columns]

            if missing_indicators:
                reason = f"Indicators missing from DataFrame: {missing_indicators}. Pipeline may have failed to add them."
                logger.error(f"❌ {symbol}: {reason}")
                results['reason'] = reason
                return results

            indicator_errors = []
            
            # Son değerlerin NaN olup olmadığını kontrol et
            if pd.isna(df['RSI_14'].iloc[-1]):
                indicator_errors.append("RSI is NaN at the last candle.")
            if pd.isna(df['ATR_14'].iloc[-1]):
                indicator_errors.append("ATR is NaN at the last candle.")
            if pd.isna(df['EMA_200'].iloc[-1]):
                indicator_errors.append("EMA_200 is NaN at the last candle.")

            if indicator_errors:
                reason = "Indicator validation failed: " + ", ".join(indicator_errors)
                logger.error(f"❌ {symbol}: {reason}")
                results['reason'] = reason
                return results

        except Exception as e:
            reason = f"Indicator validation on DataFrame failed: {e}"
            logger.error(f"❌ {symbol}: {reason}", exc_info=True)
            results['reason'] = reason
            return results
            
        # Adım 3: Başarı Durumunu Raporla
        logger.info(f"✅ {symbol}: All indicators seem healthy and ready.")
        results['status'] = 'OK'
        results['reason'] = 'All indicators validated successfully.'
        return results

    def _log_validation_summary(self, results: Dict):
        logger.info("\n" + "="*80)
        logger.info("📋 INDICATOR VALIDATION SUMMARY")
        logger.info("="*80)
        
        total = len(results)
        valid = sum(1 for res in results.values() if res['status'] == 'OK')
        
        logger.info(f"Total Symbols: {total}, Valid Symbols: {valid}, Failed Symbols: {total - valid}")
        
        if valid != total:
            logger.error("❌ SOME INDICATORS FAILED VALIDATION:")
            for symbol, res in results.items():
                if res['status'] != 'OK':
                    logger.error(f"  - {symbol}: {res['reason']}")
        else:
            logger.info("✅ ALL INDICATORS READY FOR TRADING")
            
        logger.info("="*80)

    # --- ML VALIDATION METHODS (UNCHANGED from original) ---
    def validate_ml_data(self, price_data: pd.DataFrame, symbol: str) -> Tuple[bool, List[str]]:
        """
        Validate data quality for ML model consumption.
        """
        errors = []
        
        if price_data is None or price_data.empty:
            errors.append("Price data is None or empty")
            return False, errors
        
        min_rows = 50
        if len(price_data) < min_rows:
            errors.append(f"Insufficient data: {len(price_data)} rows (need {min_rows})")
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in price_data.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        for col in required_cols:
            if col in price_data.columns:
                if price_data[col].isna().sum() > 0:
                    errors.append(f"Found NaN values in '{col}'")
                if np.isinf(price_data[col]).sum() > 0:
                    errors.append(f"Found infinite values in '{col}'")

        for col in ['open', 'high', 'low', 'close']:
            if col in price_data.columns and (price_data[col] <= 0).sum() > 0:
                errors.append(f"Found zero/negative values in '{col}'")
        
        # Datetime index kontrolü daha sağlam hale getirildi.
        try:
            if isinstance(price_data.index, pd.DatetimeIndex):
                last_timestamp = price_data.index[-1]
            else:
                last_timestamp = pd.to_datetime(price_data['timestamp'].iloc[-1])
            
            if last_timestamp.tzinfo is None:
                last_timestamp = last_timestamp.tz_localize('UTC')
            
            age_minutes = (datetime.now(timezone.utc) - last_timestamp).total_seconds() / 60
            if age_minutes > 120:  # Allow up to 2 hours for larger timeframes
                errors.append(f"Stale data: last update was {age_minutes:.1f} minutes ago")
        except Exception as e:
            logger.debug(f"Could not check data freshness: {e}")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"🧠 [ML-VALIDATION] {symbol}: Data validation failed - {len(errors)} errors")
            for error in errors:
                logger.warning(f"   - {error}")
        else:
            logger.debug(f"🧠 [ML-VALIDATION] {symbol}: ✓ Data is clean and ready for ML")
        
        return is_valid, errors
    
    def validate_ml_features(self, features: pd.DataFrame, symbol: str) -> Tuple[bool, List[str]]:
        """
        Validate extracted ML features.
        """
        errors = []
        
        if features is None or features.empty:
            errors.append("Features DataFrame is None or empty")
            return False, errors
        
        nan_cols = features.columns[features.isna().any()].tolist()
        if nan_cols:
            errors.append(f"NaN values in features: {nan_cols}")
        
        inf_cols = []
        for col in features.select_dtypes(include=[np.number]).columns:
            if np.isinf(features[col]).any():
                inf_cols.append(col)
        if inf_cols:
            errors.append(f"Infinite values in features: {inf_cols}")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"🧠 [ML-VALIDATION] {symbol}: Feature validation failed")
            for error in errors:
                logger.warning(f"   - {error}")
        
        return is_valid, errors
