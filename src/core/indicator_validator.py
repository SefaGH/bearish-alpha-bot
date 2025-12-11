"""
Indicator Warmup Validation Module
Ensures all technical indicators are properly calculated before trading begins.
(GÜNCELLENDİ: Modern mimariyle uyumlu, eksiksiz sürüm)
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any

# StreamDataCollector'ı doğrudan import etmek, tip ipuçları ve doğrudan erişim için daha iyidir.
from .stream_data_collector import StreamDataCollector
from .volume_analyzer import VolumeAnalyzer

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
    and returning valid values after the prefetch step.
    (GÜNCELLENDİ: Artık doğrudan StreamDataCollector ile çalışır ve tüm ML doğrulama metotlarını içerir)
    """
    
    REQUIRED_CANDLES = 250  # İndikatörler için gereken minimum mum sayısı
    
    def __init__(self, collector: StreamDataCollector, rest_client: Any = None, config: Dict[str, Any] = None):
        """
        Initialize the validator with a direct reference to the data collector.

        Args:
            collector (StreamDataCollector): The central data store from WebSocketManager.
            rest_client (Any): A CcxtClient instance for potential future REST fallbacks.
        """
        self.collector = collector
        self.rest_client = rest_client # Şu an için kullanılmıyor ama gelecekteki geliştirmeler için saklanıyor.
        self.config = config or {}
        validator_cfg = self.config.get('validator', {}) if isinstance(self.config, dict) else {}
        self.volume_analyzer_required: bool = bool(validator_cfg.get('volume_analyzer_required', False))

        if not self.collector:
            # Bu durum artık bir hata olmalı, çünkü sistemin çalışması için collector şart.
            logger.error(
                "❌ CRITICAL: IndicatorValidator cannot function without a StreamDataCollector instance."
            )
            raise ValueError("IndicatorValidator requires a valid StreamDataCollector.")
        
    async def validate_all(
        self, 
        symbols: List[str], 
        timeframes: List[str]
    ) -> Dict[str, Dict]:
        """
        Validates all specified symbols and their most critical timeframe.
        
        Args:
            symbols: List of trading symbols to validate (e.g., ['BTC/USDT']).
            timeframes: List of all timeframes used, to ensure we check the right one.
        
        Returns:
            A dictionary with validation results for each symbol.
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
        Validates the indicators for a single symbol.
        (GÜNCELLENDİ: `asyncio.sleep` kaldırıldı, veri erişimi düzeltildi)
        """
        logger.info(f"\n📊 Validating indicators for {symbol}...")
        results = {'status': 'FAIL', 'reason': 'Unknown failure'}
        va_result = None
        
        # En kritik ve en hızlı güncellenen zaman dilimini kontrol et, genelde '1m' olur.
        validation_tf = '1m' if '1m' in timeframes else timeframes[0]
        
        # Adım 1: Veri Varlığını Doğrudan Collector'dan Kontrol Et
        try:
            # --- DÜZELTME: Artık collector'ın doğru metodu olan get_latest_ohlcv çağrılıyor ---
            # Bu metot doğrudan mum listesini ([timestamp, o, h, l, c, v]) döndürür.
            ohlcv_list = self.collector.get_latest_ohlcv(
                exchange='bingx',  # veya dinamik bir exchange adı
                symbol=symbol, 
                timeframe=validation_tf, 
                limit=self.REQUIRED_CANDLES
            )
            
            if not ohlcv_list or len(ohlcv_list) < self.REQUIRED_CANDLES:
                live_data_count = len(ohlcv_list) if ohlcv_list else 0
                reason = (f"Insufficient data in collector for '{validation_tf}': "
                          f"found {live_data_count}, required {self.REQUIRED_CANDLES}. "
                          "This usually means the prefetch/priming step failed.")
                logger.error(f"❌ {symbol}: {reason}")
                results['reason'] = reason
                return results

            logger.info(f"✅ Data availability check passed: {len(ohlcv_list)} candles found in collector for validation.")
            
        except Exception as e:
            reason = f"Failed to retrieve data from collector: {e}"
            logger.error(f"❌ {symbol}: {reason}", exc_info=True)
            results['reason'] = reason
            return results

        # Adım 2: VolumeAnalyzer hazır olma kontrolü (sadece mevcut veriyi kullanarak)
        va_cfg = self._get_volume_analyzer_config()
        va_enabled = bool(va_cfg.get('enabled', True))
        if va_enabled:
            trade_tf = va_cfg.get('trade_timeframe') or va_cfg.get('trade_tf') or '5m'
            short_tf = va_cfg.get('baseline_short_tf')
            medium_tf = va_cfg.get('baseline_medium_tf')
            short_lb = int(va_cfg.get('short_lookback'))
            medium_lb = int(va_cfg.get('medium_lookback'))
            window_bars = int(va_cfg.get('window_bars'))

            va_result = self.validate_volume_analyzer_for_symbol(
                symbol=symbol,
                trade_tf=trade_tf,
                short_tf=short_tf,
                medium_tf=medium_tf,
                window_bars=window_bars,
                short_lookback=short_lb,
                medium_lookback=medium_lb,
            )

            log_payload = {
                "event": "volume_analyzer_validation",
                "symbol": symbol,
                "trade_tf": trade_tf,
                "short_tf": short_tf,
                "medium_tf": medium_tf,
                "trade_bars_available": va_result["trade_bars_available"],
                "required_trade_bars": va_result["required_trade_bars"],
                "short_bars_available": va_result["short_bars_available"],
                "required_short_bars": va_result["required_short_bars"],
                "medium_bars_available": va_result["medium_bars_available"],
                "required_medium_bars": va_result["required_medium_bars"],
                "ready": va_result["ready"],
            }

            if va_result["ready"]:
                logger.info(log_payload)
            else:
                logger.warning("[VOLUME VALIDATION] VolumeAnalyzer not ready for %s: %s", symbol, log_payload)
        else:
            logger.info("[VOLUME VALIDATION] Skipping volume analyzer validation for %s (disabled in config).", symbol)

        # Adım 3: Veriyi DataFrame'e Çevir ve İndikatörleri Hesapla
        try:
            df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).apply(pd.to_numeric)
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            indicator_errors = []
            
            # RSI kontrolü
            rsi = talib.RSI(close)
            if np.all(np.isnan(rsi[14:])):
                indicator_errors.append("RSI calculation resulted in all NaNs.")

            # ATR kontrolü
            atr = talib.ATR(high, low, close)
            if np.all(np.isnan(atr[14:])):
                indicator_errors.append("ATR calculation resulted in all NaNs.")

            # EMA kontrolü
            ema_slow = talib.EMA(close, timeperiod=200)
            if np.isnan(ema_slow[-1]):
                 indicator_errors.append("EMA_SLOW has NaN at the last candle.")

            if indicator_errors:
                reason = "Indicator calculation failed: " + ", ".join(indicator_errors)
                logger.error(f"❌ {symbol}: {reason}")
                results['reason'] = reason
                return results

        except Exception as e:
            reason = f"TA-Lib calculation failed: {e}"
            logger.error(f"❌ {symbol}: {reason}", exc_info=True)
            results['reason'] = reason
            return results
            
        # Adım 3: Başarı Durumunu Raporla
        logger.info(f"✅ {symbol}: All indicators seem healthy and ready.")
        results['status'] = 'OK'
        results['reason'] = 'All indicators validated successfully.'

        # Adım 4: VolumeAnalyzer zorunluluk kontrolü
        if va_enabled and va_result is not None:
            results['volume_ready'] = bool(va_result.get('ready'))
            results['volume_validation'] = va_result
            if self.volume_analyzer_required and not va_result.get('ready'):
                results['status'] = 'FAIL'
                results['reason'] = (
                    f"VolumeAnalyzer not ready: trade {va_result['trade_bars_available']}/"
                    f"{va_result['required_trade_bars']}, short {va_result['short_bars_available']}/"
                    f"{va_result['required_short_bars']}, medium {va_result['medium_bars_available']}/"
                    f"{va_result['required_medium_bars']}"
                )
        elif not va_enabled:
            results['volume_ready'] = None
            results['volume_validation'] = {'ready': None, 'skipped': True}
        return results

    def _log_validation_summary(self, results: Dict):
        logger.info("\n" + "="*80)
        logger.info("📋 INDICATOR VALIDATION SUMMARY")
        logger.info("="*80)
        
        total = len(results)
        valid = sum(1 for res in results.values() if res['status'] == 'OK')
        volume_ready = [res for res in results.values() if res.get('volume_ready') is True]
        volume_not_ready = [res for res in results.values() if res.get('volume_ready') is False]
        volume_skipped = [res for res in results.values() if res.get('volume_ready') is None]
        
        logger.info(f"Total Symbols: {total}, Valid Symbols: {valid}, Failed Symbols: {total - valid}")
        logger.info(
            "VolumeAnalyzer readiness: %s ok, %s not ready, %s skipped/disabled",
            len(volume_ready),
            len(volume_not_ready),
            len(volume_skipped),
        )
        
        if valid != total:
            logger.error("❌ SOME INDICATORS FAILED VALIDATION:")
            for symbol, res in results.items():
                if res['status'] != 'OK':
                    logger.error(f"  - {symbol}: {res['reason']}")
        else:
            logger.info("✅ ALL INDICATORS READY FOR TRADING")
            
        logger.info("="*80)

    def _get_volume_analyzer_config(self) -> Dict[str, Any]:
        va_cfg = self.config.get('volume_analyzer') if isinstance(self.config, dict) else {}
        merged = {**VolumeAnalyzer.DEFAULT_CONFIG, **(va_cfg or {})}
        return merged

    def validate_volume_analyzer_for_symbol(
        self,
        symbol: str,
        trade_tf: str,
        short_tf: str,
        medium_tf: str,
        window_bars: int,
        short_lookback: int,
        medium_lookback: int,
    ) -> Dict[str, Any]:
        def _len_for_tf(tf: str, limit: int) -> int:
            candles = self.collector.get_latest_ohlcv('bingx', symbol, tf, limit)
            return len(candles) if candles else 0

        trade_len = _len_for_tf(trade_tf, window_bars)
        short_len = _len_for_tf(short_tf, short_lookback)
        medium_len = _len_for_tf(medium_tf, medium_lookback)

        ready = (
            trade_len >= window_bars
            and short_len >= short_lookback
            and medium_len >= medium_lookback
        )

        return {
            "symbol": symbol,
            "trade_tf": trade_tf,
            "short_tf": short_tf,
            "medium_tf": medium_tf,
            "trade_bars_available": trade_len,
            "required_trade_bars": window_bars,
            "short_bars_available": short_len,
            "required_short_bars": short_lookback,
            "medium_bars_available": medium_len,
            "required_medium_bars": medium_lookback,
            "ready": ready,
        }

    # --- ML VALIDATION METHODS (UNCHANGED) ---
    def validate_ml_data(self, price_data: pd.DataFrame, symbol: str) -> Tuple[bool, List[str]]:
        """
        Validate data quality for ML model consumption.
        
        This is the ML data validation gateway - ensures data is clean and complete
        before feeding to ML models.
        
        Args:
            price_data: Price DataFrame to validate
            symbol: Trading symbol (for logging)
            
        Returns:
            Tuple of (is_valid, list_of_errors)
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
        
        if 'timestamp' in price_data.index.name or 'timestamp' in price_data.columns:
            try:
                last_timestamp = pd.to_datetime(price_data.index[-1] if 'timestamp' in price_data.index.name else price_data['timestamp'].iloc[-1])
                # Ensure timestamp is timezone-aware for correct comparison
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
        
        Args:
            features: Feature DataFrame
            symbol: Trading symbol
            
        Returns:
            Tuple of (is_valid, list_of_errors)
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
