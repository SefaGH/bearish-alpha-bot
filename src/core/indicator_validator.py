"""
Indicator Warmup Validation Module
Ensures all technical indicators are properly calculated before trading begins.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple

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
    """
    
    REQUIRED_CANDLES = 250
    
    def __init__(self, websocket_manager):
        actual_ws_manager = getattr(websocket_manager, 'ws_manager', None)
        if not actual_ws_manager or not hasattr(actual_ws_manager, 'collector'):
            raise ValueError("IndicatorValidator requires an initialized WebSocketManager with a data collector.")
        self.ws_manager = actual_ws_manager
        self.validation_results = {}
        
    async def validate_all_symbols(
        self, 
        symbols: List[str], 
        exchange: str = 'bingx'
    ) -> Tuple[bool, Dict]:
        logger.info("="*80)
        logger.info("🔍 INDICATOR WARMUP VERIFICATION (POST-PREFETCH)")
        logger.info("="*80)
        
        if not TALIB_AVAILABLE:
            logger.error("❌ TA-Lib is not installed. Cannot perform indicator validation.")
            return False, {s: {'overall_valid': False, 'errors': ['TA-Lib not installed']} for s in symbols}

        all_valid = True
        results = {}
        for symbol in symbols:
            symbol_valid, symbol_results = await self.validate_symbol(symbol, exchange)
            results[symbol] = symbol_results
            all_valid = all_valid and symbol_valid
            
        self._log_validation_summary(results)
        return all_valid, results
    
    async def validate_symbol(
        self, 
        symbol: str, 
        exchange: str
    ) -> Tuple[bool, Dict]:
        logger.info(f"\n📊 Validating indicators for {symbol}...")
        symbol_norm = f"{symbol.split(':')[0]}:USDT"
        
        results = {'symbol': symbol, 'exchange': exchange, 'indicators': {}, 'overall_valid': True, 'errors': []}
        
        # Prefetch sonrası veriyi doğrudan collector'dan al
        ohlcv_1m = self.ws_manager.collector.get_latest_ohlcv(exchange, symbol_norm, '1m', limit=self.REQUIRED_CANDLES)
        
        # Kontrol: Prefetch başarılı oldu mu?
        if ohlcv_1m is None or len(ohlcv_1m) < self.REQUIRED_CANDLES:
            error = f"Insufficient 1m data after prefetch: {len(ohlcv_1m) if ohlcv_1m else 0}/{self.REQUIRED_CANDLES} candles. Prefetch step may have failed."
            logger.error(f"❌ {error}")
            results['errors'].append(error)
            results['overall_valid'] = False
            return False, results
            
        logger.info(f"✅ Data availability: {len(ohlcv_1m)} candles found in collector.")
        
        df = pd.DataFrame(ohlcv_1m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).apply(pd.to_numeric)
        
        # TA-Lib hesaplamaları...
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Basit NaN kontrolü
            if np.isnan(talib.RSI(close)[-1]):
                results['errors'].append("RSI calculation resulted in NaN.")
            if np.isnan(talib.ATR(high, low, close)[-1]):
                results['errors'].append("ATR calculation resulted in NaN.")
            if np.isnan(talib.EMA(close, timeperiod=200)[-1]):
                 results['errors'].append("EMA_SLOW calculation resulted in NaN.")

        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            results['errors'].append(f"TA-Lib calculation failed: {e}")

        if results['errors']:
            results['overall_valid'] = False
            logger.error(f"❌ {symbol}: Indicator validation FAILED")
            for err in results['errors']: logger.error(f"   - {err}")
        else:
            logger.info(f"✅ {symbol}: All indicators seem healthy.")

        return results['overall_valid'], results

    def _log_validation_summary(self, results: Dict):
        logger.info("\n" + "="*80)
        logger.info("📋 INDICATOR VALIDATION SUMMARY")
        logger.info("="*80)
        total = len(results)
        valid = sum(1 for r in results.values() if r['overall_valid'])
        logger.info(f"Total Symbols: {total}, Valid Symbols: {valid}, Failed Symbols: {total - valid}")
        if valid == total:
            logger.info("✅ ALL INDICATORS READY FOR TRADING")
        else:
            logger.error("❌ SOME INDICATORS FAILED VALIDATION")
        logger.info("="*80)
