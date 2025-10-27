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
    and returning valid values before strategies start execution.
    """
    
    REQUIRED_INDICATORS = {
        'RSI': {'period': 14, 'min_data': 50},
        'ATR': {'period': 14, 'min_data': 50},
        'EMA_FAST': {'period': 21, 'min_data': 50},
        'EMA_MID': {'period': 50, 'min_data': 100},
        'EMA_SLOW': {'period': 200, 'min_data': 250},
        'VWAP': {'min_data': 20}
    }
    
    def __init__(self, websocket_manager):
        # --- ÖNCEKİ HATAYI DÜZELTEN KRİTİK DEĞİŞİKLİK BURADA ---
        # `websocket_manager` aslında `OptimizedWebSocketManager` sınıfıdır.
        # Bizim ise onun içindeki `ws_manager` özelliğine ve onun da içindeki `collector`'a ihtiyacımız var.
        
        # 1. Önce `OptimizedWebSocketManager`'ın içindeki `ws_manager`'a erişiyoruz.
        actual_ws_manager = getattr(websocket_manager, 'ws_manager', None)
        
        # 2. Şimdi `actual_ws_manager`'ın `collector`'ı olup olmadığını kontrol ediyoruz.
        if not actual_ws_manager or not hasattr(actual_ws_manager, 'collector'):
            raise ValueError("IndicatorValidator requires an initialized WebSocketManager with a data collector.")
        
        # 3. Doğru nesneyi sınıf değişkenine atıyoruz.
        self.ws_manager = actual_ws_manager
        # --------------------------------------------------------
        
        self.validation_results = {}
        
    async def validate_all_symbols(
        self, 
        symbols: List[str], 
        exchange: str = 'bingx'
    ) -> Tuple[bool, Dict]:
        logger.info("="*80)
        logger.info("🔍 INDICATOR WARMUP VERIFICATION")
        logger.info("="*80)
        
        all_valid = True
        results = {}
        
        if not TALIB_AVAILABLE:
            logger.error("❌ TA-Lib is not installed. Cannot perform indicator validation.")
            for symbol in symbols:
                results[symbol] = {'overall_valid': False, 'errors': ['TA-Lib not installed']}
            return False, results

        for symbol in symbols:
            symbol_valid, symbol_results = await self.validate_symbol(
                symbol, 
                exchange
            )
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
        
        results = {
            'symbol': symbol,
            'exchange': exchange,
            'indicators': {},
            'overall_valid': True,
            'errors': []
        }
        
        # Artık self.ws_manager doğru nesne olduğu için bu satır çalışacaktır.
        ohlcv_1m = self.ws_manager.collector.get_latest_ohlcv(exchange, symbol_norm, '1m', limit=250)
        
        if ohlcv_1m is None or len(ohlcv_1m) < self.REQUIRED_INDICATORS['EMA_SLOW']['min_data']:
            required_candles = self.REQUIRED_INDICATORS['EMA_SLOW']['min_data']
            error = f"Insufficient 1m data: {len(ohlcv_1m) if ohlcv_1m else 0}/{required_candles} candles"
            logger.error(f"❌ {error}")
            results['errors'].append(error)
            results['overall_valid'] = False
            return False, results
            
        logger.info(f"✅ Data availability: 1m={len(ohlcv_1m)}")
        
        df = pd.DataFrame(ohlcv_1m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
        
        close_1m = df['close'].values
        high_1m = df['high'].values
        low_1m = df['low'].values
        volume_1m = df['volume'].values
        
        results['indicators']['RSI'] = self._validate_rsi(close_1m)
        results['indicators']['ATR'] = self._validate_atr(high_1m, low_1m, close_1m)
        results['indicators']['EMA_FAST'] = self._validate_ema(close_1m, 21, 'EMA_FAST')
        results['indicators']['EMA_MID'] = self._validate_ema(close_1m, 50, 'EMA_MID')
        results['indicators']['EMA_SLOW'] = self._validate_ema(close_1m, 200, 'EMA_SLOW')
        results['indicators']['VWAP'] = self._validate_vwap(high_1m, low_1m, close_1m, volume_1m)
        
        for name, res in results['indicators'].items():
            if not res['valid']:
                results['overall_valid'] = False
                results['errors'].append(f"{name}: {res.get('error', 'Unknown error')}")
        
        if results['overall_valid']:
            logger.info(f"✅ {symbol}: All indicators ready for trading")
            self._log_indicator_values(results['indicators'])
        else:
            logger.error(f"❌ {symbol}: Indicator validation FAILED")
            for error in results['errors']:
                logger.error(f"   - {error}")
        
        return results['overall_valid'], results

    def _validate_generic(self, name: str, values: np.ndarray, min_val: float = -np.inf, max_val: float = np.inf) -> Dict:
        if values is None or len(values) == 0:
            return {'valid': False, 'error': f'{name} calculation returned None or empty'}
        if np.isnan(values[-1]):
            return {'valid': False, 'error': f'{name} returned NaN', 'value': None}
        if not (min_val <= values[-1] <= max_val):
            return {'valid': False, 'error': f'{name} out of range: {values[-1]:.2f}', 'value': values[-1]}
        valid_count = np.sum(~np.isnan(values[-50:]))
        if valid_count < 40:
            return {'valid': False, 'error': f'Insufficient valid {name} values: {valid_count}/50', 'value': values[-1]}
        return {'valid': True, 'value': values[-1], 'recent_values': values[-5:].tolist()}

    def _validate_rsi(self, close: np.ndarray) -> Dict:
        if not TALIB_AVAILABLE: return {'valid': False, 'error': 'TA-Lib not installed'}
        try:
            rsi = talib.RSI(close, timeperiod=14)
            return self._validate_generic('RSI', rsi, 0, 100)
        except Exception as e:
            return {'valid': False, 'error': f'RSI calculation failed: {str(e)}'}

    def _validate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict:
        if not TALIB_AVAILABLE: return {'valid': False, 'error': 'TA-Lib not installed'}
        try:
            atr = talib.ATR(high, low, close, timeperiod=14)
            return self._validate_generic('ATR', atr, 0)
        except Exception as e:
            return {'valid': False, 'error': f'ATR calculation failed: {str(e)}'}

    def _validate_ema(self, close: np.ndarray, period: int, name: str) -> Dict:
        if not TALIB_AVAILABLE: return {'valid': False, 'error': 'TA-Lib not installed'}
        try:
            ema = talib.EMA(close, timeperiod=period)
            current_price = close[-1]
            return self._validate_generic(name, ema, 0.5 * current_price, 1.5 * current_price)
        except Exception as e:
            return {'valid': False, 'error': f'{name} calculation failed: {str(e)}'}

    def _validate_vwap(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> Dict:
        try:
            typical_price = (high + low + close) / 3
            if np.sum(volume) == 0: return {'valid': False, 'error': 'Total volume is zero'}
            vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
            current_price = close[-1]
            return self._validate_generic('VWAP', vwap, 0.8 * current_price, 1.2 * current_price)
        except Exception as e:
            return {'valid': False, 'error': f'VWAP calculation failed: {str(e)}'}

    def _log_indicator_values(self, indicators: Dict):
        logger.info("   Indicator Values:")
        for name, result in indicators.items():
            if result['valid']:
                logger.info(f"      {name:12s}: {result['value']:10.2f} ✅")

    def _log_validation_summary(self, results: Dict):
        logger.info("\n" + "="*80)
        logger.info("📋 INDICATOR VALIDATION SUMMARY")
        logger.info("="*80)
        total = len(results)
        valid = sum(1 for r in results.values() if r['overall_valid'])
        logger.info(f"Total Symbols: {total}")
        logger.info(f"Valid Symbols: {valid}")
        logger.info(f"Failed Symbols: {total - valid}")
        if valid == total:
            logger.info("✅ ALL INDICATORS READY FOR TRADING")
        else:
            logger.error("❌ SOME INDICATORS FAILED VALIDATION")
            for symbol, result in results.items():
                if not result['overall_valid']:
                    logger.error(f"      - {symbol}: {', '.join(result['errors'])}")
        logger.info("="*80)
