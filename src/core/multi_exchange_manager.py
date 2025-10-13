"""
Multi-Exchange Manager for synchronized data fetching across multiple exchanges.
Supports KuCoin, BingX, and other exchanges with unified data collection.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from .ccxt_client import CcxtClient

logger = logging.getLogger(__name__)


class MultiExchangeManager:
    """
    Multi-Exchange Manager for coordinated data fetching and synchronization.
    
    Features:
    - Unified data fetching across multiple exchanges
    - Timestamp alignment for cross-exchange analysis
    - Support for KuCoin, BingX, and other exchanges
    """
    
    def __init__(self, exchanges: Optional[Dict[str, CcxtClient]] = None):
        """
        Initialize MultiExchangeManager.
        
        Args:
            exchanges: Dictionary mapping exchange names to CcxtClient instances
                      If None, creates default KuCoin and BingX clients
        """
        if exchanges is None:
            self.exchanges = {
                'kucoinfutures': CcxtClient('kucoinfutures'),
                'bingx': CcxtClient('bingx')
            }
        else:
            self.exchanges = exchanges
        
        logger.info(f"MultiExchangeManager initialized with exchanges: {list(self.exchanges.keys())}")
    
    def fetch_unified_data(self, symbols_per_exchange: Dict[str, List[str]], 
                          timeframe: str = '30m', 
                          limit: int = 500) -> Dict[str, Dict[str, List[List]]]:
        """
        Fetch data from multiple exchanges in a unified format.
        
        Args:
            symbols_per_exchange: Dict mapping exchange names to lists of symbols
                                 e.g., {'kucoinfutures': ['BTC/USDT:USDT'], 
                                        'bingx': ['VST/USDT:USDT']}
            timeframe: Timeframe for OHLCV data (e.g., '30m', '1h')
            limit: Number of candles to fetch per symbol
        
        Returns:
            Nested dict: {exchange_name: {symbol: ohlcv_data}}
        """
        results = {}
        
        for exchange_name, symbols in symbols_per_exchange.items():
            if exchange_name not in self.exchanges:
                logger.warning(f"Exchange '{exchange_name}' not initialized, skipping")
                continue
            
            client = self.exchanges[exchange_name]
            exchange_data = {}
            
            logger.info(f"Fetching data from {exchange_name}: {len(symbols)} symbols")
            
            for symbol in symbols:
                try:
                    if limit > 500:
                        # Use bulk fetch for large requests
                        data = client.fetch_ohlcv_bulk(symbol, timeframe, limit)
                    else:
                        # Use regular fetch for small requests
                        data = client.ohlcv(symbol, timeframe, limit)
                    
                    exchange_data[symbol] = data
                    logger.info(f"✓ {exchange_name} {symbol}: {len(data)} candles")
                    
                    # Rate limiting between symbols
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"✗ {exchange_name} {symbol} failed: {e}")
                    exchange_data[symbol] = []
            
            results[exchange_name] = exchange_data
        
        logger.info(f"Unified data fetch complete: {len(results)} exchanges")
        return results
    
    def align_timestamps(self, data_dict: Dict[str, Dict[str, List[List]]], 
                        tolerance_ms: int = 60000) -> Dict[str, Dict[str, List[List]]]:
        """
        Align timestamps across exchanges for synchronized analysis.
        
        Args:
            data_dict: Nested dict from fetch_unified_data()
            tolerance_ms: Maximum timestamp difference to consider aligned (default 60s)
        
        Returns:
            Aligned data with matching timestamps across exchanges
        """
        logger.info("Aligning timestamps across exchanges...")
        
        # Collect all unique timestamps across all exchanges
        all_timestamps = set()
        for exchange_data in data_dict.values():
            for symbol_data in exchange_data.values():
                for candle in symbol_data:
                    if candle:  # Check if candle is not empty
                        all_timestamps.add(candle[0])
        
        if not all_timestamps:
            logger.warning("No timestamps found in data")
            return data_dict
        
        aligned_data = {}
        
        for exchange_name, exchange_data in data_dict.items():
            aligned_exchange = {}
            
            for symbol, candles in exchange_data.items():
                # Create timestamp index for quick lookup
                timestamp_index = {c[0]: c for c in candles if c}
                
                # Filter candles to only include aligned timestamps
                aligned_candles = []
                for ts in sorted(all_timestamps):
                    # Look for exact match or close match within tolerance
                    exact_match = timestamp_index.get(ts)
                    if exact_match:
                        aligned_candles.append(exact_match)
                    else:
                        # Look for close match
                        for candle_ts, candle in timestamp_index.items():
                            if abs(candle_ts - ts) <= tolerance_ms:
                                aligned_candles.append(candle)
                                break
                
                aligned_exchange[symbol] = aligned_candles
                logger.debug(f"{exchange_name} {symbol}: {len(candles)} → {len(aligned_candles)} aligned")
            
            aligned_data[exchange_name] = aligned_exchange
        
        logger.info(f"Timestamp alignment complete: {len(all_timestamps)} unique timestamps")
        return aligned_data
    
    def get_exchange_summary(self) -> Dict[str, Any]:
        """
        Get summary information about configured exchanges.
        
        Returns:
            Dict with exchange information and status
        """
        summary = {
            'total_exchanges': len(self.exchanges),
            'exchanges': {}
        }
        
        for name, client in self.exchanges.items():
            try:
                # Try to get market info to verify exchange is accessible
                markets = client.markets()
                summary['exchanges'][name] = {
                    'status': 'active',
                    'markets': len(markets),
                    'name': client.name
                }
            except Exception as e:
                summary['exchanges'][name] = {
                    'status': 'error',
                    'error': str(e),
                    'name': client.name
                }
        
        return summary
    
    def validate_vst_contract(self, exchange: str = 'bingx') -> Dict[str, Any]:
        """
        Validate VST/USDT contract availability and specifications.
        
        Args:
            exchange: Exchange to check (default: 'bingx')
        
        Returns:
            Dict with VST contract information
        """
        if exchange not in self.exchanges:
            return {'error': f"Exchange '{exchange}' not configured"}
        
        client = self.exchanges[exchange]
        vst_symbol = 'VST/USDT:USDT'
        
        result = {
            'symbol': vst_symbol,
            'exchange': exchange,
            'available': False,
            'contract_type': 'perpetual'
        }
        
        try:
            markets = client.markets()
            
            if vst_symbol in markets:
                market_info = markets[vst_symbol]
                result['available'] = True
                result['market_info'] = {
                    'active': market_info.get('active', False),
                    'type': market_info.get('type', 'unknown'),
                    'settle': market_info.get('settle', 'unknown'),
                    'contract_size': market_info.get('contractSize', 1)
                }
                logger.info(f"✓ VST/USDT contract found on {exchange}")
            else:
                # Check for alternative VST symbols
                vst_variants = [s for s in markets.keys() if 'VST' in s.upper()]
                if vst_variants:
                    result['available'] = True
                    result['alternative_symbols'] = vst_variants
                    logger.info(f"✓ VST contracts found on {exchange}: {vst_variants}")
                else:
                    logger.warning(f"VST/USDT not found on {exchange}")
        
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error validating VST contract on {exchange}: {e}")
        
        return result
