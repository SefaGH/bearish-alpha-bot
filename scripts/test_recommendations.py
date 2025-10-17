#!/usr/bin/env python3
"""
Market Regime Recommendations Test Script
Tests if recommendations are working correctly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import logging
from core.market_regime import MarketRegimeAnalyzer
from core.multi_exchange import build_clients_from_env
from core.indicators import add_indicators
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_recommendations():
    """Test market regime recommendations with real data"""
    
    logger.info("=" * 60)
    logger.info("MARKET REGIME RECOMMENDATIONS TEST")
    logger.info("=" * 60)
    
    try:
        # Build exchange clients
        logger.info("Building exchange clients...")
        clients = build_clients_from_env()
        
        if not clients:
            logger.error("No exchange clients available")
            return False
        
        # Get first available client
        exchange_name, client = next(iter(clients.items()))
        logger.info(f"Using exchange: {exchange_name}")
        
        # Test symbol
        symbol = 'BTC/USDT:USDT'
        logger.info(f"Testing with symbol: {symbol}")
        
        # Validate symbol
        validated_symbol = client.validate_and_get_symbol(symbol)
        logger.info(f"Validated symbol: {validated_symbol}")
        
        # Fetch OHLCV data
        logger.info("Fetching OHLCV data...")
        df_30m_raw = client.ohlcv(validated_symbol, '30m', 200)
        df_1h_raw = client.ohlcv(validated_symbol, '1h', 200)
        df_4h_raw = client.ohlcv(validated_symbol, '4h', 200)
        
        # Convert to DataFrames
        def to_df(ohlcv_data):
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        
        df_30m = to_df(df_30m_raw)
        df_1h = to_df(df_1h_raw)
        df_4h = to_df(df_4h_raw)
        
        logger.info(f"Data fetched - 30m: {len(df_30m)} bars, 1h: {len(df_1h)} bars, 4h: {len(df_4h)} bars")
        
        # Add indicators
        logger.info("Adding indicators...")
        df_30m = add_indicators(df_30m, {})
        df_1h = add_indicators(df_1h, {})
        df_4h = add_indicators(df_4h, {})
        
        # Initialize analyzer
        logger.info("Initializing MarketRegimeAnalyzer...")
        analyzer = MarketRegimeAnalyzer()
        
        # Test 1: Basic regime analysis
        logger.info("\n" + "=" * 40)
        logger.info("TEST 1: Basic Regime Analysis")
        logger.info("=" * 40)
        
        regime = analyzer.analyze_market_regime(df_30m, df_1h, df_4h)
        
        logger.info(f"Trend: {regime.get('trend', 'unknown')}")
        logger.info(f"Momentum: {regime.get('momentum', 'unknown')}")
        logger.info(f"Volatility: {regime.get('volatility', 'unknown')}")
        logger.info(f"Risk Multiplier: {regime.get('risk_multiplier', 1.0):.2f}")
        logger.info(f"Entry Score: {regime.get('entry_score', 0.5):.2f}")
        
        # Test 2: Get recommendations
        logger.info("\n" + "=" * 40)
        logger.info("TEST 2: Trading Recommendations")
        logger.info("=" * 40)
        
        # Check if method exists
        if not hasattr(analyzer, 'get_regime_recommendations'):
            logger.error("❌ get_regime_recommendations method NOT FOUND!")
            logger.error("   The method was not properly added to the class")
            return False
        
        recommendations = analyzer.get_regime_recommendations(df_30m, df_1h, df_4h)
        
        if not recommendations:
            logger.warning("No recommendations generated")
        else:
            logger.info(f"Generated {len(recommendations)} recommendations:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")
        
        # Test 3: Strategy favorability
        logger.info("\n" + "=" * 40)
        logger.info("TEST 3: Strategy Favorability")
        logger.info("=" * 40)
        
        if hasattr(analyzer, 'is_favorable_for_strategy'):
            # Test OversoldBounce
            is_good_ob, reason_ob = analyzer.is_favorable_for_strategy('oversold_bounce', df_30m, df_1h, df_4h)
            logger.info(f"OversoldBounce: {'✅ FAVORABLE' if is_good_ob else '❌ NOT FAVORABLE'}")
            logger.info(f"  Reason: {reason_ob}")
            
            # Test ShortTheRip
            is_good_str, reason_str = analyzer.is_favorable_for_strategy('short_the_rip', df_30m, df_1h, df_4h)
            logger.info(f"ShortTheRip: {'✅ FAVORABLE' if is_good_str else '❌ NOT FAVORABLE'}")
            logger.info(f"  Reason: {reason_str}")
        else:
            logger.error("❌ is_favorable_for_strategy method NOT FOUND!")
        
        # Test 4: Adaptive RSI threshold
        logger.info("\n" + "=" * 40)
        logger.info("TEST 4: Adaptive RSI Threshold")
        logger.info("=" * 40)
        
        if hasattr(analyzer, 'get_adaptive_rsi_threshold'):
            adaptive_threshold = analyzer.get_adaptive_rsi_threshold(
                regime.get('trend', 'neutral'),
                regime.get('momentum', 'sideways'),
                regime.get('volatility', 'normal')
            )
            logger.info(f"Adaptive RSI Threshold: {adaptive_threshold:.1f}")
            logger.info(f"  (Base would be 40.0, adjusted for current regime)")
        else:
            logger.error("❌ get_adaptive_rsi_threshold method NOT FOUND!")
        
        # Test 5: Position size multiplier
        logger.info("\n" + "=" * 40)
        logger.info("TEST 5: Position Size Multiplier")
        logger.info("=" * 40)
        
        if hasattr(analyzer, 'get_position_size_multiplier'):
            size_mult = analyzer.get_position_size_multiplier(df_30m, df_1h, df_4h)
            logger.info(f"Position Size Multiplier: {size_mult:.2f}x")
            logger.info(f"  (Based on volatility: {regime.get('volatility', 'normal')})")
        else:
            logger.error("❌ get_position_size_multiplier method NOT FOUND!")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        # Check what methods are actually available
        available_methods = [
            'get_regime_recommendations',
            'is_favorable_for_strategy', 
            'get_adaptive_rsi_threshold',
            'get_position_size_multiplier'
        ]
        
        for method in available_methods:
            if hasattr(analyzer, method):
                logger.info(f"✅ {method} is available")
            else:
                logger.error(f"❌ {method} is NOT available")
        
        logger.info("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_recommendations()
    sys.exit(0 if success else 1)
