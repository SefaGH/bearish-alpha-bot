#!/usr/bin/env python3
"""
Market Regime Recommendations Test Script (Mock Version)
Tests recommendations without requiring exchange credentials
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.market_regime import MarketRegimeAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_mock_data(trend='bearish', volatility='normal', periods=200):
    """Create mock OHLCV data with indicators"""
    
    # Generate timestamps
    end_time = datetime.now()
    timestamps = [end_time - timedelta(hours=i) for i in range(periods, 0, -1)]
    
    # Base price
    base_price = 50000
    
    # Generate price data based on trend
    if trend == 'bearish':
        prices = [base_price * (1 - 0.0002 * i + np.random.randn() * 0.001 * base_price) for i in range(periods)]
    elif trend == 'bullish':
        prices = [base_price * (1 + 0.0002 * i + np.random.randn() * 0.001 * base_price) for i in range(periods)]
    else:  # neutral
        prices = [base_price * (1 + np.sin(i/10) * 0.01 + np.random.randn() * 0.001) for i in range(periods)]
    
    # Create OHLCV
    df = pd.DataFrame()
    df['timestamp'] = timestamps
    df['open'] = prices
    df['high'] = [p * 1.002 for p in prices]
    df['low'] = [p * 0.998 for p in prices]
    df['close'] = [p * 1.0001 for p in prices]
    df['volume'] = [np.random.randint(100, 1000) for _ in range(periods)]
    df.set_index('timestamp', inplace=True)
    
    # Add indicators
    df['rsi'] = 30 if trend == 'bearish' else 70 if trend == 'bullish' else 50
    df['rsi'] = df['rsi'] + np.random.randn(periods) * 5
    
    # EMAs
    df['ema21'] = df['close'].ewm(span=21).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    
    # ATR (simplified)
    df['atr'] = df['close'] * 0.01 if volatility == 'low' else df['close'] * 0.03 if volatility == 'high' else df['close'] * 0.02
    
    return df

def test_recommendations_with_mock():
    """Test recommendations using mock data"""
    
    logger.info("=" * 60)
    logger.info("MARKET REGIME RECOMMENDATIONS TEST (MOCK)")
    logger.info("=" * 60)
    
    # Initialize analyzer
    logger.info("Initializing MarketRegimeAnalyzer...")
    analyzer = MarketRegimeAnalyzer()
    
    # Test different market conditions
    test_scenarios = [
        {'trend': 'bearish', 'volatility': 'high', 'name': 'Bearish High Volatility'},
        {'trend': 'bullish', 'volatility': 'low', 'name': 'Bullish Low Volatility'},
        {'trend': 'neutral', 'volatility': 'normal', 'name': 'Neutral Normal Volatility'},
    ]
    
    all_tests_passed = True
    
    for scenario in test_scenarios:
        logger.info("\n" + "=" * 40)
        logger.info(f"SCENARIO: {scenario['name']}")
        logger.info("=" * 40)
        
        # Create mock data
        df_30m = create_mock_data(scenario['trend'], scenario['volatility'], 200)
        df_1h = create_mock_data(scenario['trend'], scenario['volatility'], 200)
        df_4h = create_mock_data(scenario['trend'], scenario['volatility'], 200)
        
        # Test 1: Basic regime analysis
        logger.info("\nTEST 1: Regime Analysis")
        try:
            regime = analyzer.analyze_market_regime(df_30m, df_1h, df_4h)
            
            logger.info(f"  Trend: {regime.get('trend', 'unknown')}")
            logger.info(f"  Momentum: {regime.get('momentum', 'unknown')}")
            logger.info(f"  Volatility: {regime.get('volatility', 'unknown')}")
            logger.info(f"  Entry Score: {regime.get('entry_score', 0.5):.2f}")
            logger.info(f"  Risk Multiplier: {regime.get('risk_multiplier', 1.0):.2f}")
            logger.info("  ✅ Regime analysis PASSED")
        except Exception as e:
            logger.error(f"  ❌ Regime analysis FAILED: {e}")
            all_tests_passed = False
        
        # Test 2: Get recommendations
        logger.info("\nTEST 2: Recommendations")
        try:
            if not hasattr(analyzer, 'get_regime_recommendations'):
                logger.error("  ❌ get_regime_recommendations method NOT FOUND!")
                all_tests_passed = False
            else:
                recommendations = analyzer.get_regime_recommendations(df_30m, df_1h, df_4h)
                
                if not recommendations:
                    logger.warning("  ⚠️ No recommendations generated")
                else:
                    logger.info(f"  Generated {len(recommendations)} recommendations:")
                    for i, rec in enumerate(recommendations[:3], 1):
                        logger.info(f"    {i}. {rec}")
                logger.info("  ✅ Recommendations PASSED")
        except Exception as e:
            logger.error(f"  ❌ Recommendations FAILED: {e}")
            all_tests_passed = False
        
        # Test 3: Strategy favorability
        logger.info("\nTEST 3: Strategy Favorability")
        try:
            if not hasattr(analyzer, 'is_favorable_for_strategy'):
                logger.error("  ❌ is_favorable_for_strategy method NOT FOUND!")
                all_tests_passed = False
            else:
                # Test OversoldBounce
                is_good_ob, reason_ob = analyzer.is_favorable_for_strategy('oversold_bounce', df_30m, df_1h, df_4h)
                logger.info(f"  OversoldBounce: {'✅ FAVORABLE' if is_good_ob else '❌ NOT FAVORABLE'}")
                logger.info(f"    Reason: {reason_ob}")
                
                # Test ShortTheRip
                is_good_str, reason_str = analyzer.is_favorable_for_strategy('short_the_rip', df_30m, df_1h, df_4h)
                logger.info(f"  ShortTheRip: {'✅ FAVORABLE' if is_good_str else '❌ NOT FAVORABLE'}")
                logger.info(f"    Reason: {reason_str}")
                logger.info("  ✅ Strategy favorability PASSED")
        except Exception as e:
            logger.error(f"  ❌ Strategy favorability FAILED: {e}")
            all_tests_passed = False
        
        # Test 4: Adaptive RSI threshold
        logger.info("\nTEST 4: Adaptive RSI Threshold")
        try:
            if not hasattr(analyzer, 'get_adaptive_rsi_threshold'):
                logger.error("  ❌ get_adaptive_rsi_threshold method NOT FOUND!")
                all_tests_passed = False
            else:
                adaptive_threshold = analyzer.get_adaptive_rsi_threshold(
                    scenario['trend'],
                    'strong' if scenario['trend'] != 'neutral' else 'sideways',
                    scenario['volatility']
                )
                logger.info(f"  Adaptive RSI Threshold: {adaptive_threshold:.1f}")
                
                # Validate threshold is in expected range
                if 25 <= adaptive_threshold <= 55:
                    logger.info("  ✅ Threshold in valid range (25-55)")
                else:
                    logger.error(f"  ❌ Threshold out of range: {adaptive_threshold}")
                    all_tests_passed = False
        except Exception as e:
            logger.error(f"  ❌ Adaptive RSI threshold FAILED: {e}")
            all_tests_passed = False
        
        # Test 5: Position size multiplier
        logger.info("\nTEST 5: Position Size Multiplier")
        try:
            if not hasattr(analyzer, 'get_position_size_multiplier'):
                logger.error("  ❌ get_position_size_multiplier method NOT FOUND!")
                all_tests_passed = False
            else:
                size_mult = analyzer.get_position_size_multiplier(df_30m, df_1h, df_4h)
                logger.info(f"  Position Size Multiplier: {size_mult:.2f}x")
                
                # Validate multiplier is in expected range
                if 0.5 <= size_mult <= 1.5:
                    logger.info("  ✅ Multiplier in valid range (0.5-1.5)")
                else:
                    logger.error(f"  ❌ Multiplier out of range: {size_mult}")
                    all_tests_passed = False
        except Exception as e:
            logger.error(f"  ❌ Position size multiplier FAILED: {e}")
            all_tests_passed = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    if all_tests_passed:
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("Market regime recommendations are working correctly.")
        return True
    else:
        logger.error("❌ SOME TESTS FAILED")
        logger.error("Please check the implementation.")
        return False

if __name__ == "__main__":
    success = test_recommendations_with_mock()
    sys.exit(0 if success else 1)
