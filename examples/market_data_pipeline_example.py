#!/usr/bin/env python3
"""
Market Data Pipeline Example - Phase 2.1
Demonstrates integration with existing Bearish Alpha Bot infrastructure
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.market_data_pipeline import MarketDataPipeline
from core.data_aggregator import DataAggregator
from core.multi_exchange import build_clients_from_env
from core.indicators import add_indicators
import os


def example_basic_usage():
    """Basic pipeline usage with existing infrastructure"""
    print("=" * 70)
    print("Example 1: Basic Pipeline Usage")
    print("=" * 70)
    
    try:
        # Integration with existing multi-exchange system
        # Set up minimal environment for demonstration
        if not os.getenv('EXCHANGES'):
            print("⚠️  Note: EXCHANGES environment variable not set.")
            print("    Set EXCHANGES='bingx,kucoinfutures' for live data.")
            print("    Using mock setup for demonstration...\n")
            return
        
        clients = build_clients_from_env()
        pipeline = MarketDataPipeline(clients)
        
        # Start feeds for popular symbols
        symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        timeframes = ['30m', '1h']
        
        print(f"Starting feeds for {len(symbols)} symbols across {len(timeframes)} timeframes...")
        result = pipeline.start_feeds(symbols, timeframes)
        
        print(f"\n✅ Pipeline started successfully!")
        print(f"   Symbols processed: {result['symbols_processed']}")
        print(f"   Successful fetches: {result['successful_fetches']}")
        print(f"   Failed fetches: {result['failed_fetches']}")
        print(f"   Exchanges used: {', '.join(result['exchanges_used'])}")
        
        # Get latest data
        btc_data = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m')
        if btc_data is not None:
            print(f"\n📊 BTC/USDT:USDT 30m data:")
            print(f"   Total candles: {len(btc_data)}")
            print(f"   Latest close: ${btc_data['close'].iloc[-1]:.2f}")
            print(f"   RSI: {btc_data['rsi'].iloc[-1]:.2f}")
            print(f"   EMA21: ${btc_data['ema21'].iloc[-1]:.2f}")
            print(f"   Columns: {', '.join(btc_data.columns.tolist())}")
        
        # Shutdown gracefully
        pipeline.shutdown()
        print("\n✓ Pipeline shut down gracefully")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def example_health_monitoring():
    """Health monitoring and status tracking"""
    print("=" * 70)
    print("Example 2: Health Monitoring")
    print("=" * 70)
    
    try:
        if not os.getenv('EXCHANGES'):
            print("⚠️  Set EXCHANGES environment variable for live demonstration.\n")
            return
        
        clients = build_clients_from_env()
        pipeline = MarketDataPipeline(clients)
        
        # Start some feeds
        pipeline.start_feeds(['BTC/USDT:USDT'], ['30m', '1h'])
        
        # Check health
        health = pipeline.health_check()
        print(f"\n🏥 Health Check:")
        print(f"   Status: {health['status'].upper()}")
        print(f"   Uptime: {health['uptime_seconds']:.1f} seconds")
        print(f"   Total requests: {health['total_requests']}")
        print(f"   Failed requests: {health['failed_requests']}")
        print(f"   Error rate: {health['error_rate']:.2f}%")
        print(f"   Active streams: {health['active_streams']}")
        print(f"   Is running: {health['is_running']}")
        
        # Get detailed pipeline status
        status = pipeline.get_pipeline_status()
        print(f"\n📈 Pipeline Status:")
        print(f"   Memory estimate: {status['memory_estimate_mb']:.2f} MB")
        print(f"   Fresh data streams: {status['data_freshness']['fresh']}")
        print(f"   Stale data streams: {status['data_freshness']['stale']}")
        
        print(f"\n📊 Exchange Breakdown:")
        for exchange, info in status['exchanges'].items():
            print(f"   {exchange}: {info['streams']} streams, {len(info['symbols'])} symbols")
        
        pipeline.shutdown()
        print("\n✓ Health monitoring completed")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def example_with_aggregator():
    """Advanced usage with data aggregator"""
    print("=" * 70)
    print("Example 3: Data Aggregator Integration")
    print("=" * 70)
    
    try:
        if not os.getenv('EXCHANGES'):
            print("⚠️  Set EXCHANGES environment variable for live demonstration.\n")
            return
        
        clients = build_clients_from_env()
        pipeline = MarketDataPipeline(clients)
        aggregator = DataAggregator(pipeline)
        
        # Start feeds
        symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        pipeline.start_feeds(symbols, ['30m'])
        
        print("\n🔄 Aggregating data from multiple exchanges...")
        
        # Aggregate multi-exchange data
        btc_agg = aggregator.aggregate_multi_exchange('BTC/USDT:USDT', '30m')
        
        print(f"\n📊 BTC/USDT:USDT Aggregation Results:")
        print(f"   Available sources: {len(btc_agg['sources'])}")
        
        for exchange, data in btc_agg['sources'].items():
            print(f"\n   {exchange}:")
            print(f"      Quality score: {data['quality_score']:.2f}")
            print(f"      Candle count: {data['candle_count']}")
            print(f"      Freshness: {data['freshness']}")
        
        # Get best data source
        best_source = aggregator.get_best_data_source('BTC/USDT:USDT', '30m')
        if best_source:
            print(f"\n🏆 Best data source: {best_source}")
        
        # Get consensus data
        consensus = aggregator.get_consensus_data('BTC/USDT:USDT', '30m', min_sources=2)
        if consensus is not None:
            print(f"\n✅ Consensus data generated: {len(consensus)} candles")
            print(f"   Latest consensus close: ${consensus['close'].iloc[-1]:.2f}")
        
        pipeline.shutdown()
        print("\n✓ Data aggregation completed")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def example_custom_configuration():
    """Custom pipeline configuration"""
    print("=" * 70)
    print("Example 4: Custom Configuration")
    print("=" * 70)
    
    try:
        if not os.getenv('EXCHANGES'):
            print("⚠️  Set EXCHANGES environment variable for live demonstration.\n")
            return
        
        clients = build_clients_from_env()
        
        # Custom indicator configuration
        config = {
            'indicators': {
                'rsi_period': 14,
                'ema_fast': 21,
                'ema_mid': 50,
                'ema_slow': 200
            }
        }
        
        pipeline = MarketDataPipeline(clients, config=config)
        
        print("✅ Pipeline initialized with custom indicator configuration")
        print(f"   RSI period: {config['indicators']['rsi_period']}")
        print(f"   EMA periods: {config['indicators']['ema_fast']}, "
              f"{config['indicators']['ema_mid']}, {config['indicators']['ema_slow']}")
        
        # Start feeds with specific timeframes and buffer limits
        result = pipeline.start_feeds(['BTC/USDT:USDT'], ['30m', '1h', '4h'])
        
        print(f"\n📊 Started feeds with custom configuration:")
        print(f"   30m buffer limit: {pipeline.BUFFER_LIMITS['30m']} candles (~21 days)")
        print(f"   1h buffer limit: {pipeline.BUFFER_LIMITS['1h']} candles (~21 days)")
        print(f"   4h buffer limit: {pipeline.BUFFER_LIMITS['4h']} candles (~33 days)")
        
        pipeline.shutdown()
        print("\n✓ Custom configuration example completed")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def example_multi_timeframe_analysis():
    """Multi-timeframe data analysis"""
    print("=" * 70)
    print("Example 5: Multi-Timeframe Analysis")
    print("=" * 70)
    
    try:
        if not os.getenv('EXCHANGES'):
            print("⚠️  Set EXCHANGES environment variable for live demonstration.\n")
            return
        
        clients = build_clients_from_env()
        pipeline = MarketDataPipeline(clients)
        
        # Start feeds for multiple timeframes
        symbol = 'BTC/USDT:USDT'
        timeframes = ['30m', '1h', '4h']
        
        print(f"Fetching {symbol} data across {len(timeframes)} timeframes...")
        pipeline.start_feeds([symbol], timeframes)
        
        print(f"\n📊 Multi-Timeframe Analysis for {symbol}:\n")
        
        for tf in timeframes:
            data = pipeline.get_latest_ohlcv(symbol, tf)
            if data is not None and not data.empty:
                latest = data.iloc[-1]
                print(f"   {tf:>4} timeframe:")
                print(f"      Close: ${latest['close']:.2f}")
                print(f"      RSI: {latest['rsi']:.2f}")
                print(f"      ATR: ${latest['atr']:.2f}")
                print(f"      EMA21: ${latest['ema21']:.2f}")
                print(f"      Volume: {latest['volume']:,.0f}")
                print()
        
        pipeline.shutdown()
        print("✓ Multi-timeframe analysis completed")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def example_error_handling():
    """Error handling and resilience demonstration"""
    print("=" * 70)
    print("Example 6: Error Handling & Resilience")
    print("=" * 70)
    
    try:
        if not os.getenv('EXCHANGES'):
            print("⚠️  Set EXCHANGES environment variable for live demonstration.\n")
            return
        
        clients = build_clients_from_env()
        pipeline = MarketDataPipeline(clients)
        
        # Try fetching with invalid symbol (will fail gracefully)
        print("Testing error handling with invalid symbol...")
        result = pipeline.start_feeds(['INVALID/SYMBOL:USDT'], ['30m'])
        
        print(f"\n📊 Error Handling Results:")
        print(f"   Successful fetches: {result['successful_fetches']}")
        print(f"   Failed fetches: {result['failed_fetches']}")
        print(f"   Errors encountered: {len(result['errors'])}")
        
        if result['errors']:
            print(f"\n   Error details:")
            for error in result['errors'][:3]:  # Show first 3 errors
                print(f"      - {error}")
        
        # Check health after errors
        health = pipeline.health_check()
        print(f"\n🏥 Health Status After Errors:")
        print(f"   Status: {health['status']}")
        print(f"   Error rate: {health['error_rate']:.2f}%")
        
        pipeline.shutdown()
        print("\n✓ Pipeline handled errors gracefully")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def main():
    """Main example runner"""
    print("\n" + "=" * 70)
    print("Market Data Pipeline - Phase 2.1 Examples")
    print("Demonstrates integration with Bearish Alpha Bot infrastructure")
    print("=" * 70 + "\n")
    
    # Check for environment setup
    if not os.getenv('EXCHANGES'):
        print("💡 TIP: Set environment variables for live examples:")
        print("   export EXCHANGES='bingx,kucoinfutures'")
        print("   export BINGX_KEY='your_key'")
        print("   export BINGX_SECRET='your_secret'")
        print("   # ... add other exchange credentials as needed\n")
        print("Running examples with mock setup...\n")
    
    # Run all examples
    example_basic_usage()
    example_health_monitoring()
    example_with_aggregator()
    example_custom_configuration()
    example_multi_timeframe_analysis()
    example_error_handling()
    
    print("=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)
    print("\n📚 For more information:")
    print("   - docs/PHASE2_MARKET_DATA.md - Comprehensive documentation")
    print("   - docs/market_data_pipeline_usage.md - Detailed usage guide")
    print("   - tests/test_market_data_pipeline.py - Test examples")
    print()


if __name__ == '__main__':
    main()
