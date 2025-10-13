#!/usr/bin/env python3
"""
Integration test for backtest scripts with mocked exchange data.
This verifies the complete flow works end-to-end.
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import logging
logging.basicConfig(level=logging.INFO)

# Mock the CCXT client to avoid real API calls
class MockExchange:
    """Mock exchange for testing."""
    def load_markets(self):
        return {
            'BTC/USDT:USDT': {'id': 'BTCUSDT', 'symbol': 'BTC/USDT:USDT'},
        }
    
    def fetch_ohlcv(self, symbol, timeframe='30m', limit=100):
        """Return mock OHLCV data."""
        import time
        now = int(time.time() * 1000)
        data = []
        for i in range(limit):
            timestamp = now - (limit - i) * 30 * 60 * 1000  # 30m candles
            # Generate some realistic-looking price data
            base_price = 50000 + i * 10
            data.append([
                timestamp,
                base_price,      # open
                base_price + 100, # high
                base_price - 100, # low
                base_price + 50,  # close
                1000.0           # volume
            ])
        return data

class MockCcxtClient:
    """Mock CCXT client."""
    def __init__(self, name, creds=None):
        self.name = name
        self.ex = MockExchange()
    
    def validate_and_get_symbol(self, symbol):
        markets = self.ex.load_markets()
        if symbol in markets:
            return symbol
        # Try common variants
        variants = ['BTC/USDT:USDT', 'BTC/USDT', 'BTCUSDT']
        for v in variants:
            if v in markets:
                print(f"✅ Symbol fallback: {symbol} → {v}")
                return v
        raise RuntimeError(f"Symbol {symbol} not found")
    
    def ohlcv(self, symbol, timeframe='30m', limit=100):
        return self.ex.fetch_ohlcv(symbol, timeframe, limit)
    
    def fetch_ohlcv_bulk(self, symbol, timeframe='30m', target_limit=100):
        """Bulk fetch using same logic as ohlcv for mock."""
        return self.ex.fetch_ohlcv(symbol, timeframe, target_limit)

# Monkey-patch the CcxtClient
import core.ccxt_client as ccxt_client_module
original_client = ccxt_client_module.CcxtClient
ccxt_client_module.CcxtClient = MockCcxtClient

try:
    # Setup environment
    os.environ['EXCHANGES'] = 'kucoinfutures'
    os.environ['KUCOIN_KEY'] = 'mock_key'
    os.environ['KUCOIN_SECRET'] = 'mock_secret'
    os.environ['KUCOIN_PASSWORD'] = 'mock_password'
    os.environ['BT_EXCHANGE'] = 'kucoinfutures'
    os.environ['BT_SYMBOL'] = 'BTC/USDT'
    os.environ['BT_LIMIT'] = '100'
    os.environ['CONFIG_PATH'] = 'config/config.example.yaml'
    os.environ['LOG_LEVEL'] = 'INFO'
    
    print("="*60)
    print("INTEGRATION TEST: Backtest with Mocked Data")
    print("="*60)
    
    # Import and run the backtest after monkeypatching
    from backtest import param_sweep
    
    # Run the backtest
    print("\nRunning backtest...")
    try:
        param_sweep.main()
        print("\n✅ Backtest completed successfully!")
        
        # Check if output file was created
        import glob
        output_files = glob.glob('data/backtests/*.csv')
        if output_files:
            print(f"✅ Output file created: {output_files[-1]}")
            
            # Read and verify the output
            import pandas as pd
            df = pd.read_csv(output_files[-1])
            if len(df) > 0:
                print(f"✅ Results contain {len(df)} parameter combinations")
                print("\nTop 3 results:")
                print(df.head(3).to_string(index=False))
            else:
                print("⚠️ Warning: Results file is empty")
        else:
            print("⚠️ Warning: No output file created")
            
    except Exception as e:
        print(f"\n❌ Backtest failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
finally:
    # Restore original client
    ccxt_client_module.CcxtClient = original_client

print("\n" + "="*60)
print("INTEGRATION TEST PASSED")
print("="*60)
