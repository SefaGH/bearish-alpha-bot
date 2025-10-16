#!/usr/bin/env python3
"""Test if signals can be generated with current config."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ccxt_client import CcxtClient
from core.indicators import add_indicators
from strategies.adaptive_ob import AdaptiveOversoldBounce
from core.market_regime import MarketRegimeAnalyzer
import yaml

# Load config
with open('config/config.example.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create client
client = CcxtClient('bingx', {
    'apiKey': os.getenv('BINGX_KEY'),
    'secret': os.getenv('BINGX_SECRET')
})

# Test symbols
symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']

for symbol in symbols:
    print(f"\nTesting {symbol}...")
    
    # Fetch data
    ohlcv = client.ohlcv(symbol, '30m', 100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Add indicators
    df = add_indicators(df, config.get('indicators', {}))
    
    # Check RSI
    last_rsi = df['rsi'].iloc[-1]
    print(f"  Current RSI: {last_rsi:.1f}")
    
    # Check signal conditions
    ob_config = config['signals']['oversold_bounce']
    if last_rsi <= ob_config['rsi_max']:
        print(f"  ✅ OVERSOLD signal possible! (RSI {last_rsi:.1f} <= {ob_config['rsi_max']})")
    
    str_config = config['signals']['short_the_rip']
    if last_rsi >= str_config['rsi_min']:
        print(f"  ✅ SHORT signal possible! (RSI {last_rsi:.1f} >= {str_config['rsi_min']})")
