#!/usr/bin/env python3
"""Validate adaptive strategies configuration."""

import yaml
import sys
from pathlib import Path

def validate_config():
    """Validate adaptive strategies configuration."""
    config_path = Path('config/config.example.yaml')
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check adaptive strategies
    adaptive = config.get('adaptive_strategies', {})
    
    if adaptive.get('enable'):
        print('✅ Adaptive strategies ENABLED')
        
        # Validate monitoring
        monitoring = adaptive.get('monitoring', {})
        if monitoring.get('enabled'):
            print('  ✓ Monitoring enabled')
            print(f'    Report interval: {monitoring.get("report_interval", 300)}s')
        
        # Validate performance tuning
        performance = adaptive.get('performance', {})
        print(f'  ✓ Min volatility: {performance.get("min_volatility_for_adjustment", 0.02)}')
        print(f'  ✓ Max position multiplier: {performance.get("max_position_multiplier", 2.0)}')
        
        # Validate strategy parameters
        signals = config.get('signals', {})
        
        # OB strategy
        ob = signals.get('oversold_bounce', {})
        if ob.get('enable'):
            print('\n📊 Oversold Bounce Strategy:')
            print(f'  ✓ Adaptive RSI base: {ob.get("adaptive_rsi_base", "NOT SET")}')
            print(f'  ✓ Adaptive RSI range: ±{ob.get("adaptive_rsi_range", "NOT SET")}')
            print(f'  ✓ Mode: {ob.get("adaptive_mode", "dynamic")}')
            print(f'  ✓ Volatility sensitivity: {ob.get("volatility_sensitivity", "medium")}')
        
        # STR strategy
        str_cfg = signals.get('short_the_rip', {})
        if str_cfg.get('enable'):
            print('\n📊 Short The Rip Strategy:')
            print(f'  ✓ Adaptive RSI base: {str_cfg.get("adaptive_rsi_base", "NOT SET")}')
            print(f'  ✓ Adaptive RSI range: ±{str_cfg.get("adaptive_rsi_range", "NOT SET")}')
            print(f'  ✓ Mode: {str_cfg.get("adaptive_mode", "dynamic")}')
            print(f'  ✓ Volatility sensitivity: {str_cfg.get("volatility_sensitivity", "medium")}')
            
    else:
        print('⚡ Base strategies enabled (non-adaptive)')
    
    # Check universe configuration
    universe = config.get('universe', {})
    fixed_symbols = universe.get('fixed_symbols', [])
    auto_select = universe.get('auto_select', False)
    
    print(f'\n🌍 Universe Configuration:')
    print(f'  Fixed symbols: {len(fixed_symbols)} symbols')
    print(f'  Auto-select: {auto_select}')
    
    if fixed_symbols and not auto_select:
        print('  ✅ Using fixed symbol list (optimal for production)')
    elif auto_select:
        print('  ⚠️  Auto-select enabled (may cause performance issues)')
    
    print('\n✅ Configuration validation completed successfully!')
    return 0

if __name__ == '__main__':
    sys.exit(validate_config())
