"""
Live Trading Configuration.
Configuration settings for production live trading.
"""

from typing import Dict, List, Any


class LiveTradingConfiguration:
    """Production live trading configuration."""
    
    # Execution configuration
    EXECUTION_CONFIG: Dict[str, Any] = {
        'default_execution_algo': 'limit',
        'max_slippage_tolerance': 0.005,  # 0.5% max slippage
        'order_timeout': 300,  # 5 minutes order timeout
        'partial_fill_threshold': 0.8,  # 80% fill threshold
        'retry_attempts': 3,  # Number of retry attempts for failed orders
        'retry_delay': 2,  # Seconds between retries
        'enable_live': True,  # Live trading enabled
        'order_type': 'market',
        'time_in_force': 'IOC',
        'fee_pct': 0.0006,
        'leverage': {
            'default': 5,
            'overrides': {}
        }
    }
    
    # Order management configuration
    ORDER_CONFIG: Dict[str, Any] = {
        'enable_smart_routing': True,  # Enable smart order routing
        'enable_iceberg_orders': False,  # Enable iceberg orders (large orders)
        'enable_twap': False,  # Enable TWAP execution
        'max_order_age': 600,  # Maximum order age in seconds (10 minutes)
        'cancel_stale_orders': True,  # Auto-cancel stale orders
    }
    
    # Position monitoring configuration
    MONITORING_CONFIG: Dict[str, Any] = {
        'position_check_interval': 10,  # 10 seconds
        'pnl_update_frequency': 5,  # 5 seconds
        'risk_check_frequency': 1,  # 1 second
        'performance_report_interval': 3600,  # 1 hour
        'enable_position_alerts': True,  # Enable position alerts
        'alert_thresholds': {
            'pnl_drop_pct': 0.05,  # Alert on 5% P&L drop
            'risk_breach': True,  # Alert on risk limit breach
        },
        'duplicate_prevention': {
            'enabled': True,
            'same_symbol_cooldown': 60,  # Reduced from 300s to 60s for testing
            'same_strategy_cooldown': 60,  # Reduced from 180s to 60s (deprecated, uses same_symbol_cooldown)
            'min_price_change': 0.002,  # 0.2% minimum price change
        }
    }
    
    # Emergency configuration
    EMERGENCY_CONFIG: Dict[str, Any] = {
        'max_daily_loss': 0.05,  # 5% max daily loss
        'max_drawdown': 0.10,  # 10% max drawdown trigger
        'emergency_shutdown_triggers': [
            'exchange_disconnection',
            'risk_limit_breach',
            'system_error',
            'manual_intervention',
            'max_daily_loss_reached'
        ],
        'emergency_close_method': 'market',  # Use market orders for emergency closes
        'enable_circuit_breaker': True,  # Enable circuit breaker
    }
    
    # Signal processing configuration
    SIGNAL_CONFIG: Dict[str, Any] = {
        'max_queue_size': 100,  # Maximum signal queue size
        'signal_timeout': 60,  # Signal validity timeout in seconds
        'enable_signal_validation': True,  # Enable signal validation
        'priority_execution': True,  # Enable priority-based execution
    }
    
    # Risk configuration
    RISK_CONFIG: Dict[str, Any] = {
        'equity_usd': 100,  # Default $100 capital
        'per_trade_risk_pct': 0.01,  # 1% risk per trade
        'daily_loss_limit_pct': 0.02,  # 2% daily loss limit
        'risk_usd_cap': 5,  # Max $5 risk per trade
        'max_notional_per_trade': 20,  # Max $20 per trade
        'min_stop_pct': 0.003,  # Min stop distance 0.3%
        'min_amount_behavior': 'skip',
        'min_notional_behavior': 'skip',
        'daily_max_trades': 5,  # Max 5 trades per day
    }
    
    # Universe configuration  
    UNIVERSE_CONFIG: Dict[str, Any] = {
        'fixed_symbols': [
            'BTC/USDT:USDT',
            'ETH/USDT:USDT', 
            'SOL/USDT:USDT',
            'BNB/USDT:USDT',
            'ADA/USDT:USDT',
            'DOT/USDT:USDT',
            'AVAX/USDT:USDT',
            'LTC/USDT:USDT'
        ],
        'auto_select': False,  # Use fixed symbols only
        'priority_order': ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT'],
        'min_quote_volume_usdt': 1000000,
        'prefer_perps': True,
        'max_symbols_per_exchange': 80,
        'top_n_per_exchange': 15,
        'only_linear': True
    }
    
    # WebSocket configuration
    WEBSOCKET_CONFIG: Dict[str, Any] = {
        'enabled': True,
        'max_streams_per_exchange': {
            'bingx': 10,
            'binance': 20,
            'kucoinfutures': 15,
            'default': 10
        },
        'stream_timeframes': ['1m', '5m'],
        'reconnect_delay': 5,
        'max_reconnect_attempts': 3
    }
    
    # Notification configuration
    NOTIFY_CONFIG: Dict[str, Any] = {
        'send_all': True,
        'push_no_signal': True,
        'push_debug': True,
        'min_cooldown_sec': 300,
        'push_trail_updates': False
    }
    
    # Indicator configuration
    INDICATORS_CONFIG: Dict[str, Any] = {
        'rsi_period': 14,
        'atr_period': 14,
        'ema_fast': 21,
        'ema_mid': 50,
        'ema_slow': 200
    }
    
    # Strategy signals configuration
    SIGNALS_CONFIG: Dict[str, Any] = {
        'oversold_bounce': {
            'enable': True,
            'ignore_regime': False,  # Production'da false
            'adaptive_rsi_base': 40,
            'adaptive_rsi_range': 15,
            'tp_pct': 0.015,
            'sl_atr_mult': 1.0
        },
        'short_the_rip': {
            'enable': True,
            'ignore_regime': False,
            'adaptive_rsi_base': 65,
            'adaptive_rsi_range': 20,
            'tp_pct': 0.012,
            'sl_atr_mult': 1.2
        }
    }
    
    # Quarantine configuration
    QUARANTINE_CONFIG: Dict[str, Any] = {
        'enable': True,
        'days': 7,
        'file': 'data/quarantine.json'
    }
    
    # Regime configuration
    REGIME_CONFIG: Dict[str, Any] = {
        'min_slow_candles': 90
    }
    
    # Execution algorithm parameters
    ALGO_PARAMS: Dict[str, Dict[str, Any]] = {
        'limit': {
            'price_offset_pct': 0.001,  # 0.1% price offset for limit orders
            'max_wait_time': 60,  # Max wait time for limit order fill
            'fallback_to_market': True,  # Fallback to market if timeout
        },
        'iceberg': {
            'slice_size_pct': 0.10,  # 10% of total order per slice
            'time_between_slices': 30,  # 30 seconds between slices
        },
        'twap': {
            'time_window': 300,  # 5 minutes execution window
            'num_slices': 10,  # Number of order slices
            'adaptive_slicing': True,  # Adapt to market conditions
        },
        'market': {
            'slippage_check': True,  # Check slippage after execution
            'max_retries': 1,  # Max retries for market orders
        }
    }
    
    # Performance tracking configuration
    PERFORMANCE_CONFIG: Dict[str, Any] = {
        'track_execution_quality': True,  # Track execution metrics
        'calculate_implementation_shortfall': True,  # Calculate shortfall
        'save_execution_history': True,  # Save execution history
        'metrics_update_interval': 60,  # Update metrics every minute
    }
    
    @classmethod
    def get_execution_algo_params(cls, algo_name: str) -> Dict[str, Any]:
        """Get parameters for specific execution algorithm."""
        return cls.ALGO_PARAMS.get(algo_name, cls.ALGO_PARAMS['limit'])
    
    @classmethod
    def get_all_configs(cls) -> Dict[str, Any]:
        """Get all configuration dictionaries - PRODUCTION_COORDINATOR İÇİN!"""
        return {
            'execution': cls.EXECUTION_CONFIG,
            'order': cls.ORDER_CONFIG,
            'monitoring': cls.MONITORING_CONFIG,
            'emergency': cls.EMERGENCY_CONFIG,
            'signal': cls.SIGNAL_CONFIG,
            'risk': cls.RISK_CONFIG,
            'universe': cls.UNIVERSE_CONFIG,
            'websocket': cls.WEBSOCKET_CONFIG,
            'notify': cls.NOTIFY_CONFIG,
            'indicators': cls.INDICATORS_CONFIG,
            'signals': cls.SIGNALS_CONFIG,
            'quarantine': cls.QUARANTINE_CONFIG,
            'regime': cls.REGIME_CONFIG,
            'algo_params': cls.ALGO_PARAMS,
            'performance': cls.PERFORMANCE_CONFIG,
        }
    
    @classmethod
    def from_yaml_config(cls, yaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert from config.yaml format to production format.
        Allows using config.yaml values to override defaults.
        """
        all_configs = cls.get_all_configs()
        
        # Override with yaml config values if present
        if 'execution' in yaml_config:
            all_configs['execution'].update(yaml_config['execution'])
        
        if 'risk' in yaml_config:
            all_configs['risk'].update(yaml_config['risk'])
            
        if 'signals' in yaml_config:
            all_configs['signals'].update(yaml_config['signals'])
            
        if 'universe' in yaml_config:
            all_configs['universe'].update(yaml_config['universe'])
            
        if 'websocket' in yaml_config:
            all_configs['websocket'].update(yaml_config['websocket'])
            
        if 'notify' in yaml_config:
            all_configs['notify'].update(yaml_config['notify'])
            
        if 'indicators' in yaml_config:
            all_configs['indicators'].update(yaml_config['indicators'])
            
        if 'quarantine' in yaml_config:
            all_configs['quarantine'].update(yaml_config['quarantine'])
            
        if 'regime' in yaml_config:
            all_configs['regime'].update(yaml_config['regime'])
        
        return all_configs
