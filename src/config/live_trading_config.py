"""
Live Trading Configuration.
Configuration settings for production live trading.
Loads default values and allows override from config/config.example.yaml
"""

from typing import Dict, List, Any
import os
import yaml


class LiveTradingConfiguration:
    """Production live trading configuration."""
    
    # Config file path - ✅ UPDATED to use config.example.yaml
    CONFIG_FILE_PATH = 'config/config.example.yaml'
    
    # Execution configuration
    EXECUTION_CONFIG: Dict[str, Any] = {
        'default_execution_algo': 'limit',
        'max_slippage_tolerance': 0.005,
        'order_timeout': 300,
        'partial_fill_threshold': 0.8,
        'retry_attempts': 3,
        'retry_delay': 2,
        'enable_live': True,
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
        'enable_smart_routing': True,
        'enable_iceberg_orders': False,
        'enable_twap': False,
        'max_order_age': 600,
        'cancel_stale_orders': True,
    }
    
    # Position monitoring configuration
    MONITORING_CONFIG: Dict[str, Any] = {
        'position_check_interval': 10,
        'pnl_update_frequency': 5,
        'risk_check_frequency': 1,
        'performance_report_interval': 3600,
        'enable_position_alerts': True,
        'alert_thresholds': {
            'pnl_drop_pct': 0.05,
            'risk_breach': True,
        },
        # ✅ REMOVED: duplicate_prevention (moved to SIGNALS_CONFIG)
    }
    
    # Emergency configuration
    EMERGENCY_CONFIG: Dict[str, Any] = {
        'max_daily_loss': 0.05,
        'max_drawdown': 0.10,
        'emergency_shutdown_triggers': [
            'exchange_disconnection',
            'risk_limit_breach',
            'system_error',
            'manual_intervention',
            'max_daily_loss_reached'
        ],
        'emergency_close_method': 'market',
        'enable_circuit_breaker': True,
    }
    
    # Signal processing configuration
    SIGNAL_CONFIG: Dict[str, Any] = {
        'max_queue_size': 100,
        'signal_timeout': 60,
        'enable_signal_validation': True,
        'priority_execution': True,
    }
    
    # Risk configuration
    RISK_CONFIG: Dict[str, Any] = {
        'equity_usd': 100,
        'per_trade_risk_pct': 0.01,
        'daily_loss_limit_pct': 0.02,
        'risk_usd_cap': 5,
        'max_notional_per_trade': 20,
        'min_stop_pct': 0.003,
        'min_amount_behavior': 'skip',
        'min_notional_behavior': 'skip',
        'daily_max_trades': 5,
    }
    
    # Universe configuration  
    UNIVERSE_CONFIG: Dict[str, Any] = {
        'fixed_symbols': [
            'BTC/USDT:USDT',
            'ETH/USDT:USDT', 
            'SOL/USDT:USDT',
            # ✅ REMOVED: Extra symbols (BNB, ADA, DOT, AVAX, LTC)
        ],
        'auto_select': False,
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
        'priority_enabled': True,  # ✅ Added
        'max_data_age': 60,  # ✅ Added
        'fallback_threshold': 3,  # ✅ Added
        'max_streams_per_exchange': {
            'bingx': 6,  # ✅ Updated from 10 to 6
            'binance': 20,
            'kucoinfutures': 15,
            'default': 10
        },
        'stream_timeframes': ['1m'],  # ✅ Only 1m
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
    
    # ✅ FIXED: Strategy signals configuration
    SIGNALS_CONFIG: Dict[str, Any] = {
        # ✅ ADDED: Duplicate prevention (moved from MONITORING_CONFIG)
        'duplicate_prevention': {
            'min_price_change_pct': 0.05,  # ✅ Match config.example.yaml
            'cooldown_seconds': 20,  # ✅ Match config.example.yaml
            'price_delta_bypass_threshold': 0.0015,  # ✅ Added
            'price_delta_bypass_enabled': True,  # ✅ Added
        },
        'oversold_bounce': {
            'enable': True,
            'ignore_regime': True,  # ✅ Updated
            'rsi_max': 45,  # ✅ Added
            'adaptive_rsi_base': 45,  # ✅ Updated
            'adaptive_rsi_range': 10,  # ✅ Updated
            'adaptive_mode': 'dynamic',  # ✅ Added
            'volatility_sensitivity': 'medium',  # ✅ Added
            'tp_atr_mult': 2.5,  # ✅ Added
            'sl_atr_mult': 1.2,  # ✅ Added
            'min_tp_pct': 0.008,  # ✅ Added
            'max_sl_pct': 0.015,  # ✅ Added
        },
        'short_the_rip': {
            'enable': True,
            'ignore_regime': True,  # ✅ Updated
            'rsi_min': 55,  # ✅ Added
            'adaptive_rsi_base': 55,  # ✅ Updated
            'adaptive_rsi_range': 10,  # ✅ Updated
            'adaptive_mode': 'dynamic',  # ✅ Added
            'volatility_sensitivity': 'medium',  # ✅ Added
            'tp_atr_mult': 3.0,  # ✅ Added
            'sl_atr_mult': 1.5,  # ✅ Added
            'min_tp_pct': 0.010,  # ✅ Added
            'max_sl_pct': 0.020,  # ✅ Added
            # ✅ ADDED: Symbol-specific RSI threshold overrides
            'symbols': {
                'BTC/USDT:USDT': {
                    'rsi_threshold': 55
                },
                'ETH/USDT:USDT': {
                    'rsi_threshold': 50
                },
                'SOL/USDT:USDT': {
                    'rsi_threshold': 50
                }
            }
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
            'price_offset_pct': 0.001,
            'max_wait_time': 60,
            'fallback_to_market': True,
        },
        'iceberg': {
            'slice_size_pct': 0.10,
            'time_between_slices': 30,
        },
        'twap': {
            'time_window': 300,
            'num_slices': 10,
            'adaptive_slicing': True,
        },
        'market': {
            'slippage_check': True,
            'max_retries': 1,
        }
    }
    
    # Performance tracking configuration
    PERFORMANCE_CONFIG: Dict[str, Any] = {
        'track_execution_quality': True,
        'calculate_implementation_shortfall': True,
        'save_execution_history': True,
        'metrics_update_interval': 60,
    }
    
    @classmethod
    def get_execution_algo_params(cls, algo_name: str) -> Dict[str, Any]:
        """Get parameters for specific execution algorithm."""
        return cls.ALGO_PARAMS.get(algo_name, cls.ALGO_PARAMS['limit'])
    
    @classmethod
    def get_all_configs(cls) -> Dict[str, Any]:
        """Get all configuration dictionaries."""
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
    def load_yaml_config(cls, config_path: str = None) -> Dict[str, Any]:
        """
        Load YAML config file (config.example.yaml by default).
        
        Args:
            config_path: Path to config file (default: config/config.example.yaml)
            
        Returns:
            Dict with loaded config values
        """
        if config_path is None:
            config_path = cls.CONFIG_FILE_PATH
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Please ensure config/config.example.yaml exists."
            )
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @classmethod
    def from_yaml_config(cls, yaml_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Convert from config.example.yaml format to production format.
        Allows using config.example.yaml values to override defaults.
        
        Args:
            yaml_config: Pre-loaded YAML config dict. If None, loads from config.example.yaml
            
        Returns:
            Dict with merged configuration (defaults + YAML overrides)
        """
        # ✅ Load YAML if not provided
        if yaml_config is None:
            yaml_config = cls.load_yaml_config()
        
        all_configs = cls.get_all_configs()
        
        # ✅ FIXED: Deep merge for signals
        if 'signals' in yaml_config:
            signals_config = yaml_config['signals']
            
            # Handle duplicate_prevention
            if 'duplicate_prevention' in signals_config:
                if 'duplicate_prevention' not in all_configs['signals']:
                    all_configs['signals']['duplicate_prevention'] = {}
                all_configs['signals']['duplicate_prevention'].update(
                    signals_config['duplicate_prevention']
                )
            
            # Handle strategy configs
            for strategy in ['oversold_bounce', 'short_the_rip']:
                if strategy in signals_config:
                    if strategy not in all_configs['signals']:
                        all_configs['signals'][strategy] = {}
                    # Deep merge
                    for key, value in signals_config[strategy].items():
                        if isinstance(value, dict):
                            if key not in all_configs['signals'][strategy]:
                                all_configs['signals'][strategy][key] = {}
                            all_configs['signals'][strategy][key].update(value)
                        else:
                            all_configs['signals'][strategy][key] = value
        
        # ✅ FIXED: Handle monitoring.duplicate_prevention for backward compatibility
        if 'monitoring' in yaml_config:
            if 'duplicate_prevention' in yaml_config['monitoring']:
                # Map to signals.duplicate_prevention
                if 'duplicate_prevention' not in all_configs['signals']:
                    all_configs['signals']['duplicate_prevention'] = {}
                all_configs['signals']['duplicate_prevention'].update(
                    yaml_config['monitoring']['duplicate_prevention']
                )
        
        # Update other sections
        if 'execution' in yaml_config:
            all_configs['execution'].update(yaml_config['execution'])
        
        if 'risk' in yaml_config:
            all_configs['risk'].update(yaml_config['risk'])
        
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
    
    @classmethod
    def get_config_with_yaml_override(cls) -> Dict[str, Any]:
        """
        Convenience method to get config with YAML overrides applied.
        Loads config/config.example.yaml and merges with defaults.
        
        Returns:
            Dict with merged configuration
        """
        return cls.from_yaml_config()
