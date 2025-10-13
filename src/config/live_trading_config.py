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
        """Get all configuration dictionaries."""
        return {
            'execution': cls.EXECUTION_CONFIG,
            'order': cls.ORDER_CONFIG,
            'monitoring': cls.MONITORING_CONFIG,
            'emergency': cls.EMERGENCY_CONFIG,
            'signal': cls.SIGNAL_CONFIG,
            'algo_params': cls.ALGO_PARAMS,
            'performance': cls.PERFORMANCE_CONFIG,
        }
