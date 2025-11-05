"""
Risk Management Configuration.
Centralized risk parameters and circuit breaker limits.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Portfolio-level risk limits."""
    max_portfolio_risk: float = 0.02  # 2% max risk per trade
    max_position_size: float = 0.10   # 10% max position size
    max_drawdown: float = 0.15        # 15% max portfolio drawdown
    max_correlation: float = 0.70     # 70% max position correlation
    stop_loss_multiplier: float = 2.0  # 2x ATR stop loss
    take_profit_ratio: float = 2.0     # 2:1 risk/reward minimum

    # Sentinel değerler - None = dinamik olarak belirlenecek
    take_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None


@dataclass
class CircuitBreakerLimits:
    """Circuit breaker triggers and thresholds."""
    daily_loss_limit: float = 0.05          # 5% daily loss limit
    position_loss_limit: float = 0.03       # 3% position loss limit
    volatility_spike_threshold: float = 3.0  # 3 sigma volatility spike
    correlation_spike_threshold: float = 0.9  # 90% correlation spike


@dataclass
class EmergencyProtocols:
    """Emergency response protocols."""
    protocols: Dict[str, str] = field(default_factory=lambda: {
        'market_crash': 'close_all_positions',
        'exchange_issue': 'redistribute_positions',
        'volatility_spike': 'reduce_position_sizes',
        'correlation_spike': 'close_correlated_positions'
    })


class RiskConfiguration:
    """Centralized risk management configuration."""
    
    DEFAULT_RISK_LIMITS = {
        'max_portfolio_risk': 0.02,  # 2% max risk per trade
        'max_position_size': 0.10,   # 10% max position size
        'max_drawdown': 0.15,        # 15% max portfolio drawdown
        'max_correlation': 0.70,     # 70% max position correlation
        'stop_loss_multiplier': 2.0, # 2x ATR stop loss
        'take_profit_ratio': 2.0,    # 2:1 risk/reward minimum
        'take_profit_pct': None,     # None = dinamik
        'stop_loss_pct': None,       # None = dinamik
    }
    
    CIRCUIT_BREAKER_LIMITS = {
        'daily_loss_limit': 0.05,         # 5% daily loss limit
        'position_loss_limit': 0.03,      # 3% position loss limit  
        'volatility_spike_threshold': 3.0, # 3 sigma volatility spike
        'correlation_spike_threshold': 0.9, # 90% correlation spike
    }
    
    EMERGENCY_PROTOCOLS = {
        'market_crash': 'close_all_positions',
        'exchange_issue': 'redistribute_positions',
        'volatility_spike': 'reduce_position_sizes',
        'correlation_spike': 'close_correlated_positions'
    }
    
    def __init__(self, custom_limits: Dict[str, Any] = None, initial_capital: float = None):
        """
        Initialize risk configuration with support for dynamic values.
        
        Args:
            custom_limits: Configuration dictionary from YAML/ENV
            initial_capital: Trading capital for USD calculations
        """
        # Store capital for USD calculations
        self.initial_capital = initial_capital or (custom_limits.get('equity_usd', 100.0) if custom_limits else 100.0)
        
        # Process each limit with sentinel value awareness
        processed_limits = {}
        
        for key, default_value in self.DEFAULT_RISK_LIMITS.items():
            if custom_limits and key in custom_limits:
                value = custom_limits[key]
            else:
                value = default_value
            
            # Sentinel value kontrolü
            if key == 'take_profit_pct' and value is None:
                logger.info("✓ Take Profit: Will be calculated dynamically by strategies")
            elif key == 'stop_loss_pct' and value is None:
                logger.info("✓ Stop Loss: Will be calculated dynamically by strategies")
            elif value is not None:
                if key in ['take_profit_pct', 'stop_loss_pct']:
                    logger.info(f"✓ {key}: {value*100:.1f}% (static value)")
            
            processed_limits[key] = value
            
        self.risk_limits = RiskLimits(**{
            k: custom_limits.get(k, v) 
            for k, v in self.DEFAULT_RISK_LIMITS.items()
        }) if custom_limits else RiskLimits()
        
        self.circuit_breaker_limits = CircuitBreakerLimits(**{
            k: custom_limits.get(k, v) 
            for k, v in self.CIRCUIT_BREAKER_LIMITS.items()
        }) if custom_limits else CircuitBreakerLimits()
        
        self.emergency_protocols = EmergencyProtocols()
        
        # Load dynamic R/R configuration
        self._load_dynamic_rr_config(custom_limits)
        
        # NEW: Calculate USD amounts after loading risk limits
        self._calculate_usd_amounts(custom_limits)
    
    def _calculate_usd_amounts(self, custom_limits: Dict[str, Any] = None):
        """Calculate USD amounts based on percentages and capital."""
        # Read ENV overrides for critical risk parameters
        # Convert from percentage (e.g., 1.0 for 1%) to decimal (0.01)
        
        # Get default per_trade_risk value from config or risk limits
        default_per_trade_risk = (
            custom_limits.get('per_trade_risk_pct', self.risk_limits.max_portfolio_risk) 
            if custom_limits 
            else self.risk_limits.max_portfolio_risk
        )
        per_trade_risk_pct = self._get_env_or_config(
            'PER_TRADE_RISK_PCT', 
            default_per_trade_risk * 100,  # Convert to percentage
            float
        )
        
        # Get default daily_loss_limit value from config or circuit breaker limits
        default_daily_loss = (
            custom_limits.get('daily_loss_limit_pct', self.circuit_breaker_limits.daily_loss_limit)
            if custom_limits 
            else self.circuit_breaker_limits.daily_loss_limit
        )
        daily_loss_limit_pct = self._get_env_or_config(
            'DAILY_LOSS_LIMIT_PCT',
            default_daily_loss * 100,  # Convert to percentage
            float
        )
        
        # Calculate USD amounts
        self.max_risk_per_trade_usd = self.initial_capital * (per_trade_risk_pct / 100)
        self.daily_loss_limit_usd = self.initial_capital * (daily_loss_limit_pct / 100)
        self.max_drawdown_usd = self.initial_capital * self.risk_limits.max_drawdown
        
        # Update circuit breaker with USD values
        self.circuit_breaker_limits_usd = {
            'daily_loss_limit': self.daily_loss_limit_usd,
            'position_loss_limit': self.initial_capital * self.circuit_breaker_limits.position_loss_limit,
            'max_drawdown': self.max_drawdown_usd
        }
        
        logger.info(f"""
===== RISK USD AMOUNTS CALCULATED =====
Capital: ${self.initial_capital:.2f}
Per-Trade Risk: {per_trade_risk_pct}% = ${self.max_risk_per_trade_usd:.2f}
Daily Loss Limit: {daily_loss_limit_pct}% = ${self.daily_loss_limit_usd:.2f}
Max Drawdown: {self.risk_limits.max_drawdown:.1%} = ${self.max_drawdown_usd:.2f}
=======================================
""")
    
    def get_risk_params_for_sizing(self) -> Dict:
        """Get risk parameters with USD amounts for position sizing."""
        return {
            'max_risk_per_trade': self.risk_limits.max_portfolio_risk,
            'max_risk_amount': self.max_risk_per_trade_usd,  # USD amount
            'daily_loss_limit': self.daily_loss_limit_usd,    # USD amount
            'circuit_breaker_limits': self.circuit_breaker_limits_usd,  # USD amounts
            'initial_capital': self.initial_capital
        }
    
    def get_risk_limits(self) -> RiskLimits:
        """Get current risk limits."""
        return self.risk_limits
    
    def get_circuit_breaker_limits(self) -> CircuitBreakerLimits:
        """Get current circuit breaker limits."""
        return self.circuit_breaker_limits
    
    def get_emergency_protocol(self, event_type: str) -> str:
        """
        Get emergency protocol for event type.
        
        Args:
            event_type: Type of emergency event
            
        Returns:
            Protocol action string
        """
        return self.emergency_protocols.protocols.get(event_type, 'close_all_positions')
    
    def update_risk_limits(self, **kwargs):
        """
        Update risk limits dynamically.
        
        Args:
            **kwargs: Risk limit parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.risk_limits, key):
                setattr(self.risk_limits, key, value)
    
    def _get_env_or_config(self, env_key: str, config_value: Any, value_type=str) -> Any:
        """
        Helper to get value with priority: ENV > config > default.
        
        Args:
            env_key: Environment variable name
            config_value: Value from config file
            value_type: Type to convert to (str, bool, float, int)
            
        Returns:
            Value with priority order applied
        """
        env_value = os.getenv(env_key)
        if env_value is not None:
            if value_type == bool:
                # Enhanced boolean parsing with more edge cases
                val_lower = str(env_value).strip().lower()
                if val_lower in ['true', '1', 'yes', 'on', 'enabled']:
                    return True
                elif val_lower in ['false', '0', 'no', 'off', 'disabled', '']:
                    return False
                else:
                    logger.warning(f"Invalid boolean value '{env_value}' for {env_key}, defaulting to config value")
                    return config_value
            elif value_type == float:
                return float(env_value)
            elif value_type == int:
                return int(env_value)
            return str(env_value)
        return config_value
    
    def _load_dynamic_rr_config(self, config_dict: Dict[str, Any]):
        """
        Load dynamic R/R configuration with GitHub Variables priority.
        
        Args:
            config_dict: Configuration dictionary from YAML
        """
        rr_config = config_dict.get('rr_dynamic', {}) if config_dict else {}
        
        # Build configuration with priority chain: ENV > config > defaults
        self.rr_dynamic = {
            'enabled': self._get_env_or_config('RR_DYNAMIC_ENABLED', 
                                              rr_config.get('enabled', True), 
                                              bool),
            
            'base_target_rr': self._get_env_or_config('RR_BASE_TARGET',
                                                     rr_config.get('base_target_rr', 1.5),
                                                     float),
            
            'lower_bound_rr': self._get_env_or_config('RR_LOWER_BOUND',
                                                     rr_config.get('lower_bound_rr', 0.8),
                                                     float),
            
            'upper_bound_rr': self._get_env_or_config('RR_UPPER_BOUND',
                                                     rr_config.get('upper_bound_rr', 2.0),
                                                     float),
            
            'weights': {
                'ml_confidence': self._get_env_or_config('RR_WEIGHT_ML',
                                                        rr_config.get('weights', {}).get('ml_confidence', 0.3),
                                                        float),
                'rl_agreement': self._get_env_or_config('RR_WEIGHT_RL',
                                                       rr_config.get('weights', {}).get('rl_agreement', 0.3),
                                                       float),
                'regime_clarity': self._get_env_or_config('RR_WEIGHT_REGIME',
                                                         rr_config.get('weights', {}).get('regime_clarity', 0.2),
                                                         float),
                'volume_strength': self._get_env_or_config('RR_WEIGHT_VOLUME',
                                                          rr_config.get('weights', {}).get('volume_strength', 0.1),
                                                          float),
                'momentum_strength': self._get_env_or_config('RR_WEIGHT_MOMENTUM',
                                                            rr_config.get('weights', {}).get('momentum_strength', 0.1),
                                                            float),
            },
            
            'fallback': {
                'missing_ml_default': self._get_env_or_config('RR_FALLBACK_ML',
                                                             rr_config.get('fallback', {}).get('missing_ml_default', 0.5),
                                                             float),
                'missing_rl_default': self._get_env_or_config('RR_FALLBACK_RL',
                                                             rr_config.get('fallback', {}).get('missing_rl_default', 0.5),
                                                             float),
                'missing_regime_default': self._get_env_or_config('RR_FALLBACK_REGIME',
                                                                 rr_config.get('fallback', {}).get('missing_regime_default', 0.3),
                                                                 float),
            },
            
            'regime_multipliers': {
                'bullish': self._get_env_or_config('RR_MULT_BULLISH',
                                                  rr_config.get('regime_multipliers', {}).get('bullish', 0.9),
                                                  float),
                'bearish': self._get_env_or_config('RR_MULT_BEARISH',
                                                  rr_config.get('regime_multipliers', {}).get('bearish', 0.9),
                                                  float),
                'neutral': self._get_env_or_config('RR_MULT_NEUTRAL',
                                                  rr_config.get('regime_multipliers', {}).get('neutral', 1.0),
                                                  float),
                'volatile': self._get_env_or_config('RR_MULT_VOLATILE',
                                                   rr_config.get('regime_multipliers', {}).get('volatile', 1.2),
                                                   float),
            }
        }
        
        # Also store min_risk_reward_ratio for backward compatibility
        self.min_risk_reward_ratio = self.rr_dynamic['base_target_rr']
        
        # Validate weights configuration
        total_weight = sum(self.rr_dynamic['weights'].values())
        if total_weight > 1.5:
            logger.warning(f"⚠️ Dynamic R/R weights sum to {total_weight:.2f}, which may cause unexpected behavior. "
                         f"Consider normalizing weights to sum to 1.0")
        
        logger.info(f"✅ Dynamic R/R Config: enabled={self.rr_dynamic['enabled']}, "
                   f"base={self.rr_dynamic['base_target_rr']:.1f}, "
                   f"bounds=[{self.rr_dynamic['lower_bound_rr']:.1f}-{self.rr_dynamic['upper_bound_rr']:.1f}], "
                   f"weights_sum={total_weight:.2f}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        result = {
            'risk_limits': {
                'max_portfolio_risk': self.risk_limits.max_portfolio_risk,
                'max_position_size': self.risk_limits.max_position_size,
                'max_drawdown': self.risk_limits.max_drawdown,
                'max_correlation': self.risk_limits.max_correlation,
                'stop_loss_multiplier': self.risk_limits.stop_loss_multiplier,
                'take_profit_ratio': self.risk_limits.take_profit_ratio,
            },
            'circuit_breaker_limits': {
                'daily_loss_limit': self.circuit_breaker_limits.daily_loss_limit,
                'position_loss_limit': self.circuit_breaker_limits.position_loss_limit,
                'volatility_spike_threshold': self.circuit_breaker_limits.volatility_spike_threshold,
                'correlation_spike_threshold': self.circuit_breaker_limits.correlation_spike_threshold,
            },
            'emergency_protocols': self.emergency_protocols.protocols
        }
        
        # Add dynamic R/R config if available
        if hasattr(self, 'rr_dynamic'):
            result['rr_dynamic'] = self.rr_dynamic
        
        return result
