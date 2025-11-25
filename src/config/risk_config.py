"""
Risk Management Configuration.
Centralized risk parameters and circuit breaker limits.
"""

from typing import Dict, Any, Optional
from copy import deepcopy
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


@dataclass
class QueueConfig:
    ttl_seconds: int = 60
    max_queue_depth: int = 50
    batch_dequeue: int = 3
    max_pending_per_symbol: int = 1
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        'explicit_priority': 0.4,
        'risk_reward': 0.3,
        'ml_confidence': 0.2,
        'urgency': 0.1,
    })


@dataclass
class ConcurrentRiskLimitsConfig:
    max_open_positions: int = 3
    max_positions_per_symbol: int = 1
    max_total_risk_pct: float = 0.06
    correlation_bucket_threshold: float = 0.8


@dataclass
class VolatilitySizingConfig:
    enabled: bool = True
    atr_window: int = 14
    atr_floor_pct: float = 0.005
    atr_ceiling_pct: float = 0.02
    low_vol_multiplier: float = 1.2
    baseline_multiplier: float = 1.0
    high_vol_multiplier: float = 0.6
    min_position_size_pct: float = 0.01


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
        # Preserve a flattened snapshot of the normalized risk section for downstream consumers
        self._raw_risk_config = deepcopy(custom_limits) if custom_limits else {}
        
        # Store daily_max_trades if provided
        self.daily_max_trades = custom_limits.get('daily_max_trades') if custom_limits else None
        
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

        self.risk_limits = RiskLimits(**processed_limits)
        
        self.circuit_breaker_limits = CircuitBreakerLimits(**{
            k: custom_limits.get(k, v) 
            for k, v in self.CIRCUIT_BREAKER_LIMITS.items()
        }) if custom_limits else CircuitBreakerLimits()
        
        self.emergency_protocols = EmergencyProtocols()

        queue_raw = custom_limits.get('queue') if custom_limits else None
        concurrent_raw = custom_limits.get('concurrent_limits') if custom_limits else None
        volatility_raw = custom_limits.get('volatility_sizing') if custom_limits else None

        self.queue_config = self._load_queue_config(queue_raw or {})
        self.concurrent_limits_config = self._load_concurrent_limits(concurrent_raw or {})
        self.volatility_sizing_config = self._load_volatility_sizing(volatility_raw or {})
        
        # Load dynamic R/R configuration
        self._load_dynamic_rr_config(custom_limits)
        
        # Load regime soft-weighting configuration
        self._load_regime_soft_weight_config(custom_limits)
        
        # Load signal scoring configuration
        self._load_signal_scoring_config(custom_limits)
        
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
        
        strategy_overrides = rr_config.get('strategy_overrides', {}) if rr_config else {}
        self.rr_dynamic_strategy_overrides = self._normalize_strategy_overrides(strategy_overrides)
        if self.rr_dynamic_strategy_overrides:
            logger.info("✓ Dynamic R/R strategy overrides loaded for %d strategies", len(self.rr_dynamic_strategy_overrides))
        
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
    
    def _normalize_strategy_overrides(self, overrides: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Normalize per-strategy override keys for case-insensitive lookups."""
        normalized = {}
        if not overrides:
            return normalized
        for name, override in overrides.items():
            if not isinstance(override, dict):
                continue
            normalized[name.lower()] = deepcopy(override)
        return normalized

    def _load_regime_soft_weight_config(self, config_dict: Dict[str, Any]):
        """
        Load regime soft-weighting configuration.
        
        Args:
            config_dict: Configuration dictionary from YAML
        """
        ml_config = config_dict.get('ml', {}) if config_dict else {}
        regime_config = ml_config.get('regime', {})
        
        # Build configuration with priority chain: ENV > config > defaults
        self.regime_soft_weight = {
            'enabled': self._get_env_or_config('REGIME_SOFT_WEIGHT_ENABLED',
                                              regime_config.get('soft_weighting_enabled', True),
                                              bool),
            
            'min_confidence_hard_reject': self._get_env_or_config('REGIME_MIN_CONF_REJECT',
                                                                 regime_config.get('min_confidence_hard_reject', 0.30),
                                                                 float),
            
            'min_confidence_full_weight': self._get_env_or_config('REGIME_MIN_CONF_FULL',
                                                                 regime_config.get('min_confidence_full_weight', 0.60),
                                                                 float),
        }
        
        logger.info(f"✅ Regime Soft-Weight Config: enabled={self.regime_soft_weight['enabled']}, "
                   f"hard_reject={self.regime_soft_weight['min_confidence_hard_reject']:.2f}, "
                   f"full_weight={self.regime_soft_weight['min_confidence_full_weight']:.2f}")
    
    def _load_signal_scoring_config(self, config_dict: Dict[str, Any]):
        """
        Load signal scoring configuration.
        
        Args:
            config_dict: Configuration dictionary from YAML
        """
        ml_config = config_dict.get('ml', {}) if config_dict else {}
        scoring_config = ml_config.get('signal_scoring', {})
        
        # Build configuration with priority chain: ENV > config > defaults
        self.signal_scoring = {
            'enabled': self._get_env_or_config('SIGNAL_SCORING_ENABLED',
                                              scoring_config.get('enabled', True),
                                              bool),
            
            'min_score_to_trade': self._get_env_or_config('SIGNAL_MIN_SCORE',
                                                         scoring_config.get('min_score_to_trade', 60),
                                                         int),
            
            'weights': {
                'strategy': self._get_env_or_config('SCORE_WEIGHT_STRATEGY',
                                                   scoring_config.get('weights', {}).get('strategy', 0.3),
                                                   float),
                'ml_price': self._get_env_or_config('SCORE_WEIGHT_ML',
                                                   scoring_config.get('weights', {}).get('ml_price', 0.3),
                                                   float),
                'regime': self._get_env_or_config('SCORE_WEIGHT_REGIME',
                                                 scoring_config.get('weights', {}).get('regime', 0.2),
                                                 float),
                'risk_reward': self._get_env_or_config('SCORE_WEIGHT_RR',
                                                      scoring_config.get('weights', {}).get('risk_reward', 0.2),
                                                      float),
            }
        }
        
        total_weight = sum(self.signal_scoring['weights'].values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"⚠️ Signal scoring weights sum to {total_weight:.2f}, expected 1.0")
        
        logger.info(f"✅ Signal Scoring Config: enabled={self.signal_scoring['enabled']}, "
                   f"min_score={self.signal_scoring['min_score_to_trade']}, "
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

        if self._raw_risk_config:
            # Provide the complete risk section (already normalized by LiveTradingConfiguration)
            result['risk'] = deepcopy(self._raw_risk_config)
        
        # Add daily_max_trades if available
        if hasattr(self, 'daily_max_trades') and self.daily_max_trades is not None:
            result['daily_max_trades'] = self.daily_max_trades
        
        # Add dynamic R/R config if available
        if hasattr(self, 'rr_dynamic'):
            result['rr_dynamic'] = self.rr_dynamic
        
        # Add regime soft-weight config if available
        if hasattr(self, 'regime_soft_weight'):
            result['regime_soft_weight'] = self.regime_soft_weight
        
        # Add signal scoring config if available
        if hasattr(self, 'signal_scoring'):
            result['signal_scoring'] = self.signal_scoring

        result['queue_config'] = {
            'ttl_seconds': self.queue_config.ttl_seconds,
            'max_queue_depth': self.queue_config.max_queue_depth,
            'batch_dequeue': self.queue_config.batch_dequeue,
            'max_pending_per_symbol': self.queue_config.max_pending_per_symbol,
            'priority_weights': self.queue_config.priority_weights,
        }

        result['concurrent_limits'] = {
            'max_open_positions': self.concurrent_limits_config.max_open_positions,
            'max_positions_per_symbol': self.concurrent_limits_config.max_positions_per_symbol,
            'max_total_risk_pct': self.concurrent_limits_config.max_total_risk_pct,
            'correlation_bucket_threshold': self.concurrent_limits_config.correlation_bucket_threshold,
        }

        result['volatility_sizing'] = {
            'enabled': self.volatility_sizing_config.enabled,
            'atr_window': self.volatility_sizing_config.atr_window,
            'atr_floor_pct': self.volatility_sizing_config.atr_floor_pct,
            'atr_ceiling_pct': self.volatility_sizing_config.atr_ceiling_pct,
            'low_vol_multiplier': self.volatility_sizing_config.low_vol_multiplier,
            'baseline_multiplier': self.volatility_sizing_config.baseline_multiplier,
            'high_vol_multiplier': self.volatility_sizing_config.high_vol_multiplier,
            'min_position_size_pct': self.volatility_sizing_config.min_position_size_pct,
        }

        return result

    def get_flat_risk_settings(self) -> Dict[str, Any]:
        """Return a deepcopy of the normalized risk section for consumers like PositionSizing."""
        return deepcopy(self._raw_risk_config)

    def get_queue_config(self) -> QueueConfig:
        return self.queue_config

    def get_concurrent_limits(self) -> ConcurrentRiskLimitsConfig:
        return self.concurrent_limits_config

    def get_volatility_sizing(self) -> VolatilitySizingConfig:
        return self.volatility_sizing_config

    def _load_queue_config(self, cfg: Dict[str, Any]) -> QueueConfig:
        weights = cfg.get('priority_weights') or {}
        priority_weights = {
            'explicit_priority': self._safe_float(weights.get('explicit_priority', 0.4), 0.4),
            'risk_reward': self._safe_float(weights.get('risk_reward', 0.3), 0.3),
            'ml_confidence': self._safe_float(weights.get('ml_confidence', 0.2), 0.2),
            'urgency': self._safe_float(weights.get('urgency', 0.1), 0.1),
        }
        total = sum(priority_weights.values())
        if total <= 0:
            priority_weights = {
                'explicit_priority': 0.4,
                'risk_reward': 0.3,
                'ml_confidence': 0.2,
                'urgency': 0.1,
            }
            total = 1.0
        priority_weights = {k: v / total for k, v in priority_weights.items()}

        return QueueConfig(
            ttl_seconds=self._safe_int(cfg.get('ttl_seconds', 60), 60),
            max_queue_depth=self._safe_int(cfg.get('max_queue_depth', 50), 50),
            batch_dequeue=self._safe_int(cfg.get('batch_dequeue', 3), 3, minimum=1),
            max_pending_per_symbol=self._safe_int(cfg.get('max_pending_per_symbol', 1), 1, minimum=1),
            priority_weights=priority_weights
        )

    def _load_concurrent_limits(self, cfg: Dict[str, Any]) -> ConcurrentRiskLimitsConfig:
        max_total_risk = cfg.get('max_total_risk_pct', 0.06)
        if max_total_risk > 1:
            max_total_risk /= 100.0
        correlation_threshold = cfg.get('correlation_bucket_threshold', 0.8)
        if correlation_threshold > 1:
            correlation_threshold /= 100.0

        return ConcurrentRiskLimitsConfig(
            max_open_positions=self._safe_int(cfg.get('max_open_positions', 3), 3, minimum=0),
            max_positions_per_symbol=self._safe_int(cfg.get('max_positions_per_symbol', 1), 1, minimum=0),
            max_total_risk_pct=float(max_total_risk),
            correlation_bucket_threshold=float(correlation_threshold)
        )

    def _load_volatility_sizing(self, cfg: Dict[str, Any]) -> VolatilitySizingConfig:
        floor_pct = cfg.get('atr_floor_pct', 0.005)
        ceiling_pct = cfg.get('atr_ceiling_pct', 0.02)
        min_pos_pct = cfg.get('min_position_size_pct', 0.01)

        def normalize_pct(value: Any, fallback: float) -> float:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return fallback
            if numeric > 1:
                numeric /= 100.0
            return max(numeric, 0.0)

        floor_pct = normalize_pct(floor_pct, 0.005)
        ceiling_pct = normalize_pct(ceiling_pct, 0.02)
        if ceiling_pct <= floor_pct:
            ceiling_pct = floor_pct * 1.5
        min_pos_pct = normalize_pct(min_pos_pct, 0.01)

        return VolatilitySizingConfig(
            enabled=bool(cfg.get('enabled', True)),
            atr_window=self._safe_int(cfg.get('atr_window', 14), 14, minimum=1),
            atr_floor_pct=floor_pct,
            atr_ceiling_pct=ceiling_pct,
            low_vol_multiplier=self._safe_float(cfg.get('low_vol_multiplier', 1.2), 1.2, minimum=0.1),
            baseline_multiplier=self._safe_float(cfg.get('baseline_multiplier', 1.0), 1.0, minimum=0.1),
            high_vol_multiplier=self._safe_float(cfg.get('high_vol_multiplier', 0.6), 0.6, minimum=0.05),
            min_position_size_pct=min_pos_pct
        )

    @staticmethod
    def _safe_int(value: Any, fallback: int, minimum: Optional[int] = None) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = fallback
        if minimum is not None and numeric < minimum:
            numeric = minimum
        return numeric

    @staticmethod
    def _safe_float(value: Any, fallback: float, minimum: Optional[float] = None) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = fallback
        if minimum is not None and numeric < minimum:
            numeric = minimum
        return numeric
