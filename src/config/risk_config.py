"""
Risk Management Configuration.
Centralized risk parameters and circuit breaker limits.
"""

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class RiskLimits:
    """Portfolio-level risk limits."""
    max_portfolio_risk: float = 0.02  # 2% max risk per trade
    max_position_size: float = 0.10   # 10% max position size
    max_drawdown: float = 0.15        # 15% max portfolio drawdown
    max_correlation: float = 0.70     # 70% max position correlation
    stop_loss_multiplier: float = 2.0  # 2x ATR stop loss
    take_profit_ratio: float = 2.0     # 2:1 risk/reward minimum


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
    
    def __init__(self, custom_limits: Dict[str, Any] = None):
        """
        Initialize risk configuration.
        
        Args:
            custom_limits: Optional custom risk limits to override defaults
        """
        self.risk_limits = RiskLimits(**{
            k: custom_limits.get(k, v) 
            for k, v in self.DEFAULT_RISK_LIMITS.items()
        }) if custom_limits else RiskLimits()
        
        self.circuit_breaker_limits = CircuitBreakerLimits(**{
            k: custom_limits.get(k, v) 
            for k, v in self.CIRCUIT_BREAKER_LIMITS.items()
        }) if custom_limits else CircuitBreakerLimits()
        
        self.emergency_protocols = EmergencyProtocols()
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
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
