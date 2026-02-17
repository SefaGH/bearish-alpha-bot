"""
Risk Management Configuration.
Centralized risk parameters and circuit breaker limits.
"""

from typing import Dict, Any, Optional, Literal
from copy import deepcopy
from dataclasses import dataclass, field
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Portfolio-level risk limits."""
    max_portfolio_risk: float = 0.06   # 6% max portfolio heat cap (balanced preset)
    max_position_size: float = 0.25    # 25% max position size (ratio)
    max_drawdown: float = 0.15        # 15% max portfolio drawdown
    max_correlation: float = 0.70     # 70% max position correlation
    stop_loss_multiplier: float = 2.0  # 2x ATR stop loss
    take_profit_ratio: float = 2.0     # 2:1 risk/reward minimum

    # Stop floor to prevent microscopic stops from inflating position size
    min_stop_pct: float = 0.001  # 0.1% default

    # New limit controls (Single Source Of Truth for RiskManager)
    max_position_notional_usd: Optional[float] = None
    position_size_policy: Literal["clip", "reject"] = "clip"
    min_notional_threshold: float = 5.0

    # Sentinel değerler - None = dinamik olarak belirlenecek
    take_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None

    def __post_init__(self):
        if self.position_size_policy not in ("clip", "reject"):
            raise ValueError(
                f"Invalid position_size_policy='{self.position_size_policy}'. Must be 'clip' or 'reject'."
            )

        if not 0 <= self.max_position_size <= 1.0:
            raise ValueError(
                f"max_position_size must be between 0 and 1.0, got {self.max_position_size}"
            )

        if self.min_notional_threshold < 0:
            raise ValueError(
                f"min_notional_threshold must be >= 0, got {self.min_notional_threshold}"
            )

        if self.min_stop_pct is not None and self.min_stop_pct < 0:
            raise ValueError(
                f"min_stop_pct must be >= 0, got {self.min_stop_pct}"
            )


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
        'regime_alignment': 0.05,
        'strategy_urgency': 0.05,
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

    PER_TRADE_RISK_DEFAULT = 0.003  # 0.3% per-trade risk (balanced preset)

    DEFAULT_RISK_LIMITS = {
        'max_portfolio_risk': 0.06,   # 6% max portfolio heat cap (balanced preset)
        'max_position_size': 0.25,    # 25% max position size
        'max_drawdown': 0.15,        # 15% max portfolio drawdown
        'max_correlation': 0.70,     # 70% max position correlation
        'stop_loss_multiplier': 2.0, # 2x ATR stop loss
        'take_profit_ratio': 2.0,    # 2:1 risk/reward minimum
        'min_stop_pct': 0.001,       # 0.1% stop floor
        'max_position_notional_usd': None,
        'position_size_policy': 'clip',
        'min_notional_threshold': 5.0,
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
        # Store capital for USD calculations (Single Source of Truth: risk.equity_usd)
        equity_from_cfg = None
        if isinstance(custom_limits, dict) and custom_limits.get('equity_usd') is not None:
            try:
                equity_from_cfg = float(custom_limits.get('equity_usd'))
            except (TypeError, ValueError):
                equity_from_cfg = None

        if initial_capital is not None:
            try:
                self.initial_capital = float(initial_capital)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid initial_capital={initial_capital!r}; expected float") from None
        elif equity_from_cfg is not None:
            self.initial_capital = float(equity_from_cfg)
        else:
            raise ValueError(
                "Missing starting capital: set `risk.equity_usd` in config (or override via `CAPITAL_USDT`)."
            )
        # Preserve a flattened snapshot of the normalized risk section for downstream consumers
        self._raw_risk_config = deepcopy(custom_limits) if custom_limits else {}

        # Polymorphic risk management: per-strategy profile overrides.
        self.strategy_profiles: Dict[str, Dict[str, Any]] = {}
        self._load_strategy_profiles(custom_limits or {})
        
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

        # Normalize percent-style inputs for key risk limits when provided via config dict.
        # LiveTradingConfiguration already normalizes these, but RiskConfiguration also supports
        # direct instantiation in tests/tools.
        for pct_key in ('max_drawdown', 'max_portfolio_risk', 'max_position_size'):
            try:
                raw = processed_limits.get(pct_key)
                if raw is None or isinstance(raw, bool):
                    continue
                numeric = float(raw)
                if numeric > 1:
                    processed_limits[pct_key] = numeric / 100.0
            except Exception:
                continue

        # ENV → YAML → default priority for min_stop_pct
        processed_limits['min_stop_pct'] = self._get_env_or_config(
            'RISK_MIN_STOP_PCT',
            processed_limits.get('min_stop_pct', 0.001),
            float
        )
        processed_limits['min_stop_pct'] = self._safe_float_optional(
            processed_limits.get('min_stop_pct'),
            default=0.001,
            field_name='risk.min_stop_pct',
        )
        if processed_limits['min_stop_pct'] is not None and processed_limits['min_stop_pct'] > 1:
            # Normalize percent-style values (e.g., 0.5 -> 0.5%, 50 -> 50%)
            processed_limits['min_stop_pct'] = processed_limits['min_stop_pct'] / 100.0

        # Max position notional (total exposure cap):
        # - explicit override (env/config) > computed (pct-based) > legacy computed > None (unlimited)
        explicit_max_notional = self._get_env_or_config(
            'MAX_POSITION_NOTIONAL_USD',
            custom_limits.get('max_position_notional_usd') if custom_limits else None,
            float
        )
        explicit_max_notional = self._safe_float_optional(
            explicit_max_notional,
            default=None,
            field_name='risk.max_position_notional_usd',
        )
        computed_max_notional = None
        if custom_limits:
            # Prefer pct-based cap when present (e.g., 10.0 == 10x equity notional cap)
            computed_pct = custom_limits.get('max_notional_pct_per_trade')
            try:
                computed_pct = float(computed_pct) if computed_pct is not None else None
            except Exception:
                computed_pct = None
            if computed_pct is not None and computed_pct > 0 and self.initial_capital:
                computed_max_notional = float(self.initial_capital) * float(computed_pct)
            else:
                computed_max_notional = custom_limits.get('computed_max_notional_usd') or custom_limits.get('max_notional_per_trade')
        computed_max_notional = self._safe_float_optional(
            computed_max_notional,
            default=None,
            field_name='risk.computed_max_notional_usd',
        )

        max_notional_choice = None
        if explicit_max_notional is not None and explicit_max_notional > 0:
            max_notional_choice = explicit_max_notional
        elif computed_max_notional is not None and computed_max_notional > 0:
            max_notional_choice = computed_max_notional

        if max_notional_choice is not None:
            processed_limits['max_position_notional_usd'] = float(max_notional_choice)
        # Else: keep None (unlimited) when not explicitly configured

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

    def _load_strategy_profiles(self, custom_limits: Dict[str, Any]) -> None:
        """Load per-strategy risk profiles with case-insensitive keys."""
        raw_profiles = custom_limits.get('strategy_profiles') if isinstance(custom_limits, dict) else None
        if raw_profiles is None:
            self.strategy_profiles = {}
            return
        if not isinstance(raw_profiles, dict):
            logger.warning("Ignoring invalid risk.strategy_profiles (expected dict): %s", type(raw_profiles))
            self.strategy_profiles = {}
            return

        profiles: Dict[str, Dict[str, Any]] = {}
        for name, profile in raw_profiles.items():
            if not isinstance(name, str):
                continue
            key = name.strip().lower()
            if not key:
                continue
            if not isinstance(profile, dict):
                logger.warning("Ignoring invalid strategy profile '%s' (expected dict): %s", name, type(profile))
                continue
            profiles[key] = deepcopy(profile)

        self.strategy_profiles = profiles
        if self.strategy_profiles:
            logger.info("✓ Strategy risk profiles loaded for %d strategies", len(self.strategy_profiles))

    @staticmethod
    def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge override values into base (override wins)."""
        merged = deepcopy(base) if isinstance(base, dict) else {}
        if not isinstance(override, dict):
            return merged
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = RiskConfiguration._deep_merge_dict(merged[key], value)
            else:
                merged[key] = deepcopy(value) if isinstance(value, dict) else value
        return merged

    def get_strategy_profile(self, strategy_name: Optional[str]) -> Dict[str, Any]:
        """Resolve a strategy's effective risk profile (global defaults + overrides).

        Precedence for R/R dynamic settings:
        1) Legacy: `risk.rr_dynamic.strategy_overrides.<strategy>`
        2) New: `risk.strategy_profiles.<strategy>.rr_dynamic` (wins if both present)
        """
        base_profile: Dict[str, Any] = {}
        if hasattr(self, 'rr_dynamic'):
            base_profile['rr_dynamic'] = deepcopy(self.rr_dynamic)

        if not strategy_name:
            return base_profile

        key = str(strategy_name).strip().lower()
        if not key:
            return base_profile

        # Legacy per-strategy overrides (risk.rr_dynamic.strategy_overrides)
        try:
            legacy_overrides = getattr(self, 'rr_dynamic_strategy_overrides', {}) or {}
        except Exception:
            legacy_overrides = {}
        legacy_override = None
        if isinstance(legacy_overrides, dict):
            legacy_override = legacy_overrides.get(key) or legacy_overrides.get(strategy_name)
        if isinstance(legacy_override, dict) and 'rr_dynamic' in base_profile:
            base_profile['rr_dynamic'] = self._deep_merge_dict(base_profile['rr_dynamic'], legacy_override)

        # New polymorphic strategy profiles (risk.strategy_profiles)
        override = self.strategy_profiles.get(key)
        if not isinstance(override, dict):
            return base_profile

        # Merge known nested blocks first.
        rr_override = override.get('rr_dynamic')
        if isinstance(rr_override, dict) and 'rr_dynamic' in base_profile:
            base_profile['rr_dynamic'] = self._deep_merge_dict(base_profile['rr_dynamic'], rr_override)

        # Merge any additional keys so other rules can adopt profiles later.
        for ov_key, ov_value in override.items():
            if ov_key == 'rr_dynamic':
                continue
            if isinstance(ov_value, dict) and isinstance(base_profile.get(ov_key), dict):
                base_profile[ov_key] = self._deep_merge_dict(base_profile[ov_key], ov_value)
            else:
                base_profile[ov_key] = deepcopy(ov_value) if isinstance(ov_value, dict) else ov_value

        return base_profile

    def get_effective_risk_pct(self, strategy_name: Optional[str]) -> float:
        """Resolve per-trade risk fraction with Strategy Profile > Global fallback.

        Returns a fraction (e.g., 0.02 for 2%).
        """
        default_pct = getattr(self, "per_trade_risk_pct", self.PER_TRADE_RISK_DEFAULT)
        if not strategy_name:
            return float(default_pct or 0.0)

        key = str(strategy_name).strip().lower()
        if not key:
            return float(default_pct or 0.0)

        override = None
        if isinstance(getattr(self, "strategy_profiles", None), dict):
            override = self.strategy_profiles.get(key)

        if isinstance(override, dict) and "per_trade_risk_pct" in override:
            normalized = self._normalize_fraction_value(
                override.get("per_trade_risk_pct"),
                f"risk.strategy_profiles.{key}.per_trade_risk_pct",
            )
            try:
                normalized = float(normalized)
            except (TypeError, ValueError):
                normalized = float(default_pct or 0.0)
            if normalized < 0:
                return 0.0
            return min(normalized, 1.0)

        return float(default_pct or 0.0)
    
    def _calculate_usd_amounts(self, custom_limits: Dict[str, Any] = None):
        """Calculate USD amounts based on percentages and capital."""
        # Read ENV overrides for critical risk parameters
        # Convert from percentage (e.g., 1.0 for 1%) to decimal (0.01)

        # Get default per_trade_risk value from config or the balanced preset
        default_per_trade_risk = (
            custom_limits.get('per_trade_risk_pct', self.PER_TRADE_RISK_DEFAULT)
            if custom_limits
            else self.PER_TRADE_RISK_DEFAULT
        )
        per_trade_risk_value = self._get_env_or_config(
            'PER_TRADE_RISK_PCT',
            default_per_trade_risk,  # already normalized to fraction
            float
        )
        per_trade_risk_pct = self._normalize_fraction_value(per_trade_risk_value, 'PER_TRADE_RISK_PCT')
        self.per_trade_risk_pct = per_trade_risk_pct
        
        # Get default daily_loss_limit value from config or circuit breaker limits
        default_daily_loss = (
            custom_limits.get('daily_loss_limit_pct', self.circuit_breaker_limits.daily_loss_limit)
            if custom_limits 
            else self.circuit_breaker_limits.daily_loss_limit
        )
        daily_loss_limit_value = self._get_env_or_config(
            'DAILY_LOSS_LIMIT_PCT',
            default_daily_loss,  # already normalized to fraction
            float
        )
        daily_loss_limit_pct = self._normalize_fraction_value(daily_loss_limit_value, 'DAILY_LOSS_LIMIT_PCT')
        self.daily_loss_limit_pct = daily_loss_limit_pct
        
        # Calculate USD amounts
        self.max_risk_per_trade_usd = self.initial_capital * per_trade_risk_pct
        self.daily_loss_limit_usd = self.initial_capital * daily_loss_limit_pct
        self.max_drawdown_usd = self.initial_capital * self.risk_limits.max_drawdown
        self.max_portfolio_risk_usd = self.initial_capital * self.risk_limits.max_portfolio_risk
        
        # Update circuit breaker with USD values
        self.circuit_breaker_limits_usd = {
            'daily_loss_limit': self.daily_loss_limit_usd,
            'position_loss_limit': self.initial_capital * self.circuit_breaker_limits.position_loss_limit,
            'max_drawdown': self.max_drawdown_usd
        }
        
        logger.info(f"""
===== RISK USD AMOUNTS CALCULATED =====
Capital: ${self.initial_capital:.2f}
Per-Trade Risk: {per_trade_risk_pct*100:.2f}% = ${self.max_risk_per_trade_usd:.2f}
Daily Loss Limit: {daily_loss_limit_pct*100:.2f}% = ${self.daily_loss_limit_usd:.2f}
Max Drawdown: {self.risk_limits.max_drawdown:.1%} = ${self.max_drawdown_usd:.2f}
Portfolio Heat Cap: {self.risk_limits.max_portfolio_risk:.1%} = ${self.max_portfolio_risk_usd:.2f}
=======================================
""")

        risk_usd_cap = None
        try:
            raw_cap = (custom_limits or {}).get('risk_usd_cap') if custom_limits else None
            if raw_cap is not None:
                risk_usd_cap = float(raw_cap)
        except Exception:
            risk_usd_cap = None
        self.risk_usd_cap = risk_usd_cap

        size_planner_mode = None
        try:
            if custom_limits and 'size_planner_enabled' in custom_limits:
                size_planner_mode = bool(custom_limits.get('size_planner_enabled'))
        except Exception:
            size_planner_mode = None

        def _safe_round(value: Any, places: int = 6):
            try:
                return round(value, places)
            except Exception:
                return None

        raw_max_notional_multiple = None
        try:
            if custom_limits and custom_limits.get('max_notional_pct_per_trade') is not None:
                raw_max_notional_multiple = float(custom_limits.get('max_notional_pct_per_trade'))
        except Exception:
            raw_max_notional_multiple = None

        logger.info(
            "[RISK-CONFIG-SNAPSHOT] %s",
            {
                'equity_usd': _safe_round(self.initial_capital, 4),
                'per_trade_risk_pct': _safe_round(per_trade_risk_pct),
                'max_portfolio_risk_pct': _safe_round(self.risk_limits.max_portfolio_risk),
                'max_position_size_pct': _safe_round(self.risk_limits.max_position_size),
                'max_notional_pct_per_trade': _safe_round(raw_max_notional_multiple),
                'max_position_notional_usd': _safe_round(self.risk_limits.max_position_notional_usd),
                'min_stop_pct': _safe_round(self.risk_limits.min_stop_pct or 0.0),
                'min_notional_threshold': _safe_round(self.risk_limits.min_notional_threshold or 0.0),
                'risk_usd_cap': risk_usd_cap,
                'size_planner_mode': size_planner_mode,
            },
        )

    @staticmethod
    def _normalize_fraction_value(value: Any, field_name: str) -> float:
        """Interpret value as fraction; allow percent-style inputs for backward compatibility."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            logger.warning(f"⚠️ {field_name} value '{value}' is invalid; defaulting to 0.0")
            return 0.0

        if numeric >= 1:
            logger.warning(
                f"⚠️ {field_name} appears to be percent-style ({numeric}); converting to fraction."
            )
            numeric = numeric / 100.0

        if numeric < 0:
            logger.warning(f"⚠️ {field_name} value {numeric} is negative; clamping to 0.")
            return 0.0

        return numeric
    
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
        
                'max_portfolio_risk': 0.06,   # 6% portfolio heat cap (balanced preset)
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
        Helper to coerce config values to expected types.

        NOTE: RiskConfiguration no longer reads environment variables directly.
        The single source of truth is the `custom_limits` dict passed during init,
        which is expected to already include any ENV/AppConfig overrides applied
        by `LiveTradingConfiguration`.
        
        Args:
            env_key: Environment variable name
            config_value: Value from config file
            value_type: Type to convert to (str, bool, float, int)
            
        Returns:
            Coerced value
        """
        _ = env_key  # kept for backward compatibility with call sites
        if config_value is None:
            return None

        if value_type == bool:
            if isinstance(config_value, bool):
                return config_value
            try:
                val_lower = str(config_value).strip().lower()
            except Exception:
                return bool(config_value)
            if val_lower in ['true', '1', 'yes', 'on', 'enabled']:
                return True
            if val_lower in ['false', '0', 'no', 'off', 'disabled', '']:
                return False
            logger.warning("Invalid boolean value '%s' for %s; using bool() fallback", config_value, env_key)
            return bool(config_value)

        if value_type == float:
            try:
                return float(config_value)
            except (TypeError, ValueError):
                logger.warning("Invalid float value '%s' for %s; leaving as-is", config_value, env_key)
                return config_value

        if value_type == int:
            try:
                return int(config_value)
            except (TypeError, ValueError):
                logger.warning("Invalid int value '%s' for %s; leaving as-is", config_value, env_key)
                return config_value

        return str(config_value)

    @staticmethod
    def _safe_float_optional(value: Any, default: Optional[float], field_name: str) -> Optional[float]:
        if value is None:
            return default
        if isinstance(value, bool):
            logger.warning("Invalid %s value '%s'; using default %s", field_name, value, default)
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (TypeError, ValueError):
                logger.warning("Invalid %s value '%s'; using default %s", field_name, value, default)
                return default
        logger.warning("Invalid %s value '%s'; using default %s", field_name, value, default)
        return default
    
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

            # Dynamic R/R model selector (default: v1 for backward compatibility)
            # NOTE: `RiskConfiguration` does not read env vars directly; `env_key` is ignored.
            'model_version': str(
                self._get_env_or_config(
                    'RR_MODEL_VERSION',
                    rr_config.get('model_version', 'v1'),
                    str,
                )
            ).strip().lower() or 'v1',
            
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
        regime_config = ml_config.get('regime_prediction', {})
        if not isinstance(regime_config, dict) or not regime_config:
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
            result['rr_dynamic'] = deepcopy(self.rr_dynamic)
            overrides = getattr(self, 'rr_dynamic_strategy_overrides', {}) or {}
            if overrides:
                result['rr_dynamic']['strategy_overrides'] = deepcopy(overrides)
        
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
        default_weights = {
            'explicit_priority': 0.4,
            'risk_reward': 0.3,
            'ml_confidence': 0.2,
            'urgency': 0.1,
            'regime_alignment': 0.05,
            'strategy_urgency': 0.05,
        }
        priority_weights = {}
        for key, default in default_weights.items():
            priority_weights[key] = self._safe_float(weights.get(key, default), default)

        extra_keys = set(weights.keys()) - set(default_weights.keys())
        if extra_keys:
            logger.warning("Ignoring unsupported priority weight keys: %s", ', '.join(sorted(extra_keys)))

        total = sum(priority_weights.values())
        if total <= 0:
            priority_weights = default_weights.copy()
            total = sum(priority_weights.values())
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
