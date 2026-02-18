"""
Risk Rules Engine - Modular Risk Validation Framework.

This module implements a rules-based approach to risk management, allowing
individual risk validation rules to be defined as separate, testable components.
Each rule can be independently configured, tested, and combined to create
sophisticated risk management strategies.

Phase 3 Refactor: Transform monolithic RiskManager validation into a
composable, extensible rules engine following the Open/Closed Principle.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
import math
from typing import Dict, Tuple, Any, Optional
from datetime import datetime, timezone
import logging

from core.logger import get_current_run_id
try:
    from core.rr_guard import (
        RR_REASON_BELOW_1,
        RR_REASON_BELOW_REQUIRED,
        build_prefill_rr_reason_code,
        evaluate_rr_gate,
    )
except ModuleNotFoundError:
    try:
        from src.core.rr_guard import (
            RR_REASON_BELOW_1,
            RR_REASON_BELOW_REQUIRED,
            build_prefill_rr_reason_code,
            evaluate_rr_gate,
        )
    except ModuleNotFoundError as e:
        if e.name in ("src", "src.core", "src.core.rr_guard"):
            from .rr_guard import (
                RR_REASON_BELOW_1,
                RR_REASON_BELOW_REQUIRED,
                build_prefill_rr_reason_code,
                evaluate_rr_gate,
            )
        else:
            raise

logger = logging.getLogger(__name__)


def compute_max_affordable_notional(available_balance: float, leverage: float, safety_factor: float = 0.95) -> float:
    """Shared helper for capital affordability (planner + CapitalLimitRule).

    Args:
        available_balance: free capital available (portfolio value minus exposure)
        leverage: leverage multiplier for the instrument
        safety_factor: haircut to avoid edge-of-limit fills
    """
    try:
        leverage = leverage or 1
        available_balance = float(available_balance)
        return max(0.0, available_balance * float(leverage) * float(safety_factor))
    except Exception:
        return 0.0


def compute_portfolio_open_risk_usd(portfolio_manager, signal: Dict = None, default_portfolio_value: float = 100.0) -> Tuple[float, float]:
    """Canonical portfolio heat calculator (risk USD).

    Returns (open_risk_usd, portfolio_value). Uses per-position risk = size * |entry - stop|,
    falling back to `risk_amount` when provided.
    """
    portfolio_value = _get_portfolio_value(portfolio_manager, signal or {}, default=default_portfolio_value)

    if isinstance(portfolio_manager, dict):
        active_positions = portfolio_manager.get('open_positions', portfolio_manager.get('active_positions', {})) or {}
    elif hasattr(portfolio_manager, 'get_open_positions'):
        active_positions = portfolio_manager.get_open_positions() or {}
    else:
        active_positions = {}

    total_risk = 0.0
    for pos in active_positions.values():
        try:
            if isinstance(pos, dict) and 'risk_amount' in pos:
                total_risk += float(pos.get('risk_amount') or 0.0)
                continue
            entry = float(pos.get('entry_price') or pos.get('entry') or 0.0)
            stop = float(pos.get('stop_loss') or pos.get('stop') or 0.0)
            size = float(pos.get('size') or pos.get('position_size') or pos.get('amount') or 0.0)
            if entry > 0 and stop > 0 and size > 0:
                total_risk += abs(entry - stop) * size
        except Exception:
            continue

    return total_risk, portfolio_value


def _get_portfolio_value(portfolio_manager, signal, default=100):
    """
    Helper to get portfolio value from either portfolio_manager object or fallback to signal.
    Provides backward compatibility for tests that pass dicts instead of objects.
    
    Args:
        portfolio_manager: PortfolioManager instance or dict
        signal: Trading signal with potential fallback values
        default: Default value if not found anywhere
        
    Returns:
        float: Portfolio value
    """
    if isinstance(portfolio_manager, dict) or not hasattr(portfolio_manager, 'get_current_equity'):
        return signal.get('portfolio_value', default)
    return portfolio_manager.get_current_equity()


def _get_portfolio_exposure(portfolio_manager, signal, default=0):
    """
    Helper to get portfolio exposure from either portfolio_manager object or fallback to signal.
    
    Args:
        portfolio_manager: PortfolioManager instance or dict
        signal: Trading signal with potential fallback values
        default: Default value if not found anywhere
        
    Returns:
        float: Current portfolio exposure
    """
    if isinstance(portfolio_manager, dict) or not hasattr(portfolio_manager, 'get_total_exposure'):
        return signal.get('current_exposure', default)
    return portfolio_manager.get_total_exposure()


class BaseRiskRule(ABC):
    """
    Abstract base class for all risk validation rules.
    
    Each rule implements a single validation concern (e.g., position size,
    drawdown, correlation) and can be independently tested and configured.
    
    The validate method must return a tuple of (is_valid, reason) where:
    - is_valid: bool indicating if the rule passes
    - reason: str describing why the rule passed or failed
    """
    
    def __init__(self, rule_name: str = None):
        """
        Initialize the base risk rule.
        
        Args:
            rule_name: Optional custom name for the rule
        """
        self.rule_name = rule_name or self.__class__.__name__
        self.enabled = True
    
    @abstractmethod
    def validate(self, signal: Dict, portfolio_manager) -> Tuple[bool, str]:
        """
        Validate a signal against this specific risk rule.
        
        Args:
            signal: Trading signal with entry, stop, size, symbol, etc.
            portfolio_manager: PortfolioManager instance for querying portfolio state
            
        Returns:
            Tuple of (is_valid: bool, reason: str)
        """
        pass
    
    def enable(self):
        """Enable this rule."""
        self.enabled = True
        logger.info(f"Rule '{self.rule_name}' enabled")
    
    def disable(self):
        """Disable this rule."""
        self.enabled = False
        logger.info(f"Rule '{self.rule_name}' disabled")
    
    def __repr__(self):
        status = "enabled" if self.enabled else "disabled"
        return f"<{self.rule_name} ({status})>"


class VolumeAwarePositionSizingRule(BaseRiskRule):
    """Adjusts sizing and R/R distances based on volume bucket context."""

    def __init__(
        self,
        risk_matrix: Dict[str, Dict[str, float]],
        rule_name: str = None,
        risk_config: Optional[Any] = None,
    ):
        super().__init__(rule_name or "VolumeAwarePositionSizingRule")
        self.risk_matrix = self._normalize_matrix(risk_matrix)
        self.risk_config = risk_config

    @staticmethod
    def _normalize_matrix(raw_matrix: Any) -> Dict[str, Dict[str, float]]:
        if not isinstance(raw_matrix, dict):
            return {}
        normalized: Dict[str, Dict[str, float]] = {}
        for raw_bucket, raw_cfg in raw_matrix.items():
            if not isinstance(raw_cfg, dict):
                continue
            bucket = str(raw_bucket or "").strip().upper()
            if not bucket:
                continue
            normalized[bucket] = dict(raw_cfg)
        return normalized

    @staticmethod
    def _resolve_strategy_name(signal: Dict[str, Any]) -> Optional[str]:
        if not isinstance(signal, dict):
            return None
        for key in ("strategy_name", "strategy", "source_strategy"):
            raw = signal.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        meta = signal.get("meta")
        if isinstance(meta, dict):
            for key in ("strategy_name", "strategy", "base_strategy", "source_strategy"):
                raw = meta.get(key)
                if isinstance(raw, str) and raw.strip():
                    return raw.strip()
        return None

    def _resolve_strategy_override_matrix(self, strategy_name: Optional[str]) -> Dict[str, Dict[str, float]]:
        if not strategy_name:
            return {}
        if self.risk_config is None or not hasattr(self.risk_config, "get_strategy_profile"):
            return {}
        try:
            profile = self.risk_config.get_strategy_profile(strategy_name)
        except Exception:
            return {}
        if not isinstance(profile, dict):
            return {}

        raw_override = profile.get("volume_bucket_risk_matrix")
        if raw_override is None:
            raw_override = profile.get("volume_override")
        return self._normalize_matrix(raw_override)

    def _resolve_effective_matrix(self, signal: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, float]], str, Optional[str]]:
        strategy_name = self._resolve_strategy_name(signal)
        override_matrix = self._resolve_strategy_override_matrix(strategy_name)
        if not override_matrix:
            return self.risk_matrix, "global", strategy_name

        effective_matrix = deepcopy(self.risk_matrix)
        for bucket, bucket_cfg in override_matrix.items():
            base_cfg = effective_matrix.get(bucket)
            if isinstance(base_cfg, dict):
                merged_cfg = dict(base_cfg)
                merged_cfg.update(bucket_cfg)
                effective_matrix[bucket] = merged_cfg
            else:
                effective_matrix[bucket] = dict(bucket_cfg)

        strategy_key = str(strategy_name or "").strip().lower()
        source = f"strategy_profile:{strategy_key}" if strategy_key else "strategy_profile"
        return effective_matrix, source, strategy_name

    def validate(self, signal: Dict, portfolio_manager) -> Tuple[bool, str]:
        if not self.enabled:
            return (True, f"{self.rule_name} disabled")

        ctx_source = signal.get('volume_ctx_source')
        if ctx_source and ctx_source != 'analyzer':
            return (True, "Volume context not analyzer-derived; skipping")

        bucket = (signal.get('volume_bucket') or '').upper()
        effective_matrix, matrix_source, strategy_name = self._resolve_effective_matrix(signal)
        if not bucket or not effective_matrix:
            return (True, "No volume bucket provided; skipping")

        cfg = effective_matrix.get(bucket) or effective_matrix.get('NORMAL')
        if not cfg:
            return (True, "No volume matrix configured; skipping")

        try:
            base_position_size = float(signal.get('position_size', 0))
            entry_price = signal.get('entry') or signal.get('entry_price') or 0
            try:
                entry_price = float(entry_price)
            except (TypeError, ValueError):
                entry_price = 0.0

            base_notional = signal.get('notional')
            try:
                if (base_notional is None or base_notional <= 0) and entry_price > 0:
                    base_notional = base_position_size * entry_price
            except Exception:
                base_notional = signal.get('notional')

            scaled_position_size = base_position_size * float(cfg.get('position_size_multiplier', 1.0))
            signal['position_size'] = scaled_position_size
            if 'stop_loss_dist' in signal:
                signal['stop_loss_dist'] = float(signal['stop_loss_dist']) * float(cfg.get('stop_loss_multiplier', 1.0))
            if 'take_profit_dist' in signal:
                signal['take_profit_dist'] = float(signal['take_profit_dist']) * float(cfg.get('take_profit_multiplier', 1.0))

            scaled_notional = None
            if entry_price and entry_price > 0:
                try:
                    scaled_notional = scaled_position_size * entry_price
                except Exception:
                    scaled_notional = None

            caps_snapshot = None
            try:
                caps = signal.get('planner_caps_snapshot') or {}
                max_notional_cap = caps.get('max_notional_cap')
                max_size_pct_notional = caps.get('max_size_pct_notional')
                heat_cap_notional = caps.get('heat_cap_notional')
                caps_snapshot = {
                    'max_notional_cap': max_notional_cap,
                    'max_size_pct_notional': max_size_pct_notional,
                    'heat_cap_notional': heat_cap_notional,
                }
            except Exception:
                caps_snapshot = None

            cap_floor = None
            try:
                candidates = [c for c in [
                    caps_snapshot.get('max_notional_cap') if isinstance(caps_snapshot, dict) else None,
                    caps_snapshot.get('max_size_pct_notional') if isinstance(caps_snapshot, dict) else None,
                    caps_snapshot.get('heat_cap_notional') if isinstance(caps_snapshot, dict) else None,
                ] if c is not None]
                if candidates:
                    cap_floor = min(candidates)
            except Exception:
                cap_floor = None

            would_breach_caps = False
            if scaled_notional is not None and cap_floor is not None:
                try:
                    would_breach_caps = scaled_notional > cap_floor
                except Exception:
                    would_breach_caps = False

            now_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            run_id = get_current_run_id()

            logger.info(
                "volume_bucket_risk %s",
                {
                    'event': 'volume_bucket_risk',
                    'timestamp': now_ts,
                    'run_id': run_id,
                    'symbol': signal.get('symbol'),
                    'timeframe': signal.get('timeframe') or signal.get('tf'),
                    'strategy_name': strategy_name,
                    'volume_matrix_source': matrix_source,
                    'volume_bucket': bucket,
                    'position_size_multiplier': cfg.get('position_size_multiplier', 1.0),
                    'stop_loss_multiplier': cfg.get('stop_loss_multiplier', 1.0),
                    'take_profit_multiplier': cfg.get('take_profit_multiplier', 1.0),
                    'base_position_size': base_position_size,
                    'base_notional': base_notional,
                    'scaled_position_size': scaled_position_size,
                    'scaled_notional': scaled_notional,
                    'caps_snapshot': caps_snapshot,
                    'would_breach_caps_after_volume': would_breach_caps,
                },
            )
            return (True, f"Applied volume bucket {bucket}")
        except Exception as exc:
            logger.error(f"[{self.rule_name}] Error applying matrix for {bucket}: {exc}")
            return (False, f"Volume sizing error: {exc}")


class CapitalLimitRule(BaseRiskRule):
    """
    Validates that total portfolio exposure does not exceed available capital.
    
    UPDATED: Now supports margin-based validation for futures.
    - Spot (leverage=1): Compare notional to available capital
    - Futures (leverage>1): Compare required margin to available capital
    
    This is the fundamental capital preservation rule - ensures we cannot
    allocate more capital than we have available.
    """
    
    def __init__(self, rule_name: str = None):
        super().__init__(rule_name or "CapitalLimitRule")
    
    def validate(self, signal: Dict, portfolio_manager) -> Tuple[bool, str]:
        """
        Validate capital requirements based on leverage.
        
        Args:
            signal: Trading signal with position_size, entry price, and leverage
            portfolio_manager: PortfolioManager instance or dict
            
        Returns:
            (is_valid, reason) tuple
        """
        if not self.enabled:
            return (True, f"{self.rule_name} disabled")
        
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            notional = signal.get('notional', 0)
            leverage = signal.get('leverage', 1)
            
            # Fallback to old calculation if notional is not provided
            if notional <= 0:
                position_size = signal.get('position_size', 0)
                entry_price = signal.get('entry', 0)
                notional = position_size * entry_price
            
            # Get portfolio state using helper functions
            portfolio_value = _get_portfolio_value(portfolio_manager, signal)
            current_exposure = _get_portfolio_exposure(portfolio_manager, signal)
            available = portfolio_value - current_exposure
            affordable_notional = compute_max_affordable_notional(available, leverage)
            logger.debug(
                f"[CapitalLimitRule] portfolio_value={portfolio_value}, current_exposure={current_exposure}, available={available}, affordable={affordable_notional}, leverage={leverage}"
            )
            
            if notional <= 0:
                return (False, f"Invalid notional value: {notional}")
            
            if notional > affordable_notional:
                logger.warning(
                    f"🚫 [CapitalLimitRule] REJECTED {symbol}: ${notional:.2f} > affordable ${affordable_notional:.2f} (avail={available:.2f}, lev={leverage})"
                )
                return (False, f"Position ${notional:.2f} exceeds affordable notional ${affordable_notional:.2f}")
            else:
                logger.info(
                    f"✅ [CapitalLimitRule] PASSED {symbol}: ${notional:.2f} <= affordable ${affordable_notional:.2f} (avail={available:.2f}, lev={leverage})"
                )
                return (True, "Capital check passed")
                
        except Exception as e:
            logger.error(f"[CapitalLimitRule] Error: {e}", exc_info=True)
            return (False, f"Validation error: {str(e)}")


class PositionSizeRule(BaseRiskRule):
    """
    Validates that position size does not exceed maximum allowed percentage of portfolio.
    
    Prevents over-concentration in a single position.
    """
    
    def __init__(self, max_position_size: float = 0.10, rule_name: str = None):
        """
        Initialize position size rule.
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio (default 10%)
            rule_name: Optional custom rule name
        """
        super().__init__(rule_name or "PositionSizeRule")
        self.max_position_size = max_position_size
    
    def validate(self, signal: Dict, portfolio_manager) -> Tuple[bool, str]:
        """
        Validate position size is within limits.
        
        Args:
            signal: Trading signal
            portfolio_manager: PortfolioManager instance or dict
            
        Returns:
            (is_valid, reason) tuple
        """
        if not self.enabled:
            return (True, f"{self.rule_name} disabled")
        
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            position_size = signal.get('position_size', 0)
            entry_price = signal.get('entry', 0)

            # Prefer notional provided by planner/size pipeline to avoid drift when prices change downstream
            position_value = signal.get('notional')
            if position_value is None or position_value <= 0:
                position_value = position_size * entry_price

            raw_leverage = signal.get('leverage', 1)
            try:
                leverage = float(raw_leverage or 1.0)
            except (TypeError, ValueError):
                leverage = 1.0
            if leverage <= 0:
                leverage = 1.0

            required_margin = position_value / leverage if position_value > 0 else 0.0
            
            # Get portfolio value using helper function
            portfolio_value = _get_portfolio_value(portfolio_manager, signal)
                
            # Leverage-aware: interpret `max_position_size` as max margin fraction of equity.
            max_position_value = portfolio_value * self.max_position_size
            
            position_size_pct = position_value / portfolio_value if portfolio_value > 0 else 0

            # Expose the value seen by the rule for downstream anomaly logging
            signal['__position_size_rule_position_value'] = position_value
            signal['__position_size_rule_required_margin'] = required_margin
            
            logger.debug(
                f"[{self.rule_name}] {symbol}: notional=${position_value:.2f} lev={leverage:.2f} "
                f"margin=${required_margin:.2f} ({position_size_pct:.1%}) vs max_margin=${max_position_value:.2f} ({self.max_position_size:.1%})"
            )
            
            if required_margin > max_position_value:
                logger.warning(
                    f"🚫 [{self.rule_name}] REJECTED: {symbol} required margin ${required_margin:.2f} exceeds max ${max_position_value:.2f}"
                )
                return (False, f"Required margin ${required_margin:.2f} exceeds max ${max_position_value:.2f}")
            
            logger.debug(f"✅ [{self.rule_name}] PASSED: {symbol}")
            return (True, f"Position size within limits")
            
        except Exception as e:
            logger.error(f"[{self.rule_name}] Validation error: {e}")
            return (False, f"Validation error: {str(e)}")


class PortfolioHeatRule(BaseRiskRule):
    """
    Validates that total portfolio risk exposure (heat) stays within acceptable limits.
    
    Portfolio heat is the sum of all risk amounts across open positions.
    This prevents excessive overall risk exposure.
    """
    
    def __init__(self, max_portfolio_heat: float = 0.10, max_portfolio_risk: float = 0.06, rule_name: str = None):
        """
        Initialize portfolio heat rule.
        
        Args:
            max_portfolio_heat: Maximum total portfolio heat (default 10%)
            max_portfolio_risk: Maximum risk per single trade / heat cap fraction (default 6%)
            rule_name: Optional custom rule name
        """
        super().__init__(rule_name or "PortfolioHeatRule")
        self.max_portfolio_heat = max_portfolio_heat
        self.max_portfolio_risk = max_portfolio_risk
    
    def validate(self, signal: Dict, portfolio_manager) -> Tuple[bool, str]:
        """
        Validate portfolio heat is within limits.
        
        Args:
            signal: Trading signal
            portfolio_manager: PortfolioManager instance
            
        Returns:
            (is_valid, reason) tuple
        """
        if not self.enabled:
            return (True, f"{self.rule_name} disabled")
        
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            position_size = signal.get('position_size', 0)
            entry_price = signal.get('entry', 0)
            stop_loss = signal.get('stop', 0)

            if not stop_loss:
                stop_loss = self._calculate_stop_loss(signal, entry_price)

            open_risk_usd, portfolio_value = compute_portfolio_open_risk_usd(portfolio_manager, signal)

            risk_amount = abs(entry_price - stop_loss) * position_size
            max_risk = portfolio_value * self.max_portfolio_risk
            risk_pct = risk_amount / portfolio_value if portfolio_value > 0 else 0

            if risk_amount > max_risk:
                logger.warning(f"🚫 [{self.rule_name}] REJECTED: {symbol} risk ${risk_amount:.2f} ({risk_pct:.2%}) exceeds max ${max_risk:.2f} ({self.max_portfolio_risk:.2%})")
                return (False, f"Risk amount ${risk_amount:.2f} exceeds max ${max_risk:.2f}")

            total_risk = open_risk_usd + risk_amount
            portfolio_heat = total_risk / portfolio_value if portfolio_value > 0 else 0

            logger.debug(f"[{self.rule_name}] {symbol}: risk ${risk_amount:.2f} ({risk_pct:.2%}), total heat {portfolio_heat:.2%}")

            if portfolio_heat > self.max_portfolio_heat:
                logger.warning(f"🚫 [{self.rule_name}] REJECTED: {symbol} portfolio heat {portfolio_heat:.2%} would exceed max {self.max_portfolio_heat:.2%}")
                return (False, f"Portfolio heat {portfolio_heat:.2%} would exceed {self.max_portfolio_heat:.2%}")

            logger.debug(f"✅ [{self.rule_name}] PASSED: {symbol}")
            return (True, f"Portfolio heat within limits")
            
        except Exception as e:
            logger.error(f"[{self.rule_name}] Validation error: {e}")
            return (False, f"Validation error: {str(e)}")
    
    def _calculate_stop_loss(self, signal: Dict, entry_price: float) -> float:
        """Calculate stop loss from signal parameters."""
        side = signal.get('side', 'buy')
        
        # Try ATR-based stop
        if signal.get('sl_atr_mult') and signal.get('atr'):
            atr = signal['atr']
            sl_mult = signal['sl_atr_mult']
            
            if side in ['buy', 'long']:
                return entry_price - (atr * sl_mult)
            else:
                return entry_price + (atr * sl_mult)
        
        # Try percentage-based stop
        if signal.get('sl_pct'):
            sl_pct = signal['sl_pct']
            
            if side in ['buy', 'long']:
                return entry_price * (1 - sl_pct)
            else:
                return entry_price * (1 + sl_pct)
        
        return 0


class MaxDrawdownRule(BaseRiskRule):
    """
    Validates that current portfolio drawdown has not exceeded maximum allowed.
    
    When drawdown limit is breached, no new positions should be opened until
    portfolio recovers.
    """
    
    def __init__(self, max_drawdown: float = 0.15, rule_name: str = None):
        """
        Initialize max drawdown rule.
        
        Args:
            max_drawdown: Maximum allowed drawdown (default 15%)
            rule_name: Optional custom rule name
        """
        super().__init__(rule_name or "MaxDrawdownRule")
        self.max_drawdown = max_drawdown
    
    def validate(self, signal: Dict, portfolio_manager) -> Tuple[bool, str]:
        """
        Validate drawdown is within limits.
        
        Args:
            signal: Trading signal
            portfolio_manager: PortfolioManager instance
            
        Returns:
            (is_valid, reason) tuple
        """
        if not self.enabled:
            return (True, f"{self.rule_name} disabled")
        
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            
            # Get current drawdown (with fallback for test compatibility)
            if isinstance(portfolio_manager, dict) or not hasattr(portfolio_manager, 'get_current_drawdown'):
                current_drawdown = signal.get('current_drawdown', 0)
            else:
                current_drawdown = portfolio_manager.get_current_drawdown()
            
            logger.debug(f"[{self.rule_name}] {symbol}: current drawdown {current_drawdown:.2%} vs max {self.max_drawdown:.2%}")
            
            if current_drawdown > self.max_drawdown:
                logger.warning(f"🚫 [{self.rule_name}] REJECTED: {symbol} current drawdown {current_drawdown:.2%} exceeds max {self.max_drawdown:.2%}")
                return (False, f"Current drawdown {current_drawdown:.2%} exceeds max {self.max_drawdown:.2%}")
            
            logger.debug(f"✅ [{self.rule_name}] PASSED: {symbol}")
            return (True, f"Drawdown within limits")
            
        except Exception as e:
            logger.error(f"[{self.rule_name}] Validation error: {e}")
            return (False, f"Validation error: {str(e)}")


class RiskRewardRatioRule(BaseRiskRule):
    """
    Intelligent Risk/Reward ratio validation with ML/RL awareness.
    
    This rule implements a dynamic R/R threshold that adjusts based on:
    - ML confidence in regime prediction
    - RL agent agreement with signal direction
    - Market regime clarity
    - Strategy's minimum R/R requirement
    
    High confidence signals get lower R/R requirements (0.8-1.2)
    Low confidence signals get higher R/R requirements (1.5-2.0)
    """
    
    def __init__(self, config=None, min_risk_reward: float = None, rule_name: str = None):
        """
        Initialize the risk/reward ratio rule.
        
        Args:
            config: RiskConfiguration instance with dynamic R/R settings (preferred)
            min_risk_reward: Legacy parameter for backward compatibility (static threshold)
            rule_name: Optional custom rule name.
        """
        super().__init__(rule_name or "RiskRewardRatioRule")
        self.risk_config = config
        self.validation_history = []  # For monitoring
        
        # Backward compatibility: If min_risk_reward provided but no config,
        # create a simple config with dynamic disabled
        if min_risk_reward is not None and config is None:
            # Import here to avoid circular dependency during module initialization
            # This is only used for backward compatibility with legacy tests
            from config.risk_config import RiskConfiguration
            self.risk_config = RiskConfiguration({
                'equity_usd': 10000,
                'rr_dynamic': {
                    'enabled': False,
                    'base_target_rr': min_risk_reward
                }
            })
    
    def validate(self, signal: Dict, portfolio_manager) -> Tuple[bool, str]:
        """
        Validate R/R with dynamic intelligence-based adjustment.
        
        Args:
            signal: Trading signal with ML/RL enrichment
            portfolio_manager: PortfolioManager instance
            
        Returns:
            (is_valid, reason) tuple
        """
        if not self.enabled:
            return (True, f"{self.rule_name} disabled")
        
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            
            # Extract price levels
            entry = float(signal.get('entry', 0) or signal.get('price', 0))
            stop = float(signal.get('stop', 0) or signal.get('stop_loss', 0))
            target = float(signal.get('target', 0) or signal.get('take_profit', 0))
            
            # Backward compatibility: Calculate stop from ATR if not provided
            # This fallback supports legacy signal formats that only provide ATR parameters
            # instead of pre-calculated stop prices. Modern signals should include explicit stop prices.
            if stop == 0 and entry > 0:
                stop = self._calculate_stop_from_signal(signal, entry)
            
            # Validate inputs
            if entry <= 0 or stop <= 0 or target <= 0:
                return (False, "Invalid price levels for R/R calculation")
            
            # Calculate actual R/R
            risk = abs(entry - stop)
            reward = abs(target - entry)
            
            if risk == 0:
                return (False, "Zero risk detected (stop equals entry)")
            
            calculated_rr = reward / risk
            risk_pct = (risk / entry) * 100
            reward_pct = (reward / entry) * 100
            
            # Get dynamic target R/R
            target_rr = self._calculate_dynamic_target(signal)
            signal['dynamic_rr_target'] = target_rr
            signal.setdefault('rr_ratio', calculated_rr)
            signal['calculated_rr_ratio'] = calculated_rr
            rr_gate = evaluate_rr_gate(
                calculated_rr,
                rr_required=target_rr,
                rr_required_source="dynamic_rr_target",
                rr_floor=1.0,
                action_on_fail="reject",
            )
            prefill_reason_code = build_prefill_rr_reason_code(rr_gate.get("reason_code"))
            prefill_meta = {
                "rr_actual": rr_gate.get("rr_actual"),
                "rr_required": rr_gate.get("rr_required"),
                "rr_required_source": rr_gate.get("rr_required_source"),
                "rr_floor": rr_gate.get("rr_floor"),
                "action": rr_gate.get("action"),
                "reason_code": rr_gate.get("reason_code"),
                "prefill_reason_code": prefill_reason_code,
            }
            signal["prefill_rr_meta"] = prefill_meta
            signal["prefill_rr_reason_code"] = prefill_reason_code
            signal["prefill_rr_reason"] = rr_gate.get("reason_code")
            signal["prefill_rr_actual"] = rr_gate.get("rr_actual")
            signal["prefill_rr_required"] = rr_gate.get("rr_required")
            signal["prefill_rr_required_source"] = rr_gate.get("rr_required_source")
            signal["prefill_rr_floor"] = rr_gate.get("rr_floor")
            
            # Enhanced diagnostic logging
            logger.info(f"📊 [R/R Analysis] {symbol}:")
            logger.info(f"   Prices: Entry=${entry:.2f}, Stop=${stop:.2f} (-{risk_pct:.1f}%), Target=${target:.2f} (+{reward_pct:.1f}%)")
            logger.info(f"   R/R: Actual={calculated_rr:.2f}, Required={target_rr:.2f}")
            
            # Show intelligence metrics if available
            ml_conf = signal.get('ml_confidence', 'N/A')
            rl_agree = signal.get('rl_is_agree', 'N/A')
            rl_prob = signal.get('rl_action_prob', 'N/A')
            regime = signal.get('regime_name', 'N/A')
            regime_conf = signal.get('regime_confidence', 'N/A')
            vol_str = signal.get('volume_strength', 'N/A')
            mom_str = signal.get('momentum_strength', 'N/A')
            
            logger.info(f"   Intelligence: ML={ml_conf if isinstance(ml_conf, str) else f'{ml_conf:.2f}'}, "
                       f"RL_agree={rl_agree}, "
                       f"RL_prob={rl_prob if isinstance(rl_prob, str) else f'{rl_prob:.2f}'}, "
                       f"Regime={regime} "
                       f"({regime_conf if isinstance(regime_conf, str) else f'{regime_conf:.2f}'}), "
                       f"Vol={vol_str if isinstance(vol_str, str) else f'{vol_str:.2f}'}, "
                       f"Mom={mom_str if isinstance(mom_str, str) else f'{mom_str:.2f}'}")
            
            # Make decision
            rr_reason_code = rr_gate.get("reason_code")
            if rr_reason_code:
                if rr_reason_code == RR_REASON_BELOW_1:
                    reason = (
                        f"Risk/reward ratio {calculated_rr:.2f} is below hard floor "
                        f"{float(rr_gate.get('rr_floor') or 1.0):.2f} "
                        f"(dynamic target {target_rr:.2f}; Risk: {risk_pct:.1f}%, Reward: {reward_pct:.1f}%)"
                    )
                elif rr_reason_code == RR_REASON_BELOW_REQUIRED:
                    reason = (
                        f"Risk/reward ratio {calculated_rr:.2f} is below dynamic target {target_rr:.2f} "
                        f"(Risk: {risk_pct:.1f}%, Reward: {reward_pct:.1f}%)"
                    )
                else:
                    reason = (
                        f"Risk/reward ratio {calculated_rr:.2f} failed pre-fill RR gate "
                        f"(Risk: {risk_pct:.1f}%, Reward: {reward_pct:.1f}%)"
                    )
                logger.warning(
                    "🚫 [RiskRewardRatioRule] REJECTED %s: %s | reason_code=%s prefill_reason_code=%s "
                    "rr_actual=%.3f rr_required=%.3f rr_floor=%.3f",
                    symbol,
                    reason,
                    rr_reason_code,
                    prefill_reason_code or "na",
                    float(rr_gate.get("rr_actual") or 0.0),
                    float(rr_gate.get("rr_required") or 0.0),
                    float(rr_gate.get("rr_floor") or 0.0),
                )
                return (False, reason)
            else:
                logger.info(f"✅ [RiskRewardRatioRule] PASSED {symbol}: R/R {calculated_rr:.2f} >= {target_rr:.2f}")
                return (True, f"R/R acceptable ({calculated_rr:.2f} >= {target_rr:.2f})")
                
        except Exception as e:
            logger.error(f"[RiskRewardRatioRule] Error: {e}", exc_info=True)
            return (False, f"R/R validation error: {str(e)}")
    
    def _calculate_dynamic_target(self, signal: Dict) -> float:
        """
        Calculate dynamic R/R target based on ML/RL confidence.
        
        Formula:
        Dynamic R/R = BASE_RR - (ML_WEIGHT × ML_CONF) - (RL_WEIGHT × RL_AGREEMENT) 
                     + (REGIME_WEIGHT × UNCERTAINTY) × REGIME_MULTIPLIER
        Final R/R = CLAMP(Dynamic R/R, LOWER_BOUND, UPPER_BOUND)
        
        Args:
            signal: Trading signal with ML/RL metrics
            
        Returns:
            Dynamic R/R target threshold
        """
        # Check if dynamic system is enabled
        if not self.risk_config or not hasattr(self.risk_config, 'rr_dynamic'):
            logger.warning("[Dynamic R/R] Config not found, using static default 1.5")
            return 1.5
        
        config = self._get_rr_config_for_signal(signal)
        
        if not config.get('enabled', False):
            # Use static target if dynamic is disabled
            static_target = getattr(self.risk_config, 'min_risk_reward_ratio', 1.5)
            logger.debug(f"[Dynamic R/R] Disabled, using static target: {static_target}")
            return static_target
        
        # Get configuration parameters
        base_rr = config['base_target_rr']
        lower_bound = config['lower_bound_rr']
        upper_bound = config['upper_bound_rr']
        weights = config['weights']
        fallback = config['fallback']
        regime_mults = config['regime_multipliers']

        symbol = signal.get('symbol', 'UNKNOWN')
        model_version = str(
            signal.get('rr_model_version')
            or config.get('model_version')
            or 'v1'
        ).strip().lower()
        if model_version not in ('v1', 'v2'):
            model_version = 'v1'

        def _safe_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _clamp01(value: float) -> float:
            if value < 0.0:
                return 0.0
            if value > 1.0:
                return 1.0
            return value
        
        # Extract intelligence metrics with fallbacks
        ml_conf = _safe_float(signal.get('ml_confidence'), _safe_float(fallback.get('missing_ml_default', 0.5), 0.5))
        rl_agree = 1.0 if signal.get('rl_is_agree', False) else 0.0
        rl_prob = _safe_float(signal.get('rl_action_prob'), _safe_float(fallback.get('missing_rl_default', 0.5), 0.5))
        regime_conf = _safe_float(signal.get('regime_confidence'), _safe_float(fallback.get('missing_regime_default', 0.3), 0.3))
        regime_name_raw = signal.get('regime_name', 'neutral')
        if isinstance(regime_name_raw, dict):
            regime_name = str(
                regime_name_raw.get('predicted_regime')
                or regime_name_raw.get('regime')
                or 'neutral'
            ).lower()
        else:
            regime_name = str(regime_name_raw or 'neutral').lower()
        
        # Get regime_weight (soft-weighting). When missing, derive from regime_conf
        # to avoid hard-reject paradox (low confidence being treated as full confidence).
        regime_weight_raw = signal.get('regime_weight')
        if regime_weight_raw is not None:
            regime_weight = _clamp01(_safe_float(regime_weight_raw, 1.0))
        else:
            soft_cfg = getattr(self.risk_config, 'regime_soft_weight', {}) if self.risk_config else {}
            hard_reject = _safe_float(
                (soft_cfg or {}).get('min_confidence_hard_reject'),
                0.30,
            )
            full_weight = _safe_float(
                (soft_cfg or {}).get('min_confidence_full_weight'),
                0.60,
            )
            if full_weight <= 0:
                full_weight = 0.60

            if regime_conf < hard_reject:
                regime_weight = 0.0
            elif regime_conf >= full_weight:
                regime_weight = 1.0
            else:
                regime_weight = _clamp01(regime_conf / full_weight)

            logger.debug(
                "[Dynamic R/R] regime_weight fallback derived from regime_conf=%.2f -> %.2f "
                "(hard_reject=%.2f full_weight=%.2f)",
                regime_conf,
                regime_weight,
                hard_reject,
                full_weight,
            )
        
        # Calculate relaxation (reduces required R/R for high confidence)
        base_relaxation = (
            weights['ml_confidence'] * ml_conf
            + weights['rl_agreement'] * rl_agree * rl_prob
        )

        vol_raw = signal.get('volume_strength')
        mom_raw = signal.get('momentum_strength')
        neutral_strength = _safe_float(fallback.get('missing_ml_default', 0.5), 0.5)
        vol_strength = _clamp01(_safe_float(vol_raw, neutral_strength))
        mom_strength = _clamp01(_safe_float(mom_raw, neutral_strength))

        vol_weight = float(weights.get('volume_strength', 0.0) or 0.0)
        mom_weight = float(weights.get('momentum_strength', 0.0) or 0.0)
        vol_contrib = vol_weight * vol_strength
        mom_contrib = mom_weight * mom_strength

        relaxation = base_relaxation
        if model_version == 'v2':
            relaxation += (vol_contrib + mom_contrib)
        
        # Calculate tightening (increases required R/R for uncertainty)
        # Apply regime_weight to tightening effect
        tightening = weights['regime_clarity'] * (1.0 - regime_conf) * regime_weight
        
        # Apply regime multiplier with soft-weighting
        regime_mult = regime_mults.get(regime_name, 1.0)
        # Soft-weighted regime adjustment: interpolate between 1.0 (no effect) and regime_mult
        regime_adjustment = 1.0 + (regime_mult - 1.0) * regime_weight
        
        # Calculate dynamic target (pre-PPO)
        dynamic_target_pre_ppo = (base_rr - relaxation + tightening) * regime_adjustment

        # Respect strategy's minimum (support both keys for interoperability):
        # - `strategy_min_rr` (common)
        # - `min_rr_ratio` (mean_reversion emits this)
        strategy_floor_raw = signal.get('strategy_min_rr')
        if strategy_floor_raw is None:
            strategy_floor_raw = signal.get('min_rr_ratio')
        strategy_floor = 0.5
        if strategy_floor_raw is not None:
            try:
                strategy_floor = float(strategy_floor_raw)
            except Exception:
                strategy_floor = 0.5

        # Apply PPO multiplier BEFORE bounds so upper_bound_rr is a true hard cap.
        ppo_rr_multiplier = float(signal.get('ppo_rr_multiplier', 1.0))
        dynamic_target = float(dynamic_target_pre_ppo) * max(0.1, ppo_rr_multiplier)

        # Apply strategy floor after multipliers (PPO down-mults should not undercut strategy requirements).
        dynamic_target = max(dynamic_target, strategy_floor)

        effective_upper_bound = upper_bound
        if strategy_floor > upper_bound:
            logger.warning(
                "[Dynamic R/R] Misconfiguration detected: Strategy floor %.2f exceeds Risk Cap (upper_bound_rr=%.2f) "
                "for %s. Enforcing floor as effective cap.",
                strategy_floor,
                upper_bound,
                symbol,
            )
            effective_upper_bound = strategy_floor

        # Apply bounds (hard caps) after all multipliers
        final_target = max(lower_bound, min(dynamic_target, effective_upper_bound))
        
        # Detailed logging
        logger.info(
            f"📊 [Dynamic R/R Calc] Base={base_rr:.2f} - Relax={relaxation:.2f} + Tight={tightening:.2f} "
            f"× Regime({regime_name}, mult={regime_mult:.1f}, weight={regime_weight:.2f})={regime_adjustment:.2f} "
            f"= {dynamic_target_pre_ppo:.2f} × PPO({ppo_rr_multiplier:.2f}) "
            f"= {dynamic_target:.2f} → Final={final_target:.2f}"
        )

        if model_version == 'v2':
            logger.info(
                "📊 [Dynamic R/R v2 Factors] %s: volume_strength raw=%s norm=%.2f weight=%.3f contrib=%.4f | "
                "momentum_strength raw=%s norm=%.2f weight=%.3f contrib=%.4f",
                symbol,
                vol_raw,
                vol_strength,
                vol_weight,
                vol_contrib,
                mom_raw,
                mom_strength,
                mom_weight,
                mom_contrib,
            )
        
        return final_target

    def _get_rr_config_for_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve dynamic R/R config for this signal using strategy risk profiles."""
        base_config = deepcopy(self.risk_config.rr_dynamic)
        strategy_name = signal.get('strategy_name') or signal.get('strategy')

        resolver = getattr(self.risk_config, 'get_strategy_profile', None)
        if callable(resolver):
            try:
                profile = resolver(strategy_name)
                rr_cfg = profile.get('rr_dynamic') if isinstance(profile, dict) else None
                if isinstance(rr_cfg, dict):
                    return deepcopy(rr_cfg)
            except Exception:
                # Fall back to legacy path below
                pass

        # Backward-compatible fallback: legacy rr_dynamic.strategy_overrides
        overrides = getattr(self.risk_config, 'rr_dynamic_strategy_overrides', {}) or {}
        if not overrides or not strategy_name:
            return base_config
        strategy_key = str(strategy_name).lower()
        override = overrides.get(strategy_key) or overrides.get(strategy_name)
        if not override:
            return base_config
        return self._deep_merge_rr_config(base_config, override)

    @staticmethod
    def _deep_merge_rr_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge override values into the base dynamic R/R configuration."""
        merged = deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = RiskRewardRatioRule._deep_merge_rr_config(merged[key], value)
            else:
                merged[key] = deepcopy(value) if isinstance(value, dict) else value
        return merged
    
    def _calculate_stop_from_signal(self, signal: Dict, entry_price: float) -> float:
        """
        Calculate stop loss from signal parameters (backward compatibility).
        
        Args:
            signal: Trading signal
            entry_price: Entry price
            
        Returns:
            Stop loss price
        """
        side = signal.get('side', 'buy')
        
        # Try ATR-based stop
        if signal.get('sl_atr_mult') and signal.get('atr'):
            atr = float(signal['atr'])
            sl_mult = float(signal['sl_atr_mult'])
            
            if side in ['buy', 'long']:
                return entry_price - (atr * sl_mult)
            else:
                return entry_price + (atr * sl_mult)
        
        # Try percentage-based stop
        if signal.get('sl_pct'):
            sl_pct = float(signal['sl_pct'])
            
            if side in ['buy', 'long']:
                return entry_price * (1 - sl_pct)
            else:
                return entry_price * (1 + sl_pct)
        
        return 0
    
    def _calculate_stop_loss(self, signal: Dict, entry_price: float) -> float:
        """Calculate stop loss from signal parameters."""
        side = signal.get('side', 'buy')
        
        # Try ATR-based stop
        if signal.get('sl_atr_mult') and signal.get('atr'):
            atr = signal['atr']
            sl_mult = signal['sl_atr_mult']
            
            if side in ['buy', 'long']:
                return entry_price - (atr * sl_mult)
            else:
                return entry_price + (atr * sl_mult)
        
        # Try percentage-based stop
        if signal.get('sl_pct'):
            sl_pct = signal['sl_pct']
            
            if side in ['buy', 'long']:
                return entry_price * (1 - sl_pct)
            else:
                return entry_price * (1 + sl_pct)
        
        return 0


class StrategyPerformanceRule(BaseRiskRule):
    """
    Validates strategy has acceptable recent performance before allowing new trades.
    
    Optional rule that can reduce position sizes or block trades from poorly
    performing strategies.
    """
    
    def __init__(self, min_win_rate: float = 0.35, performance_monitor=None, rule_name: str = None):
        """
        Initialize strategy performance rule.
        
        Args:
            min_win_rate: Minimum acceptable win rate (default 35%)
            performance_monitor: Optional performance monitor instance
            rule_name: Optional custom rule name
        """
        super().__init__(rule_name or "StrategyPerformanceRule")
        self.min_win_rate = min_win_rate
        self.performance_monitor = performance_monitor
    
    def validate(self, signal: Dict, portfolio_manager) -> Tuple[bool, str]:
        """
        Validate strategy performance.
        
        Args:
            signal: Trading signal
            portfolio_manager: PortfolioManager instance
            
        Returns:
            (is_valid, reason) tuple
        """
        if not self.enabled:
            return (True, f"{self.rule_name} disabled")
        
        # Skip if no performance monitor available
        if not self.performance_monitor:
            return (True, f"{self.rule_name}: no performance monitor")
        
        try:
            strategy_name = signal.get('strategy', 'unknown')
            symbol = signal.get('symbol', 'UNKNOWN')
            
            summary = self.performance_monitor.get_strategy_summary(strategy_name)
            metrics = summary.get('metrics', {})
            
            win_rate = metrics.get('win_rate', 0.5)
            
            logger.debug(f"[{self.rule_name}] {symbol} (strategy={strategy_name}): win rate {win_rate:.2%} vs min {self.min_win_rate:.2%}")
            
            if win_rate < self.min_win_rate:
                logger.warning(f"🚫 [{self.rule_name}] REJECTED: {symbol} strategy '{strategy_name}' win rate {win_rate:.2%} below minimum {self.min_win_rate:.2%}")
                return (False, f"Strategy win rate {win_rate:.2%} too low")
            
            logger.debug(f"✅ [{self.rule_name}] PASSED: {symbol}")
            return (True, f"Strategy performance acceptable")
            
        except Exception as e:
            logger.error(f"[{self.rule_name}] Validation error: {e}")
            return (False, f"Validation error: {str(e)}")


class DailyTradeLimitRule(BaseRiskRule):
    """
    Validates that daily trade limit has not been reached.
    
    This rule prevents over-trading by enforcing a maximum number of
    trades per day. The counter automatically resets at the start of
    each new trading day (UTC).
    
    Critical for:
    - Risk management and capital preservation
    - Preventing emotional/impulsive over-trading
    - Enforcing disciplined trading approach
    """
    
    def __init__(
        self,
        max_daily_trades: int,
        *,
        dynamic_config: Optional[Dict[str, Any]] = None,
        rule_name: str = None,
    ):
        """
        Initialize daily trade limit rule.
        
        Args:
            max_daily_trades: Maximum number of trades allowed per day
            dynamic_config: Optional `risk.daily_trade_limit` config dict for dynamic cap policies
            rule_name: Optional custom rule name
        """
        super().__init__(rule_name or "DailyTradeLimitRule")
        self.max_daily_trades = max_daily_trades
        self.dynamic_config = dynamic_config if isinstance(dynamic_config, dict) else {}
        
        if max_daily_trades <= 0:
            logger.warning(f"⚠️ DailyTradeLimitRule initialized with invalid limit: {max_daily_trades}. "
                         f"This rule will effectively block all trades.")

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            if value is None or isinstance(value, bool):
                return default
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None or isinstance(value, bool):
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _normalize_pct_threshold(value: Optional[float]) -> Optional[float]:
        if value is None or not math.isfinite(value):
            return None
        # Accept percent-style inputs (e.g., 5 -> 5%)
        if value > 1:
            return value / 100.0
        return value

    def _get_dynamic_allowed_max(self, portfolio_manager) -> Tuple[int, Optional[float], str]:
        """Return (allowed_max_trades, pnl_usd_used, policy_label)."""
        base = int(self.max_daily_trades)
        cfg = self.dynamic_config or {}
        profit_cfg = cfg.get("profit_unlock") if isinstance(cfg.get("profit_unlock"), dict) else {}

        enabled = bool(profit_cfg.get("enabled", False))
        if not enabled:
            return base, None, "static"

        pnl_source = (profit_cfg.get("pnl_source") or "daily").strip().lower()
        pnl_usd = None
        if pnl_source == "since_start":
            getter = getattr(portfolio_manager, "get_pnl_since_start_usd", None)
            if callable(getter):
                try:
                    pnl_usd = float(getter())
                except Exception:
                    pnl_usd = None
        else:
            getter = getattr(portfolio_manager, "get_todays_pnl_usd", None)
            if callable(getter):
                try:
                    pnl_usd = float(getter())
                except Exception:
                    pnl_usd = None

        min_pnl_usd = self._safe_float(profit_cfg.get("min_pnl_usd"), default=0.0) or 0.0
        if pnl_usd is None or not math.isfinite(pnl_usd) or pnl_usd < min_pnl_usd:
            return base, pnl_usd, "profit_unlock:not_eligible"

        # Optional: require pnl to exceed a fraction of start-of-day equity (safer unlock).
        min_pnl_pct = self._normalize_pct_threshold(self._safe_float(profit_cfg.get("min_pnl_pct"), default=None))
        if min_pnl_pct is not None and min_pnl_pct > 0:
            start_equity = None
            getter = getattr(portfolio_manager, "get_todays_start_equity_usd", None)
            if callable(getter):
                try:
                    start_equity = float(getter())
                except Exception:
                    start_equity = None
            if start_equity is None or not math.isfinite(start_equity) or start_equity <= 0:
                return base, pnl_usd, "profit_unlock:not_eligible_missing_equity"
            pnl_pct = pnl_usd / start_equity
            if pnl_pct < float(min_pnl_pct):
                return base, pnl_usd, "profit_unlock:not_eligible_min_pnl_pct"

        # Optional: do not unlock if drawdown is elevated (tighten again intraday).
        max_drawdown_pct = self._normalize_pct_threshold(self._safe_float(profit_cfg.get("max_drawdown_pct"), default=None))
        if max_drawdown_pct is not None and max_drawdown_pct >= 0:
            dd_source = (profit_cfg.get("drawdown_source") or "daily").strip().lower()
            drawdown = None
            if dd_source == "overall":
                getter = getattr(portfolio_manager, "get_current_drawdown", None)
                if callable(getter):
                    try:
                        drawdown = float(getter())
                    except Exception:
                        drawdown = None
            else:
                getter = getattr(portfolio_manager, "get_todays_drawdown_pct", None)
                if callable(getter):
                    try:
                        drawdown = float(getter())
                    except Exception:
                        drawdown = None

            if drawdown is None or not math.isfinite(drawdown):
                return base, pnl_usd, "profit_unlock:not_eligible_missing_drawdown"
            if drawdown >= float(max_drawdown_pct):
                return base, pnl_usd, "profit_unlock:not_eligible_drawdown"

        # If configured, scale extra trades with profits (step function).
        pnl_step_usd = self._safe_float(profit_cfg.get("pnl_step_usd"), default=None)
        extra_trades_per_step = self._safe_int(profit_cfg.get("extra_trades_per_step"), default=1)
        if pnl_step_usd is not None and pnl_step_usd > 0 and extra_trades_per_step > 0:
            steps = int(math.floor((pnl_usd - min_pnl_usd) / float(pnl_step_usd)))
            extra_trades = max(0, steps * extra_trades_per_step)
        else:
            extra_trades = max(0, self._safe_int(profit_cfg.get("extra_trades"), default=0))

        max_extra_trades = self._safe_int(profit_cfg.get("max_extra_trades"), default=0)
        if max_extra_trades > 0:
            extra_trades = min(extra_trades, max_extra_trades)

        allowed = base + extra_trades

        max_trades_cap = self._safe_int(profit_cfg.get("max_trades_cap"), default=0)
        if max_trades_cap > 0:
            allowed = min(allowed, max_trades_cap)

        allowed = max(base, int(allowed))
        return allowed, pnl_usd, "profit_unlock:eligible"
    
    def validate(self, signal: Dict, portfolio_manager) -> Tuple[bool, str]:
        """
        Validate that daily trade limit has not been reached.
        
        Args:
            signal: Trading signal
            portfolio_manager: PortfolioManager instance (must have get_todays_trade_count method)
            
        Returns:
            (is_valid, reason) tuple
        """
        if not self.enabled:
            return (True, f"{self.rule_name} disabled")
        
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            
            # Get current trade count from PortfolioManager
            if not hasattr(portfolio_manager, 'get_todays_trade_count'):
                logger.error(
                    f"[{self.rule_name}] PortfolioManager does not have get_todays_trade_count method! "
                    f"This is likely an integration issue. "
                    f"FAIL-SAFE MODE: Allowing trade (limit cannot be enforced)."
                )
                # Fail safely: if we can't check, don't block the trade
                # This prevents breaking production if PortfolioManager link is not established
                return (True, f"{self.rule_name}: cannot verify (missing method - fail-safe mode)")
            
            todays_trades = portfolio_manager.get_todays_trade_count()
            allowed_max, pnl_usd, policy = self._get_dynamic_allowed_max(portfolio_manager)
            
            if pnl_usd is None:
                logger.debug(
                    f"[{self.rule_name}] {symbol}: Today's trades: {todays_trades}/{allowed_max} (policy={policy})"
                )
            else:
                logger.debug(
                    f"[{self.rule_name}] {symbol}: Today's trades: {todays_trades}/{allowed_max} "
                    f"(policy={policy}, pnl_usd={pnl_usd:+.2f})"
                )
            
            if todays_trades >= allowed_max:
                logger.warning(
                    f"🚫 [{self.rule_name}] REJECTED: {symbol}\n"
                    f"   Daily trade limit reached: {todays_trades}/{allowed_max}\n"
                    f"   No more trades allowed until next trading day."
                )
                return (False, f"Daily trade limit reached: {todays_trades}/{allowed_max}")
            
            logger.debug(f"✅ [{self.rule_name}] PASSED: {symbol} ({todays_trades + 1}/{allowed_max})")
            return (True, f"Daily trade limit check passed ({todays_trades + 1}/{allowed_max})")
            
        except Exception as e:
            logger.error(f"[{self.rule_name}] Validation error: {e}", exc_info=True)
            # Fail safely: if there's an error, don't block the trade
            return (True, f"{self.rule_name}: error during validation")
