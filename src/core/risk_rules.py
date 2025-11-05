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
from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)


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
            logger.debug(f"[CapitalLimitRule] portfolio_value={portfolio_value}, current_exposure={current_exposure}, available={available}")
            
            if notional <= 0:
                return (False, f"Invalid notional value: {notional}")
            
            # Spot trading (leverage = 1)
            if leverage <= 1:
                if notional > available:
                    logger.warning(f"🚫 [CapitalLimitRule] REJECTED {symbol} (spot): ${notional:.2f} > ${available:.2f}")
                    return (False, f"Position ${notional:.2f} would exceed capital limit ${available:.2f} available")
                else:
                    logger.info(f"✅ [CapitalLimitRule] PASSED {symbol} (spot): ${notional:.2f} <= ${available:.2f}")
                    return (True, "Capital check passed (spot)")
            
            # Futures trading (leverage > 1)
            required_margin = notional / leverage
            
            if required_margin > available:
                logger.warning(f"🚫 [CapitalLimitRule] REJECTED {symbol} (futures): Margin ${required_margin:.2f} > ${available:.2f}")
                return (False, f"Margin ${required_margin:.2f} exceeds available ${available:.2f}")
            else:
                logger.info(f"✅ [CapitalLimitRule] PASSED {symbol} (futures): Margin ${required_margin:.2f} <= ${available:.2f}")
                return (True, f"Margin check passed (leverage {leverage}x)")
                
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
            
            # Get portfolio value using helper function
            portfolio_value = _get_portfolio_value(portfolio_manager, signal)
                
            position_value = position_size * entry_price
            max_position_value = portfolio_value * self.max_position_size
            
            position_size_pct = position_value / portfolio_value if portfolio_value > 0 else 0
            
            logger.debug(f"[{self.rule_name}] {symbol}: ${position_value:.2f} ({position_size_pct:.1%}) vs max ${max_position_value:.2f} ({self.max_position_size:.1%})")
            
            if position_value > max_position_value:
                logger.warning(f"🚫 [{self.rule_name}] REJECTED: {symbol} position size ${position_value:.2f} exceeds max ${max_position_value:.2f}")
                return (False, f"Position size ${position_value:.2f} exceeds max ${max_position_value:.2f}")
            
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
    
    def __init__(self, max_portfolio_heat: float = 0.10, max_portfolio_risk: float = 0.02, rule_name: str = None):
        """
        Initialize portfolio heat rule.
        
        Args:
            max_portfolio_heat: Maximum total portfolio heat (default 10%)
            max_portfolio_risk: Maximum risk per single trade (default 2%)
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
            
            # Calculate stop loss if missing
            if not stop_loss:
                stop_loss = self._calculate_stop_loss(signal, entry_price)
            
            # Get portfolio value using helper function
            portfolio_value = _get_portfolio_value(portfolio_manager, signal)
            
            # Get active positions (empty dict for test compatibility)
            if isinstance(portfolio_manager, dict) or not hasattr(portfolio_manager, 'get_open_positions'):
                active_positions = {}
            else:
                active_positions = portfolio_manager.get_open_positions()
            
            # Calculate risk for this position
            risk_amount = abs(entry_price - stop_loss) * position_size
            max_risk = portfolio_value * self.max_portfolio_risk
            
            risk_pct = risk_amount / portfolio_value if portfolio_value > 0 else 0
            
            # Check individual position risk
            if risk_amount > max_risk:
                logger.warning(f"🚫 [{self.rule_name}] REJECTED: {symbol} risk ${risk_amount:.2f} ({risk_pct:.2%}) exceeds max ${max_risk:.2f} ({self.max_portfolio_risk:.2%})")
                return (False, f"Risk amount ${risk_amount:.2f} exceeds max ${max_risk:.2f}")
            
            # Calculate total portfolio heat
            total_risk = sum(pos.get('risk_amount', 0) for pos in active_positions.values())
            total_risk += risk_amount
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
            if calculated_rr < target_rr:
                reason = (f"Risk/reward ratio {calculated_rr:.2f} is below dynamic target {target_rr:.2f} "
                         f"(Risk: {risk_pct:.1f}%, Reward: {reward_pct:.1f}%)")
                logger.warning(f"🚫 [RiskRewardRatioRule] REJECTED {symbol}: {reason}")
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
        
        config = self.risk_config.rr_dynamic
        
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
        
        # Extract intelligence metrics with fallbacks
        ml_conf = float(signal.get('ml_confidence', fallback['missing_ml_default']))
        rl_agree = 1.0 if signal.get('rl_is_agree', False) else 0.0
        rl_prob = float(signal.get('rl_action_prob', fallback['missing_rl_default']))
        regime_conf = float(signal.get('regime_confidence', fallback.get('missing_regime_default', 0.3)))
        regime_name = signal.get('regime_name', 'neutral').lower()
        
        # Calculate relaxation (reduces required R/R for high confidence)
        relaxation = (
            weights['ml_confidence'] * ml_conf +
            weights['rl_agreement'] * rl_agree * rl_prob
        )
        
        # Calculate tightening (increases required R/R for uncertainty)
        tightening = weights['regime_clarity'] * (1.0 - regime_conf)
        
        # Apply regime multiplier
        regime_mult = regime_mults.get(regime_name, 1.0)
        
        # Calculate dynamic target
        dynamic_target = (base_rr - relaxation + tightening) * regime_mult
        
        # Respect strategy's minimum
        strategy_min = float(signal.get('strategy_min_rr', 0.5))
        dynamic_target = max(dynamic_target, strategy_min)
        
        # Apply bounds
        final_target = max(lower_bound, min(dynamic_target, upper_bound))
        
        # Detailed logging
        logger.info(f"📊 [Dynamic R/R Calc] Base={base_rr:.2f} - Relax={relaxation:.2f} + Tight={tightening:.2f} "
                   f"× Regime({regime_name})={regime_mult:.1f} = {dynamic_target:.2f} → Final={final_target:.2f}")
        
        return final_target
    
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
    
    def __init__(self, max_daily_trades: int, rule_name: str = None):
        """
        Initialize daily trade limit rule.
        
        Args:
            max_daily_trades: Maximum number of trades allowed per day
            rule_name: Optional custom rule name
        """
        super().__init__(rule_name or "DailyTradeLimitRule")
        self.max_daily_trades = max_daily_trades
        
        if max_daily_trades <= 0:
            logger.warning(f"⚠️ DailyTradeLimitRule initialized with invalid limit: {max_daily_trades}. "
                         f"This rule will effectively block all trades.")
    
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
                logger.error(f"[{self.rule_name}] PortfolioManager does not have get_todays_trade_count method!")
                # Fail safely: if we can't check, don't block the trade
                return (True, f"{self.rule_name}: cannot verify (missing method)")
            
            todays_trades = portfolio_manager.get_todays_trade_count()
            
            logger.debug(f"[{self.rule_name}] {symbol}: Today's trades: {todays_trades}/{self.max_daily_trades}")
            
            if todays_trades >= self.max_daily_trades:
                logger.warning(
                    f"🚫 [{self.rule_name}] REJECTED: {symbol}\n"
                    f"   Daily trade limit reached: {todays_trades}/{self.max_daily_trades}\n"
                    f"   No more trades allowed until next trading day."
                )
                return (False, f"Daily trade limit reached: {todays_trades}/{self.max_daily_trades}")
            
            logger.debug(f"✅ [{self.rule_name}] PASSED: {symbol} ({todays_trades + 1}/{self.max_daily_trades})")
            return (True, f"Daily trade limit check passed ({todays_trades + 1}/{self.max_daily_trades})")
            
        except Exception as e:
            logger.error(f"[{self.rule_name}] Validation error: {e}", exc_info=True)
            # Fail safely: if there's an error, don't block the trade
            return (True, f"{self.rule_name}: error during validation")
