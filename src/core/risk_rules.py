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
            portfolio_manager: PortfolioManager instance
            
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
            
            # Get portfolio state
            portfolio = portfolio_manager.get_portfolio_summary()
            available = portfolio.get('available_capital', portfolio.get('portfolio_value', 0))
            
            if notional <= 0:
                return (False, f"Invalid notional value: {notional}")
            
            # Spot trading (leverage = 1)
            if leverage <= 1:
                if notional > available:
                    logger.warning(f"🚫 [CapitalLimitRule] REJECTED {symbol} (spot): ${notional:.2f} > ${available:.2f}")
                    return (False, f"Spot position ${notional:.2f} exceeds available ${available:.2f}")
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
            
            portfolio_value = portfolio_manager.get_current_equity()
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
            
            portfolio_value = portfolio_manager.get_current_equity()
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
    Validates that a trade has an acceptable risk/reward ratio.
    
    This rule is now DYNAMIC. It prioritizes the 'min_rr_ratio' from the
    incoming signal, making it responsive to strategy-specific or config-level changes.
    """
    
    def __init__(self, min_risk_reward: float = 1.5, rule_name: str = None):
        """
        Initialize the risk/reward ratio rule.
        
        Args:
            min_risk_reward: A fallback minimum ratio, used only if the signal
                             does not provide its own 'min_rr_ratio'. Default is 1.5.
            rule_name: Optional custom rule name.
        """
        super().__init__(rule_name or "RiskRewardRatioRule")
        # Bu değer artık sadece bir "varsayılan" veya "fallback" görevi görüyor.
        self.fallback_min_risk_reward = min_risk_reward
    
    def validate(self, signal: Dict, portfolio_manager) -> Tuple[bool, str]:
        """
        Validate risk/reward ratio using the dynamic limit from the signal itself.
        
        Args:
            signal: Trading signal, expected to contain 'min_rr_ratio'.
            portfolio_manager: PortfolioManager instance.
            
        Returns:
            (is_valid, reason) tuple.
        """
        if not self.enabled:
            return (True, f"{self.rule_name} disabled")
        
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            
            # ======================= ANA DÜZELTME =======================
            # 1. Dinamik R/R Hedefini Sinyalden Oku
            #    Eğer sinyalde bu değer yoksa, başlatmadaki varsayılan değeri kullan.
            min_rr_target = signal.get('min_rr_ratio', self.fallback_min_risk_reward)
            # ==========================================================

            entry_price = signal.get('entry', 0)
            stop_loss = signal.get('stop', 0)
            target_price = signal.get('target', 0)
            
            # Gerekli fiyat bilgileri eksikse kural başarısız olur.
            if not all([entry_price, stop_loss, target_price]):
                reason = f"Missing price data for R/R calculation (entry={entry_price}, stop={stop_loss}, target={target_price})"
                logger.warning(f"[{self.rule_name}] {symbol}: {reason}")
                return (False, reason)
            
            # Risk ve Getiri mesafelerini hesapla
            risk_distance = abs(entry_price - stop_loss)
            reward_distance = abs(target_price - entry_price)
            
            if risk_distance > 0:
                # Gerçek R/R oranını hesapla
                calculated_rr_ratio = reward_distance / risk_distance
                
                # Loglamayı, hangi hedefe göre kontrol edildiğini gösterecek şekilde güncelle.
                log_msg = (f"[{self.rule_name}] {symbol}: "
                           f"Calculated R/R Ratio {calculated_rr_ratio:.2f} vs "
                           f"Target Minimum {min_rr_target:.2f}")
                
                # 2. Karşılaştırmayı Dinamik Hedefe Göre Yap
                if calculated_rr_ratio < min_rr_target:
                    reason = f"Risk/reward ratio {calculated_rr_ratio:.2f} is below the target minimum of {min_rr_target:.2f}"
                    logger.warning(f"🚫 [{self.rule_name}] REJECTED: {symbol} - {reason}")
                    # Ret logunu log_msg ile birleştirerek daha anlaşılır hale getir
                    logger.debug(log_msg) 
                    return (False, reason)
                
                logger.debug(log_msg + " -> ✅ PASSED")
                return (True, "Risk/reward ratio is acceptable")
            else:
                # Risk mesafesi sıfır ise (örneğin stop loss giriş fiyatıyla aynıysa)
                logger.warning(f"[{self.rule_name}] {symbol}: Zero risk distance, cannot calculate R/R ratio.")
                return (False, "Zero risk distance prevents R/R calculation")
            
        except Exception as e:
            logger.error(f"[{self.rule_name}] Validation error: {e}", exc_info=True)
            return (False, f"Critical validation error: {str(e)}")
    
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
