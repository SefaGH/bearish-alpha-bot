"""
Comprehensive Risk Management Engine.
Provides portfolio-level risk management, position validation, and capital allocation.

PHASE 3 REFACTOR: Transform into Rules Engine
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import numpy as np

# Import RiskConfiguration for type-safe configuration
from config.risk_config import RiskConfiguration

# PHASE 3: Import risk rules framework
try:
    from core.risk_rules import (
        BaseRiskRule,
        CapitalLimitRule,
        PositionSizeRule,
        PortfolioHeatRule,
        MaxDrawdownRule,
        RiskRewardRatioRule,
        StrategyPerformanceRule
    )
except ModuleNotFoundError:
    try:
        from src.core.risk_rules import (
            BaseRiskRule,
            CapitalLimitRule,
            PositionSizeRule,
            PortfolioHeatRule,
            MaxDrawdownRule,
            RiskRewardRatioRule,
            StrategyPerformanceRule
        )
    except ModuleNotFoundError as e:
        if e.name in ('src', 'src.core', 'src.core.risk_rules'):
            from ..risk_rules import (
                BaseRiskRule,
                CapitalLimitRule,
                PositionSizeRule,
                PortfolioHeatRule,
                MaxDrawdownRule,
                RiskRewardRatioRule,
                StrategyPerformanceRule
            )
        else:
            raise

# Triple-fallback import strategy for maximum compatibility:
# 1. Direct utils import (when src/ is on sys.path)
# 2. Absolute src.utils import (when repo root is on sys.path)
# 3. Relative import (when imported as package module)
try:
    # Option 1: Direct import (scripts add src/ to sys.path)
    from utils.pnl_calculator import calculate_unrealized_pnl
except ModuleNotFoundError:
    try:
        # Option 2: Absolute import (repo root on sys.path)
        from src.utils.pnl_calculator import calculate_unrealized_pnl
    except ModuleNotFoundError as e:
        # Option 3: Relative import (package context)
        if e.name in ('src', 'src.utils', 'src.utils.pnl_calculator'):
            from ..utils.pnl_calculator import calculate_unrealized_pnl
        else:
            # Unknown module missing, re-raise
            raise
logger = logging.getLogger(__name__)


class RiskManager:
    """Comprehensive risk management engine for multi-strategy portfolio."""
    
    def __init__(self, portfolio_value: float, risk_config: RiskConfiguration, 
                 websocket_manager=None, performance_monitor=None, rules: List[BaseRiskRule] = None):
        """
        Initialize risk manager with standardized configuration.
        
        PHASE 3 REFACTOR: RiskManager is now a RULES ENGINE.
        Individual risk validation rules are composable and independently testable.
        
        Args:
            portfolio_value: Initial portfolio value in USD (kept for backward compatibility)
            risk_config: RiskConfiguration object with all risk parameters
            websocket_manager: Optional WebSocket manager for real-time data
            performance_monitor: Optional performance monitor for strategy metrics
            rules: Optional list of risk rules to apply (if None, uses default rules)
        """
        self.risk_config = risk_config
        self.ws_manager = websocket_manager
        self.performance_monitor = performance_monitor
        
        # Extract risk limits from standardized configuration
        self.risk_limits_dataclass = self.risk_config.get_risk_limits()
        self.risk_limits = {
            'max_portfolio_risk': self.risk_limits_dataclass.max_portfolio_risk,
            'max_position_size': self.risk_limits_dataclass.max_position_size,
            'max_drawdown': self.risk_limits_dataclass.max_drawdown,
            'max_correlation': self.risk_limits_dataclass.max_correlation,
        }
        
        # PHASE 3: Rules Engine - Composable, extensible risk validation
        if rules is not None:
            self.rules = rules
        else:
            # Default rule set based on risk configuration
            self.rules = self._create_default_rules()
        
        # PHASE 2: Removed portfolio state - RiskManager is now stateless
        # State is managed by PortfolioManager
        # Keep portfolio_value only for backward compatibility during transition
        self.portfolio_value = float(portfolio_value)
        
        # DEPRECATED: These properties are kept for backward compatibility
        # They will be removed in Phase 3
        # TODO: Add deprecation warnings in Phase 3 to encourage migration
        self.active_positions = {}  # DEPRECATED: Use PortfolioManager.get_open_positions()
        self.current_drawdown = 0.0  # DEPRECATED: Use PortfolioManager.get_current_drawdown()
        self.peak_portfolio_value = self.portfolio_value  # DEPRECATED: Use PortfolioManager.get_peak_equity()
        
        # Log initialization with standardized configuration
        logger.info(f"RiskManager initialized (PHASE 3: Rules Engine)")
        logger.info(f"Risk configuration: {self.risk_config.to_dict()}")
        logger.info(f"Risk limits: {self.risk_limits}")
        logger.info(f"Active rules: {[rule.rule_name for rule in self.rules]}")
    
    def _create_default_rules(self) -> List[BaseRiskRule]:
        """
        Create default set of risk rules based on configuration.
        
        Returns:
            List of risk rule instances
        """
        return [
            # Order matters: Check capital availability first
            CapitalLimitRule(),
            # Then check position size limits
            PositionSizeRule(max_position_size=self.risk_limits['max_position_size']),
            # Check overall portfolio risk
            PortfolioHeatRule(
                max_portfolio_heat=0.10,  # 10% max total portfolio heat
                max_portfolio_risk=self.risk_limits['max_portfolio_risk']
            ),
            # Check drawdown limits
            MaxDrawdownRule(max_drawdown=self.risk_limits['max_drawdown']),
            # Validate risk/reward ratio
            RiskRewardRatioRule(min_risk_reward=1.5),
            # Optional: Check strategy performance
            StrategyPerformanceRule(
                min_win_rate=0.35,
                performance_monitor=self.performance_monitor
            )
        ]
    
    def _calculate_total_portfolio_exposure(self, portfolio_manager=None) -> float:
        """
        Calculate total notional value of all open positions.
        
        PHASE 2: This method now queries PortfolioManager for state.
        Falls back to deprecated self.active_positions for backward compatibility.
        
        Args:
            portfolio_manager: PortfolioManager instance (preferred)
        
        Returns:
            Total exposure in USDT
        """
        # PHASE 2: Prefer PortfolioManager as source of truth
        if portfolio_manager is not None:
            return portfolio_manager.get_total_exposure()
        
        # Fallback to deprecated state for backward compatibility
        total_exposure = sum(
            pos.get('size', 0) * pos.get('entry_price', 0) 
            for pos in self.active_positions.values()
        )
        
        active_count = len(self.active_positions)
        portfolio_value = self.portfolio_value
        capital_utilization = (total_exposure / portfolio_value * 100) if portfolio_value > 0 else 0
        
        logger.debug(f"📊 [EXPOSURE] Active positions: {active_count}, Total exposure: ${total_exposure:.2f}, Capital utilization: {capital_utilization:.1f}%")
        
        return total_exposure
    
    def set_risk_limits(self, max_portfolio_risk: float = 0.02, max_position_size: float = 0.10,
                       max_drawdown: float = 0.15, max_correlation: float = 0.7):
        """
        Configure portfolio-level risk limits.
        
        Args:
            max_portfolio_risk: Maximum portfolio risk per trade (default 2%)
            max_position_size: Maximum single position size (default 10% of portfolio)
            max_drawdown: Maximum allowed drawdown (default 15%)
            max_correlation: Maximum correlation between positions (default 70%)
        """
        self.risk_limits = {
            'max_portfolio_risk': max_portfolio_risk,
            'max_position_size': max_position_size,
            'max_drawdown': max_drawdown,
            'max_correlation': max_correlation,
        }
        logger.info(f"Risk limits updated: {self.risk_limits}")
    
    async def validate_new_position(self, signal: Dict, current_portfolio: Dict = None, portfolio_manager=None) -> Tuple[bool, str, Dict]:
        """
        Validate if new position meets risk criteria using rules engine.
        
        PHASE 3 REFACTOR: Uses composable rules engine for validation.
        Each rule is independently testable and can be added/removed without
        modifying this method.
        
        Args:
            signal: Trading signal with entry, stop, size, etc.
            current_portfolio: Current portfolio state (DEPRECATED - use portfolio_manager)
            portfolio_manager: PortfolioManager instance (preferred)
            
        Returns:
            Tuple of (is_valid, reason, risk_metrics)
        """
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            
            # PHASE 2: Get portfolio manager (prefer parameter, fallback to creating mock)
            if portfolio_manager is None:
                # Backward compatibility: create a minimal portfolio manager mock
                logger.warning("No PortfolioManager provided, using deprecated fallback mode")
                portfolio_manager = self._create_fallback_portfolio_manager()
            
            logger.debug(f"🛡️ [RISK-ENGINE] Validating position for {symbol}")
            logger.debug(f"🛡️ [RISK-ENGINE] Running {len(self.rules)} risk rules")
            
            # Initialize risk metrics
            risk_metrics = self._calculate_risk_metrics(signal, portfolio_manager)
            
            # PHASE 3: Run all rules through the rules engine
            for rule in self.rules:
                if not rule.enabled:
                    logger.debug(f"⏭️  [RISK-ENGINE] Skipping disabled rule: {rule.rule_name}")
                    continue
                
                is_valid, reason = rule.validate(signal, portfolio_manager)
                
                if not is_valid:
                    logger.warning(f"🚫 [RISK-ENGINE] Position REJECTED by {rule.rule_name}")
                    logger.warning(f"   Symbol: {symbol}")
                    logger.warning(f"   Reason: {reason}")
                    return (False, reason, risk_metrics)
                
                logger.debug(f"✅ [RISK-ENGINE] {rule.rule_name} passed")
            
            # All rules passed
            logger.info(f"✅ [RISK-ENGINE] Position APPROVED for {symbol}")
            logger.info(f"   All {len([r for r in self.rules if r.enabled])} rules passed")
            return (True, "All risk rules passed", risk_metrics)
            
        except Exception as e:
            logger.error(f"[RISK-ENGINE] Error validating position: {e}", exc_info=True)
            return (False, f"Validation error: {str(e)}", {})
    
    def _calculate_risk_metrics(self, signal: Dict, portfolio_manager) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics for a signal.
        
        Args:
            signal: Trading signal
            portfolio_manager: PortfolioManager instance
            
        Returns:
            Dictionary of risk metrics
        """
        try:
            position_size = signal.get('position_size', 0)
            entry_price = signal.get('entry', 0)
            stop_loss = signal.get('stop', 0)
            target_price = signal.get('target', entry_price * 1.02)
            
            # Calculate stop loss if missing
            if not stop_loss:
                stop_loss = self._calculate_stop_loss_from_signal(signal, entry_price)
            
            portfolio_value = portfolio_manager.get_current_equity()
            current_exposure = portfolio_manager.get_total_exposure()
            active_positions = portfolio_manager.get_open_positions()
            
            # Basic metrics
            new_position_value = position_size * entry_price
            projected_exposure = current_exposure + new_position_value
            position_size_pct = new_position_value / portfolio_value if portfolio_value > 0 else 0
            
            # Risk metrics
            risk_amount = abs(entry_price - stop_loss) * position_size if stop_loss else 0
            risk_pct = risk_amount / portfolio_value if portfolio_value > 0 else 0
            
            # Risk/reward
            risk_distance = abs(entry_price - stop_loss) if stop_loss else 0
            reward_distance = abs(target_price - entry_price)
            risk_reward_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
            
            # Portfolio heat
            total_risk = sum(pos.get('risk_amount', 0) for pos in active_positions.values())
            portfolio_heat = (total_risk + risk_amount) / portfolio_value if portfolio_value > 0 else 0
            
            return {
                'portfolio_value': portfolio_value,
                'current_exposure': current_exposure,
                'new_position_value': new_position_value,
                'position_value': new_position_value,  # Backward compatibility alias
                'projected_exposure': projected_exposure,
                'position_size_pct': position_size_pct,
                'max_position_value': portfolio_value * self.risk_limits['max_position_size'],
                'risk_amount': risk_amount,
                'max_risk_amount': portfolio_value * self.risk_limits['max_portfolio_risk'],
                'risk_pct': risk_pct,
                'risk_reward_ratio': risk_reward_ratio,
                'portfolio_heat': portfolio_heat,
                'current_drawdown': portfolio_manager.get_current_drawdown(),
                'max_drawdown': self.risk_limits['max_drawdown'],
                'active_positions_count': len(active_positions)
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_stop_loss_from_signal(self, signal: Dict, entry_price: float) -> float:
        """
        Calculate stop loss from signal parameters if not explicitly provided.
        
        Args:
            signal: Trading signal
            entry_price: Entry price
            
        Returns:
            Stop loss price
        """
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
    
    def _create_fallback_portfolio_manager(self):
        """
        Create a minimal portfolio manager mock for backward compatibility.
        
        Returns:
            Mock portfolio manager with basic functionality
        """
        class FallbackPortfolioManager:
            def __init__(self, risk_manager):
                self.rm = risk_manager
            
            def get_current_equity(self):
                return self.rm.portfolio_value
            
            def get_current_drawdown(self):
                return self.rm.current_drawdown
            
            def get_open_positions(self):
                return self.rm.active_positions
            
            def get_total_exposure(self):
                return sum(
                    pos.get('size', 0) * pos.get('entry_price', 0)
                    for pos in self.rm.active_positions.values()
                )
        
        return FallbackPortfolioManager(self)
    
    async def monitor_position_risk(self, position_id: str, portfolio_manager=None) -> Dict[str, Any]:
        """
        Real-time position risk monitoring.
        
        PHASE 2: Now accepts PortfolioManager to query position state.
        
        Args:
            position_id: Unique position identifier
            portfolio_manager: PortfolioManager instance (preferred)
            
        Returns:
            Dictionary with position risk metrics and alerts
        """
        try:
            # PHASE 2: Get position from PortfolioManager or fallback
            if portfolio_manager is not None:
                position = portfolio_manager.get_position(position_id)
                if position is None:
                    return {'status': 'not_found', 'alerts': []}
                portfolio_value = portfolio_manager.get_current_equity()
            else:
                # Backward compatibility fallback
                if position_id not in self.active_positions:
                    return {'status': 'not_found', 'alerts': []}
                position = self.active_positions[position_id]
                portfolio_value = self.portfolio_value
            alerts = []
            
            # Current position state
            entry_price = position.get('entry_price', 0)
            current_price = position.get('current_price', entry_price)
            stop_loss = position.get('stop_loss', 0)
            position_size = position.get('size', 0)
            
            # Calculate unrealized P&L
            unrealized_pnl = calculate_unrealized_pnl(
                position.get('side', 'long'), entry_price, current_price, position_size
            )
            
            position['unrealized_pnl'] = unrealized_pnl
            
            # Stop-loss breach check
            if position.get('side') == 'long' and current_price <= stop_loss:
                alerts.append({
                    'type': 'stop_loss_breach',
                    'severity': 'high',
                    'message': f"Stop loss breached: {current_price} <= {stop_loss}"
                })
            elif position.get('side') == 'short' and current_price >= stop_loss:
                alerts.append({
                    'type': 'stop_loss_breach',
                    'severity': 'high',
                    'message': f"Stop loss breached: {current_price} >= {stop_loss}"
                })
            
            # Large unrealized loss check
            loss_threshold = portfolio_value * self.risk_limits['max_portfolio_risk']
            if unrealized_pnl < -loss_threshold:
                alerts.append({
                    'type': 'large_loss',
                    'severity': 'high',
                    'message': f"Unrealized loss ${unrealized_pnl:.2f} exceeds threshold ${loss_threshold:.2f}"
                })
            
            # Time-based exit (if position held too long)
            entry_time = position.get('entry_time')
            if entry_time:
                hold_duration = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
                max_hold_hours = position.get('max_hold_hours', 72)
                
                if hold_duration > max_hold_hours:
                    alerts.append({
                        'type': 'time_limit',
                        'severity': 'medium',
                        'message': f"Position held for {hold_duration:.1f}h, exceeds {max_hold_hours}h"
                    })
            
            return {
                'status': 'active',
                'position_id': position_id,
                'unrealized_pnl': unrealized_pnl,
                'current_price': current_price,
                'alerts': alerts
            }
            
        except Exception as e:
            logger.error(f"Error monitoring position {position_id}: {e}")
            return {'status': 'error', 'message': str(e), 'alerts': []}
    
    async def calculate_position_size(self, signal: Dict, market_regime: Dict = None,
                                     portfolio_state: Dict = None, portfolio_manager=None) -> float:
        """
        Calculate optimal position size based on risk parameters.
        
        PHASE 2: Now accepts PortfolioManager to query current equity.
        
        Args:
            signal: Trading signal with entry, stop, target
            market_regime: Market regime data from Phase 2 (optional)
            portfolio_state: Current portfolio state (DEPRECATED - use portfolio_manager)
            portfolio_manager: PortfolioManager instance (preferred)
            
        Returns:
            Optimal position size
        """
        try:
            # PHASE 2: Get portfolio value from PortfolioManager or fallback
            if portfolio_manager is not None:
                portfolio_value = portfolio_manager.get_current_equity()
            else:
                portfolio_value = self.portfolio_value
            
            entry_price = signal.get('entry', 0)
            stop_loss = signal.get('stop', 0)
            
            # Fallback 1: Calculate from ATR if stop missing
            if not stop_loss and signal.get('sl_atr_mult') and signal.get('atr'):
                atr = signal['atr']
                sl_mult = signal['sl_atr_mult']
                side = signal.get('side', 'buy')
                
                if side in ['buy', 'long']:
                    stop_loss = entry_price - (atr * sl_mult)
                else:  # sell/short
                    stop_loss = entry_price + (atr * sl_mult)
                
                logger.info(f"📊 [RISK] Calculated ATR-based stop: {stop_loss:.2f} (ATR={atr:.2f}, mult={sl_mult})")
            
            # Fallback 2: Calculate from sl_pct if still missing
            if not stop_loss and signal.get('sl_pct'):
                sl_pct = signal['sl_pct']
                side = signal.get('side', 'buy')
                
                if side in ['buy', 'long']:
                    stop_loss = entry_price * (1 - sl_pct)
                else:
                    stop_loss = entry_price * (1 + sl_pct)
                
                logger.info(f"📊 [RISK] Calculated percentage-based stop: {stop_loss:.2f} (pct={sl_pct:.2%})")
            
            if entry_price <= 0 or stop_loss <= 0:
                logger.warning("Invalid entry or stop price after all fallbacks")
                return 0.0
            
            # Base risk amount
            risk_per_trade = portfolio_value * self.risk_limits['max_portfolio_risk']
            
            # Adjust for market regime
            if market_regime:
                risk_multiplier = market_regime.get('risk_multiplier', 1.0)
                risk_per_trade *= risk_multiplier
                logger.debug(f"Risk adjusted by regime multiplier: {risk_multiplier:.2f}")
            
            # Adjust for strategy performance
            if self.performance_monitor and signal.get('strategy'):
                strategy_name = signal['strategy']
                summary = self.performance_monitor.get_strategy_summary(strategy_name)
                metrics = summary.get('metrics', {})
                
                win_rate = metrics.get('win_rate', 0.5)
                sharpe = metrics.get('sharpe_ratio', 0)
                
                # Reduce size for poor performing strategies
                if win_rate < 0.4:
                    risk_per_trade *= 0.5
                    logger.debug("Risk reduced due to low win rate")
                elif win_rate > 0.6 and sharpe > 1.0:
                    risk_per_trade *= 1.2
                    logger.debug("Risk increased due to good performance")
            
            # Calculate position size
            risk_distance = abs(entry_price - stop_loss)
            position_size = risk_per_trade / risk_distance
            
            # Apply maximum position size limit
            max_position_value = portfolio_value * self.risk_limits['max_position_size']
            max_size_by_limit = max_position_value / entry_price
            position_size = min(position_size, max_size_by_limit)
            
            logger.info(f"Calculated position size: {position_size:.4f} (risk: ${risk_per_trade:.2f})")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def register_position(self, position_id: str, position_data: Dict):
        """
        Register a new active position.
        
        Args:
            position_id: Unique position identifier
            position_data: Position data including entry, stop, size, etc.
        """
        self.active_positions[position_id] = {
            **position_data,
            'entry_time': datetime.now(timezone.utc),
            'current_price': position_data.get('entry_price', 0)
        }
        logger.info(f"Position registered: {position_id} - {position_data.get('symbol', 'UNKNOWN')}")
    
    def close_position(self, position_id: str, exit_price: float, realized_pnl: float):
        """
        Close and remove a position.
        
        Args:
            position_id: Position identifier
            exit_price: Exit price
            realized_pnl: Realized profit/loss
        """
        if position_id in self.active_positions:
            position = self.active_positions.pop(position_id)
            
            # Update portfolio value
            self.portfolio_value += realized_pnl
            
            # Update drawdown metrics
            if self.portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = self.portfolio_value
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            
            logger.info(f"Position closed: {position_id} - PnL: ${realized_pnl:.2f}, Portfolio: ${self.portfolio_value:.2f}")
        else:
            logger.warning(f"Attempted to close non-existent position: {position_id}")
    
    def update_position_price(self, position_id: str, current_price: float):
        """
        Update current price for a position.
        
        Args:
            position_id: Position identifier
            current_price: Current market price
        """
        if position_id in self.active_positions:
            self.active_positions[position_id]['current_price'] = current_price
    
    def get_portfolio_summary(self, portfolio_manager=None) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary.
        
        PHASE 2: Now accepts PortfolioManager to query state.
        Falls back to deprecated state for backward compatibility.
        
        Args:
            portfolio_manager: PortfolioManager instance (preferred)
        
        Returns:
            Portfolio summary dictionary
        """
        # PHASE 2: Prefer PortfolioManager as source of truth
        if portfolio_manager is not None:
            # Get data from PortfolioManager
            portfolio_value = portfolio_manager.get_current_equity()
            peak_value = portfolio_manager.get_peak_equity()
            current_drawdown = portfolio_manager.get_current_drawdown()
            active_positions = portfolio_manager.get_open_positions()
            total_exposure = portfolio_manager.get_total_exposure()
            available_capital = portfolio_manager.get_available_capital()
        else:
            # Backward compatibility fallback
            portfolio_value = self.portfolio_value
            peak_value = self.peak_portfolio_value
            current_drawdown = self.current_drawdown
            active_positions = self.active_positions
            total_exposure = self._calculate_total_portfolio_exposure()
            available_capital = portfolio_value - total_exposure
        
        total_unrealized_pnl = sum(
            pos.get('unrealized_pnl', 0) 
            for pos in active_positions.values()
        )
        
        total_risk = sum(
            pos.get('risk_amount', 0) 
            for pos in active_positions.values()
        )
        
        capital_utilization = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        return {
            'portfolio_value': portfolio_value,
            'peak_value': peak_value,
            'current_drawdown': current_drawdown,
            'active_positions': len(active_positions),
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_risk': total_risk,
            'portfolio_heat': total_risk / portfolio_value if portfolio_value > 0 else 0,
            'total_exposure': total_exposure,
            'available_capital': available_capital,
            'capital_utilization': capital_utilization,
            'risk_limits': self.risk_limits
        }
