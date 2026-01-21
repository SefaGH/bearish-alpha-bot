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
from typing import Dict, Tuple, Any
from datetime import datetime, timezone
import logging

from core.logger import get_current_run_id

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

    def __init__(self, risk_matrix: Dict[str, Dict[str, float]], rule_name: str = None):
        super().__init__(rule_name or "VolumeAwarePositionSizingRule")
        self.risk_matrix = risk_matrix or {}

    def validate(self, signal: Dict, portfolio_manager) -> Tuple[bool, str]:
        if not self.enabled:
            return (True, f"{self.rule_name} disabled")

        ctx_source = signal.get('volume_ctx_source')
        if ctx_source and ctx_source != 'analyzer':
            return (True, "Volume context not analyzer-derived; skipping")

        bucket = (signal.get('volume_bucket') or '').upper()
        if not bucket or not self.risk_matrix:
            return (True, "No volume bucket provided; skipping")

        cfg = self.risk_matrix.get(bucket) or self.risk_matrix.get('NORMAL')
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
        regime_name = signal.get('regime_name', 'neutral').lower()
        
        # Get regime_weight (soft-weighting), default to 1.0 if not available
        # Note: regime_weight is calculated from regime_conf in strategy_integration.py:
        #   - regime_weight = None if regime_conf < 0.30 (hard reject)
        #   - regime_weight = regime_conf / 0.60 if 0.30 <= regime_conf < 0.60
        #   - regime_weight = 1.0 if regime_conf >= 0.60
        # Legacy signals without regime_weight are assumed to have full confidence (1.0)
        regime_weight = _safe_float(signal.get('regime_weight', 1.0), 1.0)
        
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
        
        # Calculate dynamic target
        dynamic_target = (base_rr - relaxation + tightening) * regime_adjustment
        
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
        dynamic_target = max(dynamic_target, strategy_floor)
        
        # Apply bounds
        final_target = max(lower_bound, min(dynamic_target, upper_bound))

        ppo_rr_multiplier = float(signal.get('ppo_rr_multiplier', 1.0))
        final_target *= max(0.1, ppo_rr_multiplier)
        
        # Detailed logging
        logger.info(
            f"📊 [Dynamic R/R Calc] Base={base_rr:.2f} - Relax={relaxation:.2f} + Tight={tightening:.2f} "
            f"× Regime({regime_name}, mult={regime_mult:.1f}, weight={regime_weight:.2f})={regime_adjustment:.2f} "
            f"= {dynamic_target:.2f} × PPO({ppo_rr_multiplier:.2f}) → Final={final_target:.2f}"
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
                logger.error(
                    f"[{self.rule_name}] PortfolioManager does not have get_todays_trade_count method! "
                    f"This is likely an integration issue. "
                    f"FAIL-SAFE MODE: Allowing trade (limit cannot be enforced)."
                )
                # Fail safely: if we can't check, don't block the trade
                # This prevents breaking production if PortfolioManager link is not established
                return (True, f"{self.rule_name}: cannot verify (missing method - fail-safe mode)")
            
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
