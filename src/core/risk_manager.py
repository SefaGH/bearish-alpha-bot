"""
Comprehensive Risk Management Engine.
Provides portfolio-level risk management, position validation, and capital allocation.

PHASE 3 REFACTOR: Transform into Rules Engine
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import numpy as np

# Import RiskConfiguration and related dataclasses for type-safe configuration
from config.risk_config import RiskConfiguration, ConcurrentRiskLimitsConfig

# PHASE 3: Import risk rules framework
try:
    from core.risk_rules import (
        BaseRiskRule,
        CapitalLimitRule,
        PositionSizeRule,
        PortfolioHeatRule,
        MaxDrawdownRule,
        RiskRewardRatioRule,
        StrategyPerformanceRule,
        DailyTradeLimitRule
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
            StrategyPerformanceRule,
            DailyTradeLimitRule
        )
    except ModuleNotFoundError:
        try:
            from ..risk_rules import (
                BaseRiskRule,
                CapitalLimitRule,
                PositionSizeRule,
                PortfolioHeatRule,
                MaxDrawdownRule,
                RiskRewardRatioRule,
                StrategyPerformanceRule,
                DailyTradeLimitRule
            )
        except ImportError:
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
    except ModuleNotFoundError:
        # Option 3: Relative import (package context)
        try:
            from ..utils.pnl_calculator import calculate_unrealized_pnl
        except ImportError:
            # Unable to import, re-raise
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
        self.concurrent_limits = self.risk_config.get_concurrent_limits() if hasattr(self.risk_config, 'get_concurrent_limits') else None
        self.volatility_sizing = self.risk_config.get_volatility_sizing() if hasattr(self.risk_config, 'get_volatility_sizing') else None
        
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
        self.config = self._extract_risk_config_dict()

        logger.info(f"RiskManager initialized (PHASE 3: Rules Engine)")
        logger.info(f"Risk configuration: {self.risk_config.to_dict()}")
        logger.info(f"Risk limits: {self.risk_limits}")
        logger.info(f"Active rules: {[rule.rule_name for rule in self.rules]}")

    def _extract_risk_config_dict(self) -> Dict[str, Any]:
        """Get the flattened risk section for consumers that expect raw config values."""
        try:
            if hasattr(self.risk_config, 'get_flat_risk_settings'):
                snapshot = self.risk_config.get_flat_risk_settings()
                if isinstance(snapshot, dict):
                    return snapshot
            risk_dict = self.risk_config.to_dict().get('risk') if hasattr(self.risk_config, 'to_dict') else None
            if isinstance(risk_dict, dict):
                return risk_dict
        except Exception as exc:
            logger.warning(f"[RISK-ENGINE] Unable to extract risk config snapshot: {exc}")
        return {}
    
    def _create_default_rules(self) -> List[BaseRiskRule]:
        """
        Create default set of risk rules based on configuration.
        
        Returns:
            List of risk rule instances
        """
        rules = [
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
            # Validate risk/reward ratio with dynamic intelligence
            RiskRewardRatioRule(config=self.risk_config),
            # Optional: Check strategy performance
            StrategyPerformanceRule(
                min_win_rate=0.40,
                performance_monitor=self.performance_monitor
            )
        ]
        
        # Add daily trade limit rule if configured
        daily_max_trades = self._get_daily_max_trades_from_config()
        if daily_max_trades is not None and daily_max_trades > 0:
            rules.append(DailyTradeLimitRule(max_daily_trades=daily_max_trades))
            logger.info(f"✅ Daily trade limit rule added: {daily_max_trades} trades/day")
        else:
            logger.info("ℹ️ Daily trade limit not configured or disabled")
        
        return rules
    
    def _get_daily_max_trades_from_config(self) -> Optional[int]:
        """
        Extract daily_max_trades from risk configuration.
        
        Returns:
            Daily max trades limit or None if not configured
        """
        try:
            # Try to get from risk_config if it's a RiskConfiguration object
            if hasattr(self.risk_config, 'to_dict'):
                config_dict = self.risk_config.to_dict()
                if 'daily_max_trades' in config_dict:
                    return int(config_dict['daily_max_trades'])
            
            # Try direct attribute access
            if hasattr(self.risk_config, 'daily_max_trades'):
                value = getattr(self.risk_config, 'daily_max_trades')
                if value is not None:
                    return int(value)
            
            # If risk_config is a dict, try direct key access
            if isinstance(self.risk_config, dict) and 'daily_max_trades' in self.risk_config:
                return int(self.risk_config['daily_max_trades'])
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get daily_max_trades from config: {e}")
            return None
    
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
            (pos.get('size', pos.get('amount', 0)) or 0) * pos.get('entry_price', 0)
            for pos in self.active_positions.values()
        )
        
        active_count = len(self.active_positions)
        portfolio_value = self.portfolio_value
        capital_utilization = (total_exposure / portfolio_value * 100) if portfolio_value > 0 else 0
        
        logger.debug(f"📊 [EXPOSURE] Active positions: {active_count}, Total exposure: ${total_exposure:.2f}, Capital utilization: {capital_utilization:.1f}%")
        
        return total_exposure

    def _check_concurrent_limits(
        self,
        signal: Dict,
        risk_metrics: Dict[str, Any],
        portfolio_manager
    ) -> Tuple[bool, str]:
        """Enforce concurrent exposure/position limits with intent-aware bypass.

        PHASE 3.2 UPDATE:
        - Close/reduce/reverse intents should NOT be blocked by
          max_open_positions / per-symbol limits or portfolio heat.
        - This prevents situations where risk guardrails accidentally
          trap the engine in an over-constrained state and block de-risking operations.
        """

        if not self.concurrent_limits or portfolio_manager is None:
            return True, "OK"

        limits = self.concurrent_limits
        active_positions: Optional[Dict[str, Dict[str, Any]]] = None
        if hasattr(portfolio_manager, 'count_open_positions'):
            active_count = portfolio_manager.count_open_positions()
        else:
            if hasattr(portfolio_manager, 'get_open_positions'):
                active_positions = portfolio_manager.get_open_positions() or {}
            elif isinstance(portfolio_manager, dict):
                active_positions = portfolio_manager.get('open_positions', {}) or {}
            else:
                active_positions = {}
            active_count = len(active_positions) if isinstance(active_positions, dict) else 0

        symbol = signal.get('symbol') if signal else None

        # Intent-aware bypass: never block de-risking operations
        intent = (signal or {}).get('intent') if isinstance(signal, dict) else None
        is_derisking_intent = intent in {"close", "reduce", "reverse", "force_swap"}

        if is_derisking_intent:
            logger.info(
                "[RISK-LIMITS] Bypassing concurrent limits for de-risking intent %s on %s",
                intent,
                symbol or "UNKNOWN",
            )
            return True, "Bypassed for de-risking intent"

        if limits.max_open_positions and active_count >= limits.max_open_positions:
            return False, f"Max open positions reached ({active_count}/{limits.max_open_positions})"

        symbol_limit_override_reason: Optional[str] = None
        if limits.max_positions_per_symbol and symbol:
            if hasattr(portfolio_manager, 'count_open_positions'):
                symbol_count = portfolio_manager.count_open_positions(symbol)
            else:
                active_positions = active_positions or {}
                symbol_count = self._count_positions_for_symbol(active_positions, symbol)
            if symbol_count >= limits.max_positions_per_symbol:
                can_scale, scale_reason = self._can_dynamic_scale(signal, portfolio_manager, symbol, symbol_count, limits)
                if can_scale:
                    symbol_limit_override_reason = scale_reason
                else:
                    return False, f"Max positions for {symbol} reached ({symbol_count}/{limits.max_positions_per_symbol})"

        projected_heat = risk_metrics.get('portfolio_heat') if isinstance(risk_metrics, dict) else None
        max_heat = limits.max_total_risk_pct
        if projected_heat is not None and max_heat and projected_heat >= max_heat:
            return False, f"Portfolio heat {projected_heat:.2%} exceeds limit {max_heat:.2%}"

        if symbol_limit_override_reason:
            return True, symbol_limit_override_reason
        return True, "OK"

    def _can_dynamic_scale(
        self,
        signal: Dict,
        portfolio_manager,
        symbol: Optional[str],
        symbol_count: int,
        limits: ConcurrentRiskLimitsConfig,
    ) -> Tuple[bool, str]:
        """Determine whether dynamic scaling rules permit exceeding per-symbol limits."""
        if not symbol or portfolio_manager is None:
            return False, ""

        concurrent_cfg = (self.config.get('concurrent_limits') or {}) if isinstance(self.config, dict) else {}
        scaling_cfg = concurrent_cfg.get('dynamic_scaling', {}) if isinstance(concurrent_cfg, dict) else {}
        if not scaling_cfg.get('enabled', True):
            return False, ""

        try:
            quality_score = float(signal.get('quality_score', 0) or 0)
        except (TypeError, ValueError):
            quality_score = 0.0
        # Support percentages expressed as 0-100
        if quality_score > 1:
            quality_score /= 100.0

        quality_threshold = float(scaling_cfg.get('quality_threshold', 0.80))
        pnl_threshold = float(scaling_cfg.get('min_unrealized_pnl_pct', 0.005))
        max_extra = int(scaling_cfg.get('max_additional_positions', 2))
        max_extra = max(0, max_extra)

        is_high_quality = quality_score >= quality_threshold
        if not is_high_quality:
            logger.info(
                "📉 [RISK-SCALING] Denied scale-in for %s | quality=%.2f < %.2f",
                symbol,
                quality_score,
                quality_threshold,
            )
            return False, ""

        if max_extra == 0:
            logger.info("📉 [RISK-SCALING] Dynamic scaling disabled via config (max_extra=0)")
            return False, ""

        positions: List[Dict[str, Any]] = []
        try:
            getter = getattr(portfolio_manager, 'get_open_positions_for_symbol', None)
            if callable(getter):
                positions = getter(symbol)
            elif hasattr(portfolio_manager, 'get_open_positions'):
                raw_positions = portfolio_manager.get_open_positions() or {}
                positions = [dict(position, position_id=pid) for pid, position in raw_positions.items() if position.get('symbol') == symbol]
        except Exception as exc:
            logger.debug(f"[RISK-SCALING] Unable to inspect open positions for {symbol}: {exc}")
            positions = []

        if not positions:
            logger.info(
                "📉 [RISK-SCALING] Cannot evaluate scale-in for %s (no open positions)",
                symbol,
            )
            return False, ""

        pnl_values = []
        for pos in positions:
            pnl_val = pos.get('unrealized_pnl_pct')
            if pnl_val is None:
                metrics = pos.get('metrics') or {}
                pnl_val = metrics.get('unrealized_pnl_pct')
            try:
                pnl_values.append(float(pnl_val or 0.0))
            except (TypeError, ValueError):
                pnl_values.append(0.0)

        avg_pnl_pct = sum(pnl_values) / len(pnl_values) if pnl_values else 0.0
        is_profitable = avg_pnl_pct >= pnl_threshold

        if not is_profitable:
            logger.info(
                "📉 [RISK-SCALING] Denied scale-in for %s | avgPnL=%.2f%% < %.2f%%",
                symbol,
                avg_pnl_pct * 100,
                pnl_threshold * 100,
            )
            return False, ""

        max_allowed = limits.max_positions_per_symbol + max_extra
        if max_allowed <= limits.max_positions_per_symbol:
            max_allowed = limits.max_positions_per_symbol

        if symbol_count >= max_allowed:
            logger.info(
                "📉 [RISK-SCALING] Denied scale-in for %s | slots=%d/%d",
                symbol,
                symbol_count,
                max_allowed,
            )
            return False, ""

        logger.info(
            "📈 [RISK-SCALING] Allowing scale-in for %s | quality=%.2f | avgPnL=%.2f%% | slots=%d/%d",
            symbol,
            quality_score,
            avg_pnl_pct * 100,
            symbol_count,
            max_allowed,
        )
        return True, "OK (Dynamic Scaling Allowed)"

    def can_open_new_position(
        self,
        signal: Dict,
        portfolio_manager,
        cached_metrics: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Evaluate whether a queued signal can still be opened safely.

        Args:
            signal: Signal payload from StrategyCoordinator.
            portfolio_manager: Live PortfolioManager instance.
            cached_metrics: Optional previously computed risk metrics to merge for telemetry.

        Returns:
            Tuple of (allowed flag, rejection reason, merged risk metrics snapshot).
        """
        snapshot: Dict[str, Any] = dict(cached_metrics) if isinstance(cached_metrics, dict) else {}

        if portfolio_manager is None:
            return False, "Portfolio manager unavailable", snapshot

        if signal is None:
            return False, "Signal payload missing", snapshot

        try:
            recalculated_metrics = self._calculate_risk_metrics(signal, portfolio_manager) or {}
            if snapshot:
                merged_metrics = snapshot.copy()
                merged_metrics.update(recalculated_metrics)
                risk_metrics = merged_metrics
            else:
                risk_metrics = recalculated_metrics

            limits_ok, limit_reason = self._check_concurrent_limits(signal, risk_metrics, portfolio_manager)
            if not limits_ok:
                return False, limit_reason, risk_metrics

            return True, "OK", risk_metrics

        except Exception as exc:
            logger.error(f"[RISK-ENGINE] Failed to evaluate dispatch gating: {exc}", exc_info=True)
            return False, f"Risk gating error: {exc}", snapshot

    def has_execution_capacity(self, signal: Dict, portfolio_manager) -> Tuple[bool, str]:
        """Backward-compatible wrapper for legacy callers."""
        allowed, reason, _ = self.can_open_new_position(signal, portfolio_manager)
        return allowed, reason

    @staticmethod
    def _count_positions_for_symbol(active_positions: Dict[str, Dict[str, Any]], symbol: str) -> int:
        if not isinstance(active_positions, dict) or not symbol:
            return 0
        return sum(1 for pos in active_positions.values() if pos.get('symbol') == symbol)
    
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
    
    async def validate_new_position(self, signal: Dict, portfolio_manager) -> Tuple[bool, str, Dict]:
        """
        Validate if new position meets risk criteria using the rules engine.
        
        PHASE 3.1 UPDATE: portfolio_manager is now a required argument.
        The deprecated fallback mode has been removed for safer, more reliable
        risk validation based on real-time portfolio state.

        Args:
            signal: Trading signal with entry, stop, size, etc.
            portfolio_manager: The active PortfolioManager instance. This is REQUIRED.
            
        Returns:
            Tuple of (is_valid, reason, risk_metrics)
        """
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            
            # ======================= ANA DÜZELTME =======================
            # 1. Fallback mekanizması kaldırıldı. portfolio_manager artık zorunlu.
            if portfolio_manager is None:
                # Bu durum artık bir hata olarak kabul ediliyor.
                logger.error("[RISK-ENGINE] CRITICAL: validate_new_position called without a PortfolioManager.")
                return (False, "Internal error: Risk validation requires a portfolio manager.", {})
            # ==========================================================

            logger.debug(f"🛡️ [RISK-ENGINE] Validating position for {symbol}")
            
            # Risk metriklerini hesapla
            risk_metrics = self._calculate_risk_metrics(signal, portfolio_manager)

            # Enrich signal for rule compatibility when tests supply dict portfolio managers
            if isinstance(signal, dict) and isinstance(risk_metrics, dict):
                signal.setdefault('portfolio_value', risk_metrics.get('portfolio_value', self.portfolio_value))
                signal.setdefault('current_exposure', risk_metrics.get('current_exposure', 0))
                signal.setdefault('current_drawdown', risk_metrics.get('current_drawdown', self.current_drawdown))
                new_position_value = risk_metrics.get('new_position_value')
                if new_position_value is not None:
                    signal.setdefault('notional', new_position_value)
                signal.setdefault('risk_amount', risk_metrics.get('risk_amount', 0))
            
            # PHASE 3: Tüm kuralları çalıştır
            for rule in self.rules:
                if not rule.enabled:
                    continue
                
                # Her kurala, gerçek portfolio_manager nesnesini ver
                is_valid, reason = rule.validate(signal, portfolio_manager)
                
                if not is_valid:
                    logger.warning(f"🚫 [RISK-ENGINE] Position REJECTED by {rule.rule_name}")
                    logger.warning(f"   Symbol: {symbol}")
                    logger.warning(f"   Reason: {reason}")
                    return (False, reason, risk_metrics)
                
                logger.debug(f"✅ [RISK-ENGINE] {rule.rule_name} passed")
            
            # Tüm kurallar geçti
            limits_ok, limit_reason = self._check_concurrent_limits(signal, risk_metrics, portfolio_manager)
            if not limits_ok:
                logger.warning(f"🚫 [RISK-ENGINE] Position blocked by concurrent limits: {limit_reason}")
                return (False, limit_reason, risk_metrics)

            logger.info(f"✅ [RISK-ENGINE] Position APPROVED for {symbol}")
            return (True, "All risk rules passed", risk_metrics)
            
        except Exception as e:
            logger.error(f"[RISK-ENGINE] Error validating position: {e}", exc_info=True)
            return (False, f"Validation error: {str(e)}", {})
    
    def _safe_get_equity(self, portfolio_manager) -> float:
        """
        Safely retrieve current equity with multiple fallback strategies.
        
        Args:
            portfolio_manager: PortfolioManager instance or dict
            
        Returns:
            float: Current equity value
        """
        try:
            # Primary: Try PortfolioManager method
            if hasattr(portfolio_manager, 'get_current_equity'):
                return float(portfolio_manager.get_current_equity())
            
            # Secondary: Try dict access
            if isinstance(portfolio_manager, dict):
                logger.warning(f"[RISK-ENGINE] Received dict instead of PortfolioManager. "
                             f"Using fallback dict access.")
                if 'equity_usd' in portfolio_manager:
                    return float(portfolio_manager.get('equity_usd', self.portfolio_value))
                if 'portfolio_value' in portfolio_manager:
                    return float(portfolio_manager.get('portfolio_value', self.portfolio_value))
                return float(self.portfolio_value)
            
            # Fallback: Use internal value
            logger.warning(f"[RISK-ENGINE] Invalid portfolio_manager type: {type(portfolio_manager)}. "
                          f"Using fallback value: {self.portfolio_value}")
            return float(self.portfolio_value)
            
        except Exception as e:
            logger.error(f"[RISK-ENGINE] Failed to get equity: {e}. Using fallback: {self.portfolio_value}")
            return float(self.portfolio_value)
    
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
            
            # FIX: Use safe equity getter
            portfolio_value = self._safe_get_equity(portfolio_manager)
            
            # Safe access for other portfolio methods
            if hasattr(portfolio_manager, 'get_total_exposure'):
                current_exposure = portfolio_manager.get_total_exposure()
            elif isinstance(portfolio_manager, dict):
                current_exposure = portfolio_manager.get('total_exposure', portfolio_manager.get('current_exposure', 0))
            else:
                current_exposure = 0
            
            if hasattr(portfolio_manager, 'get_open_positions'):
                active_positions = portfolio_manager.get_open_positions()
            elif isinstance(portfolio_manager, dict):
                active_positions = portfolio_manager.get('open_positions', portfolio_manager.get('active_positions', {}))
                if not isinstance(active_positions, dict):
                    active_positions = {}
            else:
                active_positions = {}

            if hasattr(portfolio_manager, 'get_current_drawdown'):
                current_drawdown = portfolio_manager.get_current_drawdown()
            elif isinstance(portfolio_manager, dict):
                current_drawdown = portfolio_manager.get('current_drawdown', self.current_drawdown)
            else:
                current_drawdown = self.current_drawdown
            
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
            total_risk = 0.0
            if isinstance(active_positions, dict):
                for pos in active_positions.values():
                    risk_val = pos.get('risk_amount')
                    if risk_val is None:
                        risk_val = pos.get('risk_usd')
                    if risk_val is None:
                        risk_val = 0.0
                    total_risk += float(risk_val)
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
                'total_risk_amount': total_risk + risk_amount,
                'max_risk_amount': portfolio_value * self.risk_limits['max_portfolio_risk'],
                'risk_pct': risk_pct,
                'risk_reward_ratio': risk_reward_ratio,
                'portfolio_heat': portfolio_heat,
                'current_drawdown': current_drawdown,
                'max_drawdown': self.risk_limits['max_drawdown'],
                'active_positions_count': len(active_positions)
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_stop_loss_from_signal(self, signal: Dict, entry_price: float) -> float:
        """
        Calculate stop loss with support for dynamic and static values.
        """
        side = signal.get('side', 'buy')
        
        # Priority 1: Signal'de hazır stop var mı?
        if signal.get('stop'):
            return signal['stop']
        
        # Priority 2: ATR-based dynamic stop
        if signal.get('sl_atr_mult') and signal.get('atr'):
            atr = signal['atr']
            sl_mult = signal['sl_atr_mult']
            
            if side in ['buy', 'long']:
                return entry_price - (atr * sl_mult)
            else:
                return entry_price + (atr * sl_mult)
        
        # Priority 3: Percentage-based stop
        if signal.get('sl_pct'):
            sl_pct = signal['sl_pct']
        else:
            # Config'den al - None olabilir!
            sl_pct = self.risk_limits_dataclass.stop_loss_pct
            
            if sl_pct is None:
                # Config'de de yok, strateji de vermemiş
                # Son çare: Çok konservatif bir default kullan
                logger.warning(f"No stop loss defined for signal, using emergency default 2%")
                sl_pct = 0.02
        
        if side in ['buy', 'long']:
            return entry_price * (1 - sl_pct)
        else:
            return entry_price * (1 + sl_pct)
    
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
                # FIX: Use safe equity getter
                portfolio_value = self._safe_get_equity(portfolio_manager)
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
                # FIX: Use safe equity getter
                portfolio_value = self._safe_get_equity(portfolio_manager)
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

            position_size, vol_meta = self._apply_volatility_sizing(
                position_size,
                signal,
                entry_price,
                portfolio_value,
                max_size_by_limit
            )

            if vol_meta:
                sizing_meta = signal.setdefault('sizing_meta', {})
                sizing_meta.update(vol_meta)
            
            logger.info(f"Calculated position size: {position_size:.4f} (risk: ${risk_per_trade:.2f})")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def _apply_volatility_sizing(
        self,
        base_size: float,
        signal: Dict,
        entry_price: float,
        portfolio_value: float,
        max_size_by_limit: float
    ) -> Tuple[float, Dict[str, Any]]:
        if not self.volatility_sizing or not getattr(self.volatility_sizing, 'enabled', False):
            return base_size, {}

        atr_value = signal.get('atr')
        if not atr_value:
            sizing_meta = signal.get('sizing_meta', {})
            atr_value = sizing_meta.get('atr') if isinstance(sizing_meta, dict) else None
        if not atr_value or entry_price <= 0:
            return base_size, {}

        atr_pct = float(atr_value) / entry_price
        cfg = self.volatility_sizing
        floor_pct = getattr(cfg, 'atr_floor_pct', 0.005)
        ceiling_pct = getattr(cfg, 'atr_ceiling_pct', 0.02)
        baseline = getattr(cfg, 'baseline_multiplier', 1.0)
        low_mult = getattr(cfg, 'low_vol_multiplier', 1.2)
        high_mult = getattr(cfg, 'high_vol_multiplier', 0.6)

        if atr_pct <= floor_pct:
            multiplier = low_mult
            bucket = 'low'
        elif atr_pct >= ceiling_pct:
            multiplier = high_mult
            bucket = 'high'
        else:
            bucket = 'medium'
            span = ceiling_pct - floor_pct
            progress = (atr_pct - floor_pct) / span if span > 0 else 0.5
            multiplier = baseline - progress * (baseline - high_mult)

        adjusted_size = max(base_size * multiplier, 0.0)
        adjusted_size = min(adjusted_size, max_size_by_limit)

        min_position_pct = getattr(cfg, 'min_position_size_pct', 0.0) or 0.0
        min_position_units = 0.0
        if min_position_pct > 0 and portfolio_value > 0:
            min_notional = portfolio_value * min_position_pct
            min_position_units = min_notional / entry_price
            adjusted_size = max(adjusted_size, min_position_units)

        meta = {
            'atr_pct': atr_pct,
            'volatility_bucket': bucket,
            'volatility_multiplier': multiplier,
            'min_position_units': min_position_units
        }
        return adjusted_size, meta
    
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
            # FIX: Use safe equity getter
            portfolio_value = self._safe_get_equity(portfolio_manager)
            peak_value = portfolio_manager.get_peak_equity() if hasattr(portfolio_manager, 'get_peak_equity') else portfolio_value
            current_drawdown = portfolio_manager.get_current_drawdown() if hasattr(portfolio_manager, 'get_current_drawdown') else 0.0
            active_positions = portfolio_manager.get_open_positions() if hasattr(portfolio_manager, 'get_open_positions') else {}
            total_exposure = portfolio_manager.get_total_exposure() if hasattr(portfolio_manager, 'get_total_exposure') else 0.0
            available_capital = portfolio_manager.get_available_capital() if hasattr(portfolio_manager, 'get_available_capital') else portfolio_value
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
