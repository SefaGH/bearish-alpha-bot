"""
Comprehensive Risk Management Engine.
Provides portfolio-level risk management, position validation, and capital allocation.

PHASE 3 REFACTOR: Transform into Rules Engine
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple, Protocol
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

# Import RiskConfiguration and related dataclasses for type-safe configuration
from config.risk_config import RiskConfiguration, ConcurrentRiskLimitsConfig
from src.core.signal_intents import (
    INTENT_ENTRY,
    INTENT_FORCE_SWAP,
    INTENT_REDUCE,
    INTENT_REVERSE,
    INTENT_SCALE_IN,
    MAINTENANCE_INTENTS,
)

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
        DailyTradeLimitRule,
        compute_max_affordable_notional,
        compute_portfolio_open_risk_usd,
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
            DailyTradeLimitRule,
            compute_max_affordable_notional,
            compute_portfolio_open_risk_usd,
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
                DailyTradeLimitRule,
                compute_max_affordable_notional,
                compute_portfolio_open_risk_usd,
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


@dataclass
class PlannedSizeResult:
    planned_notional: float
    planned_qty: float
    capped_by_size_pct: bool
    capped_by_max_notional: bool
    capped_by_capital: bool
    capped_by_heat: bool
    below_min_notional: bool
    position_size_policy: Optional[str] = None
    reason: Optional[str] = None



class PositionPnlProviderProtocol(Protocol):
    """Lightweight PnL provider interface for scale-in gating."""

    def get_positions_for_symbol(
        self,
        symbol: str,
        strategy_name: Optional[str] = None,
        side: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        ...


class RiskManager:
    """Comprehensive risk management engine for multi-strategy portfolio."""
    
    def __init__(self, portfolio_value: float, risk_config: RiskConfiguration, 
                 websocket_manager=None, performance_monitor=None, rules: List[BaseRiskRule] = None,
                 pnl_provider: Optional[PositionPnlProviderProtocol] = None):
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
            'max_position_notional_usd': self.risk_limits_dataclass.max_position_notional_usd,
            'position_size_policy': self.risk_limits_dataclass.position_size_policy,
            'min_notional_threshold': self.risk_limits_dataclass.min_notional_threshold,
            'min_notional': self.risk_limits_dataclass.min_notional_threshold,
            'min_stop_pct': getattr(self.risk_limits_dataclass, 'min_stop_pct', 0.0),
            'max_risk_per_trade_usd': getattr(self.risk_config, 'max_risk_per_trade_usd', None),
        }
        self.concurrent_limits = self.risk_config.get_concurrent_limits() if hasattr(self.risk_config, 'get_concurrent_limits') else None
        self.volatility_sizing = self.risk_config.get_volatility_sizing() if hasattr(self.risk_config, 'get_volatility_sizing') else None

        # Flattened config snapshot for downstream consumers (used by rules)
        self.config = self._extract_risk_config_dict()

        # Compute planner flag once and log source for observability
        self._compute_size_planner_flag()
        logger.info(
            "[RISK-PLANNER] size_planner.flag",
            extra={
                'enabled': self._size_planner_enabled,
                'source': getattr(self, '_planner_flag_source', 'default'),
                'env_raw': getattr(self, '_planner_flag_details', {}).get('env_raw'),
                'risk_cfg_raw': getattr(self, '_planner_flag_details', {}).get('risk_cfg_raw'),
            },
        )
        
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

        # PnL provider for dynamic scaling (PositionManager-backed)
        self._pnl_provider: Optional[PositionPnlProviderProtocol] = pnl_provider

        # Track resize attempts per logical position to avoid repeated retries
        self._resize_attempted_for_position: Dict[str, bool] = {}

        # DCA bookkeeping
        self._dca_last_symbol_trigger: Dict[str, float] = defaultdict(float)

        logger.info(f"RiskManager initialized (PHASE 3: Rules Engine)")
        logger.info(f"Risk configuration: {self.risk_config.to_dict()}")
        logger.info(f"Risk limits: {self.risk_limits}")
        logger.info(f"Active rules: {[rule.rule_name for rule in self.rules]}")

    def set_pnl_provider(self, provider: PositionPnlProviderProtocol) -> None:
        """Inject or replace the PnL provider used for dynamic scaling decisions."""
        self._pnl_provider = provider

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
        risk_matrix = self.config.get('volume_bucket_risk_matrix') if isinstance(self.config, dict) else None

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
        ]

        if risk_matrix:
            from core.risk_rules import VolumeAwarePositionSizingRule
            rules.append(VolumeAwarePositionSizingRule(risk_matrix))

        rules.append(
            StrategyPerformanceRule(
                min_win_rate=0.40,
                performance_monitor=self.performance_monitor
            )
        )
        
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
        scale_profile = None
        dca_meta = {}
        if isinstance(signal, dict):
            dca_meta = signal.get("dca_metadata") or {}
            scale_profile = signal.get("scale_profile") or dca_meta.get("profile")
        is_dca_signal = scale_profile == "dca"

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
        intent = (signal or {}).get('intent', INTENT_ENTRY) if isinstance(signal, dict) else INTENT_ENTRY
        is_derisking_intent = intent in MAINTENANCE_INTENTS

        pyramiding_cfg = {}
        try:
            if portfolio_manager and hasattr(portfolio_manager, "cfg"):
                cfg_source = portfolio_manager.cfg or {}
                if isinstance(cfg_source, dict):
                    pyramiding_cfg = cfg_source.get("pyramiding", {}) or {}
        except Exception:
            pyramiding_cfg = {}
        pyramiding_enabled = bool(pyramiding_cfg.get("enabled", False))

        if is_derisking_intent:
            logger.info(
                "[RISK-LIMITS] Bypassing concurrent limits for de-risking intent %s on %s",
                intent,
                symbol or "UNKNOWN",
            )
            return True, "Bypassed for de-risking intent"

        # DCA branch (mutually exclusive with trend pyramiding in v1)
        if is_dca_signal and intent == INTENT_SCALE_IN:
            if limits.max_open_positions and active_count >= limits.max_open_positions:
                return False, f"Max open positions reached ({active_count}/{limits.max_open_positions})"
            if active_positions is None and hasattr(portfolio_manager, 'get_open_positions'):
                try:
                    active_positions = portfolio_manager.get_open_positions() or {}
                except Exception:
                    active_positions = {}
            ok_dca, dca_reason = self._check_dca_limits(
                signal=signal,
                portfolio_manager=portfolio_manager,
                active_positions=active_positions or {},
                pyramiding_cfg=pyramiding_cfg,
                concurrent_limits=limits,
            )
            if not ok_dca:
                return False, dca_reason

            projected_heat = risk_metrics.get('portfolio_heat') if isinstance(risk_metrics, dict) else None
            max_heat = limits.max_total_risk_pct
            if projected_heat is not None and max_heat and projected_heat >= max_heat:
                return False, f"Portfolio heat {projected_heat:.2%} exceeds limit {max_heat:.2%}"
            return True, dca_reason or "OK (DCA)"

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
                can_scale, scale_reason = self._can_dynamic_scale(
                    signal,
                    portfolio_manager,
                    symbol,
                    symbol_count,
                    limits,
                    intent=intent,
                    pyramiding_cfg=pyramiding_cfg,
                )
                if can_scale:
                    symbol_limit_override_reason = scale_reason
                else:
                    if scale_reason:
                        return False, scale_reason
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
        intent: str = INTENT_ENTRY,
        pyramiding_cfg: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Determine whether dynamic scaling rules permit exceeding per-symbol limits."""
        if not symbol or portfolio_manager is None:
            return False, ""

        concurrent_cfg = (self.config.get('concurrent_limits') or {}) if isinstance(self.config, dict) else {}
        scaling_cfg = concurrent_cfg.get('dynamic_scaling', {}) if isinstance(concurrent_cfg, dict) else {}
        if not scaling_cfg.get('enabled', True):
            return False, ""

        pyramiding_enabled = bool(pyramiding_cfg.get("enabled", False)) if isinstance(pyramiding_cfg, dict) else False

        try:
            quality_score = float(signal.get('quality_score', 0) or 0)
        except (TypeError, ValueError):
            quality_score = 0.0
        # Support percentages expressed as 0-100
        if quality_score > 1:
            quality_score /= 100.0

        base_quality_threshold = float(scaling_cfg.get('quality_threshold', 0.80))
        base_pnl_threshold = float(scaling_cfg.get('min_unrealized_pnl_pct', 0.005))
        base_distance_pct = float(scaling_cfg.get('min_distance_pct', 0.005))
        max_extra = int(scaling_cfg.get('max_additional_positions', 2))
        max_extra = max(0, max_extra)

        eff_quality_threshold = base_quality_threshold
        eff_pnl_threshold = base_pnl_threshold
        eff_distance_pct = base_distance_pct
        max_layers = None

        if pyramiding_enabled and intent == INTENT_SCALE_IN and isinstance(pyramiding_cfg, dict):
            eff_quality_threshold = float(pyramiding_cfg.get('min_scale_in_quality', base_quality_threshold))
            eff_pnl_threshold = float(pyramiding_cfg.get('min_scale_in_unrealized_pnl_pct', base_pnl_threshold))
            eff_distance_pct = float(pyramiding_cfg.get('min_scale_in_distance_pct', base_distance_pct))
            try:
                max_layers = int(pyramiding_cfg.get('max_layers_per_symbol'))
                if max_layers is not None and max_layers < 1:
                    max_layers = None
            except Exception:
                max_layers = None

        is_high_quality = quality_score >= eff_quality_threshold
        if not is_high_quality:
            logger.info(
                "📉 [RISK-SCALING] Denied scale-in for %s | quality=%.2f < %.2f",
                symbol,
                quality_score,
                eff_quality_threshold,
            )
            return False, "scale_in_quality_below_threshold"

        if max_extra == 0:
            logger.info("📉 [RISK-SCALING] Dynamic scaling disabled via config (max_extra=0)")
            return False, ""

        pnl_provider = getattr(self, "_pnl_provider", None)
        if pnl_provider is None:
            logger.warning("📉 [RISK-SCALING] PnL provider unavailable for %s", symbol or "UNKNOWN")
            return False, "scale_in_pnl_data_unavailable"

        try:
            positions: List[Dict[str, Any]] = pnl_provider.get_positions_for_symbol(
                symbol,
                strategy_name=signal.get("strategy_name") if isinstance(signal, dict) else None,
                side=signal.get("side") if isinstance(signal, dict) else None,
            )
        except Exception as exc:
            logger.warning(f"📉 [RISK-SCALING] Failed to read PnL from provider for {symbol}: {exc}")
            return False, "scale_in_pnl_data_unavailable"

        if not positions:
            logger.warning("📉 [RISK-SCALING] No PnL data available for %s", symbol)
            return False, "scale_in_pnl_data_unavailable"

        pnl_values = []
        last_entry_price = None
        if positions:
            last_position = max(
                positions,
                key=lambda pos: pos.get('entry_time') or pos.get('opened_at') or 0,
            )
            last_entry_candidate = (
                last_position.get('entry_price')
                or last_position.get('entry')
                or last_position.get('price')
            )
            try:
                last_entry_price = float(last_entry_candidate or 0.0)
            except (TypeError, ValueError):
                last_entry_price = None
        for pos in positions:
            pnl_val = pos.get('unrealized_pnl_pct')
            if pnl_val is None:
                metrics = pos.get('metrics') or {}
                pnl_val = metrics.get('unrealized_pnl_pct')
            try:
                if pnl_val is None:
                    continue
                pnl_values.append(float(pnl_val))
            except (TypeError, ValueError):
                continue

        if not pnl_values:
            logger.warning("📉 [RISK-SCALING] PnL data unavailable/invalid for %s (positions=%d)", symbol, len(positions))
            return False, "scale_in_pnl_data_unavailable"

        logger.info(
            "[RISK-SCALING-PNL] sym=%s | layers=%d | pnls=%s",
            symbol,
            len(pnl_values),
            [round(v * 100, 3) for v in pnl_values],
        )
        avg_pnl_pct = sum(pnl_values) / len(pnl_values) if pnl_values else 0.0
        is_profitable = avg_pnl_pct >= eff_pnl_threshold

        if not is_profitable:
            logger.info(
                "📉 [RISK-SCALING] Denied scale-in for %s | avgPnL=%.2f%% < %.2f%%",
                symbol,
                avg_pnl_pct * 100,
                eff_pnl_threshold * 100,
            )
            return False, "scale_in_pnl_below_threshold"

        current_entry = None
        if isinstance(signal, dict):
            current_entry = signal.get('entry')
            if not current_entry:
                current_entry = signal.get('entry_price') or signal.get('price')
        try:
            current_entry = float(current_entry or 0.0)
        except (TypeError, ValueError):
            current_entry = None

        price_diff_pct = None
        if (
            eff_distance_pct > 0
            and last_entry_price
            and current_entry
            and last_entry_price > 0
        ):
            price_diff_pct = abs(current_entry - last_entry_price) / last_entry_price
            if price_diff_pct < eff_distance_pct:
                logger.info(
                    "📉 [RISK-SCALING] Denied scale-in for %s | last=%.2f, new=%.2f, diff=%.2f%% < %.2f%%",
                    symbol,
                    last_entry_price,
                    current_entry,
                    price_diff_pct * 100,
                    eff_distance_pct * 100,
                )
                return False, "scale_in_distance_below_threshold"

        max_allowed = limits.max_positions_per_symbol + max_extra
        if max_allowed <= limits.max_positions_per_symbol:
            max_allowed = limits.max_positions_per_symbol
        if pyramiding_enabled and intent == INTENT_SCALE_IN and max_layers:
            max_allowed = min(max_allowed, max_layers)
            if max_allowed < limits.max_positions_per_symbol:
                max_allowed = limits.max_positions_per_symbol

        if symbol_count >= max_allowed:
            logger.info(
                "📉 [RISK-SCALING] Denied scale-in for %s | slots=%d/%d",
                symbol,
                symbol_count,
                max_allowed,
            )
            if pyramiding_enabled and intent == INTENT_SCALE_IN and max_layers:
                return False, "pyramiding_max_layers_reached"
            return False, ""

        if intent == INTENT_SCALE_IN and pyramiding_enabled:
            logger.info(
                "[PYRAMID] scale-in allowed | sym=%s | slots=%d/%d | quality=%.2f/%.2f | avgPnL=%.2f%%/%.2f%% | dist=%s/%s",
                symbol,
                symbol_count,
                max_allowed,
                quality_score,
                eff_quality_threshold,
                avg_pnl_pct * 100,
                eff_pnl_threshold * 100,
                f"{price_diff_pct*100:.2f}%" if price_diff_pct is not None else "n/a",
                f"{eff_distance_pct*100:.2f}%",
            )
        else:
            logger.info(
                "📈 [RISK-SCALING] Allowing scale-in for %s | quality=%.2f | avgPnL=%.2f%% | slots=%d/%d",
                symbol,
                quality_score,
                avg_pnl_pct * 100,
                symbol_count,
                max_allowed,
            )
        return True, "OK (Dynamic Scaling Allowed)"

    def _check_dca_limits(
        self,
        signal: Dict[str, Any],
        portfolio_manager: Any,
        active_positions: Dict[str, Dict[str, Any]],
        pyramiding_cfg: Dict[str, Any],
        concurrent_limits: ConcurrentRiskLimitsConfig,
    ) -> Tuple[bool, str]:
        """DCA-specific concurrent limit checks (mutually exclusive with trend in v1)."""
        import time

        dca_cfg = self._get_dca_cfg(portfolio_manager)
        if not dca_cfg or not dca_cfg.get("enabled"):
            return False, "dca_not_enabled"

        strategy_cfg = dca_cfg.get("strategy", {}) if isinstance(dca_cfg, dict) else {}
        risk_cfg = dca_cfg.get("risk_limits", {}) if isinstance(dca_cfg, dict) else {}
        allow_mix = bool(risk_cfg.get("allow_concurrent_with_trend", False))
        symbol = signal.get("symbol")

        if not allow_mix and self._has_trend_pyramiding_position(symbol, active_positions):
            return False, "dca_trend_mutually_exclusive"

        max_layers_cfg = strategy_cfg.get("max_layers", 0)
        try:
            max_layers_cfg = int(max_layers_cfg)
        except (TypeError, ValueError):
            max_layers_cfg = 0
        # max_layers includes the initial entry in the config; DCA layers are additional.
        allowed_dca_layers = max(0, max_layers_cfg - 1) if max_layers_cfg else max_layers_cfg
        current_dca_layers = self._count_dca_layers(active_positions, symbol)
        if allowed_dca_layers and current_dca_layers >= allowed_dca_layers:
            return False, f"dca_max_layers_reached:{allowed_dca_layers}"

        if concurrent_limits.max_positions_per_symbol and symbol:
            symbol_count = self._count_positions_for_symbol(active_positions, symbol)
            max_symbol_slots = concurrent_limits.max_positions_per_symbol + (allowed_dca_layers or 0)
            if symbol_count >= max_symbol_slots:
                return False, f"dca_symbol_limit:{symbol_count}/{max_symbol_slots}"

        equity = self._safe_get_equity(portfolio_manager)
        portfolio_dca_heat = self._calculate_portfolio_dca_heat(active_positions, equity)
        max_portfolio_heat = self._safe_float(risk_cfg.get("max_dca_portfolio_pct", 0.0), 0.0)
        if max_portfolio_heat and portfolio_dca_heat > max_portfolio_heat:
            return False, "dca_portfolio_heat_limit"

        symbol_dca_heat = self._calculate_symbol_dca_heat(active_positions, symbol, equity)
        max_symbol_heat = self._safe_float(risk_cfg.get("max_heat_per_symbol", 0.0), 0.0)
        if max_symbol_heat and symbol_dca_heat > max_symbol_heat:
            return False, "dca_symbol_heat_limit"

        panic_cutoff = self._safe_float(risk_cfg.get("panic_cutoff_pct", 0.0), 0.0)
        if panic_cutoff:
            drawdown_pct = self._calculate_symbol_drawdown(signal, active_positions, symbol)
            if drawdown_pct is not None and drawdown_pct > panic_cutoff:
                return False, "dca_panic_cutoff_triggered"

        cooldown_seconds = self._safe_float(strategy_cfg.get("cooldown_seconds", 0), 0.0)
        if cooldown_seconds > 0 and not self._is_dca_cooldown_passed(symbol, cooldown_seconds):
            return False, "dca_cooldown_not_passed"

        self._dca_last_symbol_trigger[symbol or ""] = time.time()
        return True, "OK (DCA)"

    def _get_dca_cfg(self, portfolio_manager: Any) -> Dict[str, Any]:
        if portfolio_manager and hasattr(portfolio_manager, "cfg"):
            cfg = getattr(portfolio_manager, "cfg", {}) or {}
            if isinstance(cfg, dict):
                return cfg.get("dca", {}) or {}
        return {}

    def _is_dca_position(self, position: Dict[str, Any]) -> bool:
        if not isinstance(position, dict):
            return False
        meta = position.get("dca_metadata") or {}
        profile = position.get("scale_profile") or meta.get("profile")
        return profile == "dca"

    def _count_dca_layers(self, active_positions: Dict[str, Dict[str, Any]], symbol: Optional[str]) -> int:
        if not isinstance(active_positions, dict) or not symbol:
            return 0
        layers = 0
        for pos in active_positions.values():
            if pos.get("symbol") != symbol:
                continue
            if self._is_dca_position(pos):
                layers += 1
        return layers

    def _extract_position_notional(self, position: Dict[str, Any]) -> float:
        if not position:
            return 0.0
        notional = position.get("notional") or position.get("position_notional")
        if notional is None:
            try:
                entry = float(position.get("entry_price") or position.get("entry") or 0.0)
                qty = float(position.get("amount") or position.get("size") or 0.0)
                notional = entry * qty
            except Exception:
                notional = 0.0
        try:
            return float(notional or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _calculate_portfolio_dca_heat(self, active_positions: Dict[str, Dict[str, Any]], equity: float) -> float:
        if equity is None or equity <= 0:
            return 0.0
        total = 0.0
        for pos in (active_positions or {}).values():
            if self._is_dca_position(pos):
                total += self._extract_position_notional(pos)
        return total / equity if equity else 0.0

    def _calculate_symbol_dca_heat(
        self,
        active_positions: Dict[str, Dict[str, Any]],
        symbol: Optional[str],
        equity: float,
    ) -> float:
        if equity is None or equity <= 0 or not symbol:
            return 0.0
        total = 0.0
        for pos in (active_positions or {}).values():
            if pos.get("symbol") != symbol:
                continue
            if self._is_dca_position(pos):
                total += self._extract_position_notional(pos)
        return total / equity if equity else 0.0

    def _calculate_symbol_drawdown(
        self,
        signal: Dict[str, Any],
        active_positions: Dict[str, Dict[str, Any]],
        symbol: Optional[str],
    ) -> Optional[float]:
        if not symbol:
            return None
        dca_meta = signal.get("dca_metadata") or {}
        price_drop_pct = dca_meta.get("price_drop_pct")
        try:
            if price_drop_pct is not None:
                drop = float(price_drop_pct)
                return max(0.0, drop)
        except (TypeError, ValueError):
            pass

        anchor_price = dca_meta.get("anchor_price")
        current_price = signal.get("entry") or signal.get("price") or signal.get("entry_price")
        direction = (signal.get("side") or "").lower()
        try:
            if anchor_price and current_price:
                anchor_val = float(anchor_price)
                current_val = float(current_price)
                if anchor_val > 0:
                    if direction in ("long", "buy"):
                        return max(0.0, (anchor_val - current_val) / anchor_val)
                    if direction in ("short", "sell"):
                        return max(0.0, (current_val - anchor_val) / anchor_val)
        except (TypeError, ValueError):
            pass

        # Fallback: use earliest position entry as anchor and latest price from positions
        anchor = None
        latest_price = None
        for pos in (active_positions or {}).values():
            if pos.get("symbol") != symbol:
                continue
            try:
                if anchor is None:
                    anchor = float(pos.get("entry_price") or pos.get("entry") or 0.0)
                price_candidate = pos.get("current_price") or pos.get("entry_price")
                if price_candidate is not None:
                    latest_price = float(price_candidate)
            except (TypeError, ValueError):
                continue
        if anchor and latest_price:
            if direction in ("short", "sell"):
                return max(0.0, (latest_price - anchor) / anchor)
            return max(0.0, (anchor - latest_price) / anchor)
        return None

    def _has_trend_pyramiding_position(self, symbol: Optional[str], active_positions: Dict[str, Dict[str, Any]]) -> bool:
        if not symbol or not isinstance(active_positions, dict):
            return False
        for pos in active_positions.values():
            if pos.get("symbol") != symbol:
                continue
            if not self._is_dca_position(pos):
                return True
        return False

    def _is_dca_cooldown_passed(self, symbol: Optional[str], cooldown_seconds: float) -> bool:
        if not symbol or cooldown_seconds <= 0:
            return True
        import time
        last_ts = self._dca_last_symbol_trigger.get(symbol, 0.0)
        now = time.time()
        return (now - last_ts) >= cooldown_seconds

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
    
    def set_risk_limits(self, max_portfolio_risk: float = 0.06, max_position_size: float = 0.10,
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

            planner_mode = 'active' if signal.get('planner_active') else 'inactive'
            if planner_mode == 'active':
                logger.info(
                    "[RISK-PLANNER] validate_path",
                    extra={
                        'symbol': symbol,
                        'mode': planner_mode,
                        'raw_notional': signal.get('planner_raw_notional'),
                        'planned_notional': signal.get('planner_planned_notional'),
                        'cap_flags': signal.get('planner_cap_flags', {}),
                    },
                )
            
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
                    if planner_mode == 'active' and getattr(rule, 'rule_name', '') == 'PositionSizeRule':
                        pos_value = signal.get('__position_size_rule_position_value')
                        if pos_value is None:
                            try:
                                pos_value = (signal.get('notional') or 0) or (signal.get('position_size', 0) * signal.get('entry', 0))
                            except Exception:
                                pos_value = None
                        logger.warning(
                            "[RISK-PLANNER] anomaly_position_size_rule",
                            extra={
                                'symbol': symbol,
                                'raw_notional': signal.get('planner_raw_notional'),
                                'planned_notional': signal.get('planner_planned_notional'),
                                'cap_flags': signal.get('planner_cap_flags', {}),
                                'position_value_seen_by_rule': pos_value,
                                'planner_mode': planner_mode,
                                'reason': reason,
                            },
                        )
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
        """Return best-effort equity reading without raising."""
        if portfolio_manager is None:
            return float(self.portfolio_value)

        try:
            if hasattr(portfolio_manager, 'get_total_equity'):
                return float(portfolio_manager.get_total_equity())
            if hasattr(portfolio_manager, 'portfolio_value'):
                return float(getattr(portfolio_manager, 'portfolio_value'))
            if hasattr(portfolio_manager, 'total_value'):
                return float(getattr(portfolio_manager, 'total_value'))
            if isinstance(portfolio_manager, dict):
                for key in ('equity_usd', 'portfolio_value', 'total_value'):
                    if key in portfolio_manager:
                        return float(portfolio_manager[key])
            logger.warning(
                f"[RISK-ENGINE] Unable to read equity from portfolio_manager type={type(portfolio_manager)}; using fallback"
            )
            return float(self.portfolio_value)
        except Exception as exc:
            logger.warning(
                f"[RISK-ENGINE] Failed to get equity from portfolio_manager: {exc}; using fallback {self.portfolio_value}"
            )
            return float(self.portfolio_value)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_bool(val: Any) -> Optional[bool]:
        if val is None:
            return None
        if isinstance(val, bool):
            return val
        try:
            sval = str(val).strip().lower()
        except Exception:
            return None
        if sval in ("1", "true", "yes", "on"):  # truthy
            return True
        if sval in ("0", "false", "no", "off"):  # falsy
            return False
        return None

    def _compute_size_planner_flag(self) -> bool:
        env_raw = os.getenv("RISK_SIZE_PLANNER_ENABLED")
        env_val = self._coerce_bool(env_raw)

        risk_cfg_raw = None
        risk_val = None
        try:
            if isinstance(self.config, dict):
                risk_cfg_raw = self.config.get('size_planner_enabled')
                risk_val = self._coerce_bool(risk_cfg_raw)
        except Exception:
            risk_cfg_raw = None
            risk_val = None

        if env_val is not None:
            enabled = env_val
            source = 'env'
        elif risk_val is not None:
            enabled = risk_val
            source = 'risk_config'
        else:
            enabled = False
            source = 'default'

        self._size_planner_enabled = enabled
        self._planner_flag_source = source
        self._planner_flag_details = {
            'env_raw': env_raw,
            'risk_cfg_raw': risk_cfg_raw,
            'source': source,
        }
        return enabled

    def _is_size_planner_enabled(self) -> bool:
        # Cached after __init__; recompute if missing
        if getattr(self, '_size_planner_enabled', None) is None:
            return self._compute_size_planner_flag()
        return self._size_planner_enabled

    def _log_planner_decision(self, symbol: str, raw_notional: float, planned: PlannedSizeResult,
                               max_portfolio_risk_usd: Optional[float], cap_heat: float, shadow: bool) -> None:
        notional_delta_abs = raw_notional - planned.planned_notional
        notional_delta_ratio = (planned.planned_notional / raw_notional) if raw_notional else 0.0
        logger.info(
            "[RISK-PLANNER] size_planner.decision",
            extra={
                'symbol': symbol,
                'raw_notional': raw_notional,
                'planned_notional': planned.planned_notional,
                'planned_qty': planned.planned_qty,
                'capped_by_size_pct': planned.capped_by_size_pct,
                'capped_by_max_notional': planned.capped_by_max_notional,
                'capped_by_capital': planned.capped_by_capital,
                'capped_by_heat': planned.capped_by_heat,
                'heat_remaining_usd': cap_heat if cap_heat != float('inf') else None,
                'max_portfolio_risk_usd': max_portfolio_risk_usd,
                'below_min_notional': planned.below_min_notional,
                'position_size_policy': planned.position_size_policy,
                'reason': planned.reason,
                'notional_delta_abs': notional_delta_abs,
                'notional_delta_ratio': notional_delta_ratio,
                'shadow_mode': shadow,
            }
        )

    def plan_position_size(
        self,
        *,
        raw_notional: float,
        symbol: str,
        equity: float,
        price: float,
        available_balance: float,
        leverage: float,
        risk_limits: Dict[str, Any],
        min_notional_threshold: float,
        max_portfolio_risk_usd: Optional[float],
        current_open_risk_usd: float,
        position_size_policy: str,
        stop_pct: Optional[float] = None,
    ) -> PlannedSizeResult:
        max_position_size_pct = float(risk_limits.get('max_position_size', 0) or 0)
        max_position_notional_usd = risk_limits.get('max_position_notional_usd')
        max_notional_pct_per_trade = risk_limits.get('max_notional_pct_per_trade') or risk_limits.get('computed_max_notional_pct_per_trade')

        cap_size_pct = equity * max_position_size_pct if max_position_size_pct else float('inf')
        cap_notional = float('inf')
        try:
            if max_position_notional_usd is not None:
                cap_notional = float(max_position_notional_usd)
            elif max_notional_pct_per_trade:
                cap_notional = equity * float(max_notional_pct_per_trade)
        except Exception:
            cap_notional = float('inf')

        cap_capital = compute_max_affordable_notional(available_balance or 0.0, leverage or 1.0)

        cap_heat_notional = float('inf')
        if max_portfolio_risk_usd is not None:
            heat_remaining_usd = max(0.0, max_portfolio_risk_usd - max(0.0, current_open_risk_usd))
            if heat_remaining_usd <= 0:
                cap_heat_notional = 0.0
            else:
                effective_stop_pct = None
                try:
                    effective_stop_pct = float(stop_pct) if stop_pct is not None else None
                except Exception:
                    effective_stop_pct = None

                min_stop_pct = float(risk_limits.get('min_stop_pct', 0) or 0.0)
                if effective_stop_pct is None or effective_stop_pct <= 0:
                    effective_stop_pct = min_stop_pct
                elif min_stop_pct and effective_stop_pct < min_stop_pct:
                    effective_stop_pct = min_stop_pct

                if effective_stop_pct and effective_stop_pct > 0:
                    cap_heat_notional = heat_remaining_usd / effective_stop_pct

        planned_notional = min(raw_notional, cap_size_pct, cap_notional, cap_capital, cap_heat_notional)

        capped_by_size_pct = planned_notional < raw_notional and planned_notional == cap_size_pct
        capped_by_max_notional = planned_notional < raw_notional and planned_notional == cap_notional
        capped_by_capital = planned_notional < raw_notional and planned_notional == cap_capital
        capped_by_heat = planned_notional < raw_notional and planned_notional == cap_heat_notional

        # position_size_policy handling
        reason = None
        if position_size_policy == 'reject' and (capped_by_size_pct or capped_by_max_notional):
            return PlannedSizeResult(
                planned_notional=0.0,
                planned_qty=0.0,
                capped_by_size_pct=capped_by_size_pct,
                capped_by_max_notional=capped_by_max_notional,
                capped_by_capital=capped_by_capital,
                capped_by_heat=capped_by_heat,
                below_min_notional=False,
                position_size_policy=position_size_policy,
                reason="REJECT_SIZE_CAP",
            )

        planned_qty = planned_notional / price if price > 0 else 0.0

        # min-notional enforcement
        if planned_notional < min_notional_threshold:
            if cap_heat_notional == planned_notional:
                reason = "portfolio_heat_exhausted"
                capped_by_heat = True
            else:
                reason = "REJECT_TOO_SMALL_AFTER_CAP"
            return PlannedSizeResult(
                planned_notional=planned_notional,
                planned_qty=planned_qty,
                capped_by_size_pct=capped_by_size_pct,
                capped_by_max_notional=capped_by_max_notional,
                capped_by_capital=capped_by_capital,
                capped_by_heat=capped_by_heat,
                below_min_notional=True,
                position_size_policy=position_size_policy,
                reason=reason,
            )

        return PlannedSizeResult(
            planned_notional=planned_notional,
            planned_qty=planned_qty,
            capped_by_size_pct=capped_by_size_pct,
            capped_by_max_notional=capped_by_max_notional,
            capped_by_capital=capped_by_capital,
            capped_by_heat=capped_by_heat,
            below_min_notional=False,
            position_size_policy=position_size_policy,
            reason=reason,
        )
    
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

    def apply_position_limits(self, signal: Dict, portfolio_manager=None) -> Tuple[float, Dict[str, Any]]:
        """Apply SSOT position limits and return the capped size plus metadata."""
        portfolio_value = self._safe_get_equity(portfolio_manager)
        entry_price = signal.get('entry', 0) or signal.get('entry_price', 0)

        proposed_size = signal.get('position_size')
        if proposed_size is None:
            proposed_size = signal.get('amount', 0) or 0
        proposed_notional = signal.get('notional')
        if proposed_notional is None:
            proposed_notional = proposed_size * entry_price if entry_price > 0 else 0

        max_by_pct = portfolio_value * self.risk_limits['max_position_size']
        max_abs_limit = self.risk_limits.get('max_position_notional_usd')
        if max_abs_limit is None:
            max_by_abs = float('inf')
        else:
            try:
                max_by_abs = float(max_abs_limit)
            except (TypeError, ValueError):
                logger.warning(f"[RISK-LIMITS] Invalid max_position_notional_usd={max_abs_limit}, treating as unlimited")
                max_by_abs = float('inf')
        min_notional = float(self.risk_limits.get('min_notional_threshold', 5.0) or 0)
        allowed_notional = min(max_by_pct, max_by_abs)
        policy = self.risk_limits.get('position_size_policy', 'clip')

        if proposed_notional > allowed_notional:
            if policy == 'clip':
                final_notional = allowed_notional
                action = 'clip'
                reason = f"Clipped from ${proposed_notional:.2f} to ${allowed_notional:.2f}"
            else:
                final_notional = 0.0
                action = 'reject'
                reason = f"Proposed ${proposed_notional:.2f} exceeds max ${allowed_notional:.2f}"
        else:
            final_notional = proposed_notional
            action = 'none'
            reason = 'Within limits'

        if final_notional > 0 and final_notional < min_notional:
            clipped_value = final_notional
            logger.warning(
                f"⚠️ Clipped notional ${clipped_value:.2f} below min ${min_notional:.2f} - rejecting"
            )
            final_notional = 0.0
            action = 'reject'
            if policy == 'clip':
                reason = f"Clipped value ${clipped_value:.2f} below exchange minimum ${min_notional:.2f}"
            else:
                reason = f"Proposed ${proposed_notional:.2f} below minimum notional ${min_notional:.2f}"

        final_size = final_notional / entry_price if entry_price > 0 and final_notional > 0 else 0.0

        limit_meta = {
            'action': action,
            'reason': reason,
            'proposed_notional': proposed_notional,
            'allowed_notional': allowed_notional,
            'final_notional': final_notional,
            'final_size': final_size,
            'max_by_pct': max_by_pct,
            'max_by_pct_ratio': self.risk_limits['max_position_size'],
            'max_by_abs': max_by_abs,
            'min_notional': min_notional,
            'policy': policy,
        }

        signal['limit_meta'] = limit_meta
        self._log_position_limit_decision(signal, limit_meta)
        return final_size, limit_meta

    def _log_position_limit_decision(self, signal: Dict, limit_meta: Dict[str, Any]) -> None:
        symbol = signal.get('symbol', 'UNKNOWN')
        action = limit_meta.get('action')
        reason = limit_meta.get('reason')
        if action == 'clip':
            logger.info(
                f"📉 [RISK-LIMITS] {symbol} clipped to ${limit_meta.get('final_notional', 0):.2f} ({reason})"
            )
        elif action == 'reject':
            logger.warning(
                f"🚫 [RISK-LIMITS] {symbol} rejected by position limits: {reason}"
            )
        else:
            logger.debug(
                f"✅ [RISK-LIMITS] {symbol} within position limits (notional=${limit_meta.get('final_notional', 0):.2f})"
            )

    async def size_and_validate_position(
        self,
        signal: Dict,
        portfolio_manager,
        sizing_engine: 'AdvancedPositionSizing' = None,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Unified API: size → apply limits → validate."""

        combined_meta: Dict[str, Any] = {}

        if portfolio_manager is None:
            logger.error("[RISK-ENGINE] size_and_validate_position requires a PortfolioManager instance")
            return False, 0.0, {'error': 'missing_portfolio_manager'}

        try:
            if sizing_engine:
                try:
                    signal = await sizing_engine.calculate_optimal_size(signal, return_signal=True)
                except ValueError as exc:
                    logger.warning(
                        "[RISK-ENGINE] Position sizing rejected trade",
                        extra={'symbol': signal.get('symbol'), 'error': str(exc)}
                    )
                    combined_meta['sizing_error'] = str(exc)
                    combined_meta['blocked_by'] = 'PositionSizing'
                    return False, 0.0, combined_meta
            combined_meta['sizing_meta'] = signal.get('sizing_meta', {})

            planner_enabled = self._is_size_planner_enabled()
            logger.info(
                "[RISK-PLANNER] size_planner.mode",
                extra={
                    'mode': 'active' if planner_enabled else 'shadow',
                    'enabled': planner_enabled,
                    'flag_source': getattr(self, '_planner_flag_source', 'unknown'),
                },
            )
            entry_price = signal.get('entry', 0) or signal.get('entry_price', 0)
            stop_pct = None
            try:
                stop_price = (
                    signal.get('stop')
                    or signal.get('stop_loss')
                    or signal.get('stop_loss_price')
                    or signal.get('stopPrice')
                    or signal.get('stopLossPrice')
                )
                if stop_price is None and entry_price:
                    stop_price = self._calculate_stop_loss_from_signal(signal, float(entry_price))
                if entry_price and stop_price:
                    stop_pct = abs(float(entry_price) - float(stop_price)) / float(entry_price)
            except Exception:
                stop_pct = None
            raw_notional = signal.get('notional')
            if raw_notional is None:
                proposed_size = signal.get('position_size') or signal.get('amount') or 0.0
                raw_notional = proposed_size * entry_price if entry_price else 0.0
            equity = self._safe_get_equity(portfolio_manager)
            available_balance = self._get_available_balance(portfolio_manager)
            leverage = signal.get('leverage', 1) or 1
            max_portfolio_risk_usd = self.config.get('max_portfolio_risk_usd') if isinstance(self.config, dict) else None
            if max_portfolio_risk_usd is None and equity is not None:
                try:
                    max_portfolio_risk_usd = equity * float(self.risk_limits.get('max_portfolio_risk', 0))
                except Exception:
                    max_portfolio_risk_usd = None
            current_open_risk_usd, _ = compute_portfolio_open_risk_usd(portfolio_manager, signal)
            cap_heat_value = max(0.0, (max_portfolio_risk_usd - current_open_risk_usd)) if max_portfolio_risk_usd is not None else float('inf')

            risk_limits_for_planner = dict(self.risk_limits)
            if isinstance(self.config, dict):
                if 'max_notional_pct_per_trade' in self.config:
                    risk_limits_for_planner['max_notional_pct_per_trade'] = self.config.get('max_notional_pct_per_trade')

            planner_result = self.plan_position_size(
                raw_notional=raw_notional,
                symbol=signal.get('symbol', 'UNKNOWN'),
                equity=equity,
                price=entry_price,
                available_balance=available_balance if available_balance is not None else 0.0,
                leverage=leverage,
                risk_limits=risk_limits_for_planner,
                min_notional_threshold=float(self.risk_limits.get('min_notional_threshold', 0) or 0.0),
                max_portfolio_risk_usd=max_portfolio_risk_usd,
                current_open_risk_usd=current_open_risk_usd,
                position_size_policy=self.risk_limits.get('position_size_policy', 'clip'),
                stop_pct=stop_pct,
            )

            # Telemetry helper: surface the cap components used by the planner to downstream consumers
            try:
                signal['planner_caps_snapshot'] = {
                    'max_notional_cap': cap_notional,
                    'max_size_pct_notional': cap_size_pct,
                    'heat_cap_notional': cap_heat_value,
                }
            except Exception:
                # Best-effort only; never block sizing
                signal['planner_caps_snapshot'] = None

            shadow_mode = not planner_enabled
            self._log_planner_decision(
                signal.get('symbol', 'UNKNOWN'),
                raw_notional,
                planner_result,
                max_portfolio_risk_usd,
                cap_heat_value,
                shadow_mode,
            )

            combined_meta['planner'] = planner_result
            combined_meta['planner_reason'] = planner_result.reason
            combined_meta['planner_raw_notional'] = raw_notional
            combined_meta['planner_delta_abs'] = raw_notional - planner_result.planned_notional
            combined_meta['planner_delta_ratio'] = (
                planner_result.planned_notional / raw_notional
            ) if raw_notional else 0.0

            if planner_enabled:
                if planner_result.below_min_notional or (planner_result.reason and planner_result.planned_notional == 0):
                    combined_meta['blocked_by'] = 'SizePlanner'
                    return False, 0.0, combined_meta

                final_size = planner_result.planned_qty
                signal['position_size'] = planner_result.planned_qty
                signal['amount'] = planner_result.planned_qty
                signal['notional'] = planner_result.planned_notional
                signal['planner_active'] = True
                signal['planner_cap_flags'] = {
                    'capped_by_size_pct': planner_result.capped_by_size_pct,
                    'capped_by_max_notional': planner_result.capped_by_max_notional,
                    'capped_by_capital': planner_result.capped_by_capital,
                    'capped_by_heat': planner_result.capped_by_heat,
                }
                signal['planner_raw_notional'] = raw_notional
                signal['planner_planned_notional'] = planner_result.planned_notional

                if 'sizing_meta' in signal:
                    signal['sizing_meta']['capped'] = any([
                        planner_result.capped_by_size_pct,
                        planner_result.capped_by_max_notional,
                        planner_result.capped_by_capital,
                        planner_result.capped_by_heat,
                    ])
                    signal['sizing_meta']['cap_applied_by'] = 'RiskManager.plan_position_size'
                    signal['sizing_meta']['final_notional'] = planner_result.planned_notional

            else:
                # Legacy Sprint1 path
                final_size, limit_meta = self.apply_position_limits(signal, portfolio_manager)
                combined_meta['limit_meta'] = limit_meta
                signal['limit_meta'] = limit_meta

                if limit_meta['action'] == 'reject':
                    return False, 0.0, combined_meta

                signal['position_size'] = final_size
                signal['amount'] = final_size
                signal['notional'] = limit_meta['final_notional']

                if 'sizing_meta' in signal:
                    signal['sizing_meta']['capped'] = limit_meta['action'] == 'clip'
                    signal['sizing_meta']['cap_applied_by'] = 'RiskManager.apply_position_limits'
                    signal['sizing_meta']['final_notional'] = limit_meta['final_notional']

            is_valid, reason, risk_metrics = await self.validate_new_position(signal, portfolio_manager)
            combined_meta['risk_metrics'] = risk_metrics
            combined_meta['validation_reason'] = reason

            if not is_valid:
                if self._should_attempt_resize(signal, reason):
                    resize_outcome = self._attempt_resize(signal, portfolio_manager)
                    if resize_outcome and resize_outcome.get('signal') is not None:
                        combined_meta['resize_meta'] = resize_outcome.get('meta', {})
                        resized_signal = resize_outcome['signal']

                        is_valid, reason, risk_metrics = await self.validate_new_position(resized_signal, portfolio_manager)
                        combined_meta['risk_metrics'] = risk_metrics
                        combined_meta['validation_reason'] = reason

                        if is_valid:
                            return True, resized_signal.get('position_size', 0.0), combined_meta
                    else:
                        combined_meta['resize_failed'] = True

                return False, 0.0, combined_meta

            return True, final_size, combined_meta

        except Exception as exc:
            logger.error(f"[RISK-ENGINE] size_and_validate_position failed: {exc}", exc_info=True)
            combined_meta['error'] = str(exc)
            return False, 0.0, combined_meta

    def _generate_position_key(self, signal: Dict) -> str:
        """Deterministic key for tracking resize attempts (no timestamps)."""
        symbol = signal.get('symbol', 'UNKNOWN')
        entry_price = signal.get('entry') or signal.get('entry_price') or 0
        stop_loss = signal.get('stop') or signal.get('stop_loss') or 0
        qty = signal.get('position_size') or signal.get('amount') or signal.get('size') or 0
        leverage = signal.get('leverage') or 0
        return f"{symbol}-{entry_price}-{stop_loss}-{qty}-{leverage}"

    def _should_attempt_resize(self, signal: Dict, reason: str) -> bool:
        """Attempt auto-resize only once per logical position and only for margin/capital errors."""
        position_key = self._generate_position_key(signal)
        if self._resize_attempted_for_position.get(position_key, False):
            return False

        if not reason:
            return False

        reason_lower = str(reason).lower()
        keywords = ('margin', 'balance', 'capital', 'available', 'affordable')
        return any(word in reason_lower for word in keywords)

    def _get_available_balance(self, portfolio_manager) -> Optional[float]:
        """Best-effort available balance reader for auto-resize."""
        try:
            if hasattr(portfolio_manager, 'get_available_balance'):
                return float(portfolio_manager.get_available_balance())
            if hasattr(portfolio_manager, 'get_available_capital'):
                return float(portfolio_manager.get_available_capital())
            equity = self._safe_get_equity(portfolio_manager)
            exposure = portfolio_manager.get_total_exposure() if hasattr(portfolio_manager, 'get_total_exposure') else 0.0
            return float(equity - exposure)
        except Exception as exc:
            logger.warning(f"[RISK-ENGINE] Failed to read available balance for auto-resize: {exc}")
            return None

    def _attempt_resize(self, signal: Dict, portfolio_manager) -> Optional[Dict[str, Any]]:
        """Try to shrink position to fit available margin/capital."""
        position_key = self._generate_position_key(signal)
        self._resize_attempted_for_position[position_key] = True

        entry_price = signal.get('entry', 0) or signal.get('entry_price', 0)
        leverage = signal.get('leverage', 1) or 1

        proposed_notional = signal.get('notional')
        if proposed_notional is None:
            proposed_size = signal.get('position_size') or signal.get('amount') or signal.get('size') or 0.0
            proposed_notional = proposed_size * entry_price

        available_balance = self._get_available_balance(portfolio_manager)
        if available_balance is None:
            return None

        safety_factor = 0.95
        max_affordable = available_balance * leverage * safety_factor

        max_notional_limit = self.risk_limits.get('max_position_notional_usd')
        try:
            if max_notional_limit is not None:
                max_affordable = min(max_affordable, float(max_notional_limit))
        except (TypeError, ValueError):
            logger.warning(f"[RISK-ENGINE] Invalid max_position_notional_usd={max_notional_limit}, skipping clamp")

        min_notional = float(self.risk_limits.get('min_notional_threshold', 0) or 0)

        if max_affordable <= 0 or max_affordable < min_notional:
            logger.warning(
                "[RISK-ENGINE] Auto-resize failed: affordable notional too low",
                extra={'max_affordable': max_affordable, 'min_notional': min_notional}
            )
            return None

        if proposed_notional is None or proposed_notional <= 0:
            return None

        if max_affordable >= proposed_notional:
            return None  # Nothing to do

        scale_factor = max_affordable / proposed_notional
        proposed_size = signal.get('position_size') or signal.get('amount') or signal.get('size') or 0.0
        new_size = proposed_size * scale_factor
        new_notional = new_size * entry_price

        resized_signal = dict(signal)
        resized_signal['position_size'] = new_size
        resized_signal['amount'] = new_size
        resized_signal['notional'] = new_notional

        final_size, limit_meta = self.apply_position_limits(resized_signal, portfolio_manager)
        resized_signal['position_size'] = final_size
        resized_signal['amount'] = final_size
        resized_signal['notional'] = limit_meta.get('final_notional', new_notional)

        if limit_meta.get('action') == 'reject' or final_size <= 0:
            return None

        meta = {
            'attempted': True,
            'position_key': position_key,
            'max_affordable': max_affordable,
            'available_balance': available_balance,
            'leverage': leverage,
            'safety_factor': safety_factor,
            'used_notional': limit_meta.get('final_notional', new_notional),
            'clipped_after_resize': limit_meta.get('action') == 'clip'
        }

        logger.warning(
            "[RISK-ENGINE] Position auto-resized due to capital limits",
            extra={
                'symbol': signal.get('symbol'),
                'original_notional': proposed_notional,
                'new_notional': meta['used_notional'],
                'available_balance': available_balance,
                'leverage': leverage
            }
        )

        return {'signal': resized_signal, 'meta': meta}
    
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
                logger.info(
                    "📊 [VOL-SIZING] ATR-Adj Applied | Multiplier: %.2fx | Bucket: %s | ATR/Price: %.2f%% | MinUnits: %.6f",
                    vol_meta.get('volatility_multiplier', 1.0),
                    vol_meta.get('volatility_bucket', 'unknown'),
                    (vol_meta.get('atr_pct', 0.0) or 0.0) * 100,
                    vol_meta.get('min_position_units', 0.0),
                )
            
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

    def run_health_check(self) -> Dict[str, Any]:
        """Lightweight internal health check for critical risk settings."""
        health = {
            'status': 'HEALTHY',
            'timestamp': datetime.now().isoformat(),
            'component': 'RiskManager',
            'checks': {}
        }

        critical_keys = [
            'max_position_notional_usd',
            'min_stop_pct',
            'min_notional_threshold',
            'max_risk_per_trade_usd',
            'max_position_size',
            'max_portfolio_risk'
        ]

        for key in critical_keys:
            value = self.risk_limits.get(key)
            if key == 'max_position_notional_usd':
                # Optional clamp: None means "no clamp" and is acceptable
                is_ok = value is None or (isinstance(value, (int, float)) and value > 0)
            else:
                is_ok = value is not None and isinstance(value, (int, float)) and value > 0

            health['checks'][f'config_{key}'] = {
                'ok': is_ok,
                'value': value,
                'required': True
            }

            if not is_ok:
                health['status'] = 'UNHEALTHY'

        deps = ['portfolio_value', 'performance_monitor', 'ws_manager']
        for dep in deps:
            exists = hasattr(self, dep)
            health['checks'][f'dep_{dep}'] = {
                'ok': exists,
                'detail': f"Attribute {dep} exists: {exists}"
            }
            if not exists:
                health['status'] = 'UNHEALTHY'

        if health['status'] == 'HEALTHY':
            logger.info("RiskManager health check PASSED", extra=health)
        else:
            logger.error("RiskManager health check FAILED", extra=health)

        return health
