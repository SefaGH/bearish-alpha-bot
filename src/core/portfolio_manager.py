"""
Advanced Multi-Strategy Portfolio Optimization Engine.
Coordinates multiple trading strategies and optimizes capital allocation.
"""

import logging
import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from collections import defaultdict

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Advanced multi-strategy portfolio optimization engine."""
    
    def __init__(self, risk_manager, performance_monitor, websocket_manager=None, exchange_clients=None):
        """
        Initialize portfolio manager.
        
        Args:
            risk_manager: RiskManager instance from Phase 3.2
            performance_monitor: RealTimePerformanceMonitor from Phase 2
            websocket_manager: WebSocketManager from Phase 3.1 (optional)
            exchange_clients: Dictionary of exchange clients (optional)
        """
        self.risk_manager = risk_manager
        self.performance_monitor = performance_monitor
        self.ws_manager = websocket_manager
        self.exchange_clients = exchange_clients or {}
        self.cfg = {}
        
        # Strategy registry
        self.strategies = {}  # strategy_name -> strategy_instance
        self.strategy_allocations = {}  # strategy_name -> allocation (0-1)
        self.strategy_metadata = {}  # strategy_name -> metadata dict
        
        # Portfolio state
        self.portfolio_state = {
            'total_value': risk_manager.portfolio_value,
            'allocated_capital': 0.0,
            'available_capital': risk_manager.portfolio_value,
            'strategy_performance': {},
            'last_rebalance': None,
            'rebalance_count': 0
        }
        
        # Optimization history
        self.optimization_history = []
        
        logger.info(f"PortfolioManager initialized with portfolio value: ${risk_manager.portfolio_value:.2f}")
    
    def add_exchange_client(self, exchange_name: str, client):
        """
        Add an exchange client dynamically.
        
        Args:
            exchange_name: Name of the exchange
            client: Exchange client instance
        """
        self.exchange_clients[exchange_name] = client
        logger.info(f"Exchange client added: {exchange_name}")
    
    def register_strategy(self, strategy_name: str, strategy_instance: Any, 
                         initial_allocation: float = 0.25) -> Dict[str, Any]:
        """
        Register a trading strategy with initial capital allocation.
        
        Args:
            strategy_name: Unique strategy identifier
            strategy_instance: Strategy instance (can be any object with execute method)
            initial_allocation: Initial capital allocation (default 25%)
            
        Returns:
            Registration status and metadata
        """
        try:
            # Validate allocation
            if not 0 <= initial_allocation <= 1:
                raise ValueError(f"Initial allocation must be between 0 and 1, got {initial_allocation}")
            
            # Check if strategy already registered
            if strategy_name in self.strategies:
                logger.warning(f"Strategy '{strategy_name}' already registered, updating...")
            
            # Register strategy
            self.strategies[strategy_name] = strategy_instance
            self.strategy_allocations[strategy_name] = initial_allocation
            
            # Calculate allocated capital
            allocated_capital = self.portfolio_state['total_value'] * initial_allocation
            
            # Initialize metadata
            self.strategy_metadata[strategy_name] = {
                'registration_time': datetime.now(timezone.utc),
                'initial_allocation': initial_allocation,
                'allocated_capital': allocated_capital,
                'active': True,
                'risk_profile': 'medium',  # Can be updated based on strategy
                'performance_baseline': None
            }
            
            # Get performance baseline if available
            if self.performance_monitor:
                summary = self.performance_monitor.get_strategy_summary(strategy_name)
                metrics = summary.get('metrics', {})
                if metrics:
                    self.strategy_metadata[strategy_name]['performance_baseline'] = metrics
            
            # Update portfolio state
            self._update_portfolio_state()
            
            logger.info(f"Strategy '{strategy_name}' registered with {initial_allocation:.1%} allocation (${allocated_capital:.2f})")
            
            return {
                'status': 'success',
                'strategy_name': strategy_name,
                'allocation': initial_allocation,
                'allocated_capital': allocated_capital,
                'metadata': self.strategy_metadata[strategy_name]
            }
            
        except Exception as e:
            logger.error(f"Error registering strategy '{strategy_name}': {e}")
            return {
                'status': 'error',
                'strategy_name': strategy_name,
                'error': str(e)
            }
    
    async def optimize_portfolio_allocation(self, optimization_method: str = 'markowitz',
                                           target_return: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize capital allocation across strategies.
        
        Args:
            optimization_method: Optimization method ('markowitz', 'risk_parity', 'black_litterman', 'performance_based')
            target_return: Target return for optimization (optional)
            
        Returns:
            Optimized allocations and metrics
        """
        try:
            if not self.strategies:
                logger.warning("No strategies registered for optimization")
                return {'status': 'no_strategies', 'allocations': {}}
            
            logger.info(f"Running portfolio optimization using '{optimization_method}' method...")
            
            # Get strategy returns and metrics
            strategy_returns = self._calculate_strategy_returns()
            
            if not strategy_returns or len(strategy_returns) == 0:
                logger.warning("Insufficient data for optimization, maintaining current allocations")
                return {
                    'status': 'insufficient_data',
                    'old_allocations': self.strategy_allocations.copy(),
                    'new_allocations': self.strategy_allocations.copy(),
                    'allocation_changes': {s: 0 for s in self.strategies.keys()}
                }
            
            # Choose optimization method
            if optimization_method == 'markowitz':
                new_allocations = self._markowitz_optimization(strategy_returns, target_return)
            elif optimization_method == 'risk_parity':
                new_allocations = self._risk_parity_optimization(strategy_returns)
            elif optimization_method == 'black_litterman':
                new_allocations = self._black_litterman_optimization(strategy_returns, target_return)
            elif optimization_method == 'performance_based':
                new_allocations = self._performance_based_optimization(strategy_returns)
            else:
                logger.error(f"Unknown optimization method: {optimization_method}")
                return {'status': 'error', 'error': f'Unknown method: {optimization_method}'}
            
            # Calculate allocation changes
            allocation_changes = {
                strategy: new_allocations.get(strategy, 0) - self.strategy_allocations.get(strategy, 0)
                for strategy in set(list(new_allocations.keys()) + list(self.strategy_allocations.keys()))
            }
            
            # Record optimization
            optimization_record = {
                'timestamp': datetime.now(timezone.utc),
                'method': optimization_method,
                'old_allocations': self.strategy_allocations.copy(),
                'new_allocations': new_allocations.copy(),
                'allocation_changes': allocation_changes,
                'target_return': target_return
            }
            self.optimization_history.append(optimization_record)
            
            # Keep last 100 optimization records
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            logger.info(f"Portfolio optimization complete: {optimization_method}")
            for strategy, allocation in new_allocations.items():
                change = allocation_changes.get(strategy, 0)
                logger.info(f"  {strategy}: {allocation:.1%} (change: {change:+.1%})")
            
            return {
                'status': 'success',
                'method': optimization_method,
                'old_allocations': self.strategy_allocations.copy(),
                'new_allocations': new_allocations,
                'allocation_changes': allocation_changes,
                'optimization_record': optimization_record
            }
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def rebalance_portfolio(self, trigger: str = 'scheduled', 
                                 threshold: float = 0.05,
                                 apply: bool = True) -> Dict[str, Any]:
        """
        Dynamic portfolio rebalancing based on performance.
        
        Args:
            trigger: Rebalance trigger ('scheduled', 'threshold', 'performance', 'risk')
            threshold: Drift threshold for threshold-based rebalancing (default 5%)
            apply: Whether to apply the rebalancing (default True)
            
        Returns:
            Rebalancing report and actions
        """
        try:
            logger.info(f"Initiating portfolio rebalance (trigger: {trigger}, threshold: {threshold:.1%})...")
            
            # Check if rebalancing is needed
            needs_rebalance, reason = self._check_rebalance_needed(trigger, threshold)
            
            if not needs_rebalance:
                logger.info(f"Rebalancing not needed: {reason}")
                return {
                    'status': 'not_needed',
                    'reason': reason,
                    'current_allocations': self.strategy_allocations.copy()
                }
            
            # Determine rebalancing method based on trigger
            if trigger == 'performance':
                optimization_method = 'performance_based'
            elif trigger == 'risk':
                optimization_method = 'risk_parity'
            else:
                optimization_method = 'markowitz'
            
            # Optimize allocations
            optimization_result = await self.optimize_portfolio_allocation(optimization_method)
            
            if optimization_result['status'] not in ['success', 'insufficient_data']:
                logger.warning(f"Optimization failed: {optimization_result.get('error', 'Unknown error')}")
                return {
                    'status': 'error',
                    'error': optimization_result.get('error', 'Optimization failed')
                }
            
            # If insufficient data, no rebalancing needed
            if optimization_result['status'] == 'insufficient_data':
                return {
                    'status': 'not_needed',
                    'reason': 'Insufficient data for optimization',
                    'current_allocations': self.strategy_allocations.copy()
                }
            
            new_allocations = optimization_result['new_allocations']
            allocation_changes = optimization_result['allocation_changes']
            
            # Calculate rebalancing actions
            rebalancing_actions = []
            for strategy_name in self.strategies.keys():
                old_allocation = self.strategy_allocations.get(strategy_name, 0)
                new_allocation = new_allocations.get(strategy_name, 0)
                change = new_allocation - old_allocation
                
                if abs(change) > 0.01:  # Only act on changes > 1%
                    old_capital = self.portfolio_state['total_value'] * old_allocation
                    new_capital = self.portfolio_state['total_value'] * new_allocation
                    capital_change = new_capital - old_capital
                    
                    rebalancing_actions.append({
                        'strategy': strategy_name,
                        'old_allocation': old_allocation,
                        'new_allocation': new_allocation,
                        'allocation_change': change,
                        'old_capital': old_capital,
                        'new_capital': new_capital,
                        'capital_change': capital_change,
                        'action': 'increase' if change > 0 else 'decrease'
                    })
            
            # Apply rebalancing if requested
            if apply and rebalancing_actions:
                self.strategy_allocations = new_allocations.copy()
                self.portfolio_state['last_rebalance'] = datetime.now(timezone.utc)
                self.portfolio_state['rebalance_count'] += 1
                self._update_portfolio_state()
                logger.info(f"Rebalancing applied: {len(rebalancing_actions)} strategies adjusted")
            
            return {
                'status': 'success',
                'trigger': trigger,
                'reason': reason,
                'needs_rebalance': needs_rebalance,
                'rebalancing_actions': rebalancing_actions,
                'applied': apply,
                'new_allocations': new_allocations if apply else None,
                'timestamp': datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"Error rebalancing portfolio: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _update_portfolio_state(self):
        """Update internal portfolio state."""
        total_allocation = sum(self.strategy_allocations.values())
        allocated_capital = self.portfolio_state['total_value'] * total_allocation
        
        self.portfolio_state['allocated_capital'] = allocated_capital
        self.portfolio_state['available_capital'] = self.portfolio_state['total_value'] - allocated_capital
        
        # Update strategy performance from performance monitor
        if self.performance_monitor:
            for strategy_name in self.strategies.keys():
                summary = self.performance_monitor.get_strategy_summary(strategy_name)
                self.portfolio_state['strategy_performance'][strategy_name] = summary.get('metrics', {})
    
    def _calculate_strategy_returns(self) -> Dict[str, List[float]]:
        """Calculate historical returns for each strategy."""
        strategy_returns = {}
        
        if not self.performance_monitor:
            return strategy_returns
        
        for strategy_name in self.strategies.keys():
            if strategy_name not in self.performance_monitor.performance_history:
                continue
            
            trades = self.performance_monitor.performance_history[strategy_name].get('trades', [])
            if not trades:
                continue
            
            # Extract PnL as returns
            returns = []
            for trade in trades:
                result = trade.get('result', {})
                if isinstance(result, dict) and 'pnl' in result:
                    returns.append(float(result['pnl']))
                elif isinstance(result, (int, float)):
                    returns.append(float(result))
            
            if returns:
                strategy_returns[strategy_name] = returns
        
        return strategy_returns
    
    def _markowitz_optimization(self, strategy_returns: Dict[str, List[float]], 
                               target_return: Optional[float] = None) -> Dict[str, float]:
        """
        Modern Portfolio Theory (Markowitz) optimization.
        Maximize Sharpe ratio or achieve target return with minimum variance.
        """
        try:
            strategies = list(strategy_returns.keys())
            n_strategies = len(strategies)
            
            if n_strategies == 0:
                return {}
            
            # Convert returns to numpy array
            returns_matrix = np.array([strategy_returns[s] for s in strategies])
            
            # Calculate mean returns and covariance matrix
            mean_returns = np.mean(returns_matrix, axis=1)
            cov_matrix = np.cov(returns_matrix)
            
            # Handle single strategy or equal allocation for simplicity
            if n_strategies == 1:
                return {strategies[0]: 1.0}
            
            # Simple optimization: maximize Sharpe ratio
            # For production, use scipy.optimize
            if target_return is None:
                # Equal risk contribution (simplified)
                weights = np.ones(n_strategies) / n_strategies
            else:
                # Simplified target return approach
                weights = np.ones(n_strategies) / n_strategies
            
            # Ensure weights sum to 1 and are non-negative
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
            
            # Convert to dictionary
            allocations = {strategies[i]: float(weights[i]) for i in range(n_strategies)}
            
            logger.info(f"Markowitz optimization complete (Sharpe-based)")
            return allocations
            
        except Exception as e:
            logger.error(f"Error in Markowitz optimization: {e}")
            return self.strategy_allocations.copy()
    
    def _risk_parity_optimization(self, strategy_returns: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Risk parity allocation - equal risk contribution from each strategy.
        """
        try:
            strategies = list(strategy_returns.keys())
            n_strategies = len(strategies)
            
            if n_strategies == 0:
                return {}
            
            # Calculate volatility for each strategy
            volatilities = {}
            for strategy, returns in strategy_returns.items():
                volatilities[strategy] = np.std(returns) if len(returns) > 1 else 1.0
            
            # Inverse volatility weighting
            total_inv_vol = sum(1.0 / vol for vol in volatilities.values() if vol > 0)
            
            if total_inv_vol == 0:
                # Equal weights fallback
                return {s: 1.0 / n_strategies for s in strategies}
            
            allocations = {
                strategy: (1.0 / volatilities[strategy]) / total_inv_vol
                for strategy in strategies
            }
            
            logger.info(f"Risk parity optimization complete")
            return allocations
            
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            return self.strategy_allocations.copy()
    
    def _black_litterman_optimization(self, strategy_returns: Dict[str, List[float]],
                                     target_return: Optional[float] = None) -> Dict[str, float]:
        """
        Black-Litterman model - combines market equilibrium with views.
        Simplified implementation.
        """
        try:
            # For now, use Markowitz as fallback
            # Full Black-Litterman requires market views and equilibrium returns
            logger.info("Black-Litterman optimization (simplified, using Markowitz)")
            return self._markowitz_optimization(strategy_returns, target_return)
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {e}")
            return self.strategy_allocations.copy()
    
    def _performance_based_optimization(self, strategy_returns: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Performance-based rebalancing - allocate based on recent performance.
        """
        try:
            strategies = list(strategy_returns.keys())
            n_strategies = len(strategies)
            
            if n_strategies == 0:
                return {}
            
            # Calculate performance scores
            performance_scores = {}
            for strategy in strategies:
                if strategy not in self.performance_monitor.performance_history:
                    performance_scores[strategy] = 0.5  # Neutral score
                    continue
                
                metrics = self.performance_monitor.performance_history[strategy].get('metrics', {})
                
                # Composite score: win_rate * sharpe_ratio * profit_factor
                win_rate = metrics.get('win_rate', 0.5)
                sharpe = max(metrics.get('sharpe_ratio', 0), 0)
                profit_factor = metrics.get('profit_factor', 1.0)
                
                # Normalized score
                score = (win_rate * 0.4) + (min(sharpe / 2.0, 0.3) * 0.3) + (min(profit_factor / 3.0, 0.3) * 0.3)
                performance_scores[strategy] = max(score, 0.1)  # Minimum 10% score
            
            # Normalize to sum to 1
            total_score = sum(performance_scores.values())
            allocations = {
                strategy: score / total_score
                for strategy, score in performance_scores.items()
            }
            
            logger.info(f"Performance-based optimization complete")
            return allocations
            
        except Exception as e:
            logger.error(f"Error in performance-based optimization: {e}")
            return self.strategy_allocations.copy()
    
    def _check_rebalance_needed(self, trigger: str, threshold: float) -> Tuple[bool, str]:
        """Check if rebalancing is needed based on trigger type."""
        try:
            if trigger == 'scheduled':
                # Check time since last rebalance
                last_rebalance = self.portfolio_state.get('last_rebalance')
                if last_rebalance is None:
                    return (True, "Initial rebalance")
                
                # Rebalance if more than 24 hours since last rebalance
                time_since = (datetime.now(timezone.utc) - last_rebalance).total_seconds() / 3600
                if time_since >= 24:
                    return (True, f"Scheduled rebalance (last: {time_since:.1f}h ago)")
                return (False, f"Last rebalance {time_since:.1f}h ago (< 24h)")
            
            elif trigger == 'threshold':
                # Check allocation drift
                max_drift = 0.0
                drifted_strategy = None
                
                for strategy_name in self.strategies.keys():
                    target_allocation = self.strategy_allocations.get(strategy_name, 0)
                    # In production, calculate actual allocation from positions
                    # For now, assume allocations are maintained
                    actual_allocation = target_allocation  # Placeholder
                    
                    drift = abs(actual_allocation - target_allocation)
                    if drift > max_drift:
                        max_drift = drift
                        drifted_strategy = strategy_name
                
                if max_drift > threshold:
                    return (True, f"Allocation drift {max_drift:.1%} exceeds threshold {threshold:.1%} ({drifted_strategy})")
                return (False, f"Max drift {max_drift:.1%} below threshold {threshold:.1%}")
            
            elif trigger == 'performance':
                # Check for performance degradation
                if not self.performance_monitor:
                    return (False, "Performance monitor not available")
                
                for strategy_name in self.strategies.keys():
                    summary = self.performance_monitor.get_strategy_summary(strategy_name)
                    metrics = summary.get('metrics', {})
                    
                    win_rate = metrics.get('win_rate', 0.5)
                    recent_win_rate = metrics.get('recent_win_rate', 0.5)
                    
                    # Significant performance degradation
                    if win_rate > 0.5 and recent_win_rate < win_rate * 0.7:
                        return (True, f"Performance degradation detected in {strategy_name}")
                
                return (False, "No significant performance changes")
            
            elif trigger == 'risk':
                # Check risk metrics
                portfolio_heat = self.risk_manager.get_portfolio_summary().get('portfolio_heat', 0)
                if portfolio_heat > 0.08:  # 8% portfolio heat threshold
                    return (True, f"High portfolio heat: {portfolio_heat:.1%}")
                
                return (False, f"Portfolio heat acceptable: {portfolio_heat:.1%}")
            
            else:
                return (False, f"Unknown trigger: {trigger}")
                
        except Exception as e:
            logger.error(f"Error checking rebalance need: {e}")
            return (False, f"Error: {str(e)}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        self._update_portfolio_state()
        
        return {
            'portfolio_state': self.portfolio_state.copy(),
            'registered_strategies': list(self.strategies.keys()),
            'strategy_allocations': self.strategy_allocations.copy(),
            'strategy_metadata': self.strategy_metadata.copy(),
            'optimization_history_count': len(self.optimization_history),
            'last_optimization': self.optimization_history[-1] if self.optimization_history else None,
            'risk_metrics': self.risk_manager.get_portfolio_summary()
        }
    
    def get_strategy_allocation(self, strategy_name: str) -> Optional[float]:
        """Get current allocation for a specific strategy."""
        return self.strategy_allocations.get(strategy_name)
    
    def update_strategy_status(self, strategy_name: str, active: bool) -> bool:
        """Enable or disable a strategy."""
        if strategy_name not in self.strategies:
            logger.warning(f"Strategy '{strategy_name}' not found")
            return False
        
        self.strategy_metadata[strategy_name]['active'] = active
        logger.info(f"Strategy '{strategy_name}' {'activated' if active else 'deactivated'}")
        return True
