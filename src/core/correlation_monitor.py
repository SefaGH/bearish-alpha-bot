"""
Correlation and Diversification Monitoring.
Tracks position correlations and portfolio diversification metrics.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class CorrelationMonitor:
    """Monitor position correlations and diversification."""
    
    def __init__(self, websocket_manager=None):
        """
        Initialize correlation monitor.
        
        Args:
            websocket_manager: WebSocket manager for real-time price data
        """
        self.ws_manager = websocket_manager
        
        # Correlation data
        self.correlation_matrix = {}
        self.price_history = {}  # symbol -> deque of prices
        self.diversification_metrics = {}
        
        # Configuration
        self.lookback_period = 30  # days
        self.price_buffer_size = 500
        
        logger.info("CorrelationMonitor initialized")
    
    async def update_correlation_matrix(self, symbols: List[str], lookback_period: int = 30):
        """
        Update real-time correlation matrix.
        
        Args:
            symbols: List of symbols to analyze
            lookback_period: Number of periods for correlation calculation
        """
        try:
            self.lookback_period = lookback_period
            
            # Collect price series for each symbol
            price_series = {}
            
            for symbol in symbols:
                if symbol in self.price_history:
                    prices = list(self.price_history[symbol])
                    if len(prices) >= 2:
                        price_series[symbol] = [p['price'] for p in prices[-lookback_period:]]
            
            if len(price_series) < 2:
                logger.warning("Insufficient data for correlation calculation")
                return
            
            # Calculate returns
            returns_data = {}
            for symbol, prices in price_series.items():
                if len(prices) >= 2:
                    returns = np.diff(prices) / prices[:-1]
                    returns_data[symbol] = returns
            
            # Create returns DataFrame
            min_length = min(len(r) for r in returns_data.values())
            aligned_returns = {
                symbol: returns[-min_length:] 
                for symbol, returns in returns_data.items()
            }
            
            df_returns = pd.DataFrame(aligned_returns)
            
            # Calculate correlation matrix
            corr_matrix = df_returns.corr()
            
            # Store as dictionary
            self.correlation_matrix = {
                'matrix': corr_matrix.to_dict(),
                'symbols': list(corr_matrix.columns),
                'timestamp': datetime.now(timezone.utc),
                'lookback_period': lookback_period
            }
            
            logger.info(f"Correlation matrix updated for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error updating correlation matrix: {e}")
    
    def calculate_portfolio_diversification(self, positions: Dict[str, Dict]) -> Dict:
        """
        Calculate portfolio diversification metrics.
        
        Args:
            positions: Dictionary of active positions
            
        Returns:
            Dictionary with diversification metrics
        """
        try:
            if not positions:
                return {
                    'diversification_ratio': 0.0,
                    'effective_positions': 0,
                    'concentration_risk': 0.0
                }
            
            # Extract symbols and weights
            symbols = [pos.get('symbol') for pos in positions.values()]
            weights = []
            total_value = sum(pos.get('size', 0) * pos.get('entry_price', 0) 
                            for pos in positions.values())
            
            for pos in positions.values():
                pos_value = pos.get('size', 0) * pos.get('entry_price', 0)
                weight = pos_value / total_value if total_value > 0 else 0
                weights.append(weight)
            
            weights = np.array(weights)
            
            # Effective number of positions (Herfindahl index)
            # 1 / sum(weight^2)
            herfindahl = np.sum(weights ** 2) if len(weights) > 0 else 1.0
            effective_positions = 1 / herfindahl if herfindahl > 0 else 0
            
            # Concentration risk (max weight)
            concentration_risk = np.max(weights) if len(weights) > 0 else 0.0
            
            # Diversification ratio
            # Requires correlation matrix
            diversification_ratio = 1.0
            if self.correlation_matrix and len(symbols) > 1:
                corr_dict = self.correlation_matrix.get('matrix', {})
                
                # Calculate portfolio variance
                portfolio_var = 0.0
                for i, sym1 in enumerate(symbols):
                    for j, sym2 in enumerate(symbols):
                        if sym1 in corr_dict and sym2 in corr_dict.get(sym1, {}):
                            corr = corr_dict[sym1][sym2]
                            portfolio_var += weights[i] * weights[j] * corr
                
                # Individual variances (assuming correlation with itself = 1)
                individual_var = np.sum(weights ** 2)
                
                # Diversification ratio = sqrt(individual_var) / sqrt(portfolio_var)
                if portfolio_var > 0:
                    diversification_ratio = np.sqrt(individual_var) / np.sqrt(portfolio_var)
            
            metrics = {
                'diversification_ratio': float(diversification_ratio),
                'effective_positions': float(effective_positions),
                'concentration_risk': float(concentration_risk),
                'num_positions': len(positions),
                'herfindahl_index': float(herfindahl),
                'timestamp': datetime.now(timezone.utc)
            }
            
            self.diversification_metrics = metrics
            
            logger.debug(f"Diversification calculated: effective_positions={effective_positions:.2f}, "
                        f"concentration={concentration_risk:.2%}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating diversification: {e}")
            return {
                'diversification_ratio': 0.0,
                'effective_positions': 0,
                'concentration_risk': 0.0
            }
    
    async def validate_new_position_correlation(self, new_symbol: str, 
                                               existing_positions: Dict[str, Dict],
                                               max_correlation: float = 0.7) -> Tuple[bool, str, Dict]:
        """
        Validate correlation impact of new position.
        
        Args:
            new_symbol: Symbol of new position to add
            existing_positions: Currently active positions
            max_correlation: Maximum allowed correlation threshold
            
        Returns:
            Tuple of (is_valid, reason, correlation_data)
        """
        try:
            if not existing_positions:
                return (True, "No existing positions to correlate with", {})
            
            if not self.correlation_matrix:
                return (True, "No correlation data available", {})
            
            corr_dict = self.correlation_matrix.get('matrix', {})
            
            if new_symbol not in corr_dict:
                return (True, "New symbol not in correlation matrix", {})
            
            # Check correlations with existing positions
            high_correlations = []
            correlation_data = {}
            
            for pos_id, position in existing_positions.items():
                existing_symbol = position.get('symbol', '')
                
                if existing_symbol in corr_dict.get(new_symbol, {}):
                    correlation = corr_dict[new_symbol][existing_symbol]
                    correlation_data[existing_symbol] = correlation
                    
                    if abs(correlation) > max_correlation:
                        high_correlations.append({
                            'symbol': existing_symbol,
                            'correlation': correlation,
                            'position_id': pos_id
                        })
            
            if high_correlations:
                reason = f"High correlation detected with {len(high_correlations)} positions: "
                reason += ", ".join([f"{hc['symbol']}({hc['correlation']:.2f})" 
                                   for hc in high_correlations[:3]])
                return (False, reason, correlation_data)
            
            # Check portfolio concentration
            diversification = self.calculate_portfolio_diversification(existing_positions)
            concentration = diversification.get('concentration_risk', 0)
            
            if concentration > 0.5:  # More than 50% in single position
                return (False, f"High concentration risk: {concentration:.2%}", correlation_data)
            
            return (True, "Correlation validation passed", correlation_data)
            
        except Exception as e:
            logger.error(f"Error validating correlation: {e}")
            return (False, f"Validation error: {str(e)}", {})
    
    def get_correlation_alerts(self) -> List[Dict]:
        """
        Generate correlation-based risk alerts.
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        try:
            if not self.correlation_matrix:
                return alerts
            
            corr_dict = self.correlation_matrix.get('matrix', {})
            symbols = self.correlation_matrix.get('symbols', [])
            
            # Find high correlations
            high_corr_threshold = 0.8
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    if i >= j:  # Skip diagonal and duplicates
                        continue
                    
                    if sym1 in corr_dict and sym2 in corr_dict.get(sym1, {}):
                        corr = corr_dict[sym1][sym2]
                        
                        if abs(corr) > high_corr_threshold:
                            alerts.append({
                                'type': 'high_correlation',
                                'severity': 'medium',
                                'symbol1': sym1,
                                'symbol2': sym2,
                                'correlation': corr,
                                'message': f"High correlation between {sym1} and {sym2}: {corr:.2f}"
                            })
            
            # Check concentration risk
            if self.diversification_metrics:
                concentration = self.diversification_metrics.get('concentration_risk', 0)
                
                if concentration > 0.4:
                    alerts.append({
                        'type': 'concentration_risk',
                        'severity': 'medium',
                        'concentration': concentration,
                        'message': f"High concentration risk: {concentration:.2%}"
                    })
                
                # Low diversification
                effective_positions = self.diversification_metrics.get('effective_positions', 0)
                num_positions = self.diversification_metrics.get('num_positions', 0)
                
                if num_positions > 2 and effective_positions < num_positions * 0.5:
                    alerts.append({
                        'type': 'low_diversification',
                        'severity': 'low',
                        'effective_positions': effective_positions,
                        'num_positions': num_positions,
                        'message': f"Low diversification: {effective_positions:.1f} effective positions out of {num_positions}"
                    })
            
        except Exception as e:
            logger.error(f"Error generating correlation alerts: {e}")
        
        return alerts
    
    def update_price_history(self, symbol: str, price: float, timestamp: datetime = None):
        """
        Update price history for correlation calculations.
        
        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Price timestamp (default: now)
        """
        try:
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.price_buffer_size)
            
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)
            
            self.price_history[symbol].append({
                'price': price,
                'timestamp': timestamp
            })
            
        except Exception as e:
            logger.error(f"Error updating price history for {symbol}: {e}")
    
    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """
        Get correlation between two symbols.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            
        Returns:
            Correlation coefficient or None if not available
        """
        try:
            if not self.correlation_matrix:
                return None
            
            corr_dict = self.correlation_matrix.get('matrix', {})
            
            if symbol1 in corr_dict and symbol2 in corr_dict.get(symbol1, {}):
                return corr_dict[symbol1][symbol2]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting correlation: {e}")
            return None
