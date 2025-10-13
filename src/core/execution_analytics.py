"""
Execution Quality Analytics.
Analyze and optimize trade execution quality.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
import statistics

logger = logging.getLogger(__name__)


class ExecutionAnalytics:
    """Analyze and optimize trade execution quality."""
    
    def __init__(self, order_manager, position_manager):
        """
        Initialize execution analytics.
        
        Args:
            order_manager: SmartOrderManager instance
            position_manager: AdvancedPositionManager instance
        """
        self.order_manager = order_manager
        self.position_manager = position_manager
        
        # Execution metrics tracking
        self.execution_metrics = {
            'total_trades': 0,
            'avg_slippage': 0.0,
            'avg_execution_time': 0.0,
            'fill_rates': [],
            'implementation_shortfalls': []
        }
        
        logger.info("ExecutionAnalytics initialized")
    
    def analyze_execution_quality(self, trade_id: str) -> Dict[str, Any]:
        """
        Analyze execution quality metrics for a specific trade.
        
        Args:
            trade_id: Trade identifier (order_id)
            
        Returns:
            Execution quality analysis
        """
        try:
            # Get order details
            order = self.order_manager.get_order_status(trade_id)
            
            if not order:
                return {'success': False, 'reason': 'Order not found'}
            
            # Calculate execution metrics
            metrics = {
                'trade_id': trade_id,
                'symbol': order.get('symbol'),
                'side': order.get('side'),
                'order_type': order.get('type'),
                'timestamp': order.get('created_at'),
            }
            
            # Slippage analysis
            if 'slippage' in order:
                metrics['slippage'] = order['slippage']
                metrics['slippage_bps'] = order['slippage'] * 10000  # basis points
                metrics['slippage_quality'] = self._assess_slippage_quality(order['slippage'])
            
            # Fill rate analysis
            if 'amount' in order and 'filled_amount' in order:
                fill_rate = order['filled_amount'] / order['amount'] if order['amount'] > 0 else 0
                metrics['fill_rate'] = fill_rate
                metrics['fill_quality'] = 'excellent' if fill_rate >= 0.95 else 'good' if fill_rate >= 0.80 else 'poor'
            
            # Execution speed
            if 'created_at' in order and 'filled_at' in order:
                execution_time = (order['filled_at'] - order['created_at']).total_seconds()
                metrics['execution_time_seconds'] = execution_time
                metrics['execution_speed'] = 'fast' if execution_time < 1 else 'normal' if execution_time < 5 else 'slow'
            
            # Price improvement (for limit orders)
            if order.get('type') == 'limit' and 'limit_price' in order and 'avg_fill_price' in order:
                limit_price = order['limit_price']
                fill_price = order['avg_fill_price']
                side = order.get('side')
                
                if side in ['buy', 'long']:
                    price_improvement = (limit_price - fill_price) / limit_price
                else:
                    price_improvement = (fill_price - limit_price) / limit_price
                
                metrics['price_improvement'] = price_improvement
                metrics['price_improvement_bps'] = price_improvement * 10000
            
            # Market impact assessment
            metrics['market_impact'] = self._assess_market_impact(order)
            
            logger.info(f"Execution quality analyzed for {trade_id}: slippage={metrics.get('slippage_bps', 0):.2f}bps")
            
            return {
                'success': True,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error analyzing execution quality: {e}")
            return {'success': False, 'reason': str(e)}
    
    def _assess_slippage_quality(self, slippage: float) -> str:
        """Assess slippage quality based on thresholds."""
        slippage_bps = slippage * 10000
        
        if slippage_bps < 5:
            return 'excellent'
        elif slippage_bps < 10:
            return 'good'
        elif slippage_bps < 25:
            return 'acceptable'
        else:
            return 'poor'
    
    def _assess_market_impact(self, order: Dict) -> Dict[str, Any]:
        """Assess market impact of order execution."""
        # Simplified market impact assessment
        amount = order.get('amount', 0)
        avg_price = order.get('avg_fill_price', 0)
        
        notional_value = amount * avg_price
        
        # Classify based on order size
        if notional_value < 1000:
            impact_level = 'minimal'
        elif notional_value < 10000:
            impact_level = 'low'
        elif notional_value < 50000:
            impact_level = 'moderate'
        else:
            impact_level = 'high'
        
        return {
            'impact_level': impact_level,
            'notional_value': notional_value,
            'assessment': f'{impact_level.capitalize()} market impact expected'
        }
    
    def calculate_implementation_shortfall(self, order_id: str) -> Dict[str, Any]:
        """
        Calculate implementation shortfall for execution.
        
        Implementation shortfall = Opportunity cost + Market impact + Timing cost
        
        Args:
            order_id: Order identifier
            
        Returns:
            Implementation shortfall analysis
        """
        try:
            order = self.order_manager.get_order_status(order_id)
            
            if not order:
                return {'success': False, 'reason': 'Order not found'}
            
            # Extract order details
            side = order.get('side')
            expected_price = order.get('expected_price', order.get('limit_price', 0))
            avg_fill_price = order.get('avg_fill_price', 0)
            amount = order.get('filled_amount', 0)
            
            if expected_price == 0 or avg_fill_price == 0:
                return {'success': False, 'reason': 'Insufficient price data'}
            
            # Calculate slippage cost
            if side in ['buy', 'long']:
                slippage_cost = (avg_fill_price - expected_price) * amount
            else:
                slippage_cost = (expected_price - avg_fill_price) * amount
            
            # Calculate as percentage of notional
            notional = expected_price * amount
            slippage_pct = (slippage_cost / notional) * 100 if notional > 0 else 0
            
            # Timing cost (simplified - based on execution time)
            timing_cost = 0.0
            if 'created_at' in order and 'filled_at' in order:
                execution_delay = (order['filled_at'] - order['created_at']).total_seconds()
                # Assume 0.01% cost per minute of delay
                timing_cost = notional * 0.0001 * (execution_delay / 60)
            
            # Total implementation shortfall
            total_shortfall = slippage_cost + timing_cost
            shortfall_pct = (total_shortfall / notional) * 100 if notional > 0 else 0
            
            result = {
                'order_id': order_id,
                'expected_price': expected_price,
                'avg_fill_price': avg_fill_price,
                'slippage_cost': slippage_cost,
                'timing_cost': timing_cost,
                'total_shortfall': total_shortfall,
                'shortfall_pct': shortfall_pct,
                'notional_value': notional,
                'efficiency_score': max(0, 100 - abs(shortfall_pct * 100))  # 0-100 score
            }
            
            logger.info(f"Implementation shortfall for {order_id}: {shortfall_pct:.4f}% (${total_shortfall:.2f})")
            
            return {
                'success': True,
                'analysis': result
            }
            
        except Exception as e:
            logger.error(f"Error calculating implementation shortfall: {e}")
            return {'success': False, 'reason': str(e)}
    
    def generate_execution_report(self, time_period: str = '1d') -> Dict[str, Any]:
        """
        Generate comprehensive execution quality report.
        
        Args:
            time_period: Time period for report ('1h', '1d', '1w', '1m')
            
        Returns:
            Execution quality report
        """
        try:
            # Calculate time window
            now = datetime.now(timezone.utc)
            if time_period == '1h':
                start_time = now - timedelta(hours=1)
            elif time_period == '1d':
                start_time = now - timedelta(days=1)
            elif time_period == '1w':
                start_time = now - timedelta(weeks=1)
            elif time_period == '1m':
                start_time = now - timedelta(days=30)
            else:
                start_time = now - timedelta(days=1)  # Default to 1 day
            
            # Get orders from order manager
            order_stats = self.order_manager.get_execution_statistics()
            
            # Get position summary
            position_summary = self.position_manager.get_position_summary()
            
            # Compile report
            report = {
                'report_period': time_period,
                'generated_at': now,
                'start_time': start_time,
                'execution_statistics': order_stats,
                'position_summary': position_summary,
            }
            
            # Calculate aggregate metrics
            if order_stats['total_orders'] > 0:
                report['performance_metrics'] = {
                    'success_rate': order_stats.get('success_rate', 0),
                    'avg_execution_time': order_stats.get('avg_execution_time', 0),
                    'total_orders': order_stats['total_orders'],
                    'successful_orders': order_stats['successful_orders'],
                    'failed_orders': order_stats['failed_orders'],
                }
                
                # Calculate slippage statistics
                if order_stats.get('total_slippage', 0) > 0 and order_stats['successful_orders'] > 0:
                    avg_slippage = order_stats['total_slippage'] / order_stats['successful_orders']
                    report['performance_metrics']['avg_slippage_bps'] = avg_slippage * 10000
            
            # Add position performance
            if position_summary['total_positions'] > 0:
                report['position_performance'] = {
                    'active_positions': position_summary['active_positions'],
                    'closed_positions': position_summary['closed_positions'],
                    'total_pnl': position_summary['total_pnl'],
                    'unrealized_pnl': position_summary['total_unrealized_pnl'],
                    'realized_pnl': position_summary['total_realized_pnl'],
                }
                
                # Calculate win rate for closed positions
                if position_summary['closed_positions'] > 0:
                    winning_positions = sum(
                        1 for p in self.position_manager.closed_positions 
                        if p.get('realized_pnl', 0) > 0
                    )
                    win_rate = winning_positions / position_summary['closed_positions']
                    report['position_performance']['win_rate'] = win_rate
            
            # Generate optimization recommendations
            report['recommendations'] = self._generate_recommendations(report)
            
            logger.info(f"Execution report generated for period: {time_period}")
            
            return {
                'success': True,
                'report': report
            }
            
        except Exception as e:
            logger.error(f"Error generating execution report: {e}")
            return {'success': False, 'reason': str(e)}
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate optimization recommendations based on report data."""
        recommendations = []
        
        perf_metrics = report.get('performance_metrics', {})
        
        # Success rate recommendations
        success_rate = perf_metrics.get('success_rate', 0)
        if success_rate < 0.90:
            recommendations.append(
                f"Low success rate ({success_rate:.1%}). Consider reviewing order validation logic."
            )
        
        # Slippage recommendations
        avg_slippage_bps = perf_metrics.get('avg_slippage_bps', 0)
        if avg_slippage_bps > 25:
            recommendations.append(
                f"High average slippage ({avg_slippage_bps:.1f}bps). Consider using limit orders or smaller order sizes."
            )
        
        # Execution time recommendations
        avg_exec_time = perf_metrics.get('avg_execution_time', 0)
        if avg_exec_time > 5:
            recommendations.append(
                f"Slow execution time ({avg_exec_time:.1f}s). Consider optimizing order routing."
            )
        
        # Position performance recommendations
        pos_perf = report.get('position_performance', {})
        if 'win_rate' in pos_perf:
            win_rate = pos_perf['win_rate']
            if win_rate < 0.40:
                recommendations.append(
                    f"Low win rate ({win_rate:.1%}). Review strategy signals and entry criteria."
                )
        
        if not recommendations:
            recommendations.append("Execution quality is good. Continue monitoring.")
        
        return recommendations
    
    def get_best_execution_algorithm(self, order_size: float, urgency: str = 'normal') -> str:
        """
        Recommend best execution algorithm based on order characteristics.
        
        Args:
            order_size: Order size in notional value
            urgency: Execution urgency ('high', 'normal', 'low')
            
        Returns:
            Recommended execution algorithm
        """
        # Large orders should use TWAP or Iceberg
        if order_size > 50000:
            if urgency == 'high':
                return 'iceberg'  # Faster than TWAP
            else:
                return 'twap'  # Minimize market impact
        
        # Medium orders
        elif order_size > 10000:
            if urgency == 'high':
                return 'market'
            else:
                return 'limit'
        
        # Small orders
        else:
            if urgency == 'high':
                return 'market'
            else:
                return 'limit'
