"""
Advanced Position Management System.
Comprehensive position lifecycle management.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """Position status enumeration."""
    OPEN = 'open'
    CLOSED = 'closed'
    PARTIALLY_CLOSED = 'partially_closed'
    PENDING_CLOSE = 'pending_close'


class ExitReason(Enum):
    """Position exit reason enumeration."""
    TAKE_PROFIT = 'take_profit'
    STOP_LOSS = 'stop_loss'
    TRAILING_STOP = 'trailing_stop'
    TIME_EXIT = 'time_exit'
    SIGNAL_EXIT = 'signal_exit'
    MANUAL = 'manual'
    EMERGENCY = 'emergency'


class AdvancedPositionManager:
    """Comprehensive position lifecycle management."""
    
    def __init__(self, portfolio_manager, risk_manager, websocket_manager=None):
        """
        Initialize position manager.
        
        Args:
            portfolio_manager: PortfolioManager instance from Phase 3.3
            risk_manager: RiskManager instance from Phase 3.2
            websocket_manager: WebSocketManager from Phase 3.1 (optional)
        """
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.ws_manager = websocket_manager
        
        # Position tracking
        self.positions = {}  # position_id -> position_data
        self.pnl_tracker = {}  # position_id -> pnl_history
        self.closed_positions = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        
        logger.info("AdvancedPositionManager initialized")
    
    async def open_position(self, signal: Dict, execution_result: Dict) -> Dict[str, Any]:
        """
        Initialize new position with full tracking.
        
        Args:
            signal: Trading signal that triggered the position
            execution_result: Order execution result
            
        Returns:
            Position initialization result
        """
        try:
            if not execution_result.get('success'):
                return {
                    'success': False,
                    'reason': 'Execution failed',
                    'position_id': None
                }
            
            # Generate position ID
            position_id = f"pos_{signal.get('symbol', 'UNKNOWN')}_{int(datetime.now(timezone.utc).timestamp())}"
            
            # Extract position details
            symbol = signal.get('symbol')
            side = signal.get('side', 'long')
            entry_price = execution_result.get('avg_price', 0)
            amount = execution_result.get('filled_amount', 0)
            
            # Calculate stop-loss and take-profit levels
            stop_loss = signal.get('stop', entry_price * 0.95)  # Default 5% stop
            take_profit = signal.get('target', entry_price * 1.05)  # Default 5% target
            
            # Create position record
            position = {
                'position_id': position_id,
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'current_price': entry_price,
                'amount': amount,
                'initial_amount': amount,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': PositionStatus.OPEN.value,
                'opened_at': datetime.now(timezone.utc),
                'strategy': signal.get('strategy', 'unknown'),
                'exchange': signal.get('exchange', 'unknown'),
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'trailing_stop_enabled': False,
                'trailing_stop_distance': 0.02,  # 2% trailing distance
                'highest_price': entry_price if side == 'long' else entry_price,
                'lowest_price': entry_price if side == 'short' else entry_price,
                'max_adverse_excursion': 0.0,
                'max_favorable_excursion': 0.0,
                'exit_reason': None,
            }
            
            # Register with risk manager
            self.risk_manager.register_position(position_id, position)
            
            # Store position
            self.positions[position_id] = position
            
            # Initialize P&L tracker
            self.pnl_tracker[position_id] = [{
                'timestamp': datetime.now(timezone.utc),
                'price': entry_price,
                'unrealized_pnl': 0.0,
                'pnl_pct': 0.0
            }]
            
            logger.info(f"Position opened: {position_id} - {symbol} {side} {amount} @ {entry_price:.4f}")
            logger.info(f"  Stop-loss: {stop_loss:.4f}, Take-profit: {take_profit:.4f}")
            
            return {
                'success': True,
                'position_id': position_id,
                'position': position
            }
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return {
                'success': False,
                'reason': str(e),
                'position_id': None
            }
    
    async def monitor_position_pnl(self, position_id: str, current_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Real-time P&L monitoring and alerting.
        
        Args:
            position_id: Position identifier
            current_price: Current market price (optional, will fetch if not provided)
            
        Returns:
            P&L monitoring result
        """
        try:
            if position_id not in self.positions:
                return {'success': False, 'reason': 'Position not found'}
            
            position = self.positions[position_id]
            
            # Update current price
            if current_price:
                position['current_price'] = current_price
            
            # Calculate unrealized P&L
            entry_price = position['entry_price']
            amount = position['amount']
            side = position['side']
            
            if side in ['long', 'buy']:
                unrealized_pnl = (position['current_price'] - entry_price) * amount
            else:  # short
                unrealized_pnl = (entry_price - position['current_price']) * amount
            
            position['unrealized_pnl'] = unrealized_pnl
            
            # Calculate P&L percentage
            position_value = entry_price * amount
            pnl_pct = (unrealized_pnl / position_value) * 100 if position_value > 0 else 0
            
            # Update max adverse/favorable excursion
            if unrealized_pnl < 0 and abs(unrealized_pnl) > abs(position['max_adverse_excursion']):
                position['max_adverse_excursion'] = unrealized_pnl
            elif unrealized_pnl > 0 and unrealized_pnl > position['max_favorable_excursion']:
                position['max_favorable_excursion'] = unrealized_pnl
            
            # Update highest/lowest price for trailing stop
            if side in ['long', 'buy']:
                if position['current_price'] > position['highest_price']:
                    position['highest_price'] = position['current_price']
                    # Update trailing stop if enabled
                    if position['trailing_stop_enabled']:
                        trailing_distance = position['trailing_stop_distance']
                        position['stop_loss'] = position['highest_price'] * (1 - trailing_distance)
            else:  # short
                if position['current_price'] < position['lowest_price']:
                    position['lowest_price'] = position['current_price']
                    if position['trailing_stop_enabled']:
                        trailing_distance = position['trailing_stop_distance']
                        position['stop_loss'] = position['lowest_price'] * (1 + trailing_distance)
            
            # Record P&L snapshot
            self.pnl_tracker[position_id].append({
                'timestamp': datetime.now(timezone.utc),
                'price': position['current_price'],
                'unrealized_pnl': unrealized_pnl,
                'pnl_pct': pnl_pct
            })
            
            # Check for exit conditions
            exit_signal = await self._check_exit_conditions(position)
            
            return {
                'success': True,
                'position_id': position_id,
                'unrealized_pnl': unrealized_pnl,
                'pnl_pct': pnl_pct,
                'current_price': position['current_price'],
                'exit_signal': exit_signal
            }
            
        except Exception as e:
            logger.error(f"Error monitoring position P&L: {e}")
            return {'success': False, 'reason': str(e)}
    
    async def _check_exit_conditions(self, position: Dict) -> Optional[Dict[str, Any]]:
        """Check if position should be exited based on conditions."""
        current_price = position['current_price']
        side = position['side']
        
        # Check stop-loss
        if side in ['long', 'buy']:
            if current_price <= position['stop_loss']:
                return {
                    'should_exit': True,
                    'reason': ExitReason.STOP_LOSS.value,
                    'price': current_price
                }
            # Check take-profit
            if current_price >= position['take_profit']:
                return {
                    'should_exit': True,
                    'reason': ExitReason.TAKE_PROFIT.value,
                    'price': current_price
                }
        else:  # short
            if current_price >= position['stop_loss']:
                return {
                    'should_exit': True,
                    'reason': ExitReason.STOP_LOSS.value,
                    'price': current_price
                }
            if current_price <= position['take_profit']:
                return {
                    'should_exit': True,
                    'reason': ExitReason.TAKE_PROFIT.value,
                    'price': current_price
                }
        
        return None
    
    async def close_position(self, position_id: str, exit_price: float, 
                           exit_reason: str = ExitReason.MANUAL.value) -> Dict[str, Any]:
        """
        Close position and finalize P&L.
        
        Args:
            position_id: Position identifier
            exit_price: Exit execution price
            exit_reason: Reason for exit
            
        Returns:
            Position close result
        """
        try:
            if position_id not in self.positions:
                return {'success': False, 'reason': 'Position not found'}
            
            position = self.positions[position_id]
            
            # Calculate final P&L
            entry_price = position['entry_price']
            amount = position['amount']
            side = position['side']
            
            if side in ['long', 'buy']:
                realized_pnl = (exit_price - entry_price) * amount
            else:  # short
                realized_pnl = (entry_price - exit_price) * amount
            
            position['realized_pnl'] = realized_pnl
            position['exit_price'] = exit_price
            position['exit_reason'] = exit_reason
            position['status'] = PositionStatus.CLOSED.value
            position['closed_at'] = datetime.now(timezone.utc)
            
            # Calculate return percentage
            position_value = entry_price * amount
            return_pct = (realized_pnl / position_value) * 100 if position_value > 0 else 0
            position['return_pct'] = return_pct
            
            # Close with risk manager
            self.risk_manager.close_position(position_id, exit_price, realized_pnl)
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[position_id]
            
            logger.info(f"Position closed: {position_id}")
            logger.info(f"  Entry: {entry_price:.4f}, Exit: {exit_price:.4f}")
            logger.info(f"  P&L: ${realized_pnl:.2f} ({return_pct:.2f}%)")
            logger.info(f"  Reason: {exit_reason}")
            
            return {
                'success': True,
                'position_id': position_id,
                'realized_pnl': realized_pnl,
                'return_pct': return_pct,
                'exit_reason': exit_reason
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'success': False, 'reason': str(e)}
    
    async def _get_current_price_from_ws(self, symbol: str) -> Optional[float]:
        """
        Get current price from WebSocket with API fallback.
        Phase 3.4 - Issue #100: Exit Monitoring
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None
        """
        try:
            # Try WebSocket first
            if self.ws_manager:
                try:
                    ticker = self.ws_manager.get_latest_ticker(symbol)
                    if ticker and 'last' in ticker:
                        return float(ticker['last'])
                except Exception as ws_error:
                    logger.debug(f"WebSocket price fetch failed for {symbol}: {ws_error}")
            
            # Fallback to API
            for ex_name, client in self.portfolio_manager.exchange_clients.items():
                try:
                    ticker = client.fetch_ticker(symbol)
                    last_price = ticker.get('last', ticker.get('close', 0))
                    if last_price > 0:
                        return float(last_price)
                except Exception as api_error:
                    logger.debug(f"API price fetch failed for {symbol} on {ex_name}: {api_error}")
            
            logger.warning(f"Could not fetch current price for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def _check_stop_loss_hit(self, position: Dict, current_price: float) -> bool:
        """Check if stop-loss is hit."""
        if current_price <= 0:
            logger.warning(f"_check_stop_loss_hit called with invalid current_price={current_price} for position_id={position.get('id', 'unknown')}. Data quality issue.")
            return False
        
        side = position['side']
        stop_loss = position['stop_loss']
        
        if side in ['long', 'buy']:
            return current_price <= stop_loss
        else:  # short
            return current_price >= stop_loss
    
    def _check_take_profit_hit(self, position: Dict, current_price: float) -> bool:
        """Check if take-profit is hit."""
        if current_price <= 0:
            return False
        
        side = position['side']
        take_profit = position['take_profit']
        
        if side in ['long', 'buy']:
            return current_price >= take_profit
        else:  # short
            return current_price <= take_profit
    
    def _check_timeout_exit(self, position: Dict) -> bool:
        """Check if position should exit due to timeout."""
        # Get configuration
        config = self.portfolio_manager.cfg if hasattr(self.portfolio_manager, 'cfg') else {}
        position_config = config.get('position_management', {}).get('time_based_exit', {})
        max_duration = position_config.get('max_position_duration', 3600)
        
        opened_at = position.get('opened_at')
        if not opened_at:
            return False
        
        current_time = datetime.now(timezone.utc)
        duration = (current_time - opened_at).total_seconds()
        
        return duration >= max_duration
    
    async def manage_position_exits(self, position_id: str) -> Dict[str, Any]:
        """
        Intelligent position exit management with priority checks.
        Phase 3.4 - Issue #100: Position Exit Monitoring
        
        Priority order: Stop Loss > Take Profit > Timeout
        
        Args:
            position_id: Position identifier
            
        Returns:
            Exit management result
        """
        try:
            if position_id not in self.positions:
                return {'success': False, 'reason': 'Position not found'}
            
            position = self.positions[position_id]
            
            # Get current price from WebSocket or API
            current_price = await self._get_current_price_from_ws(position['symbol'])
            
            if current_price is None or current_price <= 0:
                logger.warning(f"Could not get valid price for {position_id}, skipping exit check")
                return {
                    'success': True,
                    'should_exit': False,
                    'position_id': position_id,
                    'reason': 'price_unavailable'
                }
            
            # Update position's current price
            position['current_price'] = current_price
            
            # Priority 1: Check stop-loss
            if self._check_stop_loss_hit(position, current_price):
                logger.info(f"Stop-loss hit for {position_id} at {current_price}")
                # Auto-close position
                close_result = await self.close_position(position_id, current_price, ExitReason.STOP_LOSS.value)
                return {
                    'success': True,
                    'should_exit': True,
                    'position_id': position_id,
                    'exit_reason': ExitReason.STOP_LOSS.value,
                    'exit_price': current_price,
                    'close_result': close_result
                }
            
            # Priority 2: Check take-profit
            if self._check_take_profit_hit(position, current_price):
                logger.info(f"Take-profit hit for {position_id} at {current_price}")
                # Auto-close position
                close_result = await self.close_position(position_id, current_price, ExitReason.TAKE_PROFIT.value)
                return {
                    'success': True,
                    'should_exit': True,
                    'position_id': position_id,
                    'exit_reason': ExitReason.TAKE_PROFIT.value,
                    'exit_price': current_price,
                    'close_result': close_result
                }
            
            # Priority 3: Check timeout
            if self._check_timeout_exit(position):
                logger.info(f"Timeout exit for {position_id} at {current_price}")
                # Auto-close position
                close_result = await self.close_position(position_id, current_price, ExitReason.TIME_EXIT.value)
                return {
                    'success': True,
                    'should_exit': True,
                    'position_id': position_id,
                    'exit_reason': ExitReason.TIME_EXIT.value,
                    'exit_price': current_price,
                    'close_result': close_result
                }
            
            # No exit conditions met
            return {
                'success': True,
                'should_exit': False,
                'position_id': position_id,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"Error managing position exit: {e}")
            return {'success': False, 'reason': str(e)}
    
    def calculate_position_metrics(self, position_id: str) -> Dict[str, Any]:
        """
        Calculate comprehensive position metrics.
        
        Args:
            position_id: Position identifier
            
        Returns:
            Position metrics
        """
        try:
            # Check both active and closed positions
            position = self.positions.get(position_id)
            if not position:
                # Check closed positions
                position = next((p for p in self.closed_positions if p['position_id'] == position_id), None)
            
            if not position:
                return {'success': False, 'reason': 'Position not found'}
            
            # Basic metrics
            metrics = {
                'position_id': position_id,
                'symbol': position['symbol'],
                'side': position['side'],
                'entry_price': position['entry_price'],
                'current_price': position.get('current_price', position.get('exit_price', 0)),
                'amount': position['amount'],
                'status': position['status'],
            }
            
            # P&L metrics
            if position['status'] == PositionStatus.CLOSED.value:
                metrics['realized_pnl'] = position['realized_pnl']
                metrics['return_pct'] = position['return_pct']
                metrics['exit_reason'] = position['exit_reason']
                
                # Calculate holding period
                if 'opened_at' in position and 'closed_at' in position:
                    holding_period = position['closed_at'] - position['opened_at']
                    metrics['holding_period_seconds'] = holding_period.total_seconds()
                    metrics['holding_period_hours'] = holding_period.total_seconds() / 3600
            else:
                metrics['unrealized_pnl'] = position['unrealized_pnl']
                pnl_pct = (position['unrealized_pnl'] / (position['entry_price'] * position['amount'])) * 100
                metrics['unrealized_pnl_pct'] = pnl_pct
            
            # Risk metrics
            metrics['max_adverse_excursion'] = position['max_adverse_excursion']
            metrics['max_favorable_excursion'] = position['max_favorable_excursion']
            metrics['stop_loss'] = position['stop_loss']
            metrics['take_profit'] = position['take_profit']
            
            # Calculate risk-reward ratio
            if position['side'] in ['long', 'buy']:
                risk = position['entry_price'] - position['stop_loss']
                reward = position['take_profit'] - position['entry_price']
            else:
                risk = position['stop_loss'] - position['entry_price']
                reward = position['entry_price'] - position['take_profit']
            
            if risk > 0:
                metrics['risk_reward_ratio'] = reward / risk
            else:
                metrics['risk_reward_ratio'] = 0
            
            return {
                'success': True,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error calculating position metrics: {e}")
            return {'success': False, 'reason': str(e)}
    
    def enable_trailing_stop(self, position_id: str, trailing_distance: float = 0.02) -> Dict[str, Any]:
        """
        Enable trailing stop-loss for a position.
        
        Args:
            position_id: Position identifier
            trailing_distance: Trailing distance as decimal (e.g., 0.02 for 2%)
            
        Returns:
            Operation result
        """
        try:
            if position_id not in self.positions:
                return {'success': False, 'reason': 'Position not found'}
            
            position = self.positions[position_id]
            position['trailing_stop_enabled'] = True
            position['trailing_stop_distance'] = trailing_distance
            
            logger.info(f"Trailing stop enabled for {position_id} with {trailing_distance*100:.1f}% distance")
            
            return {'success': True, 'position_id': position_id}
            
        except Exception as e:
            logger.error(f"Error enabling trailing stop: {e}")
            return {'success': False, 'reason': str(e)}
    
    def get_all_positions(self) -> Dict[str, Any]:
        """Get all active positions."""
        return {
            'active_positions': list(self.positions.values()),
            'count': len(self.positions)
        }
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions."""
        active_count = len(self.positions)
        closed_count = len(self.closed_positions)
        
        # Calculate total P&L
        total_unrealized_pnl = sum(p['unrealized_pnl'] for p in self.positions.values())
        total_realized_pnl = sum(p['realized_pnl'] for p in self.closed_positions)
        
        return {
            'active_positions': active_count,
            'closed_positions': closed_count,
            'total_positions': active_count + closed_count,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'total_pnl': total_unrealized_pnl + total_realized_pnl
        }
    
    async def start_exit_monitoring(self):
        """
        Start continuous exit monitoring loop.
        Phase 3.4 - Issue #100: Position Exit Monitoring
        
        Monitors all active positions every 5 seconds for exit conditions.
        """
        if self.monitoring_active:
            logger.warning("Exit monitoring already active")
            return
        
        # Get configuration
        config = self.portfolio_manager.cfg if hasattr(self.portfolio_manager, 'cfg') else {}
        exit_config = config.get('position_management', {}).get('exit_monitoring', {})
        
        if not exit_config.get('enabled', True):
            logger.info("Exit monitoring disabled in configuration")
            return
        
        check_frequency = exit_config.get('check_frequency', 5)
        
        self.monitoring_active = True
        logger.info(f"Starting exit monitoring (check every {check_frequency}s)")
        
        async def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Check all active positions
                    position_ids = list(self.positions.keys())
                    
                    for position_id in position_ids:
                        try:
                            result = await self.manage_position_exits(position_id)
                            
                            if result.get('should_exit'):
                                logger.info(f"Position {position_id} exited: {result.get('exit_reason')}")
                        
                        except Exception as e:
                            logger.error(f"Error checking position {position_id}: {e}")
                    
                    # Wait before next check
                    await asyncio.sleep(check_frequency)
                    
                except Exception as e:
                    logger.error(f"Error in exit monitoring loop: {e}")
                    await asyncio.sleep(check_frequency)
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(monitoring_loop())
        logger.info("Exit monitoring task started")
    
    async def stop_exit_monitoring(self):
        """Stop exit monitoring loop."""
        if not self.monitoring_active:
            logger.warning("Exit monitoring not active")
            return
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Exit monitoring stopped")
