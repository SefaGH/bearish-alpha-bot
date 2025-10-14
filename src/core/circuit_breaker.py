"""
Circuit Breaker and Emergency Stop System.
Implements emergency protocols for extreme market conditions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class CircuitBreakerSystem:
    """Emergency stop and circuit breaker mechanisms."""
    
    def __init__(self, risk_manager, websocket_manager=None):
        """
        Initialize circuit breaker system.
        
        Args:
            risk_manager: Risk manager instance
            websocket_manager: WebSocket manager for real-time monitoring
        """
        self.risk_manager = risk_manager
        self.ws_manager = websocket_manager
        
        # Circuit breaker states
        self.circuit_breakers = {}
        self.emergency_protocols = {}
        
        # Monitoring data
        self.daily_pnl_history = deque(maxlen=1000)
        self.volatility_history = {}
        
        # Breaker status
        self.breakers_active = {}
        self.monitoring_active = False
        self.monitoring_task = None
        
        logger.info("CircuitBreakerSystem initialized")
    
    def set_circuit_breakers(self, daily_loss_limit: float = 0.05,
                            position_loss_limit: float = 0.03,
                            volatility_spike_threshold: float = 3.0):
        """
        Configure circuit breaker triggers.
        
        Args:
            daily_loss_limit: Daily portfolio loss limit (default 5%)
            position_loss_limit: Individual position loss limit (default 3%)
            volatility_spike_threshold: Volatility spike threshold in std devs (default 3.0)
        """
        self.circuit_breakers = {
            'daily_loss': {
                'threshold': daily_loss_limit,
                'enabled': True,
                'triggered': False
            },
            'position_loss': {
                'threshold': position_loss_limit,
                'enabled': True,
                'triggered': False
            },
            'volatility_spike': {
                'threshold': volatility_spike_threshold,
                'enabled': True,
                'triggered': False
            }
        }
        
        logger.info(f"Circuit breakers configured: {self.circuit_breakers}")
    
    async def monitor_circuit_breakers(self):
        """Monitor all circuit breaker conditions continuously."""
        try:
            self.monitoring_active = True
            logger.info("Circuit breaker monitoring started")
            
            while self.monitoring_active:
                # Check daily loss limit
                await self._check_daily_loss()
                
                # Check position loss limits
                await self._check_position_losses()
                
                # Check volatility spikes
                await self._check_volatility_spikes()
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except asyncio.CancelledError:
            logger.info("Circuit breaker monitoring cancelled")
        except Exception as e:
            logger.error(f"Error in circuit breaker monitoring: {e}")
        finally:
            self.monitoring_active = False
    
    async def _check_daily_loss(self):
        """Check daily portfolio loss limit."""
        try:
            breaker = self.circuit_breakers.get('daily_loss', {})
            if not breaker.get('enabled') or breaker.get('triggered'):
                return
            
            threshold = breaker.get('threshold', 0.05)
            
            # Calculate daily P&L
            portfolio_value = self.risk_manager.portfolio_value
            peak_value = self.risk_manager.peak_portfolio_value
            
            daily_loss_pct = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
            
            logger.debug(f"🔥 [CIRCUIT] Daily P&L: {daily_loss_pct:.2%} (limit: {threshold:.2%})")
            
            if daily_loss_pct > threshold:
                logger.critical(f"DAILY LOSS LIMIT BREACHED: {daily_loss_pct:.2%} > {threshold:.2%}")
                logger.debug(f"🔥 [CIRCUIT] TRIGGERED: Daily loss limit breached")
                await self.trigger_circuit_breaker('daily_loss', severity='critical')
                
        except Exception as e:
            logger.error(f"Error checking daily loss: {e}")
    
    async def _check_position_losses(self):
        """Check individual position loss limits."""
        try:
            breaker = self.circuit_breakers.get('position_loss', {})
            if not breaker.get('enabled') or breaker.get('triggered'):
                return
            
            threshold = breaker.get('threshold', 0.03)
            portfolio_value = self.risk_manager.portfolio_value
            
            positions_to_close = []
            
            for pos_id, position in self.risk_manager.active_positions.items():
                unrealized_pnl = position.get('unrealized_pnl', 0)
                loss_pct = abs(unrealized_pnl) / portfolio_value if portfolio_value > 0 else 0
                
                if unrealized_pnl < 0 and loss_pct > threshold:
                    logger.warning(f"Position loss limit breached: {pos_id} - {loss_pct:.2%}")
                    positions_to_close.append(pos_id)
            
            if positions_to_close:
                await self.trigger_circuit_breaker('position_loss', 
                                                  severity='high',
                                                  affected_positions=positions_to_close)
                
        except Exception as e:
            logger.error(f"Error checking position losses: {e}")
    
    async def _check_volatility_spikes(self):
        """Check for extreme volatility spikes."""
        try:
            breaker = self.circuit_breakers.get('volatility_spike', {})
            if not breaker.get('enabled') or breaker.get('triggered'):
                return
            
            threshold = breaker.get('threshold', 3.0)
            
            # Check volatility for each active position
            for pos_id, position in self.risk_manager.active_positions.items():
                symbol = position.get('symbol', '')
                
                if symbol not in self.volatility_history:
                    continue
                
                vol_history = self.volatility_history[symbol]
                if len(vol_history) < 20:
                    continue
                
                # Calculate recent volatility vs historical
                recent_vol = vol_history[-1] if vol_history else 0
                historical_vols = list(vol_history)[:-10]  # Exclude recent for baseline
                
                if len(historical_vols) < 10:
                    continue
                
                mean_vol = sum(historical_vols) / len(historical_vols)
                std_vol = (sum((v - mean_vol) ** 2 for v in historical_vols) / len(historical_vols)) ** 0.5
                
                if std_vol > 0:
                    z_score = (recent_vol - mean_vol) / std_vol
                    
                    logger.debug(f"🔥 [CIRCUIT] Volatility spike check: {symbol} z-score={z_score:.2f} (threshold: {threshold})")
                    
                    if abs(z_score) > threshold:
                        logger.warning(f"Volatility spike detected: {symbol} - z-score: {z_score:.2f}")
                        logger.debug(f"🔥 [CIRCUIT] TRIGGERED: Volatility spike on {symbol}")
                        await self.trigger_circuit_breaker('volatility_spike',
                                                          severity='high',
                                                          details={'symbol': symbol, 'z_score': z_score})
                        
        except Exception as e:
            logger.error(f"Error checking volatility spikes: {e}")
    
    async def trigger_circuit_breaker(self, breaker_type: str, severity: str = 'high',
                                     affected_positions: List[str] = None,
                                     details: Dict = None):
        """
        Execute circuit breaker protocol.
        
        Args:
            breaker_type: Type of breaker ('daily_loss', 'position_loss', 'volatility_spike')
            severity: Severity level ('low', 'medium', 'high', 'critical')
            affected_positions: List of position IDs affected
            details: Additional details about the trigger
        """
        try:
            if breaker_type in self.circuit_breakers:
                self.circuit_breakers[breaker_type]['triggered'] = True
                self.circuit_breakers[breaker_type]['triggered_at'] = datetime.now(timezone.utc)
            
            logger.critical(f"CIRCUIT BREAKER TRIGGERED: {breaker_type} - Severity: {severity}")
            
            # Record trigger
            trigger_record = {
                'breaker_type': breaker_type,
                'severity': severity,
                'timestamp': datetime.now(timezone.utc),
                'portfolio_state': self.risk_manager.get_portfolio_summary(),
                'affected_positions': affected_positions or [],
                'details': details or {}
            }
            
            self.breakers_active[datetime.now(timezone.utc).isoformat()] = trigger_record
            
            # Execute appropriate protocol
            if severity == 'critical':
                await self.execute_emergency_protocol('close_all')
            elif breaker_type == 'position_loss' and affected_positions:
                await self.execute_emergency_protocol('close_positions', affected_positions)
            elif breaker_type == 'volatility_spike':
                await self.execute_emergency_protocol('reduce_positions')
            
            logger.critical(f"Circuit breaker {breaker_type} executed")
            
        except Exception as e:
            logger.error(f"Error triggering circuit breaker: {e}")
    
    async def execute_emergency_protocol(self, protocol_name: str, positions: List[str] = None):
        """
        Execute predefined emergency protocols.
        
        Args:
            protocol_name: Name of protocol ('close_all', 'close_positions', 'reduce_positions')
            positions: List of position IDs (for selective protocols)
        """
        try:
            logger.critical(f"EXECUTING EMERGENCY PROTOCOL: {protocol_name}")
            
            if protocol_name == 'close_all':
                # Close all positions immediately
                all_positions = list(self.risk_manager.active_positions.keys())
                
                for pos_id in all_positions:
                    position = self.risk_manager.active_positions.get(pos_id, {})
                    logger.critical(f"Emergency close: {pos_id} - {position.get('symbol', 'UNKNOWN')}")
                    
                    # In real implementation, this would execute market orders
                    # For now, simulate closure
                    current_price = position.get('current_price', position.get('entry_price', 0))
                    entry_price = position.get('entry_price', 0)
                    size = position.get('size', 0)
                    side = position.get('side', 'long')
                    
                    if side == 'long':
                        realized_pnl = (current_price - entry_price) * size
                    else:
                        realized_pnl = (entry_price - current_price) * size
                    
                    self.risk_manager.close_position(pos_id, current_price, realized_pnl)
                
                logger.critical(f"Emergency protocol 'close_all' completed: {len(all_positions)} positions closed")
                
            elif protocol_name == 'close_positions' and positions:
                # Close specific positions
                for pos_id in positions:
                    if pos_id in self.risk_manager.active_positions:
                        position = self.risk_manager.active_positions[pos_id]
                        logger.critical(f"Emergency close: {pos_id} - {position.get('symbol', 'UNKNOWN')}")
                        
                        current_price = position.get('current_price', position.get('entry_price', 0))
                        entry_price = position.get('entry_price', 0)
                        size = position.get('size', 0)
                        side = position.get('side', 'long')
                        
                        if side == 'long':
                            realized_pnl = (current_price - entry_price) * size
                        else:
                            realized_pnl = (entry_price - current_price) * size
                        
                        self.risk_manager.close_position(pos_id, current_price, realized_pnl)
                
                logger.critical(f"Emergency protocol 'close_positions' completed: {len(positions)} positions closed")
                
            elif protocol_name == 'reduce_positions':
                # Reduce all position sizes by 50%
                for pos_id, position in self.risk_manager.active_positions.items():
                    original_size = position.get('size', 0)
                    position['size'] = original_size * 0.5
                    logger.warning(f"Position reduced: {pos_id} - {original_size} -> {position['size']}")
                
                logger.critical("Emergency protocol 'reduce_positions' completed")
            
        except Exception as e:
            logger.error(f"Error executing emergency protocol: {e}")
    
    def reset_circuit_breaker(self, breaker_type: str):
        """
        Reset a circuit breaker after issue resolution.
        
        Args:
            breaker_type: Type of breaker to reset
        """
        if breaker_type in self.circuit_breakers:
            self.circuit_breakers[breaker_type]['triggered'] = False
            logger.info(f"Circuit breaker reset: {breaker_type}")
        else:
            logger.warning(f"Unknown circuit breaker type: {breaker_type}")
    
    def update_volatility(self, symbol: str, volatility: float):
        """
        Update volatility history for a symbol.
        
        Args:
            symbol: Trading symbol
            volatility: Current volatility measure (e.g., ATR)
        """
        if symbol not in self.volatility_history:
            self.volatility_history[symbol] = deque(maxlen=100)
        
        self.volatility_history[symbol].append(volatility)
    
    async def check_circuit_breaker(self) -> Dict[str, Any]:
        """
        Check all circuit breaker conditions and return status.
        
        Returns:
            Dictionary with breaker status and severity
        """
        try:
            # Check if any breakers are currently triggered
            for breaker_name, breaker in self.circuit_breakers.items():
                if breaker.get('triggered', False):
                    return {
                        'tripped': True,
                        'breaker': breaker_name,
                        'severity': 'critical',  # Adjust based on breaker type
                        'threshold': breaker.get('threshold'),
                        'message': f"Circuit breaker '{breaker_name}' is active"
                    }
            
            # Run active checks (non-blocking)
            await self._check_daily_loss()
            await self._check_position_losses() 
            await self._check_volatility_spikes()
            
            # Check again after running checks
            for breaker_name, breaker in self.circuit_breakers.items():
                if breaker.get('triggered', False):
                    return {
                        'tripped': True,
                        'breaker': breaker_name,
                        'severity': 'critical',
                        'threshold': breaker.get('threshold'),
                        'message': f"Circuit breaker '{breaker_name}' just triggered"
                    }
            
            # All clear
            return {
                'tripped': False,
                'severity': 'none',
                'message': 'All circuit breakers normal'
            }
            
        except Exception as e:
            logger.error(f"Error checking circuit breakers: {e}")
            return {
                'tripped': True,
                'breaker': 'system_error',
                'severity': 'critical',
                'message': f"Circuit breaker check failed: {e}"
            }
    
    def get_breaker_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        return {
            'circuit_breakers': self.circuit_breakers,
            'monitoring_active': self.monitoring_active,
            'active_triggers': len(self.breakers_active),
            'last_trigger': max(self.breakers_active.keys()) if self.breakers_active else None
        }
    
    def start_monitoring(self):
        """Start circuit breaker monitoring."""
        if not self.monitoring_active:
            self.monitoring_task = asyncio.create_task(self.monitor_circuit_breakers())
            logger.info("Circuit breaker monitoring started")
    
    def stop_monitoring(self):
        """Stop circuit breaker monitoring."""
        self.monitoring_active = False
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
        logger.info("Circuit breaker monitoring stopped")
    
    async def close(self):
        """Cleanup and close monitoring."""
        self.stop_monitoring()
        if self.monitoring_task:
            await asyncio.gather(self.monitoring_task, return_exceptions=True)
        logger.info("CircuitBreakerSystem closed")
