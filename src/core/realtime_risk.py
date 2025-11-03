"""
Real-Time Risk Monitoring.
Uses WebSocket feeds for immediate risk assessment and alerts.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import numpy as np
from collections import deque

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


class RealTimeRiskMonitor:
    """
    Real-time risk monitoring using WebSocket feeds.
    
    PHASE 2 REFACTOR: Event-driven architecture.
    - Monitors risk conditions and EMITS events
    - Does NOT take direct actions (no position closing)
    - Other components listen to events and take appropriate actions
    """
    
    def __init__(self, risk_manager, websocket_manager, portfolio_manager=None):
        """
        Initialize real-time risk monitor.
        
        PHASE 2: Now accepts PortfolioManager for state queries.
        
        Args:
            risk_manager: Risk manager instance (for risk calculations)
            websocket_manager: WebSocket manager for real-time data
            portfolio_manager: PortfolioManager instance (for state queries)
        """
        self.risk_manager = risk_manager
        self.ws_manager = websocket_manager
        self.portfolio_manager = portfolio_manager
        
        # PHASE 2: Event publishing queue
        # Risk events are published here for other components to consume
        self.risk_alerts = asyncio.Queue()
        self.risk_events = asyncio.Queue()  # New: dedicated event queue
        
        # Emergency stop triggers (for logging only, not for action)
        self.emergency_stops = {}
        
        # Price monitoring
        self.price_buffers = {}  # symbol -> deque of recent prices
        self.buffer_size = 100
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_tasks = []
        
        logger.info("RealTimeRiskMonitor initialized (PHASE 2: Event-Driven)")
    
    async def start_risk_monitoring(self):
        """
        Start real-time risk monitoring for all active positions.
        
        PHASE 2: Gets positions from PortfolioManager.
        """
        try:
            self.monitoring_active = True
            
            # PHASE 2: Get active positions from PortfolioManager or fallback to RiskManager
            if self.portfolio_manager:
                active_positions = self.portfolio_manager.get_open_positions()
            else:
                active_positions = self.risk_manager.active_positions
            
            if not active_positions:
                logger.info("No active positions to monitor")
                return
            
            # Extract symbols to monitor
            symbols_per_exchange = {}
            for pos_id, position in active_positions.items():
                symbol = position.get('symbol', '')
                exchange = position.get('exchange', 'kucoinfutures')
                
                if exchange not in symbols_per_exchange:
                    symbols_per_exchange[exchange] = []
                
                if symbol and symbol not in symbols_per_exchange[exchange]:
                    symbols_per_exchange[exchange].append(symbol)
            
            logger.info(f"Starting risk monitoring for {len(active_positions)} positions")
            logger.info(f"Symbols per exchange: {symbols_per_exchange}")
            
            # Start WebSocket streams if manager available
            if self.ws_manager:
                # Create price update callback
                async def price_callback(exchange, symbol, ticker):
                    await self.on_price_update(symbol, ticker)
                
                # Start ticker streams
                tasks = await self.ws_manager.stream_tickers(
                    symbols_per_exchange,
                    callback=price_callback
                )
                
                self.monitoring_tasks.extend(tasks)
                logger.info(f"Started {len(tasks)} monitoring streams")
            
            # Start portfolio metrics monitoring
            portfolio_monitor_task = asyncio.create_task(self._monitor_portfolio_metrics())
            self.monitoring_tasks.append(portfolio_monitor_task)
            
        except Exception as e:
            logger.error(f"Error starting risk monitoring: {e}")
    
    async def on_price_update(self, symbol: str, price_data: Dict):
        """
        Process price updates for risk assessment.
        
        Args:
            symbol: Trading symbol
            price_data: Ticker data with current price
        """
        try:
            current_price = price_data.get('last', 0)
            if current_price <= 0:
                return
            
            # Update price buffer
            if symbol not in self.price_buffers:
                self.price_buffers[symbol] = deque(maxlen=self.buffer_size)
            
            self.price_buffers[symbol].append({
                'price': current_price,
                'timestamp': datetime.now(timezone.utc)
            })
            
            # PHASE 2: Get positions from PortfolioManager
            if self.portfolio_manager:
                active_positions = self.portfolio_manager.get_open_positions()
            else:
                active_positions = self.risk_manager.active_positions
            
            # Find positions for this symbol
            for pos_id, position in active_positions.items():
                if position.get('symbol') == symbol:
                    # PHASE 2: Update position price via PortfolioManager
                    if self.portfolio_manager:
                        self.portfolio_manager.update_position_price(pos_id, current_price)
                    else:
                        self.risk_manager.update_position_price(pos_id, current_price)
                    
                    # Check stop-loss triggers (emits events, doesn't take action)
                    await self._check_stop_loss(pos_id, position, current_price)
                    
                    # Monitor unrealized P&L (emits events, doesn't take action)
                    await self._check_unrealized_pnl(pos_id, position, current_price)
            
            # Assess portfolio heat
            await self._assess_portfolio_heat()
            
        except Exception as e:
            logger.error(f"Error processing price update for {symbol}: {e}")
    
    async def _check_stop_loss(self, position_id: str, position: Dict, current_price: float):
        """Check if stop-loss has been triggered."""
        try:
            stop_loss = position.get('stop_loss', 0)
            side = position.get('side', 'long')
            symbol = position.get('symbol', 'UNKNOWN')
            
            triggered = False
            if side == 'long' and current_price <= stop_loss:
                triggered = True
            elif side == 'short' and current_price >= stop_loss:
                triggered = True
            
            if triggered:
                # PHASE 2: Create event for stop-loss trigger
                event = {
                    'event_type': 'stop_loss_triggered',
                    'position_id': position_id,
                    'symbol': symbol,
                    'trigger_price': current_price,
                    'stop_loss': stop_loss,
                    'side': side,
                    'timestamp': datetime.now(timezone.utc),
                    'severity': 'high',
                    'action_required': 'close_position'
                }
                
                # Emit event to both queues
                await self.risk_events.put(event)  # For action listeners
                
                # Also create alert for monitoring/logging
                alert = {
                    'type': 'stop_loss_trigger',
                    'severity': 'high',
                    'position_id': position_id,
                    'symbol': symbol,
                    'current_price': current_price,
                    'stop_loss': stop_loss,
                    'side': side,
                    'timestamp': datetime.now(timezone.utc),
                    'message': f"Stop loss triggered for {symbol} at {current_price}"
                }
                await self.risk_alerts.put(alert)
                
                logger.warning(f"⚠️ [EVENT] STOP_LOSS_TRIGGERED: {position_id} - {symbol} at {current_price}")
                logger.warning(f"   Event emitted for action listeners to handle")
                
        except Exception as e:
            logger.error(f"Error checking stop loss: {e}")
    
    async def _check_unrealized_pnl(self, position_id: str, position: Dict, current_price: float):
        """
        Monitor unrealized P&L and generate alerts if needed.
        
        PHASE 2: Emits events, doesn't take action.
        """
        try:
            entry_price = position.get('entry_price', 0)
            size = position.get('size', 0)
            side = position.get('side', 'long')
            
            # Calculate unrealized P&L
            unrealized_pnl = calculate_unrealized_pnl(side, entry_price, current_price, size)
            
            # PHASE 2: Get portfolio value from PortfolioManager
            if self.portfolio_manager:
                portfolio_value = self.portfolio_manager.get_current_equity()
            else:
                portfolio_value = self.risk_manager.portfolio_value
            
            # Check against thresholds
            loss_threshold = portfolio_value * 0.03  # 3% loss threshold
            
            if unrealized_pnl < -loss_threshold:
                # PHASE 2: Emit event for large loss
                event = {
                    'event_type': 'large_unrealized_loss',
                    'position_id': position_id,
                    'symbol': position.get('symbol', 'UNKNOWN'),
                    'unrealized_pnl': unrealized_pnl,
                    'loss_pct': abs(unrealized_pnl) / portfolio_value if portfolio_value > 0 else 0,
                    'timestamp': datetime.now(timezone.utc),
                    'severity': 'high',
                    'action_required': 'review_position'
                }
                await self.risk_events.put(event)
                
                # Also create alert
                alert = {
                    'type': 'large_unrealized_loss',
                    'severity': 'high',
                    'position_id': position_id,
                    'symbol': position.get('symbol', 'UNKNOWN'),
                    'unrealized_pnl': unrealized_pnl,
                    'loss_pct': abs(unrealized_pnl) / portfolio_value if portfolio_value > 0 else 0,
                    'timestamp': datetime.now(timezone.utc),
                    'message': f"Large unrealized loss: ${unrealized_pnl:.2f}"
                }
                
                await self.risk_alerts.put(alert)
                logger.warning(f"⚠️ [EVENT] LARGE_LOSS: {position_id} - ${unrealized_pnl:.2f}")
                
        except Exception as e:
            logger.error(f"Error checking unrealized PnL: {e}")
    
    async def _assess_portfolio_heat(self):
        """
        Assess total portfolio risk exposure.
        
        PHASE 2: Uses PortfolioManager for state queries.
        """
        try:
            # PHASE 2: Get summary from RiskManager with PortfolioManager
            summary = self.risk_manager.get_portfolio_summary(self.portfolio_manager)
            portfolio_heat = summary.get('portfolio_heat', 0)
            
            # High heat threshold (8%)
            if portfolio_heat > 0.08:
                # PHASE 2: Emit event for high portfolio heat
                event = {
                    'event_type': 'high_portfolio_heat',
                    'portfolio_heat': portfolio_heat,
                    'active_positions': summary.get('active_positions', 0),
                    'total_risk': summary.get('total_risk', 0),
                    'timestamp': datetime.now(timezone.utc),
                    'severity': 'medium',
                    'action_required': 'reduce_exposure'
                }
                await self.risk_events.put(event)
                
                # Also create alert
                alert = {
                    'type': 'high_portfolio_heat',
                    'severity': 'medium',
                    'portfolio_heat': portfolio_heat,
                    'active_positions': summary.get('active_positions', 0),
                    'total_risk': summary.get('total_risk', 0),
                    'timestamp': datetime.now(timezone.utc),
                    'message': f"High portfolio heat: {portfolio_heat:.2%}"
                }
                
                await self.risk_alerts.put(alert)
                logger.warning(f"⚠️ [EVENT] HIGH_PORTFOLIO_HEAT: {portfolio_heat:.2%}")
                
        except Exception as e:
            logger.error(f"Error assessing portfolio heat: {e}")
    
    async def _monitor_portfolio_metrics(self):
        """
        Continuously monitor portfolio-level metrics.
        
        PHASE 2: Uses PortfolioManager and emits events.
        """
        try:
            while self.monitoring_active:
                # PHASE 2: Get portfolio summary with PortfolioManager
                summary = self.risk_manager.get_portfolio_summary(self.portfolio_manager)
                
                # Check drawdown
                drawdown = summary.get('current_drawdown', 0)
                max_drawdown = self.risk_manager.risk_limits.get('max_drawdown', 0.15)
                
                if drawdown > max_drawdown * 0.8:  # 80% of max drawdown
                    # PHASE 2: Emit event for approaching max drawdown
                    event = {
                        'event_type': 'approaching_max_drawdown',
                        'current_drawdown': drawdown,
                        'max_drawdown': max_drawdown,
                        'timestamp': datetime.now(timezone.utc),
                        'severity': 'high',
                        'action_required': 'halt_new_positions'
                    }
                    await self.risk_events.put(event)
                    
                    # Also create alert
                    alert = {
                        'type': 'approaching_max_drawdown',
                        'severity': 'high',
                        'current_drawdown': drawdown,
                        'max_drawdown': max_drawdown,
                        'timestamp': datetime.now(timezone.utc),
                        'message': f"Approaching max drawdown: {drawdown:.2%}"
                    }
                    
                    await self.risk_alerts.put(alert)
                    logger.warning(f"⚠️ [EVENT] APPROACHING_MAX_DRAWDOWN: {drawdown:.2%}")
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
        except asyncio.CancelledError:
            logger.info("Portfolio metrics monitoring cancelled")
        except Exception as e:
            logger.error(f"Error in portfolio monitoring: {e}")
    
    async def trigger_emergency_stop(self, reason: str, affected_positions: List[str] = None):
        """
        Emergency position closure mechanism.
        
        PHASE 2: ONLY emits event - does NOT close positions directly.
        Action listeners will handle actual position closures.
        
        Args:
            reason: Reason for emergency stop
            affected_positions: List of position IDs to close (None = all)
        """
        try:
            logger.critical(f"🚨 [EVENT] EMERGENCY STOP TRIGGERED: {reason}")
            
            # PHASE 2: Get active positions from PortfolioManager
            if self.portfolio_manager:
                active_pos = self.portfolio_manager.get_open_positions()
            else:
                active_pos = self.risk_manager.active_positions
            
            if affected_positions is None:
                # All positions
                affected_positions = list(active_pos.keys())
            
            # Record emergency stop
            stop_record = {
                'reason': reason,
                'timestamp': datetime.now(timezone.utc),
                'affected_positions': affected_positions,
                'portfolio_state': self.risk_manager.get_portfolio_summary(self.portfolio_manager)
            }
            
            self.emergency_stops[datetime.now(timezone.utc).isoformat()] = stop_record
            
            # PHASE 2: Emit emergency stop event for action listeners
            event = {
                'event_type': 'emergency_stop',
                'reason': reason,
                'affected_positions': affected_positions,
                'timestamp': datetime.now(timezone.utc),
                'severity': 'critical',
                'action_required': 'close_all_positions'
            }
            await self.risk_events.put(event)
            
            # Generate emergency alert
            alert = {
                'type': 'emergency_stop',
                'severity': 'critical',
                'reason': reason,
                'affected_positions': len(affected_positions),
                'timestamp': datetime.now(timezone.utc),
                'message': f"Emergency stop activated: {reason}"
            }
            
            await self.risk_alerts.put(alert)
            
            # PHASE 2: Only log - don't take action
            logger.critical(f"🚨 Emergency stop event emitted for {len(affected_positions)} positions")
            logger.critical(f"   Action listeners will handle position closures")
            
        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")
    
    def calculate_portfolio_var(self, confidence: float = 0.05, time_horizon: int = 1) -> Dict[str, float]:
        """
        Calculate portfolio Value at Risk (VaR).
        
        Args:
            confidence: Confidence level (default 5% = 95% confidence)
            time_horizon: Time horizon in days (default 1)
            
        Returns:
            Dictionary with VaR metrics
        """
        try:
            # Collect returns from price buffers
            all_returns = []
            
            for symbol, price_buffer in self.price_buffers.items():
                if len(price_buffer) < 2:
                    continue
                
                prices = [p['price'] for p in price_buffer]
                returns = np.diff(prices) / prices[:-1]
                all_returns.extend(returns)
            
            if not all_returns:
                return {
                    'historical_var': 0.0,
                    'parametric_var': 0.0,
                    'expected_shortfall': 0.0
                }
            
            returns_array = np.array(all_returns)
            
            # PHASE 2: Get portfolio value from PortfolioManager
            if self.portfolio_manager:
                portfolio_value = self.portfolio_manager.get_current_equity()
            else:
                portfolio_value = self.risk_manager.portfolio_value
            
            # Historical VaR
            historical_var = np.percentile(returns_array, confidence * 100)
            historical_var_dollar = abs(historical_var * portfolio_value)
            
            # Parametric VaR (assuming normal distribution)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            z_score = -1.645  # 95% confidence
            parametric_var = (mean_return + z_score * std_return) * portfolio_value
            parametric_var_dollar = abs(parametric_var)
            
            # Expected Shortfall (Conditional VaR)
            tail_returns = returns_array[returns_array <= historical_var]
            expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else historical_var
            expected_shortfall_dollar = abs(expected_shortfall * portfolio_value)
            
            var_metrics = {
                'historical_var': historical_var_dollar,
                'parametric_var': parametric_var_dollar,
                'expected_shortfall': expected_shortfall_dollar,
                'confidence_level': 1 - confidence,
                'time_horizon': time_horizon
            }
            
            logger.debug(f"VaR calculated: Historical=${historical_var_dollar:.2f}, "
                        f"Parametric=${parametric_var_dollar:.2f}, "
                        f"ES=${expected_shortfall_dollar:.2f}")
            
            return var_metrics
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return {
                'historical_var': 0.0,
                'parametric_var': 0.0,
                'expected_shortfall': 0.0
            }
    
    async def get_risk_alerts(self, count: int = 10) -> List[Dict]:
        """
        Get recent risk alerts.
        
        Args:
            count: Number of alerts to retrieve
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        try:
            for _ in range(min(count, self.risk_alerts.qsize())):
                alert = await asyncio.wait_for(self.risk_alerts.get(), timeout=0.1)
                alerts.append(alert)
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.error(f"Error retrieving alerts: {e}")
        
        return alerts
    
    async def get_risk_events(self, count: int = 10) -> List[Dict]:
        """
        Get recent risk events (PHASE 2).
        
        Args:
            count: Number of events to retrieve
            
        Returns:
            List of event dictionaries
        """
        events = []
        try:
            for _ in range(min(count, self.risk_events.qsize())):
                event = await asyncio.wait_for(self.risk_events.get(), timeout=0.1)
                events.append(event)
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.error(f"Error retrieving events: {e}")
        
        return events
    
    def update_price_history(self, symbol: str, price: float):
        """
        Update price history buffer for a symbol.
        
        Args:
            symbol: Trading symbol
            price: Current price
        """
        if symbol not in self.price_buffers:
            self.price_buffers[symbol] = deque(maxlen=self.buffer_size)
        
        self.price_buffers[symbol].append({
            'price': price,
            'timestamp': datetime.now(timezone.utc)
        })
    
    def stop_monitoring(self):
        """Stop all risk monitoring tasks."""
        self.monitoring_active = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
        
        logger.info("Risk monitoring stopped")
    
    async def close(self):
        """Cleanup and close monitoring."""
        self.stop_monitoring()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        self.monitoring_tasks.clear()
        logger.info("RealTimeRiskMonitor closed")
