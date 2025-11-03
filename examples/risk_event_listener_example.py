"""
Example: Risk Event Listener Implementation (Phase 2)

This example demonstrates how to implement an event listener that consumes
risk events from RealTimeRiskMonitor and takes appropriate actions.

ARCHITECTURE:
- RealTimeRiskMonitor: Detects risk conditions and EMITS events
- Event Listener: Consumes events and TAKES actions
- Separation of concerns: Detection vs Action

Author: Phase 2 Refactoring
Date: 2025
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RiskEventListener:
    """
    Event listener for risk events from RealTimeRiskMonitor.
    
    This component listens to the risk_events queue and takes appropriate
    actions based on event type and severity.
    """
    
    def __init__(self, risk_monitor, order_manager, position_manager, portfolio_manager):
        """
        Initialize risk event listener.
        
        Args:
            risk_monitor: RealTimeRiskMonitor instance
            order_manager: OrderManager for placing orders
            position_manager: PositionManager for position management
            portfolio_manager: PortfolioManager for state queries
        """
        self.risk_monitor = risk_monitor
        self.order_manager = order_manager
        self.position_manager = position_manager
        self.portfolio_manager = portfolio_manager
        
        self.listener_active = False
        self.listener_task = None
        
        logger.info("RiskEventListener initialized")
    
    async def start_listening(self):
        """Start listening to risk events."""
        self.listener_active = True
        self.listener_task = asyncio.create_task(self._event_loop())
        logger.info("✅ Risk event listener started")
    
    async def stop_listening(self):
        """Stop listening to risk events."""
        self.listener_active = False
        if self.listener_task and not self.listener_task.done():
            self.listener_task.cancel()
            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Risk event listener stopped")
    
    async def _event_loop(self):
        """
        Main event processing loop.
        
        Continuously monitors the risk_events queue and processes events.
        """
        logger.info("🎧 Event listener loop started")
        
        try:
            while self.listener_active:
                try:
                    # Wait for event with timeout to allow checking listener_active
                    event = await asyncio.wait_for(
                        self.risk_monitor.risk_events.get(),
                        timeout=1.0
                    )
                    
                    # Process the event
                    await self._handle_event(event)
                    
                except asyncio.TimeoutError:
                    # No event received, continue loop
                    continue
                except Exception as e:
                    logger.error(f"Error in event loop: {e}", exc_info=True)
                    await asyncio.sleep(1)  # Brief pause before continuing
        
        except asyncio.CancelledError:
            logger.info("Event listener loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in event loop: {e}", exc_info=True)
    
    async def _handle_event(self, event: Dict):
        """
        Handle a single risk event.
        
        Routes the event to the appropriate handler based on event_type.
        
        Args:
            event: Risk event dictionary
        """
        event_type = event.get('event_type', 'unknown')
        severity = event.get('severity', 'unknown')
        timestamp = event.get('timestamp', datetime.now())
        
        logger.info(f"📩 Received event: {event_type} (severity: {severity})")
        
        # Route to appropriate handler
        if event_type == 'stop_loss_triggered':
            await self._handle_stop_loss_triggered(event)
        elif event_type == 'large_unrealized_loss':
            await self._handle_large_unrealized_loss(event)
        elif event_type == 'high_portfolio_heat':
            await self._handle_high_portfolio_heat(event)
        elif event_type == 'approaching_max_drawdown':
            await self._handle_approaching_max_drawdown(event)
        elif event_type == 'emergency_stop':
            await self._handle_emergency_stop(event)
        else:
            logger.warning(f"⚠️  Unknown event type: {event_type}")
    
    async def _handle_stop_loss_triggered(self, event: Dict):
        """
        Handle stop-loss trigger event.
        
        Action: Close the position immediately with a market order.
        
        Args:
            event: Stop-loss trigger event
        """
        position_id = event.get('position_id')
        symbol = event.get('symbol')
        trigger_price = event.get('trigger_price')
        
        logger.warning(f"🚨 STOP LOSS TRIGGERED: {position_id} - {symbol} at {trigger_price}")
        
        try:
            # Get position details from PortfolioManager
            position = self.portfolio_manager.get_position(position_id)
            
            if not position:
                logger.error(f"❌ Position {position_id} not found in portfolio")
                return
            
            # Close position via OrderManager
            logger.info(f"📤 Sending market order to close position {position_id}")
            
            # This is where you would call OrderManager to close the position
            # Expected OrderManager interface:
            # await self.order_manager.place_order(
            #     symbol=position['symbol'],
            #     side='sell' if position['side'] == 'long' else 'buy',
            #     order_type='market',
            #     quantity=position['size'],
            #     reduce_only=True
            # )
            # 
            # Or simpler interface if available:
            # await self.order_manager.close_position(
            #     position_id=position_id,
            #     order_type='market'
            # )
            
            # For demonstration, we'll just log the action
            logger.info(f"✅ Position {position_id} closed due to stop-loss trigger")
            
            # Update metrics or send notifications if needed
            await self._send_notification(
                f"Position {symbol} closed at {trigger_price} due to stop-loss trigger"
            )
            
        except Exception as e:
            logger.error(f"❌ Error closing position {position_id}: {e}", exc_info=True)
    
    async def _handle_large_unrealized_loss(self, event: Dict):
        """
        Handle large unrealized loss event.
        
        Action: Review position and potentially tighten stop-loss.
        
        Args:
            event: Large unrealized loss event
        """
        position_id = event.get('position_id')
        symbol = event.get('symbol')
        unrealized_pnl = event.get('unrealized_pnl')
        loss_pct = event.get('loss_pct')
        
        logger.warning(f"⚠️  LARGE UNREALIZED LOSS: {position_id} - {symbol}: ${unrealized_pnl:.2f} ({loss_pct:.2%})")
        
        try:
            # Get position details
            position = self.portfolio_manager.get_position(position_id)
            
            if not position:
                logger.error(f"❌ Position {position_id} not found")
                return
            
            # Option 1: Tighten stop-loss
            current_price = position.get('current_price', 0)
            current_stop = position.get('stop_loss', 0)
            
            # Move stop-loss closer to current price (e.g., 1% away)
            new_stop = current_price * 0.99  # 1% below current price for long
            
            if new_stop > current_stop:
                logger.info(f"📊 Tightening stop-loss: {current_stop} -> {new_stop}")
                # Update stop-loss via PositionManager
                # await self.position_manager.update_stop_loss(position_id, new_stop)
            
            # Option 2: Send alert for manual review
            await self._send_notification(
                f"Position {symbol} has large unrealized loss: ${unrealized_pnl:.2f}. "
                f"Consider reviewing or closing."
            )
            
        except Exception as e:
            logger.error(f"❌ Error handling large loss for {position_id}: {e}")
    
    async def _handle_high_portfolio_heat(self, event: Dict):
        """
        Handle high portfolio heat event.
        
        Action: Stop opening new positions until heat decreases.
        
        Args:
            event: High portfolio heat event
        """
        portfolio_heat = event.get('portfolio_heat')
        active_positions = event.get('active_positions')
        total_risk = event.get('total_risk')
        
        logger.warning(f"🔥 HIGH PORTFOLIO HEAT: {portfolio_heat:.2%} "
                      f"({active_positions} positions, ${total_risk:.2f} total risk)")
        
        try:
            # Pause new position openings
            # This could be done via a flag in TradingEngine or similar
            logger.info("🛑 Halting new position openings until portfolio heat decreases")
            
            # Optionally: Close smallest/worst performing positions
            logger.info("📊 Consider reducing exposure by closing underperforming positions")
            
            await self._send_notification(
                f"Portfolio heat at {portfolio_heat:.2%}. New positions halted."
            )
            
        except Exception as e:
            logger.error(f"❌ Error handling high portfolio heat: {e}")
    
    async def _handle_approaching_max_drawdown(self, event: Dict):
        """
        Handle approaching max drawdown event.
        
        Action: Halt new positions and prepare to close losing positions.
        
        Args:
            event: Approaching max drawdown event
        """
        current_drawdown = event.get('current_drawdown')
        max_drawdown = event.get('max_drawdown')
        
        logger.critical(f"🚨 APPROACHING MAX DRAWDOWN: {current_drawdown:.2%} "
                       f"(max: {max_drawdown:.2%})")
        
        try:
            # Halt all new trading
            logger.info("🛑 HALT ALL NEW TRADING")
            
            # Close most risky positions
            positions = self.portfolio_manager.get_open_positions()
            
            # Sort by unrealized PnL (worst first)
            losing_positions = sorted(
                positions.items(),
                key=lambda x: x[1].get('unrealized_pnl', 0)
            )
            
            # Close bottom 25% of positions
            num_to_close = max(1, len(losing_positions) // 4)
            logger.info(f"📤 Closing {num_to_close} worst performing positions")
            
            for pos_id, position in losing_positions[:num_to_close]:
                symbol = position.get('symbol', 'UNKNOWN')
                logger.info(f"   Closing {symbol} ({pos_id})")
                # await self.order_manager.close_position(pos_id, order_type='market')
            
            await self._send_notification(
                f"⚠️ CRITICAL: Drawdown at {current_drawdown:.2%}. "
                f"Closing {num_to_close} positions and halting trading."
            )
            
        except Exception as e:
            logger.error(f"❌ Error handling max drawdown: {e}")
    
    async def _handle_emergency_stop(self, event: Dict):
        """
        Handle emergency stop event.
        
        Action: Close ALL positions immediately.
        
        Args:
            event: Emergency stop event
        """
        reason = event.get('reason')
        affected_positions = event.get('affected_positions', [])
        
        logger.critical(f"🚨🚨🚨 EMERGENCY STOP: {reason}")
        logger.critical(f"        Closing {len(affected_positions)} positions")
        
        try:
            # Close all affected positions
            for pos_id in affected_positions:
                position = self.portfolio_manager.get_position(pos_id)
                if position:
                    symbol = position.get('symbol', 'UNKNOWN')
                    logger.critical(f"   ⚡ EMERGENCY CLOSE: {symbol} ({pos_id})")
                    # await self.order_manager.close_position(pos_id, order_type='market', priority='urgent')
            
            # Send critical notification
            await self._send_notification(
                f"🚨 EMERGENCY STOP: {reason}. All positions closed.",
                priority='critical'
            )
            
            logger.critical(f"✅ Emergency stop completed: {len(affected_positions)} positions closed")
            
        except Exception as e:
            logger.critical(f"❌ CRITICAL ERROR during emergency stop: {e}", exc_info=True)
    
    async def _send_notification(self, message: str, priority: str = 'normal'):
        """
        Send notification (e.g., Telegram, email).
        
        Args:
            message: Notification message
            priority: Priority level ('normal', 'critical')
        """
        # This is a placeholder - implement your notification system
        logger.info(f"📬 NOTIFICATION ({priority}): {message}")
        
        # Example: Send via Telegram bot
        # await self.telegram_bot.send_message(message, priority=priority)


async def main_example():
    """
    Example usage of RiskEventListener.
    
    This demonstrates the complete event-driven risk management workflow.
    """
    print("\n" + "="*80)
    print("RISK EVENT LISTENER EXAMPLE - PHASE 2 ARCHITECTURE")
    print("="*80 + "\n")
    
    # This is pseudo-code showing how to integrate the event listener
    # In real usage, you would have actual instances of these managers
    
    print("Step 1: Initialize components")
    print("-" * 40)
    # risk_manager = RiskManager(...)
    # portfolio_manager = PortfolioManager(risk_manager, ...)
    # risk_monitor = RealTimeRiskMonitor(risk_manager, ws_manager, portfolio_manager)
    # order_manager = OrderManager(...)
    # position_manager = PositionManager(...)
    
    print("Step 2: Create event listener")
    print("-" * 40)
    # listener = RiskEventListener(
    #     risk_monitor=risk_monitor,
    #     order_manager=order_manager,
    #     position_manager=position_manager,
    #     portfolio_manager=portfolio_manager
    # )
    
    print("Step 3: Start monitoring and listening")
    print("-" * 40)
    # await risk_monitor.start_risk_monitoring()
    # await listener.start_listening()
    
    print("Step 4: Event flow")
    print("-" * 40)
    print("RealTimeRiskMonitor detects risk condition")
    print("    ↓")
    print("Emits event to risk_events queue")
    print("    ↓")
    print("RiskEventListener receives event")
    print("    ↓")
    print("Takes appropriate action (close position, etc.)")
    
    print("\n" + "="*80)
    print("KEY BENEFITS OF PHASE 2 ARCHITECTURE:")
    print("="*80)
    print("✅ Separation of Concerns: Detection vs Action")
    print("✅ Testability: Components can be tested independently")
    print("✅ Flexibility: Easy to add new event types and handlers")
    print("✅ Maintainability: Clear responsibilities for each component")
    print("✅ Extensibility: Multiple listeners can consume the same events")
    print("\n")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main_example())
