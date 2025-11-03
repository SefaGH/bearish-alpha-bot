# Phase 2 Architecture: Separation of Responsibilities and Layered Architecture

## Overview

Phase 2 implements a clean separation of concerns in the risk management system, transforming it from a monolithic architecture to a layered, event-driven design that follows the Single Responsibility Principle (SRP).

## Before Phase 2 (Monolithic Architecture)

### Problems

```
┌─────────────────────────────────┐
│       RiskManager               │
│  (God Object - Too Many Jobs)   │
│                                 │
│  ❌ Tracks portfolio state      │
│  ❌ Makes risk decisions        │
│  ❌ Monitors positions          │
│  ❌ Closes positions            │
│  ❌ Manages drawdowns           │
│  ❌ Calculates exposure         │
└─────────────────────────────────┘
```

**Issues:**
- **Testing Nightmare**: Can't test risk logic without portfolio state
- **Tight Coupling**: Everything depends on RiskManager
- **Hard to Maintain**: Changes to one feature affect everything
- **No Event System**: Direct action taking prevents flexibility
- **State Confusion**: Who owns what data?

## After Phase 2 (Layered Architecture)

### Solution

```
┌──────────────────────────────────────────────────────────────────┐
│                         LAYERED ARCHITECTURE                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────┐  ┌─────────────────────┐               │
│  │  PortfolioManager   │  │    RiskManager      │               │
│  │   (Accountant)      │  │   (Risk Analyst)    │               │
│  │                     │  │                     │               │
│  │ ✅ "What do we      │  │ ✅ "Is this safe?"  │               │
│  │    have?"           │  │                     │               │
│  │                     │  │ • Validation        │               │
│  │ • Portfolio value   │  │ • Calculations      │               │
│  │ • Positions         │  │ • Decisions         │               │
│  │ • Drawdown          │  │ • No state!         │               │
│  │ • Exposure          │  │                     │               │
│  └─────────────────────┘  └─────────────────────┘               │
│           │                         │                            │
│           │                         │                            │
│           v                         v                            │
│  ┌────────────────────────────────────────────────┐             │
│  │        RealTimeRiskMonitor                      │             │
│  │             (Watchdog)                          │             │
│  │                                                 │             │
│  │ ✅ "What's happening?"                          │             │
│  │                                                 │             │
│  │ • Monitors positions                            │             │
│  │ • Detects risk events                           │             │
│  │ • EMITS events (asyncio.Queue)                  │             │
│  │ • NO direct actions!                            │             │
│  └────────────────────────────────────────────────┘             │
│                         │                                        │
│                         │ Events                                 │
│                         v                                        │
│  ┌────────────────────────────────────────────────┐             │
│  │        Risk Event Listeners                     │             │
│  │          (Action Takers)                        │             │
│  │                                                 │             │
│  │ ✅ "What should we do?"                         │             │
│  │                                                 │             │
│  │ • Listen to events                              │             │
│  │ • Take actions (close positions, etc.)          │             │
│  │ • Can be multiple listeners!                    │             │
│  └────────────────────────────────────────────────┘             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### 1. PortfolioManager (The Accountant)

**Role:** Single source of truth for portfolio state

**Responsibilities:**
- Track portfolio value (current, peak)
- Manage active positions (register, update, close)
- Calculate metrics (exposure, drawdown, available capital)
- Answer: "What do we have?"

**Key Methods:**
```python
# State Queries
portfolio_manager.get_current_equity()       # Current portfolio value
portfolio_manager.get_peak_equity()          # Peak value (for drawdown)
portfolio_manager.get_current_drawdown()     # Current drawdown %
portfolio_manager.get_open_positions()       # All active positions
portfolio_manager.get_position(pos_id)       # Specific position
portfolio_manager.get_total_exposure()       # Total notional exposure
portfolio_manager.get_available_capital()    # Unallocated capital

# State Modifications
portfolio_manager.register_position(pos_id, data)
portfolio_manager.update_position_price(pos_id, price)
portfolio_manager.close_position(pos_id, exit_price, pnl)
```

**Example:**
```python
# Creating PortfolioManager with initial capital
portfolio_manager = PortfolioManager(
    risk_manager=risk_manager,
    performance_monitor=performance_monitor
)

# Registering a new position
portfolio_manager.register_position('pos_1', {
    'symbol': 'BTC/USDT:USDT',
    'entry_price': 50000,
    'stop_loss': 49000,
    'size': 0.1,
    'side': 'long',
    'risk_amount': 100
})

# Querying state
equity = portfolio_manager.get_current_equity()  # $10,000
positions = portfolio_manager.get_open_positions()  # {'pos_1': {...}}
exposure = portfolio_manager.get_total_exposure()  # $5,000
```

---

### 2. RiskManager (The Risk Analyst)

**Role:** Stateless decision and validation engine

**Responsibilities:**
- Validate if positions meet risk criteria
- Calculate optimal position sizes
- Assess portfolio risk metrics
- Answer: "Is this safe?"
- **NO STATE MANAGEMENT** - queries PortfolioManager

**Key Methods:**
```python
# All methods now accept optional portfolio_manager parameter

# Validation (Decision Making)
is_valid, reason, metrics = await risk_manager.validate_new_position(
    signal=signal,
    portfolio_manager=portfolio_manager  # NEW: Query state from here
)

# Position Sizing (Calculation)
size = await risk_manager.calculate_position_size(
    signal=signal,
    portfolio_manager=portfolio_manager  # NEW
)

# Portfolio Summary (Analysis)
summary = risk_manager.get_portfolio_summary(
    portfolio_manager=portfolio_manager  # NEW
)
```

**Example:**
```python
# Validating a new position
signal = {
    'symbol': 'BTC/USDT:USDT',
    'entry': 50000,
    'stop': 49000,
    'target': 52000,
    'position_size': 0.02,
    'side': 'long'
}

# PHASE 2: Pass PortfolioManager for state queries
is_valid, reason, metrics = await risk_manager.validate_new_position(
    signal=signal,
    portfolio_manager=portfolio_manager  # RiskManager queries this
)

if is_valid:
    print("✅ Position approved")
    print(f"Risk amount: ${metrics['risk_amount']:.2f}")
    print(f"R/R ratio: {metrics['risk_reward_ratio']:.2f}")
else:
    print(f"❌ Position rejected: {reason}")
```

**Backward Compatibility:**
```python
# OLD WAY (still works, but deprecated)
is_valid, reason, metrics = await risk_manager.validate_new_position(
    signal=signal,
    current_portfolio={}  # Falls back to deprecated internal state
)

# NEW WAY (Phase 2 - preferred)
is_valid, reason, metrics = await risk_manager.validate_new_position(
    signal=signal,
    portfolio_manager=portfolio_manager  # Clean separation!
)
```

---

### 3. RealTimeRiskMonitor (The Watchdog)

**Role:** Event-driven risk monitoring system

**Responsibilities:**
- Monitor positions in real-time (via WebSocket feeds)
- Detect risk conditions (stop-loss, large losses, etc.)
- **EMIT events** (not take actions!)
- Answer: "What's happening?"

**Event Types:**
```python
{
    'event_type': 'stop_loss_triggered',     # Stop-loss hit
    'event_type': 'large_unrealized_loss',   # Position losing too much
    'event_type': 'high_portfolio_heat',     # Total risk too high
    'event_type': 'approaching_max_drawdown', # Near drawdown limit
    'event_type': 'emergency_stop'           # Critical situation
}
```

**Event Structure:**
```python
{
    'event_type': 'stop_loss_triggered',
    'position_id': 'pos_123',
    'symbol': 'BTC/USDT:USDT',
    'trigger_price': 48500,
    'stop_loss': 49000,
    'side': 'long',
    'timestamp': datetime.now(timezone.utc),
    'severity': 'high',
    'action_required': 'close_position'  # Suggestion for listeners
}
```

**Example:**
```python
# Initialize with PortfolioManager
monitor = RealTimeRiskMonitor(
    risk_manager=risk_manager,
    websocket_manager=ws_manager,
    portfolio_manager=portfolio_manager  # NEW: For state queries
)

# Start monitoring
await monitor.start_risk_monitoring()

# Monitor emits events automatically when risks detected
# Events go into monitor.risk_events queue

# Check for events
events = await monitor.get_risk_events(count=10)
for event in events:
    print(f"Event: {event['event_type']}")
    print(f"Severity: {event['severity']}")
    print(f"Action needed: {event['action_required']}")
```

---

### 4. Risk Event Listeners (The Action Takers)

**Role:** Consume events and execute actions

**Responsibilities:**
- Listen to `RealTimeRiskMonitor.risk_events` queue
- Take appropriate actions based on event type
- Can be multiple listeners (e.g., one for trading, one for notifications)
- Answer: "What should we do?"

**Example Implementation:**
```python
class RiskEventListener:
    def __init__(self, risk_monitor, order_manager, portfolio_manager):
        self.risk_monitor = risk_monitor
        self.order_manager = order_manager
        self.portfolio_manager = portfolio_manager
        self.listener_active = False
    
    async def start_listening(self):
        """Start consuming events from risk_monitor.risk_events"""
        self.listener_active = True
        while self.listener_active:
            # Get event from queue
            event = await self.risk_monitor.risk_events.get()
            
            # Handle based on type
            if event['event_type'] == 'stop_loss_triggered':
                await self._close_position(event['position_id'])
            elif event['event_type'] == 'high_portfolio_heat':
                await self._halt_new_positions()
            # ... handle other event types
    
    async def _close_position(self, position_id):
        """Close a position via OrderManager"""
        position = self.portfolio_manager.get_position(position_id)
        await self.order_manager.close_position(
            position_id,
            order_type='market'
        )
        logger.info(f"✅ Position {position_id} closed")
```

**Full Example:**
```python
# Create listener
listener = RiskEventListener(
    risk_monitor=monitor,
    order_manager=order_manager,
    portfolio_manager=portfolio_manager
)

# Start listening (runs in background)
await listener.start_listening()

# Now when monitor detects a stop-loss:
# 1. Monitor emits event to risk_events queue
# 2. Listener picks up event
# 3. Listener closes position via OrderManager
# 4. Clean separation of detection vs action!
```

---

## Data Flow Examples

### Example 1: Opening a New Position

```
┌──────────────────────────────────────────────────────────────┐
│                   OPENING A NEW POSITION                      │
└──────────────────────────────────────────────────────────────┘

1. Trading Signal Arrives
   ↓
2. Query Portfolio State
   TradingEngine → PortfolioManager.get_current_equity()
   TradingEngine → PortfolioManager.get_open_positions()
   ↓
3. Validate Position with RiskManager
   TradingEngine → RiskManager.validate_new_position(
       signal=signal,
       portfolio_manager=portfolio_manager
   )
   ↓
   RiskManager queries PortfolioManager:
   - portfolio_manager.get_current_equity()
   - portfolio_manager.get_total_exposure()
   - portfolio_manager.get_current_drawdown()
   ↓
   RiskManager returns: is_valid=True, metrics={}
   ↓
4. Open Position
   TradingEngine → OrderManager.place_order(...)
   ↓
5. Register Position in Portfolio
   TradingEngine → PortfolioManager.register_position(pos_id, data)
   ↓
6. Start Real-Time Monitoring
   RealTimeRiskMonitor detects new position
   Subscribes to price updates
```

### Example 2: Stop-Loss Trigger Event Flow

```
┌──────────────────────────────────────────────────────────────┐
│              STOP-LOSS TRIGGER EVENT FLOW                     │
└──────────────────────────────────────────────────────────────┘

1. Price Update via WebSocket
   WebSocketManager → RealTimeRiskMonitor.on_price_update()
   ↓
2. Monitor Checks Position
   RealTimeRiskMonitor queries:
   - portfolio_manager.get_open_positions()
   ↓
   Detects: price <= stop_loss
   ↓
3. Monitor EMITS Event (No Action!)
   event = {
       'event_type': 'stop_loss_triggered',
       'position_id': 'pos_123',
       'trigger_price': 48500,
       'action_required': 'close_position'
   }
   ↓
   RealTimeRiskMonitor.risk_events.put(event)
   ↓
4. Event Listener Receives Event
   RiskEventListener picks up event from queue
   ↓
5. Listener Takes Action
   RiskEventListener → OrderManager.close_position(pos_id)
   ↓
6. Update Portfolio State
   OrderManager → PortfolioManager.close_position(
       pos_id,
       exit_price,
       realized_pnl
   )
   ↓
   PortfolioManager updates:
   - portfolio_value
   - current_drawdown
   - removes position from active_positions
```

---

## Migration Guide

### For Existing Code

**Old Pattern (Pre-Phase 2):**
```python
# RiskManager held state
risk_manager = RiskManager(portfolio_value=10000, risk_config=config)

# Validate position (uses internal state)
is_valid, reason, metrics = await risk_manager.validate_new_position(
    signal=signal,
    current_portfolio={}
)

# Register position in RiskManager
risk_manager.register_position(pos_id, data)

# Close position in RiskManager
risk_manager.close_position(pos_id, exit_price, pnl)

# RealTimeRiskMonitor closes positions directly
monitor = RealTimeRiskMonitor(risk_manager, ws_manager)
# Monitor would close positions when stop-loss hit (tight coupling!)
```

**New Pattern (Phase 2):**
```python
# Step 1: Create managers
risk_manager = RiskManager(portfolio_value=10000, risk_config=config)
portfolio_manager = PortfolioManager(risk_manager, performance_monitor)

# Step 2: RiskManager queries PortfolioManager
is_valid, reason, metrics = await risk_manager.validate_new_position(
    signal=signal,
    portfolio_manager=portfolio_manager  # Pass state manager!
)

# Step 3: Register position in PortfolioManager (single source of truth)
portfolio_manager.register_position(pos_id, data)

# Step 4: Close position via PortfolioManager
portfolio_manager.close_position(pos_id, exit_price, pnl)

# Step 5: Event-driven monitoring
monitor = RealTimeRiskMonitor(
    risk_manager,
    ws_manager,
    portfolio_manager  # Pass state manager!
)

# Step 6: Create event listener
listener = RiskEventListener(monitor, order_manager, portfolio_manager)
await listener.start_listening()

# Now monitor EMITS events, listener TAKES actions
# Clean separation of concerns!
```

---

## Benefits of Phase 2 Architecture

### 1. Testability
```python
# BEFORE: Hard to test RiskManager logic without full state setup
# AFTER: Easy to test with mock PortfolioManager

# Test position validation in isolation
mock_portfolio = Mock()
mock_portfolio.get_current_equity.return_value = 10000
mock_portfolio.get_total_exposure.return_value = 5000

is_valid, reason, metrics = await risk_manager.validate_new_position(
    signal=test_signal,
    portfolio_manager=mock_portfolio
)

assert is_valid == True
```

### 2. Flexibility
```python
# Multiple event listeners can consume the same events!

# Listener 1: Takes trading actions
trading_listener = TradingActionListener(...)

# Listener 2: Sends notifications
notification_listener = NotificationListener(...)

# Listener 3: Logs to database
logging_listener = DatabaseLogger(...)

# All listen to the same risk_events queue
await trading_listener.start_listening()
await notification_listener.start_listening()
await logging_listener.start_listening()
```

### 3. Maintainability
```python
# Clear responsibilities:
# - Need to change position tracking? → Edit PortfolioManager only
# - Need to change risk logic? → Edit RiskManager only
# - Need to change monitoring? → Edit RealTimeRiskMonitor only
# - Need to add action? → Add new event listener

# No cascading changes!
```

### 4. Extensibility
```python
# Easy to add new event types

# In RealTimeRiskMonitor
async def _check_correlation_risk(self):
    if correlation > 0.9:
        event = {
            'event_type': 'high_correlation_risk',  # NEW EVENT TYPE
            'correlation': correlation,
            'action_required': 'diversify'
        }
        await self.risk_events.put(event)

# In RiskEventListener, add handler
async def _handle_high_correlation_risk(self, event):
    # Handle new event type
    pass
```

---

## Testing Strategy

### Unit Tests

**PortfolioManager Tests:**
```python
def test_portfolio_manager_state():
    """Test PortfolioManager manages state correctly"""
    pm = PortfolioManager(risk_manager, performance_monitor)
    
    # Test equity tracking
    assert pm.get_current_equity() == 10000
    
    # Test position registration
    pm.register_position('pos_1', {...})
    assert 'pos_1' in pm.get_open_positions()
    
    # Test position closure and P&L
    pm.close_position('pos_1', 51000, 100)
    assert pm.get_current_equity() == 10100
```

**RiskManager Tests (Stateless):**
```python
@pytest.mark.asyncio
async def test_risk_manager_validation():
    """Test RiskManager validation logic (no state)"""
    rm = RiskManager(portfolio_value=10000, risk_config=config)
    
    # Create mock PortfolioManager
    mock_pm = Mock()
    mock_pm.get_current_equity.return_value = 10000
    mock_pm.get_total_exposure.return_value = 5000
    mock_pm.get_open_positions.return_value = {}
    
    # Test validation (pure logic, no side effects)
    is_valid, reason, metrics = await rm.validate_new_position(
        signal=signal,
        portfolio_manager=mock_pm
    )
    
    assert is_valid == True
    # RiskManager doesn't modify any state!
```

**RealTimeRiskMonitor Tests (Event Emission):**
```python
@pytest.mark.asyncio
async def test_monitor_emits_stop_loss_event():
    """Test monitor emits events instead of taking actions"""
    monitor = RealTimeRiskMonitor(rm, ws_manager, portfolio_manager)
    
    # Register position
    portfolio_manager.register_position('pos_1', {
        'symbol': 'BTC/USDT:USDT',
        'stop_loss': 49000,
        'side': 'long'
    })
    
    # Price breaches stop
    await monitor.on_price_update('BTC/USDT:USDT', {'last': 48500})
    
    # Check event was emitted
    assert monitor.risk_events.qsize() == 1
    event = await monitor.risk_events.get()
    assert event['event_type'] == 'stop_loss_triggered'
    
    # Position still active (monitor doesn't close it!)
    assert 'pos_1' in portfolio_manager.get_open_positions()
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_full_event_flow():
    """Test complete event flow from detection to action"""
    # Setup
    portfolio_manager = PortfolioManager(...)
    risk_manager = RiskManager(...)
    monitor = RealTimeRiskMonitor(rm, ws_manager, portfolio_manager)
    listener = RiskEventListener(monitor, order_manager, portfolio_manager)
    
    # Register position
    portfolio_manager.register_position('pos_1', {...})
    
    # Start monitoring and listening
    await monitor.start_risk_monitoring()
    await listener.start_listening()
    
    # Trigger stop-loss
    await monitor.on_price_update('BTC/USDT:USDT', {'last': 48500})
    
    # Wait for event processing
    await asyncio.sleep(0.1)
    
    # Verify position was closed by listener
    assert 'pos_1' not in portfolio_manager.get_open_positions()
```

---

## Backward Compatibility

All Phase 2 changes maintain **100% backward compatibility**:

```python
# OLD CODE (still works):
risk_manager = RiskManager(portfolio_value=10000, risk_config=config)
is_valid, reason, metrics = await risk_manager.validate_new_position(
    signal=signal,
    current_portfolio={}
)
risk_manager.register_position(pos_id, data)

# NEW CODE (Phase 2 - preferred):
portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
is_valid, reason, metrics = await risk_manager.validate_new_position(
    signal=signal,
    portfolio_manager=portfolio_manager
)
portfolio_manager.register_position(pos_id, data)
```

**Deprecation Plan:**
- Phase 2: Deprecated properties kept, warnings logged
- Phase 3: Migrate all code to use PortfolioManager
- Phase 4: Remove deprecated properties

---

## Summary

### Key Takeaways

1. **PortfolioManager** = Single source of truth for state
2. **RiskManager** = Stateless decision engine
3. **RealTimeRiskMonitor** = Event emitter (no actions!)
4. **Event Listeners** = Action takers

### Architecture Principles

- ✅ Single Responsibility Principle (SRP)
- ✅ Separation of Concerns
- ✅ Event-Driven Design
- ✅ Dependency Injection
- ✅ Testability First

### Next Steps (Phase 3)

1. Implement production event listeners
2. Migrate all code to use PortfolioManager
3. Remove deprecated RiskManager properties
4. Add more event types as needed
5. Enhance monitoring capabilities

---

## Questions & Support

For questions about Phase 2 architecture:
1. See `examples/risk_event_listener_example.py` for full example
2. Check test files for usage patterns
3. Review method docstrings for parameter details

Phase 2 provides a solid foundation for future enhancements while maintaining backward compatibility. 🚀
