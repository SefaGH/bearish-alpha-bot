# Phase 2 Migration Checklist

## Overview

This checklist helps teams migrate from the old monolithic RiskManager architecture to the new Phase 2 layered architecture.

## Pre-Migration

### ✅ Preparation
- [ ] Read `PHASE2_ARCHITECTURE.md` documentation
- [ ] Review `examples/risk_event_listener_example.py`
- [ ] Understand the new component responsibilities
- [ ] Identify all code that interacts with RiskManager
- [ ] Plan migration strategy (gradual vs all-at-once)

### ✅ Testing Setup
- [ ] Ensure all existing tests pass (38/38 for risk_management)
- [ ] Create backup branch before migration
- [ ] Set up test environment for validation

---

## Migration Steps

### Step 1: Update Component Initialization

#### Old Pattern:
```python
# Only RiskManager held state
risk_manager = RiskManager(
    portfolio_value=10000,
    risk_config=risk_config
)
```

#### New Pattern:
```python
# Create both RiskManager and PortfolioManager
risk_manager = RiskManager(
    portfolio_value=10000,
    risk_config=risk_config
)

portfolio_manager = PortfolioManager(
    risk_manager=risk_manager,
    performance_monitor=performance_monitor,
    websocket_manager=websocket_manager
)
```

**Checklist:**
- [ ] Identify all RiskManager instantiations
- [ ] Add PortfolioManager creation after each RiskManager
- [ ] Pass necessary dependencies to PortfolioManager
- [ ] Update initialization logs/documentation

---

### Step 2: Migrate State Management

#### Position Registration

**Old:**
```python
risk_manager.register_position(position_id, position_data)
```

**New:**
```python
portfolio_manager.register_position(position_id, position_data)
```

**Checklist:**
- [ ] Find all `risk_manager.register_position()` calls
- [ ] Replace with `portfolio_manager.register_position()`
- [ ] Verify position data structure is compatible
- [ ] Test position registration

#### Position Updates

**Old:**
```python
risk_manager.update_position_price(position_id, current_price)
```

**New:**
```python
portfolio_manager.update_position_price(position_id, current_price)
```

**Checklist:**
- [ ] Find all `risk_manager.update_position_price()` calls
- [ ] Replace with `portfolio_manager.update_position_price()`
- [ ] Test price updates

#### Position Closure

**Old:**
```python
risk_manager.close_position(position_id, exit_price, realized_pnl)
```

**New:**
```python
portfolio_manager.close_position(position_id, exit_price, realized_pnl)
```

**Checklist:**
- [ ] Find all `risk_manager.close_position()` calls
- [ ] Replace with `portfolio_manager.close_position()`
- [ ] Verify P&L calculations still work
- [ ] Test drawdown updates

#### State Queries

**Old:**
```python
portfolio_value = risk_manager.portfolio_value
active_positions = risk_manager.active_positions
drawdown = risk_manager.current_drawdown
```

**New:**
```python
portfolio_value = portfolio_manager.get_current_equity()
active_positions = portfolio_manager.get_open_positions()
drawdown = portfolio_manager.get_current_drawdown()
```

**Checklist:**
- [ ] Find all direct state property accesses
- [ ] Replace with PortfolioManager method calls
- [ ] Update any caching logic if needed
- [ ] Test state queries

---

### Step 3: Update Risk Decision Calls

#### Position Validation

**Old:**
```python
is_valid, reason, metrics = await risk_manager.validate_new_position(
    signal=signal,
    current_portfolio={}
)
```

**New:**
```python
is_valid, reason, metrics = await risk_manager.validate_new_position(
    signal=signal,
    portfolio_manager=portfolio_manager
)
```

**Checklist:**
- [ ] Find all `validate_new_position()` calls
- [ ] Add `portfolio_manager` parameter
- [ ] Remove `current_portfolio` parameter (or set to None)
- [ ] Test position validation logic
- [ ] Verify risk metrics are correct

#### Position Sizing

**Old:**
```python
size = await risk_manager.calculate_position_size(
    signal=signal,
    market_regime=regime
)
```

**New:**
```python
size = await risk_manager.calculate_position_size(
    signal=signal,
    market_regime=regime,
    portfolio_manager=portfolio_manager
)
```

**Checklist:**
- [ ] Find all `calculate_position_size()` calls
- [ ] Add `portfolio_manager` parameter
- [ ] Test position sizing calculations
- [ ] Verify risk-adjusted sizing works

#### Portfolio Summary

**Old:**
```python
summary = risk_manager.get_portfolio_summary()
```

**New:**
```python
summary = risk_manager.get_portfolio_summary(portfolio_manager)
```

**Checklist:**
- [ ] Find all `get_portfolio_summary()` calls
- [ ] Add `portfolio_manager` parameter
- [ ] Verify summary includes all expected fields
- [ ] Test summary accuracy

---

### Step 4: Migrate RealTimeRiskMonitor

#### Initialization

**Old:**
```python
monitor = RealTimeRiskMonitor(
    risk_manager=risk_manager,
    websocket_manager=websocket_manager
)
```

**New:**
```python
monitor = RealTimeRiskMonitor(
    risk_manager=risk_manager,
    websocket_manager=websocket_manager,
    portfolio_manager=portfolio_manager  # NEW!
)
```

**Checklist:**
- [ ] Find all RealTimeRiskMonitor instantiations
- [ ] Add `portfolio_manager` parameter
- [ ] Test monitor initialization

#### Remove Direct Actions

**Old (if present):**
```python
# Monitor might have code like:
if stop_loss_triggered:
    await self.order_manager.close_position(pos_id)
```

**New:**
```python
# Monitor only emits events:
if stop_loss_triggered:
    event = {
        'event_type': 'stop_loss_triggered',
        'position_id': pos_id,
        'action_required': 'close_position'
    }
    await self.risk_events.put(event)
```

**Checklist:**
- [ ] Review RealTimeRiskMonitor code for direct actions
- [ ] Remove any position closing logic
- [ ] Ensure events are emitted instead
- [ ] Verify event structure matches documentation

---

### Step 5: Implement Event Listeners

#### Create Event Listener Class

**Use the example as a template:**
```python
# See: examples/risk_event_listener_example.py

class RiskEventListener:
    def __init__(self, risk_monitor, order_manager, portfolio_manager):
        self.risk_monitor = risk_monitor
        self.order_manager = order_manager
        self.portfolio_manager = portfolio_manager
    
    async def start_listening(self):
        # Consume events from risk_monitor.risk_events
        ...
```

**Checklist:**
- [ ] Create RiskEventListener class (or similar)
- [ ] Implement event handling for each event type
- [ ] Add proper error handling
- [ ] Add logging for debugging
- [ ] Test each event handler individually

#### Event Handlers to Implement

**Required Handlers:**
- [ ] `stop_loss_triggered` → Close position
- [ ] `large_unrealized_loss` → Review/adjust position
- [ ] `high_portfolio_heat` → Halt new positions
- [ ] `approaching_max_drawdown` → Emergency measures
- [ ] `emergency_stop` → Close all positions

**Optional Handlers (based on needs):**
- [ ] Notification system integration
- [ ] Database logging
- [ ] Analytics/metrics collection

#### Integration

**Checklist:**
- [ ] Instantiate event listener in main application
- [ ] Start listener alongside monitor: `await listener.start_listening()`
- [ ] Verify events are being consumed
- [ ] Test end-to-end event flow
- [ ] Add graceful shutdown logic

---

### Step 6: Testing

#### Unit Tests

**Checklist:**
- [ ] Test PortfolioManager state management
- [ ] Test RiskManager decision logic (with mock PortfolioManager)
- [ ] Test RealTimeRiskMonitor event emission
- [ ] Test event listener handlers
- [ ] Verify no regressions in existing tests

#### Integration Tests

**Checklist:**
- [ ] Test full position lifecycle (open → monitor → close)
- [ ] Test stop-loss event flow (detect → emit → close)
- [ ] Test multiple concurrent events
- [ ] Test error handling at each layer
- [ ] Test with real WebSocket data (if possible)

#### Performance Tests

**Checklist:**
- [ ] Measure event queue latency
- [ ] Test with multiple positions
- [ ] Test event listener throughput
- [ ] Verify no memory leaks
- [ ] Check CPU usage is acceptable

---

## Post-Migration

### Verification

**Checklist:**
- [ ] All tests pass (38/38 risk_management tests minimum)
- [ ] No regression in trading functionality
- [ ] Event flow works correctly
- [ ] Logging shows clear separation of concerns
- [ ] Performance is acceptable

### Documentation

**Checklist:**
- [ ] Update internal documentation
- [ ] Document any custom event types added
- [ ] Update API documentation
- [ ] Create runbook for operations team
- [ ] Add migration notes to CHANGELOG

### Monitoring

**Checklist:**
- [ ] Add metrics for event queue depth
- [ ] Monitor event processing latency
- [ ] Track event handler failures
- [ ] Set up alerts for critical events
- [ ] Create dashboard for Phase 2 components

---

## Rollback Plan

### If Issues Occur

**Checklist:**
- [ ] Document the issue clearly
- [ ] Check if it's a Phase 2 specific issue
- [ ] Review recent changes to identify cause
- [ ] Consider temporary rollback if critical

### Rollback Steps

1. **Partial Rollback** (recommended):
   - Keep PortfolioManager but revert specific components
   - Use backward compatibility mode
   - Fix issue, then re-migrate

2. **Full Rollback** (if necessary):
   - Revert to backup branch
   - Remove PortfolioManager references
   - Remove event listener
   - Restore direct RiskManager state access

**Note:** Full rollback should be rare due to backward compatibility.

---

## Timeline Estimation

### Small Codebase (< 10 RiskManager usages)
- **Preparation:** 2 hours
- **Migration:** 4-6 hours
- **Testing:** 4 hours
- **Total:** 1-2 days

### Medium Codebase (10-50 usages)
- **Preparation:** 4 hours
- **Migration:** 1-2 days
- **Testing:** 1 day
- **Total:** 3-5 days

### Large Codebase (> 50 usages)
- **Preparation:** 1 day
- **Migration:** 3-5 days
- **Testing:** 2-3 days
- **Total:** 1-2 weeks

**Tip:** Consider gradual migration over time rather than big-bang approach.

---

## Common Issues & Solutions

### Issue 1: PortfolioManager Not Available

**Problem:**
```python
AttributeError: 'NoneType' object has no attribute 'get_current_equity'
```

**Solution:**
Ensure PortfolioManager is created and passed to all components:
```python
# Create it
portfolio_manager = PortfolioManager(risk_manager, performance_monitor)

# Pass it to RiskManager calls
risk_manager.validate_new_position(signal, portfolio_manager=portfolio_manager)

# Pass it to RealTimeRiskMonitor
monitor = RealTimeRiskMonitor(rm, ws, portfolio_manager)
```

---

### Issue 2: Positions Registered in Wrong Place

**Problem:**
Position registered in RiskManager instead of PortfolioManager.

**Solution:**
```python
# WRONG:
risk_manager.register_position(pos_id, data)

# CORRECT:
portfolio_manager.register_position(pos_id, data)
```

---

### Issue 3: Events Not Being Processed

**Problem:**
Risk events emitted but no action taken.

**Solution:**
Ensure event listener is started:
```python
listener = RiskEventListener(monitor, order_manager, portfolio_manager)
await listener.start_listening()  # Don't forget this!
```

---

### Issue 4: State Inconsistency

**Problem:**
PortfolioManager and RiskManager show different state.

**Solution:**
Always use PortfolioManager as single source of truth:
```python
# DON'T:
value_from_risk = risk_manager.portfolio_value
value_from_portfolio = portfolio_manager.get_current_equity()
# These might differ!

# DO:
value = portfolio_manager.get_current_equity()  # Single source of truth
```

---

## Support & Resources

- **Documentation:** `docs/PHASE2_ARCHITECTURE.md`
- **Example:** `examples/risk_event_listener_example.py`
- **Tests:** `tests/test_risk_management.py`
- **Backward Compatibility:** All old code continues to work

---

## Success Criteria

Migration is complete when:

✅ All tests pass (100% pass rate)
✅ PortfolioManager is single source of truth for state
✅ RiskManager queries PortfolioManager (not internal state)
✅ RealTimeRiskMonitor emits events (doesn't take actions)
✅ Event listener handles all critical events
✅ No performance degradation
✅ Clear logging shows component separation
✅ Team understands new architecture

---

## Next Steps After Migration

1. **Monitor Production:**
   - Watch for any issues in live trading
   - Monitor event queue metrics
   - Verify event processing latency

2. **Optimize:**
   - Fine-tune event handling
   - Optimize state queries if needed
   - Add caching if beneficial

3. **Enhance:**
   - Add new event types
   - Create specialized event listeners
   - Improve monitoring/alerting

4. **Phase 3 Planning:**
   - Remove deprecated RiskManager properties
   - Add advanced features
   - Further optimize architecture

Good luck with your migration! 🚀
