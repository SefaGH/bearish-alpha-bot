# P&L Calculation Guide

## Overview

This guide documents the centralized P&L (Profit & Loss) calculation utilities in the Bearish Alpha Bot. All P&L calculations throughout the codebase now use these standardized functions to ensure consistency and maintainability.

## Module Location

```
src/utils/pnl_calculator.py
```

## Core Functions

### 1. `calculate_unrealized_pnl()`

Calculates unrealized P&L for open positions.

**Signature:**
```python
def calculate_unrealized_pnl(side: str, entry_price: float, 
                             current_price: float, amount: float) -> float
```

**Parameters:**
- `side`: Position side - accepts 'long', 'buy', 'short', or 'sell'
- `entry_price`: Entry price of the position
- `current_price`: Current market price
- `amount`: Position size/amount

**Returns:**
- Unrealized P&L in quote currency (positive = profit, negative = loss)

**Examples:**
```python
from utils.pnl_calculator import calculate_unrealized_pnl

# Long position with profit
pnl = calculate_unrealized_pnl('long', 50000, 51000, 0.1)
# Returns: 100.0 (profit of $100)

# Short position with profit
pnl = calculate_unrealized_pnl('short', 50000, 49000, 0.1)
# Returns: 100.0 (profit of $100)

# Long position with loss
pnl = calculate_unrealized_pnl('long', 50000, 49000, 0.1)
# Returns: -100.0 (loss of $100)
```

### 2. `calculate_realized_pnl()`

Calculates realized P&L for closed positions.

**Signature:**
```python
def calculate_realized_pnl(side: str, entry_price: float,
                           exit_price: float, amount: float) -> float
```

**Parameters:**
- `side`: Position side - accepts 'long', 'buy', 'short', or 'sell'
- `entry_price`: Entry price of the position
- `exit_price`: Exit price of the position
- `amount`: Position size/amount

**Returns:**
- Realized P&L in quote currency (positive = profit, negative = loss)

**Examples:**
```python
from utils.pnl_calculator import calculate_realized_pnl

# Close long position with profit
pnl = calculate_realized_pnl('long', 50000, 51500, 0.1)
# Returns: 150.0 (profit of $150)

# Close short position with loss
pnl = calculate_realized_pnl('short', 50000, 51000, 0.1)
# Returns: -100.0 (loss of $100)
```

### 3. `calculate_pnl_percentage()`

Converts P&L amount to percentage of initial position value.

**Signature:**
```python
def calculate_pnl_percentage(pnl: float, entry_price: float, amount: float) -> float
```

**Parameters:**
- `pnl`: Profit/Loss amount in quote currency
- `entry_price`: Entry price of the position
- `amount`: Position size/amount

**Returns:**
- P&L percentage (e.g., 2.0 for +2%, -3.5 for -3.5%)

**Examples:**
```python
from utils.pnl_calculator import calculate_pnl_percentage

# Calculate percentage for $100 profit on $5000 position
pnl_pct = calculate_pnl_percentage(100, 50000, 0.1)
# Returns: 2.0 (2% profit)

# Zero division protection
pnl_pct = calculate_pnl_percentage(100, 0, 0)
# Returns: 0.0 (safe handling of edge case)
```

### 4. `calculate_return_percentage()`

Calculates return percentage for a position based on price movement.

**Signature:**
```python
def calculate_return_percentage(entry_price: float, exit_price: float, side: str) -> float
```

**Parameters:**
- `entry_price`: Entry price of the position
- `exit_price`: Exit price of the position
- `side`: Position side - accepts 'long', 'buy', 'short', or 'sell'

**Returns:**
- Return percentage relative to entry price

**Examples:**
```python
from utils.pnl_calculator import calculate_return_percentage

# Long position return
ret = calculate_return_percentage(50000, 51000, 'long')
# Returns: 2.0 (2% return)

# Short position return
ret = calculate_return_percentage(50000, 49000, 'short')
# Returns: 2.0 (2% return)
```

### 5. `calculate_position_value()`

Calculates total position value at entry.

**Signature:**
```python
def calculate_position_value(entry_price: float, amount: float) -> float
```

**Parameters:**
- `entry_price`: Entry price of the position
- `amount`: Position size/amount

**Returns:**
- Total position value in quote currency

**Examples:**
```python
from utils.pnl_calculator import calculate_position_value

# Calculate position value
value = calculate_position_value(50000, 0.1)
# Returns: 5000.0 ($5000 position)
```

## Usage in Codebase

The P&L calculator functions are used in the following modules:

### Position Manager (`src/core/position_manager.py`)

Used in 3 locations:
1. **`monitor_position_pnl()`** - Calculates unrealized P&L for open positions
2. **`close_position()`** - Calculates realized P&L when closing positions
3. **`calculate_position_metrics()`** - Calculates P&L percentage for metrics

### Real-Time Risk Monitor (`src/core/realtime_risk.py`)

Used in `_check_unrealized_pnl()` to monitor real-time P&L against risk thresholds.

### Risk Manager (`src/core/risk_manager.py`)

Used in `monitor_position_risk()` to calculate unrealized P&L for risk assessment.

### Production Coordinator (`src/core/production_coordinator.py`)

Used in `_print_position_dashboard()` to display current P&L for all active positions.

## Implementation Details

### Calculation Logic

**For Long/Buy Positions:**
```python
unrealized_pnl = (current_price - entry_price) * amount
```

**For Short/Sell Positions:**
```python
unrealized_pnl = (entry_price - current_price) * amount
```

### Edge Case Handling

All functions include robust edge case handling:

1. **Zero Division Protection**: Returns 0.0 when position value is zero
2. **Negative Entry Price Protection**: Returns 0.0 for invalid entry prices
3. **Side Aliases**: Supports both 'long'/'buy' and 'short'/'sell' terminology

## Testing

Comprehensive unit tests are available in `tests/test_pnl_calculator.py`:

```bash
# Run tests with pytest
pytest tests/test_pnl_calculator.py -v

# Run specific test class
pytest tests/test_pnl_calculator.py::TestUnrealizedPnL -v
```

### Test Coverage

The test suite includes:
- ✅ Long position profit/loss scenarios
- ✅ Short position profit/loss scenarios
- ✅ Side alias validation ('buy'/'sell')
- ✅ Zero division protection
- ✅ Edge cases (zero prices, negative values)
- ✅ Integration scenarios (complete trade flows)

## Migration Notes

### Before (Duplicated Code)

```python
# This pattern appeared 6 times across the codebase
if side in ['long', 'buy']:
    unrealized_pnl = (current_price - entry_price) * amount
else:
    unrealized_pnl = (entry_price - current_price) * amount
```

### After (Centralized)

```python
from utils.pnl_calculator import calculate_unrealized_pnl

unrealized_pnl = calculate_unrealized_pnl(side, entry_price, current_price, amount)
```

## Benefits

1. **DRY Principle**: Single source of truth for P&L calculations
2. **Consistency**: All modules use identical calculation logic
3. **Testability**: Isolated unit tests for calculation logic
4. **Maintainability**: Bug fixes and improvements in one location
5. **Type Safety**: Clear function signatures with type hints
6. **Documentation**: Centralized calculation documentation

## Best Practices

### When to Use Each Function

- Use `calculate_unrealized_pnl()` for **open positions** during monitoring
- Use `calculate_realized_pnl()` when **closing positions**
- Use `calculate_pnl_percentage()` to **display P&L as percentage**
- Use `calculate_return_percentage()` for **performance analysis**
- Use `calculate_position_value()` for **risk calculations**

### Import Pattern

```python
from utils.pnl_calculator import (
    calculate_unrealized_pnl,
    calculate_realized_pnl,
    calculate_pnl_percentage
)
```

## Example: Complete Trade Flow

```python
from utils.pnl_calculator import (
    calculate_position_value,
    calculate_unrealized_pnl,
    calculate_pnl_percentage,
    calculate_realized_pnl,
    calculate_return_percentage
)

# Entry
entry_price = 50000
amount = 0.1
side = 'long'

# Calculate initial position value
position_value = calculate_position_value(entry_price, amount)
print(f"Position Value: ${position_value:.2f}")  # $5000.00

# During trade - monitor unrealized P&L
current_price = 51000
unrealized_pnl = calculate_unrealized_pnl(side, entry_price, current_price, amount)
print(f"Unrealized P&L: ${unrealized_pnl:.2f}")  # $100.00

# Calculate percentage
pnl_pct = calculate_pnl_percentage(unrealized_pnl, entry_price, amount)
print(f"P&L: {pnl_pct:+.2f}%")  # +2.00%

# Close trade
exit_price = 51500
realized_pnl = calculate_realized_pnl(side, entry_price, exit_price, amount)
print(f"Realized P&L: ${realized_pnl:.2f}")  # $150.00

# Calculate return
return_pct = calculate_return_percentage(entry_price, exit_price, side)
print(f"Return: {return_pct:+.2f}%")  # +3.00%
```

## Troubleshooting

### Import Errors

If you encounter `ModuleNotFoundError: No module named 'utils'`:

1. **Install the project in editable mode** (recommended for development):

### Calculation Discrepancies

If you notice calculation differences:

1. Verify the `side` parameter is correct ('long'/'buy' vs 'short'/'sell')
2. Check that `entry_price`, `current_price`, and `amount` values are correct
3. Ensure no fees or slippage are being double-counted

## Version History

- **v1.0.0** (2025-10-19): Initial release
  - Extracted from 6 duplicated locations
  - Added comprehensive unit tests
  - Created documentation

## Support

For questions or issues:
1. Check unit tests for usage examples
2. Review this documentation
3. Open an issue on GitHub

---

**Last Updated:** 2025-10-19  
**Module:** `src/utils/pnl_calculator.py`  
**Tests:** `tests/test_pnl_calculator.py`
