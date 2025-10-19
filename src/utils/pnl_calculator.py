"""
P&L Calculation Utilities

Centralized P&L calculation functions to ensure consistency across the codebase.
All P&L calculations should use these standardized functions to prevent calculation drift.
"""


def calculate_unrealized_pnl(side: str, entry_price: float, 
                             current_price: float, amount: float) -> float:
    """
    Calculate unrealized P&L for an open position.
    
    Args:
        side: Position side ('long', 'buy', 'short', 'sell')
        entry_price: Entry price of the position
        current_price: Current market price
        amount: Position size/amount
        
    Returns:
        Unrealized P&L in quote currency (positive = profit, negative = loss)
        
    Examples:
        >>> calculate_unrealized_pnl('long', 50000, 51000, 0.1)
        100.0
        >>> calculate_unrealized_pnl('short', 50000, 49000, 0.1)
        100.0
    """
    if side in ['long', 'buy']:
        return (current_price - entry_price) * amount
    else:  # short, sell
        return (entry_price - current_price) * amount


def calculate_realized_pnl(side: str, entry_price: float,
                           exit_price: float, amount: float) -> float:
    """
    Calculate realized P&L for a closed position.
    
    Args:
        side: Position side ('long', 'buy', 'short', 'sell')
        entry_price: Entry price of the position
        exit_price: Exit price of the position
        amount: Position size/amount
        
    Returns:
        Realized P&L in quote currency (positive = profit, negative = loss)
        
    Examples:
        >>> calculate_realized_pnl('long', 50000, 51000, 0.1)
        100.0
        >>> calculate_realized_pnl('short', 50000, 51000, 0.1)
        -100.0
    """
    return calculate_unrealized_pnl(side, entry_price, exit_price, amount)


def calculate_pnl_percentage(pnl: float, entry_price: float, amount: float) -> float:
    """
    Calculate P&L as percentage of initial position value.
    
    Args:
        pnl: Profit/Loss amount in quote currency
        entry_price: Entry price of the position
        amount: Position size/amount
        
    Returns:
        P&L percentage (e.g., 2.0 for +2%, -3.5 for -3.5%)
        
    Examples:
        >>> calculate_pnl_percentage(100, 50000, 0.1)
        2.0
        >>> calculate_pnl_percentage(-150, 50000, 0.1)
        -3.0
    """
    position_value = entry_price * amount
    return (pnl / position_value) * 100 if position_value > 0 else 0.0


def calculate_return_percentage(entry_price: float, exit_price: float, side: str) -> float:
    """
    Calculate return percentage for a position.
    
    Args:
        entry_price: Entry price of the position
        exit_price: Exit price of the position
        side: Position side ('long', 'buy', 'short', 'sell')
        
    Returns:
        Return percentage relative to entry price
        
    Examples:
        >>> calculate_return_percentage(50000, 51000, 'long')
        2.0
        >>> calculate_return_percentage(50000, 51000, 'short')
        -2.0
    """
    if entry_price <= 0:
        return 0.0
    if side in ['long', 'buy']:
        return ((exit_price - entry_price) / entry_price) * 100
    else:
        return ((entry_price - exit_price) / entry_price) * 100


def calculate_position_value(entry_price: float, amount: float) -> float:
    """
    Calculate total position value.
    
    Args:
        entry_price: Entry price of the position
        amount: Position size/amount
        
    Returns:
        Total position value in quote currency
        
    Examples:
        >>> calculate_position_value(50000, 0.1)
        5000.0
        >>> calculate_position_value(100, 5)
        500.0
    """
    return entry_price * amount
