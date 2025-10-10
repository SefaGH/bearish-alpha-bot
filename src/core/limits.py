def clamp_amount(client, symbol, amount: float) -> float:
    m = (client.ex.markets or {}).get(symbol, {})
    lim = (m.get('limits') or {}).get('amount') or {}
    min_a = lim.get('min') or 0.0
    max_a = lim.get('max') or float('inf')
    if min_a and amount < min_a: 
        amount = min_a
    if amount > max_a:
        amount = max_a
    return max(amount, 0.0)

def meets_notional(client, symbol, price: float, amount: float) -> bool:
    m = (client.ex.markets or {}).get(symbol, {})
    lim = (m.get('limits') or {}).get('cost') or {}
    min_n = lim.get('min') or 0.0
    notional = price * amount
    return notional >= (min_n or 0.0)

def clamp_price(client, symbol, price: float) -> float:
    # simple tickSize clamp via precision
    try:
        p = float(client.ex.price_to_precision(symbol, price))
        return p
    except Exception:
        return price
