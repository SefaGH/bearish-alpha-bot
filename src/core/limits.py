def _market(client, symbol):
    try:
        return (client.ex.markets or {}).get(symbol, {}) or {}
    except Exception:
        return {}

def _contract_size(m):
    cs = m.get('contractSize')
    try:
        return float(cs) if cs else 1.0
    except Exception:
        return 1.0

def clamp_amount(client, symbol, amount: float, behavior: str = "skip") -> float:
    """Clamp by min/max lot. If amount < min:
        - behavior == 'skip'  -> return 0.0 (signal skipped)
        - behavior == 'scale' -> raise to min
    """
    m = _market(client, symbol)
    lim = (m.get('limits') or {}).get('amount') or {}
    min_a = float(lim.get('min') or 0.0)
    max_a = float(lim.get('max') or float('inf'))
    if min_a and amount < min_a:
        return 0.0 if behavior == "skip" else min_a
    if amount > max_a:
        return max_a
    return max(amount, 0.0)

def notional_info(client, symbol, price: float, amount: float):
    m = _market(client, symbol)
    cs = _contract_size(m)
    notional = price * amount * cs
    lim = (m.get('limits') or {}).get('cost') or {}
    min_n = float(lim.get('min') or 0.0)
    return notional, min_n, cs

def meets_or_scale_notional(client, symbol, price: float, amount: float, behavior: str = "skip"):
    notional, min_n, cs = notional_info(client, symbol, price, amount)
    if min_n <= 0:
        return amount  # no requirement
    if notional >= min_n:
        return amount
    if behavior == "skip":
        return 0.0
    # scale: increase amount to meet min notional (no step-size here; main will precision-round)
    needed = min_n / (price * cs) if price * cs > 0 else 0.0
    return max(amount, needed)

def clamp_price(client, symbol, price: float) -> float:
    try:
        return float(client.ex.price_to_precision(symbol, price))
    except Exception:
        return price
