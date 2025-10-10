def price_to_precision(client, symbol, price: float) -> str:
    try:
        return client.ex.price_to_precision(symbol, price)
    except Exception:
        return f"{price:.6f}"

def amount_to_precision(client, symbol, amount: float) -> str:
    try:
        return client.ex.amount_to_precision(symbol, amount)
    except Exception:
        return f"{amount:.6f}"
