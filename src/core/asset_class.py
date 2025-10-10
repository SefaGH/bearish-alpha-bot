MEME_BASES_DEFAULT = {
    'PEPE','WIF','FLOKI','SHIB','DOGE','BONK','BABYDOGE','MOG','MEW','TURBO','WEN','HOSHI','GROK'
}

def base_from_symbol(symbol: str) -> str:
    # 'PEPE/USDT:USDT' -> 'PEPE'
    s = symbol.split('/')[0].upper().strip()
    return s

def classify_symbol(symbol: str, config: dict) -> str:
    """Return one of: 'meme', 'other' (extendable later)"""
    bases_cfg = set([b.upper() for b in (config.get('classes',{}).get('meme_bases', []) or [])])
    if not bases_cfg:
        bases_cfg = MEME_BASES_DEFAULT
    base = base_from_symbol(symbol)
    if base in bases_cfg:
        return 'meme'
    return 'other'
