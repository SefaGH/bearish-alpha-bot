# core/asset_class.py — gelişmiş sınıflandırma (meme / microcap / bluechip / other)

MEME_BASES_DEFAULT = {
    'PEPE','WIF','FLOKI','SHIB','DOGE','BONK','BABYDOGE','MOG','MEW','TURBO','WEN','HOSHI','GROK','TRUMP'
}

MICROCAP_BASES_DEFAULT = {
    'SUI','ARB','JASMY','CELO','WLD','TIA','OP','NEAR','APT','MANTA','SEI','INJ'
}

BLUECHIP_BASES_DEFAULT = {
    'BTC','ETH','BNB','SOL','XRP','ADA','DOT','LTC','AVAX','LINK','XLM','ATOM'
}


def base_from_symbol(symbol: str) -> str:
    """
    'PEPE/USDT:USDT' -> 'PEPE'
    'BTC/USDT'       -> 'BTC'
    """
    try:
        return symbol.split('/')[0].upper().strip()
    except Exception:
        return str(symbol).upper().strip()


def classify_symbol(symbol: str, config: dict) -> str:
    """
    Dönen sınıflar: 'meme', 'microcap', 'bluechip', 'other'
    Config override desteği:
      classes:
        meme_bases: ['PEPE', 'DOGE', 'TRUMP']
        microcap_bases: ['SUI', 'ARB', 'WLD']
        bluechip_bases: ['BTC', 'ETH', 'SOL']
    """
    classes_cfg = (config or {}).get('classes', {}) or {}

    meme_bases = set([b.upper() for b in classes_cfg.get('meme_bases', [])]) or MEME_BASES_DEFAULT
    micro_bases = set([b.upper() for b in classes_cfg.get('microcap_bases', [])]) or MICROCAP_BASES_DEFAULT
    blue_bases = set([b.upper() for b in classes_cfg.get('bluechip_bases', [])]) or BLUECHIP_BASES_DEFAULT

    base = base_from_symbol(symbol)

    if base in meme_bases:
        return 'meme'
    if base in micro_bases:
        return 'microcap'
    if base in blue_bases:
        return 'bluechip'
    return 'other'
