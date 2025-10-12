import os
from typing import Dict, List, Set

def _is_usdt_candidate(market: dict, only_linear: bool = True) -> bool:
    """
    only_linear=True  -> sadece USDT-quoted linear swap (perps)
    only_linear=False -> USDT-quoted linear swap **veya** USDT-quoted spot
    """
    if not market.get('active', True):
        return False
    if market.get('quote') != 'USDT':
        return False

    is_swap = bool(market.get('swap', False))
    is_spot = not is_swap and bool(market.get('spot', not is_swap))
    is_linear = (market.get('linear') is not False)  # yoksa True varsay

    if only_linear:
        return is_swap and is_linear
    else:
        return (is_swap and is_linear) or is_spot

def _synced_lists(u: dict) -> (Set[str], Set[str]):
    include = set(u.get('include', []) or []).union(set(u.get('allow_list', []) or []))
    deny = set(u.get('exclude', []) or []).union(set(u.get('deny_list', []) or [])).union(set(u.get('blacklist', []) or []))
    return include, deny

def _is_stable_base(symbol: str) -> bool:
    base = (symbol.split('/')[0] if '/' in symbol else symbol).upper()
    return base in {'USDT', 'USDC', 'FDUSD', 'TUSD', 'DAI'}

def build_universe(exchanges: Dict[str, object], cfg: dict) -> Dict[str, List[str]]:
    u = cfg.get('universe', {}) or {}

    # thresholds (sağlam parse)
    try:
        min_qv = float(u.get('min_quote_volume_usdt', u.get('min_quote_vol_usd', u.get('min_quote_volume', 0))) or 0)
    except Exception:
        min_qv = 0.0
    try:
        top_n = int(u.get('top_n_per_exchange', u.get('max_symbols_per_exchange', 20)) or 20)
    except Exception:
        top_n = 20

    only_linear = bool(u.get('only_linear', u.get('prefer_perps', True)))
    exclude_stables = bool(u.get('exclude_stables', True))
    allow_set, deny_set = _synced_lists(u)

    per_ex: Dict[str, List[str]] = {}

    for name, client in exchanges.items():
        # 1) markets
        try:
            mkts = client.markets()
        except Exception as e:
            print(f"[universe] skip {name}: markets() failed: {e}")
            continue

        # 2) adaylar
        candidates = []
        for sym, m in mkts.items():
            try:
                if _is_usdt_candidate(m, only_linear):
                    if sym in deny_set:
                        continue
                    if exclude_stables and _is_stable_base(sym):
                        continue
                    candidates.append(sym)
            except Exception:
                continue

        # 3) tickers & hacim (quoteVolume yoksa baseVolume fallback)
        try:
            tks = client.tickers()
        except Exception as e:
            print(f"[universe] {name}: tickers() failed: {e}")
            tks = {}

        def qv(sym: str) -> float:
            t = tks.get(sym) or {}
            return float(t.get('quoteVolume', 0) or t.get('baseVolume', 0) or 0)

        ranked = sorted(candidates, key=qv, reverse=True)

        # include'ları en üste al
        for a in allow_set:
            if a not in ranked and a in mkts:
                ranked.insert(0, a)

        # min qv filtresi
        if min_qv > 0:
            ranked = [s for s in ranked if qv(s) >= min_qv]

        if not ranked:
            print(f"[universe] {name}: no eligible symbols after filtering; skipping.")
            continue

        per_ex[name] = ranked[:top_n]

    if not per_ex:
        raise SystemExit("Universe is empty. Lower min_quote_volume_usdt, increase max_symbols_per_exchange, "
                         "or set prefer_perps: false to include spot.")
    return per_ex

def pick_execution_exchange() -> str:
    return os.getenv('EXECUTION_EXCHANGE', 'kucoinfutures')
