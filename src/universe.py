import os
from typing import Dict, List

def _is_linear_usdt(market: dict, only_linear: bool=True) -> bool:
    if not market.get('swap', False):
        return False
    if market.get('quote') != 'USDT':
        return False
    if only_linear and market.get('linear') is False:
        return False
    return market.get('active', True)

def build_universe(exchanges: Dict[str, object], cfg: dict) -> Dict[str, List[str]]:
    per_ex = {}
    min_qv = cfg['universe'].get('min_quote_volume_usdt', 0)
    top_n = cfg['universe'].get('top_n_per_exchange', 20)
    allow = set(cfg['universe'].get('allow_list', []) or [])
    deny  = set(cfg['universe'].get('deny_list',  []) or [])
    only_linear = cfg['universe'].get('only_linear', True)

    for name, client in exchanges.items():
        try:
            mkts = client.markets()
        except Exception as e:
            print(f"[universe] skip {name}: markets() failed: {e}")
            continue

        candidates = []
        for sym, m in mkts.items():
            if _is_linear_usdt(m, only_linear):
                if sym in deny:
                    continue
                candidates.append(sym)

        try:
            tks = client.tickers()
        except Exception as e:
            print(f"[universe] {name}: tickers() failed: {e}")
            tks = {}

        def qv(sym: str):
            t = tks.get(sym) or {}
            return float(t.get('quoteVolume', 0) or t.get('baseVolume', 0) or 0)

        ranked = sorted(candidates, key=qv, reverse=True)

        for a in allow:
            if a not in ranked and a in mkts:
                ranked.insert(0, a)

        if min_qv > 0:
            ranked = [s for s in ranked if qv(s) >= min_qv]

        if not ranked:
            print(f"[universe] {name}: no eligible symbols after filtering; skipping.")
            continue

        per_ex[name] = ranked[:top_n]

    if not per_ex:
        raise SystemExit("Universe is empty. All exchanges failed or no symbols passed filters. "
                         "Consider removing geo-blocked exchanges (e.g., binance) from EXCHANGES or reduce filters.")
    return per_ex

def pick_execution_exchange() -> str:
    return os.getenv('EXECUTION_EXCHANGE','bingx')
