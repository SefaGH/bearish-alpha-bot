import os
from typing import Dict, List, Set

# ----- helpers -----

def _is_linear_usdt(market: dict, only_linear: bool = True) -> bool:
    """
    Piyasanın USDT-quoted, aktif ve (only_linear=True ise) linear swap olup olmadığını kontrol eder.
    Spot/Inverse/Inactive olanları eler.
    """
    if not market.get('active', True):
        return False

    # Sadece USDT quoted piyasalar
    if market.get('quote') != 'USDT':
        return False

    # Swap olmayanları ele (spot ise swap=False olur)
    if not market.get('swap', False):
        return False

    # Sadece linear istiyorsak inverse/perpetual inverse'leri ele
    if only_linear and market.get('linear') is False:
        return False

    return True


def _synced_lists(u: dict) -> (Set[str], Set[str]):
    """
    include/allow_list ve exclude/deny_list/blacklist alanlarını birleştirir.
    Farklı isimlendirmeleri destekler.
    """
    include = set(u.get('include', []) or [])
    allow = set(u.get('allow_list', []) or [])
    merged_allow = include.union(allow)

    exclude = set(u.get('exclude', []) or [])
    deny = set(u.get('deny_list', []) or [])
    blacklist = set(u.get('blacklist', []) or [])
    merged_deny = exclude.union(deny).union(blacklist)
    return merged_allow, merged_deny


def _is_stable_base(symbol: str) -> bool:
    """
    Base tarafı stable olan çiftleri (USDT/..., USDC/..., FDUSD/...) elemek için basit kontrol.
    """
    base = (symbol.split('/')[0] if '/' in symbol else symbol).upper()
    return base in {'USDT', 'USDC', 'FDUSD', 'TUSD', 'DAI'}


# ----- main -----

def build_universe(exchanges: Dict[str, object], cfg: dict) -> Dict[str, List[str]]:
    """
    Borsalardan linear USDT perps evrenini kurar.
    Config'te universe bloğu olmasa bile güvenli default'larla çalışır.

    Desteklenen universe alanları ve eşanlamlıları:
      - min_quote_volume_usdt  | min_quote_vol_usd | min_quote_volume   (default: 0)
      - top_n_per_exchange     | max_symbols_per_exchange                (default: 20)
      - only_linear            | prefer_perps (bool; default: True → linear swap tercih)
      - include / allow_list
      - exclude / deny_list / blacklist
      - exclude_stables (bool; default: True)
    """
    u = cfg.get('universe', {}) or {}

    # Eşikler / parametreler (çok isimli desteği)
    min_qv = u.get('min_quote_volume_usdt',
              u.get('min_quote_vol_usd',
              u.get('min_quote_volume', 0)))
    try:
        min_qv = float(min_qv or 0)
    except Exception:
        min_qv = 0.0

    top_n = u.get('top_n_per_exchange',
             u.get('max_symbols_per_exchange', 20))
    try:
        top_n = int(top_n or 20)
    except Exception:
        top_n = 20

    # prefer_perps=True → only_linear=True davranışıyla eşitlenir
    only_linear = u.get('only_linear', u.get('prefer_perps', True))
    only_linear = bool(only_linear)

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

        # 2) adaylar: USDT linear perps ve deny listesinde olmayanlar
        candidates = []
        for sym, m in mkts.items():
            try:
                if _is_linear_usdt(m, only_linear):
                    if sym in deny_set:
                        continue
                    if exclude_stables and _is_stable_base(sym):
                        continue
                    candidates.append(sym)
            except Exception:
                # tekil markette alanlar eksikse sessiz geç
                continue

        # 3) tickers ile hacim sıralaması
        try:
            tks = client.tickers()
        except Exception as e:
            print(f"[universe] {name}: tickers() failed: {e}")
            tks = {}

        def qv(sym: str) -> float:
            t = tks.get(sym) or {}
            # ccxt çoğu borsada quoteVolume döndürür; yoksa baseVolume'a düş
            return float(t.get('quoteVolume', 0) or t.get('baseVolume', 0) or 0)

        ranked = sorted(candidates, key=qv, reverse=True)

        # 4) allow/include önceliklendir (listede yoksa başa ekle)
        for a in allow_set:
            if a not in ranked and a in mkts:
                ranked.insert(0, a)

        # 5) min quote volume filtresi
        if min_qv > 0:
            ranked = [s for s in ranked if qv(s) >= min_qv]

        if not ranked:
            print(f"[universe] {name}: no eligible symbols after filtering; skipping.")
            continue

        # 6) top-N kırp
        per_ex[name] = ranked[:top_n]

    if not per_ex:
        raise SystemExit(
            "Universe is empty. All exchanges failed or no symbols passed filters. "
            "Consider reducing filters (e.g., lower min_quote_volume), increasing top_n_per_exchange, "
            "or removing geo-blocked exchanges from EXCHANGES."
        )

    return per_ex


def pick_execution_exchange() -> str:
    # Bildirim/exec varsayılan borsa (ENV ile override edilebilir)
    return os.getenv('EXECUTION_EXCHANGE', 'bingx')
