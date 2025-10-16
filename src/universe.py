import os
from typing import Dict, List, Set

# src/universe.py - _is_usdt_candidate() fonksiyonu

def _is_usdt_candidate(market: dict, only_linear: bool = True) -> bool:
    """
    only_linear=True  -> sadece USDT-quoted linear swap (perps)
    only_linear=False -> USDT-quoted linear swap **veya** USDT-quoted spot
    """
    symbol = market.get('symbol', 'UNKNOWN')
    is_active = market.get('active', True)
    quote = market.get('quote')
    
    # BingX için özel durum: swap/spot anahtarları yoksa, type'a bak
    market_type = market.get('type', '')
    
    # Önce explicit swap/spot anahtarlarına bak
    is_swap = market.get('swap', False)
    is_spot = market.get('spot', False)
    
    # Eğer her ikisi de yoksa veya False ise, type'dan çıkar
    if not is_swap and not is_spot:
        if market_type in ['swap', 'future', 'perpetual']:
            is_swap = True
        elif market_type == 'spot':
            is_spot = True
        else:
            # BingX futures API'de olduğumuz için varsayılan swap
            is_swap = True
    
    is_linear = market.get('linear', True)  # Varsayılan True (BingX hepsi linear)
    
    # Debug log
    print(f"[UNIVERSE] {symbol}: active={is_active}, quote={quote}, swap={is_swap}, "
          f"spot={is_spot}, linear={is_linear}, type={market_type}")
    
    if not is_active:
        return False
    if quote != 'USDT':
        return False

    if only_linear:
        # Sadece linear USDT perpetuals
        result = is_swap and is_linear
        if result:
            print(f"[UNIVERSE] ✅ {symbol} accepted as linear USDT perpetual")
        return result
    else:
        # Linear perpetuals veya spot
        result = (is_swap and is_linear) or is_spot
        if result:
            print(f"[UNIVERSE] ✅ {symbol} accepted")
        return result

def _synced_lists(u: dict) -> (Set[str], Set[str]):
    include = set(u.get('include', []) or []).union(set(u.get('allow_list', []) or []))
    deny = set(u.get('exclude', []) or []).union(set(u.get('deny_list', []) or [])).union(set(u.get('blacklist', []) or []))
    return include, deny

def _is_stable_base(symbol: str) -> bool:
    base = (symbol.split('/')[0] if '/' in symbol else symbol).upper()
    return base in {'USDT', 'USDC', 'FDUSD', 'TUSD', 'DAI'}

def build_universe(exchanges: Dict[str, object], cfg: dict) -> Dict[str, List[str]]:
    """
    Build universe with fixed symbols (no market loading!) or dynamic auto-select.
    """
    u = cfg.get('universe', {}) or {}
    
    # Config'den sabit sembol listesi
    fixed_symbols = u.get('fixed_symbols', [])
    auto_select = u.get('auto_select', False)  # Varsayılan FALSE!
    
    # Sabit liste varsa ve auto_select kapalıysa
    if fixed_symbols and not auto_select:
        print(f"[UNIVERSE] ✅ Using FIXED symbol list: {len(fixed_symbols)} symbols")
        print(f"[UNIVERSE] No market loading needed! 🚀")
        
        per_ex = {}
        for name in exchanges.keys():
            # Her borsa için aynı listeyi kullan
            per_ex[name] = fixed_symbols.copy()
            print(f"[UNIVERSE] {name}: Assigned {len(fixed_symbols)} symbols")
        
        # Debug: İlk 5 sembolü göster
        if fixed_symbols:
            print(f"[UNIVERSE] Symbols: {', '.join(fixed_symbols[:5])}...")
        
        return per_ex
    
    # AUTO_SELECT = TRUE (eski yöntem, önerilmez)
    print(f"[UNIVERSE] ⚠️ Auto-select mode active (will load all markets)")

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
    
    print(f"[UNIVERSE] Building universe with: min_qv={min_qv}, top_n={top_n}, only_linear={only_linear}")

    per_ex: Dict[str, List[str]] = {}

    for name, client in exchanges.items():
        print(f"[UNIVERSE] Processing exchange: {name}")
        
        # 1) markets
        try:
            mkts = client.markets()
            print(f"[UNIVERSE] {name}: loaded {len(mkts)} markets")
        except Exception as e:
            print(f"[UNIVERSE] skip {name}: markets() failed: {e}")
            continue

        # 2) adaylar
        candidates = []
        for sym, m in mkts.items():
            try:
                if _is_usdt_candidate(m, only_linear):
                    if sym in deny_set:
                        print(f"[UNIVERSE] {sym} denied (blacklist)")
                        continue
                    if exclude_stables and _is_stable_base(sym):
                        print(f"[UNIVERSE] {sym} excluded (stable base)")
                        continue
                    candidates.append(sym)
            except Exception:
                continue
        
        print(f"[UNIVERSE] {name}: {len(candidates)} candidates after filtering")

        # 3) tickers & hacim (quoteVolume yoksa baseVolume fallback)
        try:
            tks = client.tickers()
        except Exception as e:
            print(f"[UNIVERSE] {name}: tickers() failed: {e}")
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
            before_filter = len(ranked)
            ranked = [s for s in ranked if qv(s) >= min_qv]
            print(f"[UNIVERSE] {name}: volume filter removed {before_filter - len(ranked)} symbols")

        if not ranked:
            print(f"[UNIVERSE] {name}: no eligible symbols after filtering; skipping.")
            continue

        per_ex[name] = ranked[:top_n]
        print(f"[UNIVERSE] {name}: selected {len(per_ex[name])} symbols: {per_ex[name][:5]}...")

    if not per_ex:
        raise SystemExit("Universe is empty. Lower min_quote_volume_usdt, increase max_symbols_per_exchange, "
                         "or set prefer_perps: false to include spot.")
    
    print(f"[UNIVERSE] Built universe: {per_ex}")
    return per_ex

def pick_execution_exchange() -> str:
    return os.getenv('EXECUTION_EXCHANGE', 'kucoinfutures')
