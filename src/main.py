# main.py (routing to EXECUTION_EXCHANGE + per-run base dedup + oversold ignore_regime)
import os, yaml, pandas as pd, time, csv, datetime, math, traceback, json
from datetime import datetime as dt, timedelta, timezone
from dotenv import load_dotenv
from core.multi_exchange import build_clients_from_env
from core.indicators import add_indicators
from core.regime import is_bearish_regime
from core.risk import RiskGuard, RiskConfig
from core.sizing import position_size_usdt
from core.exec_engine import ExecEngine
from core.notify import Telegram
from core.normalize import price_to_precision, amount_to_precision
from core.trailing import initial_stops, trail_level
from core.limits import clamp_amount, meets_or_scale_notional, clamp_price
from core.state import load_state, save_state, load_day_stats, save_day_stats
from core.asset_class import classify_symbol
# ⬇️ base bazlı dedup için
from core.asset_class import base_from_symbol
from strategies.short_the_rip import ShortTheRip
from strategies.oversold_bounce import OversoldBounce
from universe import build_universe, pick_execution_exchange

load_dotenv()

def _utcnow():
    return dt.now(timezone.utc)

# -------- Config load --------
cfg_path = os.getenv('CONFIG_PATH', 'config/config.yaml')
if not os.path.isfile(cfg_path):
    alt = 'config/config.example.yaml'
    if os.path.isfile(alt):
        print(f"[config] {cfg_path} not found. Falling back to {alt}")
        cfg_path = alt
    else:
        raise FileNotFoundError(f"Neither {cfg_path} nor {alt} found. Provide a config file.")
with open(cfg_path, 'r') as f:
    CFG = yaml.safe_load(f) or {}
print(f"[config] Loaded: {cfg_path}")

# ---- safe defaults for execution ----
_exec = CFG.get('execution') or {}
FEE_PCT = float(_exec.get('fee_pct', 0.0006))
MAX_SLIPPAGE_PCT = float(_exec.get('max_slippage_pct', 0.001))
print(f"[exec] fee_pct={FEE_PCT} max_slippage_pct={MAX_SLIPPAGE_PCT}")

MODE = os.getenv('MODE','paper')
TG = Telegram(os.getenv('TELEGRAM_BOT_TOKEN',''), os.getenv('TELEGRAM_CHAT_ID','')) if os.getenv('TELEGRAM_BOT_TOKEN') else None

clients = build_clients_from_env()
if not clients:
    raise SystemExit('EXCHANGES boş. .env/.secrets ayarla.')

sym_source = os.getenv('SYM_SOURCE','AUTO').upper()
if sym_source == 'AUTO':
    UNIVERSE = build_universe(clients, CFG)
else:
    manual = [s.strip() for s in os.getenv('SYMBOLS','BTC/USDT').split(',') if s.strip()]
    UNIVERSE = { pick_execution_exchange(): manual }

# -------- Quarantine list (temporary exclude) --------
QUAR_CFG = CFG.get('quarantine', {}) or {}
QUAR_ENABLE = bool(QUAR_CFG.get('enable', True))
QUAR_DAYS = int(QUAR_CFG.get('days', 7))
QUAR_FILE = QUAR_CFG.get('file', 'data/quarantine.json')

def load_quarantine(path=QUAR_FILE):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            q = json.load(f)
            out = {}
            for k, v in q.items():
                try:
                    exp = dt.fromisoformat(v['expiry'])
                except Exception:
                    exp = _utcnow()
                out[str(k)] = {'added': v.get('added'), 'reason': v.get('reason',''), 'expiry': exp.isoformat()}
            return out
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[quarantine] failed to load: {e}")
        return {}

def save_quarantine(q, path=QUAR_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump = {}
    for k, v in q.items():
        exp = v.get('expiry')
        if not isinstance(exp, str):
            try:
                exp = dt.fromisoformat(str(exp)).isoformat()
            except Exception:
                exp = (_utcnow() + timedelta(days=QUAR_DAYS)).isoformat()
        dump[k] = {'added': v.get('added', _utcnow().isoformat()),
                   'reason': v.get('reason',''),
                   'expiry': exp}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)

def is_quarantined(qmap, ex_name, sym):
    key = f"{ex_name}:{sym}"
    rec = qmap.get(key)
    if not rec: return False
    try:
        expiry = dt.fromisoformat(rec['expiry'])
    except Exception:
        return False
    if _utcnow() >= expiry:
        return False
    return True

def add_quarantine(qmap, ex_name, sym, reason):
    key = f"{ex_name}:{sym}"
    expiry = (_utcnow() + timedelta(days=QUAR_DAYS)).isoformat()
    qmap[key] = {'added': _utcnow().isoformat(), 'reason': reason, 'expiry': expiry}

def cleanup_quarantine(qmap):
    removed = []
    for k, v in list(qmap.items()):
        try:
            expiry = dt.fromisoformat(v['expiry'])
        except Exception:
            del qmap[k]; removed.append(k); continue
        if _utcnow() >= expiry:
            del qmap[k]; removed.append(k)
    return removed

QUAR = load_quarantine() if QUAR_ENABLE else {}
expired = cleanup_quarantine(QUAR)
if expired:
    print(f"[quarantine] cleaned {len(expired)} expired entries")

# Filter UNIVERSE by quarantine
if QUAR_ENABLE:
    filtered = {}
    for ex, syms in UNIVERSE.items():
        kept = [s for s in syms if not is_quarantined(QUAR, ex, s)]
        dropped = len(syms) - len(kept)
        if dropped > 0:
            print(f"[quarantine] filtered {dropped} symbols on {ex}")
        filtered[ex] = kept
    UNIVERSE = filtered

# --- Universe breakdown (after quarantine) ---
total_syms = sum(len(v) for v in UNIVERSE.values())
print("[universe] breakdown (post-quarantine):")
for ex, syms in UNIVERSE.items():
    sample = ", ".join(syms[:8]) + ("..." if len(syms) > 8 else "")
    print(f"  - {ex}: {len(syms)} symbols (e.g., {sample})")
print(f"[universe] total symbols: {total_syms}")
if TG:
    TG.send("[universe] " + " | ".join([f"{ex}:{len(syms)}" for ex, syms in UNIVERSE.items()]))

# -------- Notify routing --------
send_all = bool(CFG.get('notify', {}).get('send_all', True))
exec_ex_env = os.getenv('EXECUTION_EXCHANGE', pick_execution_exchange())
exec_ex = (exec_ex_env or '').strip().lower() or next(iter(clients.keys()))
SEND_EX = exec_ex if exec_ex in UNIVERSE else (next(iter(UNIVERSE.keys())) if UNIVERSE else exec_ex)
print(f"[notify] send_all={send_all} SEND_EX={SEND_EX}")
print(f"[info] universe exchanges: {list(UNIVERSE.keys())}")

def should_notify(ex_name: str) -> bool:
    return send_all or (ex_name == SEND_EX)

exec_client = clients.get(SEND_EX) or next(iter(clients.values()))
exec_eng = ExecEngine(MODE, exec_client, FEE_PCT, MAX_SLIPPAGE_PCT, TG)

# -------- Risk / params --------
min_amount_behavior = str(CFG['risk'].get('min_amount_behavior', 'skip')).lower()
min_notional_behavior = str(CFG['risk'].get('min_notional_behavior', 'skip')).lower()
MIN_STOP_PCT = float(CFG['risk'].get('min_stop_pct', 0.003))

GLOBAL_MAX_NOTIONAL = float(CFG['risk'].get('max_notional_per_trade', 100000.0))
GLOBAL_RISK_USD_CAP = float(CFG['risk'].get('risk_usd_cap', 1200.0))

CLASS_LIMITS = CFG.get('class_limits', {})
def class_caps(symbol: str):
    cls = classify_symbol(symbol, CFG)
    lim = CLASS_LIMITS.get(cls, {})
    return cls, float(lim.get('max_notional_per_trade', GLOBAL_MAX_NOTIONAL)), float(lim.get('risk_usd_cap', GLOBAL_RISK_USD_CAP))

risk_cfg = RiskConfig(
    per_trade_risk_pct=CFG['risk']['per_trade_risk_pct'],
    daily_loss_limit_pct=CFG['risk']['daily_loss_limit_pct'],
    cool_down_min=CFG['risk']['cool_down_min']
)
risk = RiskGuard(equity_usd=CFG['risk'].get('equity_usd', 10_000), cfg=risk_cfg)

str_short = ShortTheRip(CFG['signals']['short_the_rip']) if CFG['signals']['short_the_rip']['enable'] else None
str_bounce = OversoldBounce(CFG['signals']['oversold_bounce']) if CFG['signals']['oversold_bounce']['enable'] else None

TF_FAST = CFG['timeframes']['fast']
TF_SLOW = CFG['timeframes']['slow']
TF_MID  = CFG['timeframes']['mid']

MIN_COOLDOWN_SEC = int(CFG.get('notify',{}).get('min_cooldown_sec', 300))
PUSH_NO_SIGNAL = bool(CFG.get('notify',{}).get('push_no_signal', True))
PUSH_DEBUG = bool(CFG.get('notify',{}).get('push_debug', False))
DAILY_MAX_TRADES = int(CFG['risk'].get('daily_max_trades', 20))
MIN_SLOW_CANDLES = int(CFG.get('regime',{}).get('min_slow_candles', 120))

LAST_SENT = {}
def can_notify(key: str) -> bool:
    now = time.time()
    last = LAST_SENT.get(key, 0)
    if now - last >= MIN_COOLDOWN_SEC:
        LAST_SENT[key] = now
        return True
    return False

def fetch_df(client, symbol, tf):
    o = client.ohlcv(symbol, tf, limit=400)
    df = pd.DataFrame(o, columns=['ts','open','high','low','close','vol'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df

def log_signal(row: dict):
    os.makedirs('data', exist_ok=True)
    path = 'data/signals.csv'
    file_exists = os.path.isfile(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def _safe_contract_size(mkt: dict) -> float:
    cs = mkt.get('contractSize', 1.0)
    if isinstance(cs, (int, float)):
        return float(cs)
    if isinstance(cs, str):
        try: return float(cs)
        except Exception: return 1.0
    if isinstance(cs, dict):
        for k in ('value','size','contractSize'):
            v = cs.get(k)
            if isinstance(v, (int, float)): return float(v)
            if isinstance(v, str):
                try: return float(v)
                except Exception: pass
    return 1.0

def fmt_usd(x: float) -> str:
    try: x = float(x)
    except Exception: return str(x)
    if x >= 1e9: return f"${x/1e9:.2f}B"
    if x >= 1e6: return f"${x/1e6:.2f}M"
    if x >= 1e3: return f"${x/1e3:.1f}k"
    return f"${x:.2f}"

state = load_state()
day = load_day_stats()

def register_open(ex_name, sym, side, entry, tp, sl, trail, qty, extra=None):
    key = f"{ex_name}:{sym}:{side}"
    state['open'][key] = {'ts': datetime.datetime.utcnow().isoformat(),
                          'exchange': ex_name, 'symbol': sym, 'side': side,
                          'entry': entry, 'tp': tp, 'sl': sl, 'trail': trail, 'qty': qty,
                          'status': 'open', **(extra or {})}
    day['signals'] += 1

def evaluate_tp_sl(current_price, side, tp, sl):
    if side == 'SELL':
        if current_price <= tp: return 'TP'
        if current_price >= sl: return 'SL'
    else:
        if current_price >= tp: return 'TP'
        if current_price <= sl: return 'SL'
    return None

def paper_pnl(side, entry, exitp, qty, fee_pct):
    gross = (entry - exitp) * qty if side == 'SELL' else (exitp - entry) * qty
    fees = (entry + exitp) * qty * fee_pct
    return gross - fees

def apply_caps(entry, sl, qty, cs, cls, max_notional_cls, risk_cap_cls):
    notional = entry * qty * cs
    risk_usd = abs(sl - entry) * qty * cs
    cap_note = ""
    max_notional = min(max_notional_cls, GLOBAL_MAX_NOTIONAL)
    if notional > max_notional and max(entry*cs, 1e-12) > 0:
        qty = max_notional / (entry * cs)
        cap_note = "capped:notional"
        notional = entry * qty * cs
        risk_usd = abs(sl - entry) * qty * cs
    risk_cap = min(risk_cap_cls, GLOBAL_RISK_USD_CAP)
    if risk_usd > risk_cap and abs(sl - entry) * cs > 0:
        qty = risk_cap / (abs(sl - entry) * cs)
        cap_note = ("; " if cap_note else "") + "capped:risk"
        notional = entry * qty * cs
        risk_usd = abs(sl - entry) * cs * qty
    return qty, notional, risk_usd, cap_note

# ---------- LIVE EXECUTION HOOK (always routes to EXECUTION_EXCHANGE) ----------
def maybe_execute_live(exec_eng, symbol, side, qty, entry, tp_f, sl_f, trailp):
    """
    Canlı mod + enable_live True ise emri her zaman EXECUTION_EXCHANGE üzerinde aç.
    ExecEngine.open_market kullanılır; yoksa sessizce None döner.
    """
    try:
        live_mode = (os.getenv('MODE','paper').lower() == 'live')
        if not (live_mode and CFG.get('execution',{}).get('enable_live', False)):
            return None
        if hasattr(exec_eng, 'open_market'):
            return exec_eng.open_market(side, symbol, qty, sl_f, tp_f, trailp)
        return None
    except Exception as e:
        print(f"[execute] routed {symbol} -> {SEND_EX} failed -> {e}")
        if TG:
            TG.send(f"⚠️ Live order failed routed {symbol} — {e}")
        return None
# -----------------------------------------------------------------------------

scanned = 0
bear_ok = 0
signals_found = 0
sent = 0
# ⬇️ per-run base dedup
DEDUP_BASES = set()

try:
    for ex_name, syms in UNIVERSE.items():
        c = clients[ex_name]
        for sym in syms:
            try:
                scanned += 1

                # Fetch raw 4h first
                raw4h = fetch_df(c, sym, TF_SLOW)
                # (opsiyonel) BingX ping
                if ex_name.lower() == "bingx":
                    print(f"[ping] bingx:{sym} fetched {len(raw4h)} bars @ {TF_SLOW}")
                    if TG and bool(CFG.get('notify',{}).get('push_debug', False)):
                        TG.send(f"📡 BingX ping — {sym}: fetched {len(raw4h)} bars @ {TF_SLOW}")

                if len(raw4h) < MIN_SLOW_CANDLES:
                    if QUAR_ENABLE:
                        add_quarantine(QUAR, ex_name, sym, reason=f"short 4h history ({len(raw4h)} < {MIN_SLOW_CANDLES})")
                    raise ValueError(f"short 4h history ({len(raw4h)} < {MIN_SLOW_CANDLES})")
                df4h = add_indicators(raw4h, CFG['indicators'])

                # ✅ Rejim durumunu değişkende tut (SHORT buna bağlı, LONG opsiyonel)
                bearish = is_bearish_regime(df4h)
                if bearish:
                    bear_ok += 1

                df30 = add_indicators(fetch_df(c, sym, TF_FAST), CFG['indicators'])
                df30c = df30.dropna()
                if df30c.empty or 'atr' not in df30c.columns:
                    if QUAR_ENABLE:
                        add_quarantine(QUAR, ex_name, sym, reason="no fast frame data")
                    raise ValueError("no fast frame data")
                last = df30c.iloc[-1]
                price = float(last['close'])

                mkt = (c.ex.markets or {}).get(sym, {}) or {}
                is_contract = bool(mkt.get('contract', mkt.get('swap', False)))
                cs_eff = _safe_contract_size(mkt) or 1.0

                # SHORT — sadece bearish rejimde
                if bearish and str_short and risk.can_trade() and day['signals'] < DAILY_MAX_TRADES:
                    sig = str_short.signal(df30, df30)
                    if sig:
                        base = base_from_symbol(sym)   # dedup key
                        if base in DEDUP_BASES:
                            if TG and bool(CFG.get('notify',{}).get('push_debug', False)):
                                TG.send(f"🟡 Dedup (base): {base} on {ex_name}:{sym}")
                        else:
                            atr = float(last['atr'])
                            tp, sl = initial_stops('sell', price, atr, sig['sl_atr_mult'], sig['tp_pct'])
                            qty_tokens = position_size_usdt(price, sl, risk.per_trade_risk_usd(), 'short')
                            qty = (qty_tokens / cs_eff) if is_contract else qty_tokens
                            qty = clamp_amount(c, sym, qty, behavior=min_amount_behavior)
                            if qty > 0:
                                qty = meets_or_scale_notional(c, sym, price, qty, behavior=min_notional_behavior)
                            if qty > 0:
                                entry = float(price_to_precision(c, sym, price))
                                tp_f  = float(price_to_precision(c, sym, tp))
                                sl_f  = float(price_to_precision(c, sym, sl))
                                qty   = float(amount_to_precision(c, sym, qty))
                                cls, max_notional_cls, risk_cap_cls = class_caps(sym)
                                qty, notional, risk_usd, cap_note = apply_caps(entry, sl_f, qty, (cs_eff if is_contract else 1.0), cls, max_notional_cls, risk_cap_cls)
                                signals_found += 1

                                # LIVE EXEC — her zaman EXECUTION_EXCHANGE'te
                                trailp_val = float(price_to_precision(c, sym, trail_level('sell', price, atr, CFG['signals']['short_the_rip'].get('trail_atr_mult', 1.0))))
                                live_res = maybe_execute_live(exec_eng, sym, 'SELL', qty, entry, tp_f, sl_f, trailp_val)
                                order_note = f"\nOrderID={live_res['order_id']}" if (live_res and live_res.get('order_id')) else ""
                                DEDUP_BASES.add(base)  # tek emir

                                if TG and should_notify(ex_name) and qty > 0 and can_notify(f"{ex_name}:{sym}:SELL"):
                                    trailp = price_to_precision(c, sym, trail_level('sell', price, atr, CFG['signals']['short_the_rip'].get('trail_atr_mult', 1.0)))
                                    TG.send(
                                        f"🔴 [{ex_name}→{SEND_EX}] SHORT {sym} @ {entry}\n"
                                        f"TP~{tp_f} SL~{sl_f} Trail~{trailp} Qty~{qty}\n"
                                        f"Notional≈{fmt_usd(notional)}  Risk≈{fmt_usd(risk_usd)}{order_note}"
                                    )
                                    sent += 1

                                register_open(ex_name, sym, 'SELL', entry, tp_f, sl_f, trailp_val, qty)

                # LONG — oversold_bounce için rejim opsiyonel
                allow_long = CFG['signals']['oversold_bounce'].get('ignore_regime', False) or bearish
                if allow_long and str_bounce and risk.can_trade() and day['signals'] < DAILY_MAX_TRADES:
                    sig = str_bounce.signal(df30)
                    if sig:
                        base = base_from_symbol(sym)   # dedup key
                        if base in DEDUP_BASES:
                            if TG and bool(CFG.get('notify',{}).get('push_debug', False)):
                                TG.send(f"🟡 Dedup (base): {base} on {ex_name}:{sym}")
                        else:
                            atr = float(last['atr'])
                            sl_pct_cfg = CFG['signals']['oversold_bounce'].get('sl_pct', None)
                            sl = price * (1 - float(sl_pct_cfg)) if sl_pct_cfg is not None else price - CFG['signals']['oversold_bounce'].get('sl_atr_mult', 1.0) * atr
                            tp = price * (1 + sig['tp_pct'])
                            if (price - sl) / max(price, 1e-12) < MIN_STOP_PCT:
                                sig = None
                    if sig:
                        qty_tokens = position_size_usdt(price, sl, risk.per_trade_risk_usd(), 'long')
                        qty = (qty_tokens / cs_eff) if is_contract else qty_tokens
                        qty = clamp_amount(c, sym, qty, behavior=min_amount_behavior)
                        if qty > 0:
                            qty = meets_or_scale_notional(c, sym, price, qty, behavior=min_notional_behavior)
                        if qty > 0:
                            entry = float(price_to_precision(c, sym, price))
                            tp_f  = float(price_to_precision(c, sym, tp))
                            sl_f  = float(price_to_precision(c, sym, sl))
                            qty   = float(amount_to_precision(c, sym, qty))
                            cls, max_notional_cls, risk_cap_cls = class_caps(sym)
                            qty, notional, risk_usd, cap_note = apply_caps(entry, sl_f, qty, (cs_eff if is_contract else 1.0), cls, max_notional_cls, risk_cap_cls)
                            signals_found += 1

                            # LIVE EXEC — her zaman EXECUTION_EXCHANGE'te
                            trailp_val = float(price_to_precision(c, sym, trail_level('buy', price, atr, CFG['signals']['oversold_bounce'].get('trail_atr_mult', 1.0))))
                            live_res = maybe_execute_live(exec_eng, sym, 'BUY', qty, entry, tp_f, sl_f, trailp_val)
                            order_note = f"\nOrderID={live_res['order_id']}" if (live_res and live_res.get('order_id')) else ""
                            DEDUP_BASES.add(base_from_symbol(sym))  # tek emir

                            if TG and should_notify(ex_name) and qty > 0 and can_notify(f"{ex_name}:{sym}:BUY"):
                                trailp = price_to_precision(c, sym, trail_level('buy', price, atr, CFG['signals']['oversold_bounce'].get('trail_atr_mult', 1.0)))
                                TG.send(
                                    f"🟢 [{ex_name}→{SEND_EX}] LONG {sym} @ {entry}\n"
                                    f"TP~{tp_f} SL~{sl_f} Trail~{trailp} Qty~{qty}\n"
                                    f"Notional≈{fmt_usd(notional)}  Risk≈{fmt_usd(risk_usd)}{order_note}"
                                )
                                sent += 1

                            register_open(ex_name, sym, 'BUY', entry, tp_f, sl_f, trailp_val, qty)

            except Exception as se:
                warn = f"[warn] {ex_name}:{sym} error -> {se}"
                print(warn)
                if TG and bool(CFG.get('notify',{}).get('push_debug', False)):
                    TG.send(f"🟡 Skip {ex_name}:{sym} — {se}")
                continue

except Exception as e:
    tb = traceback.format_exc()
    if TG:
        TG.send(f"⚠️ Run error: {e}\n{tb[-900:]}")
    print(f"[error] {e}\n{tb}")
finally:
    save_state(state); save_day_stats(day)
    if QUAR_ENABLE:
        save_quarantine(QUAR)
        print(f"[quarantine] saved {len(QUAR)} entries to {QUAR_FILE}")
    print(f"[summary] scanned={scanned} bearish_ok={bear_ok} signals_found={signals_found} sent={sent} open_now={len(state['open'])} closed_today_tp={day['tp']} sl={day['sl']} pnl≈{day['pnl']:.2f}")
    if TG and bool(CFG.get('notify',{}).get('push_no_signal', True)) and sent == 0:
        TG.send(f"ℹ️ No signals this run. scanned={scanned} bearish_ok={bear_ok} signals_found={signals_found} sent={sent} open={len(state['open'])}")
    if TG and bool(CFG.get('notify',{}).get('push_debug', False)):
        TG.send(f"🧪 Debug: scanned={scanned} bearish_ok={bear_ok} signals_found={signals_found} sent={sent} open={len(state['open'])} pnl≈$0.00")
