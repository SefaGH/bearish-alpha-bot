import os, yaml, pandas as pd, time, csv, datetime
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
from strategies.short_the_rip import ShortTheRip
from strategies.oversold_bounce import OversoldBounce
from universe import build_universe, pick_execution_exchange

load_dotenv()

with open('config/config.yaml','r') as f:
    CFG = yaml.safe_load(f)

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

# ---- Notification routing (send_all or single exchange) ----
send_all = bool(CFG.get('notify', {}).get('send_all', True))  # DEFAULT: True
exec_ex_env = os.getenv('EXECUTION_EXCHANGE', pick_execution_exchange())
exec_ex = (exec_ex_env or '').strip().lower() or next(iter(clients.keys()))
SEND_EX = exec_ex
if SEND_EX not in UNIVERSE and UNIVERSE:
    # fallback to first universe exchange to avoid "sent=0" if names differ
    SEND_EX = next(iter(UNIVERSE.keys()))
print(f"[notify] send_all={send_all} SEND_EX={SEND_EX}")
print(f"[info] universe exchanges: {list(UNIVERSE.keys())}")

def should_notify(ex_name: str) -> bool:
    return send_all or (ex_name == SEND_EX)

exec_client = clients.get(SEND_EX) or next(iter(clients.values()))
exec_eng = ExecEngine(MODE, exec_client, CFG['execution']['fee_pct'], CFG['execution']['max_slippage_pct'], TG)

# --- sizing & limits behavior flags ---
min_amount_behavior = str(CFG['risk'].get('min_amount_behavior', 'skip')).lower()   # 'skip' or 'scale'
min_notional_behavior = str(CFG['risk'].get('min_notional_behavior', 'skip')).lower()
print(f"[limits] min_amount_behavior={min_amount_behavior} min_notional_behavior={min_notional_behavior}")

# --- minimal stop distance for longs; same-run gate ---
MIN_STOP_PCT = float(CFG['risk'].get('min_stop_pct', 0.003))  # 0.3% default
opened_this_run = set()

risk_cfg = RiskConfig(
    per_trade_risk_pct=CFG['risk']['per_trade_risk_pct'],
    daily_loss_limit_pct=CFG['risk']['daily_loss_limit_pct'],
    cool_down_min=CFG['risk']['cool_down_min']
)
risk = RiskGuard(equity_usd=CFG['risk'].get('equity_usd', 10_000), cfg=risk_cfg)

str_short = ShortTheRip(CFG['signals']['short_the_rip']) if CFG['signals']['short_the_rip']['enable'] else None
str_bounce = OversoldBounce(CFG['signals']['oversold_bounce']) if CFG['signals']['oversold_bounce']['enable'] else None

TF_FAST = CFG['timeframes']['fast']
TF_MID  = CFG['timeframes']['mid']
TF_SLOW = CFG['timeframes']['slow']

MIN_COOLDOWN_SEC = int(CFG.get('notify',{}).get('min_cooldown_sec', 300))
PUSH_NO_SIGNAL = bool(CFG.get('notify',{}).get('push_no_signal', True))
PUSH_DEBUG = bool(CFG.get('notify',{}).get('push_debug', False))
DAILY_MAX_TRADES = int(CFG['risk'].get('daily_max_trades', 20))
MAX_NOTIONAL_PER_TRADE = float(CFG['risk'].get('max_notional_per_trade', 1e9))  # large default

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

# ---- State & day stats ----
state = load_state()
day = load_day_stats()

def register_open(ex_name, sym, side, entry, tp, sl, trail, qty, extra=None):
    key = f"{ex_name}:{sym}:{side}"
    state['open'][key] = {
        'ts': datetime.datetime.utcnow().isoformat(),
        'exchange': ex_name, 'symbol': sym, 'side': side,
        'entry': entry, 'tp': tp, 'sl': sl, 'trail': trail, 'qty': qty,
        'status': 'open', **(extra or {})
    }
    day['signals'] += 1

def evaluate_tp_sl(current_price, side, tp, sl):
    if side == 'SELL':
        if current_price <= tp: return 'TP'
        if current_price >= sl: return 'SL'
    else:  # BUY
        if current_price >= tp: return 'TP'
        if current_price <= sl: return 'SL'
    return None

def paper_pnl(side, entry, exitp, qty, fee_pct):
    gross = (entry - exitp) * qty if side == 'SELL' else (exitp - entry) * qty
    fees = (entry + exitp) * qty * fee_pct
    return gross - fees

def market_info(client, symbol):
    m = (client.ex.markets or {}).get(symbol, {}) or {}
    is_contract = bool(m.get('contract', False))
    cs = float(m.get('contractSize') or 1.0)
    return m, is_contract, cs

# ---- One-shot run with tracking ----
scanned = 0
bear_ok = 0
signals_found = 0
sent = 0
now_iso = datetime.datetime.utcnow().isoformat()

try:
    # 1) scan & generate signals
    for ex_name, syms in UNIVERSE.items():
        c = clients[ex_name]
        for sym in syms:
            scanned += 1
            df30 = add_indicators(fetch_df(c, sym, TF_FAST), CFG['indicators'])
            df1h = add_indicators(fetch_df(c, sym, TF_MID),  CFG['indicators'])
            df4h = add_indicators(fetch_df(c, sym, TF_SLOW), CFG['indicators'])

            if not is_bearish_regime(df4h):
                continue
            bear_ok += 1

            last = df30.dropna().iloc[-1]
            price = float(last['close'])
            mkt, is_contract, cs = market_info(c, sym)

            # SHORT
            if str_short and risk.can_trade() and day['signals'] < DAILY_MAX_TRADES:
                sig = str_short.signal(df30, df1h)
                if sig:
                    atr = float(last['atr'])
                    tp, sl = initial_stops('sell', price, atr, sig['sl_atr_mult'], sig['tp_pct'])
                    qty_tokens = position_size_usdt(price, sl, risk.per_trade_risk_usd(), 'short')
                    qty = qty_tokens / cs if is_contract else qty_tokens
                    qty = clamp_amount(c, sym, qty, behavior=min_amount_behavior)
                    if qty > 0:
                        if price * qty * (cs if is_contract else 1.0) > MAX_NOTIONAL_PER_TRADE:
                            qty = MAX_NOTIONAL_PER_TRADE / (price * (cs if is_contract else 1.0))
                            qty = clamp_amount(c, sym, qty, behavior=min_amount_behavior)
                        qty = meets_or_scale_notional(c, sym, price, qty, behavior=min_notional_behavior)
                    if qty > 0:
                        qty_s = amount_to_precision(c, sym, qty)
                        entry_s = price_to_precision(c, sym, price)
                        tp_s    = price_to_precision(c, sym, tp)
                        sl_s    = price_to_precision(c, sym, sl)
                        trail_s = price_to_precision(c, sym, trail_level('sell', price, atr, CFG['signals']['short_the_rip'].get('trail_atr_mult', 1.0)))
                        signals_found += 1
                        note = f""
                        if TG and should_notify(ex_name) and float(qty_s) > 0:
                            key = f"{ex_name}:{sym}:SELL"
                            if can_notify(key):
                                cs_note = f" cs≈{cs}" if is_contract else ""
                                TG.send(f"🔴 [{ex_name}] SHORT {sym} @ {entry_s}\nTP~{tp_s} SL~{sl_s} Trail~{trail_s} Qty~{qty_s}{cs_note} — {sig['reason']}")
                                sent += 1
                        register_open(ex_name, sym, 'SELL', float(entry_s), float(tp_s), float(sl_s), float(trail_s), float(qty_s),
                                      extra={"is_contract": is_contract, "contractSize": cs})
                        opened_this_run.add(f"{ex_name}:{sym}:SELL")
                        log_signal({"ts": now_iso, "exchange": ex_name, "symbol": sym, "side": "SELL",
                                   "entry": entry_s, "tp": tp_s, "sl": sl_s, "trail": trail_s, "qty": qty_s,
                                   "is_contract": is_contract, "contractSize": cs, "reason": sig['reason']})

            # LONG
            if str_bounce and risk.can_trade() and day['signals'] < DAILY_MAX_TRADES:
                sig = str_bounce.signal(df30)
                if sig:
                    atr = float(last['atr'])
                    sl_pct_cfg = CFG['signals']['oversold_bounce'].get('sl_pct', None)
                    if sl_pct_cfg is not None:
                        sl = price * (1 - float(sl_pct_cfg))
                    else:
                        sl = price - CFG['signals']['oversold_bounce'].get('sl_atr_mult', 1.0) * atr
                    tp = price * (1 + sig['tp_pct'])
                    if (price - sl) / max(price, 1e-12) < MIN_STOP_PCT:
                        sig = None
                if sig:
                    qty_tokens = position_size_usdt(price, sl, risk.per_trade_risk_usd(), 'long')
                    qty = qty_tokens / cs if is_contract else qty_tokens
                    qty = clamp_amount(c, sym, qty, behavior=min_amount_behavior)
                    if qty > 0:
                        if price * qty * (cs if is_contract else 1.0) > MAX_NOTIONAL_PER_TRADE:
                            qty = MAX_NOTIONAL_PER_TRADE / (price * (cs if is_contract else 1.0))
                            qty = clamp_amount(c, sym, qty, behavior=min_amount_behavior)
                        qty = meets_or_scale_notional(c, sym, price, qty, behavior=min_notional_behavior)
                    if qty > 0:
                        qty_s = amount_to_precision(c, sym, qty)
                        entry_s = price_to_precision(c, sym, price)
                        tp_s    = price_to_precision(c, sym, tp)
                        sl_s    = price_to_precision(c, sym, sl)
                        trail_s = price_to_precision(c, sym, trail_level('buy', price, atr, CFG['signals']['oversold_bounce'].get('trail_atr_mult', 1.0)))
                        signals_found += 1
                        if TG and should_notify(ex_name) and float(qty_s) > 0:
                            key = f"{ex_name}:{sym}:BUY"
                            if can_notify(key):
                                cs_note = f" cs≈{cs}" if is_contract else ""
                                TG.send(f"🟢 [{ex_name}] LONG {sym} @ {entry_s}\nTP~{tp_s} SL~{sl_s} Trail~{trail_s} Qty~{qty_s}{cs_note} — {sig['reason']}")
                                sent += 1
                        register_open(ex_name, sym, 'BUY', float(entry_s), float(tp_s), float(sl_s), float(trail_s), float(qty_s),
                                      extra={"is_contract": is_contract, "contractSize": cs})
                        opened_this_run.add(f"{ex_name}:{sym}:BUY")
                        log_signal({"ts": now_iso, "exchange": ex_name, "symbol": sym, "side": "BUY",
                                   "entry": entry_s, "tp": tp_s, "sl": sl_s, "trail": trail_s, "qty": qty_s,
                                   "is_contract": is_contract, "contractSize": cs, "reason": sig['reason']})

    # 2) track open positions (paper): check TP/SL/trail
    for key, pos in list(state['open'].items()):
        ex = pos['exchange']; sym = pos['symbol']; side = pos['side']
        if key in opened_this_run:
            continue
        c = clients.get(ex)
        if not c:
            continue
        df30 = add_indicators(fetch_df(c, sym, TF_FAST), CFG['indicators'])
        last = df30.dropna().iloc[-1]
        price = float(last['close'])
        atr = float(last['atr'])
        new_trail = trail_level('sell' if side=='SELL' else 'buy', price, atr,
                                CFG['signals']['short_the_rip'].get('trail_atr_mult', 1.0) if side=='SELL' else CFG['signals']['oversold_bounce'].get('trail_atr_mult', 1.0))
        pos['trail'] = float(price_to_precision(c, sym, new_trail))
        hit = evaluate_tp_sl(price, side, pos['tp'], pos['sl'])
        if hit in ('TP','SL'):
            pnl = paper_pnl(side, pos['entry'], price, pos['qty'], CFG['execution']['fee_pct'])
            pos['exit'] = price
            pos['pnl'] = pnl
            pos['status'] = hit
            state['closed'].append(pos)
            del state['open'][key]
            if hit == 'TP':
                day['tp'] += 1
            if hit == 'SL':
                day['sl'] += 1
            day['pnl'] += pnl
            if TG and should_notify(ex):
                TG.send(f"📌 {hit} — [{ex}] {sym} {side} exit={price_to_precision(c, sym, price)} PnL≈{pnl:.2f} USDT")
        else:
            if bool(CFG.get('notify',{}).get('push_trail_updates', False)) and TG and should_notify(ex):
                TG.send(f"↘️ Trail upd — [{ex}] {sym} {side} new_stop≈{pos['trail']}")
except Exception as e:
    if TG: TG.send(f"⚠️ Run error: {e}")
    print(f"[error] {e}")
finally:
    save_state(state)
    save_day_stats(day)
    print(f"[summary] scanned={scanned} bearish_ok={bear_ok} signals_found={signals_found} sent={sent} open_now={len(state['open'])} closed_today_tp={day['tp']} sl={day['sl']} pnl≈{day['pnl']:.2f}")
    if TG and PUSH_NO_SIGNAL and sent == 0:
        TG.send(f"ℹ️ No signals this run. scanned={scanned} bearish_ok={bear_ok} signals_found={signals_found} sent={sent} open={len(state['open'])}")
    if TG and PUSH_DEBUG:
        TG.send(f"🧪 Debug: scanned={scanned} bearish_ok={bear_ok} signals_found={signals_found} sent={sent} open={len(state['open'])} pnl≈{day['pnl']:.2f}")
