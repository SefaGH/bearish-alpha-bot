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

exec_ex_env = os.getenv('EXECUTION_EXCHANGE', pick_execution_exchange())
exec_ex = (exec_ex_env or '').strip().lower() or next(iter(clients.keys()))
exec_client = clients.get(exec_ex) or next(iter(clients.values()))
exec_eng = ExecEngine(MODE, exec_client, CFG['execution']['fee_pct'], CFG['execution']['max_slippage_pct'], TG)

# Determine notification routing
send_all = bool(CFG.get('notify', {}).get('send_all', True))  # default True
exec_ex_env = os.getenv('EXECUTION_EXCHANGE', pick_execution_exchange())
exec_ex = (exec_ex_env or '').strip().lower() or next(iter(clients.keys()))
SEND_EX = exec_ex
if SEND_EX not in UNIVERSE and UNIVERSE:
    SEND_EX = next(iter(UNIVERSE.keys()))
print(f"[notify] send_all={send_all} SEND_EX={SEND_EX}")

def should_notify(ex_name: str) -> bool:
    return send_all or (ex_name == SEND_EX)

risk = RiskGuard(equity_usd=10_000, cfg=RiskConfig(
    per_trade_risk_pct=CFG['risk']['per_trade_risk_pct'],
    daily_loss_limit_pct=CFG['risk']['daily_loss_limit_pct'],
    cool_down_min=CFG['risk']['cool_down_min']
))

str_short = ShortTheRip(CFG['signals']['short_the_rip']) if CFG['signals']['short_the_rip']['enable'] else None
str_bounce = OversoldBounce(CFG['signals']['oversold_bounce']) if CFG['signals']['oversold_bounce']['enable'] else None

TF_FAST = CFG['timeframes']['fast']
TF_MID  = CFG['timeframes']['mid']
TF_SLOW = CFG['timeframes']['slow']

MIN_COOLDOWN_SEC = int(CFG.get('notify',{}).get('min_cooldown_sec', 300))
PUSH_NO_SIGNAL = bool(CFG.get('notify',{}).get('push_no_signal', True))
PUSH_DEBUG = bool(CFG.get('notify',{}).get('push_debug', False))

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

# ---- One-shot run with trailing suggestions ----
scanned = 0
bear_ok = 0
signals_found = 0
sent = 0
now_iso = datetime.datetime.utcnow().isoformat()

try:
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

            # SHORT THE RIP
            if str_short and risk.can_trade():
                sig = str_short.signal(df30, df1h)
                if sig:
                    signals_found += 1
                    entry = float(df30.dropna().iloc[-1]['close'])
                    atr = float(df30.dropna().iloc[-1]['atr'])
                    tp, sl = initial_stops('sell', entry, atr, sig['sl_atr_mult'], sig['tp_pct'])
                    # Sizing (paper) with normalization
                    qty = position_size_usdt(entry, sl, risk.per_trade_risk_usd(), 'short')
                    qty_s = amount_to_precision(c, sym, qty)
                    entry_s = price_to_precision(c, sym, entry)
                    tp_s    = price_to_precision(c, sym, tp)
                    sl_s    = price_to_precision(c, sym, sl)
                    trail_s = price_to_precision(c, sym, trail_level('sell', entry, atr, CFG['signals']['short_the_rip'].get('trail_atr_mult', 1.0)))
                    if ex_name == SEND_EX:
                        key = f"{ex_name}:{sym}:SELL"
                        if TG and can_notify(key):
                            TG.send(f"🔴 [{ex_name}] SHORT {sym} @ {entry_s}\nTP~{tp_s} SL~{sl_s} Trail~{trail_s} Qty~{qty_s} — {sig['reason']}")
                            sent += 1
                    log_signal({
                        "ts": now_iso, "exchange": ex_name, "symbol": sym, "side": "SELL",
                        "entry": entry_s, "tp": tp_s, "sl": sl_s, "trail": trail_s, "qty": qty_s, "reason": sig['reason']
                    })

            # OVERSOLD BOUNCE
            if str_bounce and risk.can_trade():
                sig = str_bounce.signal(df30)
                if sig:
                    signals_found += 1
                    entry = float(df30.dropna().iloc[-1]['close'])
                    atr = float(df30.dropna().iloc[-1]['atr'])
                    tp, sl = initial_stops('buy', entry, atr, CFG['signals']['oversold_bounce'].get('sl_atr_mult', 0.0), sig['tp_pct'])
                    qty = position_size_usdt(entry, sl, risk.per_trade_risk_usd(), 'long')
                    qty_s = amount_to_precision(c, sym, qty)
                    entry_s = price_to_precision(c, sym, entry)
                    tp_s    = price_to_precision(c, sym, tp)
                    sl_s    = price_to_precision(c, sym, sl)
                    trail_s = price_to_precision(c, sym, trail_level('buy', entry, atr, CFG['signals']['oversold_bounce'].get('trail_atr_mult', 1.0)))
                    if ex_name == SEND_EX:
                        key = f"{ex_name}:{sym}:BUY"
                        if TG and can_notify(key):
                            TG.send(f"🟢 [{ex_name}] LONG {sym} @ {entry_s}\nTP~{tp_s} SL~{sl_s} Trail~{trail_s} Qty~{qty_s} — {sig['reason']}")
                            sent += 1
                    log_signal({
                        "ts": now_iso, "exchange": ex_name, "symbol": sym, "side": "BUY",
                        "entry": entry_s, "tp": tp_s, "sl": sl_s, "trail": trail_s, "qty": qty_s, "reason": sig['reason']
                    })

except Exception as e:
    if TG: TG.send(f"⚠️ Run error: {e}")
    print(f"[error] {e}")
finally:
    print(f"[summary] scanned={scanned} bearish_ok={bear_ok} signals_found={signals_found} sent={sent}")
    if TG and bool(CFG.get('notify',{}).get('push_no_signal', True)) and sent == 0:
        TG.send(f"ℹ️ No signals this run. scanned={scanned} bearish_ok={bear_ok} signals_found={signals_found} sent={sent}")
    if TG and bool(CFG.get('notify',{}).get('push_debug', False)):
        TG.send(f"🧪 Debug: scanned={scanned} bearish_ok={bear_ok} signals_found={signals_found} sent={sent} SEND_EX={SEND_EX}")
