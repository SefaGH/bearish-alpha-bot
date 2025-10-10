import os, yaml, pandas as pd, time
from dotenv import load_dotenv
from core.multi_exchange import build_clients_from_env
from core.indicators import add_indicators
from core.regime import is_bearish_regime
from core.risk import RiskGuard, RiskConfig
from core.sizing import position_size_usdt
from core.exec_engine import ExecEngine
from core.notify import Telegram
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

exec_ex = pick_execution_exchange()
exec_client = clients.get(exec_ex) or next(iter(clients.values()))
exec_eng = ExecEngine(MODE, exec_client, CFG['execution']['fee_pct'], CFG['execution']['max_slippage_pct'], TG)

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

SEND_EX = os.getenv('EXECUTION_EXCHANGE', exec_ex)
MIN_COOLDOWN_SEC = int(CFG.get('notify',{}).get('min_cooldown_sec', 300))
LAST_SENT = {}

def can_notify(key: str) -> bool:
    now = time.time()
    last = LAST_SENT.get(key, 0)
    if now - last >= MIN_COOLDOWN_SEC:
        LAST_SENT[key] = now
        return True
    return False

def fmt_price(client, symbol, price: float) -> str:
    try:
        return client.ex.price_to_precision(symbol, price)
    except Exception:
        return f"{price:.6f}"

def fetch_df(client, symbol, tf):
    o = client.ohlcv(symbol, tf, limit=400)
    df = pd.DataFrame(o, columns=['ts','open','high','low','close','vol'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df

# ---- One-shot run ----
try:
    for ex_name, syms in UNIVERSE.items():
        c = clients[ex_name]
        for sym in syms:
            df30 = add_indicators(fetch_df(c, sym, TF_FAST), CFG['indicators'])
            df1h = add_indicators(fetch_df(c, sym, TF_MID),  CFG['indicators'])
            df4h = add_indicators(fetch_df(c, sym, TF_SLOW), CFG['indicators'])

            if not is_bearish_regime(df4h):
                continue

            # SHORT THE RIP
            if str_short and risk.can_trade():
                sig = str_short.signal(df30, df1h)
                if sig:
                    entry = float(df30.dropna().iloc[-1]['close'])
                    atr = float(df30.dropna().iloc[-1]['atr'])
                    sl = entry + sig['sl_atr_mult']*atr
                    tp = entry * (1 - sig['tp_pct'])
                    qty = position_size_usdt(entry, sl, risk.per_trade_risk_usd(), 'short')
                    if ex_name == SEND_EX:
                        key = f"{ex_name}:{sym}:SELL"
                        if TG and can_notify(key):
                            entry_s = fmt_price(c, sym, entry)
                            tp_s    = fmt_price(c, sym, tp)
                            sl_s    = fmt_price(c, sym, sl)
                            TG.send(f"🔴 [{ex_name}] SHORT {sym} @ {entry_s}\nTP~{tp_s} SL~{sl_s} — {sig['reason']}")
                    # Live emir (manuel aç)
                    # if ex_name == exec_ex: exec_eng.market_order(sym, 'sell', qty)

            # OVERSOLD BOUNCE
            if str_bounce and risk.can_trade():
                sig = str_bounce.signal(df30)
                if sig:
                    entry = float(df30.dropna().iloc[-1]['close'])
                    sl = entry * (1 - sig['sl_pct'])
                    tp = entry * (1 + sig['tp_pct'])
                    qty = position_size_usdt(entry, sl, risk.per_trade_risk_usd(), 'long')
                    if ex_name == SEND_EX:
                        key = f"{ex_name}:{sym}:BUY"
                        if TG and can_notify(key):
                            entry_s = fmt_price(c, sym, entry)
                            tp_s    = fmt_price(c, sym, tp)
                            sl_s    = fmt_price(c, sym, sl)
                            TG.send(f"🟢 [{ex_name}] LONG {sym} @ {entry_s}\nTP~{tp_s} SL~{sl_s} — {sig['reason']}")
                    # Live emir (manuel aç)
                    # if ex_name == exec_ex: exec_eng.market_order(sym, 'buy', qty)
except Exception as e:
    if TG: TG.send(f"⚠️ Run error: {e}")
