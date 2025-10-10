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
    raise SystemExit('EXCHANGES boş. .env dosyasını doldurun.')

sym_source = os.getenv('SYM_SOURCE','AUTO').upper()
if sym_source == 'AUTO':
    UNIVERSE = build_universe(clients, CFG)
else:
    manual = [s.strip() for s in os.getenv('SYMBOLS','BTC/USDT').split(',') if s.strip()]
    UNIVERSE = { pick_execution_exchange(): manual }

exec_ex = pick_execution_exchange()
exec_client = clients[exec_ex]

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

def fetch_df(client, symbol, tf):
    o = client.ohlcv(symbol, tf, limit=400)
    df = pd.DataFrame(o, columns=['ts','open','high','low','close','vol'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df

while True:
    try:
        for ex_name, syms in UNIVERSE.items():
            c = clients[ex_name]
            for sym in syms:
                df30 = add_indicators(fetch_df(c, sym, TF_FAST), CFG['indicators'])
                df1h = add_indicators(fetch_df(c, sym, TF_MID),  CFG['indicators'])
                df4h = add_indicators(fetch_df(c, sym, TF_SLOW), CFG['indicators'])

                if not is_bearish_regime(df4h):
                    continue

                if str_short and risk.can_trade():
                    sig = str_short.signal(df30, df1h)
                    if sig:
                        entry = float(df30.dropna().iloc[-1]['close'])
                        atr = float(df30.dropna().iloc[-1]['atr'])
                        sl = entry + sig['sl_atr_mult']*atr
                        tp = entry * (1 - sig['tp_pct'])
                        qty = position_size_usdt(entry, sl, risk.per_trade_risk_usd(), 'short')
                        if TG: TG.send(f"🔴 [{ex_name}] SHORT {sym} @ {entry:.4f}\nTP~{tp:.4f} SL~{sl:.4f} — {sig['reason']}")
                        # if ex_name == exec_ex: exec_eng.market_order(sym, 'sell', qty)

                if str_bounce and risk.can_trade():
                    sig = str_bounce.signal(df30)
                    if sig:
                        entry = float(df30.dropna().iloc[-1]['close'])
                        sl = entry * (1 - sig['sl_pct'])
                        tp = entry * (1 + sig['tp_pct'])
                        qty = position_size_usdt(entry, sl, risk.per_trade_risk_usd(), 'long')
                        if TG: TG.send(f"🟢 [{ex_name}] LONG {sym} @ {entry:.4f}\nTP~{tp:.4f} SL~{sl:.4f} — {sig['reason']}")
                        # if ex_name == exec_ex: exec_eng.market_order(sym, 'buy', qty)
        time.sleep(60)
    except Exception as e:
        if TG: TG.send(f"⚠️ Loop error: {e}")
        time.sleep(10)
