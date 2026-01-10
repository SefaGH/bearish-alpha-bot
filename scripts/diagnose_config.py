import os
from config.live_trading_config import LiveTradingConfiguration as L

print("--- CONFIG DIAGNOSIS ---")
print(f"Current Working Directory: {os.getcwd()}")
print(f"ENV CONFIG_PATH: {os.getenv('CONFIG_PATH')}")
print(f"ENV DAILY_LOSS_LIMIT_PCT: {os.getenv('DAILY_LOSS_LIMIT_PCT')}")

try:
    cfg = L.load(force_reload=True, log_summary=False)
    risk = cfg.get('risk', {})
    print(f"LOADED risk.daily_loss_limit_pct: {risk.get('daily_loss_limit_pct')}")
    print(f"LOADED risk.max_portfolio_risk: {risk.get('max_portfolio_risk')}")
    print(f"LOADED risk.max_notional_pct_per_trade: {risk.get('max_notional_pct_per_trade')}")
    print(f"LOADED risk.max_position_size: {risk.get('max_position_size')}")
except Exception as e:
    print(f"ERROR loading config: {e}")
print("------------------------")

