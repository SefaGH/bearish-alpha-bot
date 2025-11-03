import pandas as pd
from .base_strategy import BaseStrategy

class ShortTheRip(BaseStrategy):
    def __init__(self, cfg):
        super().__init__(strategy_name="short_the_rip", config=cfg)

    # 🔥 DEĞİŞİKLİK: Metodun adı kontrata uyacak şekilde 'generate_signal' olarak değiştirildi
    # ve `async` anahtar kelimesi eklendi. Parametreler kontrata uyarlandı.
    async def generate_signal(self, symbol: str, ml_context=None) -> dict | None:
        # Veriler pipeline üzerinden çekilecek
        df_30m = await self.market_data_pipeline.get_ohlcv(symbol, '30m')
        df_1h = await self.market_data_pipeline.get_ohlcv(symbol, '1h')
        
        if df_30m is None or df_30m.empty or df_1h is None or df_1h.empty:
            return None # Gerekli verilerden biri yoksa sinyal üretme

        last30 = df_30m.dropna().iloc[-1]
        last1h = df_1h.dropna().iloc[-1]

        rsi_min = self.strategy_config.get('rsi_min', 61)
        try:
            rsi_min = float(rsi_min)
        except Exception:
            rsi_min = 61.0

        rsi_val = float(last30['rsi'])

        ema_ok = True
        if all(col in last30.index for col in ('ema21','ema50','ema200')):
            ema_ok = float(last30['ema21']) < float(last30['ema50']) <= float(last30['ema200'])

        if rsi_val >= rsi_min and ema_ok:
            return {
                "strategy_name": self.strategy_name,
                "side": "sell",
                "symbol": symbol,
                "reason": f"RSI overbought {rsi_val:.1f} (rip)",
                "tp_pct": float(self.strategy_config.get("tp_pct", 0.012)),
                "sl_atr_mult": float(self.strategy_config.get("sl_atr_mult", 1.2)),
            }
        return None
