import pandas as pd
from .base_strategy import BaseStrategy 

class OversoldBounce(BaseStrategy):
    def __init__(self, cfg):
        super().__init__(strategy_name="oversold_bounce", config=cfg)

    # 🔥 DEĞİŞİKLİK: Metodun adı kontrata uyacak şekilde 'generate_signal' olarak değiştirildi
    # ve `async` anahtar kelimesi eklendi. df_30m parametresi artık kullanılmayacak,
    # veri market_data_pipeline'dan alınacak.
    async def generate_signal(self, symbol: str, ml_context=None) -> dict | None:
        # Veri artık pipeline'dan çekilecek (bu daha modern bir yaklaşım)
        df_30m = await self.market_data_pipeline.get_ohlcv(symbol, '30m')
        if df_30m is None or df_30m.empty:
            return None # Veri yoksa sinyal üretme

        last = df_30m.dropna().iloc[-1]

        rsi_max = self.strategy_config.get('rsi_max', self.strategy_config.get('rsi_min', 25))
        try:
            rsi_max = float(rsi_max)
        except Exception:
            rsi_max = 25.0

        rsi_val = float(last['rsi'])

        if rsi_val <= rsi_max:
            return {
                "strategy_name": self.strategy_name, # Sinyale strateji adını ekleyelim
                "side": "buy",
                "symbol": symbol, # Sinyale sembolü ekleyelim
                "reason": f"RSI oversold {rsi_val:.1f}",
                "tp_pct": float(self.strategy_config.get("tp_pct", 0.015)),
                "sl_pct": (float(self.strategy_config["sl_pct"]) if "sl_pct" in self.strategy_config else None),
                "sl_atr_mult": float(self.strategy_config.get("sl_atr_mult", 1.0)),
            }
        return None
