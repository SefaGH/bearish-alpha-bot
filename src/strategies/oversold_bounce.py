 class OversoldBounce:
     def __init__(self, cfg):
-        self.cfg = cfg
+        self.cfg = cfg

     def signal(self, df_30m: pd.DataFrame):
         last = df_30m.dropna().iloc[-1]
-        if last['rsi'] <= self.cfg['rsi_max']:
+        # Geriye dönük uyumluluk: rsi_max yoksa rsi_min'ı ters mantıkla kabul et
+        rsi_max = self.cfg.get('rsi_max')
+        if rsi_max is None:
+            # Eski konfig desteği: rsi_min verildiyse onu rsi_max gibi kullan
+            rsi_max = self.cfg.get('rsi_min', 25)
+        if last['rsi'] <= float(rsi_max):
             return {
                 'side': 'buy',
                 'reason': f"RSI oversold {last['rsi']:.1f}",
-                'tp_pct': self.cfg['tp_pct'],
-                'sl_pct': self.cfg['sl_pct']
+                'tp_pct': float(self.cfg.get('tp_pct', 0.015)),
+                'sl_pct': float(self.cfg.get('sl_pct', 0.008)) if 'sl_pct' in self.cfg else None
             }
         return None
