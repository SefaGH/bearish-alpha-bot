# Konfigürasyon Referansı

> Varsayılan yol: `config/config.example.yaml` (Actions’ta `CONFIG_PATH` ile değiştirilebilir)

## Bölümler

### `execution`
```yaml
execution:
  mode: paper            # veya live (öneri: önce paper)
  order_type: market
  tif: IOC
  fee_bps: 8             # örnek
  slippage_bps: 5        # örnek
  leverage: 2
```

### `risk`
```yaml
risk:
  equity_usd: 1000
  per_trade_risk_pct: 0.5
  daily_loss_limit_pct: 5
  risk_usd_cap: 25
  max_notional_per_trade: 500
```

### `signals`
```yaml
signals:
  oversold_bounce:
    enable: true
    ignore_regime: true     # testte true; prod'da false yap
    rsi_max: 38             # test önerisi (default daha sıkı olabilir)
    tp_pct: 0.010
    sl_atr_mult: 1.2

  short_the_rip:
    enable: true
    # ignore_regime: true   # opsiyonel
    rsi_min: 55
    tp_pct: 0.010
    sl_atr_mult: 1.2
```

> Not: Eğer dosyada `indicators` ve `universe` yoksa, kod **varsayılan değerleri** kullanır. Eklemek istersen:

### (Opsiyonel) `universe`
```yaml
universe:
  top_n_per_exchange: 20
  only_linear: true
  min_quote_volume_usdt: 500000
```

### (Opsiyonel) `indicators`
```yaml
indicators:
  rsi_period: 14
  atr_period: 14
  ema_fast: 21
  ema_mid: 50
  ema_slow: 200
```
