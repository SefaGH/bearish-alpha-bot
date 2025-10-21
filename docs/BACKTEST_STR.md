# ShortTheRip Backtest (30m entries + 1h context)

Bu döküman, **ShortTheRip** stratejisi için parametre taramasını (RSI, TP, SL-ATR, EMA koşulları) sadece **GitHub Actions** üzerinden nasıl koşturacağını anlatır.

## Dosya
- `src/backtest/param_sweep_str.py`

## Çalışma Mantığı
- 30m ve 1h OHLCV verisi ccxt ile çekilir.
- İki timeframe indikatörleri (RSI/EMA/ATR) hesaplanır.
- 30m üzerinde **RSI >= rsi_min** koşulunda, opsiyonel olarak:
  - **EMA alignment**: 30m’de `ema21 < ema50 <= ema200`
  - **Band touch proxy**: 1h’de fiyat `ema50_1h`’yi **dokunmuş/üstünde** kabul edilir (yoksa 30m’de `close >= ema50`).
- Short girişi için **sonraki mum** üzerinden TP/SL testi yapılır.

## GitHub Actions (örnek workflow)
`.github/workflows/backtest_str.yml` içine aşağıdaki şablonu koyabilirsiniz (bu dosya zip’te var):

```yaml
name: Run Backtest (ShortTheRip)

on:
  workflow_dispatch:
    inputs:
      BT_SYMBOL:
        description: "Symbol (e.g. BTC/USDT)"
        required: true
        default: "BTC/USDT"
      BT_EXCHANGE:
        description: "Exchange key (must be in EXCHANGES env)"
        required: true
        default: "bingx"
      BT_LIMIT_30M:
        description: "30m bars to fetch"
        required: true
        default: "1000"
      BT_LIMIT_1H:
        description: "1h bars to fetch"
        required: true
        default: "1000"

jobs:
  backtest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install deps
        run: pip install --no-cache-dir -r requirements.txt
      - name: Run STR sweep
        env:
          EXCHANGES: ${{ secrets.EXCHANGES }}
          EXECUTION_EXCHANGE: ${{ github.event.inputs.BT_EXCHANGE }}
          BINGX_KEY:   ${{ secrets.BINGX_KEY }}
          BINGX_SECRET:${{ secrets.BINGX_SECRET }}
          BITGET_KEY:  ${{ secrets.BITGET_KEY }}
          BITGET_SECRET:${{ secrets.BITGET_SECRET }}
          BITGET_PASSWORD:${{ secrets.BITGET_PASSWORD }}
          BINANCE_KEY: ${{ secrets.BINANCE_KEY }}
          BINANCE_SECRET:${{ secrets.BINANCE_SECRET }}
          KUCOIN_KEY:  ${{ secrets.KUCOIN_KEY }}
          KUCOIN_SECRET:${{ secrets.KUCOIN_SECRET }}
          KUCOIN_PASSWORD:${{ secrets.KUCOIN_PASSWORD }}
          ASCENDEX_KEY:${{ secrets.ASCENDEX_KEY }}
          ASCENDEX_SECRET:${{ secrets.ASCENDEX_SECRET }}
          ASCENDEX_PASSWORD:${{ secrets.ASCENDEX_PASSWORD }}
          CONFIG_PATH: config/config.example.yaml
          BT_SYMBOL:   ${{ github.event.inputs.BT_SYMBOL }}
          BT_EXCHANGE: ${{ github.event.inputs.BT_EXCHANGE }}
          BT_LIMIT_30M:${{ github.event.inputs.BT_LIMIT_30M }}
          BT_LIMIT_1H: ${{ github.event.inputs.BT_LIMIT_1H }}
        run: python -u src/backtest/param_sweep_str.py
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: backtest-results-str
          path: data/backtests/**
```

## Çıktıların Yorumlanması
CSV sütunları:
- Parametreler: `rsi_min`, `tp_pct`, `sl_atr_mult`, `require_band_touch`, `require_ema_align`
- Metrikler: `trades`, `win_rate`, `avg_pnl`, `rr`, `net_pnl`

Sonuçlar `avg_pnl`, `win_rate`, `trades` önceliğiyle sıralanır. İlk 10 satır CI log’unda görünür, tüm sonuçları workflow artefact’ından indirirsin.