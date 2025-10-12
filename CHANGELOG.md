# Değişiklik Günlüğü

## 2025-10-12
- **Python 3.12**’ye geçiş; `requirements.txt` 3.12 uyumlu hale getirildi
  - `pandas-ta==0.4.67b0`, `numpy>=2.2.6`, `pandas>=2.2.3,<3`
- `src/core/ccxt_client.py`:
  - `ohlcv()` retry loop düzeltildi; son istisna aynen fırlatılıyor
- `.github/README_BOT.md` ve `README.md` güncellendi (çalışma şekli + kurulum)
- `src/main.py`:
  - veri yeterlilik guard’ları eklendi (min bar kontrolü, `dropna()`)
  - sinyal çağrılarında min 50 bar şartı
  - her koşuda `data/RUN_SUMMARY.txt` yazımı (artifact garantisi)
- **Backtest araçları** eklendi:
  - `src/backtest/param_sweep.py` (OB)
  - `src/backtest/param_sweep_str.py` (STR)
  - Nightly workflow ve raporlama (`scripts/summarize_backtests.py`)
- Workflow düzeltmeleri:
  - Nightly için matrix yerine **bash döngüsü** (split fonksiyonu hatası giderildi)
  - Upload artifact adımlarında `if-no-files-found: ignore` önerisi
- Dokümantasyonda yazım/isim tutarlılığı:
  - `dokümantasyon` yazımı
  - Risk belge anahtarları (cool_down_min/cooldown_min notu giderildi)
