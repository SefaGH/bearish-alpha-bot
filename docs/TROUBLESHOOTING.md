# Troubleshooting Guide - Sorun Giderme Kılavuzu

## Yaygın Sorunlar ve Çözümler

### 1. "No module named 'core'" Hatası

**Hata:**
```
ModuleNotFoundError: No module named 'core'
```

**Çözüm:**
Bot'u `src` dizininden çalıştırın:
```bash
cd src
python main.py
```

VEYA Python path'i ayarlayın:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/main.py
```

---

### 2. "Universe is empty" Hatası

**Hata:**
```
Universe is empty. Lower min_quote_volume_usdt, increase max_symbols_per_exchange, 
or set prefer_perps: false to include spot.
```

**Nedenler:**
- EXCHANGES environment variable boş veya hatalı
- Exchange API key'leri eksik/yanlış
- Config'deki filtreler çok katı

**Çözüm:**

**a) EXCHANGES'i kontrol edin:**
```bash
echo $EXCHANGES
# Boşsa veya yanlışsa:
export EXCHANGES=binance,bingx
```

**b) API credentials kontrol edin:**
```bash
# Binance için
echo $BINANCE_KEY
echo $BINANCE_SECRET

# Eğer boşlarsa .env dosyasını kontrol edin
```

**c) Config'i gevşetin:**
`config/config.example.yaml`:
```yaml
universe:
  min_quote_volume_usdt: 100000  # Düşürün (örn: 1000000 → 100000)
  max_symbols_per_exchange: 50   # Artırın
  prefer_perps: false            # Spot'u da dahil edin
```

---

### 3. Test Hataları

**Hata:**
```
No module named pytest
```

**Çözüm:**
```bash
pip install pytest
# VEYA
pip install -r requirements.txt
```

**Hata:**
```
Test functions should return None
```

**Açıklama:**
Bu bir warning, test çalışıyor. Görmezden gelebilirsiniz.

---

### 4. API Authentication Hataları

**Hata:**
```
ccxt.AuthenticationError: Invalid API-key, IP, or permissions
```

**Çözümler:**

**a) API Key/Secret kontrol edin:**
- Doğru exchange için key'leri girdiniz mi?
- Boşluk veya özel karakter var mı?

**b) IP Whitelist:**
- Bazı exchange'ler IP whitelist gerektirir
- Exchange ayarlarından IP'nizi ekleyin

**c) API Permissions:**
Gereken izinler:
- ✅ Read - Market data
- ✅ Trade - Order açma/kapatma (live mode için)
- ❌ Withdraw - GEREKLİ DEĞİL (güvenlik için kapalı tutun)

**d) Password/Passphrase:**
KuCoin, Bitget, AscendEX için password gerekir:
```bash
export KUCOIN_PASSWORD=your_api_passphrase
export BITGET_PASSWORD=your_api_passphrase
```

---

### 5. Telegram Bildirimleri Çalışmıyor

**Sorun:** Bot çalışıyor ama Telegram mesajı gelmiyor.

**Kontroller:**

**a) Token ve Chat ID:**
```bash
echo $TELEGRAM_BOT_TOKEN
echo $TELEGRAM_CHAT_ID

# Boşlarsa:
export TELEGRAM_BOT_TOKEN=123456:ABC...
export TELEGRAM_CHAT_ID=-1001234567890
```

**b) Bot ile konuşma başlattınız mı?**
- Telegram'da botunuza `/start` gönderin
- Veya botu gruba ekleyin

**c) Chat ID'yi doğru aldınız mı?**
```bash
# Bot token ile test edin:
curl https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
```

**d) Network bağlantısı:**
```bash
# Telegram API'ye erişebildiğinizden emin olun:
curl https://api.telegram.org/
```

---

### 6. "IndexError: single positional indexer is out-of-bounds"

**Hata:**
```
IndexError: single positional indexer is out-of-bounds
```

**Neden:**
DataFrame boş veya yeterli veri yok.

**Çözüm:**
Bu hata artık düzeltildi. En son kodda:
- `has_min_bars()` kontrolü var
- `.dropna()` kullanılıyor
- `.iloc[-1]` öncesi kontrol var

Eğer hala görüyorsanız, kodu güncelleyin:
```bash
git pull origin main
```

---

### 7. Pozisyon Büyüklüğü Çok Büyük/Küçük

**Sorun:** Hesaplanan pozisyon büyüklüğü beklenenden farklı.

**Kontrol edin:**

**a) Risk parametreleri:**
```bash
echo $RISK_EQUITY_USD        # Default: 1000
echo $RISK_PER_TRADE_RISK_PCT # Default: 0.01 (1%)
echo $RISK_RISK_USD_CAP      # Default: 50
```

**b) Stop loss mesafesi:**
- Çok dar stop = büyük pozisyon
- Çok geniş stop = küçük pozisyon

**c) Minimum lot size:**
Exchange'in minimum lot size'ını kontrol edin.

**Örnek hesaplama:**
```
Entry: $100
Stop: $99 (1% risk)
Risk amount: $10
Position size: $10 / $1 = 10 units
```

---

### 8. Backtest Sonuç Üretmiyor

**Hata:**
```
No results produced. Check data length or grid ranges.
```

**Çözümler:**

**a) Yeterli veri var mı?**
```bash
# Daha fazla candle çekin:
export BT_LIMIT=2000
export BT_LIMIT_30M=2000
export BT_LIMIT_1H=2000
```

**b) Exchange'de sembol var mı?**
```bash
# Sembolü kontrol edin:
export BT_SYMBOL=BTC/USDT  # Doğru format: BASE/QUOTE
```

**c) Exchange erişimi:**
```bash
# Exchange credentials doğru mu?
echo $BT_EXCHANGE
echo ${BT_EXCHANGE^^}_KEY  # Uppercase exchange name + _KEY
```

**d) Debug logging açın:**
```bash
# Detaylı logları görmek için:
export LOG_LEVEL=DEBUG
python src/backtest/param_sweep.py
```

---

### 8.1. KuCoin API Veri Çekme Sorunları (GitHub Actions)

**Sorun:**
KuCoin API anahtarları doğru eklenmiş ama GitHub Actions'da veri çekilemiyor ve hata mesajı görünmüyor.

**Root Cause:**
- Backtest scriptleri logging yapılandırması eksikti
- Hatalar SystemExit ile sessizce sonlandırılıyordu
- API hataları yakalanmıyor ve loglanmıyordu

**Çözüm (artık otomatik):**
✅ v1.x.x ve sonrasında bu sorunlar düzeltildi:
- Tüm backtest scriptleri artık detaylı logging yapıyor
- API hataları stderr'a yazılıyor (GitHub Actions'da görünür)
- Credential sorunları için açıklayıcı mesajlar gösteriliyor
- RuntimeError kullanılarak hatalar düzgün yakalanıyor

**GitHub Actions'da Debug:**

1. **Workflow loglarını kontrol edin:**
   - Actions sekmesinde workflow run'ı açın
   - "Run param sweep" step'inin loglarını inceleyin
   - Artık tüm hatalar burada görünecek

2. **Credential doğrulama:**
   ```yaml
   - name: Debug credentials
     env:
       EXCHANGES: ${{ secrets.EXCHANGES }}
       KUCOIN_KEY: ${{ secrets.KUCOIN_KEY }}
     run: |
       echo "EXCHANGES set: $([[ -n \"$EXCHANGES\" ]] && echo 'YES' || echo 'NO')"
       echo "KUCOIN_KEY set: $([[ -n \"$KUCOIN_KEY\" ]] && echo 'YES' || echo 'NO')"
   ```

3. **LOG_LEVEL ayarlayın:**
   ```yaml
   - name: Run backtest
     env:
       LOG_LEVEL: DEBUG  # Detaylı loglar için
       EXCHANGES: ${{ secrets.EXCHANGES }}
       # ... diğer env vars
     run: python src/backtest/param_sweep.py
   ```

**KuCoin Özel Notlar:**

- `kucoin` (spot) ve `kucoinfutures` her ikisi de `KUCOIN_*` credentials kullanabilir
- 3 credential gerekli: `KUCOIN_KEY`, `KUCOIN_SECRET`, `KUCOIN_PASSWORD`
- Alternatif olarak `KUCOINFUTURES_*` da kullanılabilir

**Hata mesajı örneği (artık görünür):**
```
ERROR - ❌ Symbol validation failed: RuntimeError: Symbol validation failed for kucoinfutures
ERROR - ⚠️ AUTHENTICATION ERROR: Please verify your KUCOINFUTURES API credentials are correct
ERROR -    KuCoin Futures can use either KUCOIN_* or KUCOINFUTURES_* credentials
ERROR -    Required: KUCOIN_KEY + KUCOIN_SECRET + KUCOIN_PASSWORD
```

---

### 9. Deprecation Warnings

**Warning:**
```
DeprecationWarning: datetime.datetime.utcnow() is deprecated
```

**Çözüm:**
Bu sorun düzeltildi. Eğer hala görüyorsanız:
```bash
git pull origin copilot/improve-bot-functionality
```

Veya manuel olarak:
```python
# Eski
from datetime import datetime
datetime.utcnow()

# Yeni
from datetime import datetime, timezone
datetime.now(timezone.utc)
```

---

### 10. GitHub Actions Çalışmıyor

**Sorun:** Workflow başarısız oluyor.

**Kontrol edin:**

**a) Secrets ayarlandı mı?**
- Repo → Settings → Secrets and variables → Actions
- Gerekli secrets:
  - EXCHANGES
  - {EXCHANGE}_KEY
  - {EXCHANGE}_SECRET
  - (Opsiyonel) TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

**b) Python version:**
Tüm workflow'larda `python-version: "3.12"` olmalı.

**c) Artifacts:**
"No files found" hatası normaldir eğer sinyal yoksa.

---

## Debug Yöntemleri

### 1. Verbose Logging
```bash
export LOG_LEVEL=DEBUG
python src/main.py
```

### 2. Smoke Test
```bash
python tests/smoke_test.py
```

### 3. Dry Run
```bash
MODE=paper EXCHANGES="" python src/main.py
# Universe hatası beklenir ama import hataları olmamalı
```

### 4. Test Bir Exchange
```bash
export EXCHANGES=binance
export BINANCE_KEY=your_key
export BINANCE_SECRET=your_secret
python src/main.py
```

### 5. Manuel Test
```python
import sys
sys.path.insert(0, 'src')

# Test exchange client
from core.multi_exchange import build_clients_from_env
clients = build_clients_from_env()
print(clients)

# Test market data
client = clients['binance']
data = client.ohlcv('BTC/USDT', '1h', 10)
print(data)
```

---

## Hızlı Kontrol Listesi

Sorun yaşıyorsanız sırayla kontrol edin:

- [ ] Python 3.12 kullanıyor musunuz? `python --version`
- [ ] Requirements yüklü mü? `pip install -r requirements.txt`
- [ ] EXCHANGES set edilmiş mi? `echo $EXCHANGES`
- [ ] API credentials doğru mu? `echo $BINANCE_KEY`
- [ ] Config dosyası var mı? `ls config/config.example.yaml`
- [ ] Smoke test geçiyor mu? `python tests/smoke_test.py`
- [ ] Log seviyesi DEBUG mi? `export LOG_LEVEL=DEBUG`

---

## Yardım Alma

Hala sorun çözemediyseniz:

1. **Log'ları toplayın:**
   ```bash
   LOG_LEVEL=DEBUG python src/main.py 2>&1 | tee bot_debug.log
   ```

2. **Smoke test sonuçları:**
   ```bash
   python tests/smoke_test.py > smoke_test.log 2>&1
   ```

3. **Environment bilgileri:**
   ```bash
   echo "Python: $(python --version)"
   echo "OS: $(uname -a)"
   echo "EXCHANGES: $EXCHANGES"
   pip list | grep -E "ccxt|pandas|numpy"
   ```

4. **Issue açın:**
   GitHub'da issue açarken yukarıdaki bilgileri ekleyin.

---

## Faydalı Komutlar

```bash
# Tam test suite
pytest tests/ -v

# Sadece smoke test
python tests/smoke_test.py

# Pozisyon sizing test
pytest tests/test_sizing.py -v

# Bot çalıştır (paper mode)
cd src && python main.py

# Backtest çalıştır
python src/backtest/param_sweep.py

# Log dosyası oluştur
LOG_LEVEL=INFO python src/main.py 2>&1 | tee logs/bot_$(date +%Y%m%d_%H%M%S).log
```
