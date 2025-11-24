# Bearish Alpha Bot - Azure VM Operations Playbook

Bu doküman, Azure VM üzerindeki Docker tabanlı prod ortam için günlük operasyon ve temel sorun giderme adımlarını özetler.

## 1. VM'ye Bağlanma

```pwsh
ssh azureuser@<VM_IP>
```

VM'ye bağlandıktan sonra komutlar genellikle `azureuser` ev dizininde (`/home/azureuser`) çalıştırılır.

## 2. Container Durumunu Kontrol Etme

```bash
# Çalışan container'ları listele
sudo docker ps

# Tüm container'lar (duranlar dahil)
sudo docker ps -a
```

Beklenen:
- `bearish-bot` isimli container `STATUS` kolonunda `Up` olarak görünmelidir.

## 3. Logları ve Seans Özetini İnceleme

```bash
# Son 200 satır
sudo docker logs bearish-bot --tail 200

# Log akışını canlı izlemek
sudo docker logs -f bearish-bot
```

Önemli log tipleri:
- VM boot ve environment setup (`Bearish Alpha Bot - VM Boot` satırları)
- GEMMA / PPO / TA-Lib health check mesajları
- `WATCHDOG` heartbeat ve engine state
- Aktif pozisyonlar ve P&L özetleri
- Seans sonu **EXIT SUMMARY** ve **Bot shutdown complete** satırları (graceful shutdown doğrulaması için)

Seans bittikten sonra en güncel `live_trading_*.log` dosyası üzerinde özet analiz almak için iki yol vardır:

1. Doğrudan analyzer scriptini çalıştırmak:

```bash
sudo docker exec bearish-bot python diagnostics/log_analyzer_auto_plus.py
```

2. Operasyon için önerilen helper scripti kullanmak (önerilen):

```bash
sudo docker exec bearish-bot python scripts/run_last_session_analysis.py
```

Her iki komut da container içindeki `logs/` klasöründe bulunan **son** `live_trading_*.log` üzerinde çalışır ve
golden graceful shutdown pattern'ini, P&L özetini ve **Signal → Trade funnel** metriklerini raporlar. Helper script,
seansın TRADING_DURATION ile bitip bitmediğine veya manuel/acil stop ile sonlanmasına bakmaz; her zaman en son seansı
loglar üzerinden analiz eder.

## 4. Container'ı Güvenli Şekilde Yeniden Başlatma

Yeni imaj deploy edildiğinde veya config değiştiğinde iki yol kullanabilirsin:

1. Klasik docker komutlarıyla manuel yönetim:

```bash
# VM üzerinde
sudo docker stop bearish-bot || true
sudo docker rm bearish-bot || true

sudo docker pull bearishalphabot.azurecr.io/bearish-bot:vm-vmboot-4

sudo docker run -d \
   --name bearish-bot \
   --restart unless-stopped \
   --env-file /home/azureuser/bearish-bot.env \
   bearishalphabot.azurecr.io/bearish-bot:vm-vmboot-4
```

2. Ops helper script ile tek adımda yönetim (önerilen):

```bash
cd /app  # container içinde repo kök dizini varsayımı
python scripts/vm_run_session.py
```

Bu helper, sırasıyla `docker stop`, `docker rm`, `docker pull` ve `docker run` komutlarını standart parametrelerle
çalıştırır. Volume mount'larını aktif tutmak için varsayılan olarak `/mnt/bearish/logs` → `/app/logs` ve
`/mnt/bearish/data` → `/app/data` mapping'lerini kullanır. Sadece hangi komutların çalışacağını görmek için:

```bash
python scripts/vm_run_session.py --just-print
```

Notlar:
- `bearish-bot.env` dosyası credential ve runtime ayarlarını içerir.
- `--restart unless-stopped` VM reboot sonrası container'ın otomatik başlamasını sağlar.
- Container entry point **değiştirilmemelidir**: Dockerfile içindeki `CMD ["python", "vm_boot.py"]`
  prod kontratının bir parçasıdır ve içerideki `scripts/live_trading_launcher.py` + core pipeline yapısını bozmaz.

## 5. Env Dosyasını Güncelleme (`bearish-bot.env`)

```bash
# VM üzerinde
nano /home/azureuser/bearish-bot.env
```

Sık kullanılan değişkenler:
- `EXCHANGES=bingx`
- `TRADING_SYMBOLS=BTC/USDT:USDT`
- `TRADING_MODE=paper` veya `live`
- `TRADING_DURATION=0` (0 = sınırsız, >0 = saniye cinsinden süre; golden graceful shutdown TRADING_DURATION dolduğunda tetiklenir)
- Borsa API key/secret değerleri

Değişiklikten sonra container'ı yeniden başlat:

```bash
sudo docker restart bearish-bot
```

## 6. Uptime ve Sağlık Kontrolü

```bash
# Son birkaç saniyelik log
sudo docker logs bearish-bot --tail 50
```

Kontrol et:
- `WATCHDOG-XXXX Heartbeat - is_running=True`
- `Engine state: running`
- Aktif pozisyon ve P&L özetleri

Container seviyesinde uptime görmek için:

```bash
sudo docker ps --format 'table {{.Names}}\t{{.Status}}'
```

## 7. Disk ve Kaynak Kullanımı

```bash
# Disk kullanımı
 df -h

# Docker'ın kullandığı disk alanı
 sudo docker system df

# Temizlik (dikkatli kullan)
 sudo docker system prune -f
```

## 8. Log ve Data Volume Stratejisi

Prod ortamda log ve state dosyalarına VM seviyesinden kolay erişim için aşağıdaki volume'leri kullanman önerilir:

```bash
sudo docker run -d \
   --name bearish-bot \
   --restart unless-stopped \
   --env-file /home/azureuser/bearish-bot.env \
   -v /mnt/bearish/logs:/app/logs \
   -v /mnt/bearish/data:/app/data \
   bearishalphabot.azurecr.io/bearish-bot:vm-vmboot-4
```

- Container içindeki `logs/` klasörü golden graceful shutdown pattern'ini, exit summary'leri ve
   tüm seans loglarını tutar; host tarafında `/mnt/bearish/logs` altında görünür.
- Container içindeki `data/state.json` ve `data/day_stats.json` gibi runtime state dosyaları,
   host tarafında `/mnt/bearish/data` altında saklanır.
- `diagnostics/log_analyzer_auto_plus.py` scripti, container içindeki `logs/` dizinini
   kullandığı için bu volume stratejisi ile hem CI (`act`), hem de Azure VM prod ortamında
   aynı path'ler korunmuş olur.

## 9. Olası Sorun Senaryoları

### a) Container Çalışmıyor / Hemen Çıkıyor

1. `sudo docker ps -a` ile EXIT kodunu kontrol et.
2. Detaylı log için:
   ```bash
   sudo docker logs bearish-bot --tail 200
   ```
3. Genellikle aşağıdaki sebepler görülür:
   - Eksik veya hatalı env değişkenleri (`bearish-bot.env`)
   - ACR'den imaj çekilememesi (network / auth)
   - GEMMA / artefact dosyalarında eksik dosya (genelde build aşamasında yakalanır)

### b) Bot Çalışıyor Ama İşlem Açmıyor

1. Loglarda şu alanlara bak:
   - Rejim filtresi (market regime uygunsuz olabilir)
   - Risk motoru ve duplicate prevention uyarıları
   - PPO / GEMMA tarafından sinyanın zayıf veya riskli işaretlenmesi
2. Gerekirse `TRADING_SYMBOLS`, risk parametreleri veya duplicate prevention ayarlarını gözden geçir.

---

Bu playbook, günlük operasyonlar ve hızlı müdahale için tasarlanmıştır. Daha ileri otomasyon (ör. health endpoint polling, uptime metrikleri, otomatik raporlama) ileride bu dokümana eklenebilir.

## 9. Kritik Alarm Senaryoları – Hızlı Checklist

### a) P&L Sert Negatif / Beklenenden Fazla Zarar

1. Son loglara bak:
   ```bash
   sudo docker logs bearish-bot --tail 50
   ```
2. Kontrol et:
   - Açık pozisyon sayısı (beklediğinden fazla mı?)
   - Son exit nedenleri (STOP-LOSS / TAKE-PROFIT / TRAILING-STOP / LIQUIDATION)
3. İlk aksiyon:
   - Gerekirse `TRADING_MODE=paper` yapıp container'ı yeniden başlat.
   - Risk parametrelerini (max position, leverage, TP/SL multiplier) gözden geçir.

### b) WebSocket Kopmaları veya Ağ Problemleri

1. Loglarda `WS-ERROR`, `WebSocket reconnect`, `REST fallback` yoğunluğu var mı?
2. Hızlı kontroller:
   ```bash
   ping api.bingx.com -c 4
   ```
3. Aksiyonlar:
   - Ağ tarafında ciddi sorun yoksa container'ı yeniden başlat:
     ```bash
     sudo docker restart bearish-bot
     ```
   - Sorun sürekli ise borsanın status sayfasını / duyurularını kontrol et.

### c) Bot Çalışıyor Ama Uzun Süre Hiç Sinyal / İşlem Yok

1. Loglarda şunlara bak:
   - `WATCHDOG-XXXX Heartbeat - is_running=True`
   - `Engine state: running`
   - Rejim filtresi / risk motoru çok agresif mi?
2. Hızlı kontroller:
   - `TRADING_SYMBOLS` doğru mu? (ör. `BTC/USDT:USDT`)
   - `TRADING_MODE` gerçekten `paper` veya `live` mı?
   - Duplicate prevention ve min price change threshold çok mu yüksek ayarlanmış?

### d) Container Sürekli Crash / Boot Loop

1. Container durumunu kontrol et:
   ```bash
   sudo docker ps -a
   ```
2. Son hata mesajını görmek için:
   ```bash
   sudo docker logs bearish-bot --tail 200
   ```
3. Tipik sebepler:
   - Env dosyasında eksik / bozuk satır (ör. boş `EXCHANGES=`)
   - Yanlış ACR tag'i (mevcut olmayan imaj)
   - GEMMA / PPO artefact path problemi (deploy sırasında eksik kopyalanmış dosya)

Bu checklist, acil durumda hızlıca neye bakman gerektiğini hatırlatmak içindir; detaylı kök neden analizi için logları ve config'i ayrıca incelemek gerekir.
