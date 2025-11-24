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

## 3. Logları İnceleme

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

## 4. Container'ı Güvenli Şekilde Yeniden Başlatma

Yeni imaj deploy edildiğinde veya config değiştiğinde:

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

Notlar:
- `bearish-bot.env` dosyası credential ve runtime ayarlarını içerir.
- `--restart unless-stopped` VM reboot sonrası container'ın otomatik başlamasını sağlar.

## 5. Env Dosyasını Güncelleme (`bearish-bot.env`)

```bash
# VM üzerinde
nano /home/azureuser/bearish-bot.env
```

Sık kullanılan değişkenler:
- `EXCHANGES=bingx`
- `TRADING_SYMBOLS=BTC/USDT:USDT`
- `TRADING_MODE=paper` veya `live`
- `TRADING_DURATION=0` (0 = sınırsız, >0 = saniye cinsinden süre)
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

## 8. Olası Sorun Senaryoları

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
