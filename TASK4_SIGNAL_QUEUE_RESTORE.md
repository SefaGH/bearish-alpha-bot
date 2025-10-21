# TASK 4 – Live Trading Signal Queue Restore (Version 4)

## Amaç
Live trading motorunun strateji sinyallerini kuyruğa aktarmasını sağlayarak Phase 4 yapısının planlanan işlem akışını eksiksiz hale getirmek.

## Problem Özeti
- `_signal_processing_loop` içinde oluşturulan strateji sinyalleri `pass` ile yok sayılıyor.
- `duplicate_prevention` ve risk yönetimi katmanları kuyruğa sinyal düşmediği için devreye giremiyor.
- Bot pre-flight kontrolleri geçmesine rağmen "Signals in queue: 0" durumunda takılı kalıyor ve hiç emir üretmiyor.

## Kapsam
- `src/core/live_trading_engine.py` içindeki sinyal işleme bloğu.
- Sinyal kuyruğu üzerinde duplicate-prevention metriklerinin güncellenmesi.
- Kağıt modunda smoke test (GitHub Actions `live_trading_launcher.yml`).

## Adımlar
1. `_signal_processing_loop` fonksiyonunda stratejiden dönen sinyali zenginleştir (strateji adı, adaptive bayrakları, risk metrikleri) ve `self.signal_queue`'ya koy.
2. Kuyruğa alınan sinyaller için log mesajı ekle (ör. `Queued signal for BTC/USDT via adaptive_ob`).
3. `duplicate_prevention` sisteminin kuyruğa aktarılan sinyallerle tetiklenmesini sağlayacak şekilde gerekli metadata'yı aktar.
4. Kağıt modunda (`python scripts/live_trading_launcher.py --paper --duration 120`) smoke test çalıştır; loglarda sinyal kuyruğu ve yürütme adımlarını doğrula.
5. Gerekirse `live_trading_launcher.yml` workflow'unda ek log saklama/artifact toplama adımı ekleyerek CI gözlemini kolaylaştır.

## Deliverables
- Kod düzeltmeleri (`live_trading_engine` ve gerekiyorsa ilgili koordinatörler).
- Test logları ve gerekiyorsa CI workflow güncellemesi.
- Güncellenmiş changelog/özet notu.

## Kabul Kriterleri
- Bot kağıt modunda en az bir sinyal oluşturup kuyruğa alarak `execute_signal` yolunu tetikliyor.
- Duplicate-prevention metrikleri loglarda güncelleniyor ve cooldown mantığı çalışıyor.
- Workflow çıktıları hata içermiyor, duration boyunca bot takılmadan çalışıyor.

## Riskler & Azaltım
- **Yanlış sinyal metadata'sı** duplicate-prevention kurallarını atlatabilir → metadata haritası test edilerek doğrulanmalı.
- **Websocket gecikmesi** sinyal kuyruğunu dolduramayıp eski duruma dönebilir → log seviyesini geçici olarak DEBUG’a çekip akışı izleyin.
- **CI süresi** 120s smoke test ile sınırlı tutulmalı; gerekirse `duration` parametresi ayarlanmalı.

## Zaman Çizelgesi
- Geliştirme & birim testleri: 0.5 gün
- Smoke test & log incelemesi: 0.25 gün
- Son kontrol & PR: 0.25 gün

