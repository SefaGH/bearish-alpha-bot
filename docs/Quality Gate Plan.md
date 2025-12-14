## Kurmak istediğimiz sistem: “Quality Gate” nedir?

Hedef mimari:

* **Volume bucket (VERY_LOW → EXTREME)**

  * **ML confidence / sinyal kalitesi**
  * **Momentum / spread vb.**
    ⇒ tek bir metrikte toplanıyor: **`quality_score ∈ [0, 1]`**

Ve:

* **Hiçbir bucket tek başına trade’i veto etmiyor.**

  * Yani artık “LOW → direkt trade açma” yok.
* Tüm volume bucket’lar **yalnızca skora etki ediyor.**
* Trade kararı da şu mantıkla veriliyor:

> “Eğer `quality_score` belirlediğimiz eşiklerin altındaysa → **trade açma**.
> Üstündeyse → normal risk pipeline’a gönder.”

Yani VolumeAnalyzer:

1. Sinyalin konteksini (bucket, strength) üretir.
2. Bu context quality_score’a katkı yapar.
3. Nihai karar, **skor temelli gate** ile verilir.

Şimdi bunu offline kalibrasyonla nasıl somutlayacağımızı adım adım yazalım.

---

## 1. Ön koşullar – Log ve veri tarafında neye ihtiyacımız var?

Kalibrasyon yapabilmek için her **kapanmış trade** için aşağıdaki bilgileri görebiliyor olmamız gerekiyor (çoğu zaten log’larda var):

Her trade için (TRADE_CLOSED event + diğer loglardan join):

* `run_id`
* `timestamp`
* `symbol`, `timeframe`, `strategy_name` (adaptive_ob / adaptive_str)
* `side` (LONG/SHORT)
* `entry_price`, `exit_price`
* `rr` (veya `rr_achieved`)
* `realized_pnl_usdt`
* `volume_bucket_at_entry` (VERY_LOW / LOW / NORMAL / HIGH / EXTREME)
* `volume_strength_at_entry`
* `momentum_strength_at_entry` (varsa)
* **`quality_score`** (mümkünse sinyal üretilirken hesaplanan değer)

Şu anda:

* Volume + trade tarafı log’larda var.
* Quality_score loglarda yoksa, iki seçenek var:

  1. Küçük bir ek log: StrategyCoordinator’da, sinyal RiskManager’a gitmeden hemen önce:

     * `signal_evaluation` benzeri bir event ile
       `quality_score`, `ml_confidence`, `volume_bucket`, `rr` vs. loglamak.
  2. Ya da offline script’te quality_score formülünü birebir tekrar hesaplamak
     (mümkün ama daha uğraştırıcı).

Playbook açısından:
**İdeal varsayım** – kalibrasyona geçmeden önce quality_score’u logluyoruz.

---

## 2. Veri setini oluşturma

1. Paper run / micro live run’lardan **yeterli sayıda trade** topla
   (minimum 100+, mümkünse 300+ trade).

2. Loglardan tek bir tablo/dosya çıkar:

Her satır bir trade olacak şekilde:

* `run_id`
* `strategy_name`
* `volume_bucket_at_entry`
* `volume_strength_at_entry`
* `momentum_strength_at_entry` (varsa)
* `quality_score`
* `ml_confidence` (price_confidence vb.)
* `rr`
* `realized_pnl_usdt`
* `time_of_day` (opsiyonel – saat dilimine göre segment için)
* `regime` (trend/range gibi, eğer logluyorsak)

Bunu:

* Var olan `analyze_volume_buckets.py` scriptini genişleterek,
* Ya da yeni bir `scripts/analyze_quality_gate.py` yazarak yapabilirsin.

---

## 3. Bakılacak metrikler (rapor içeriği)

### 3.1. Quality band’lerine göre performans

Quality_score’u band’lara böl:

* [0.0, 0.2)
* [0.2, 0.3)
* [0.3, 0.4)
* [0.4, 0.5)
* [0.5, 0.6)
* [0.6, 0.7)
* [0.7, 0.8)
* [0.8, 1.0]

Her band için:

* `n_trades`
* `win_rate` (% pozitif trade)
* `avg_pnl_usdt`
* `median_rr`
* `max_drawdown_contribution` (opsiyonel – o band trades’inin toplam DD’ye katkısı)

Amaç:
“Quality < X iken ortalama sonuç net negatif mi?” sorusuna cevap bulmak.

---

### 3.2. Bucket + quality kombinasyonları

Sonra bunu volume bucket’a böl:

Örneğin her bucket için:

* `volume_bucket = VERY_LOW / LOW / NORMAL / HIGH / EXTREME`

ve bu bucket içinde yine quality band’lerine göre:

* `n_trades`
* `win_rate`
* `avg_pnl_usdt`
* `median_rr`

Böylece şuna bakabiliyoruz:

* VERY_LOW & quality 0.3 altı → tam çöp mü?
* VERY_LOW ama quality 0.6+ → yine de kötü mü?
* NORMAL / HIGH / EXTREME + quality 0.4+ → gerçekten çok mu daha iyi?

İstediğimiz resim:

* **Belirgin bir kırılma noktası**:
  Örneğin quality < 0.35 → her bucket’ta net negatif;
  quality > 0.45 → özellikle NORMAL / HIGH / EXTREME’de belirgin pozitif.

---

### 3.3. Strateji bazında ayrım (OB vs STR)

Aynı analizleri:

* `strategy_name = adaptive_ob`
* `strategy_name = adaptive_str`

için ayrı ayrı yapmak önemli, çünkü:

* OB ve STR farklı rejimlerde oynuyor,
* ML ve volume etkisi iki stratejide farklı davranabilir.

Her strateji için:

* quality band’lerine göre win_rate / avg_pnl,
* bucket + quality kombinasyonları.

Sonuç:
OB için eşik 0.38, STR için 0.42 gibi farklı “sweet spot”lar çıkabilir.

---

### 3.4. İleri seviye (opsiyonel)

İstersen şunlara da bakabiliriz:

* **Time-of-day**: Örn. quality 0.4–0.5 ama sadece Asya seansında kötü mü?
* **Regime**: Trend / range tag’imiz varsa, quality eşiği regime’e göre değişiyor mu?

Bu, ileride “regime-aware quality gate” kurmak için veri sağlar ama ilk fazda şart değil.

---

## 4. Eşik (threshold) belirleme metodolojisi

Playbook’un kalbi burası.

### 4.1. Basit görsel / tablo analizi

Her strateji için:

* Quality band → avg_pnl_usdt grafiği
* Quality band → win_rate grafiği

Beklenen form:

* 0.0–0.2: ağır negatif
* 0.2–0.3: hala negatif / düzensiz
* 0.3–0.4: break-even civarı
* 0.4–0.5 ve üzeri: net pozitif, daha stabil

Buna göre:

* OB için “trade açılabilir” bandı:
  quality ≥ Q_OB
* STR için:
  quality ≥ Q_STR

gibi iki eşik seçilebilir.

### 4.2. “Gate simülasyonu” yap

Offline olarak:

* Eşik uyguladığını varsay:

  Örn. OB için Q_OB = 0.35, STR için Q_STR = 0.40.

* `quality_score < threshold` olan trade’leri veri setinden çıkar.

* Sonra “filtrelenmiş portföy” için:

  * Toplam trade sayısı,
  * Toplam PnL,
  * Ortalama PnL,
  * Win rate,
  * Max drawdown (yaklaşık olarak).

Bunu birkaç farklı eşik kombinasyonu için yap:

* Senaryo A: Q_OB=0.35, Q_STR=0.40
* Senaryo B: Q_OB=0.38, Q_STR=0.42
* Senaryo C: tek global eşik: Q_ALL=0.38

Ve kıyasla:

* “Trade sayısı ne kadar azalıyor?” (% kaçını temizliyoruz)
* “Toplam PnL ve DD nasıl değişiyor?”

İdeal hedef:

* Trade sayısında çok büyük düşüş olmadan
  (örneğin en fazla %30–40 azalma),
* Ortalama PnL ve özellikle **kuyrukta aşırı kötü trade’lerin** ciddi azalması.

---

## 5. Sonuç: Quality gate taslağı

Bu analizden sonra ortaya çıkacak olan şey:

Örneğin (tamamen örnek rakam):

* OB için: quality < 0.36 → trade açma.
* STR için: quality < 0.40 → trade açma.

Ve **volume bucket’ı burada nasıl kullanmış oluyoruz?**

* Volume bucket doğrudan gate etmiyor.
* VolumeAnalyzer, quality_score’un volume bileşenini besliyor:

  * VERY_LOW → düşük volume bileşeni → quality genel olarak daha düşük.
  * HIGH/EXTREME → volume bileşeni daha yüksek → quality biraz boost alıyor.
* Gate, sadece quality_score üzerinden çalışıyor:

```text
if quality_score < Q_threshold_for_strategy:
    reject_reason = "QUALITY_GATE"
    trade açma
else:
    normal risk pipeline
```

Yani:

* VolumeAnalyzer’ın rolü:
  “Bu sinyalin likidite bağlamını skora yansıtmak.”
* Quality gate’in rolü:
  “Sadece yeterince kaliteli (ML + volume + momentum) sinyalleri risk tarafına geçirmek.”

---

Şimdi quality gate kalibrasyonu için offline analiz tarafını netleştirelim. Bunu iki parçaya böleceğim:

1. Ne yapacağımızın mantığı (playbook)
2. `scripts/analyze_quality_gate.py` için teknik taslak (CLI, veri modeli, JSON/CSV şeması)

---

## 1. Hedef: Quality gate + volume’un rolü

Kurmak istediğimiz yapı:

* VolumeAnalyzer, her sinyal için:

  * `volume_bucket` (VERY_LOW / LOW / NORMAL / HIGH / EXTREME)
  * `volume_strength` (0–1)
    üretir ve StrategyCoordinator bunu quality_score’a katkı olarak kullanır.

* `quality_score` şu bileşenlerden oluşur (mevcut mimariye uygun):

  * ML bileşeni (price_confidence / ml_confidence)
  * volume bileşeni (volume_strength → volume_score)
  * momentum bileşeni (momentum_strength)
  * spread vb. ek bileşenler (varsa)

* **Quality gate mantığı:**

  * Hiçbir bucket doğrudan “trade açma / açma” kararı vermiyor.
  * Volume sadece quality_score’u yükseltip/alçaltıyor.
  * Kararı veren mekanizma:

    ```text
    if quality_score < strategy_threshold:
        trade açma (QUALITY_GATE_REJECT)
    else:
        trade’i normal risk pipeline’a gönder
    ```

Bu gate’i doğru eşikle kurmak için offline analiz script’iyle geçmiş paper run / log verisine bakacağız.

---

## 2. Analiz script’i: `scripts/analyze_quality_gate.py` taslağı

### 2.1. Beklenen input (log formatı varsayımı)

Script, JSONL (her satır bir JSON event) log’lardan beslenecek.
Şu event’lere ihtiyaç var:

1. `TRADE_CLOSED` event’i (zaten var):

   Örneğin (sadeleştirilmiş):

   ```json
   {
     "event": "TRADE_CLOSED",
     "run_id": "run-2025-12-12-01",
     "timestamp": "2025-12-12T19:07:10Z",
     "symbol": "BTC/USDT:USDT",
     "timeframe": "5m",
     "strategy_name": "adaptive_ob",
     "side": "LONG",
     "entry_price": 91000.0,
     "exit_price": 91800.0,
     "rr": 1.8,
     "realized_pnl_usdt": 3.5,
     "volume_bucket_at_entry": "HIGH",
     "volume_strength_at_entry": 0.72,
     "momentum_strength_at_entry": 0.55,
     "quality_score_at_entry": 0.62   // BUNU LOGLAMIŞ OLMAMIZ İDEAL
   }
   ```

   Eğer `quality_score_at_entry` henüz log’da yoksa, ileride küçük bir ek log ile bunu TRADE_CLOSED’a taşımamız gerekecek. Script tasarımını buna göre yapıyorum.

2. (Opsiyonel) Kalite snapshot event’i

   İleride istersen şu tarz bir event de eklenebilir:

   ```json
   {
     "event": "signal_evaluation",
     "run_id": "...",
     "strategy_name": "adaptive_ob",
     "symbol": "BTC/USDT:USDT",
     "timeframe": "5m",
     "quality_score": 0.61,
     "ml_confidence": 0.74,
     "volume_bucket": "HIGH",
     "volume_strength": 0.7,
     "rr_at_signal": 1.9
   }
   ```

   Ama ilk faz için **TRADE_CLOSED + quality_score_at_entry** yeterli.

---

### 2.2. CLI taslağı

Script’i şu şekilde tasarlayabiliriz:

```bash
python -m scripts.analyze_quality_gate \
  --log-dir ./logs/run_2025_12_12 \
  --run-id run-2025-12-12-01 \
  --timeframe 5m \
  --output ./reports/quality_gate_report.json \
  --csv-output ./reports/quality_gate_trades.csv
```

Önerilen argümanlar:

* `--log-dir` veya `--log-file`

  * Varsayılan: `./logs`
  * Eğer `--log-file` verilirse sadece o dosya okunur; `--log-dir` verilirse tüm `.log` / `.jsonl` dosyalar taranır.

* `--run-id` (opsiyonel ama önerilir)

  * Boş bırakılırsa tüm run_id’ler dahil edilir.

* `--timeframe` (opsiyonel)

  * Örn. sadece `5m` analiz edilsin.

* `--strategy` (opsiyonel)

  * `adaptive_ob`, `adaptive_str`, veya `all`.

* `--min-rr` (opsiyonel filtre)

  * Örn. `--min-rr 1.0` diyerek R/R 1’in üzerinde olan trade’leri süzebilirsin.

* `--output` (JSON rapor dosyası, zorunlu)

* `--csv-output` (opsiyonel, trade-level CSV export)

---

### 2.3. Script’in iç mantığı (yüksek seviye)

Pseudo-akış:

```python
def main():
    args = parse_args()
    trades = load_trades(args)
    trades = apply_filters(trades, args)

    report = build_quality_gate_report(trades, args)

    write_json(report, args.output)
    if args.csv_output:
        write_csv(trades, args.csv_output)
```

**`load_trades`**

* Belirtilen log dosyalarını satır satır oku.

* JSON parse et.

* `event == "TRADE_CLOSED"` olanları al.

* Her trade için şu alanları çek:

  ```python
  {
    "run_id": ...,
    "symbol": ...,
    "timeframe": ...,
    "strategy_name": ...,
    "side": ...,
    "entry_price": ...,
    "exit_price": ...,
    "rr": ...,
    "realized_pnl_usdt": ...,
    "volume_bucket_at_entry": ...,
    "volume_strength_at_entry": ...,
    "momentum_strength_at_entry": ... (opsiyonel),
    "quality_score_at_entry": ...   # zorunlu kabul edelim
  }
  ```

* Eksik `quality_score_at_entry` olan trade’leri ya discarda edeceğiz ya da metastatistikte ayrı bir kategori olarak raporlayacağız.

**`apply_filters`**

* `run_id` filtresi
* `timeframe` filtresi
* `strategy_name` filtresi
* `min_rr` filtresi (opsiyonel)

---

## 3. JSON rapor şeması (taslak)

JSON raporunu 4 ana blokta kurgulayalım:

```json
{
  "meta": { ... },
  "overall": { ... },
  "by_quality_band": { ... },
  "by_bucket_and_quality": { ... },
  "by_strategy": { ... },
  "threshold_scenarios": [ ... ]
}
```

Detaylar:

### 3.1. `meta`

```json
"meta": {
  "generated_at": "2025-12-14T22:30:00Z",
  "run_id_filter": "run-2025-12-12-01",
  "timeframe_filter": "5m",
  "strategy_filter": "all",
  "n_trades_total": 185,
  "n_trades_with_quality": 180,
  "quality_bands": [
    [0.0, 0.2],
    [0.2, 0.3],
    [0.3, 0.4],
    [0.4, 0.5],
    [0.5, 0.6],
    [0.6, 0.7],
    [0.7, 0.8],
    [0.8, 1.0]
  ]
}
```

### 3.2. `overall`

```json
"overall": {
  "n_trades": 180,
  "win_rate": 0.54,
  "avg_pnl_usdt": 0.85,
  "median_rr": 1.4,
  "pnl_std_dev": 2.1,
  "approx_max_drawdown_usdt": -12.5
}
```

> Max drawdown için script tam PnL serisi üzerinden kabaca bir DD tahmini yapabilir (sırayla trade’leri timestamp’e göre sort edip kümülatif equity curve türetmek).

### 3.3. `by_quality_band`

Ana kalite resmi:

```json
"by_quality_band": {
  "0.0-0.2": {
    "n_trades": 20,
    "win_rate": 0.25,
    "avg_pnl_usdt": -0.9,
    "median_rr": 0.7
  },
  "0.2-0.3": {
    "n_trades": 30,
    "win_rate": 0.30,
    "avg_pnl_usdt": -0.4,
    "median_rr": 0.9
  },
  "0.3-0.4": {
    "n_trades": 40,
    "win_rate": 0.45,
    "avg_pnl_usdt": 0.1,
    "median_rr": 1.1
  },
  "0.4-0.5": {
    "n_trades": 35,
    "win_rate": 0.55,
    "avg_pnl_usdt": 0.6,
    "median_rr": 1.4
  },
  "0.5-0.6": {
    "n_trades": 25,
    "win_rate": 0.60,
    "avg_pnl_usdt": 1.1,
    "median_rr": 1.7
  },
  ...
}
```

Buradan ilk “kırılma” noktalarını görebilirsin.

### 3.4. `by_bucket_and_quality`

Volume bucket ile quality birlikte:

```json
"by_bucket_and_quality": {
  "VERY_LOW": {
    "0.0-0.3": { "n_trades": 15, "win_rate": 0.15, "avg_pnl_usdt": -1.2 },
    "0.3-0.5": { "n_trades": 5,  "win_rate": 0.40, "avg_pnl_usdt": 0.1 },
    "0.5-1.0": { "n_trades": 2,  "win_rate": 0.50, "avg_pnl_usdt": 0.3 }
  },
  "LOW": {
    ...
  },
    "NORMAL": { ... },
    "HIGH": { ... },
    "EXTREME": { ... }
}
```

Burada özellikle şunlara bakacağız:

* VERY_LOW & LOW katmanında quality yükselse bile performans çöp mü?
* NORMAL / HIGH / EXTREME’de quality yükseldikçe PnL belirgin artıyor mu?

### 3.5. `by_strategy`

Strateji bazlı ayrım:

```json
"by_strategy": {
  "adaptive_ob": {
    "overall": { ... },
    "by_quality_band": { ... },
    "by_bucket_and_quality": { ... }
  },
  "adaptive_str": {
    "overall": { ... },
    "by_quality_band": { ... },
    "by_bucket_and_quality": { ... }
  }
}
```

---

### 3.6. `threshold_scenarios`

Quality gate için senaryo simülasyonu:

Script, belirli threshold set’leri için portföy simülasyonu yapabilir. Örn:

* Senaryo A: global eşik 0.38
* Senaryo B: OB=0.36, STR=0.40
* Senaryo C: OB=0.40, STR=0.45

Her senaryo için:

```json
"threshold_scenarios": [
  {
    "name": "global_0.38",
    "thresholds": {
      "global": 0.38
    },
    "kept_trades": 120,
    "filtered_out_trades": 60,
    "kept_trades_ratio": 0.67,
    "total_pnl_usdt": 110.5,
    "avg_pnl_usdt": 0.92,
    "win_rate": 0.60,
    "approx_max_drawdown_usdt": -8.0
  },
  {
    "name": "per_strategy_ob_0.36_str_0.40",
    "thresholds": {
      "adaptive_ob": 0.36,
      "adaptive_str": 0.40
    },
    "kept_trades": 130,
    "filtered_out_trades": 50,
    "kept_trades_ratio": 0.72,
    "total_pnl_usdt": 115.0,
    "avg_pnl_usdt": 0.88,
    "win_rate": 0.58,
    "approx_max_drawdown_usdt": -7.5
  }
]
```

Bu blok, kalibrasyon kararını çok somut hale getirir:

* Gate eşiğini nereye koyarsam:

  * Kaç trade kaybediyorum?
  * PnL/toplam risk profili nasıl değişiyor?

---

## 4. CSV çıktısı

`--csv-output` verilirse, trade-level ham veri şu sütunlarla export edilebilir:

```text
run_id,
timestamp,
strategy_name,
symbol,
timeframe,
side,
entry_price,
exit_price,
rr,
realized_pnl_usdt,
volume_bucket_at_entry,
volume_strength_at_entry,
momentum_strength_at_entry,
quality_score_at_entry
```

Bunu hem Excel’de hızlı inceleme için, hem de farklı analizler (örneğin Jupyter, R) için kullanabilirsin.

---

